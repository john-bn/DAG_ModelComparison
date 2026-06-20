"""Command-line interface: ``compare run | list | latest``.

This replaces the interactive ``input()`` flow so the comparator can run
unattended under cron and be queried on demand from a terminal.

    compare run    --model hrrr --var TMP [--verif rtma] [--rolling opts | explicit opts]
    compare list   [--model ...] [--var ...] [--since YYYY-MM-DD] ...
    compare latest --model hrrr --var TMP [--open]

Design notes:
* Rolling real-time is the default for ``run`` (cron); pass an explicit
  ``--date``/``--init``/``--fxx`` to pin a specific target (backfills).
* ``--dry-run`` resolves and prints the target without fetching data or writing
  anything — handy for debugging a crontab line, and import-light (the heavy
  ``pipeline`` module is only imported when real work happens).
* "No data yet" is a clean exit 0 (transient for a real-time job, so cron does
  not raise an alarm); unexpected errors exit non-zero.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import logging
import sys
import time

from comparator import appconfig, manifest, normalize, timesel

logger = logging.getLogger("comparator")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare",
        description="Verify NWP forecasts against the RTMA/URMA analysis.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared config/path options.
    def add_common(p):
        p.add_argument("--config", help="Path to config.yaml")
        p.add_argument("--data-dir", help="GRIB/cache directory (overrides config)")
        p.add_argument("--out-dir", help="Figure/manifest output dir (overrides config)")
        p.add_argument("--log-dir", help="Log directory (overrides config)")

    # --- run ---
    run = sub.add_parser("run", help="Generate a comparison (single frame or GIF).")
    add_common(run)
    run.add_argument("--model", required=True, help="NWP model, e.g. hrrr, nam12k, gfs")
    run.add_argument("--var", required=True, help="Variable: TMP, DPT, VIS, WIND, GUST")
    run.add_argument("--verif", help="Verification analysis: rtma or urma")
    run.add_argument("--mode", choices=["single", "gif"], default="single",
                     help="single frame (default) or animated GIF over all runs")
    # Rolling real-time knobs (used when no explicit --date is given).
    run.add_argument("--lag", type=int, metavar="HOURS",
                     help="Rolling data-latency buffer (default from config).")
    run.add_argument("--lead", type=int, metavar="HOURS",
                     help="Rolling preferred forecast lead (single mode).")
    # Explicit target (overrides rolling).
    run.add_argument("--date", help="Explicit target date YYYY-MM-DD (pins the run).")
    run.add_argument("--init", type=int, metavar="HH",
                     help="single: init cycle hour (Z); gif: analysis valid hour (Z).")
    run.add_argument("--fxx", type=int, metavar="HH", help="single: forecast lead hour.")
    run.add_argument("--duration", type=int, default=500,
                     help="GIF frame duration in ms (default 500).")
    run.add_argument("--dry-run", action="store_true",
                     help="Resolve and print the target; write nothing.")
    run.add_argument("--log-level", default="INFO",
                     help="Logging level (DEBUG, INFO, WARNING, ...).")

    # --- list ---
    lst = sub.add_parser("list", help="List past outputs from the run manifest.")
    add_common(lst)
    lst.add_argument("--model")
    lst.add_argument("--var")
    lst.add_argument("--verif")
    lst.add_argument("--mode", choices=["single", "gif"])
    lst.add_argument("--since", help="Filter valid_dt >= this date (YYYY-MM-DD).")
    lst.add_argument("--until", help="Filter valid_dt <= this date (YYYY-MM-DD).")
    lst.add_argument("--limit", type=int, default=20, help="Max rows (default 20).")

    # --- latest ---
    latest = sub.add_parser("latest", help="Print the newest matching output path.")
    add_common(latest)
    latest.add_argument("--model")
    latest.add_argument("--var")
    latest.add_argument("--verif")
    latest.add_argument("--mode", choices=["single", "gif"])
    latest.add_argument("--open", action="store_true",
                        help="Open the file (interactive sessions only).")

    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(cfg, level="INFO"):
    """Configure root logging to a rotating file + stderr, timestamped in UTC."""
    from logging.handlers import RotatingFileHandler

    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    # Avoid duplicate handlers if called twice in one process.
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fmt.converter = time.gmtime  # UTC timestamps

    fh = RotatingFileHandler(cfg.log_path, maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)


# --------------------------------------------------------------------------- #
# Target resolution
# --------------------------------------------------------------------------- #
def _resolve_target(args, cfg, model_key, now_utc):
    """Return (cycle_dt, fxx, valid_dt) for the requested run.

    cycle_dt/fxx are None for GIF mode (which spans every covering cycle).
    """
    explicit = args.date is not None
    if explicit:
        if args.init is None:
            raise ValueError("--date requires --init (the hour in Z-time).")
        if args.mode == "single":
            if args.fxx is None:
                raise ValueError("single mode with --date requires --fxx.")
            cycle_dt = datetime.fromisoformat(f"{args.date} {args.init:02d}:00")
            valid_dt = cycle_dt + timedelta(hours=args.fxx)
            return cycle_dt, args.fxx, valid_dt
        # gif: --date/--init describe the analysis VALID time
        valid_dt = datetime.fromisoformat(f"{args.date} {args.init:02d}:00")
        return None, None, valid_dt

    # Rolling real-time.
    if args.mode == "single":
        return timesel.resolve_rolling_target(
            model_key, now_utc, lag_hours=cfg.lag_hours, lead_hours=cfg.default_lead,
        )
    valid_dt = timesel.floor_to_hour(now_utc.replace(tzinfo=None)) \
        - timedelta(hours=cfg.lag_hours)
    return None, None, valid_dt


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_run(args) -> int:
    cfg = appconfig.load_config(
        config_path=args.config, data_dir=args.data_dir, out_dir=args.out_dir,
        log_dir=args.log_dir, verif=args.verif, lag_hours=args.lag,
        default_lead=args.lead,
    )

    # Validate model/variable/verif up front (clear usage error, no work done).
    try:
        model_key = normalize.normalize_model_key(args.model)
        var_key = normalize.normalize_var_key(args.var)
        verif_key = normalize.normalize_verif_key(cfg.verif)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE

    now_utc = datetime.now(timezone.utc)
    try:
        cycle_dt, fxx, valid_dt = _resolve_target(args, cfg, model_key, now_utc)
    except timesel.NoDataYet as e:
        # Transient for a real-time job; log + clean exit so cron stays quiet.
        if not args.dry_run:
            setup_logging(cfg, args.log_level)
        logger.warning("%s", e)
        return EXIT_OK
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE

    if args.dry_run:
        print("DRY RUN — resolved target (nothing fetched or written):")
        print(f"  model   : {model_key}")
        print(f"  var     : {var_key}")
        print(f"  verif   : {verif_key}")
        print(f"  mode    : {args.mode}")
        if cycle_dt is not None:
            print(f"  cycle   : {cycle_dt:%Y-%m-%d %H}Z")
            print(f"  fxx     : F{fxx:03d}")
        print(f"  valid   : {valid_dt:%Y-%m-%d %H}Z")
        print(f"  data_dir: {cfg.data_dir}")
        print(f"  out_dir : {cfg.out_dir}")
        return EXIT_OK

    setup_logging(cfg, args.log_level)
    from comparator import pipeline  # heavy import deferred until real work

    ts = datetime.now(timezone.utc).replace(tzinfo=None)  # naive UTC, uniform manifest
    try:
        if args.mode == "single":
            result = pipeline.generate_comparison_frame(
                model_key, var_key, cycle_dt, fxx, verif_key,
                data_dir=cfg.data_dir, out_dir=cfg.out_dir,
            )
            if result is None:
                logger.warning("No output produced for %s %s %sZ.",
                               model_key, var_key, f"{valid_dt:%Y-%m-%d %H}")
                return EXIT_OK
            manifest.append(cfg.manifest_path, manifest.make_record(
                ts_utc=ts, model=model_key, var=var_key, verif=verif_key,
                mode="single", path=result.path, cycle_dt=cycle_dt, fxx=fxx,
                valid_dt=valid_dt, mean=result.mean, rmse=result.rmse, n=result.n,
            ))
            print(result.path)
            return EXIT_OK

        gif_path = pipeline.generate_gif(
            model_key, var_key, valid_dt, verif_key,
            data_dir=cfg.data_dir, out_dir=cfg.out_dir, duration=args.duration,
        )
        if gif_path is None:
            logger.warning("No GIF produced for %s %s %sZ.",
                           model_key, var_key, f"{valid_dt:%Y-%m-%d %H}")
            return EXIT_OK
        manifest.append(cfg.manifest_path, manifest.make_record(
            ts_utc=ts, model=model_key, var=var_key, verif=verif_key,
            mode="gif", path=gif_path, valid_dt=valid_dt,
        ))
        print(gif_path)
        return EXIT_OK
    except Exception as e:  # genuine failure -> non-zero so cron surfaces it
        logger.exception("Comparison run failed: %s", e)
        return EXIT_ERROR


def _fmt(value, spec):
    """Format a possibly-None numeric value for the table."""
    return format(value, spec) if isinstance(value, (int, float)) else "-"


def cmd_list(args) -> int:
    cfg = appconfig.load_config(
        config_path=args.config, data_dir=args.data_dir, out_dir=args.out_dir,
        log_dir=args.log_dir,
    )
    rows = manifest.read(
        cfg.manifest_path, model=args.model, var=args.var, verif=args.verif,
        mode=args.mode, since=args.since, until=args.until, limit=args.limit,
    )
    if not rows:
        print("No matching runs.", file=sys.stderr)
        return EXIT_OK

    header = f"{'VALID (Z)':16} {'MODEL':7} {'VAR':5} {'VRF':4} {'MODE':6} " \
             f"{'F':>4} {'MEAN':>7} {'RMSE':>7} {'N':>8}  FILE"
    print(header)
    print("-" * len(header))
    for r in rows:
        valid = (r.get("valid_dt") or "")[:16].replace("T", " ")
        fxx = r.get("fxx")
        print(
            f"{valid:16} {r.get('model',''):7} {r.get('var',''):5} "
            f"{r.get('verif',''):4} {r.get('mode',''):6} "
            f"{('F%03d' % fxx) if isinstance(fxx, int) else '-':>4} "
            f"{_fmt(r.get('mean'), '7.2f')} {_fmt(r.get('rmse'), '7.2f')} "
            f"{_fmt(r.get('n'), '8d')}  {Path(r.get('path','')).name}"
        )
    return EXIT_OK


def cmd_latest(args) -> int:
    cfg = appconfig.load_config(
        config_path=args.config, data_dir=args.data_dir, out_dir=args.out_dir,
        log_dir=args.log_dir,
    )
    rows = manifest.read(
        cfg.manifest_path, model=args.model, var=args.var, verif=args.verif,
        mode=args.mode, limit=1,
    )
    if not rows:
        print("No matching runs.", file=sys.stderr)
        return EXIT_ERROR

    path = rows[0].get("path", "")
    print(path)

    if args.open:
        _open_file(path)
    return EXIT_OK


def _open_file(path):
    """Open *path* with the OS viewer — interactive sessions only."""
    import os
    interactive = sys.stdout.isatty() and (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        or sys.platform == "darwin"
    )
    if not interactive:
        return
    import subprocess
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    try:
        subprocess.Popen([opener, str(path)])
    except OSError as e:
        print(f"Could not open {path}: {e}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "latest":
        return cmd_latest(args)
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
