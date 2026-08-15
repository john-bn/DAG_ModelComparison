"""Front-end-independent request resolution, logging setup, and build.

Every interface answers the same five questions — model, variable, verification
source, output mode, target time — and then needs the identical work done:
validate the answers, decide which cycle/lead to fetch, call
:mod:`comparator.pipeline`, and append the run to ``runs.jsonl``. That logic
lives here so a front end is only widgets: the Streamlit app
(:mod:`comparator.streamlit_app`) and the stdlib HTML form server
(:mod:`comparator.webserver`) both hand their answers to :func:`resolve_target`
and the resulting :class:`ResolvedTarget` to :func:`run_build`.

Values arrive already typed from Streamlit (``date``/``int``) and as strings
from an HTML form, so the parsing helpers accept both.

The heavy scientific stack is imported lazily inside :func:`run_build` — the
Streamlit sidebar re-resolves a target on every keystroke, and ``compare-web
render-form`` never builds anything, so neither should pay for matplotlib.
"""

from dataclasses import dataclass
from datetime import date as date_cls, datetime, timedelta, timezone
from pathlib import Path
import logging
import sys
import time

from comparator import manifest, normalize, timesel

MODES = ("single", "gif")
TARGETS = ("latest", "specific")


# --------------------------------------------------------------------------- #
# Resolved target
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResolvedTarget:
    """A validated, fully-resolved comparison request.

    ``cycle_dt``/``fxx`` are ``None`` for GIF mode (which spans every covering
    cycle); ``valid_dt`` is always set.
    """

    model_key: str
    var_key: str
    verif_key: str
    mode: str            # "single" | "gif"
    cycle_dt: datetime | None
    fxx: int | None
    valid_dt: datetime


# --------------------------------------------------------------------------- #
# Input parsing
# --------------------------------------------------------------------------- #
def _require(value, label):
    """Return *value*, or raise if it is missing (``None`` / blank string).

    ``0`` is a legitimate answer for an hour or a forecast lead, so emptiness is
    tested explicitly rather than by truthiness.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Missing required field: {label}.")
    return value


def _parse_int(value, label):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a whole number (got {value!r}).")


def _parse_date(value, label):
    """Accept a ``date``/``datetime`` (Streamlit) or a YYYY-MM-DD string (form)."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date_cls):
        return value
    try:
        return date_cls.fromisoformat(str(value).strip())
    except ValueError:
        raise ValueError(f"Invalid {label.lower()} {value!r} (expected YYYY-MM-DD).")


def _anchor_datetime(date, hour, date_label, hour_label):
    """Combine a date and an hour-of-day into a naive-UTC datetime."""
    day = _parse_date(_require(date, date_label), date_label)
    hour = _parse_int(_require(hour, hour_label), hour_label)
    if not 0 <= hour <= 23:
        raise ValueError(f"{hour_label} must be between 0 and 23 (Z-time).")
    return datetime(day.year, day.month, day.day, hour)


# --------------------------------------------------------------------------- #
# Target resolution
# --------------------------------------------------------------------------- #
def resolve_target(
    *,
    model,
    var,
    cfg,
    verif=None,
    mode="single",
    target="latest",
    date=None,
    init=None,
    fxx=None,
    now_utc=None,
) -> ResolvedTarget:
    """Validate the submitted choices and resolve what to build.

    *target* ``"latest"`` picks the newest analysis that should already be
    published (``now - cfg.lag_hours``); ``"specific"`` uses *date* + *init*,
    which anchor the forecast **cycle** in single mode and the analysis **valid
    time** in GIF mode (a GIF spans every cycle covering that analysis).

    *now_utc* is injected for determinism/testability. Raises
    :class:`ValueError` on any invalid or missing input and
    :class:`comparator.timesel.NoDataYet` when a rolling target has no covering
    cycle yet.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    model_key = normalize.normalize_model_key(_require(model, "Model"))
    var_key = normalize.normalize_var_key(_require(var, "Variable"))
    verif_key = normalize.normalize_verif_key(
        verif if (verif or "").strip() else cfg.verif
    )

    mode = str(mode or "single").strip().lower()
    if mode not in MODES:
        raise ValueError(f"Invalid mode: {mode!r} (expected 'single' or 'gif').")

    target = str(target or "latest").strip().lower()
    if target not in TARGETS:
        raise ValueError(f"Invalid target: {target!r} (expected 'latest' or 'specific').")

    common = dict(model_key=model_key, var_key=var_key, verif_key=verif_key, mode=mode)

    if target == "specific":
        if mode == "single":
            anchor = _anchor_datetime(date, init, "Date", "Init hour")
            lead = _parse_int(_require(fxx, "Forecast lead"), "Forecast lead")
            if lead < 0:
                raise ValueError("Forecast lead must be 0 or greater.")
            return ResolvedTarget(
                **common, cycle_dt=anchor, fxx=lead, valid_dt=anchor + timedelta(hours=lead)
            )
        # gif: date/init describe the analysis VALID time
        anchor = _anchor_datetime(date, init, "Date", "Valid hour")
        return ResolvedTarget(**common, cycle_dt=None, fxx=None, valid_dt=anchor)

    # Rolling "most recent available".
    if mode == "single":
        cycle_dt, lead, valid_dt = timesel.resolve_rolling_target(
            model_key, now_utc, lag_hours=cfg.lag_hours, lead_hours=cfg.default_lead,
        )
        return ResolvedTarget(**common, cycle_dt=cycle_dt, fxx=lead, valid_dt=valid_dt)

    valid_dt = timesel.floor_to_hour(now_utc.replace(tzinfo=None)) \
        - timedelta(hours=cfg.lag_hours)
    return ResolvedTarget(**common, cycle_dt=None, fxx=None, valid_dt=valid_dt)


# --------------------------------------------------------------------------- #
# Build (downloads GRIB2 + renders the image) — heavy import deferred
# --------------------------------------------------------------------------- #
def run_build(target: ResolvedTarget, cfg) -> dict:
    """Download data and render the comparison for *target*.

    Returns a result dict the front ends render:
        {"ok": True, "mode": ..., "path": ..., "filename": ..., "model": ...,
         "var": ..., "verif": ..., "valid_dt": ..., ["cycle_dt", "fxx", "mean",
         "rmse", "n"]}
    or {"ok": False, "message": ...} when nothing could be produced. Appends a
    manifest record on success, so ``compare list`` also sees runs built from a
    browser.
    """
    from comparator import pipeline  # heavy scientific stack, deferred

    ts = datetime.now(timezone.utc).replace(tzinfo=None)  # naive UTC, uniform manifest

    if target.mode == "single":
        result = pipeline.generate_comparison_frame(
            target.model_key, target.var_key, target.cycle_dt, target.fxx,
            target.verif_key, data_dir=cfg.data_dir, out_dir=cfg.out_dir,
        )
        if result is None:
            return {"ok": False, "message": (
                f"No {target.verif_key.upper()} / {target.model_key.upper()} data "
                f"available yet for valid {target.valid_dt:%Y-%m-%d %H}Z. "
                "The target may be too recent — try again shortly."
            )}
        manifest.append(cfg.manifest_path, manifest.make_record(
            ts_utc=ts, model=target.model_key, var=target.var_key,
            verif=target.verif_key, mode="single", path=result.path,
            cycle_dt=target.cycle_dt, fxx=target.fxx, valid_dt=target.valid_dt,
            mean=result.mean, rmse=result.rmse, n=result.n,
        ))
        return {
            "ok": True, "mode": "single", "path": str(result.path),
            "filename": Path(result.path).name,
            "model": target.model_key, "var": target.var_key,
            "verif": target.verif_key, "valid_dt": target.valid_dt,
            "cycle_dt": target.cycle_dt, "fxx": target.fxx,
            "mean": result.mean, "rmse": result.rmse, "n": result.n,
        }

    gif_path = pipeline.generate_gif(
        target.model_key, target.var_key, target.valid_dt, target.verif_key,
        data_dir=cfg.data_dir, out_dir=cfg.out_dir, max_workers=cfg.gif_workers,
    )
    if gif_path is None:
        return {"ok": False, "message": (
            f"No {target.model_key.upper()} runs cover the "
            f"{target.verif_key.upper()} analysis at "
            f"{target.valid_dt:%Y-%m-%d %H}Z, or no frames could be built."
        )}
    manifest.append(cfg.manifest_path, manifest.make_record(
        ts_utc=ts, model=target.model_key, var=target.var_key,
        verif=target.verif_key, mode="gif", path=gif_path, valid_dt=target.valid_dt,
    ))
    return {
        "ok": True, "mode": "gif", "path": str(gif_path),
        "filename": Path(gif_path).name,
        "model": target.model_key, "var": target.var_key,
        "verif": target.verif_key, "valid_dt": target.valid_dt,
    }


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(cfg, level="INFO"):
    """Configure root logging to a rotating file + stderr, timestamped in UTC.

    Shared by every entry point so a run is traceable in ``comparison.log``
    whether it was started by cron, a terminal, or a click in the browser.
    """
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
