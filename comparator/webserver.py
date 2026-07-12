"""On-demand web front-end for the comparator.

This replaces the former cron scheduling: instead of a scheduled job picking what
to verify, a person opens an HTML form, enters the values (model, variable,
verification source, mode, target time) and clicks a button. The server then
downloads the GRIB2 files on command and builds the image — exactly the same
:mod:`comparator.pipeline` calls the interactive shim and the CLI make.

Two ways to run the same core logic:

* ``compare-web serve``      — a standalone :class:`ThreadingHTTPServer` for local
  testing on a workstation (and as a fallback if CGI is unavailable).
* ``compare-web cgi``        — the CGI protocol, invoked per request by an
  existing web server (e.g. the company intranet Apache) via a small ``build.cgi``
  wrapper. This is the on-demand, no-daemon deployment: the form and the output
  images live under the user's web-served directory.
* ``compare-web render-form`` — print the concrete static ``index.html`` (with the
  dropdown options filled from the registry) for a CGI deployment.

The heavy scientific stack is imported lazily inside :func:`run_build` (like the
CLI) so ``render-form`` / ``--help`` stay fast and import-light.
"""

import os

# Must be set before matplotlib (pulled in by comparator.pipeline) is imported:
# the server has no display, and Herbie's HERBIE_SAVE_DIR would otherwise
# silently override the configured data_dir (same reason the old cron wrapper
# cleared it).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.pop("HERBIE_SAVE_DIR", None)

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote
import argparse
import html
import logging
import sys

from comparator import appconfig, manifest, normalize, timesel

logger = logging.getLogger("comparator.webserver")

TEMPLATE_PATH = Path(__file__).parent / "web" / "index.html"

_CONTENT_TYPES = {".png": "image/png", ".gif": "image/gif"}


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
# Form → target resolution (mirrors cli._resolve_target, reuses public timesel)
# --------------------------------------------------------------------------- #
def _require(form, key, label):
    value = (form.get(key) or "").strip()
    if not value:
        raise ValueError(f"Missing required field: {label}.")
    return value


def _parse_int(value, label):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a whole number (got {value!r}).")


def resolve_params(form, cfg, now_utc=None) -> ResolvedTarget:
    """Validate a submitted *form* dict and resolve the run target.

    *form* maps field name -> single string value. *now_utc* is injected for
    determinism/testability (defaults to the real current UTC time). Raises
    :class:`ValueError` on any invalid/missing input (the caller turns that into
    a friendly error panel) and :class:`timesel.NoDataYet` when a rolling target
    has no covering cycle yet.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    model_key = normalize.normalize_model_key(_require(form, "model", "Model"))
    var_key = normalize.normalize_var_key(_require(form, "var", "Variable"))
    verif_key = normalize.normalize_verif_key(
        (form.get("verif") or cfg.verif or "").strip() or cfg.verif
    )

    mode = (form.get("mode") or "single").strip().lower()
    if mode not in ("single", "gif"):
        raise ValueError(f"Invalid mode: {mode!r} (expected 'single' or 'gif').")

    target = (form.get("target") or "latest").strip().lower()
    if target not in ("latest", "specific"):
        raise ValueError(f"Invalid target: {target!r} (expected 'latest' or 'specific').")

    common = dict(model_key=model_key, var_key=var_key, verif_key=verif_key, mode=mode)

    if target == "specific":
        date = _require(form, "date", "Date")
        init = _parse_int(_require(form, "init", "Init hour"), "Init hour")
        if not 0 <= init <= 23:
            raise ValueError("Init hour must be between 0 and 23 (Z-time).")
        try:
            anchor = datetime.fromisoformat(f"{date} {init:02d}:00")
        except ValueError:
            raise ValueError(f"Invalid date {date!r} (expected YYYY-MM-DD).")

        if mode == "single":
            fxx = _parse_int(_require(form, "fxx", "Forecast lead"), "Forecast lead")
            if fxx < 0:
                raise ValueError("Forecast lead must be 0 or greater.")
            valid_dt = anchor + timedelta(hours=fxx)
            return ResolvedTarget(**common, cycle_dt=anchor, fxx=fxx, valid_dt=valid_dt)
        # gif: date/init describe the analysis VALID time
        return ResolvedTarget(**common, cycle_dt=None, fxx=None, valid_dt=anchor)

    # Rolling "most recent available".
    if mode == "single":
        cycle_dt, fxx, valid_dt = timesel.resolve_rolling_target(
            model_key, now_utc, lag_hours=cfg.lag_hours, lead_hours=cfg.default_lead,
        )
        return ResolvedTarget(**common, cycle_dt=cycle_dt, fxx=fxx, valid_dt=valid_dt)

    valid_dt = timesel.floor_to_hour(now_utc.replace(tzinfo=None)) - timedelta(hours=cfg.lag_hours)
    return ResolvedTarget(**common, cycle_dt=None, fxx=None, valid_dt=valid_dt)


# --------------------------------------------------------------------------- #
# Build (downloads GRIB2 + renders the image) — heavy import deferred
# --------------------------------------------------------------------------- #
def run_build(target: ResolvedTarget, cfg) -> dict:
    """Download data and render the comparison for *target*.

    Returns a result dict consumed by :func:`render_result_html`:
        {"ok": True, "filename": ..., "valid_dt": ..., ["cycle_dt", "fxx",
         "mean", "rmse", "n"]}
    or {"ok": False, "message": ...} when nothing could be produced.
    Appends a manifest record on success so ``compare list`` also shows
    web-built runs.
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
            "ok": True, "mode": "single", "filename": Path(result.path).name,
            "model": target.model_key, "var": target.var_key,
            "verif": target.verif_key, "valid_dt": target.valid_dt,
            "cycle_dt": target.cycle_dt, "fxx": target.fxx,
            "mean": result.mean, "rmse": result.rmse, "n": result.n,
        }

    gif_path = pipeline.generate_gif(
        target.model_key, target.var_key, target.valid_dt, target.verif_key,
        data_dir=cfg.data_dir, out_dir=cfg.out_dir,
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
        "ok": True, "mode": "gif", "filename": Path(gif_path).name,
        "model": target.model_key, "var": target.var_key,
        "verif": target.verif_key, "valid_dt": target.valid_dt,
    }


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #
def form_options_html():
    """Build the (model, var, verif) <option> lists from the registries.

    Returns a dict of ready-to-inject HTML strings. Analysis-only keys
    (rtma/urma) are excluded from the model list — they are verification
    sources, not forecast models.
    """
    models = [k for k in normalize.MODEL_REGISTRY
              if k not in normalize.VERIFICATION_SOURCES]
    model_opts = "".join(
        f'<option value="{html.escape(k)}">{html.escape(k.upper())}</option>'
        for k in models
    )
    var_opts = "".join(
        f'<option value="{html.escape(k)}">'
        f'{html.escape(k)} — {html.escape(meta["title"])}</option>'
        for k, meta in normalize.VAR_REGISTRY.items()
    )
    verif_opts = "".join(
        f'<option value="{html.escape(v)}">{html.escape(v.upper())}</option>'
        for v in normalize.VERIFICATION_SOURCES
    )
    return {"model": model_opts, "var": var_opts, "verif": verif_opts}


def render_index_html(action="build.cgi") -> str:
    """Render the form page: fill the option lists and the POST *action*."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    opts = form_options_html()
    return (
        template
        .replace("{{MODEL_OPTIONS}}", opts["model"])
        .replace("{{VAR_OPTIONS}}", opts["var"])
        .replace("{{VERIF_OPTIONS}}", opts["verif"])
        .replace("{{ACTION}}", html.escape(action, quote=True))
    )


def _fmt_num(value, spec):
    return format(value, spec) if isinstance(value, (int, float)) else "—"


def render_error_html(message) -> str:
    return f'<div class="error">{html.escape(str(message))}</div>'


def render_result_html(result: dict) -> str:
    """Render the results fragment: the built image + stats, or an error panel."""
    if not result.get("ok"):
        return render_error_html(result.get("message", "No output was produced."))

    filename = result["filename"]
    # filename is engine-generated (safe chars); quote defensively for the URL.
    src = "output/" + quote(filename)
    rows = [
        ("Model", result["model"].upper()),
        ("Variable", result["var"]),
        ("Verified against", result["verif"].upper()),
        ("Valid (Z)", f"{result['valid_dt']:%Y-%m-%d %H:%M}"),
    ]
    if result.get("mode") == "single":
        if result.get("cycle_dt") is not None:
            rows.append(("Init cycle (Z)", f"{result['cycle_dt']:%Y-%m-%d %H}"))
        if isinstance(result.get("fxx"), int):
            rows.append(("Forecast lead", f"F{result['fxx']:03d}"))
        rows.append(("Mean error", _fmt_num(result.get("mean"), ".2f")))
        rows.append(("RMSE", _fmt_num(result.get("rmse"), ".2f")))
        rows.append(("Finite points", _fmt_num(result.get("n"), ",d")))

    stats = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in rows
    )
    alt = html.escape(filename)
    return (
        f'<div class="card">'
        f'<img src="{html.escape(src, quote=True)}" alt="{alt}">'
        f'<table class="stats">{stats}</table>'
        f'<p class="hint">Saved as {alt}</p>'
        f'</div>'
    )


# --------------------------------------------------------------------------- #
# Output file serving (standalone server only; under CGI the web server does it)
# --------------------------------------------------------------------------- #
def safe_output_path(out_dir, name):
    """Resolve *name* under *out_dir*, or return None if it escapes / is missing.

    Rejects ``..``/absolute traversal by requiring the resolved path to stay
    within *out_dir* and to be a regular file.
    """
    out_dir = Path(out_dir).resolve()
    candidate = (out_dir / name).resolve()
    if not candidate.is_relative_to(out_dir):
        return None
    if not candidate.is_file():
        return None
    return candidate


# --------------------------------------------------------------------------- #
# CGI protocol
# --------------------------------------------------------------------------- #
def _flatten(parsed):
    """parse_qs result (name -> [values]) -> name -> first value."""
    return {k: v[0] for k, v in parsed.items() if v}


def serve_cgi(cfg, action="build.cgi", now_utc=None, stdin=None, stdout=None) -> int:
    """Handle a single CGI request from the process environment.

    On POST: parse the form, resolve, build, and emit the result HTML. On GET:
    emit the form (handy if build.cgi is visited directly). Always returns 0 —
    errors are shown in-page, not as HTTP failures, so the browser sees them.
    """
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout
    method = os.environ.get("REQUEST_METHOD", "GET").upper()

    if method != "POST":
        _write_cgi(stdout, render_index_html(action=action))
        return 0

    try:
        length = int(os.environ.get("CONTENT_LENGTH") or 0)
    except ValueError:
        length = 0
    body = stdin.read(length) if length > 0 else b""
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    form = _flatten(parse_qs(body))

    try:
        target = resolve_params(form, cfg, now_utc=now_utc)
        result = run_build(target, cfg)
        page = render_result_html(result)
    except (ValueError, timesel.NoDataYet) as e:
        page = render_error_html(e)
    except Exception as e:  # unexpected — log for the web server error log
        logger.exception("Web build failed: %s", e)
        page = render_error_html(f"Internal error: {e}")

    _write_cgi(stdout, page)
    return 0


def _write_cgi(stdout, body):
    """Emit a minimal CGI response (headers + body) to *stdout*."""
    stdout.write("Content-Type: text/html; charset=utf-8\r\n")
    stdout.write("Cache-Control: no-store\r\n")
    stdout.write("\r\n")
    stdout.write(body)
    stdout.flush()


# --------------------------------------------------------------------------- #
# Standalone development server
# --------------------------------------------------------------------------- #
def _make_handler(cfg):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ComparatorWeb/1.0"

        def log_message(self, fmt, *args):  # route to logging instead of stderr
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send_html(self, body, status=200):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send_html(render_index_html(action="/build"))
                return
            if path.startswith("/output/"):
                self._serve_output(path[len("/output/"):])
                return
            self.send_error(404, "Not found")

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path != "/build":
                self.send_error(404, "Not found")
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length).decode("utf-8", errors="replace") if length else ""
            form = _flatten(parse_qs(body))
            try:
                target = resolve_params(form, cfg)
                result = run_build(target, cfg)
                page = render_result_html(result)
            except (ValueError, timesel.NoDataYet) as e:
                page = render_error_html(e)
            except Exception as e:
                logger.exception("Web build failed: %s", e)
                page = render_error_html(f"Internal error: {e}")
            self._send_html(page)

        def _serve_output(self, name):
            path = safe_output_path(cfg.out_dir, name)
            if path is None:
                self.send_error(404, "Not found")
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type",
                             _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def run_server(cfg, host="127.0.0.1", port=8000):
    httpd = ThreadingHTTPServer((host, port), _make_handler(cfg))
    logger.info("Serving the comparison form on http://%s:%d/  (Ctrl-C to stop)",
                host, port)
    logger.info("data_dir=%s  out_dir=%s", cfg.data_dir, cfg.out_dir)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        httpd.server_close()
    return 0


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare-web",
        description="On-demand web form for the NWP-vs-analysis comparator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--config", help="Path to config.yaml")
        p.add_argument("--data-dir", help="GRIB/cache directory (overrides config)")
        p.add_argument("--out-dir", help="Figure/manifest output dir (overrides config)")
        p.add_argument("--log-dir", help="Log directory (overrides config)")

    serve = sub.add_parser("serve", help="Run the standalone dev server (local testing).")
    add_common(serve)
    serve.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1).")
    serve.add_argument("--port", type=int, default=8000, help="Port (default 8000).")

    cgi = sub.add_parser("cgi", help="Handle one CGI request (invoked by build.cgi).")
    add_common(cgi)

    rf = sub.add_parser("render-form", help="Print the static index.html for a CGI deploy.")
    rf.add_argument("--action", default="build.cgi",
                    help="Form POST target relative to the page (default build.cgi).")

    return parser


def _load_cfg(args):
    return appconfig.load_config(
        config_path=getattr(args, "config", None),
        data_dir=getattr(args, "data_dir", None),
        out_dir=getattr(args, "out_dir", None),
        log_dir=getattr(args, "log_dir", None),
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "render-form":
        sys.stdout.write(render_index_html(action=args.action))
        return 0

    # serve/cgi: log to stderr (under CGI this lands in the web server's error log).
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)sZ %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    cfg = _load_cfg(args)

    if args.command == "cgi":
        return serve_cgi(cfg)
    if args.command == "serve":
        return run_server(cfg, host=args.host, port=args.port)
    return 2


if __name__ == "__main__":
    sys.exit(main())
