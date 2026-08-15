"""Streamlit front end for the comparator.

``streamlit run new_comparison.py`` serves this page: pick the NWP model, the
field, the verification analysis (RTMA/URMA) and the time from drop-down menus,
click **Build comparison**, and the engine downloads the GRIB2 files and renders
the figure on the spot — the same :mod:`comparator.pipeline` calls the ``compare``
CLI makes, appending to the same ``runs.jsonl`` manifest. The rendered PNG (or
GIF) is shown in the page with a download button, and stays there while you
change the menus, until the next build replaces it.

Every menu is built from :mod:`comparator.normalize`, so the only choices
offered are ones the model actually produces: GFS lists just its 00/06/12/18Z
cycles, an HRRR 03Z cycle stops at F018 while 00Z runs to F048, and picking a
valid time first lists exactly the forecast runs covering it. Resolution and the
build itself live in :mod:`comparator.runner`, shared with the stdlib HTML form
server — this module is only the widgets.
"""

import os

# Must be set before matplotlib (pulled in by comparator.pipeline) is imported:
# the server has no display, and Herbie's HERBIE_SAVE_DIR would otherwise
# silently override the configured data_dir.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.pop("HERBIE_SAVE_DIR", None)

from datetime import datetime, time as time_cls, timedelta, timezone
from pathlib import Path
import logging
import threading

import streamlit as st

from comparator import appconfig, manifest, normalize, runner, timesel

logger = logging.getLogger("comparator.streamlit_app")

PAGE_TITLE = "John's Real-Time Model Comparator"
PAGE_ICON = "🌡️"
RESULT_KEY = "last_result"

MIME_TYPES = {".png": "image/png", ".gif": "image/gif"}

MODE_LABELS = {
    "single": "Single frame (PNG)",
    "gif": "Animated GIF (every run)",
}
TARGET_LABELS = {
    "latest": "Most recent available",
    "specific": "Specific time",
}
ANCHOR_LABELS = {
    "valid": "Valid time",
    "cycle": "Init + lead",
}

# matplotlib's pyplot interface keeps global figure-manager state that is not
# safe to touch from two threads at once. Streamlit runs each browser session's
# script on its own thread, so concurrent builds serialize here rather than risk
# corrupting each other's in-progress figure.
_BUILD_LOCK = threading.Lock()

# Module globals survive Streamlit's script re-runs (the module stays in
# sys.modules), so logging is wired up exactly once per server process.
_LOGGING_READY = False


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_cfg():
    """Resolve config.yaml / ``DAG_*`` settings, wiring up logging on first use."""
    global _LOGGING_READY
    cfg = appconfig.load_config()
    if not _LOGGING_READY:
        runner.setup_logging(cfg)
        _LOGGING_READY = True
    return cfg


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def var_label(var_key: str) -> str:
    """``"TMP"`` -> ``"TMP — 2 Meter Temperature"`` for the field menu."""
    return f"{var_key} — {normalize.VAR_REGISTRY[var_key]['title']}"


def hour_label(hour: int) -> str:
    return f"{hour:02d}Z"


def run_label(cycle_dt, fxx) -> str:
    return f"F{fxx:03d} — init {cycle_dt:%Y-%m-%d %H}Z"


def units_for(var_key: str) -> str:
    """The display unit of a difference field, e.g. ``"°F"`` for TMP.

    Pulled out of VAR_REGISTRY's colorbar label (``"ΔT (°F)"``) so the error
    metrics carry units without duplicating them here.
    """
    label = normalize.VAR_REGISTRY[var_key]["diff_label"]
    if label.endswith(")") and "(" in label:
        return label[label.rindex("(") + 1:-1]
    return ""


def nearest_index(values, wanted) -> int:
    """Index of the entry closest to *wanted* (ties favor the larger value).

    Used to open a menu on a sensible default — the lead closest to the
    configured rolling default, the cycle closest to the current hour — instead
    of always on its first entry.
    """
    if not values:
        return 0
    return min(range(len(values)), key=lambda i: (abs(values[i] - wanted), -values[i]))


def target_summary(target) -> str:
    """One-line description of what the Build button would produce."""
    parts = [
        f"**{target.model_key.upper()}** {target.var_key} "
        f"vs **{target.verif_key.upper()}**"
    ]
    if target.cycle_dt is not None:
        parts.append(f"init {target.cycle_dt:%Y-%m-%d %H}Z, lead F{target.fxx:03d}")
    parts.append(f"valid **{target.valid_dt:%Y-%m-%d %H}Z**")
    if target.mode == "gif":
        parts.append("one frame per covering run")
    return " · ".join(parts)


def _fmt(value, spec) -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "—"


def _utc_now():
    """Current UTC time as a naive datetime (the pipeline works in naive UTC)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# --------------------------------------------------------------------------- #
# Sidebar: the controls
# --------------------------------------------------------------------------- #
def sidebar_controls(cfg) -> dict:
    """Draw the control panel; return :func:`runner.resolve_target` kwargs."""
    bar = st.sidebar
    bar.header("Comparison")

    model_key = bar.selectbox(
        "NWP model", normalize.forecast_models(), format_func=str.upper,
        help="The forecast verified against the analysis.",
    )
    var_key = bar.selectbox(
        "Field", list(normalize.VAR_REGISTRY), format_func=var_label,
    )
    verifs = list(normalize.VERIFICATION_SOURCES)
    verif_key = bar.selectbox(
        "Verify against", verifs,
        index=verifs.index(cfg.verif) if cfg.verif in verifs else 0,
        format_func=str.upper,
        help="The analysis the forecast is differenced against, on its own grid.",
    )
    mode = bar.radio(
        "Output", list(MODE_LABELS), format_func=MODE_LABELS.get,
        help="A GIF animates every forecast run covering the valid time, so it "
             "downloads many more files and can take several minutes.",
    )

    bar.divider()
    bar.header("Valid time")
    target = bar.radio("Choose", list(TARGET_LABELS), format_func=TARGET_LABELS.get)

    params = dict(model=model_key, var=var_key, verif=verif_key,
                  mode=mode, target=target)
    if target == "latest":
        bar.caption(
            f"Verifies the newest analysis that should be published — "
            f"{cfg.lag_hours} h behind the current hour"
            + (f", at the lead nearest F{cfg.default_lead:03d}." if mode == "single"
               else ".")
        )
        return params

    if mode == "gif":
        params.update(_gif_time_controls(model_key, cfg))
    else:
        params.update(_single_time_controls(model_key, cfg))
    return params


def _anchor_default(cfg):
    """The time the 'most recent available' target would use — menu defaults."""
    return timesel.floor_to_hour(_utc_now()) - timedelta(hours=cfg.lag_hours)


def _valid_time_controls(label, cfg):
    """Date + hour menus for an analysis valid time, defaulted to the newest one."""
    default = _anchor_default(cfg)
    valid_date = st.sidebar.date_input(label, value=default.date())
    hours = list(range(24))
    hour = st.sidebar.selectbox(
        "Valid hour (Z)", hours, index=default.hour, format_func=hour_label,
        help="RTMA/URMA analyses are hourly.",
    )
    return datetime.combine(valid_date, time_cls(hour=hour))


def _single_time_controls(model_key, cfg) -> dict:
    """Menus for a single frame: anchored on the valid time, or on the cycle.

    Both routes end at the same three values (cycle date, init hour, lead) and
    only ever offer combinations *model_key* actually runs.
    """
    anchor = st.sidebar.radio(
        "Specify by", list(ANCHOR_LABELS), format_func=ANCHOR_LABELS.get,
        horizontal=True,
        help="Pick the analysis time and choose among the runs covering it, or "
             "pick a forecast cycle and lead directly.",
    )

    if anchor == "valid":
        valid_dt = _valid_time_controls("Valid date (UTC)", cfg)
        runs = sorted(normalize.find_runs_for_valid_time(model_key, valid_dt),
                      key=lambda run: run[1])
        if not runs:
            raise ValueError(
                f"No {model_key.upper()} cycle covers {valid_dt:%Y-%m-%d %H}Z."
            )
        picked = st.sidebar.selectbox(
            "Forecast run", range(len(runs)),
            index=nearest_index([fxx for _, fxx in runs], cfg.default_lead),
            format_func=lambda i: run_label(*runs[i]),
            help="Every cycle of this model whose forecast reaches the valid time.",
        )
        cycle_dt, fxx = runs[picked]
        return {"date": cycle_dt.date(), "init": cycle_dt.hour, "fxx": fxx}

    default = _anchor_default(cfg)
    init_date = st.sidebar.date_input("Init date (UTC)", value=default.date())
    init_hours = normalize.valid_init_hours(model_key)
    # Open on the most recent cycle hour this model has run today.
    recent = [h for h in init_hours if h <= default.hour] or init_hours
    init = st.sidebar.selectbox(
        "Init hour (Z)", init_hours, index=init_hours.index(recent[-1]),
        format_func=hour_label,
        help=f"{model_key.upper()} runs every "
             f"{normalize.MODEL_FORECAST_META[model_key]['cycle_interval']} h.",
    )
    cycle_dt = datetime.combine(init_date, time_cls(hour=init))
    leads = normalize.valid_forecast_hours(model_key, init)
    fxx = st.sidebar.selectbox(
        "Forecast lead", leads, index=nearest_index(leads, cfg.default_lead),
        format_func=lambda f: f"F{f:03d} — valid "
                              f"{cycle_dt + timedelta(hours=f):%Y-%m-%d %H}Z",
        help=f"The {init:02d}Z cycle runs out to "
             f"F{normalize.max_fxx_for_cycle(model_key, init):03d}.",
    )
    return {"date": init_date, "init": init, "fxx": fxx}


def _gif_time_controls(model_key, cfg) -> dict:
    """Menus for a GIF: the analysis time whose covering runs are animated."""
    valid_dt = _valid_time_controls("Analysis date (UTC)", cfg)
    runs = normalize.find_runs_for_valid_time(model_key, valid_dt)
    st.sidebar.caption(
        f"{len(runs)} {model_key.upper()} run(s) cover this analysis — "
        f"one frame each."
    )
    return {"date": valid_dt.date(), "init": valid_dt.hour}


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
class _StatusLogHandler(logging.Handler):
    """Mirror the engine's log records into a Streamlit status container.

    The pipeline reports progress by logging (it has no idea a browser is
    attached), and a build that downloads several GRIB2 files is slow enough
    that a silent spinner is not good enough.
    """

    def __init__(self, status):
        super().__init__(level=logging.INFO)
        self._status = status

    def emit(self, record):
        try:
            message = record.getMessage()
            self._status.write(
                f"⚠️ {message}" if record.levelno >= logging.WARNING else message
            )
        except Exception:  # never let a UI hiccup break the build
            pass


def build(target, cfg) -> dict:
    """Run the build for *target*, streaming engine progress into the page."""
    label = ("Building the GIF — this can take several minutes…"
             if target.mode == "gif" else "Building the comparison…")
    engine_log = logging.getLogger("comparator")

    with st.status(label, expanded=True) as status:
        handler = _StatusLogHandler(status)
        engine_log.addHandler(handler)
        try:
            with _BUILD_LOCK:
                result = runner.run_build(target, cfg)
        except Exception as e:
            logger.exception("Streamlit build failed: %s", e)
            status.update(label="Build failed", state="error")
            return {"ok": False, "message": f"Internal error: {e}"}
        finally:
            engine_log.removeHandler(handler)

        ok = bool(result.get("ok"))
        status.update(
            label="Build complete" if ok else "Nothing was produced",
            state="complete" if ok else "error",
            expanded=not ok,
        )
    return result


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def render_result(result: dict) -> None:
    """Show the built figure with its download button and error statistics."""
    if not result.get("ok"):
        st.error(result.get("message", "No output was produced."))
        return

    path = Path(result["path"])
    if not path.is_file():
        st.error(f"The output file is no longer on disk: {path}")
        return

    st.subheader(
        f"{result['model'].upper()} {result['var']} − {result['verif'].upper()}, "
        f"valid {result['valid_dt']:%Y-%m-%d %H}Z"
    )
    st.download_button(
        f"⬇  Download {path.suffix.lstrip('.').upper()}",
        data=path.read_bytes(),
        file_name=path.name,
        mime=MIME_TYPES.get(path.suffix.lower(), "application/octet-stream"),
    )
    st.image(str(path), width="stretch")

    if result.get("mode") == "single":
        units = units_for(result["var"])
        suffix = f" ({units})" if units else ""
        left, middle, right = st.columns(3)
        left.metric(f"Mean error{suffix}", _fmt(result.get("mean"), ".2f"),
                    help="Forecast minus analysis, averaged over the grid.")
        middle.metric(f"RMSE{suffix}", _fmt(result.get("rmse"), ".2f"))
        right.metric("Grid points", _fmt(result.get("n"), ",d"),
                     help="Points where both fields have data.")

    st.caption(f"Saved to {path}")


def render_recent(cfg, limit=10) -> None:
    """List the most recent builds from the shared runs.jsonl manifest."""
    rows = manifest.read(cfg.manifest_path, limit=limit)
    if not rows:
        return
    with st.expander(f"Recent runs ({len(rows)})"):
        st.dataframe(
            [
                {
                    "Valid (Z)": (r.get("valid_dt") or "").replace("T", " ")[:16],
                    "Model": (r.get("model") or "").upper(),
                    "Field": r.get("var"),
                    "Verif": (r.get("verif") or "").upper(),
                    "Mode": r.get("mode"),
                    "Lead": f"F{r['fxx']:03d}" if isinstance(r.get("fxx"), int) else "—",
                    "Mean": r.get("mean"),
                    "RMSE": r.get("rmse"),
                    "File": Path(r.get("path") or "").name,
                }
                for r in rows
            ],
            hide_index=True,
            width="stretch",
        )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def render() -> None:
    """Draw the whole page. Streamlit calls this on every interaction."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    cfg = load_cfg()

    st.title(PAGE_TITLE)
    st.caption(
        "Verify an NWP forecast field against the RTMA or URMA analysis. Choose "
        "a run from the menus on the left; the GRIB2 data is downloaded and the "
        "figure rendered on demand."
    )

    try:
        target = runner.resolve_target(cfg=cfg, **sidebar_controls(cfg))
        problem = None
    except (ValueError, timesel.NoDataYet) as e:
        target, problem = None, str(e)

    if problem:
        st.warning(problem)
    else:
        st.markdown(target_summary(target))

    if st.button("Build comparison", type="primary", disabled=target is None):
        st.session_state[RESULT_KEY] = build(target, cfg)

    result = st.session_state.get(RESULT_KEY)
    if result:
        st.divider()
        render_result(result)

    render_recent(cfg)
