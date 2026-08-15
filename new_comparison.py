# new_comparison.py
#
# Driver for the Streamlit UI — this is the script Streamlit runs, and (when
# started with a plain interpreter) the launcher that re-execs itself under
# Streamlit, so both of these do the same thing:
#
#     streamlit run new_comparison.py
#     python new_comparison.py
#
# The page itself lives in comparator.streamlit_app, the comparison engine in
# comparator.pipeline, and everything a front end shares — validating the
# choices, resolving which cycle covers a valid time, writing the run manifest —
# in comparator.runner.
#
# The unattended interfaces are unchanged and use the same engine: `compare`
# (comparator.cli) for cron and the terminal, `compare-web` (comparator.webserver)
# for the dependency-free HTML form server behind the intranet httpd.

import sys

from comparator.streamlit_app import render


def _running_under_streamlit() -> bool:
    """True when this file is being executed by Streamlit (or its test runner)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:  # streamlit too old for this API — fall back to the runtime
        from streamlit import runtime
        return runtime.exists()
    return get_script_run_ctx(suppress_warning=True) is not None


def _relaunch_under_streamlit() -> int:
    """Re-exec this file via `streamlit run`, forwarding any Streamlit options."""
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
    return stcli.main()


if __name__ == "__main__":
    if _running_under_streamlit():
        render()
    else:
        sys.exit(_relaunch_under_streamlit())
