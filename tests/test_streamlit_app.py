import base64
from datetime import datetime

import pytest

pytest.importorskip("streamlit", reason="Streamlit UI not installed in this env")

from streamlit.testing.v1 import AppTest

from comparator import runner, streamlit_app as app

APP_FILE = "new_comparison.py"

# A real (1x1) PNG, so the result panel has an actual image file to render.
_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
    "hQGAhKmMIQAAAABJRU5ErkJggg=="
)


# --- formatting helpers -----------------------------------------------------
def test_var_label_pairs_key_with_title():
    assert app.var_label("TMP") == "TMP — 2 Meter Temperature"


def test_units_for_reads_the_colorbar_label():
    assert app.units_for("TMP") == "°F"
    assert app.units_for("WIND") == "mph"
    assert app.units_for("VIS") == "SM"


def test_run_and_hour_labels():
    assert app.hour_label(3) == "03Z"
    assert app.run_label(datetime(2026, 6, 18, 12), 24) == "F024 — init 2026-06-18 12Z"


def test_nearest_index_picks_closest_then_larger():
    assert app.nearest_index([0, 6, 12, 18], 13) == 2
    assert app.nearest_index([0, 12, 24, 36], 24) == 2
    # Equidistant -> the longer lead wins (the more interesting comparison).
    assert app.nearest_index([12, 36], 24) == 1
    assert app.nearest_index([], 24) == 0


def test_target_summary_mentions_cycle_only_for_single_frames():
    single = runner.ResolvedTarget(
        model_key="hrrr", var_key="TMP", verif_key="rtma", mode="single",
        cycle_dt=datetime(2026, 6, 18, 12), fxx=24,
        valid_dt=datetime(2026, 6, 19, 12),
    )
    text = app.target_summary(single)
    assert "HRRR" in text and "RTMA" in text and "F024" in text
    assert "2026-06-19 12Z" in text

    gif = runner.ResolvedTarget(
        model_key="hrrr", var_key="TMP", verif_key="urma", mode="gif",
        cycle_dt=None, fxx=None, valid_dt=datetime(2026, 6, 19, 12),
    )
    text = app.target_summary(gif)
    assert "init" not in text and "covering run" in text


# --- the page itself (rendered headlessly; nothing is fetched or built) -----
@pytest.fixture
def isolated_app(tmp_path, monkeypatch):
    """Point the app's directories at a temp dir so nothing lands in the repo."""
    monkeypatch.setenv("DAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DAG_OUT_DIR", str(tmp_path / "figures"))
    monkeypatch.setenv("DAG_LOG_DIR", str(tmp_path / "logs"))
    for var in ("DAG_CONFIG", "DAG_VERIF", "DAG_LAG_HOURS", "DAG_DEFAULT_LEAD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    return AppTest.from_file(_app_path(), default_timeout=60).run()


def _app_path():
    from pathlib import Path
    return str(Path(__file__).resolve().parent.parent / APP_FILE)


def test_page_renders_without_exception(isolated_app):
    assert not isolated_app.exception
    assert isolated_app.title[0].value == app.PAGE_TITLE


def test_model_menu_offers_forecast_models_not_analyses(isolated_app):
    # AppTest reports a menu's options as the labels the user sees.
    options = isolated_app.sidebar.selectbox[0].options
    assert "HRRR" in options and "GFS" in options
    assert "RTMA" not in options and "URMA" not in options


def test_verification_menu_offers_rtma_and_urma(isolated_app):
    assert list(isolated_app.sidebar.selectbox[2].options) == ["RTMA", "URMA"]


def test_build_button_is_enabled_with_a_resolved_target(isolated_app):
    assert isolated_app.button[0].label == "Build comparison"
    assert isolated_app.button[0].disabled is False
    # The resolved rolling target is summarized above the button.
    assert "valid" in isolated_app.markdown[0].value


def test_specific_valid_time_lists_only_covering_runs(isolated_app):
    from comparator import normalize

    at = isolated_app
    at.sidebar.selectbox[0].set_value("gfs").run()           # model
    at.sidebar.radio[1].set_value("specific").run()          # valid time: specific
    assert not at.exception

    valid_dt = datetime.combine(
        at.sidebar.date_input[0].value, datetime.min.time()
    ).replace(hour=at.sidebar.selectbox[3].value)             # valid hour menu

    run_menu = at.sidebar.selectbox[-1]
    assert run_menu.label == "Forecast run"
    assert len(run_menu.options) == len(
        normalize.find_runs_for_valid_time("gfs", valid_dt)
    )
    # Every offered run is a real GFS cycle (00/06/12/18Z) reaching that time.
    for label in run_menu.options:
        init_hour = int(label.split("init ")[1][-3:-1])
        assert init_hour in (0, 6, 12, 18)


def test_result_panel_shows_figure_download_and_stats(tmp_path, monkeypatch):
    """A finished build renders as the image, a download button, and error stats."""
    monkeypatch.setenv("DAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DAG_OUT_DIR", str(tmp_path / "figures"))
    monkeypatch.setenv("DAG_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.chdir(tmp_path)

    figure = tmp_path / "hrrr_rtma_TMP_init20260618_12Z_F024_valid20260619_1200Z.png"
    figure.write_bytes(_ONE_PIXEL_PNG)

    at = AppTest.from_file(_app_path(), default_timeout=60)
    at.session_state[app.RESULT_KEY] = {
        "ok": True, "mode": "single", "path": str(figure), "filename": figure.name,
        "model": "hrrr", "var": "TMP", "verif": "rtma",
        "valid_dt": datetime(2026, 6, 19, 12), "cycle_dt": datetime(2026, 6, 18, 12),
        "fxx": 24, "mean": 0.229, "rmse": 2.109, "n": 1905141,
    }
    at.run()

    assert not at.exception
    assert len(at.get("image")) == 1
    assert at.get("download_button")[0].label.endswith("Download PNG")
    assert [(m.label, m.value) for m in at.get("metric")] == [
        ("Mean error (°F)", "0.23"),
        ("RMSE (°F)", "2.11"),
        ("Grid points", "1,905,141"),
    ]
    assert at.subheader[0].value == "HRRR TMP − RTMA, valid 2026-06-19 12Z"


def test_build_button_runs_the_engine_and_streams_its_log(tmp_path, monkeypatch):
    """Clicking Build calls runner.run_build once and shows what it produced.

    The engine itself is stubbed out — this is about the wiring: the resolved
    target reaching the build, its log records reaching the status container,
    and the result landing in session state.
    """
    import logging

    monkeypatch.setenv("DAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DAG_OUT_DIR", str(tmp_path / "figures"))
    monkeypatch.setenv("DAG_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.chdir(tmp_path)

    figure = tmp_path / "hrrr_rtma_TMP.png"
    figure.write_bytes(_ONE_PIXEL_PNG)
    calls = []

    def fake_build(target, cfg):
        calls.append(target)
        logging.getLogger("comparator.pipeline").info("Saved frame: %s", figure)
        return {
            "ok": True, "mode": "single", "path": str(figure),
            "filename": figure.name, "model": target.model_key,
            "var": target.var_key, "verif": target.verif_key,
            "valid_dt": target.valid_dt, "cycle_dt": target.cycle_dt,
            "fxx": target.fxx, "mean": 1.0, "rmse": 2.0, "n": 3,
        }

    monkeypatch.setattr(runner, "run_build", fake_build)

    at = AppTest.from_file(_app_path(), default_timeout=60).run()
    at.button[0].click().run()

    assert not at.exception
    assert len(calls) == 1 and calls[0].model_key == "hrrr"
    assert at.session_state[app.RESULT_KEY]["ok"] is True
    assert len(at.get("image")) == 1
    # The engine's progress line was mirrored into the page.
    assert any("Saved frame" in md.value for md in at.get("markdown"))


def test_result_panel_reports_a_failed_build(tmp_path, monkeypatch):
    monkeypatch.setenv("DAG_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.chdir(tmp_path)

    at = AppTest.from_file(_app_path(), default_timeout=60)
    at.session_state[app.RESULT_KEY] = {"ok": False, "message": "No RTMA data yet."}
    at.run()

    assert not at.exception
    assert not at.get("image")
    assert at.error[0].value == "No RTMA data yet."


def test_cycle_anchored_leads_stop_at_the_cycle_maximum(isolated_app):
    at = isolated_app
    at.sidebar.selectbox[0].set_value("hrrr").run()
    at.sidebar.radio[1].set_value("specific").run()
    at.sidebar.radio[2].set_value("cycle").run()              # specify by init+lead
    assert not at.exception

    at.sidebar.selectbox[3].set_value(13).run()               # a non-extended cycle
    leads = at.sidebar.selectbox[4]
    assert leads.label == "Forecast lead"
    assert leads.options[-1].startswith("F018")               # 13Z HRRR stops at F18

    at.sidebar.selectbox[3].set_value(12).run()               # an extended cycle
    assert at.sidebar.selectbox[4].options[-1].startswith("F048")
