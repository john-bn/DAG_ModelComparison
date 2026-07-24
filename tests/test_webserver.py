import io
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from comparator import appconfig, webserver


def _cfg():
    """A minimal resolved Config pointing at temp dirs (no network touched)."""
    base = Path(tempfile.mkdtemp())
    return appconfig.Config(
        data_dir=base / "data", out_dir=base / "figures", log_dir=base / "logs",
        verif="rtma", lag_hours=2, default_lead=24, gif_workers=1,
    )


# --- form / template rendering ---------------------------------------------
def test_form_options_exclude_analysis_sources():
    opts = webserver.form_options_html()
    assert '<option value="hrrr">HRRR</option>' in opts["model"]
    assert '<option value="gfs">GFS</option>' in opts["model"]
    # rtma/urma are verification sources, not selectable forecast models.
    assert 'value="rtma"' not in opts["model"]
    assert 'value="urma"' not in opts["model"]
    # ...but they ARE the verification choices.
    assert 'value="rtma"' in opts["verif"]
    assert 'value="urma"' in opts["verif"]
    assert 'value="TMP"' in opts["var"]


def test_render_index_substitutes_all_tokens():
    html = webserver.render_index_html(action="build.cgi")
    assert 'action="build.cgi"' in html
    assert '<option value="hrrr">HRRR</option>' in html
    assert '<option value="TMP">' in html
    # No template tokens left behind.
    assert "{{" not in html and "}}" not in html


# --- resolve_params: the four target cases ---------------------------------
def test_resolve_single_specific():
    form = {"model": "hrrr", "var": "TMP", "verif": "rtma", "mode": "single",
            "target": "specific", "date": "2026-06-18", "init": "12", "fxx": "24"}
    t = webserver.resolve_params(form, cfg=_cfg(), now_utc=None)
    assert t.model_key == "hrrr" and t.var_key == "TMP" and t.verif_key == "rtma"
    assert t.mode == "single"
    assert t.cycle_dt == datetime(2026, 6, 18, 12, 0)
    assert t.fxx == 24
    assert t.valid_dt == datetime(2026, 6, 19, 12, 0)


def test_resolve_gif_specific_uses_valid_time():
    form = {"model": "hrrr", "var": "TMP", "mode": "gif",
            "target": "specific", "date": "2026-06-18", "init": "12"}
    t = webserver.resolve_params(form, cfg=_cfg())
    assert t.mode == "gif"
    assert t.cycle_dt is None and t.fxx is None
    assert t.valid_dt == datetime(2026, 6, 18, 12, 0)


def test_resolve_single_latest_uses_rolling():
    now = datetime(2026, 6, 18, 14, 30, tzinfo=timezone.utc)
    form = {"model": "hrrr", "var": "TMP", "mode": "single", "target": "latest"}
    t = webserver.resolve_params(form, cfg=_cfg(), now_utc=now)
    # lag=2 -> valid 12:00Z; a covering cycle/lead must reconstruct it.
    assert t.valid_dt == datetime(2026, 6, 18, 12, 0)
    assert t.cycle_dt is not None and t.fxx is not None


def test_resolve_gif_latest_floors_now_minus_lag():
    now = datetime(2026, 6, 18, 14, 30, tzinfo=timezone.utc)
    form = {"model": "hrrr", "var": "TMP", "mode": "gif", "target": "latest"}
    t = webserver.resolve_params(form, cfg=_cfg(), now_utc=now)
    assert t.cycle_dt is None and t.fxx is None
    assert t.valid_dt == datetime(2026, 6, 18, 12, 0)


def test_resolve_defaults_verif_to_config():
    form = {"model": "hrrr", "var": "TMP", "mode": "single", "target": "latest"}
    t = webserver.resolve_params(form, cfg=_cfg(), now_utc=datetime(2026, 6, 18, 14, 0, tzinfo=timezone.utc))
    assert t.verif_key == "rtma"


# --- resolve_params: validation --------------------------------------------
def test_resolve_invalid_model_raises():
    form = {"model": "bogus", "var": "TMP", "mode": "single", "target": "latest"}
    with pytest.raises(ValueError):
        webserver.resolve_params(form, cfg=_cfg())


def test_resolve_specific_requires_date():
    form = {"model": "hrrr", "var": "TMP", "mode": "single", "target": "specific",
            "init": "12", "fxx": "24"}
    with pytest.raises(ValueError):
        webserver.resolve_params(form, cfg=_cfg())


def test_resolve_single_specific_requires_fxx():
    form = {"model": "hrrr", "var": "TMP", "mode": "single", "target": "specific",
            "date": "2026-06-18", "init": "12"}
    with pytest.raises(ValueError):
        webserver.resolve_params(form, cfg=_cfg())


def test_resolve_rejects_bad_init_hour():
    form = {"model": "hrrr", "var": "TMP", "mode": "single", "target": "specific",
            "date": "2026-06-18", "init": "25", "fxx": "24"}
    with pytest.raises(ValueError):
        webserver.resolve_params(form, cfg=_cfg())


# --- safe_output_path -------------------------------------------------------
def test_safe_output_path_accepts_file_in_dir(tmp_path):
    out = tmp_path / "figures"
    out.mkdir()
    f = out / "hrrr_rtma_TMP.png"
    f.write_bytes(b"\x89PNG")
    assert webserver.safe_output_path(out, "hrrr_rtma_TMP.png") == f.resolve()


def test_safe_output_path_rejects_traversal(tmp_path):
    out = tmp_path / "figures"
    out.mkdir()
    (tmp_path / "secret.txt").write_text("nope")
    assert webserver.safe_output_path(out, "../secret.txt") is None
    assert webserver.safe_output_path(out, "/etc/passwd") is None


def test_safe_output_path_missing_file(tmp_path):
    out = tmp_path / "figures"
    out.mkdir()
    assert webserver.safe_output_path(out, "does_not_exist.png") is None


# --- result / error rendering ----------------------------------------------
def test_render_result_ok_embeds_image_and_stats():
    result = {
        "ok": True, "mode": "single", "filename": "hrrr_rtma_TMP.png",
        "model": "hrrr", "var": "TMP", "verif": "rtma",
        "valid_dt": datetime(2026, 6, 19, 12, 0),
        "cycle_dt": datetime(2026, 6, 18, 12, 0), "fxx": 24,
        "mean": 1.23, "rmse": 2.34, "n": 100,
    }
    html = webserver.render_result_html(result)
    assert '<img src="output/hrrr_rtma_TMP.png"' in html
    assert "F024" in html and "1.23" in html and "2.34" in html


def test_render_result_error_escapes():
    html = webserver.render_result_html({"ok": False, "message": "<boom> & fail"})
    assert 'class="error"' in html
    assert "&lt;boom&gt;" in html  # escaped, not injected


# --- CGI protocol -----------------------------------------------------------
def test_cgi_post_runs_build(monkeypatch):
    monkeypatch.setattr(webserver, "run_build", lambda target, cfg: {
        "ok": True, "mode": "single", "filename": "x.png", "model": "hrrr",
        "var": "TMP", "verif": "rtma", "valid_dt": datetime(2026, 6, 19, 12, 0),
        "cycle_dt": datetime(2026, 6, 18, 12, 0), "fxx": 24,
        "mean": 0.0, "rmse": 0.0, "n": 1,
    })
    body = "model=hrrr&var=TMP&verif=rtma&mode=single&target=specific&date=2026-06-18&init=12&fxx=24"
    monkeypatch.setenv("REQUEST_METHOD", "POST")
    monkeypatch.setenv("CONTENT_LENGTH", str(len(body)))
    stdin = io.BytesIO(body.encode("utf-8"))
    stdout = io.StringIO()
    rc = webserver.serve_cgi(_cfg(), stdin=stdin, stdout=stdout)
    out = stdout.getvalue()
    assert rc == 0
    assert "Content-Type: text/html" in out
    assert '<img src="output/x.png"' in out


def test_cgi_post_bad_input_shows_error(monkeypatch):
    body = "model=bogus&var=TMP&mode=single&target=latest"
    monkeypatch.setenv("REQUEST_METHOD", "POST")
    monkeypatch.setenv("CONTENT_LENGTH", str(len(body)))
    stdout = io.StringIO()
    rc = webserver.serve_cgi(_cfg(), stdin=io.BytesIO(body.encode()), stdout=stdout)
    assert rc == 0
    assert 'class="error"' in stdout.getvalue()


def test_cgi_get_serves_form(monkeypatch):
    monkeypatch.setenv("REQUEST_METHOD", "GET")
    stdout = io.StringIO()
    webserver.serve_cgi(_cfg(), action="build.cgi", stdout=stdout)
    out = stdout.getvalue()
    assert "Content-Type: text/html" in out
    assert 'action="build.cgi"' in out
