import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from comparator import appconfig, runner


def _cfg():
    """A minimal resolved Config pointing at temp dirs (no network touched)."""
    base = Path(tempfile.mkdtemp())
    return appconfig.Config(
        data_dir=base / "data", out_dir=base / "figures", log_dir=base / "logs",
        verif="rtma", lag_hours=2, default_lead=24, gif_workers=1,
    )


def _base(**overrides):
    params = dict(model="hrrr", var="TMP", cfg=_cfg())
    params.update(overrides)
    return params


# --- the four target cases --------------------------------------------------
def test_single_specific_typed_values():
    t = runner.resolve_target(**_base(
        mode="single", target="specific", date=date(2026, 6, 18), init=12, fxx=24,
    ))
    assert (t.model_key, t.var_key, t.verif_key, t.mode) == ("hrrr", "TMP", "rtma", "single")
    assert t.cycle_dt == datetime(2026, 6, 18, 12, 0)
    assert t.fxx == 24
    assert t.valid_dt == datetime(2026, 6, 19, 12, 0)


def test_single_specific_string_values():
    # The HTML form hands everything over as strings; same result.
    t = runner.resolve_target(**_base(
        mode="single", target="specific", date="2026-06-18", init="12", fxx="24",
    ))
    assert t.cycle_dt == datetime(2026, 6, 18, 12, 0)
    assert t.valid_dt == datetime(2026, 6, 19, 12, 0)


def test_gif_specific_uses_valid_time():
    t = runner.resolve_target(**_base(
        mode="gif", target="specific", date=date(2026, 6, 18), init=12,
    ))
    assert t.mode == "gif"
    assert t.cycle_dt is None and t.fxx is None
    assert t.valid_dt == datetime(2026, 6, 18, 12, 0)


def test_single_latest_uses_rolling():
    now = datetime(2026, 6, 18, 14, 30, tzinfo=timezone.utc)
    t = runner.resolve_target(**_base(mode="single", target="latest", now_utc=now))
    # lag=2 -> valid 12:00Z; a covering cycle/lead must reconstruct it.
    assert t.valid_dt == datetime(2026, 6, 18, 12, 0)
    assert t.cycle_dt + timedelta(hours=t.fxx) == t.valid_dt


def test_gif_latest_floors_now_minus_lag():
    now = datetime(2026, 6, 18, 14, 30, tzinfo=timezone.utc)
    t = runner.resolve_target(**_base(mode="gif", target="latest", now_utc=now))
    assert t.cycle_dt is None and t.fxx is None
    assert t.valid_dt == datetime(2026, 6, 18, 12, 0)


# --- validation -------------------------------------------------------------
def test_verif_defaults_to_config():
    assert runner.resolve_target(**_base(target="latest")).verif_key == "rtma"
    assert runner.resolve_target(**_base(target="latest", verif="urma")).verif_key == "urma"


def test_zero_lead_is_not_treated_as_missing():
    # F000 is a legitimate answer -- it must not trip the "required field" check.
    t = runner.resolve_target(**_base(
        mode="single", target="specific", date=date(2026, 6, 18), init=0, fxx=0,
    ))
    assert t.fxx == 0
    assert t.valid_dt == t.cycle_dt == datetime(2026, 6, 18, 0, 0)


def test_invalid_model_raises():
    with pytest.raises(ValueError):
        runner.resolve_target(**_base(model="bogus", target="latest"))


def test_specific_requires_date():
    with pytest.raises(ValueError, match="Date"):
        runner.resolve_target(**_base(target="specific", init=12, fxx=24))


def test_single_specific_requires_lead():
    with pytest.raises(ValueError, match="Forecast lead"):
        runner.resolve_target(**_base(target="specific", date="2026-06-18", init=12))


def test_rejects_bad_init_hour():
    with pytest.raises(ValueError):
        runner.resolve_target(**_base(
            target="specific", date="2026-06-18", init=25, fxx=24,
        ))


def test_rejects_negative_lead():
    with pytest.raises(ValueError):
        runner.resolve_target(**_base(
            target="specific", date="2026-06-18", init=12, fxx=-1,
        ))


def test_rejects_unknown_mode_and_target():
    with pytest.raises(ValueError):
        runner.resolve_target(**_base(mode="movie", target="latest"))
    with pytest.raises(ValueError):
        runner.resolve_target(**_base(target="whenever"))


def test_rejects_malformed_date():
    with pytest.raises(ValueError, match="expected YYYY-MM-DD"):
        runner.resolve_target(**_base(
            target="specific", date="18/06/2026", init=12, fxx=24,
        ))


# --- logging setup ----------------------------------------------------------
def test_setup_logging_creates_log_dir_and_is_idempotent(tmp_path):
    import logging

    cfg = appconfig.Config(
        data_dir=tmp_path / "data", out_dir=tmp_path / "figures",
        log_dir=tmp_path / "logs", verif="rtma", lag_hours=2, default_lead=24,
        gif_workers=1,
    )
    root = logging.getLogger()
    saved = list(root.handlers)
    try:
        runner.setup_logging(cfg)
        first = len(root.handlers)
        runner.setup_logging(cfg)
        assert (tmp_path / "logs").is_dir()
        assert len(root.handlers) == first  # handlers replaced, not stacked
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()
        for h in saved:
            root.addHandler(h)
