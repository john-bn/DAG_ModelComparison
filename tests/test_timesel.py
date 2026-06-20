from datetime import datetime, timedelta, timezone

import pytest

from comparator import normalize, timesel


def test_valid_time_is_floored_now_minus_lag():
    now = datetime(2026, 6, 18, 12, 59, 30)
    cycle_dt, fxx, valid_dt = timesel.resolve_rolling_target("hrrr", now, lag_hours=2)
    assert valid_dt == datetime(2026, 6, 18, 10, 0, 0)
    # The chosen cycle + lead must reconstruct the valid time exactly.
    assert cycle_dt + timedelta(hours=fxx) == valid_dt


def test_tz_aware_now_yields_naive_valid_dt():
    now = datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc)
    _, _, valid_dt = timesel.resolve_rolling_target("hrrr", now, lag_hours=1)
    assert valid_dt.tzinfo is None
    assert valid_dt == datetime(2026, 6, 18, 11, 0)


def test_default_lead_picks_run_closest_to_24():
    now = datetime(2026, 6, 18, 12, 0)
    cycle_dt, fxx, valid_dt = timesel.resolve_rolling_target("hrrr", now, lag_hours=2)
    runs = normalize.find_runs_for_valid_time("hrrr", valid_dt)
    expected = min(runs, key=lambda cf: (abs(cf[1] - 24), -cf[1]))
    assert (cycle_dt, fxx) == expected


def test_lead_override_picks_short_lead():
    now = datetime(2026, 6, 18, 12, 0)
    cycle_dt, fxx, valid_dt = timesel.resolve_rolling_target(
        "hrrr", now, lag_hours=2, lead_hours=6
    )
    assert fxx == 6
    assert cycle_dt + timedelta(hours=6) == valid_dt


def test_tie_breaks_toward_longer_lead():
    # valid 10Z has both F18 and F22 available; target 20 is equidistant,
    # so the tie-break favors the longer lead (F22).
    now = datetime(2026, 6, 18, 12, 0)
    _, fxx, _ = timesel.resolve_rolling_target("hrrr", now, lag_hours=2, lead_hours=20)
    assert fxx == 22


def test_no_runs_raises_nodatayet(monkeypatch):
    monkeypatch.setattr(normalize, "find_runs_for_valid_time", lambda *a, **k: [])
    with pytest.raises(timesel.NoDataYet):
        timesel.resolve_rolling_target("hrrr", datetime(2026, 6, 18, 12, 0))
