"""Comprehensive tests for find_runs_for_valid_time and MODEL_FORECAST_META.

Covers: cycle-interval filtering, extended-vs-base max_fxx, result ordering,
edge cases (midnight boundary, fxx=0), all registered models, and error handling.
"""

from datetime import datetime, timedelta

import pytest

from comparator.normalize import (
    MODEL_FORECAST_META,
    MODEL_REGISTRY,
    find_runs_for_valid_time,
)


# ---------------------------------------------------------------------------
# MODEL_FORECAST_META structure validation
# ---------------------------------------------------------------------------

class TestForecastMetaRegistry:
    """Every model (except rtma) should have forecast metadata."""

    def test_every_nwp_model_has_meta(self):
        """All non-RTMA models in MODEL_REGISTRY should appear in META."""
        for key in MODEL_REGISTRY:
            if key == "rtma":
                continue
            assert key in MODEL_FORECAST_META, (
                f"Model '{key}' is in MODEL_REGISTRY but missing from "
                f"MODEL_FORECAST_META"
            )

    @pytest.mark.parametrize("model_key", list(MODEL_FORECAST_META))
    def test_required_fields(self, model_key):
        meta = MODEL_FORECAST_META[model_key]
        assert "cycle_interval" in meta
        assert "max_fxx" in meta
        assert isinstance(meta["cycle_interval"], int)
        assert isinstance(meta["max_fxx"], int)
        assert meta["cycle_interval"] > 0
        assert meta["max_fxx"] > 0

    @pytest.mark.parametrize("model_key", list(MODEL_FORECAST_META))
    def test_extended_cycles_consistency(self, model_key):
        """If extended_cycles is set, base_max_fxx must also be present and < max_fxx."""
        meta = MODEL_FORECAST_META[model_key]
        if "extended_cycles" in meta:
            assert "base_max_fxx" in meta, (
                f"Model '{model_key}' has extended_cycles but no base_max_fxx"
            )
            assert meta["base_max_fxx"] < meta["max_fxx"]
            # Extended cycles should be valid hours (0–23)
            for h in meta["extended_cycles"]:
                assert 0 <= h <= 23

    def test_cycle_interval_divides_24(self):
        """Cycle intervals should divide evenly into 24 hours."""
        for key, meta in MODEL_FORECAST_META.items():
            interval = meta["cycle_interval"]
            assert 24 % interval == 0, (
                f"Model '{key}' cycle_interval={interval} does not divide 24"
            )


# ---------------------------------------------------------------------------
# find_runs_for_valid_time: basic behavior
# ---------------------------------------------------------------------------

class TestFindRunsBasic:
    def test_returns_list_of_tuples(self):
        runs = find_runs_for_valid_time("hrrr", datetime(2026, 4, 6, 12))
        assert isinstance(runs, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in runs)

    def test_all_runs_hit_valid_time(self):
        """Every (cycle_dt, fxx) pair must satisfy cycle_dt + fxx == valid_dt."""
        valid_dt = datetime(2026, 4, 6, 12)
        runs = find_runs_for_valid_time("hrrr", valid_dt)
        for cycle_dt, fxx in runs:
            assert cycle_dt + timedelta(hours=fxx) == valid_dt

    def test_sorted_oldest_first(self):
        runs = find_runs_for_valid_time("hrrr", datetime(2026, 4, 6, 12))
        cycle_dts = [c for c, _ in runs]
        assert cycle_dts == sorted(cycle_dts)

    def test_fxx_zero_always_included(self):
        """The analysis itself (fxx=0) should always be a valid run."""
        valid_dt = datetime(2026, 4, 6, 12)
        runs = find_runs_for_valid_time("hrrr", valid_dt)
        assert (valid_dt, 0) in runs

    def test_no_negative_fxx(self):
        runs = find_runs_for_valid_time("hrrr", datetime(2026, 4, 6, 12))
        for _, fxx in runs:
            assert fxx >= 0

    def test_no_duplicate_runs(self):
        runs = find_runs_for_valid_time("hrrr", datetime(2026, 4, 6, 12))
        assert len(runs) == len(set(runs))


# ---------------------------------------------------------------------------
# HRRR-specific tests (extended cycles)
# ---------------------------------------------------------------------------

class TestHRRR:
    """HRRR: hourly cycles, 18h base / 48h for 00/06/12/18Z."""

    def test_hrrr_max_fxx_from_extended_cycle(self):
        """The oldest run should be from an extended cycle 48h back."""
        valid_dt = datetime(2026, 4, 6, 12)
        runs = find_runs_for_valid_time("hrrr", valid_dt)
        oldest_cycle, oldest_fxx = runs[0]
        assert oldest_fxx == 48
        assert oldest_cycle.hour == 12  # 12Z is an extended cycle

    def test_hrrr_non_extended_cycle_limited_to_18h(self):
        """Non-extended cycles (e.g. 01Z, 02Z) should max out at 18h."""
        valid_dt = datetime(2026, 4, 6, 12)
        runs = find_runs_for_valid_time("hrrr", valid_dt)
        for cycle_dt, fxx in runs:
            if cycle_dt.hour not in [0, 6, 12, 18]:
                assert fxx <= 18, (
                    f"Non-extended HRRR cycle {cycle_dt.hour}Z has fxx={fxx}"
                )

    def test_hrrr_extended_cycles_can_exceed_18h(self):
        """Extended cycles (00/06/12/18Z) CAN have fxx > 18."""
        valid_dt = datetime(2026, 4, 6, 12)
        runs = find_runs_for_valid_time("hrrr", valid_dt)
        extended_fxx = [
            fxx for cycle_dt, fxx in runs
            if cycle_dt.hour in [0, 6, 12, 18] and fxx > 18
        ]
        assert len(extended_fxx) > 0

    def test_hrrr_count(self):
        """HRRR targeting 12Z: 4 extended cycles + 18 base-range hourly cycles + 2 extended at fxx<=18 = 24 total."""
        valid_dt = datetime(2026, 4, 6, 12)
        runs = find_runs_for_valid_time("hrrr", valid_dt)
        # Extended cycles: 12Z-48h, 18Z-42h, 00Z-36h, 06Z-30h, 12Z-24h, 18Z-18h → 6 at extended range
        # Plus 18 hourly (19Z–12Z next day) at ≤18h
        # But some overlap: 00Z, 06Z, 12Z at ≤18h are also extended
        # Total should be 24 (verified empirically)
        assert len(runs) == 24


# ---------------------------------------------------------------------------
# Model with simple (non-extended) cycles
# ---------------------------------------------------------------------------

class TestNAM12k:
    """NAM12k: 6-hourly cycles, max 84h."""

    def test_nam12k_cycle_interval(self):
        valid_dt = datetime(2026, 4, 6, 12)
        runs = find_runs_for_valid_time("nam12k", valid_dt)
        for cycle_dt, _ in runs:
            assert cycle_dt.hour % 6 == 0

    def test_nam12k_max_fxx(self):
        valid_dt = datetime(2026, 4, 6, 12)
        runs = find_runs_for_valid_time("nam12k", valid_dt)
        max_fxx = max(fxx for _, fxx in runs)
        assert max_fxx == 84

    def test_nam12k_count(self):
        """84h / 6h intervals + 1 = 15 runs."""
        valid_dt = datetime(2026, 4, 6, 12)
        runs = find_runs_for_valid_time("nam12k", valid_dt)
        assert len(runs) == 15


class TestGFS:
    """GFS: 6-hourly cycles, max 384h."""

    def test_gfs_max_fxx(self):
        valid_dt = datetime(2026, 4, 6, 12)
        runs = find_runs_for_valid_time("gfs", valid_dt)
        max_fxx = max(fxx for _, fxx in runs)
        assert max_fxx == 384

    def test_gfs_count(self):
        """384h / 6h intervals + 1 = 65 runs."""
        valid_dt = datetime(2026, 4, 6, 12)
        runs = find_runs_for_valid_time("gfs", valid_dt)
        assert len(runs) == 65


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_midnight_valid_time(self):
        """00Z valid time should work correctly across date boundaries."""
        valid_dt = datetime(2026, 4, 6, 0)
        runs = find_runs_for_valid_time("hrrr", valid_dt)
        assert len(runs) > 0
        # The analysis run at 00Z should be present
        assert (valid_dt, 0) in runs

    def test_valid_time_not_on_cycle_hour(self):
        """If valid_dt.hour is not a cycle hour, fxx=0 still works
        (HRRR is hourly, so every hour is a valid cycle)."""
        valid_dt = datetime(2026, 4, 6, 7)
        runs = find_runs_for_valid_time("hrrr", valid_dt)
        assert (valid_dt, 0) in runs

    def test_6h_model_valid_time_between_cycles(self):
        """For a 6-hourly model, if valid_dt falls between cycles (e.g. 03Z),
        there is no fxx=0 run, but earlier cycles still cover it."""
        valid_dt = datetime(2026, 4, 6, 3)
        runs = find_runs_for_valid_time("nam12k", valid_dt)
        # 03Z is not a NAM cycle hour, so fxx=0 should NOT be present
        assert (valid_dt, 0) not in runs
        # But there should be runs (e.g. 00Z init with fxx=3)
        assert len(runs) > 0
        # Verify that 00Z with fxx=3 is present
        assert (datetime(2026, 4, 6, 0), 3) in runs

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="No forecast metadata"):
            find_runs_for_valid_time("not_a_model", datetime(2026, 4, 6, 12))

    def test_rtma_not_in_forecast_meta(self):
        """RTMA is an analysis product, not a forecast model."""
        with pytest.raises(ValueError):
            find_runs_for_valid_time("rtma", datetime(2026, 4, 6, 12))


# ---------------------------------------------------------------------------
# RAP (also has extended cycles)
# ---------------------------------------------------------------------------

class TestRAP:
    """RAP: hourly cycles, 21h base / 51h for 03/09/15/21Z."""

    def test_rap_extended_cycles(self):
        valid_dt = datetime(2026, 4, 6, 12)
        runs = find_runs_for_valid_time("rap", valid_dt)
        for cycle_dt, fxx in runs:
            if cycle_dt.hour not in [3, 9, 15, 21]:
                assert fxx <= 21

    def test_rap_has_long_range_runs(self):
        valid_dt = datetime(2026, 4, 6, 12)
        runs = find_runs_for_valid_time("rap", valid_dt)
        max_fxx = max(fxx for _, fxx in runs)
        assert max_fxx > 21  # Should reach up to 51 from an extended cycle


# ---------------------------------------------------------------------------
# All models smoke test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_key", list(MODEL_FORECAST_META))
def test_find_runs_returns_nonempty_for_all_models(model_key):
    """Every model should find at least one run for a valid time."""
    valid_dt = datetime(2026, 4, 6, 12)
    runs = find_runs_for_valid_time(model_key, valid_dt)
    assert len(runs) > 0
    # All must satisfy the valid-time invariant
    for cycle_dt, fxx in runs:
        assert cycle_dt + timedelta(hours=fxx) == valid_dt
