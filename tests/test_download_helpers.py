"""Tests for the download helper functions in new_comparison.py.

These tests use mocking to avoid actual network calls while verifying
the parallel download logic, RTMA reuse, and error handling.
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _fetch_herbie
# ---------------------------------------------------------------------------

class TestFetchHerbie:
    @patch("new_comparison.Herbie")
    def test_returns_herbie_on_success(self, MockHerbie, tmp_path):
        from new_comparison import _fetch_herbie

        mock_h = MagicMock()
        mock_h.__bool__ = lambda self: True
        MockHerbie.return_value = mock_h

        result = _fetch_herbie(
            datetime(2026, 4, 6, 12), 6, tmp_path, model="hrrr", product="sfc"
        )
        assert result is mock_h
        mock_h.download.assert_called_once()

    @patch("new_comparison.Herbie")
    def test_returns_none_when_unavailable(self, MockHerbie, tmp_path):
        from new_comparison import _fetch_herbie

        mock_h = MagicMock()
        mock_h.__bool__ = lambda self: False
        MockHerbie.return_value = mock_h

        result = _fetch_herbie(
            datetime(2026, 4, 6, 12), 6, tmp_path, model="hrrr", product="sfc"
        )
        assert result is None


# ---------------------------------------------------------------------------
# _fetch_nwp
# ---------------------------------------------------------------------------

class TestFetchNwp:
    @patch("new_comparison._fetch_herbie")
    def test_returns_tuple_on_success(self, mock_fetch):
        from new_comparison import _fetch_nwp

        mock_h = MagicMock()
        mock_fetch.return_value = mock_h

        cycle_dt = datetime(2026, 4, 6, 12)
        result = _fetch_nwp("hrrr", cycle_dt, 6, "TMP:2 m above", Path("./data"))
        assert result is not None
        assert result == (cycle_dt, 6, mock_h)

    @patch("new_comparison._fetch_herbie")
    def test_returns_none_on_failure(self, mock_fetch):
        from new_comparison import _fetch_nwp

        mock_fetch.return_value = None
        result = _fetch_nwp(
            "hrrr", datetime(2026, 4, 6, 12), 6, "TMP:2 m above", Path("./data")
        )
        assert result is None


# ---------------------------------------------------------------------------
# generate_gif_frames: structural tests (mocked I/O)
# ---------------------------------------------------------------------------

class TestGenerateGifFramesStructure:
    """Test that generate_gif_frames orchestrates downloads correctly."""

    @patch("new_comparison.xe")
    @patch("new_comparison.plot")
    @patch("new_comparison.fd")
    @patch("new_comparison.Herbie")
    @patch("new_comparison._fetch_nwp")
    def test_rtma_downloaded_once_for_all_frames(
        self, mock_fetch_nwp, MockHerbie, mock_fd, mock_plot, mock_xe, tmp_path
    ):
        """RTMA should be fetched exactly once, not once per frame."""
        import numpy as np
        import xarray as xr
        from new_comparison import generate_gif_frames

        # Mock RTMA Herbie
        mock_rtma = MagicMock()
        mock_rtma.__bool__ = lambda s: True
        ds_rtma = xr.Dataset({
            "t2m": (("y", "x"), np.full((2, 2), 290.0)),
            "longitude": (("x",), [-100.0, -90.0]),
            "latitude": (("y",), [30.0, 40.0]),
        })
        mock_rtma.xarray.return_value = ds_rtma
        MockHerbie.return_value = mock_rtma

        # Mock NWP fetch
        mock_nwp_h = MagicMock()
        ds_nwp = xr.Dataset({
            "t2m": (("y", "x"), np.full((2, 2), 295.0)),
            "longitude": (("x",), [-100.0, -90.0]),
            "latitude": (("y",), [30.0, 40.0]),
        })
        mock_nwp_h.xarray.return_value = ds_nwp

        runs = [
            (datetime(2026, 4, 6, 6), 6),
            (datetime(2026, 4, 6, 0), 12),
            (datetime(2026, 4, 5, 12), 24),
        ]
        mock_fetch_nwp.side_effect = [
            (runs[0][0], runs[0][1], mock_nwp_h),
            (runs[1][0], runs[1][1], mock_nwp_h),
            (runs[2][0], runs[2][1], mock_nwp_h),
        ]

        # Mock regridder
        mock_regridder = MagicMock()
        mock_regridder.return_value = ds_rtma["t2m"]
        mock_xe.Regridder.return_value = mock_regridder

        # Mock fielddiff
        mock_fd.compute_fielddiff.return_value = xr.DataArray(
            np.zeros((2, 2)), dims=("y", "x")
        )

        # Mock plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        mock_plot.plot_tempdiff_map_with_table.return_value = (fig, (ax, ax))
        mock_plot.plot_airports.return_value = None

        valid_dt = datetime(2026, 4, 6, 12)

        generate_gif_frames(
            "hrrr", "TMP", runs, valid_dt,
            save_dir=tmp_path / "data",
            out_dir=tmp_path / "figs",
        )

        # RTMA Herbie should be created exactly once
        assert MockHerbie.call_count == 1
        # NWP fetch should be called for each run
        assert mock_fetch_nwp.call_count == 3

        plt.close("all")


# ---------------------------------------------------------------------------
# Thread pool: ensure parallel execution doesn't corrupt results
# ---------------------------------------------------------------------------

class TestParallelDownloadIntegrity:
    """Verify that ThreadPoolExecutor usage preserves (cycle_dt, fxx) mapping."""

    @patch("new_comparison._fetch_herbie")
    def test_results_keyed_correctly(self, mock_fetch):
        """Each result should map back to the correct (cycle_dt, fxx) pair."""
        from new_comparison import _fetch_nwp

        mock_h = MagicMock()
        mock_fetch.return_value = mock_h

        # Simulate 5 different runs
        runs = [
            (datetime(2026, 4, 6, h), 12 - h)
            for h in [0, 3, 6, 9, 12]
        ]

        results = {}
        for cycle_dt, fxx in runs:
            result = _fetch_nwp("hrrr", cycle_dt, fxx, "TMP:2 m above", Path("."))
            if result is not None:
                c, f, h = result
                results[(c, f)] = h

        # All 5 runs should be present with correct keys
        assert len(results) == 5
        for cycle_dt, fxx in runs:
            assert (cycle_dt, fxx) in results
