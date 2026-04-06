"""Comprehensive tests for plotting constants, bounds, and helper functions.

Covers: CONUS geographic bounds, TwoSlopeNorm parameters, _wrap180,
_to_2d_lonlat, _nearest_values_on_geo_grid, airport filtering, and
plot_tempdiff_map_with_table invariants.
"""

import numpy as np
import pytest
import xarray as xr
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from comparator.plotting import (
    CONUS_LON_MIN,
    CONUS_LON_MAX,
    CONUS_LAT_MIN,
    CONUS_LAT_MAX,
    _wrap180,
    _to_2d_lonlat,
    _nearest_values_on_geo_grid,
    plot_tempdiff_map_with_table,
    plot_airports,
)
from comparator.util import major_airports_df
from comparator.normalize import VAR_REGISTRY


# ---------------------------------------------------------------------------
# CONUS bounds
# ---------------------------------------------------------------------------

class TestCONUSBounds:
    """Verify the fixed CONUS geographic constants are sane."""

    def test_lon_range(self):
        assert CONUS_LON_MIN < CONUS_LON_MAX
        # CONUS is roughly -125 to -66.5
        assert -130 < CONUS_LON_MIN < -120
        assert -70 < CONUS_LON_MAX < -60

    def test_lat_range(self):
        assert CONUS_LAT_MIN < CONUS_LAT_MAX
        # CONUS is roughly 20 to 50
        assert 15 < CONUS_LAT_MIN < 25
        assert 45 < CONUS_LAT_MAX < 55

    def test_all_airports_within_conus_bounds(self):
        """Every airport in the hardcoded list should fall within CONUS bounds."""
        airports = major_airports_df()
        assert (airports["lon"] >= CONUS_LON_MIN).all()
        assert (airports["lon"] <= CONUS_LON_MAX).all()
        assert (airports["lat"] >= CONUS_LAT_MIN).all()
        assert (airports["lat"] <= CONUS_LAT_MAX).all()


# ---------------------------------------------------------------------------
# _wrap180
# ---------------------------------------------------------------------------

class TestWrap180:
    def test_positive_longitudes(self):
        result = _wrap180(np.array([0.0, 90.0, 180.0, 270.0, 360.0]))
        np.testing.assert_allclose(result, [0.0, 90.0, -180.0, -90.0, 0.0])

    def test_already_negative(self):
        vals = np.array([-120.0, -90.0, 0.0, 50.0])
        result = _wrap180(vals)
        np.testing.assert_allclose(result, vals)

    def test_single_value(self):
        assert _wrap180(np.array([270.0]))[0] == pytest.approx(-90.0)

    def test_empty_array(self):
        result = _wrap180(np.array([]))
        assert len(result) == 0

    def test_edge_180(self):
        """180 should wrap to -180."""
        assert _wrap180(np.array([180.0]))[0] == pytest.approx(-180.0)

    def test_edge_negative_180(self):
        assert _wrap180(np.array([-180.0]))[0] == pytest.approx(-180.0)


# ---------------------------------------------------------------------------
# _to_2d_lonlat
# ---------------------------------------------------------------------------

class TestTo2DLonLat:
    def test_1d_inputs_meshgridded(self):
        lon = xr.DataArray([10.0, 20.0, 30.0], dims=("x",))
        lat = xr.DataArray([40.0, 50.0], dims=("y",))
        LON2, LAT2 = _to_2d_lonlat(lon, lat)
        assert LON2.shape == (2, 3)
        assert LAT2.shape == (2, 3)
        # First row should have lat=40
        np.testing.assert_allclose(LAT2[0, :], 40.0)
        # First column should have lon=10
        np.testing.assert_allclose(LON2[:, 0], 10.0)

    def test_2d_inputs_passthrough(self):
        lon_2d = np.array([[10.0, 20.0], [10.0, 20.0]])
        lat_2d = np.array([[40.0, 40.0], [50.0, 50.0]])
        lon = xr.DataArray(lon_2d, dims=("y", "x"))
        lat = xr.DataArray(lat_2d, dims=("y", "x"))
        LON2, LAT2 = _to_2d_lonlat(lon, lat)
        np.testing.assert_allclose(LON2, lon_2d)
        np.testing.assert_allclose(LAT2, lat_2d)


# ---------------------------------------------------------------------------
# _nearest_values_on_geo_grid
# ---------------------------------------------------------------------------

class TestNearestValues:
    def _simple_grid(self):
        """2x3 grid with known values."""
        lon = xr.DataArray([-100.0, -90.0, -80.0], dims=("x",))
        lat = xr.DataArray([30.0, 40.0], dims=("y",))
        # Values: row0=[1,2,3], row1=[4,5,6]
        data = xr.DataArray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dims=("y", "x"),
        )
        return lon, lat, data

    def test_exact_gridpoint_match(self):
        lon, lat, data = self._simple_grid()
        pts_lon = np.array([-100.0])
        pts_lat = np.array([30.0])
        result = _nearest_values_on_geo_grid(lon, lat, data, pts_lon, pts_lat)
        assert result[0] == pytest.approx(1.0)

    def test_nearest_neighbor_selection(self):
        lon, lat, data = self._simple_grid()
        # Point close to (-90, 40) → value 5.0
        pts_lon = np.array([-89.5])
        pts_lat = np.array([39.5])
        result = _nearest_values_on_geo_grid(lon, lat, data, pts_lon, pts_lat)
        assert result[0] == pytest.approx(5.0)

    def test_multiple_points(self):
        lon, lat, data = self._simple_grid()
        pts_lon = np.array([-100.0, -80.0])
        pts_lat = np.array([30.0, 40.0])
        result = _nearest_values_on_geo_grid(lon, lat, data, pts_lon, pts_lat)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(6.0)

    def test_nan_values_handled(self):
        """NaN cells should be excluded from nearest-neighbor lookup."""
        lon = xr.DataArray([-100.0, -90.0], dims=("x",))
        lat = xr.DataArray([30.0], dims=("y",))
        data = xr.DataArray([[np.nan, 5.0]], dims=("y", "x"))
        pts_lon = np.array([-100.0])  # Closest is NaN, should skip to -90
        pts_lat = np.array([30.0])
        result = _nearest_values_on_geo_grid(lon, lat, data, pts_lon, pts_lat)
        assert result[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# plot_tempdiff_map_with_table: invariants
# ---------------------------------------------------------------------------

class TestPlotTempdiffInvariants:
    """Test that the plotting function preserves key analysis invariants."""

    def _make_test_data(self):
        """Create minimal plottable data."""
        lon = xr.DataArray(
            np.linspace(-120, -70, 10), dims=("x",)
        )
        lat = xr.DataArray(
            np.linspace(25, 48, 8), dims=("y",)
        )
        diff = xr.DataArray(
            np.random.uniform(-10, 10, (8, 10)),
            dims=("y", "x"),
        )
        return lon, lat, diff

    def test_returns_fig_and_axes(self):
        import matplotlib.pyplot as plt
        lon, lat, diff = self._make_test_data()
        from datetime import datetime
        vdt = datetime(2026, 4, 6, 12)
        cdt = datetime(2026, 4, 6, 0)
        meta = VAR_REGISTRY["TMP"]

        fig, (ax_map, ax_tbl) = plot_tempdiff_map_with_table(
            lon, lat, diff, vdt, cdt, 12, "hrrr",
            major_airports_df(), plot_meta=meta,
        )
        assert fig is not None
        assert ax_map is not None
        assert ax_tbl is not None
        plt.close(fig)

    def test_norm_is_two_slope(self):
        """The color norm must be TwoSlopeNorm centered at 0."""
        import matplotlib.pyplot as plt
        lon, lat, diff = self._make_test_data()
        from datetime import datetime
        meta = VAR_REGISTRY["TMP"]

        fig, (ax_map, _) = plot_tempdiff_map_with_table(
            lon, lat, diff,
            datetime(2026, 4, 6, 12),
            datetime(2026, 4, 6, 0),
            12, "hrrr", major_airports_df(), plot_meta=meta,
        )
        # Find the pcolormesh artist and check its norm
        mesh = [
            c for c in ax_map.get_children()
            if hasattr(c, "norm") and isinstance(c.norm, TwoSlopeNorm)
        ]
        assert len(mesh) > 0
        assert mesh[0].norm.vcenter == 0.0
        plt.close(fig)

    def test_table_sorted_alphabetically(self):
        """The airport table should be sorted by ICAO code."""
        import matplotlib.pyplot as plt
        lon, lat, diff = self._make_test_data()
        from datetime import datetime
        meta = VAR_REGISTRY["TMP"]

        fig, (_, ax_tbl) = plot_tempdiff_map_with_table(
            lon, lat, diff,
            datetime(2026, 4, 6, 12),
            datetime(2026, 4, 6, 0),
            12, "hrrr", major_airports_df(), plot_meta=meta,
        )
        # Extract ICAO column from the table
        table = [
            c for c in ax_tbl.get_children()
            if hasattr(c, "_cells")
        ]
        assert len(table) > 0
        cells = table[0]._cells
        # Row 0 is header; rows 1..N are data. Column 0 is ICAO.
        icaos = []
        row = 1
        while (row, 0) in cells:
            icaos.append(cells[(row, 0)].get_text().get_text())
            row += 1
        assert icaos == sorted(icaos)
        plt.close(fig)

    def test_figure_can_be_saved(self, tmp_path):
        """Smoke test: the figure saves to disk without error."""
        import matplotlib.pyplot as plt
        lon, lat, diff = self._make_test_data()
        from datetime import datetime
        meta = VAR_REGISTRY["TMP"]

        fig, _ = plot_tempdiff_map_with_table(
            lon, lat, diff,
            datetime(2026, 4, 6, 12),
            datetime(2026, 4, 6, 0),
            12, "hrrr", major_airports_df(), plot_meta=meta,
        )
        out = tmp_path / "test_plot.png"
        fig.savefig(out, dpi=50)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close(fig)


# ---------------------------------------------------------------------------
# VAR_REGISTRY plot_meta defaults
# ---------------------------------------------------------------------------

class TestVarRegistryPlotMeta:
    """Ensure VAR_REGISTRY entries produce valid TwoSlopeNorm defaults."""

    @pytest.mark.parametrize("var_key", ["TMP", "DPT"])
    def test_default_norm_is_valid(self, var_key):
        meta = VAR_REGISTRY[var_key]
        vmin = meta.get("vmin", -15)
        vmax = meta.get("vmax", 15)
        vcenter = meta.get("vcenter", 0.0)
        # TwoSlopeNorm requires vmin < vcenter < vmax
        assert vmin < vcenter < vmax
        # Should construct without error
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        assert norm.vcenter == vcenter

    @pytest.mark.parametrize("var_key", ["TMP", "DPT"])
    def test_cmap_is_string(self, var_key):
        assert isinstance(VAR_REGISTRY[var_key]["cmap"], str)

    @pytest.mark.parametrize("var_key", ["TMP", "DPT"])
    def test_title_is_nonempty_string(self, var_key):
        title = VAR_REGISTRY[var_key]["title"]
        assert isinstance(title, str)
        assert len(title) > 0
