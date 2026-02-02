import numpy as np
import pytest
import xarray as xr

from DAG_ModelComparison.comparator.plotting import (
    _wrap180,
    _to_2d_lonlat,
    _nearest_values_on_geo_grid,
)


def test_wrap180_basic():
    vals = np.array([0, 180, 181, 359, 360, 540, -181], dtype=float)
    out = _wrap180(vals)
    # all should end up in [-180, 180)
    assert np.all(out >= -180)
    assert np.all(out < 180)
    assert out[0] == pytest.approx(0)
    assert out[2] == pytest.approx(-179)  # 181 -> -179
    assert out[3] == pytest.approx(-1)    # 359 -> -1


def test_to_2d_lonlat_from_1d():
    lon = xr.DataArray(np.array([-100, -99, -98], dtype=float), dims=("x",))
    lat = xr.DataArray(np.array([30, 31], dtype=float), dims=("y",))
    LON2, LAT2 = _to_2d_lonlat(lon, lat)
    assert LON2.shape == (2, 3)
    assert LAT2.shape == (2, 3)
    assert np.all(LON2[0, :] == np.array([-100, -99, -98]))
    assert np.all(LAT2[:, 0] == np.array([30, 31]))


def test_nearest_values_on_geo_grid_1d_coords():
    # Build a 2x3 grid using 1D lon/lat
    lon = xr.DataArray(np.array([-100, -99, -98], dtype=float), dims=("x",))
    lat = xr.DataArray(np.array([30, 31], dtype=float), dims=("y",))

    # Values laid out as:
    # lat=30: 10, 11, 12
    # lat=31: 20, 21, 22
    da = xr.DataArray(
        np.array([[10, 11, 12], [20, 21, 22]], dtype=float),
        dims=("y", "x"),
        coords={"y": lat.values, "x": lon.values},
    )

    pts_lon = np.array([-99.1, -97.6], dtype=float)  # nearest -99 and -98
    pts_lat = np.array([30.2, 30.9], dtype=float)    # nearest 30 and 31
    out = _nearest_values_on_geo_grid(lon, lat, da, pts_lon, pts_lat)

    assert out.shape == (2,)
    assert out[0] == pytest.approx(11)  # (-99, 30)
    assert out[1] == pytest.approx(22)  # (-98, 31)


def test_nearest_values_on_geo_grid_all_nan_returns_nan():
    lon = xr.DataArray(np.array([-100, -99], dtype=float), dims=("x",))
    lat = xr.DataArray(np.array([30, 31], dtype=float), dims=("y",))
    da = xr.DataArray(np.full((2, 2), np.nan), dims=("y", "x"))

    out = _nearest_values_on_geo_grid(
        lon, lat, da, np.array([-99.5]), np.array([30.5])
    )
    assert np.isnan(out[0])
