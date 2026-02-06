import numpy as np
import pytest
import xarray as xr

from comparator.fielddiff import compute_fielddiff


def _da(vals, name="t2m"):
    vals = np.asarray(vals, dtype=float)
    # minimal coords/dims that still exercise xr.align
    return xr.DataArray(vals, dims=("y", "x"), name=name)


def test_compute_fielddiff_basic_math_and_units():
    # 300K - 299K = 1K -> 1.8°F
    h = _da([[300.0, 299.0]])
    r = _da([[299.0, 299.0]])
    out = compute_fielddiff(h, r)

    assert np.isfinite(out.values).all()
    assert out.shape == (1, 2)
    assert out.values[0, 0] == pytest.approx(1.8)
    assert out.values[0, 1] == pytest.approx(0.0)


def test_compute_fielddiff_masks_nonfinite_and_out_of_range():
    h = _da([[np.nan, 100.0, 400.0, 300.0]])
    r = _da([[290.0, 290.0, 290.0, np.nan]])

    out = compute_fielddiff(h, r)
    # positions:
    # 0: h nan -> masked
    # 1: h=100 (<150) -> masked
    # 2: h=400 (>330) -> masked
    # 3: r nan -> masked
    assert np.isnan(out.values).all()


def test_compute_fielddiff_requires_exact_alignment():
    h = xr.DataArray([300.0, 301.0], dims=("x",), coords={"x": [0, 1]})
    r = xr.DataArray([300.0, 301.0], dims=("x",), coords={"x": [0, 2]})

    with pytest.raises(ValueError):
        compute_fielddiff(h, r)


def test_compute_fielddiff_preserves_coords_and_dims():
    h = xr.DataArray(
        [[300.0, 301.0], [302.0, 303.0]],
        dims=("y", "x"),
        coords={"y": [10, 20], "x": [1, 2]},
        name="h",
    )
    r = xr.DataArray(
        [[299.0, 300.0], [301.0, 302.0]],
        dims=("y", "x"),
        coords={"y": [10, 20], "x": [1, 2]},
        name="r",
    )

    out = compute_fielddiff(h, r)
    assert out.dims == ("y", "x")
    assert np.all(out["y"].values == np.array([10, 20]))
    assert np.all(out["x"].values == np.array([1, 2]))