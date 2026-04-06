"""Comprehensive tests for comparator.fielddiff.compute_fielddiff.

Covers: Kelvin-to-Fahrenheit conversion factor, valid-range bounds (150–330 K),
NaN / inf masking, exact-alignment requirement, shape/coord preservation,
symmetric-difference sign, and edge-case inputs.
"""

import numpy as np
import pytest
import xarray as xr

from comparator.fielddiff import compute_fielddiff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _da(vals, name="t2m", coords=None):
    """Build a 2-D DataArray from a nested list or ndarray."""
    vals = np.asarray(vals, dtype=float)
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    dims = ("y", "x")
    if coords is None:
        coords = {"y": np.arange(vals.shape[0]), "x": np.arange(vals.shape[1])}
    return xr.DataArray(vals, dims=dims, coords=coords, name=name)


# ---------------------------------------------------------------------------
# K → °F conversion factor
# ---------------------------------------------------------------------------

class TestConversionFactor:
    """ΔK * 9/5 = Δ°F  (1 K difference = 1.8 °F difference)."""

    def test_1K_diff_equals_1_8F(self):
        h = _da([[301.0]])
        r = _da([[300.0]])
        assert float(compute_fielddiff(h, r)) == pytest.approx(1.8)

    def test_negative_diff(self):
        h = _da([[299.0]])
        r = _da([[300.0]])
        assert float(compute_fielddiff(h, r)) == pytest.approx(-1.8)

    def test_zero_diff(self):
        h = _da([[300.0]])
        r = _da([[300.0]])
        assert float(compute_fielddiff(h, r)) == pytest.approx(0.0)

    def test_fractional_kelvin(self):
        h = _da([[300.5]])
        r = _da([[300.0]])
        assert float(compute_fielddiff(h, r)) == pytest.approx(0.5 * 9 / 5)

    def test_large_diff(self):
        """10 K difference → 18 °F."""
        h = _da([[310.0]])
        r = _da([[300.0]])
        assert float(compute_fielddiff(h, r)) == pytest.approx(18.0)


# ---------------------------------------------------------------------------
# Valid-range bounds (150 < T < 330 K)
# ---------------------------------------------------------------------------

class TestValidRangeBounds:
    """Temperatures outside (150, 330) K should be masked to NaN."""

    @pytest.mark.parametrize("bad_val", [149.9, 150.0, 0.0, -50.0, -273.15])
    def test_low_bound_masks_nwp(self, bad_val):
        h = _da([[bad_val]])
        r = _da([[290.0]])
        assert np.isnan(float(compute_fielddiff(h, r)))

    @pytest.mark.parametrize("bad_val", [330.0, 330.1, 500.0, 1000.0])
    def test_high_bound_masks_nwp(self, bad_val):
        h = _da([[bad_val]])
        r = _da([[290.0]])
        assert np.isnan(float(compute_fielddiff(h, r)))

    @pytest.mark.parametrize("bad_val", [149.9, 150.0, 330.0, 330.1])
    def test_bounds_mask_rtma(self, bad_val):
        h = _da([[290.0]])
        r = _da([[bad_val]])
        assert np.isnan(float(compute_fielddiff(h, r)))

    def test_just_inside_lower_bound(self):
        """150.1 K is valid."""
        h = _da([[200.0]])
        r = _da([[200.0]])
        # Both 200 K are in (150, 330) → valid, diff = 0
        assert float(compute_fielddiff(h, r)) == pytest.approx(0.0)

    def test_just_inside_upper_bound(self):
        """329.9 K is valid."""
        h = _da([[329.9]])
        r = _da([[329.9]])
        assert float(compute_fielddiff(h, r)) == pytest.approx(0.0)

    def test_boundary_exactly_150_is_excluded(self):
        """The bound uses strict > 150, so 150.0 is masked."""
        h = _da([[150.0]])
        r = _da([[200.0]])
        assert np.isnan(float(compute_fielddiff(h, r)))

    def test_boundary_exactly_330_is_excluded(self):
        """The bound uses strict < 330, so 330.0 is masked."""
        h = _da([[330.0]])
        r = _da([[200.0]])
        assert np.isnan(float(compute_fielddiff(h, r)))

    def test_mixed_valid_and_invalid(self):
        """Only the valid cells should produce finite output."""
        h = _da([[300.0, 100.0, 310.0, 400.0]])
        r = _da([[299.0, 290.0, 309.0, 290.0]])
        out = compute_fielddiff(h, r)
        assert np.isfinite(out.values[0, 0])
        assert np.isnan(out.values[0, 1])  # h < 150
        assert np.isfinite(out.values[0, 2])
        assert np.isnan(out.values[0, 3])  # h > 330


# ---------------------------------------------------------------------------
# NaN and Inf masking
# ---------------------------------------------------------------------------

class TestNonFiniteMasking:
    def test_nan_in_nwp(self):
        h = _da([[np.nan]])
        r = _da([[290.0]])
        assert np.isnan(float(compute_fielddiff(h, r)))

    def test_nan_in_rtma(self):
        h = _da([[290.0]])
        r = _da([[np.nan]])
        assert np.isnan(float(compute_fielddiff(h, r)))

    def test_inf_in_nwp(self):
        h = _da([[np.inf]])
        r = _da([[290.0]])
        assert np.isnan(float(compute_fielddiff(h, r)))

    def test_neg_inf_in_rtma(self):
        h = _da([[290.0]])
        r = _da([[-np.inf]])
        assert np.isnan(float(compute_fielddiff(h, r)))

    def test_both_nan(self):
        h = _da([[np.nan]])
        r = _da([[np.nan]])
        assert np.isnan(float(compute_fielddiff(h, r)))

    def test_all_nan_grid(self):
        h = _da([[np.nan, np.nan], [np.nan, np.nan]])
        r = _da([[290.0, 291.0], [292.0, 293.0]])
        out = compute_fielddiff(h, r)
        assert np.isnan(out.values).all()


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

class TestAlignment:
    def test_mismatched_coords_raises(self):
        h = xr.DataArray([300.0], dims=("x",), coords={"x": [0]})
        r = xr.DataArray([300.0], dims=("x",), coords={"x": [1]})
        with pytest.raises(ValueError):
            compute_fielddiff(h, r)

    def test_mismatched_shapes_raises(self):
        h = _da([[300.0, 301.0]])
        r = _da([[300.0, 301.0, 302.0]])
        with pytest.raises(ValueError):
            compute_fielddiff(h, r)

    def test_matching_coords_succeed(self):
        coords = {"y": [10], "x": [20, 30]}
        h = _da([[300.0, 301.0]], coords=coords)
        r = _da([[299.0, 300.0]], coords=coords)
        out = compute_fielddiff(h, r)
        assert out.shape == (1, 2)
        np.testing.assert_allclose(out.values, [[1.8, 1.8]])


# ---------------------------------------------------------------------------
# Shape and coordinate preservation
# ---------------------------------------------------------------------------

class TestShapePreservation:
    def test_2d_shape_preserved(self):
        h = _da(np.full((5, 10), 300.0))
        r = _da(np.full((5, 10), 299.0))
        out = compute_fielddiff(h, r)
        assert out.shape == (5, 10)

    def test_dims_preserved(self):
        h = _da([[300.0]], coords={"y": [42], "x": [99]})
        r = _da([[299.0]], coords={"y": [42], "x": [99]})
        out = compute_fielddiff(h, r)
        assert out.dims == ("y", "x")
        assert int(out.coords["y"]) == 42
        assert int(out.coords["x"]) == 99

    def test_large_grid(self):
        """Smoke test with a realistic grid size."""
        shape = (100, 200)
        h = _da(np.random.uniform(270, 310, shape))
        r = _da(np.random.uniform(270, 310, shape))
        out = compute_fielddiff(h, r)
        assert out.shape == shape
        # All inputs in valid range, so all outputs should be finite
        assert np.isfinite(out.values).all()


# ---------------------------------------------------------------------------
# Sign symmetry
# ---------------------------------------------------------------------------

class TestSignSymmetry:
    def test_swap_inputs_flips_sign(self):
        h = _da([[305.0]])
        r = _da([[300.0]])
        diff_hr = float(compute_fielddiff(h, r))
        diff_rh = float(compute_fielddiff(r, h))
        assert diff_hr == pytest.approx(-diff_rh)
