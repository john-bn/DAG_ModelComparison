import numpy as np
import pytest
import xarray as xr

xe = pytest.importorskip("xesmf", reason="xesmf (+ esmpy) required for regrid tests")

from comparator.regrid import regrid_chunked


# ── helpers ──────────────────────────────────────────────────────────

def _make_rectilinear_grids(src_ny=20, src_nx=30, tgt_ny=15, tgt_nx=25):
    """Return (src_field, src_grid, tgt_grid) on simple rectilinear grids."""
    src_lon = np.linspace(-110, -90, src_nx)
    src_lat = np.linspace(25, 45, src_ny)
    tgt_lon = np.linspace(-108, -92, tgt_nx)
    tgt_lat = np.linspace(27, 43, tgt_ny)

    LON, LAT = np.meshgrid(src_lon, src_lat)
    # smooth field: linear ramp so bilinear interpolation is exact
    vals = 2.0 * LAT + 0.5 * LON
    src_field = xr.DataArray(vals, dims=("y", "x"))

    src_grid = {"lon": src_lon, "lat": src_lat}
    tgt_grid = {"lon": tgt_lon, "lat": tgt_lat}
    return src_field, src_grid, tgt_grid


def _make_curvilinear_grids(src_ny=20, src_nx=30, tgt_ny=15, tgt_nx=25):
    """Return (src_field, src_grid, tgt_grid) with 2-D lon/lat arrays."""
    src_lon_1d = np.linspace(-110, -90, src_nx)
    src_lat_1d = np.linspace(25, 45, src_ny)
    src_lon, src_lat = np.meshgrid(src_lon_1d, src_lat_1d)

    tgt_lon_1d = np.linspace(-108, -92, tgt_nx)
    tgt_lat_1d = np.linspace(27, 43, tgt_ny)
    tgt_lon, tgt_lat = np.meshgrid(tgt_lon_1d, tgt_lat_1d)

    vals = 2.0 * src_lat + 0.5 * src_lon
    src_field = xr.DataArray(vals, dims=("y", "x"))

    src_grid = {"lon": src_lon, "lat": src_lat}
    tgt_grid = {"lon": tgt_lon, "lat": tgt_lat}
    return src_field, src_grid, tgt_grid


def _regrid_reference(src_field, src_grid, tgt_grid, method="bilinear"):
    """Single-shot xESMF regrid (the baseline we compare against)."""
    regridder = xe.Regridder(src_grid, tgt_grid, method=method, periodic=False)
    return regridder(src_field)


# ── tests ────────────────────────────────────────────────────────────

def test_chunked_matches_unchunked_rectilinear():
    """Chunked regridding must produce the same values as a single-shot regrid."""
    src_field, src_grid, tgt_grid = _make_rectilinear_grids()
    ref = _regrid_reference(src_field, src_grid, tgt_grid)
    # chunk_rows=5 forces several chunks on a 15-row target
    chunked = regrid_chunked(src_field, src_grid, tgt_grid, chunk_rows=5)

    np.testing.assert_allclose(
        np.asarray(chunked), np.asarray(ref), atol=1e-6,
    )


def test_chunked_matches_unchunked_curvilinear():
    """Same comparison but with 2-D lon/lat grids."""
    src_field, src_grid, tgt_grid = _make_curvilinear_grids()
    ref = _regrid_reference(src_field, src_grid, tgt_grid)
    chunked = regrid_chunked(src_field, src_grid, tgt_grid, chunk_rows=5)

    np.testing.assert_allclose(
        np.asarray(chunked), np.asarray(ref), atol=1e-6,
    )


def test_output_shape_matches_target_grid():
    """The output must have the same shape as the target grid."""
    src_field, src_grid, tgt_grid = _make_rectilinear_grids(
        tgt_ny=18, tgt_nx=22,
    )
    result = regrid_chunked(src_field, src_grid, tgt_grid, chunk_rows=7)
    assert result.shape == (18, 22)


def test_single_chunk_path():
    """When chunk_rows >= n_target_rows the fast single-shot path is used."""
    src_field, src_grid, tgt_grid = _make_rectilinear_grids(tgt_ny=10)
    ref = _regrid_reference(src_field, src_grid, tgt_grid)
    # chunk_rows larger than target rows → single shot inside regrid_chunked
    chunked = regrid_chunked(src_field, src_grid, tgt_grid, chunk_rows=999)

    np.testing.assert_allclose(
        np.asarray(chunked), np.asarray(ref), atol=1e-6,
    )


def test_no_nan_in_interior():
    """A smooth source field should produce no NaN inside the target domain."""
    src_field, src_grid, tgt_grid = _make_rectilinear_grids()
    result = regrid_chunked(src_field, src_grid, tgt_grid, chunk_rows=4)
    assert np.all(np.isfinite(np.asarray(result)))


def test_chunk_boundary_continuity():
    """Values at chunk boundaries should be continuous, not show seams."""
    src_field, src_grid, tgt_grid = _make_rectilinear_grids(tgt_ny=20)
    result = regrid_chunked(src_field, src_grid, tgt_grid, chunk_rows=5)
    vals = np.asarray(result)

    # Check row-to-row differences — on a smooth linear ramp they should
    # be roughly constant (no jumps at chunk boundaries 5, 10, 15).
    row_diffs = np.diff(vals, axis=0)
    # Max delta between consecutive row-diffs should be small
    diff_of_diffs = np.abs(np.diff(row_diffs, axis=0))
    assert diff_of_diffs.max() < 0.1


def test_preserves_nan_from_source():
    """NaN values in the source should propagate to the output."""
    src_field, src_grid, tgt_grid = _make_rectilinear_grids(
        src_ny=10, src_nx=10, tgt_ny=8, tgt_nx=8,
    )
    # Inject a block of NaN into the source
    vals = src_field.values.copy()
    vals[3:6, 3:6] = np.nan
    src_field = xr.DataArray(vals, dims=("y", "x"))

    result = regrid_chunked(src_field, src_grid, tgt_grid, chunk_rows=3)
    # There should be at least some NaN in the output (near the NaN source region)
    assert np.any(np.isnan(np.asarray(result)))


def test_different_methods():
    """regrid_chunked should accept the nearest_s2d method too."""
    src_field, src_grid, tgt_grid = _make_rectilinear_grids()
    ref = _regrid_reference(src_field, src_grid, tgt_grid, method="nearest_s2d")
    chunked = regrid_chunked(
        src_field, src_grid, tgt_grid, method="nearest_s2d", chunk_rows=5,
    )

    np.testing.assert_allclose(
        np.asarray(chunked), np.asarray(ref), atol=1e-6,
    )
