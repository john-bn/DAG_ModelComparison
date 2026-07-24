import numpy as np
import pytest

from comparator.regrid import KDTreeRegridder


# A small 2x2 source grid near the equator (so unit-sphere distances between
# the four corners are ~symmetric and easy to reason about).
SRC_LON = [0.0, 1.0]
SRC_LAT = [0.0, 1.0]


def _field(values):
    """A (2, 2) field laid out as meshgrid(lon, lat) -> shape (lat, lon)."""
    return np.asarray(values, dtype=float).reshape(2, 2)


def test_exact_hits_reproduce_source():
    """Target points coincident with source points return source values."""
    rg = KDTreeRegridder.from_grids(SRC_LON, SRC_LAT, SRC_LON, SRC_LAT, k=4)
    field = _field([[1.0, 2.0], [3.0, 4.0]])
    out = rg.apply(field)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, field)


def test_equidistant_center_averages_neighbours():
    """A target at the grid centre weights the four corners ~equally."""
    rg = KDTreeRegridder.from_grids(SRC_LON, SRC_LAT, [0.5], [0.5], k=4)
    field = _field([[1.0, 2.0], [3.0, 4.0]])
    out = rg.apply(field)
    assert out.shape == (1, 1)
    # ~equal weights (corners are near-equidistant on the sphere, not exactly).
    np.testing.assert_allclose(out[0, 0], np.mean([1.0, 2.0, 3.0, 4.0]), rtol=1e-3)


def test_nan_neighbour_is_dropped_and_renormalised():
    """A non-finite source neighbour is excluded; weights renormalise."""
    rg = KDTreeRegridder.from_grids(SRC_LON, SRC_LAT, [0.5], [0.5], k=4)
    field = _field([[np.nan, 2.0], [3.0, 4.0]])
    out = rg.apply(field)
    np.testing.assert_allclose(out[0, 0], np.mean([2.0, 3.0, 4.0]), rtol=1e-3)


def test_all_nan_neighbours_yield_nan():
    rg = KDTreeRegridder.from_grids(SRC_LON, SRC_LAT, [0.5], [0.5], k=4)
    out = rg.apply(_field([[np.nan] * 4]))
    assert np.isnan(out[0, 0])


def test_save_load_round_trip(tmp_path):
    rg = KDTreeRegridder.from_grids(SRC_LON, SRC_LAT, [0.25, 0.75], [0.25, 0.75], k=4)
    field = _field([[1.0, 2.0], [3.0, 4.0]])
    expected = rg.apply(field)

    path = tmp_path / "weights.npz"
    rg.save(path)
    loaded = KDTreeRegridder.load(path)

    np.testing.assert_array_equal(loaded.indices, rg.indices)
    np.testing.assert_allclose(loaded.weights, rg.weights)
    assert loaded.tgt_shape == rg.tgt_shape
    np.testing.assert_allclose(loaded.apply(field), expected)


def test_from_grids_cached_writes_then_reuses(tmp_path):
    path = tmp_path / "cache.npz"
    assert not path.exists()

    first = KDTreeRegridder.from_grids_cached(
        SRC_LON, SRC_LAT, [0.5], [0.5], cache_path=path
    )
    assert path.exists()  # built + saved

    # Second call must load the cache (same target shape) and match.
    second = KDTreeRegridder.from_grids_cached(
        SRC_LON, SRC_LAT, [0.5], [0.5], cache_path=path
    )
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_allclose(first.weights, second.weights)


def test_nearest_neighbour_k1():
    """k=1 collapses to nearest-neighbour and keeps the (n, 1) shape."""
    rg = KDTreeRegridder.from_grids(SRC_LON, SRC_LAT, [0.4], [0.1], k=1)
    assert rg.indices.shape[1] == 1
    field = _field([[10.0, 20.0], [30.0, 40.0]])
    out = rg.apply(field)
    # Nearest source point to (lon=0.4, lat=0.1) is the (0,0) corner -> 10.0.
    np.testing.assert_allclose(out[0, 0], 10.0)
