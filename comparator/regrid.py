"""Lightweight KDTree-based regridder (analysis -> model grid).

This replaces the former xESMF/ESMF bilinear regridder. ESMF/ESMPy held the
full source + target meshes in memory while generating weights, and the source
is the RTMA/URMA 2.5 km CONUS grid (~2.9M points) -- a large fixed memory cost
in every process, and single-frame mode rebuilt the weights on every run.

Instead we use :class:`scipy.spatial.cKDTree` (already a dependency) to build a
k-nearest-neighbour **inverse-distance-weighted** map from source to target
grid points. IDW with k=4 closely approximates bilinear at these resolutions
(2.5 km analysis -> ~3 km model), at a fraction of the memory, and with no
compiled Fortran dependency.

The interpolation is defined entirely by two arrays -- ``indices`` and
``weights``, both shaped ``(n_target, k)`` -- so it precomputes once and caches
cheaply to a ``.npz`` for reuse across frames and runs.
"""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _to_2d_lonlat(lon, lat):
    """Return 2-D lon/lat arrays whether inputs are 1-D or 2-D.

    Mirrors ``plotting._to_2d_lonlat``: for 1-D inputs, lon varies across
    columns (x) and lat across rows (y).
    """
    lon_vals = np.asarray(lon, dtype=float)
    lat_vals = np.asarray(lat, dtype=float)
    if lon_vals.ndim == 1 and lat_vals.ndim == 1:
        lon2, lat2 = np.meshgrid(lon_vals, lat_vals)
    else:
        lon2, lat2 = lon_vals, lat_vals
    return lon2, lat2


def _lonlat_to_xyz(lon2, lat2):
    """Map lon/lat (degrees) to 3-D unit-sphere coordinates.

    Using Cartesian xyz on the unit sphere lets a Euclidean KDTree distance
    stand in for great-circle distance, and sidesteps antimeridian/pole
    discontinuities that plague raw lon/lat nearest-neighbour search.
    """
    lon_r = np.deg2rad(np.asarray(lon2, dtype=float).ravel())
    lat_r = np.deg2rad(np.asarray(lat2, dtype=float).ravel())
    cos_lat = np.cos(lat_r)
    x = cos_lat * np.cos(lon_r)
    y = cos_lat * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.column_stack([x, y, z])


class KDTreeRegridder:
    """k-nearest-neighbour inverse-distance-weighted regridder.

    Construct from source/target lon/lat grids (1-D or 2-D), then :meth:`apply`
    a source field (2-D, matching the source grid shape) to obtain the field on
    the target grid. The precomputed ``indices``/``weights`` can be persisted
    with :meth:`save` and reloaded with :meth:`load`.
    """

    def __init__(self, indices, weights, tgt_shape):
        self.indices = np.asarray(indices)
        # Preserve the stored dtype (float32) so a loaded map stays compact in
        # RAM; apply() promotes as needed during the arithmetic.
        self.weights = np.asarray(weights)
        self.tgt_shape = tuple(int(n) for n in tgt_shape)

    # --- construction -----------------------------------------------------
    @classmethod
    def from_grids(cls, src_lon, src_lat, tgt_lon, tgt_lat, k=4):
        """Build the k-NN IDW map from source to target grids."""
        src_lon2, src_lat2 = _to_2d_lonlat(src_lon, src_lat)
        tgt_lon2, tgt_lat2 = _to_2d_lonlat(tgt_lon, tgt_lat)
        tgt_shape = tgt_lon2.shape

        src_xyz = _lonlat_to_xyz(src_lon2, src_lat2)
        tgt_xyz = _lonlat_to_xyz(tgt_lon2, tgt_lat2)

        n_src = src_xyz.shape[0]
        k = int(min(k, n_src))
        tree = cKDTree(src_xyz)
        dist, idx = tree.query(tgt_xyz, k=k)

        # cKDTree drops the trailing axis when k == 1; restore it so the rest
        # of the code can assume shape (n_target, k).
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]

        # Inverse-distance weights. An exact hit (d == 0) should take that
        # neighbour's value outright, so give it all the weight for that point.
        with np.errstate(divide="ignore"):
            weights = 1.0 / dist
        exact = ~np.isfinite(weights)  # d == 0 -> inf
        if exact.any():
            rows = exact.any(axis=1)
            weights[rows] = 0.0
            weights[exact] = 1.0

        # Downcast to halve the in-RAM and on-disk footprint: source point count
        # is far below 2**31, and float32 weights are plenty precise for IDW.
        return cls(idx.astype(np.int32), weights.astype(np.float32), tgt_shape)

    @classmethod
    def from_grids_cached(cls, src_lon, src_lat, tgt_lon, tgt_lat, *, cache_path,
                          k=4):
        """Load a cached map when its target shape matches, else build + save.

        The source/target grids for a given (verif -> model) pair are fixed, so
        a shape match on the cache is a sufficient key in practice.
        """
        cache_path = Path(cache_path)
        tgt_shape = _to_2d_lonlat(tgt_lon, tgt_lat)[0].shape
        if cache_path.exists():
            try:
                obj = cls.load(cache_path)
                if obj.tgt_shape == tuple(tgt_shape):
                    return obj
            except Exception:
                pass  # corrupt/stale cache -> rebuild below
        obj = cls.from_grids(src_lon, src_lat, tgt_lon, tgt_lat, k=k)
        try:
            obj.save(cache_path)
        except Exception:
            pass  # caching is best-effort; a read-only dir shouldn't fail a run
        return obj

    # --- application ------------------------------------------------------
    def apply(self, field):
        """Interpolate a source *field* (2-D) onto the target grid.

        Non-finite source neighbours are dropped and the remaining weights are
        renormalised per target point; a target point whose every neighbour is
        non-finite becomes NaN.
        """
        src = np.asarray(field, dtype=float).ravel()
        gathered = src[self.indices]              # (n_target, k)
        finite = np.isfinite(gathered)

        w = self.weights * finite                 # zero out non-finite neighbours
        wsum = w.sum(axis=1)
        gathered_filled = np.where(finite, gathered, 0.0)
        numer = (w * gathered_filled).sum(axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            out = numer / wsum
        out[wsum == 0] = np.nan
        return out.reshape(self.tgt_shape)

    # --- persistence ------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            indices=self.indices,
            weights=self.weights,
            tgt_shape=np.asarray(self.tgt_shape, dtype=np.int64),
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data["indices"], data["weights"], tuple(data["tgt_shape"]))
