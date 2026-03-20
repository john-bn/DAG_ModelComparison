"""Memory-efficient chunked regridding via xESMF.

Instead of building one weight matrix for the entire source grid,
this module splits the source grid into horizontal strips (chunks of
rows), builds a small regridder for each strip, and combines the
results.  Peak memory stays proportional to *chunk_size* rather than
the full grid.
"""

import gc
import numpy as np
import xarray as xr
import xesmf as xe


def _grid_dict(lon, lat):
    """Build the {lon, lat} dict xESMF expects."""
    return {"lon": lon, "lat": lat}


def regrid_chunked(
    src_field: xr.DataArray,
    src_grid: dict,
    tgt_grid: dict,
    method: str = "bilinear",
    chunk_rows: int = 300,
) -> xr.DataArray:
    """Regrid *src_field* onto *tgt_grid* in row-wise chunks.

    Parameters
    ----------
    src_field : xr.DataArray
        The 2-D source data (e.g. RTMA temperature).
    src_grid : dict
        ``{"lon": ..., "lat": ...}`` arrays for the source grid.
    tgt_grid : dict
        ``{"lon": ..., "lat": ...}`` arrays for the target grid.
    method : str
        xESMF interpolation method (default ``"bilinear"``).
    chunk_rows : int
        Number of source-grid rows per chunk.  Lower values use less
        memory but require more iterations.  300 is a reasonable
        default for CONUS-scale grids on 4-8 GB machines.

    Returns
    -------
    xr.DataArray
        The regridded field on *tgt_grid*.
    """
    src_lon = np.asarray(src_grid["lon"])
    src_lat = np.asarray(src_grid["lat"])
    field_vals = np.asarray(src_field)

    # Determine the number of rows in the source grid.
    if src_lat.ndim == 1:
        n_rows = src_lat.shape[0]
    else:
        n_rows = src_lat.shape[0]

    # If the grid is small enough, just regrid in one shot.
    if n_rows <= chunk_rows:
        regridder = xe.Regridder(src_grid, tgt_grid, method=method, periodic=False)
        result = regridder(src_field)
        del regridder
        gc.collect()
        return result

    # --- chunked path ---
    # We accumulate the regridded result by filling NaN gaps from each
    # successive chunk.  Each chunk's regridder only produces valid
    # (non-NaN) values for target points that spatially overlap the
    # source-chunk strip; the rest are NaN.
    result = None

    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)

        # Slice source grid and field for this strip of rows.
        if src_lat.ndim == 1:
            chunk_lat = src_lat[start:end]
            chunk_lon = src_lon          # lon is shared across all rows
            chunk_field = field_vals[start:end, :]
        else:
            chunk_lat = src_lat[start:end, :]
            chunk_lon = src_lon[start:end, :]
            chunk_field = field_vals[start:end, :]

        chunk_src = _grid_dict(chunk_lon, chunk_lat)

        # Wrap the chunk in a DataArray so xESMF handles dims properly.
        chunk_da = xr.DataArray(
            chunk_field,
            dims=src_field.dims if src_field.ndim == 2 else ["y", "x"],
        )

        regridder = xe.Regridder(
            chunk_src, tgt_grid, method=method, periodic=False
        )
        regridded_chunk = regridder(chunk_da)

        # Free the weight matrix immediately.
        del regridder, chunk_da, chunk_src, chunk_field
        gc.collect()

        # Combine: fill NaN regions of the running result with this
        # chunk's valid values.  Where chunks overlap at boundaries,
        # the first chunk's values are kept (overlap is typically only
        # a pixel or two and values are nearly identical).
        if result is None:
            result = regridded_chunk
        else:
            result = result.fillna(regridded_chunk)
            del regridded_chunk
            gc.collect()

    return result
