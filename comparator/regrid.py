"""Memory-efficient chunked regridding via xESMF.

Splits the *target* grid into horizontal strips, builds a small
regridder (full source → target strip) for each, and concatenates
the results.  Peak memory stays proportional to *chunk_rows* rather
than the full target grid, and there are no extrapolation artefacts
because each strip only receives the values it actually needs.
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
    """Regrid *src_field* onto *tgt_grid* in target-row chunks.

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
        Number of *target*-grid rows per chunk.  Lower values use less
        memory but require more iterations.  300 is a reasonable
        default for CONUS-scale grids on 4-8 GB machines.

    Returns
    -------
    xr.DataArray
        The regridded field on *tgt_grid*.
    """
    tgt_lon = np.asarray(tgt_grid["lon"])
    tgt_lat = np.asarray(tgt_grid["lat"])

    # Number of rows in the target grid (first axis for both 1-D and 2-D).
    n_tgt_rows = tgt_lat.shape[0]

    # If the target grid is small enough, regrid in one shot.
    if n_tgt_rows <= chunk_rows:
        regridder = xe.Regridder(src_grid, tgt_grid, method=method, periodic=False)
        result = regridder(src_field)
        del regridder
        gc.collect()
        return result

    # --- chunked path: iterate over strips of target rows ---
    strips: list[np.ndarray] = []

    for start in range(0, n_tgt_rows, chunk_rows):
        end = min(start + chunk_rows, n_tgt_rows)

        # Slice the target grid for this strip of rows.
        if tgt_lat.ndim == 1:
            strip_lat = tgt_lat[start:end]
            strip_lon = tgt_lon            # lon shared across all rows
        else:
            strip_lat = tgt_lat[start:end, :]
            strip_lon = tgt_lon[start:end, :]

        strip_tgt = _grid_dict(strip_lon, strip_lat)

        regridder = xe.Regridder(
            src_grid, strip_tgt, method=method, periodic=False
        )
        regridded_strip = regridder(src_field)

        # Keep only the numpy values; we will reconstruct coordinates later.
        strips.append(np.asarray(regridded_strip))

        del regridder, regridded_strip, strip_tgt
        gc.collect()

    # Concatenate strips along the row (y) axis to reconstruct
    # the full target grid.
    full_vals = np.concatenate(strips, axis=0)
    del strips
    gc.collect()

    # Build the output DataArray with the target grid's coordinates.
    # Use the same dimension names xESMF would produce.
    result = xr.DataArray(
        full_vals,
        dims=["y", "x"],
        coords={"lat": (["y", "x"], tgt_lat) if tgt_lat.ndim == 2 else (["y"], tgt_lat),
                "lon": (["y", "x"], tgt_lon) if tgt_lon.ndim == 2 else (["x"], tgt_lon)},
    )
    return result
