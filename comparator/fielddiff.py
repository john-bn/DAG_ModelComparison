import numpy as np
import xarray as xr

_METERS_PER_SM = 1609.344
_MPH_PER_MPS = 2.23694


def compute_fielddiff(nwp_field: xr.DataArray, anl_field: xr.DataArray, var_key: str = "TMP") -> xr.DataArray:
    """Compute field difference NWP - analysis on the SAME grid.

    The analysis field is RTMA or URMA (see VERIFICATION_SOURCES).
    TMP / DPT (Kelvin inputs): returns Fahrenheit difference.
    VIS (meter inputs): returns statute-mile difference.
    WIND / GUST (m/s inputs): returns mph difference.
    Assumes inputs are on identical (y,x) coords.
    """
    h, r = xr.align(nwp_field, anl_field, join="exact")

    if var_key in ("TMP", "DPT"):
        valid = (np.isfinite(h) & np.isfinite(r) & (h > 150) & (h < 330) & (r > 150) & (r < 330))
        return (h - r).where(valid) * 9 / 5
    elif var_key == "VIS":
        valid = (np.isfinite(h) & np.isfinite(r) & (h >= 0) & (h <= 24000) & (r >= 0) & (r <= 24000))
        return (h - r).where(valid) / _METERS_PER_SM
    elif var_key in ("WIND", "GUST"):
        valid = (np.isfinite(h) & np.isfinite(r) & (h >= 0) & (h <= 150) & (r >= 0) & (r <= 150))
        return (h - r).where(valid) * _MPH_PER_MPS
    else:
        raise ValueError(f"No fielddiff logic for var_key='{var_key}'")
