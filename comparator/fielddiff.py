import numpy as np
import xarray as xr

_METERS_PER_SM = 1609.344
_MPS_PER_MPH = 0.44704


def compute_fielddiff(nwp_field: xr.DataArray, rtma_field: xr.DataArray, var_key: str = "TMP") -> xr.DataArray:
    """Compute field difference NWP - RTMA on the SAME grid.

    TMP / DPT (Kelvin inputs): returns Fahrenheit difference.
    VIS (meter inputs): returns statute-mile difference.
    WSP / GUST (m/s inputs): returns mph difference.
    Assumes inputs are on identical (y,x) coords.
    """
    h, r = xr.align(nwp_field, rtma_field, join="exact")

    if var_key in ("TMP", "DPT"):
        valid = (np.isfinite(h) & np.isfinite(r) & (h > 150) & (h < 330) & (r > 150) & (r < 330))
        return (h - r).where(valid) * 9 / 5
    elif var_key == "VIS":
        valid = (np.isfinite(h) & np.isfinite(r) & (h >= 0) & (h <= 24000) & (r >= 0) & (r <= 24000))
        return (h - r).where(valid) / _METERS_PER_SM
    elif var_key in ("WSP", "GUST"):
        valid = (np.isfinite(h) & np.isfinite(r) & (h >= 0) & (h <= 120) & (r >= 0) & (r <= 120))
        return (h - r).where(valid) / _MPS_PER_MPH
    else:
        raise ValueError(f"No fielddiff logic for var_key='{var_key}'")
