### TODO: Develop new fielddiff to allow calculations to be made for 2m Relative Humidity
### TODO: Develop or modify fielddiff to all calculations for visibility

import numpy as np
import xarray as xr

### Computes the difference between the NWP data & the RTMA dataset
def compute_fielddiff(hrrr_t2m: xr.DataArray, rtma_on_hrrr_t2m) -> xr.DataArray:
    """Compute Fahrenheit temperature difference HRRR - RTMA on the SAME grid.
    Assumes inputs are Kelvin on identical (y,x,time) coords. Only works for temp & dewpoint
    """
    h, r = xr.align(hrrr_t2m, rtma_on_hrrr_t2m, join="exact")
    valid = (np.isfinite(h) & np.isfinite(r) & (h > 150) & (h < 330) & (r > 150) & (r < 330))
    return (h - r).where(valid) * 9/5
