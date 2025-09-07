"""Helpers for streaming HRRR and RTMA data using xarray and fsspec."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import fsspec  # type: ignore
import xarray as xr  # type: ignore


def _open_grib(url: str, field: str, level: int) -> xr.DataArray:
    """Open a remote GRIB2 file and extract a single field."""
    backend_kwargs = {
        "filter_by_keys": {
            "typeOfLevel": "heightAboveGround",
            "level": level,
            "shortName": field,
        }
    }
    fs = fsspec.open(url)
    with fs.open() as f:
        ds = xr.open_dataset(f, engine="cfgrib", backend_kwargs=backend_kwargs)
    return ds[field]


def get_hrrr(
    time: datetime,
    field: str = "2t",
    level: int = 2,
    forecast_hour: int = 0,
) -> xr.DataArray:
    """Stream a field from the HRRR model on AWS."""
    date = time.strftime("%Y%m%d")
    hour = time.hour
    url = (
        f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date}/conus/"
        f"hrrr.t{hour:02d}z.wrfsfcf{forecast_hour:02d}.grib2"
    )
    return _open_grib(url, field, level)


def get_rtma(time: datetime, field: str = "2t", level: int = 2) -> xr.DataArray:
    """Stream a field from the RTMA analysis on AWS."""
    date = time.strftime("%Y%m%d")
    hour = time.hour
    url = (
        f"https://noaa-rtma2p5-pds.s3.amazonaws.com/rtma2p5.{date}/"
        f"rtma2p5.t{hour:02d}z.2dvaranl_ndfd.grib2"
    )
    return _open_grib(url, field, level)
