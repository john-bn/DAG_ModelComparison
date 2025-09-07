# DAG_ModelComparison

Utilities for comparing HRRR model output with RTMA analysis fields.
The code streams GRIB2 files directly from NOAA's AWS S3 buckets using
xarray and fsspec, avoiding the need to download data to disk.  It also
provides helper functions to compute simple difference statistics between
model output and analysis fields.  The ``comparison.py`` module is backend
agnostic and can be applied to any RTMA field.

Example::

    from datetime import datetime
    from data_fetcher import get_hrrr, get_rtma
    from comparison import evaluate

    time = datetime(2024, 5, 1, 0)
    hrrr_t2m = get_hrrr(time)
    rtma_t2m = get_rtma(time)
    diff, metrics = evaluate(hrrr_t2m, rtma_t2m)
    print(metrics)

Tests only require the standard library and can be run with ``pytest``.
