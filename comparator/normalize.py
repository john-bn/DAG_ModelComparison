### TODO: Add URMA as an option 
### Big Registries for NWP & RTMA kwargs
MODEL_REGISTRY = {
    "hrrr": {
        "aliases": ["hrrr"],
        "kwargs": {"model": "hrrr", "product": "sfc"},
    },
    "nam5k": {
        "aliases": ["nam5k", "nam-conusnest", "namnest", "nam hi-res nest"],
        "kwargs": {"model": "nam", "product": "conusnest.hiresf"},
    },
    "nam12k": {
        "aliases": ["nam12k", "nam-12km", "nam"],
        "kwargs": {"model": "nam", "product": "awip12"},
    },
    "nbm": {
        "aliases": ["nbm"],
        "kwargs": {"model": "nbm", "product": "co"},
    },
    "rap": {
        "aliases": ["rap", "rapid refresh"],
        "kwargs": {"model": "rap", "product": "awp130bgrb"},
    },
    "arw": {
        "aliases": ["arw", "ncar-arw"],
        "kwargs": {"model": "hiresw", "product": "arw_5km", "domain": "conus", "member": 2},
    },
    "fv3": {
        "aliases": ["fv3"],
        "kwargs": {"model": "hiresw", "product": "fv3_5km", "domain": "conus", "member": 1},
    },
    "href": {
        "aliases": ["href"],
        "kwargs": {"model": "href", "product": "mean", "domain": "conus"},
        "selector_map": {
            "TMP": r"TMP:2 m above ground:\d+ hour fcst",
            "DPT": r"DPT:2 m above ground:\d+ hour fcst",
        },
        "xarray_kwargs": {
            "backend_kwargs": {
                "read_keys": [
                    "derivedForecast",
                    "parameterName",
                    "parameterUnits",
                    "stepRange",
                    "uvRelativeToGrid",
                    "shapeOfTheEarth",
                    "orientationOfTheGridInDegrees",
                    "southPoleOnProjectionPlane",
                    "LaDInDegrees",
                    "LoVInDegrees",
                    "Latin1InDegrees",
                    "Latin2InDegrees",
                ],
            },
        },
    },
    "gfs": {
        "aliases": ["gfs"],
        "kwargs": {"model": "gfs", "product": "pgrb2.0p25"},
    },
    "ifs": {
        "aliases": ["ifs", "ecmwf"],
        "kwargs": {"model": "ifs", "product": "oper"},
        "selector_map": {
            "TMP": ":2t:",
            "DPT": ":2d:",
        },
    },
    "rtma": {
        "aliases": ["rtma"],
        "kwargs": {"model": "rtma", "product": "anl"},
    },
}
### Forecast-range metadata per model
# cycle_interval : hours between init cycles (e.g. 1 for hourly, 6 for 4x/day)
# max_fxx        : longest forecast hour produced by any cycle
# For models whose extended runs are only on certain cycles, use
# extended_cycles / base_max_fxx so the shorter runs aren't over-counted.
MODEL_FORECAST_META = {
    "hrrr": {
        "cycle_interval": 1,
        "max_fxx": 48,
        "extended_cycles": [0, 6, 12, 18],
        "base_max_fxx": 18,
    },
    "nam5k": {"cycle_interval": 6, "max_fxx": 60},
    "nam12k": {"cycle_interval": 6, "max_fxx": 84},
    "rap": {
        "cycle_interval": 1,
        "max_fxx": 51,
        "extended_cycles": [3, 9, 15, 21],
        "base_max_fxx": 21,
    },
    "nbm": {"cycle_interval": 1, "max_fxx": 264},
    "arw": {"cycle_interval": 12, "max_fxx": 48},
    "fv3": {"cycle_interval": 12, "max_fxx": 48},
    "href": {"cycle_interval": 12, "max_fxx": 48},
    "gfs": {"cycle_interval": 6, "max_fxx": 384},
    "ifs": {"cycle_interval": 12, "max_fxx": 240},
}


def find_runs_for_valid_time(model_key: str, valid_dt) -> list[tuple]:
    """Return every (cycle_dt, fxx) pair whose forecast covers *valid_dt*.

    Results are sorted oldest-cycle-first so a GIF shows forecast evolution
    from long-range down to the shortest lead time.
    """
    from datetime import timedelta

    meta = MODEL_FORECAST_META.get(model_key)
    if meta is None:
        raise ValueError(
            f"No forecast metadata for model '{model_key}'. "
            f"Known models: {', '.join(MODEL_FORECAST_META)}"
        )

    interval = meta["cycle_interval"]
    global_max = meta["max_fxx"]
    extended_cycles = meta.get("extended_cycles")
    base_max = meta.get("base_max_fxx", global_max)

    results = []
    for hours_back in range(0, global_max + 1):
        candidate = valid_dt - timedelta(hours=hours_back)
        # Only keep cycles that land on a valid init hour
        if candidate.hour % interval != 0:
            continue
        fxx = hours_back
        # Determine the max forecast hour for this specific cycle
        if extended_cycles is not None:
            cycle_max = global_max if candidate.hour in extended_cycles else base_max
        else:
            cycle_max = global_max
        if fxx <= cycle_max:
            results.append((candidate, fxx))

    # Oldest init first → GIF animates from long-range to short-range
    results.sort(key=lambda pair: pair[0])
    return results


### TODO: Add 2m RH & Vis as variables
### Registry for variable kwargs
VAR_REGISTRY = {
    "TMP": {
        "selector": "TMP:2 m above",
        "aliases": ["2 meter temperature", "t2m", "temperature", "tmp"],
        "ds_candidates": ["t2m", "tmp2m", "temperature", "t", "2t"],
        "units_hint": "K",
        "title": "2 Meter Temperature",
        "cmap": "coolwarm"
    },
    "DPT": {
        "selector": "DPT:2 m above",
        "aliases": ["2 meter dew point", "dewpoint", "dpt"],
        "ds_candidates": ["d2m", "dpt2m", "dpt", "dewpoint", "d", "2d"],
        "units_hint": "K",
        "title": "2 Meter Dew Point",
        "cmap": "BrBG"
    }
}

### Normalization of user inputs
def normalize_model_key(user_text: str) -> str:
    """Map user input to our model registry key."""
    key = user_text.strip().lower()
    if key in MODEL_REGISTRY:
        return key
    for reg_key, entry in MODEL_REGISTRY.items():
        if key in entry.get("aliases", []):
            return reg_key
    raise ValueError(f"Invalid NWP model selected: {user_text}")

def herbie_kwargs_for(model_key: str) -> dict:
    """Return kwargs for Herbie(...)"""
    entry = MODEL_REGISTRY[model_key]
    return dict(entry["kwargs"])

def get_xarray_kwargs(model_key: str) -> dict:
    """Return extra kwargs to pass to Herbie.xarray() for a given model.

    Models with an ``xarray_kwargs`` entry (e.g. HREF) may need extra
    cfgrib backend_kwargs such as ``read_keys`` to correctly decode
    ensemble-derived GRIB2 products (PDT 4.2).
    """
    return dict(MODEL_REGISTRY[model_key].get("xarray_kwargs", {}))

def get_selector(model_key: str, var_key: str) -> str:
    """Return the Herbie search string for a model+variable pair.

    Models with a selector_map (eccodes-indexed, e.g. IFS) use their
    own selectors; all others fall back to VAR_REGISTRY.
    """
    selector_map = MODEL_REGISTRY[model_key].get("selector_map")
    if selector_map and var_key in selector_map:
        return selector_map[var_key]
    return VAR_REGISTRY[var_key]["selector"]

def wrap_longitude(ds):
    """Convert 0-360 longitudes to -180..180 if needed, then sort."""
    lon = ds["longitude"]
    if float(lon.max()) > 180.0:
        wrapped = ((lon.values + 180.0) % 360.0) - 180.0
        ds["longitude"].values[:] = wrapped
        if ds["longitude"].ndim == 1:
            ds = ds.sortby("longitude")
    return ds

def ensure_dataset(ds_or_list, var_key=None):
    """If Herbie returned a list of datasets (multi-hypercube), pick the best.

    When *var_key* is provided we scan the list for a dataset that contains
    one of the expected variable names (from VAR_REGISTRY ds_candidates).
    Falls back to the first dataset if no candidate matches.
    """
    if not isinstance(ds_or_list, list):
        return ds_or_list
    if var_key is not None:
        candidates = VAR_REGISTRY.get(var_key, {}).get("ds_candidates", [])
        for ds in ds_or_list:
            data_vars = list(ds.data_vars)
            for cand in candidates:
                if cand in data_vars:
                    return ds
            # also try substring match
            for cand in candidates:
                for dv in data_vars:
                    if cand in dv:
                        return ds
    return ds_or_list[0]

def normalize_var_key(user_text: str) -> str:
    """Map user input to our variable registry key."""
    key = user_text.strip().lower()
    for var_key, entry in VAR_REGISTRY.items():
        if key == var_key.lower() or key in entry["aliases"]:
            return var_key
    raise ValueError(f"Invalid analysis variable: {user_text}")

### Select variable from xarray.Dataset using VAR_REGISTRY
def pick_data_varname_from_ds(ds, var_key: str) -> str:
    """
    Pick the right data variable from a Herbie-returned xarray.Dataset.
    - If only one data_var exists, use it
    - Else use VAR_REGISTRY[var_key]['ds_candidates']
    """
    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        return data_vars[0]

    candidates = VAR_REGISTRY[var_key].get("ds_candidates", [])
    for cand in candidates:
        if cand in ds:
            return cand

    # fallback: substring match (helps with odd naming)
    for cand in candidates:
        for dv in data_vars:
            if cand in dv:
                return dv

    raise ValueError(
        f"Could not determine data variable for {var_key}. "
        f"Dataset contains data_vars={data_vars}"
    )
