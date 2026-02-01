# new_comparison.py
from herbie.core import Herbie
import xesmf as xe
import matplotlib.pyplot as plt
from comparator import tempdiff as td
from comparator import plotting as plot
from comparator import util
from datetime import datetime, timedelta
from tempfile import gettempdir

### Model & Variable Registries
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
        "aliases": ["rap"],
        "kwargs": {"model": "rap", "product": "awp130pgrb"},
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
    },
    "gfs": {
        "aliases": ["gfs"],
        "kwargs": {"model": "gfs", "product": "pgrb2.0p25"},
    },
    "rtma": {
        "aliases": ["rtma"],
        "kwargs": {"model": "rtma", "product": "anl"},
    },
}

VAR_REGISTRY = {
    "TMP": {
        "selector": "TMP:2 m above",
        "aliases": ["2 meter temperature", "t2m", "temperature", "tmp"],
        "ds_candidates": ["t2m", "tmp2m", "temperature"],
        "units_hint": "K",
        "title": "2 Meter Temperature",
        "cmap": "coolwarm"
    },
    "DPT": {
        "selector": "DPT:2 m above",
        "aliases": ["2 meter dew point", "dewpoint", "dpt"],
        "ds_candidates": ["dpt2m", "dpt", "dewpoint"],
        "units_hint": "K",
        "title": "2 Meter Dew Point",
        "cmap": "BrBG"
    }
}

### Normalization of user inputs
def normalize_model_key(user_text: str) -> str:
    """Map user input to our registry key."""
    key = user_text.strip().lower()
    # Direct hit
    if key in MODEL_REGISTRY:
        return key
    # Alias hit
    for reg_key, entry in MODEL_REGISTRY.items():
        if key in entry.get("aliases", []):
            return reg_key
    raise ValueError(f"Invalid NWP model selected: {user_text}")

def herbie_kwargs_for(model_key: str) -> dict:
    """Return kwargs for Herbie(...)"""
    entry = MODEL_REGISTRY[model_key]
    return dict(entry["kwargs"])

def normalize_var_key(user_text: str) -> str:
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

def main():
    nwp_model = input(
        "Enter NWP model to compare against RTMA : HRRR, NBM, ARW, FV3, GFS: "
    ).strip()

    date = input("Enter date (YYYY-MM-DD): ").strip()
    init_hour = int(input("Enter a valid initialization hour, in 24-hour Z-time: "))
    forecast = int(input("Enter a valid forecast hour, in 24-hour Z-time: "))
    anl_var = input("Enter analysis variable (TMP = 2m temperature, DPT = 2m dew point): ").strip()
    var_key = normalize_var_key(anl_var)
    var_meta = VAR_REGISTRY[var_key]
    var_cmap = var_meta["cmap"]
    var_title = var_meta["title"]

    # 1) Build cycle + valid datetimes
    cycle_dt = datetime.fromisoformat(f"{date} {init_hour:02d}:00")
    valid_dt = cycle_dt + timedelta(hours=forecast)

    # 2) Normalize model + variable keys, pull kwargs + selector
    try:
        model_key = normalize_model_key(nwp_model)
        nwp_kwargs = herbie_kwargs_for(model_key)
    except ValueError as e:
        print(e)
        return

    try:
        var_key = normalize_var_key(anl_var)
        selector = VAR_REGISTRY[var_key]["selector"]
    except ValueError as e:
        print(e)
        return

    # 3) Build Herbie objects using kwargs registry
    nwp = Herbie(
        cycle_dt,
        fxx=forecast,
        save_dir=gettempdir(),
        overwrite=True,
        **nwp_kwargs,
    )

    rtma_kwargs = herbie_kwargs_for("rtma")
    rtma = Herbie(
        valid_dt,
        fxx=0,
        save_dir=gettempdir(),
        overwrite=True,
        **rtma_kwargs,
    )

    # 4) Load fields (use selector, not hard-coded TMP)
    ds_nwp = nwp.xarray(selector, remove_grib=True)
    ds_rtma = rtma.xarray(selector, remove_grib=True)

    # Pick correct data variable names from each dataset
    try:
        nwp_varname = pick_data_varname_from_ds(ds_nwp, var_key)
        rtma_varname = pick_data_varname_from_ds(ds_rtma, var_key)
    except ValueError as e:
        print(e)
        return

    # 5) Regrid RTMA to NWP grid
    src_grid = {"lon": ds_rtma["longitude"], "lat": ds_rtma["latitude"]}
    tgt_grid = {"lon": ds_nwp["longitude"], "lat": ds_nwp["latitude"]}
    regridder_bilin = xe.Regridder(
        src_grid, tgt_grid, method="bilinear", periodic=False, reuse_weights=False
    )
    rtma_on_nwp_bilin = regridder_bilin(ds_rtma[rtma_varname])

    # 6) Compute difference in NWP to RTMA fields (use selected variable)
    diff = td.compute_fielddiff(ds_nwp[nwp_varname], rtma_on_nwp_bilin)

    display_name = model_key

    fig, (ax_map, ax_tbl) = plot.plot_tempdiff_map_with_table(
        ds_nwp["longitude"],
        ds_nwp["latitude"],
        diff,
        valid_dt,
        cycle_dt,
        forecast,
        display_name,
        util.major_airports_df(),
        max_rows=20,
        var_title=var_title, 
        var_cmap=var_cmap       
    )

    # Add airport markers/labels on the map axis
    plot.plot_airports(ax_map, util.major_airports_df())

    plt.show()

if __name__ == "__main__":
    main()
