# run_comparison.py
from herbie.core import Herbie
import xesmf as xe
import matplotlib.pyplot as plt
from comparator import tempdiff as td
from comparator import plotting as plot
from comparator import util
from datetime import datetime, timedelta
from tempfile import gettempdir

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

PRODUCT_REGISTRY = {
    "t2m": {
        "aliases": ["2m temp", "temp", "temperature", "2m temperature", "t2m"],
        "search": "TMP:2 m above",
        "var": "t2m",
        "plot": {
            "title": "2-m Temperature Difference (°F)",
            "cmap": "RdBu_r",
            "units": "°F",
            "vmin": -15,
            "vmax": 15,
        },
    },
    "d2m": {
        "aliases": ["2m dewpoint", "dewpoint", "dpt", "2m dpt", "d2m"],
        "search": "DPT:2 m above",
        "var": "d2m",
        "plot": {
            "title": "2-m Dewpoint Difference (°F)",
            "cmap": "BrBG",
            "units": "°F",
            "vmin": -15,
            "vmax": 15,
        },
    },
}


def normalize_product_key(user_text: str) -> str:
    key = user_text.strip().lower()
    # direct hit (if user types "t2m" or "d2m")
    if key in PRODUCT_REGISTRY:
        return key
    # alias hit
    for reg_key, entry in PRODUCT_REGISTRY.items():
        if key in entry.get("aliases", []):
            return reg_key
    raise ValueError(f"Invalid product selected: {user_text}")


def herbie_kwargs_for(model_key: str) -> dict:
        """Return kwargs for Herbie(...)"""
        entry = MODEL_REGISTRY[model_key]
        return dict(entry["kwargs"])  # shallow copy

def main():
    nwp_model = input("Enter NWP model to compare against RTMA : HRRR, NAM5K, NAM12K, NBM, RAP, ARW, FV3, HREF, or GFS: ").strip()

    product_text = input("Enter the product to analyze: 2m Temp, 2m Dewpoint: ").strip()

    date = input("Enter date (YYYY-MM-DD): ").strip()
    init_hour = int(input("Enter a valid initialization hour, in 24-hour Z-time: "))
    forecast = int(input("Enter a valid forecast hour, in 24-hour Z-time: "))

    try:
        product_key = normalize_product_key(product_text)   # <-- THIS is the important step
    except ValueError as e:
        print(e)
        return

    # 1) Build cycle + valid datetimes
    cycle_dt = datetime.fromisoformat(f"{date} {init_hour:02d}:00")
    valid_dt = cycle_dt + timedelta(hours=forecast)

    try:
        model_key = normalize_model_key(nwp_model)
        nwp_kwargs = herbie_kwargs_for(model_key)
    except ValueError as e:
        print(e)
        return
    meta = PRODUCT_REGISTRY[product_key]

    # 2) Build Herbie objects using kwargs registry
    nwp = Herbie(
        cycle_dt,
        fxx=forecast,
        save_dir=gettempdir(),
        overwrite=True,
        **nwp_kwargs,
    )

    rtma_kwargs = herbie_kwargs_for("rtma")
    rtma = Herbie(
        valid_dt,                 # RTMA is analysis valid time
        fxx=0,                    # analysis
        save_dir=gettempdir(),
        overwrite=True,
        **rtma_kwargs,
    )

    # 3) Load fields
    ds_nwp  = nwp.xarray(meta["search"], remove_grib=True)
    ds_rtma = rtma.xarray(meta["search"], remove_grib=True)

    # 4) Regrid RTMA to NWP grid
    src_grid = {"lon": ds_rtma["longitude"], "lat": ds_rtma["latitude"]}
    tgt_grid = {"lon": ds_nwp["longitude"], "lat": ds_nwp["latitude"]}
    regridder_bilin = xe.Regridder(src_grid, tgt_grid, method="bilinear", periodic=False, reuse_weights=False)
    rtma_on_nwp_bilin = regridder_bilin(ds_rtma[PRODUCT_REGISTRY[product_key]["var"]])

    # 5) Diff in °F
    diffF = td.compute_tempdiff_f(ds_nwp[PRODUCT_REGISTRY[product_key]["var"]], rtma_on_nwp_bilin)

    display_name = model_key

    fig, (ax_map, ax_tbl) = plot.plot_tempdiff_map_with_table(
        ds_nwp["longitude"],
        ds_nwp["latitude"],
        diffF,
        valid_dt,
        cycle_dt,
        forecast,
        display_name,                     
        util.major_airports_df(),
        max_rows=20,
    )

    # Add airport markers/labels on the map axis
    plot.plot_airports(ax_map, util.major_airports_df())

    plt.show()

if __name__ == "__main__":
    main()