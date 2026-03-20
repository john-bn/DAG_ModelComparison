# new_comparison.py
import os
if os.environ.get("CODESPACES"):
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — saves memory in Codespaces

from herbie.core import Herbie
import xesmf as xe
import matplotlib.pyplot as plt
from comparator import fielddiff as fd
from comparator import plotting as plot
from comparator import util
from comparator import normalize as norm
from datetime import datetime, timedelta
from pathlib import Path
import gc

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

def main():
    nwp_model = input("Enter NWP model to compare against RTMA : HRRR, RAP, NBM, ARW, FV3, GFS, IFS, HREF: ").strip()

    date = input("Enter date (YYYY-MM-DD): ").strip()
    init_hour = int(input("Enter a valid initialization hour, in 24-hour Z-time: "))
    forecast = int(input("Enter a valid forecast hour, in 24-hour Z-time: "))
    anl_var = input("Enter analysis variable (TMP = 2m temperature, DPT = 2m dew point): ").strip()
    var_key = norm.normalize_var_key(anl_var)
    var_meta = norm.VAR_REGISTRY[var_key]
    var_cmap = var_meta["cmap"]
    var_title = var_meta["title"]

    # 1) Build cycle & valid datetimes
    cycle_dt = datetime.fromisoformat(f"{date} {init_hour:02d}:00")
    valid_dt = cycle_dt + timedelta(hours=forecast)

    # 2) Normalize model & variable keys, pull kwargs, & selector
    try:
        model_key = norm.normalize_model_key(nwp_model)
        nwp_kwargs = norm.herbie_kwargs_for(model_key)
    except ValueError as e:
        print(e)
        return

    try:
        var_key = norm.normalize_var_key(anl_var)
        selector = norm.get_selector(model_key, var_key)
    except ValueError as e:
        print(e)
        return

    # 3) Build Herbie objects using kwargs registry
    nwp = Herbie(
        cycle_dt,
        fxx=forecast,
        save_dir=str(DATA_DIR),
        overwrite=False,
        **nwp_kwargs,
    )
    if not nwp:
        print(f"Could not find {model_key.upper()} data for {cycle_dt:%Y-%m-%d %H}Z F{forecast:02d}.")
        print("The file may not be available yet, or may have been removed from the server.")
        return

    rtma_kwargs = norm.herbie_kwargs_for("rtma")
    rtma = Herbie(
        valid_dt,
        fxx=0,
        save_dir=str(DATA_DIR),
        overwrite=False,
        **rtma_kwargs,
    )
    if not rtma:
        print(f"Could not find RTMA data for {valid_dt:%Y-%m-%d %H}Z.")
        print("The file may not be available yet, or may have been removed from the server.")
        return

    # 4) Load NWP field — extract only what we need, then free the dataset
    nwp_xr_kwargs = norm.get_xarray_kwargs(model_key)
    try:
        ds_nwp = norm.ensure_dataset(
            nwp.xarray(selector, remove_grib=True, **nwp_xr_kwargs),
            var_key=var_key,
        )
    except Exception as e:
        print(f"Failed to load {model_key} GRIB data: {e}")
        return
    ds_nwp = norm.wrap_longitude(ds_nwp)

    try:
        nwp_varname = norm.pick_data_varname_from_ds(ds_nwp, var_key)
    except ValueError as e:
        print(e)
        return

    # Extract lon/lat/data as float32 in-memory arrays, then drop the full dataset
    nwp_lon = ds_nwp["longitude"].load().astype("float32")
    nwp_lat = ds_nwp["latitude"].load().astype("float32")
    nwp_field = ds_nwp[nwp_varname].load().astype("float32")
    tgt_grid = {"lon": nwp_lon, "lat": nwp_lat}
    del ds_nwp, nwp
    gc.collect()

    # 5) Load RTMA field — same pattern: extract and free
    rtma_selector = norm.get_selector("rtma", var_key)
    try:
        ds_rtma = norm.ensure_dataset(
            rtma.xarray(rtma_selector, remove_grib=True),
            var_key=var_key,
        )
    except Exception as e:
        print(f"Failed to load RTMA GRIB data: {e}")
        return

    try:
        rtma_varname = norm.pick_data_varname_from_ds(ds_rtma, var_key)
    except ValueError as e:
        print(e)
        return

    rtma_field = ds_rtma[rtma_varname].load().astype("float32")
    src_grid = {"lon": ds_rtma["longitude"].load().astype("float32"),
                "lat": ds_rtma["latitude"].load().astype("float32")}
    del ds_rtma, rtma
    gc.collect()

    # 6) Regrid RTMA to model grid — free regridder immediately after use
    regridder_bilin = xe.Regridder(
        src_grid, tgt_grid, method="bilinear", periodic=False
    )
    rtma_on_nwp_bilin = regridder_bilin(rtma_field)
    del regridder_bilin, rtma_field, src_grid, tgt_grid
    gc.collect()

    # 7) Compute difference in NWP to RTMA fields
    diff = fd.compute_fielddiff(nwp_field, rtma_on_nwp_bilin)
    del nwp_field, rtma_on_nwp_bilin
    gc.collect()

    display_name = model_key

    # Coarsen grids for plotting — full resolution is invisible at output DPI.
    # Keep full-res lon/lat/diff for the airport ΔT table (computed inside plot fn),
    # but pass coarsened versions for the pcolormesh map.
    import numpy as _np
    _MAX_PLOT_CELLS = 500_000  # target ceiling for pcolormesh
    _total = max(nwp_lon.size, 1)
    _stride = max(1, int(_np.ceil(_np.sqrt(_total / _MAX_PLOT_CELLS))))
    if _stride > 1 and nwp_lon.ndim >= 1:
        _s = _stride
        plot_lon = nwp_lon.values[::_s, ::_s] if nwp_lon.ndim == 2 else nwp_lon.values[::_s]
        plot_lat = nwp_lat.values[::_s, ::_s] if nwp_lat.ndim == 2 else nwp_lat.values[::_s]
        plot_diff = diff.values[::_s, ::_s] if diff.ndim == 2 else diff.values[::_s]
        import xarray as _xr
        plot_lon = _xr.DataArray(plot_lon)
        plot_lat = _xr.DataArray(plot_lat)
        plot_diff = _xr.DataArray(plot_diff)
    else:
        plot_lon, plot_lat, plot_diff = nwp_lon, nwp_lat, diff

    fig, (ax_map, ax_tbl) = plot.plot_tempdiff_map_with_table(
        nwp_lon,
        nwp_lat,
        diff,
        valid_dt,
        cycle_dt,
        forecast,
        display_name,
        util.major_airports_df(),
        max_rows=20,
        var_title=var_title,
        var_cmap=var_cmap,
        plot_meta=var_meta,
        plot_lon=plot_lon,
        plot_lat=plot_lat,
        plot_diff=plot_diff,
    )

    # Add airport markers & labels on the map axis
    plot.plot_airports(ax_map, util.major_airports_df())

    # Free data arrays before rendering — figure already holds a rasterized copy
    del diff, nwp_lon, nwp_lat
    gc.collect()

    # Save plot to figures/ directory
    out_dir = Path("./figures")
    out_dir.mkdir(exist_ok=True)
    filename = f"{display_name}_rtma_{var_key}_{valid_dt:%Y%m%d_%H%MZ}.png"
    out_path = out_dir / filename
    save_dpi = 100 if os.environ.get("CODESPACES") else 150
    fig.savefig(out_path, dpi=save_dpi, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    # Show the plot to the user
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        plt.show()
    elif os.environ.get("CODESPACES") or os.environ.get("TERM_PROGRAM") == "vscode":
        import subprocess
        subprocess.Popen(["code", str(out_path)])

    plt.close(fig)

if __name__ == "__main__":
    main()
