# new_comparison.py
from herbie.core import Herbie
import xesmf as xe
import matplotlib.pyplot as plt
from comparator import tempdiff as td
from comparator import plotting as plot
from comparator import util
from comparator import normalize as norm
from datetime import datetime, timedelta
from tempfile import gettempdir

def main():
    nwp_model = input("Enter NWP model to compare against RTMA : HRRR, NBM, ARW, FV3, GFS: ").strip()

    date = input("Enter date (YYYY-MM-DD): ").strip()
    init_hour = int(input("Enter a valid initialization hour, in 24-hour Z-time: "))
    forecast = int(input("Enter a valid forecast hour, in 24-hour Z-time: "))
    anl_var = input("Enter analysis variable (TMP = 2m temperature, DPT = 2m dew point): ").strip()
    var_key = norm.normalize_var_key(anl_var)
    var_meta = norm.VAR_REGISTRY[var_key]
    var_cmap = var_meta["cmap"]
    var_title = var_meta["title"]

    # 1) Build cycle + valid datetimes
    cycle_dt = datetime.fromisoformat(f"{date} {init_hour:02d}:00")
    valid_dt = cycle_dt + timedelta(hours=forecast)

    # 2) Normalize model + variable keys, pull kwargs + selector
    try:
        model_key = norm.normalize_model_key(nwp_model)
        nwp_kwargs = norm.herbie_kwargs_for(model_key)
    except ValueError as e:
        print(e)
        return

    try:
        var_key = norm.normalize_var_key(anl_var)
        selector = norm.VAR_REGISTRY[var_key]["selector"]
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

    rtma_kwargs = norm.herbie_kwargs_for("rtma")
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
        nwp_varname = norm.pick_data_varname_from_ds(ds_nwp, var_key)
        rtma_varname = norm.pick_data_varname_from_ds(ds_rtma, var_key)
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
        var_cmap=var_cmap,
        plot_meta=var_meta
    )

    # Add airport markers/labels on the map axis
    plot.plot_airports(ax_map, util.major_airports_df())

    plt.show()

if __name__ == "__main__":
    main()
