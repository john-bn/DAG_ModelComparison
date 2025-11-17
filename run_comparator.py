# run_comparison.py
from herbie.core import Herbie
import xesmf as xe
import matplotlib.pyplot as plt
from comparator import tempdiff as td
from comparator import plotting as plot
from comparator import util
from datetime import datetime, timedelta
from tempfile import gettempdir

def main():
    nwp_model = input("Enter NWP model to compare against RTMA : HRRR, NAM5K, NAM12K, IFS (WIP), NBM, RAP, ARW (WIP), ARW2 (WIP), FV3 (WIP), HREF (WIP), RRFS (WIP), or GFS (WIP): ").strip().lower()
    date = input("Enter date (YYYY-MM-DD): ")
    init_hour = int(input("Enter a valid initialization hour, in 24-hour Z-time: "))
    forecast = int(input("Enter a valid forecast hour, in 24-hour Z-time: "))    
    model_dict = {"hrrr": "sfc", "nam5k": "conusnest.hiresf", "nam12k": "awip12", "ifs": "oper", "nbm": "co", "rap": "awp130pgrb", "arw": "arw_5km", "arw2": "arw_2p5km", "fv3": "fv3_5km", "href": "???", "rrfs": "prslev", "gfs": "pgrb2.0p25"}
    model_product = model_dict.get(nwp_model)
    
    if nwp_model.startswith("nam"):
        nwp_model = "nam"

    if model_product is None:
        print("Invalid NWP model selected.")
        return

    # 1) Build the user model cycle datetime (initialization time)
    cycle_dt = datetime.fromisoformat(f"{date} {init_hour:02d}:00")

    # Compute the forecast valid time (handles day/month/year rollovers)
    valid_dt = cycle_dt + timedelta(hours=forecast)

    # User model: pass the cycle time + forecast hour
    nwp = Herbie(
        cycle_dt,
        model=nwp_model,
        product=model_product,
        fxx=forecast,
        save_dir=gettempdir(),
        overwrite=True
    )

    # RTMA: pass the *valid* analysis time so it matches nwp's valid time
    rtma = Herbie(
        valid_dt,
        model="rtma",
        product="anl",
        save_dir=gettempdir(),
        overwrite=True
    )

    ### Load into xarray dataarrays
    ds_nwp = nwp.xarray("TMP:2 m above", remove_grib=True)
    ds_rtma = rtma.xarray("TMP:2 m above", remove_grib=True)

    # 2) Curvilinear grids from your nwp/RTMA Datasets
    src_grid = {"lon": ds_rtma["longitude"], "lat": ds_rtma["latitude"]}
    tgt_grid = {"lon": ds_nwp["longitude"], "lat": ds_nwp["latitude"]}

    # Fast sanity first (nearest) & wrap in xarray
    regridder_bilin = xe.Regridder(src_grid, tgt_grid, method="bilinear", periodic=False, reuse_weights=False)
    rtma_on_nwp_bilin = regridder_bilin(ds_rtma["t2m"])

    # 3️⃣ Compute Fahrenheit temperature difference
    diffF = td.compute_tempdiff_f(ds_nwp["t2m"], rtma_on_nwp_bilin)

    # 4️⃣ Plot the map + airports
    fig, ax = plot.plot_tempdiff_map(ds_nwp["longitude"], ds_nwp["latitude"], diffF, valid_dt=valid_dt, cycle_dt=cycle_dt, forecast=forecast, model_name=nwp_model.upper())
    plot.plot_airports(ax, util.major_airports_df())
    plt.show()

if __name__ == "__main__":
    main()
