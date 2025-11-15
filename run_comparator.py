# run_comparison.py
from herbie.core import Herbie
import xesmf as xe
import matplotlib.pyplot as plt
from comparator import tempdiff as td
from comparator import plotting as plot
from comparator import util
from datetime import datetime, timedelta

def main():
    date = input("Enter date (YYYY-MM-DD): ")
    init_hour = int(input("Enter initialization hour (HH): "))
    forecast = int(input("Enter forecast hour (0...18 for off-hour HRRR runs & 48 for long-range runs): "))

    # 1) Build the HRRR cycle datetime (initialization time)
    cycle_dt = datetime.fromisoformat(f"{date} {init_hour:02d}:00")

    # Compute the forecast valid time (handles day/month/year rollovers)
    valid_dt = cycle_dt + timedelta(hours=forecast)

    # Optional: a display string if you need it
    displayhour = valid_dt.strftime("%H:%M")

    # Compute display hour safely
    displayhour = (forecast + init_hour) % 24

    # HRRR: pass the cycle time + forecast hour
    hrrr = Herbie(
        cycle_dt,
        model="hrrr",
        product="sfc",
        fxx=forecast
    )

    # RTMA: pass the *valid* analysis time so it matches HRRR's valid time
    rtma = Herbie(
        valid_dt,
        model="rtma",
        product="anl"
    )

    ### Load into xarray dataarrays
    ds_hrrr = hrrr.xarray("TMP:2 m above")
    ds_rtma = rtma.xarray("TMP:2 m above")

    # 2) Curvilinear grids from your HRRR/RTMA Datasets
    src_grid = {"lon": ds_rtma["longitude"], "lat": ds_rtma["latitude"]}
    tgt_grid = {"lon": ds_hrrr["longitude"], "lat": ds_hrrr["latitude"]}

    # Fast sanity first (nearest) & wrap in xarray
    regridder_fast = xe.Regridder(src_grid, tgt_grid, method="nearest_s2d", periodic=True, reuse_weights=False)
    rtma_on_hrrr_fast = regridder_fast(ds_rtma["t2m"])

    # 3️⃣ Compute Fahrenheit temperature difference
    diffF = td.compute_tempdiff_f(ds_hrrr["t2m"], rtma_on_hrrr_fast)

    # 4️⃣ Plot the map + airports
    fig, ax = plot.plot_tempdiff_map(ds_hrrr["longitude"], ds_hrrr["latitude"], diffF)
    plot.plot_airports(ax, util.major_airports_df())
    plt.show()

if __name__ == "__main__":
    main()
