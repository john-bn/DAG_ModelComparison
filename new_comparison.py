# new_comparison.py
from herbie.core import Herbie
import xesmf as xe
import matplotlib.pyplot as plt
from comparator import fielddiff as fd
from comparator import plotting as plot
from comparator import util
from comparator import normalize as norm
from comparator.build_gif import create_gif
from datetime import datetime, timedelta
from pathlib import Path
import os

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

FIGURE_DIR = Path("./figures")
FIGURE_DIR.mkdir(exist_ok=True)


def generate_comparison_frame(
    model_key,
    var_key,
    cycle_dt,
    forecast_hour,
    save_dir=DATA_DIR,
    out_dir=FIGURE_DIR,
):
    """Generate a single NWP-vs-RTMA comparison plot and return the saved path.

    Returns the Path to the saved PNG, or None if the frame could not be built.
    """
    var_meta = norm.VAR_REGISTRY[var_key]
    var_cmap = var_meta["cmap"]
    var_title = var_meta["title"]
    valid_dt = cycle_dt + timedelta(hours=forecast_hour)

    nwp_kwargs = norm.herbie_kwargs_for(model_key)
    selector = norm.get_selector(model_key, var_key)

    # --- Fetch NWP data ---
    nwp = Herbie(
        cycle_dt,
        fxx=forecast_hour,
        save_dir=str(save_dir),
        overwrite=True,
        **nwp_kwargs,
    )
    if not nwp:
        print(
            f"  Could not find {model_key.upper()} data for "
            f"{cycle_dt:%Y-%m-%d %H}Z F{forecast_hour:02d}. Skipping."
        )
        return None

    # --- Fetch RTMA data ---
    rtma_kwargs = norm.herbie_kwargs_for("rtma")
    rtma = Herbie(
        valid_dt,
        fxx=0,
        save_dir=str(save_dir),
        overwrite=True,
        **rtma_kwargs,
    )
    if not rtma:
        print(
            f"  Could not find RTMA data for {valid_dt:%Y-%m-%d %H}Z. Skipping."
        )
        return None

    # --- Load fields ---
    nwp_xr_kwargs = norm.get_xarray_kwargs(model_key)
    try:
        ds_nwp = norm.ensure_dataset(
            nwp.xarray(selector, remove_grib=True, **nwp_xr_kwargs),
            var_key=var_key,
        )
    except Exception as e:
        print(f"  Failed to load {model_key} GRIB data (F{forecast_hour:02d}): {e}")
        return None
    ds_nwp = norm.wrap_longitude(ds_nwp)

    rtma_selector = norm.get_selector("rtma", var_key)
    try:
        ds_rtma = norm.ensure_dataset(
            rtma.xarray(rtma_selector, remove_grib=True),
            var_key=var_key,
        )
    except Exception as e:
        print(f"  Failed to load RTMA GRIB data ({valid_dt:%Y-%m-%d %H}Z): {e}")
        return None

    # --- Variable name resolution ---
    try:
        nwp_varname = norm.pick_data_varname_from_ds(ds_nwp, var_key)
        rtma_varname = norm.pick_data_varname_from_ds(ds_rtma, var_key)
    except ValueError as e:
        print(f"  {e}")
        return None

    # --- Regrid RTMA to model grid ---
    src_grid = {"lon": ds_rtma["longitude"], "lat": ds_rtma["latitude"]}
    tgt_grid = {"lon": ds_nwp["longitude"], "lat": ds_nwp["latitude"]}
    regridder = xe.Regridder(
        src_grid, tgt_grid, method="bilinear", periodic=False, reuse_weights=False
    )
    rtma_on_nwp = regridder(ds_rtma[rtma_varname])

    # --- Compute difference ---
    diff = fd.compute_fielddiff(ds_nwp[nwp_varname], rtma_on_nwp)

    display_name = model_key

    fig, (ax_map, ax_tbl) = plot.plot_tempdiff_map_with_table(
        ds_nwp["longitude"],
        ds_nwp["latitude"],
        diff,
        valid_dt,
        cycle_dt,
        forecast_hour,
        display_name,
        util.major_airports_df(),
        max_rows=20,
        var_title=var_title,
        var_cmap=var_cmap,
        plot_meta=var_meta,
    )

    plot.plot_airports(ax_map, util.major_airports_df())

    # --- Save ---
    filename = f"{display_name}_rtma_{var_key}_{valid_dt:%Y%m%d_%H%MZ}.png"
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved frame: {out_path}")
    return out_path


def main():
    nwp_model = input(
        "Enter NWP model to compare against RTMA : "
        "HRRR, NAM5k, NAM12k, RAP, NBM, ARW, FV3, GFS, IFS, HREF: "
    ).strip()

    date = input("Enter date (YYYY-MM-DD): ").strip()
    init_hour = int(input("Enter a valid initialization hour, in 24-hour Z-time: "))
    forecast = int(input("Enter a valid forecast hour, in 24-hour Z-time: "))
    anl_var = input(
        "Enter analysis variable (TMP = 2m temperature, DPT = 2m dew point): "
    ).strip()
    animate = input("Animate the plot? (y/n): ").strip().lower()

    # --- Validate model & variable early ---
    try:
        model_key = norm.normalize_model_key(nwp_model)
    except ValueError as e:
        print(e)
        return

    try:
        var_key = norm.normalize_var_key(anl_var)
    except ValueError as e:
        print(e)
        return

    cycle_dt = datetime.fromisoformat(f"{date} {init_hour:02d}:00")

    if animate == "y":
        # --- GIF mode: ask for timeframe range ---
        start_fxx = int(
            input("Enter the START forecast hour for the GIF timeframe: ")
        )
        end_fxx = int(
            input("Enter the END forecast hour for the GIF timeframe: ")
        )
        if start_fxx > end_fxx:
            print("Start forecast hour must be <= end forecast hour.")
            return

        print(
            f"\nGenerating frames for {model_key.upper()} "
            f"F{start_fxx:02d}–F{end_fxx:02d} "
            f"(init {cycle_dt:%Y-%m-%d %H}Z) ..."
        )

        frame_paths = []
        for fxx in range(start_fxx, end_fxx + 1):
            print(f"\n--- Forecast hour {fxx:02d} ---")
            path = generate_comparison_frame(
                model_key, var_key, cycle_dt, fxx
            )
            if path is not None:
                frame_paths.append(path)

        if not frame_paths:
            print("No frames were generated. Cannot create GIF.")
            return

        gif_name = (
            f"{model_key}_rtma_{var_key}_{cycle_dt:%Y%m%d_%H}Z_"
            f"F{start_fxx:02d}-F{end_fxx:02d}.gif"
        )
        gif_path = FIGURE_DIR / gif_name
        create_gif(frame_paths, gif_path, duration=500)
        print(f"\nGIF saved to {gif_path}  ({len(frame_paths)} frames)")

    else:
        # --- Single-frame mode ---
        out_path = generate_comparison_frame(
            model_key, var_key, cycle_dt, forecast
        )
        if out_path is None:
            return
        print(f"Plot saved to {out_path}")

        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            from matplotlib import pyplot as _plt

            _plt.show()
        elif os.environ.get("CODESPACES") or os.environ.get("TERM_PROGRAM") == "vscode":
            import subprocess

            subprocess.Popen(["code", str(out_path)])


if __name__ == "__main__":
    main()
