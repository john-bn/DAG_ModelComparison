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
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

FIGURE_DIR = Path("./figures")
FIGURE_DIR.mkdir(exist_ok=True)


def fetch_rtma(var_key, valid_dt, save_dir=DATA_DIR):
    """Download and decode the RTMA dataset for a given valid time.

    Returns (ds_rtma, rtma_varname) or (None, None) on failure.
    """
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
            f"  Could not find RTMA data for {valid_dt:%Y-%m-%d %H}Z."
        )
        return None, None

    rtma_selector = norm.get_selector("rtma", var_key)
    try:
        ds_rtma = norm.ensure_dataset(
            rtma.xarray(rtma_selector, remove_grib=True),
            var_key=var_key,
        )
    except Exception as e:
        print(f"  Failed to load RTMA GRIB data ({valid_dt:%Y-%m-%d %H}Z): {e}")
        return None, None

    try:
        rtma_varname = norm.pick_data_varname_from_ds(ds_rtma, var_key)
    except ValueError as e:
        print(f"  {e}")
        return None, None

    return ds_rtma, rtma_varname


def generate_comparison_frame(
    model_key,
    var_key,
    cycle_dt,
    forecast_hour,
    save_dir=DATA_DIR,
    out_dir=FIGURE_DIR,
    rtma_ds=None,
    rtma_varname=None,
):
    """Generate a single NWP-vs-RTMA comparison plot and return the saved path.

    When *rtma_ds* and *rtma_varname* are supplied the RTMA download is
    skipped entirely, reusing the pre-loaded dataset.

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

    # --- Load NWP fields ---
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

    # --- RTMA: reuse pre-loaded dataset or fetch on demand ---
    if rtma_ds is not None and rtma_varname is not None:
        ds_rtma = rtma_ds
    else:
        ds_rtma, rtma_varname = fetch_rtma(var_key, valid_dt, save_dir)
        if ds_rtma is None:
            return None

    # --- NWP variable name resolution ---
    try:
        nwp_varname = norm.pick_data_varname_from_ds(ds_nwp, var_key)
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

    # --- Save (include init cycle in filename so each frame is unique) ---
    filename = (
        f"{display_name}_rtma_{var_key}_"
        f"init{cycle_dt:%Y%m%d_%H}Z_F{forecast_hour:03d}_"
        f"valid{valid_dt:%Y%m%d_%H%MZ}.png"
    )
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

    if animate == "y":
        # --- GIF mode: user provides the RTMA analysis time ---
        analysis_date = input(
            "Enter the RTMA analysis date (YYYY-MM-DD): "
        ).strip()
        analysis_hour = int(
            input("Enter the RTMA analysis hour, in 24-hour Z-time: ")
        )
        valid_dt = datetime.fromisoformat(
            f"{analysis_date} {analysis_hour:02d}:00"
        )

        # Auto-discover every init cycle that covers this valid time
        runs = norm.find_runs_for_valid_time(model_key, valid_dt)
        if not runs:
            print(
                f"No {model_key.upper()} init cycles found whose forecast "
                f"range covers {valid_dt:%Y-%m-%d %H}Z."
            )
            return

        print(
            f"\nFound {len(runs)} {model_key.upper()} run(s) covering "
            f"RTMA analysis {valid_dt:%Y-%m-%d %H}Z:"
        )
        for cycle, fxx in runs:
            print(f"  Init {cycle:%Y-%m-%d %H}Z  F{fxx:03d}")

        # Pre-fetch RTMA once — every frame shares the same valid time
        print(f"\nDownloading RTMA analysis for {valid_dt:%Y-%m-%d %H}Z ...")
        rtma_ds, rtma_varname = fetch_rtma(var_key, valid_dt)
        if rtma_ds is None:
            print("RTMA download failed. Cannot generate frames.")
            return

        max_workers = min(os.cpu_count() or 4, len(runs), 8)
        print(
            f"Generating {len(runs)} comparison frames "
            f"using {max_workers} parallel workers ..."
        )

        frame_results = {}  # cycle_dt -> path
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_run = {}
            for cycle_dt, fxx in runs:
                future = executor.submit(
                    generate_comparison_frame,
                    model_key,
                    var_key,
                    cycle_dt,
                    fxx,
                    rtma_ds=rtma_ds,
                    rtma_varname=rtma_varname,
                )
                future_to_run[future] = (cycle_dt, fxx)

            for future in as_completed(future_to_run):
                cycle_dt, fxx = future_to_run[future]
                try:
                    path = future.result()
                    if path is not None:
                        frame_results[cycle_dt] = path
                    else:
                        print(
                            f"  Skipped: Init {cycle_dt:%Y-%m-%d %H}Z "
                            f"F{fxx:03d}"
                        )
                except Exception as e:
                    print(
                        f"  Failed:  Init {cycle_dt:%Y-%m-%d %H}Z "
                        f"F{fxx:03d}: {e}"
                    )

        # Preserve chronological order (oldest init first) for the GIF
        frame_paths = [
            frame_results[dt] for dt, _ in runs if dt in frame_results
        ]

        if not frame_paths:
            print("No frames were generated. Cannot create GIF.")
            return

        gif_name = (
            f"{model_key}_rtma_{var_key}_"
            f"valid{valid_dt:%Y%m%d_%H}Z_all_runs.gif"
        )
        gif_path = FIGURE_DIR / gif_name
        create_gif(frame_paths, gif_path, duration=500)
        print(f"\nGIF saved to {gif_path}  ({len(frame_paths)} frames)")

    else:
        # --- Single-frame mode ---
        date = input("Enter date (YYYY-MM-DD): ").strip()
        init_hour = int(
            input("Enter a valid initialization hour, in 24-hour Z-time: ")
        )
        forecast = int(
            input("Enter a valid forecast hour, in 24-hour Z-time: ")
        )
        cycle_dt = datetime.fromisoformat(f"{date} {init_hour:02d}:00")

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
