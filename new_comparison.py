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
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

FIGURE_DIR = Path("./figures")
FIGURE_DIR.mkdir(exist_ok=True)

# Maximum parallel Herbie download threads (matches FastHerbie default)
MAX_DOWNLOAD_THREADS = 5


# ---------------------------------------------------------------------------
# Low-level helpers: download one Herbie object
# ---------------------------------------------------------------------------

def _fetch_herbie(date, fxx, save_dir, **kwargs):
    """Create a Herbie object and download the GRIB file.

    Returns the Herbie object, or None if the file is unavailable.
    """
    h = Herbie(date, fxx=fxx, save_dir=str(save_dir), overwrite=True, **kwargs)
    if not h:
        return None
    h.download()
    return h


def _fetch_nwp(model_key, cycle_dt, fxx, selector, save_dir):
    """Download one NWP GRIB file.  Returns (cycle_dt, fxx, Herbie) or None."""
    kwargs = norm.herbie_kwargs_for(model_key)
    h = _fetch_herbie(cycle_dt, fxx, save_dir, **kwargs)
    if h is None:
        print(
            f"  [skip] {model_key.upper()} {cycle_dt:%Y-%m-%d %H}Z F{fxx:03d} "
            f"not available"
        )
        return None
    return (cycle_dt, fxx, h)


# ---------------------------------------------------------------------------
# Single-frame workflow (unchanged logic, cleaner structure)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# GIF workflow – parallel NWP downloads, single RTMA, cached regridder
# ---------------------------------------------------------------------------

def generate_gif_frames(
    model_key,
    var_key,
    runs,
    valid_dt,
    save_dir=DATA_DIR,
    out_dir=FIGURE_DIR,
    max_threads=MAX_DOWNLOAD_THREADS,
):
    """Generate comparison PNGs for every run, returning the list of frame paths.

    Optimisations over the sequential approach:
    • NWP GRIB files are downloaded in parallel via ThreadPoolExecutor.
    • RTMA is downloaded once and reused for every frame.
    • The xesmf regridder is built once and reused (same target grid).
    """
    var_meta = norm.VAR_REGISTRY[var_key]
    var_cmap = var_meta["cmap"]
    var_title = var_meta["title"]
    selector = norm.get_selector(model_key, var_key)
    nwp_xr_kwargs = norm.get_xarray_kwargs(model_key)
    rtma_selector = norm.get_selector("rtma", var_key)

    # ------------------------------------------------------------------
    # 1) Download RTMA once (single valid time, fxx=0)
    # ------------------------------------------------------------------
    print(f"\nDownloading RTMA for {valid_dt:%Y-%m-%d %H}Z ...")
    rtma_kwargs = norm.herbie_kwargs_for("rtma")
    rtma = Herbie(
        valid_dt, fxx=0, save_dir=str(save_dir), overwrite=True, **rtma_kwargs
    )
    if not rtma:
        print(
            f"Could not find RTMA data for {valid_dt:%Y-%m-%d %H}Z. Aborting."
        )
        return []

    try:
        ds_rtma = norm.ensure_dataset(
            rtma.xarray(rtma_selector, remove_grib=True), var_key=var_key
        )
        rtma_varname = norm.pick_data_varname_from_ds(ds_rtma, var_key)
    except Exception as e:
        print(f"Failed to load RTMA data: {e}")
        return []

    rtma_src_grid = {
        "lon": ds_rtma["longitude"],
        "lat": ds_rtma["latitude"],
    }

    # ------------------------------------------------------------------
    # 2) Download all NWP GRIB files in parallel
    # ------------------------------------------------------------------
    n = len(runs)
    print(f"Downloading {n} NWP files in parallel (max {max_threads} threads) ...")

    # Submit downloads
    nwp_results = {}  # (cycle_dt, fxx) → Herbie
    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        future_to_run = {
            pool.submit(
                _fetch_nwp, model_key, cycle_dt, fxx, selector, save_dir
            ): (cycle_dt, fxx)
            for cycle_dt, fxx in runs
        }
        for future in as_completed(future_to_run):
            result = future.result()
            if result is not None:
                cyc, fxx, herbie_obj = result
                nwp_results[(cyc, fxx)] = herbie_obj

    downloaded = len(nwp_results)
    print(f"Downloaded {downloaded}/{n} NWP files successfully.\n")

    if not nwp_results:
        return []

    # ------------------------------------------------------------------
    # 3) Process each frame (load → regrid → diff → plot)
    #    Regridder is cached per unique NWP grid shape.
    # ------------------------------------------------------------------
    regridder_cache = {}
    frame_paths = []

    for cycle_dt, fxx in runs:
        herbie_obj = nwp_results.get((cycle_dt, fxx))
        if herbie_obj is None:
            continue

        label = f"{model_key.upper()} init {cycle_dt:%Y-%m-%d %H}Z F{fxx:03d}"
        try:
            ds_nwp = norm.ensure_dataset(
                herbie_obj.xarray(selector, remove_grib=True, **nwp_xr_kwargs),
                var_key=var_key,
            )
        except Exception as e:
            print(f"  [{label}] Failed to load GRIB: {e}")
            continue
        ds_nwp = norm.wrap_longitude(ds_nwp)

        try:
            nwp_varname = norm.pick_data_varname_from_ds(ds_nwp, var_key)
        except ValueError as e:
            print(f"  [{label}] {e}")
            continue

        # Build or reuse regridder (keyed on target grid shape)
        grid_key = (
            ds_nwp["longitude"].shape,
            ds_nwp["latitude"].shape,
        )
        if grid_key not in regridder_cache:
            tgt_grid = {
                "lon": ds_nwp["longitude"],
                "lat": ds_nwp["latitude"],
            }
            regridder_cache[grid_key] = xe.Regridder(
                rtma_src_grid,
                tgt_grid,
                method="bilinear",
                periodic=False,
                reuse_weights=False,
            )
        regridder = regridder_cache[grid_key]
        rtma_on_nwp = regridder(ds_rtma[rtma_varname])

        # Compute difference
        diff = fd.compute_fielddiff(ds_nwp[nwp_varname], rtma_on_nwp)

        # Plot
        display_name = model_key
        fig, (ax_map, ax_tbl) = plot.plot_tempdiff_map_with_table(
            ds_nwp["longitude"],
            ds_nwp["latitude"],
            diff,
            valid_dt,
            cycle_dt,
            fxx,
            display_name,
            util.major_airports_df(),
            max_rows=20,
            var_title=var_title,
            var_cmap=var_cmap,
            plot_meta=var_meta,
        )
        plot.plot_airports(ax_map, util.major_airports_df())

        filename = (
            f"{display_name}_rtma_{var_key}_"
            f"init{cycle_dt:%Y%m%d_%H}Z_F{fxx:03d}_"
            f"valid{valid_dt:%Y%m%d_%H%MZ}.png"
        )
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved frame: {out_path}")
        frame_paths.append(out_path)

    return frame_paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

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

        # Parallel download + sequential processing
        frame_paths = generate_gif_frames(
            model_key, var_key, runs, valid_dt
        )

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
