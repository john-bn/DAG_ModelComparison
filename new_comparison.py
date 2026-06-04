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

# --- Shared analysis state for GIF workers --------------------------------
# In GIF mode every frame validates against the SAME analysis time on the SAME
# model grid, so the regridded analysis is identical for all frames. We compute
# it once in the parent and hand it to each worker process via the pool
# initializer (set once per process rather than pickled per task).
_SHARED_ANL_ON_NWP = None
_SHARED_TGT_LON = None
_SHARED_TGT_LAT = None


def _init_worker(anl_on_nwp, tgt_lon, tgt_lat):
    """Pool initializer: stash the precomputed analysis in module globals."""
    global _SHARED_ANL_ON_NWP, _SHARED_TGT_LON, _SHARED_TGT_LAT
    _SHARED_ANL_ON_NWP = anl_on_nwp
    _SHARED_TGT_LON = tgt_lon
    _SHARED_TGT_LAT = tgt_lat


def generate_comparison_frame(
    model_key,
    var_key,
    cycle_dt,
    forecast_hour,
    verif_key="rtma",
    save_dir=DATA_DIR,
    out_dir=FIGURE_DIR,
):
    """Generate a single NWP-vs-analysis comparison plot and return the saved path.

    *verif_key* is the verification analysis source ("rtma" or "urma").
    Returns the Path to the saved PNG, or None if the frame could not be built.
    """
    verif_label = verif_key.upper()
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
        overwrite=False,
        **nwp_kwargs,
    )
    if not nwp:
        print(
            f"  Could not find {model_key.upper()} data for "
            f"{cycle_dt:%Y-%m-%d %H}Z F{forecast_hour:02d}. Skipping."
        )
        return None

    # --- Fetch analysis (RTMA/URMA) data ---
    anl_kwargs = norm.herbie_kwargs_for(verif_key)
    anl = Herbie(
        valid_dt,
        fxx=0,
        save_dir=str(save_dir),
        overwrite=False,
        **anl_kwargs,
    )
    if not anl:
        print(
            f"  Could not find {verif_label} data for {valid_dt:%Y-%m-%d %H}Z. Skipping."
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

    anl_selector = norm.get_selector(verif_key, var_key)
    try:
        ds_anl = norm.ensure_dataset(
            anl.xarray(anl_selector, remove_grib=True),
            var_key=var_key,
        )
    except Exception as e:
        print(f"  Failed to load {verif_label} GRIB data ({valid_dt:%Y-%m-%d %H}Z): {e}")
        return None

    # --- Variable resolution (derives wind speed from U/V when needed) ---
    try:
        nwp_field = norm.resolve_field_da(ds_nwp, var_key)
        anl_field = norm.resolve_field_da(ds_anl, var_key)
    except ValueError as e:
        print(f"  {e}")
        return None

    # --- Regrid analysis to model grid ---
    src_grid = {"lon": ds_anl["longitude"], "lat": ds_anl["latitude"]}
    tgt_grid = {"lon": ds_nwp["longitude"], "lat": ds_nwp["latitude"]}
    regridder = xe.Regridder(
        src_grid, tgt_grid, method="bilinear", periodic=False, reuse_weights=False
    )
    anl_on_nwp = regridder(anl_field)

    # --- Compute difference ---
    diff = fd.compute_fielddiff(nwp_field, anl_on_nwp, var_key)

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
        verif_name=verif_label,
    )

    plot.plot_airports(ax_map, util.major_airports_df())

    # --- Save (include init cycle in filename so each frame is unique) ---
    filename = (
        f"{display_name}_{verif_key}_{var_key}_"
        f"init{cycle_dt:%Y%m%d_%H}Z_F{forecast_hour:03d}_"
        f"valid{valid_dt:%Y%m%d_%H%MZ}.png"
    )
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved frame: {out_path}")
    return out_path


def precompute_analysis_on_model_grid(
    model_key,
    var_key,
    valid_dt,
    runs,
    verif_key="rtma",
    save_dir=DATA_DIR,
    weights_dir=DATA_DIR,
):
    """Fetch + load the analysis once and regrid it onto the model grid.

    Every frame in a GIF validates against the same *valid_dt* on the same model
    grid, so the regridded analysis is identical for all of them. We do that work
    here, in the parent, exactly once.

    *runs* is the list of (cycle_dt, fxx) pairs; any one of them yields the model
    target grid, so we try them in order until one loads.

    Returns (anl_on_nwp, tgt_lon, tgt_lat), or None if the analysis or every
    reference NWP file could not be loaded (caller should abort the GIF).
    """
    verif_label = verif_key.upper()

    # --- Fetch + load analysis once (keep GRIB on disk for re-runs) ---
    anl_kwargs = norm.herbie_kwargs_for(verif_key)
    anl = Herbie(
        valid_dt,
        fxx=0,
        save_dir=str(save_dir),
        overwrite=False,
        **anl_kwargs,
    )
    if not anl:
        print(f"  Could not find {verif_label} data for {valid_dt:%Y-%m-%d %H}Z.")
        return None

    anl_selector = norm.get_selector(verif_key, var_key)
    try:
        ds_anl = norm.ensure_dataset(
            anl.xarray(anl_selector, remove_grib=False),
            var_key=var_key,
        )
        anl_field = norm.resolve_field_da(ds_anl, var_key)
    except Exception as e:
        print(f"  Failed to load {verif_label} GRIB data ({valid_dt:%Y-%m-%d %H}Z): {e}")
        return None

    # --- Load ONE reference NWP file to obtain the model target grid ---
    nwp_kwargs = norm.herbie_kwargs_for(model_key)
    nwp_xr_kwargs = norm.get_xarray_kwargs(model_key)
    selector = norm.get_selector(model_key, var_key)

    ds_nwp = None
    for cycle_dt, fxx in runs:
        nwp = Herbie(
            cycle_dt,
            fxx=fxx,
            save_dir=str(save_dir),
            overwrite=False,
            **nwp_kwargs,
        )
        if not nwp:
            continue
        try:
            ds_nwp = norm.wrap_longitude(
                norm.ensure_dataset(
                    nwp.xarray(selector, remove_grib=False, **nwp_xr_kwargs),
                    var_key=var_key,
                )
            )
            break
        except Exception as e:
            print(
                f"  Reference grid load failed for {model_key.upper()} "
                f"{cycle_dt:%Y-%m-%d %H}Z F{fxx:03d}: {e}"
            )
            ds_nwp = None

    if ds_nwp is None:
        print(f"  Could not load any {model_key.upper()} reference file for the target grid.")
        return None

    # --- Build the regridder once (cache weights to disk) ---
    src_grid = {"lon": ds_anl["longitude"], "lat": ds_anl["latitude"]}
    tgt_grid = {"lon": ds_nwp["longitude"], "lat": ds_nwp["latitude"]}
    weights_path = Path(weights_dir) / f"weights_{verif_key}_to_{model_key}_bilinear.nc"
    try:
        regridder = xe.Regridder(
            src_grid, tgt_grid, method="bilinear", periodic=False,
            reuse_weights=weights_path.exists(), filename=str(weights_path),
        )
    except Exception as e:
        # Stale/mismatched weights file: rebuild from scratch.
        print(f"  Rebuilding regridder weights ({weights_path.name}): {e}")
        if weights_path.exists():
            weights_path.unlink()
        regridder = xe.Regridder(
            src_grid, tgt_grid, method="bilinear", periodic=False,
            reuse_weights=False, filename=str(weights_path),
        )

    # Materialize so the result pickles cleanly to worker processes
    # (no dask graph or open GRIB/netCDF file handle attached).
    anl_on_nwp = regridder(anl_field).compute()
    return anl_on_nwp, ds_nwp["longitude"], ds_nwp["latitude"]


def _render_frame_worker(
    model_key,
    var_key,
    cycle_dt,
    forecast_hour,
    verif_key="rtma",
    save_dir=DATA_DIR,
    out_dir=FIGURE_DIR,
):
    """GIF worker: render one frame against the shared precomputed analysis.

    Reads the regridded analysis and target grid from module globals set by
    *_init_worker*, so it only fetches/loads the per-frame NWP forecast.
    Returns the saved PNG Path, or None if the frame could not be built.
    """
    anl_on_nwp = _SHARED_ANL_ON_NWP
    tgt_lon = _SHARED_TGT_LON
    tgt_lat = _SHARED_TGT_LAT

    verif_label = verif_key.upper()
    var_meta = norm.VAR_REGISTRY[var_key]
    var_cmap = var_meta["cmap"]
    var_title = var_meta["title"]
    valid_dt = cycle_dt + timedelta(hours=forecast_hour)

    nwp_kwargs = norm.herbie_kwargs_for(model_key)
    selector = norm.get_selector(model_key, var_key)

    # --- Fetch NWP data (this frame's unique forecast) ---
    nwp = Herbie(
        cycle_dt,
        fxx=forecast_hour,
        save_dir=str(save_dir),
        overwrite=False,
        **nwp_kwargs,
    )
    if not nwp:
        print(
            f"  Could not find {model_key.upper()} data for "
            f"{cycle_dt:%Y-%m-%d %H}Z F{forecast_hour:02d}. Skipping."
        )
        return None

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

    try:
        nwp_field = norm.resolve_field_da(ds_nwp, var_key)
    except ValueError as e:
        print(f"  {e}")
        return None

    # --- Compute difference against the shared regridded analysis ---
    diff = fd.compute_fielddiff(nwp_field, anl_on_nwp, var_key)

    display_name = model_key

    fig, (ax_map, ax_tbl) = plot.plot_tempdiff_map_with_table(
        tgt_lon,
        tgt_lat,
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
        verif_name=verif_label,
    )

    plot.plot_airports(ax_map, util.major_airports_df())

    filename = (
        f"{display_name}_{verif_key}_{var_key}_"
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
        "Enter NWP model to compare against the analysis : "
        "HRRR, NAM5k, NAM12k, RAP, NBM, ARW, FV3, GFS, IFS, HREF: "
    ).strip()

    anl_var = input(
        "Enter analysis variable (TMP = 2m temperature, DPT = 2m dew point, "
        "VIS = visibility, WIND = 10m wind, GUST = wind gust): "
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

    # --- Select verification source (no default; re-prompt until valid) ---
    while True:
        verif_in = input("Verify against which analysis? (RTMA / URMA): ").strip()
        try:
            verif_key = norm.normalize_verif_key(verif_in)
            break
        except ValueError as e:
            print(e)
    verif_label = verif_key.upper()

    if animate == "y":
        # --- GIF mode: user provides the analysis time ---
        analysis_date = input(
            f"Enter the {verif_label} analysis date (YYYY-MM-DD): "
        ).strip()
        analysis_hour = int(
            input(f"Enter the {verif_label} analysis hour, in 24-hour Z-time: ")
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
            f"{verif_label} analysis {valid_dt:%Y-%m-%d %H}Z:"
        )
        for cycle, fxx in runs:
            print(f"  Init {cycle:%Y-%m-%d %H}Z  F{fxx:03d}")

        # Fetch + load + regrid the analysis ONCE (identical for every frame)
        print(f"\nPreparing {verif_label} analysis {valid_dt:%Y-%m-%d %H}Z ...")
        shared = precompute_analysis_on_model_grid(
            model_key, var_key, valid_dt, runs, verif_key
        )
        if shared is None:
            print(
                f"Could not prepare {verif_label} analysis for "
                f"{valid_dt:%Y-%m-%d %H}Z. Aborting GIF."
            )
            return
        anl_on_nwp, tgt_lon, tgt_lat = shared

        max_workers = min(os.cpu_count() or 4, len(runs), 8)
        print(
            f"\nGenerating {len(runs)} comparison frames "
            f"using {max_workers} parallel workers ..."
        )

        frame_results = {}  # cycle_dt -> path
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(anl_on_nwp, tgt_lon, tgt_lat),
        ) as executor:
            future_to_run = {}
            for cycle_dt, fxx in runs:
                future = executor.submit(
                    _render_frame_worker,
                    model_key,
                    var_key,
                    cycle_dt,
                    fxx,
                    verif_key,
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
            f"{model_key}_{verif_key}_{var_key}_"
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
            model_key, var_key, cycle_dt, forecast, verif_key
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
