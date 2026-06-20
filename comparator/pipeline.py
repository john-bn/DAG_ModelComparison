"""Comparison engine: NWP forecast vs RTMA/URMA analysis.

This is the reusable core, extracted from the former interactive
``new_comparison.py`` so it can be driven by the CLI (cron / on-demand) and by
the interactive shim alike. The science is unchanged from the original; the
differences are operational:

* directories are explicit arguments (no ``mkdir`` side effects at import time),
* progress/skip/error messages go through :mod:`logging` instead of ``print``,
* :func:`generate_comparison_frame` also returns summary error statistics
  (mean / RMSE / finite-point count) for the run manifest,
* the GIF orchestration formerly inlined in ``main()`` now lives in
  :func:`generate_gif`.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import logging
import os

from herbie.core import Herbie
import numpy as np
import xesmf as xe
import matplotlib.pyplot as plt

from comparator import fielddiff as fd
from comparator import plotting as plot
from comparator import util
from comparator import normalize as norm
from comparator.build_gif import create_gif

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """A rendered single-frame comparison and its summary error stats."""

    path: Path
    mean: float | None = None
    rmse: float | None = None
    n: int = 0


def _field_stats(diff):
    """Summarize a difference field (NWP - analysis) in its display units.

    Returns (mean, rmse, n_finite). When no finite points remain (all masked),
    returns (None, None, 0).
    """
    values = np.asarray(diff.values, dtype=float)
    finite = np.isfinite(values)
    n = int(finite.sum())
    if n == 0:
        return None, None, 0
    vals = values[finite]
    mean = float(np.mean(vals))
    rmse = float(np.sqrt(np.mean(np.square(vals))))
    return mean, rmse, n


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
    *,
    data_dir,
    out_dir,
):
    """Generate a single NWP-vs-analysis comparison plot.

    *verif_key* is the verification analysis source ("rtma" or "urma").
    Returns a :class:`FrameResult` (path + error stats), or None if the frame
    could not be built.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        save_dir=str(data_dir),
        overwrite=False,
        **nwp_kwargs,
    )
    if not nwp:
        logger.warning(
            "Could not find %s data for %sZ F%02d. Skipping.",
            model_key.upper(), f"{cycle_dt:%Y-%m-%d %H}", forecast_hour,
        )
        return None

    # --- Fetch analysis (RTMA/URMA) data ---
    anl_kwargs = norm.herbie_kwargs_for(verif_key)
    anl = Herbie(
        valid_dt,
        fxx=0,
        save_dir=str(data_dir),
        overwrite=False,
        **anl_kwargs,
    )
    if not anl:
        logger.warning(
            "Could not find %s data for %sZ. Skipping.",
            verif_label, f"{valid_dt:%Y-%m-%d %H}",
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
        logger.error("Failed to load %s GRIB data (F%02d): %s", model_key, forecast_hour, e)
        return None
    ds_nwp = norm.wrap_longitude(ds_nwp)

    anl_selector = norm.get_selector(verif_key, var_key)
    try:
        ds_anl = norm.ensure_dataset(
            anl.xarray(anl_selector, remove_grib=True),
            var_key=var_key,
        )
    except Exception as e:
        logger.error(
            "Failed to load %s GRIB data (%sZ): %s",
            verif_label, f"{valid_dt:%Y-%m-%d %H}", e,
        )
        return None

    # --- Variable resolution (derives wind speed from U/V when needed) ---
    try:
        nwp_field = norm.resolve_field_da(ds_nwp, var_key)
        anl_field = norm.resolve_field_da(ds_anl, var_key)
    except ValueError as e:
        logger.error("%s", e)
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
    mean, rmse, n = _field_stats(diff)

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
    logger.info("Saved frame: %s", out_path)
    return FrameResult(path=out_path, mean=mean, rmse=rmse, n=n)


def precompute_analysis_on_model_grid(
    model_key,
    var_key,
    valid_dt,
    runs,
    verif_key="rtma",
    *,
    save_dir,
    weights_dir,
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
    save_dir = Path(save_dir)
    weights_dir = Path(weights_dir)
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
        logger.warning("Could not find %s data for %sZ.", verif_label, f"{valid_dt:%Y-%m-%d %H}")
        return None

    anl_selector = norm.get_selector(verif_key, var_key)
    try:
        ds_anl = norm.ensure_dataset(
            anl.xarray(anl_selector, remove_grib=False),
            var_key=var_key,
        )
        anl_field = norm.resolve_field_da(ds_anl, var_key)
    except Exception as e:
        logger.error(
            "Failed to load %s GRIB data (%sZ): %s",
            verif_label, f"{valid_dt:%Y-%m-%d %H}", e,
        )
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
            logger.warning(
                "Reference grid load failed for %s %sZ F%03d: %s",
                model_key.upper(), f"{cycle_dt:%Y-%m-%d %H}", fxx, e,
            )
            ds_nwp = None

    if ds_nwp is None:
        logger.error(
            "Could not load any %s reference file for the target grid.",
            model_key.upper(),
        )
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
        logger.warning("Rebuilding regridder weights (%s): %s", weights_path.name, e)
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
    verif_key,
    save_dir,
    out_dir,
):
    """GIF worker: render one frame against the shared precomputed analysis.

    Reads the regridded analysis and target grid from module globals set by
    *_init_worker*, so it only fetches/loads the per-frame NWP forecast.
    Returns the saved PNG Path, or None if the frame could not be built.
    """
    save_dir = Path(save_dir)
    out_dir = Path(out_dir)
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
        logger.warning(
            "Could not find %s data for %sZ F%02d. Skipping.",
            model_key.upper(), f"{cycle_dt:%Y-%m-%d %H}", forecast_hour,
        )
        return None

    nwp_xr_kwargs = norm.get_xarray_kwargs(model_key)
    try:
        ds_nwp = norm.ensure_dataset(
            nwp.xarray(selector, remove_grib=True, **nwp_xr_kwargs),
            var_key=var_key,
        )
    except Exception as e:
        logger.error("Failed to load %s GRIB data (F%02d): %s", model_key, forecast_hour, e)
        return None
    ds_nwp = norm.wrap_longitude(ds_nwp)

    try:
        nwp_field = norm.resolve_field_da(ds_nwp, var_key)
    except ValueError as e:
        logger.error("%s", e)
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
    logger.info("Saved frame: %s", out_path)
    return out_path


def generate_gif(
    model_key,
    var_key,
    valid_dt,
    verif_key="rtma",
    *,
    data_dir,
    out_dir,
    duration=500,
    max_workers=None,
):
    """Build an animated GIF of every forecast covering *valid_dt*.

    Auto-discovers all (cycle, fxx) runs that verify against *valid_dt*, renders
    a frame per run in parallel (sharing one precomputed regridded analysis),
    and stitches them oldest-cycle-first. Returns the GIF Path, or None if no
    frames could be produced.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    verif_label = verif_key.upper()

    runs = norm.find_runs_for_valid_time(model_key, valid_dt)
    if not runs:
        logger.warning(
            "No %s init cycles cover %s analysis %sZ.",
            model_key.upper(), verif_label, f"{valid_dt:%Y-%m-%d %H}",
        )
        return None

    logger.info(
        "Found %d %s run(s) covering %s analysis %sZ.",
        len(runs), model_key.upper(), verif_label, f"{valid_dt:%Y-%m-%d %H}",
    )

    # Fetch + load + regrid the analysis ONCE (identical for every frame).
    logger.info("Preparing %s analysis %sZ ...", verif_label, f"{valid_dt:%Y-%m-%d %H}")
    shared = precompute_analysis_on_model_grid(
        model_key, var_key, valid_dt, runs, verif_key,
        save_dir=data_dir, weights_dir=data_dir,
    )
    if shared is None:
        logger.error(
            "Could not prepare %s analysis for %sZ. Aborting GIF.",
            verif_label, f"{valid_dt:%Y-%m-%d %H}",
        )
        return None
    anl_on_nwp, tgt_lon, tgt_lat = shared

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(runs), 8)
    logger.info(
        "Generating %d comparison frames using %d parallel workers ...",
        len(runs), max_workers,
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
                data_dir,
                out_dir,
            )
            future_to_run[future] = (cycle_dt, fxx)

        for future in as_completed(future_to_run):
            cycle_dt, fxx = future_to_run[future]
            try:
                path = future.result()
                if path is not None:
                    frame_results[cycle_dt] = path
                else:
                    logger.warning("Skipped: Init %sZ F%03d", f"{cycle_dt:%Y-%m-%d %H}", fxx)
            except Exception as e:
                logger.error("Failed: Init %sZ F%03d: %s", f"{cycle_dt:%Y-%m-%d %H}", fxx, e)

    # Preserve chronological order (oldest init first) for the GIF.
    frame_paths = [frame_results[dt] for dt, _ in runs if dt in frame_results]

    if not frame_paths:
        logger.warning("No frames were generated. Cannot create GIF.")
        return None

    gif_name = (
        f"{model_key}_{verif_key}_{var_key}_"
        f"valid{valid_dt:%Y%m%d_%H}Z_all_runs.gif"
    )
    gif_path = out_dir / gif_name
    create_gif(frame_paths, gif_path, duration=duration)
    logger.info("GIF saved to %s  (%d frames)", gif_path, len(frame_paths))
    return gif_path
