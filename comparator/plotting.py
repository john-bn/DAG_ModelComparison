import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes  # for Pylance typing help
import xarray as xr
import pandas as pd
from matplotlib.gridspec import GridSpec

# Fixed CONUS bounds in lon/lat
CONUS_LON_MIN, CONUS_LON_MAX = -125.0, -66.5
CONUS_LAT_MIN, CONUS_LAT_MAX = 20.0, 50.0
PC = ccrs.PlateCarree()


def _lock_conus_view(ax: GeoAxes):
    """
    Convert CONUS lon/lat corners to the current projection's x/y and lock x/y limits.
    Also disables autoscaling so artists can't change the view.
    """
    proj = ax.projection
    lons = np.array([CONUS_LON_MIN, CONUS_LON_MAX, CONUS_LON_MIN, CONUS_LON_MAX], dtype=float)
    lats = np.array([CONUS_LAT_MIN, CONUS_LAT_MIN, CONUS_LAT_MAX, CONUS_LAT_MAX], dtype=float)

    xy = proj.transform_points(PC, lons, lats)
    xs, ys = xy[:, 0], xy[:, 1]
    pad_x = 0.02 * (xs.max() - xs.min())
    pad_y = 0.02 * (ys.max() - ys.min())

    ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
    ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
    ax.set_autoscale_on(False)  # <- prevent later draws from changing limits


def plot_tempdiff_map(
    lon: xr.DataArray,
    lat: xr.DataArray,
    tempdiff_f: xr.DataArray,
    valid_dt,
    cycle_dt,
    forecast,
    model_name: str,
):
    """Original plotting routine (kept intact)."""
    lon_wrapped = ((lon + 180) % 360) - 180
    T = tempdiff_f.where(np.isfinite(tempdiff_f))

    fig = plt.figure(figsize=(10, 8))
    proj = ccrs.LambertConformal(central_longitude=-95, standard_parallels=(33, 45))
    ax: GeoAxes = plt.axes(projection=proj) # type: ignore

    _lock_conus_view(ax)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)  # type: ignore
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)    # type: ignore
    ax.add_feature(cfeature.STATES, linewidth=0.4)     # type: ignore

    p = ax.pcolormesh(
        lon_wrapped, lat, T,
        transform=PC, cmap="coolwarm",
        shading="nearest", vmin=-10, vmax=10, rasterized=True
    )

    cb = plt.colorbar(p, ax=ax, orientation="horizontal", pad=0.02, shrink=0.9)
    cb.set_label("ΔT (°F)")

    ax.set_title(
        f"{model_name.upper()} − RTMA: 2 m Temperature Difference\n"
        f"Valid: {valid_dt:%Y-%m-%d %H:%MZ} | Init: {cycle_dt:%Y-%m-%d %H:%MZ} | Forecast Hour: {forecast}",
        fontsize=11
    )

    _lock_conus_view(ax)
    plt.tight_layout()
    return fig, ax


def plot_airports(ax: GeoAxes, airports: pd.DataFrame):
    """Scatter and label airports on an existing axis."""
    a = airports[
        (airports.lon >= CONUS_LON_MIN) & (airports.lon <= CONUS_LON_MAX) &
        (airports.lat >= CONUS_LAT_MIN) & (airports.lat <= CONUS_LAT_MAX)
    ]
    ax.scatter(a.lon, a.lat, transform=PC, marker="^", s=30, zorder=6, color="black")
    for _, r in a.iterrows():
        ax.text(
            float(r.lon) + 0.3, float(r.lat) + 0.2, r.icao,
            transform=PC, fontsize=8, fontweight="bold", zorder=7, color="black",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )

# ---------------------------------------------------------------------
# NEW: Robust point sampling on 2-D or 1-D lat/lon grids for airport ΔT table
# ---------------------------------------------------------------------

# Always define cKDTree so it's not "possibly unbound" for type-checkers
try:
    from scipy.spatial import cKDTree as _cKDTree  # type: ignore
    cKDTree = _cKDTree  # rebind to a stable name
    _HAS_KDTREE = True
except Exception:
    cKDTree = None  # type: ignore[assignment]
    _HAS_KDTREE = False


def _wrap180(lon_vals: np.ndarray) -> np.ndarray:
    """Wrap longitudes to [-180, 180]."""
    return ((lon_vals.astype(float) + 180.0) % 360.0) - 180.0


def _as_float_array(a) -> np.ndarray:
    """Coerce xarray/pandas/np scalars/arrays to float64 ndarray."""
    return np.asarray(a, dtype=float)


def _to_2d_lonlat(lon_da: xr.DataArray, lat_da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return 2-D lon/lat arrays no matter if inputs are 1-D or 2-D.
    Assumes lon varies across columns (x), lat across rows (y) if 1-D.
    """
    lon_vals = np.asarray(lon_da.values)
    lat_vals = np.asarray(lat_da.values)
    if lon_vals.ndim == 1 and lat_vals.ndim == 1:
        LON2, LAT2 = np.meshgrid(lon_vals, lat_vals)
    else:
        LON2, LAT2 = lon_vals, lat_vals
    return _as_float_array(LON2), _as_float_array(LAT2)


def _nearest_values_on_geo_grid(
    lon_da: xr.DataArray,
    lat_da: xr.DataArray,
    da: xr.DataArray,
    pts_lon: np.ndarray,
    pts_lat: np.ndarray
) -> np.ndarray:
    """
    Return nearest-neighbor values from `da` for (pts_lon, pts_lat).
    Works for both 1-D and 2-D lon/lat grids.
    """
    LON2, LAT2 = _to_2d_lonlat(lon_da, lat_da)
    VAL = _as_float_array(da.values)

    # Broadcast VAL to same 2-D shape if needed (e.g., if it's DataArray with matching dims already this is no-op)
    if VAL.shape != LON2.shape:
        # Try to broadcast (common when VAL is (y, x) and lon/lat are (y, x))
        try:
            VAL = np.broadcast_to(VAL, LON2.shape)
        except Exception:
            # Last resort: ravel checks; if totally incompatible, bail to NaNs
            return np.full(len(pts_lon), np.nan, dtype=float)

    # Flatten valid cells
    mask = np.isfinite(LON2) & np.isfinite(LAT2) & np.isfinite(VAL)
    if not np.any(mask):
        return np.full(len(pts_lon), np.nan, dtype=float)

    lon_flat = _wrap180(LON2[mask].ravel())
    lat_flat = LAT2[mask].ravel()
    val_flat = VAL[mask].ravel()

    # Query points
    qlon = _wrap180(_as_float_array(pts_lon))
    qlat = _as_float_array(pts_lat)

    if _HAS_KDTREE and cKDTree is not None:
        tree = cKDTree(np.column_stack([lon_flat, lat_flat]))  # type: ignore[call-arg]
        _, idx = tree.query(np.column_stack([qlon, qlat]), k=1)  # type: ignore[call-arg]
        out = val_flat[idx]
    else:
        # NumPy fallback (fine for tens of airports)
        q = np.column_stack([qlon, qlat])[:, None, :]                  # (N,1,2)
        g = np.column_stack([lon_flat, lat_flat])[None, :, :]          # (1,M,2)
        idx = np.argmin(np.sum((q - g) ** 2, axis=2), axis=1)          # (N,)
        out = val_flat[idx]

    return out.astype(float)


def plot_tempdiff_map_with_table(
    lon: xr.DataArray,
    lat: xr.DataArray,
    tempdiff_f: xr.DataArray,
    valid_dt,
    cycle_dt,
    forecast,
    model_name: str,
    airports_df: pd.DataFrame,
    max_rows: int = 20,
):
    """Draw the CONUS map and add a ΔT table of selected airports."""

    # --- Compute ΔT at airport locations (robust for 2-D or 1-D lon/lat grids) ---
    airports_df = airports_df.copy()
    for col in ("lon", "lat"):
        airports_df[col] = pd.to_numeric(airports_df[col], errors="coerce")

    pts_lon = airports_df["lon"].to_numpy(dtype=float, copy=False)
    pts_lat = airports_df["lat"].to_numpy(dtype=float, copy=False)

    deltas = _nearest_values_on_geo_grid(lon, lat, tempdiff_f, pts_lon, pts_lat)

    # Build table in alphabetical order
    table_df = pd.DataFrame({
    "ICAO": airports_df["icao"].astype(str),
    "ΔT (°F)": deltas,
    })
    table_df["ΔT (°F)"] = pd.to_numeric(table_df["ΔT (°F)"], errors="coerce").round(1)
    table_df = (
        table_df.sort_values("ICAO", ascending=True, kind="mergesort")
        .head(max_rows)
        .reset_index(drop=True)
    )

    # --- Layout: map (left) + table (right) ---
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[3.3, 1.0])

    # Left: map
    proj = ccrs.LambertConformal(central_longitude=-95, standard_parallels=(33, 45))
    ax_map: GeoAxes = fig.add_subplot(gs[0, 0], projection=proj) # type: ignore

    _lock_conus_view(ax_map)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6)  # type: ignore
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.6)    # type: ignore
    ax_map.add_feature(cfeature.STATES, linewidth=0.4)     # type: ignore

    lon_wrapped = ((lon + 180) % 360) - 180
    T = tempdiff_f.where(np.isfinite(tempdiff_f))
    p = ax_map.pcolormesh(
        lon_wrapped, lat, T,
        transform=PC, cmap="coolwarm",
        shading="nearest", vmin=-10, vmax=10, rasterized=True
    )

    plt.colorbar(p, ax=ax_map, orientation="horizontal", pad=0.02, shrink=0.8, label="ΔT (°F)")
    ax_map.set_title(
        f"{model_name.upper()} − RTMA: 2 m Temperature Difference\n"
        f"Valid: {valid_dt:%Y-%m-%d %H:%MZ} | Init: {cycle_dt:%Y-%m-%d %H:%MZ} | Forecast Hour: {forecast}",
        fontsize=11
    )
    _lock_conus_view(ax_map)

    # Right: table
    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.axis("off")

    cell_text = table_df.values.tolist()
    col_labels = table_df.columns.tolist()
    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.2)
    ax_tbl.set_title("Airport ΔT (°F)", fontsize=11, pad=8)

    # Optional: tint ΔT column cells by sign (add small ignore to hush Pylance)
    try:
        col_idx = table_df.columns.get_loc("ΔT (°F)")
        for i in range(len(table_df)):
            val = table_df.iloc[i, col_idx]
            try:
                fval = float(val)
            except Exception:
                fval = np.nan

            if np.isnan(fval):
                color = (0.9, 0.9, 0.9, 1.0)  # grey
            elif fval >= 0:
                color = (1.0, 0.9, 0.9, 1.0)  # light red
            else:
                color = (0.9, 0.9, 1.0, 1.0)  # light blue

            tbl[(int(i) + 1, int(col_idx))].set_facecolor(color)  # type: ignore[index]
    except Exception:
        pass

    return fig, (ax_map, ax_tbl)
