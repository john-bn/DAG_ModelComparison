import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes  # for Pylance typing help
import xarray as xr
import pandas as pd
import numpy as np

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
    # corners of the rectangle in lon/lat
    lons = np.array([CONUS_LON_MIN, CONUS_LON_MAX, CONUS_LON_MIN, CONUS_LON_MAX])
    lats = np.array([CONUS_LAT_MIN, CONUS_LAT_MIN, CONUS_LAT_MAX, CONUS_LAT_MAX])

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
    # Wrap longitude to [-180, 180] for consistent plotting
    lon_wrapped = ((lon + 180) % 360) - 180
    T = tempdiff_f.where(np.isfinite(tempdiff_f))

    fig = plt.figure(figsize=(10, 8))
    proj = ccrs.LambertConformal(central_longitude=-95, standard_parallels=(33, 45))
    ax: GeoAxes = plt.axes(projection=proj)

    # Lock the view BEFORE plotting anything
    _lock_conus_view(ax)

    # Map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)  # type: ignore
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)    # type: ignore
    ax.add_feature(cfeature.STATES, linewidth=0.4)     # type: ignore

    # Plot data (this would normally trigger autoscale, but we've disabled it)
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

    # As a belt-and-suspenders, re-apply the lock after plotting too
    _lock_conus_view(ax)

    plt.tight_layout()
    return fig, ax

def plot_airports(ax: GeoAxes, airports: pd.DataFrame):
    # Clip to CONUS box so labels don’t spill outside
    a = airports[
        (airports.lon >= CONUS_LON_MIN) & (airports.lon <= CONUS_LON_MAX) &
        (airports.lat >= CONUS_LAT_MIN) & (airports.lat <= CONUS_LAT_MAX)
    ]
    ax.scatter(a.lon, a.lat, transform=PC, marker="^", s=30, zorder=6, color="black")
    for _, r in a.iterrows():
        ax.text(
            r.lon + 0.3, r.lat + 0.2, r.icao,
            transform=PC, fontsize=8, fontweight="bold", zorder=7, color="black",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )
