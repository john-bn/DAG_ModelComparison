import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import pandas as pd

def plot_tempdiff_map(lon: xr.DataArray, lat: xr.DataArray, tempdiff_f: xr.DataArray,
                      title: str = "HRRR − RTMA: 2 m Temperature Difference (°F)"):
    """Render a Cartopy map of temp difference on a Lambert Conformal projection."""
    lon_wrapped = ((lon + 180) % 360) - 180
    T = tempdiff_f.where(np.isfinite(tempdiff_f))
    vmin, vmax = -20, 20

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-95))
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    ax.add_feature(cfeature.STATES, linewidth=0.4)

    pc = ccrs.PlateCarree()
    p = ax.pcolormesh(lon_wrapped, lat, T, transform=pc, cmap="coolwarm",
                      shading="nearest", vmin=vmin, vmax=vmax, rasterized=True)
    cb = plt.colorbar(p, ax=ax, orientation="horizontal", pad=0.02, shrink=0.9)
    cb.set_label("ΔT (°F)")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax

def plot_airports(ax, airports: pd.DataFrame):
    """Overlay airports (icao, lat, lon) onto an existing Cartopy axis."""
    pc = ccrs.PlateCarree()
    ax.scatter(airports.lon, airports.lat, transform=pc, marker="^", s=30, zorder=6, color="black")
    for _, r in airports.iterrows():
        ax.text(r.lon + 0.35, r.lat + 0.25, r.icao,
                transform=pc, fontsize=8, fontweight="bold", zorder=7, color="black",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
