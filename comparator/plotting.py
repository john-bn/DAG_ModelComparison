import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import pandas as pd

def plot_tempdiff_map(lon: xr.DataArray, lat: xr.DataArray, tempdiff_f: xr.DataArray, valid_dt, cycle_dt, forecast):
    lon_wrapped = ((lon + 180) % 360) - 180
    T = tempdiff_f.where(np.isfinite(tempdiff_f))
    v = np.nanpercentile(np.abs(T.values), 95)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-95))
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6) # type: ignore
    ax.add_feature(cfeature.BORDERS, linewidth=0.6) # type: ignore
    ax.add_feature(cfeature.STATES, linewidth=0.4) # type: ignore

    pc = ccrs.PlateCarree()
    p = ax.pcolormesh(lon_wrapped, lat, T, transform=pc, cmap="coolwarm",
                      shading="nearest", vmin = -10, vmax = 10,  rasterized=True)
    cb = plt.colorbar(p, ax=ax, orientation="horizontal", pad=0.02, shrink=0.9)
    cb.set_label("ΔT (°F)")
    ax.set_title("HRRR − RTMA: 2 m Temperature Difference at "+ valid_dt.strftime("%Y-%m-%d %H:%M Z")+ "\nHRRR initialized "+ cycle_dt.strftime("%Y-%m-%d %H:%M Z")+ f" Forecast Hour: {forecast}")

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
