import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from comparator.plotting import plot_airports
from comparator.utils import major_airports_df

def test_airports_within_extent():
    """All major airports should fall within a standard CONUS map extent."""
    proj = ccrs.LambertConformal(central_longitude=-95)
    pc = ccrs.PlateCarree()
    fig, ax = plt.subplots(subplot_kw={"projection": proj}, figsize=(6, 5))

    airports = major_airports_df()
    plot_airports(ax, airports)

    # Default extent may be global; set a reasonable CONUS extent first
    ax.set_extent([-130, -65, 23, 50], pc)
    x0, x1, y0, y1 = ax.get_extent(pc)

    within = (
        (airports.lon >= x0) & (airports.lon <= x1) &
        (airports.lat >= y0) & (airports.lat <= y1)
    )
    assert within.all(), "Some airports fall outside the map extent."

def test_projection_roundtrip_accuracy():
    """Transform lon/lat -> map coords -> lon/lat roundtrip should be precise."""
    proj = ccrs.LambertConformal(central_longitude=-95)
    pc = ccrs.PlateCarree()
    airports = major_airports_df()

    pts = proj.transform_points(pc, airports.lon.values, airports.lat.values)
    lon2, lat2 = pc.transform_points(proj, pts[:,0], pts[:,1])[:, :2].T

    err_lon = np.abs(lon2 - airports.lon.values)
    err_lat = np.abs(lat2 - airports.lat.values)
    assert err_lon.max() < 1e-6 and err_lat.max() < 1e-6, "Projection roundtrip error too large."
