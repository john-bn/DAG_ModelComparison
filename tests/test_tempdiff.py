import numpy as np
import pandas as pd
import xarray as xr

from comparator.tempdiff import compute_tempdiff_f
from comparator.util import major_airports_df


def test_tempdiff_uses_airport_locations_and_units():
    """Compute difference on airport grid and ensure Fahrenheit conversion."""
    airports = major_airports_df()
    times = pd.date_range("2024-01-01", periods=2, freq="h")

    # Use airport lat/lon as coordinates to ensure spatial alignment is honored
    coords = {
        "airport": airports.icao.values,
        "time": times,
        "lat": ("airport", airports.lat.values),
        "lon": ("airport", airports.lon.values),
    }

    # HRRR is 5 K warmer than RTMA everywhere except one intentionally invalid point
    hrrr = xr.DataArray(
        np.full((len(airports), len(times)), 300.0),
        coords=coords,
        dims=("airport", "time"),
    )
    rtma = xr.DataArray(
        np.full((len(airports), len(times)), 295.0),
        coords=coords,
        dims=("airport", "time"),
    )

    # Make a single airport/time invalid to ensure masking occurs
    hrrr = hrrr.assign_coords()
    hrrr = hrrr.copy()
    hrrr[0, 0] = 100.0

    diff_f = compute_tempdiff_f(hrrr, rtma)

    # Valid locations should show a 9 F (5 C) warm HRRR bias
    expected = xr.DataArray(
        np.full((len(airports), len(times)), 9.0),
        coords=coords,
        dims=("airport", "time"),
    )
    expected[0, 0] = np.nan

    xr.testing.assert_allclose(diff_f, expected)

    # Airport coordinates should be preserved so locations remain traceable
    np.testing.assert_array_equal(diff_f.lat, airports.lat.values)
    np.testing.assert_array_equal(diff_f.lon, airports.lon.values)