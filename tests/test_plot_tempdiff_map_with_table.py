import numpy as np
import pandas as pd
import pytest
import xarray as xr
from datetime import datetime
from matplotlib.collections import QuadMesh
from DAG_ModelComparison.comparator.plotting import plot_tempdiff_map_with_table


@pytest.mark.mpl_image_compare(remove_text=True)  # optional if you later want image tests
def test_plot_tempdiff_map_with_table_norm_and_table_sort(tmp_path):
    # Tiny 2x3 grid with 1D lon/lat
    lon = xr.DataArray(np.array([-100, -99, -98], dtype=float), dims=("x",))
    lat = xr.DataArray(np.array([30, 31], dtype=float), dims=("y",))

    tempdiff_f = xr.DataArray(
        np.array([[ -3.0, 0.0, 2.0],
                  [ -1.0, 4.0, 6.0]], dtype=float),
        dims=("y", "x"),
        coords={"y": lat.values, "x": lon.values},
        name="tempdiff_f",
    )

    airports = pd.DataFrame(
        [
            ("KDEN", "Denver", 39.8617, -104.6731),  # out of this tiny grid, but OK
            ("KATL", "Atlanta", 33.6367, -84.4281),  # out of this tiny grid
            ("KORD", "Chicago", 41.9742, -87.9073),  # out of this tiny grid
        ],
        columns=["icao", "city", "lat", "lon"],
    )

    plot_meta = {"title": "ΔT", "cmap": "coolwarm", "vmin": -15, "vmax": 15, "vcenter": 0.0}
    fig, (ax_map, ax_tbl) = plot_tempdiff_map_with_table(
        lon=lon,
        lat=lat,
        tempdiff_f=tempdiff_f,
        valid_dt=datetime(2026, 2, 1, 0, 0),
        cycle_dt=datetime(2026, 2, 1, 0, 0),
        forecast="F00",
        model_name="hrrr",
        airports_df=airports,
        plot_meta=plot_meta,
        max_rows=20,
    )

    # --- Assert the pcolormesh on the map uses TwoSlopeNorm centered at zero ---
    meshes = [c for c in ax_map.collections if isinstance(c, QuadMesh)]
    assert meshes, "Expected a QuadMesh (pcolormesh) on ax_map"
    norm = meshes[-1].norm  # last is often safest if multiple
    assert norm is not None
    assert getattr(norm, "vcenter", None) == 0.0
    assert getattr(norm, "vmin", None) == -15
    assert getattr(norm, "vmax", None) == 15


    # --- Assert the table is alphabetically sorted by ICAO ---
    # Matplotlib table stores text objects; pull the first column (ICAO) rows 1..N (row 0 is header)
    table = None
    for child in ax_tbl.get_children():
        if child.__class__.__name__ == "Table":
            table = child
            break
    assert table is not None, "Expected a matplotlib Table on ax_tbl"

    # Extract ICAO column values (col 0), rows 1..3
    icaos = []
    for r in range(1, 1 + len(airports)):
        cell = table[(r, 0)]  # type: ignore
        icaos.append(cell.get_text().get_text())

    assert icaos == sorted(icaos)

    # Smoke save
    fig.savefig(tmp_path / "smoke.png")
