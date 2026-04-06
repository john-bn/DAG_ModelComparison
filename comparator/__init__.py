import os as _os
if not _os.environ.get("DISPLAY") and not _os.environ.get("WAYLAND_DISPLAY"):
    import matplotlib as _mpl
    _mpl.use("Agg")

from .fielddiff import compute_fielddiff
from .plotting import plot_tempdiff_map_with_table, plot_airports
from .util import major_airports_df
from .normalize import normalize_model_key, herbie_kwargs_for, normalize_var_key, pick_data_varname_from_ds, get_selector, get_xarray_kwargs, wrap_longitude, ensure_dataset, find_runs_for_valid_time