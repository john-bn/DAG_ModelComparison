import pytest
import xarray as xr

from DAG_ModelComparison.comparator.normalize import (
    MODEL_REGISTRY,
    VAR_REGISTRY,
    normalize_model_key,
    herbie_kwargs_for,
    normalize_var_key,
    pick_data_varname_from_ds,
    get_selector,
    get_xarray_kwargs,
    wrap_longitude,
    ensure_dataset,
)


def test_normalize_model_key_direct_hit():
    assert normalize_model_key("hrrr") == "hrrr"
    assert normalize_model_key(" HRRR ") == "hrrr"


def test_normalize_model_key_alias_hit():
    assert normalize_model_key("namnest") == "nam5k"
    assert normalize_model_key("NAM-12km") == "nam12k"
    assert normalize_model_key("ncar-arw") == "arw"
    assert normalize_model_key("rapid refresh") == "rap"


def test_normalize_model_key_invalid_raises():
    with pytest.raises(ValueError) as e:
        normalize_model_key("not-a-model")
    assert "Invalid NWP model selected" in str(e.value)


def test_herbie_kwargs_for_returns_copy_not_reference():
    kw = herbie_kwargs_for("hrrr")
    assert kw == MODEL_REGISTRY["hrrr"]["kwargs"]
    kw["model"] = "mutated"
    # Ensure registry is unchanged (we returned a new dict)
    assert MODEL_REGISTRY["hrrr"]["kwargs"]["model"] == "hrrr"


def test_normalize_var_key_direct_or_alias():
    assert normalize_var_key("TMP") == "TMP"
    assert normalize_var_key("tmp") == "TMP"
    assert normalize_var_key("temperature") == "TMP"
    assert normalize_var_key("dewpoint") == "DPT"


def test_normalize_var_key_invalid_raises():
    with pytest.raises(ValueError) as e:
        normalize_var_key("wind_speed")
    assert "Invalid analysis variable" in str(e.value)


def test_pick_data_varname_single_data_var():
    ds = xr.Dataset({"weird_name": (("x",), [1, 2, 3])})
    assert pick_data_varname_from_ds(ds, "TMP") == "weird_name"


def test_pick_data_varname_candidate_exact_match():
    # TMP candidates: ["t2m", "tmp2m", "temperature"]
    ds = xr.Dataset(
        {
            "something_else": (("x",), [0, 0, 0]),
            "t2m": (("x",), [1, 2, 3]),
        }
    )
    assert pick_data_varname_from_ds(ds, "TMP") == "t2m"


def test_pick_data_varname_candidate_substring_match():
    # candidate "t2m" should match "t2m_surface" in fallback substring loop
    ds = xr.Dataset(
        {
            "foo": (("x",), [0]),
            "t2m_surface": (("x",), [1]),
        }
    )
    assert pick_data_varname_from_ds(ds, "TMP") == "t2m_surface"


def test_pick_data_varname_no_match_raises():
    ds = xr.Dataset({"foo": (("x",), [1]), "bar": (("x",), [2])})
    with pytest.raises(ValueError) as e:
        pick_data_varname_from_ds(ds, "TMP")
    msg = str(e.value)
    assert "Could not determine data variable for TMP" in msg
    assert "data_vars" in msg


def test_var_registry_has_expected_fields():
    # This catches accidental edits that remove required pieces your app relies on.
    for k, entry in VAR_REGISTRY.items():
        assert "aliases" in entry
        assert "ds_candidates" in entry
        assert "title" in entry
        assert "cmap" in entry


def test_normalize_ifs_aliases():
    assert normalize_model_key("ifs") == "ifs"
    assert normalize_model_key("ecmwf") == "ifs"


def test_get_selector_ifs_uses_eccodes():
    assert get_selector("ifs", "TMP") == ":2t:"
    assert get_selector("ifs", "DPT") == ":2d:"


def test_get_selector_wgrib2_fallback():
    assert get_selector("hrrr", "TMP") == "TMP:2 m above"
    assert get_selector("gfs", "DPT") == "DPT:2 m above"


def test_d2m_in_dpt_candidates():
    assert "d2m" in VAR_REGISTRY["DPT"]["ds_candidates"]


def test_wrap_longitude_converts_0_360():
    ds = xr.Dataset({"longitude": (("x",), [0.0, 90.0, 180.0, 270.0])})
    result = wrap_longitude(ds)
    assert float(result["longitude"].max()) <= 180.0
    assert float(result["longitude"].min()) >= -180.0


def test_wrap_longitude_noop_for_negative180():
    ds = xr.Dataset({"longitude": (("x",), [-120.0, -90.0, 0.0, 50.0])})
    result = wrap_longitude(ds)
    assert list(result["longitude"].values) == [-120.0, -90.0, 0.0, 50.0]


def test_get_selector_href_uses_precise_regex():
    sel = get_selector("href", "TMP")
    assert "hour fcst" in sel
    assert r"\d+" in sel
    sel_dpt = get_selector("href", "DPT")
    assert "hour fcst" in sel_dpt


def test_href_selector_matches_forecast_not_max_min():
    import re
    sel = get_selector("href", "TMP")
    # Should match standard forecast entries
    assert re.search(sel, ":TMP:2 m above ground:6 hour fcst:")
    # Should NOT match max/min ensemble statistics
    assert not re.search(sel, ":TMP:2 m above ground:0-6 hour max fcst:")
    assert not re.search(sel, ":TMP:2 m above ground:0-6 hour min fcst:")


def test_ensure_dataset_returns_dataset():
    ds = xr.Dataset({"t2m": (("x",), [1, 2, 3])})
    assert ensure_dataset(ds) is ds


def test_ensure_dataset_returns_first_from_list():
    ds1 = xr.Dataset({"t2m": (("x",), [1, 2, 3])})
    ds2 = xr.Dataset({"t2m": (("x",), [4, 5, 6])})
    assert ensure_dataset([ds1, ds2]) is ds1


def test_ensure_dataset_picks_dataset_with_target_var():
    """When var_key is given, pick the dataset containing a candidate variable."""
    ds_other = xr.Dataset({"wind": (("x",), [1, 2, 3])})
    ds_temp = xr.Dataset({"t2m": (("x",), [4, 5, 6])})
    # t2m is a TMP candidate — should pick ds_temp even though it's second
    assert ensure_dataset([ds_other, ds_temp], var_key="TMP") is ds_temp


def test_ensure_dataset_falls_back_to_first_when_no_match():
    """If no dataset contains a candidate variable, fall back to the first."""
    ds1 = xr.Dataset({"foo": (("x",), [1])})
    ds2 = xr.Dataset({"bar": (("x",), [2])})
    assert ensure_dataset([ds1, ds2], var_key="TMP") is ds1


def test_ensure_dataset_substring_match_in_list():
    """Substring candidate matching works across multi-hypercube datasets."""
    ds_other = xr.Dataset({"wind": (("x",), [1])})
    ds_temp = xr.Dataset({"t2m_surface": (("x",), [2])})
    assert ensure_dataset([ds_other, ds_temp], var_key="TMP") is ds_temp


def test_get_xarray_kwargs_href_has_derived_forecast():
    kw = get_xarray_kwargs("href")
    assert "backend_kwargs" in kw
    assert "derivedForecast" in kw["backend_kwargs"]["read_keys"]


def test_get_xarray_kwargs_returns_empty_for_standard_models():
    assert get_xarray_kwargs("hrrr") == {}
    assert get_xarray_kwargs("gfs") == {}


def test_get_xarray_kwargs_returns_copy():
    kw1 = get_xarray_kwargs("href")
    kw2 = get_xarray_kwargs("href")
    assert kw1 == kw2
    assert kw1 is not kw2


def test_ds_candidates_include_eccodes_short_names():
    """TMP and DPT candidates should include eccodes short names for PDT 4.2."""
    tmp_cands = VAR_REGISTRY["TMP"]["ds_candidates"]
    assert "t" in tmp_cands
    assert "2t" in tmp_cands
    dpt_cands = VAR_REGISTRY["DPT"]["ds_candidates"]
    assert "d" in dpt_cands
    assert "2d" in dpt_cands
