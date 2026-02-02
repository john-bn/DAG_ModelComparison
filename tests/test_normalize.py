import pytest
import xarray as xr

from DAG_ModelComparison.comparator.normalize import (
    MODEL_REGISTRY,
    VAR_REGISTRY,
    normalize_model_key,
    herbie_kwargs_for,
    normalize_var_key,
    pick_data_varname_from_ds,
)


def test_normalize_model_key_direct_hit():
    assert normalize_model_key("hrrr") == "hrrr"
    assert normalize_model_key(" HRRR ") == "hrrr"


def test_normalize_model_key_alias_hit():
    assert normalize_model_key("namnest") == "nam5k"
    assert normalize_model_key("NAM-12km") == "nam12k"
    assert normalize_model_key("ncar-arw") == "arw"


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
