import pytest

from DAG_ModelComparison.comparator.normalize import (
    MODEL_REGISTRY,
    normalize_model_key,
    herbie_kwargs_for,
)

# Define what "kwargs contract" you require for every model.
# You can expand this over time.
REQUIRED_KWARG_KEYS = {"model", "product"}


@pytest.mark.parametrize("model_key, entry", MODEL_REGISTRY.items())
def test_model_registry_entry_has_expected_shape(model_key, entry):
    assert isinstance(model_key, str) and model_key

    # aliases
    assert "aliases" in entry
    assert isinstance(entry["aliases"], list)
    assert all(isinstance(a, str) and a for a in entry["aliases"])

    # kwargs
    assert "kwargs" in entry
    assert isinstance(entry["kwargs"], dict)
    assert REQUIRED_KWARG_KEYS.issubset(entry["kwargs"].keys())

    # kwargs values should be simple JSON-like values (strings/ints)
    for k, v in entry["kwargs"].items():
        assert isinstance(k, str) and k
        assert isinstance(v, (str, int, float, bool)), f"{model_key} kwargs[{k}] has unsupported type: {type(v)}"


@pytest.mark.parametrize("model_key", list(MODEL_REGISTRY.keys()))
def test_herbie_kwargs_for_returns_copy(model_key):
    kw1 = herbie_kwargs_for(model_key)
    kw2 = herbie_kwargs_for(model_key)
    assert kw1 == kw2
    assert kw1 is not kw2  # ensure it's a copy, not the same dict


def test_all_aliases_normalize_to_their_model_key():
    # For each alias, normalization should return that model's canonical key.
    for model_key, entry in MODEL_REGISTRY.items():
        for alias in entry.get("aliases", []):
            assert normalize_model_key(alias) == model_key


def test_no_duplicate_aliases_across_models():
    # Duplicate aliases cause ambiguous normalization.
    seen = {}
    for model_key, entry in MODEL_REGISTRY.items():
        for alias in entry.get("aliases", []):
            a = alias.strip().lower()
            if a in seen:
                raise AssertionError(f"Alias '{alias}' is used for BOTH '{seen[a]}' and '{model_key}'")
            seen[a] = model_key
