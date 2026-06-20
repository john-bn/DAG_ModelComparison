from pathlib import Path

import pytest

from comparator import appconfig

DAG_ENV_VARS = [
    "DAG_CONFIG", "DAG_DATA_DIR", "DAG_OUT_DIR", "DAG_LOG_DIR",
    "DAG_VERIF", "DAG_LAG_HOURS", "DAG_DEFAULT_LEAD",
]


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Run in a clean tmp cwd with no DAG_* env vars or ambient config.yaml."""
    for var in DAG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_builtin_defaults(isolated_env):
    cfg = appconfig.load_config()
    assert cfg.data_dir == (isolated_env / "data").resolve()
    assert cfg.out_dir == (isolated_env / "figures").resolve()
    assert cfg.verif == "rtma"
    assert cfg.lag_hours == 2
    assert cfg.default_lead == 24
    assert cfg.manifest_path == cfg.out_dir / "runs.jsonl"


def test_file_values_used(isolated_env):
    cfg_file = isolated_env / "config.yaml"
    cfg_file.write_text(
        "data_dir: /srv/data\n"
        "out_dir: /srv/figs\n"
        "log_dir: /srv/logs\n"
        "verif: urma\n"
        "rolling:\n"
        "  lag_hours: 4\n"
        "  default_lead: 12\n"
    )
    cfg = appconfig.load_config(config_path=str(cfg_file))
    assert cfg.data_dir == Path("/srv/data")
    assert cfg.out_dir == Path("/srv/figs")
    assert cfg.verif == "urma"
    assert cfg.lag_hours == 4
    assert cfg.default_lead == 12


def test_env_overrides_file(isolated_env, monkeypatch):
    cfg_file = isolated_env / "config.yaml"
    cfg_file.write_text("data_dir: /srv/data\nverif: urma\n")
    monkeypatch.setenv("DAG_DATA_DIR", "/env/data")
    monkeypatch.setenv("DAG_VERIF", "rtma")
    monkeypatch.setenv("DAG_LAG_HOURS", "6")
    cfg = appconfig.load_config(config_path=str(cfg_file))
    assert cfg.data_dir == Path("/env/data")
    assert cfg.verif == "rtma"
    assert cfg.lag_hours == 6


def test_cli_overrides_env(isolated_env, monkeypatch):
    monkeypatch.setenv("DAG_DATA_DIR", "/env/data")
    cfg = appconfig.load_config(data_dir="/cli/data", lag_hours=9, default_lead=36)
    assert cfg.data_dir == Path("/cli/data")
    assert cfg.lag_hours == 9
    assert cfg.default_lead == 36


def test_relative_paths_made_absolute(isolated_env):
    cfg = appconfig.load_config(data_dir="mydata")
    assert cfg.data_dir.is_absolute()
    assert cfg.data_dir == (isolated_env / "mydata").resolve()
