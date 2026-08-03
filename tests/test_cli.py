import pytest

from comparator import cli

DAG_ENV_VARS = [
    "DAG_CONFIG", "DAG_DATA_DIR", "DAG_OUT_DIR", "DAG_LOG_DIR",
    "DAG_VERIF", "DAG_LAG_HOURS", "DAG_DEFAULT_LEAD", "DAG_GIF_WORKERS",
]


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    for var in DAG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- argument parsing -------------------------------------------------------
def test_parse_run_defaults():
    args = cli.parse_args(["run", "--model", "hrrr", "--var", "TMP"])
    assert args.command == "run"
    assert args.model == "hrrr"
    assert args.var == "TMP"
    assert args.mode == "single"
    assert args.dry_run is False


def test_parse_requires_model_and_var():
    with pytest.raises(SystemExit):
        cli.parse_args(["run", "--model", "hrrr"])


def test_parse_workers_flag():
    args = cli.parse_args(["run", "--model", "hrrr", "--var", "TMP", "--workers", "3"])
    assert args.workers == 3
    # Unset -> None so config/default wins.
    args = cli.parse_args(["run", "--model", "hrrr", "--var", "TMP"])
    assert args.workers is None


# --- run --dry-run ----------------------------------------------------------
def test_dry_run_rolling_writes_nothing(isolated_env, capsys):
    rc = cli.main(["run", "--model", "hrrr", "--var", "TMP", "--dry-run"])
    assert rc == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "hrrr" in out and "valid" in out
    # No working directories should have been created.
    for d in ("data", "figures", "logs"):
        assert not (isolated_env / d).exists()


def test_dry_run_explicit_target(capsys):
    rc = cli.main([
        "run", "--model", "hrrr", "--var", "TMP",
        "--date", "2026-06-18", "--init", "12", "--fxx", "24", "--dry-run",
    ])
    assert rc == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "2026-06-18 12Z" in out      # cycle
    assert "F024" in out
    assert "2026-06-19 12Z" in out      # valid = cycle + 24h


def test_invalid_model_is_usage_error(capsys):
    rc = cli.main(["run", "--model", "bogus", "--var", "TMP", "--dry-run"])
    assert rc == cli.EXIT_USAGE
    assert "error" in capsys.readouterr().err.lower()


def test_explicit_date_without_init_is_usage_error():
    rc = cli.main(["run", "--model", "hrrr", "--var", "TMP",
                   "--date", "2026-06-18", "--dry-run"])
    assert rc == cli.EXIT_USAGE


# --- list / latest on an empty manifest ------------------------------------
def test_list_empty(isolated_env, capsys):
    rc = cli.main(["list", "--out-dir", str(isolated_env)])
    assert rc == cli.EXIT_OK
    assert "No matching runs." in capsys.readouterr().err


def test_latest_empty_is_error(isolated_env, capsys):
    rc = cli.main(["latest", "--model", "hrrr", "--out-dir", str(isolated_env)])
    assert rc == cli.EXIT_ERROR
    assert "No matching runs." in capsys.readouterr().err


def test_list_reads_manifest(isolated_env, capsys):
    from comparator import manifest
    out_dir = isolated_env / "figures"
    out_dir.mkdir()
    manifest.append(out_dir / "runs.jsonl", manifest.make_record(
        ts_utc="2026-06-18T13:00:00", model="hrrr", var="TMP", verif="rtma",
        mode="single", path=str(out_dir / "hrrr_TMP.png"),
        valid_dt="2026-06-18T12:00:00", fxx=24, mean=1.23, rmse=2.34, n=100,
    ))
    rc = cli.main(["list", "--out-dir", str(out_dir)])
    assert rc == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "hrrr" in out and "F024" in out and "hrrr_TMP.png" in out
