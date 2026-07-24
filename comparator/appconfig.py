"""Application configuration: directories, verification source, rolling defaults.

This finally wires up the previously-aspirational ``config.yaml``. Settings are
resolved with the precedence (highest first):

    CLI flag  >  environment variable  >  config.yaml  >  built-in default

Every directory is resolved to an **absolute** path. This matters under cron:
its working directory is ``$HOME``, so a relative ``./data`` would otherwise
land in the wrong place. (The cron wrapper also ``cd``s into the project dir,
but absolute paths make the behavior robust regardless.)

Environment variables:
    DAG_CONFIG       path to the YAML config file
    DAG_DATA_DIR     GRIB/cache directory
    DAG_OUT_DIR      figure/manifest output directory
    DAG_LOG_DIR      log directory
    DAG_VERIF        default verification source (rtma|urma)
    DAG_LAG_HOURS    rolling data-latency buffer (hours)
    DAG_DEFAULT_LEAD rolling default forecast lead (hours)
    DAG_GIF_WORKERS  GIF render parallelism (1 = sequential, low memory)
"""

from dataclasses import dataclass
from pathlib import Path
import os

from comparator import timesel

DEFAULT_VERIF = "rtma"
DEFAULT_CONFIG_NAME = "config.yaml"
DEFAULT_GIF_WORKERS = 1


@dataclass(frozen=True)
class Config:
    """Resolved, absolute-path application settings."""

    data_dir: Path
    out_dir: Path
    log_dir: Path
    verif: str
    lag_hours: int
    default_lead: int
    gif_workers: int

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / "runs.jsonl"

    @property
    def log_path(self) -> Path:
        return self.log_dir / "comparison.log"


def _load_yaml(path: Path) -> dict:
    """Read a YAML mapping from *path*, or {} if missing/empty."""
    if not path or not path.exists():
        return {}
    import yaml  # local import so non-config code paths don't need pyyaml

    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")
    return data


def _find_config_path(cli_config: str | None) -> Path | None:
    """Resolve which config file to read (CLI > env > ./config.yaml)."""
    candidate = cli_config or os.environ.get("DAG_CONFIG")
    if candidate:
        return Path(candidate).expanduser()
    default = Path.cwd() / DEFAULT_CONFIG_NAME
    return default if default.exists() else None


def _pick(*values, default=None):
    """First non-None value, else *default*."""
    for v in values:
        if v is not None:
            return v
    return default


def _abspath(value) -> Path:
    """Expand ~ and resolve to an absolute path (against CWD if relative)."""
    return Path(value).expanduser().resolve()


def _int_or_none(value):
    return None if value is None else int(value)


def load_config(
    *,
    config_path: str | None = None,
    data_dir: str | None = None,
    out_dir: str | None = None,
    log_dir: str | None = None,
    verif: str | None = None,
    lag_hours: int | None = None,
    default_lead: int | None = None,
    gif_workers: int | None = None,
) -> Config:
    """Resolve a :class:`Config`, applying CLI > env > file > default.

    All keyword args are CLI-level overrides (highest precedence); pass the
    parsed argparse values straight through (``None`` where unset).
    """
    file_cfg = _load_yaml(_find_config_path(config_path))
    rolling_cfg = file_cfg.get("rolling", {}) or {}

    env = os.environ.get

    data = _pick(data_dir, env("DAG_DATA_DIR"), file_cfg.get("data_dir"), default="data")
    out = _pick(out_dir, env("DAG_OUT_DIR"), file_cfg.get("out_dir"), default="figures")
    logs = _pick(log_dir, env("DAG_LOG_DIR"), file_cfg.get("log_dir"), default="logs")

    verification = _pick(
        verif, env("DAG_VERIF"), file_cfg.get("verif"), default=DEFAULT_VERIF
    )

    lag = _pick(
        _int_or_none(lag_hours),
        _int_or_none(env("DAG_LAG_HOURS")),
        rolling_cfg.get("lag_hours"),
        default=timesel.DEFAULT_LAG_HOURS,
    )
    lead = _pick(
        _int_or_none(default_lead),
        _int_or_none(env("DAG_DEFAULT_LEAD")),
        rolling_cfg.get("default_lead"),
        default=timesel.DEFAULT_LEAD_HOURS,
    )
    workers = _pick(
        _int_or_none(gif_workers),
        _int_or_none(env("DAG_GIF_WORKERS")),
        file_cfg.get("gif_workers"),
        default=DEFAULT_GIF_WORKERS,
    )

    return Config(
        data_dir=_abspath(data),
        out_dir=_abspath(out),
        log_dir=_abspath(logs),
        verif=str(verification).strip().lower(),
        lag_hours=int(lag),
        default_lead=int(lead),
        gif_workers=max(1, int(workers)),
    )
