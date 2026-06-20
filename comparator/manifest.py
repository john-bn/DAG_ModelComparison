"""Append-only JSONL manifest of generated comparison outputs.

Each produced PNG/GIF appends one JSON record (one line) to
``<out_dir>/runs.jsonl``. The ``compare list`` / ``compare latest`` commands
read it back to let you browse what the cron job produced — filterable by
model/variable/verif/date — without re-deriving anything from filenames.

JSONL (not a database) keeps this dependency-free and append-safe across
concurrent cron lines: each record is written with a single ``write`` of one
newline-terminated line, which POSIX appends atomically.

Datetimes are stored as ISO-8601 strings (``YYYY-MM-DDTHH:MM:SS``).
"""

from datetime import datetime
from pathlib import Path
import json

# Field order used for both records and the `list` table.
FIELDS = (
    "ts_utc",     # when this record was written (UTC)
    "model",
    "var",
    "verif",
    "mode",       # "single" | "gif"
    "cycle_dt",   # forecast init cycle (ISO)
    "fxx",        # forecast lead hour
    "valid_dt",   # analysis/valid time (ISO)
    "path",       # output file
    "mean",       # mean signed error (NWP - analysis), in display units
    "rmse",       # root-mean-square error, display units
    "n",          # count of finite points
)


def _iso(dt) -> str | None:
    """ISO-format a datetime; pass through strings/None unchanged."""
    if dt is None or isinstance(dt, str):
        return dt
    return dt.replace(microsecond=0).isoformat()


def make_record(
    *,
    ts_utc,
    model,
    var,
    verif,
    mode,
    path,
    cycle_dt=None,
    fxx=None,
    valid_dt=None,
    mean=None,
    rmse=None,
    n=None,
) -> dict:
    """Assemble a manifest record (datetimes ISO-formatted)."""
    return {
        "ts_utc": _iso(ts_utc),
        "model": model,
        "var": var,
        "verif": verif,
        "mode": mode,
        "cycle_dt": _iso(cycle_dt),
        "fxx": fxx,
        "valid_dt": _iso(valid_dt),
        "path": str(path),
        "mean": mean,
        "rmse": rmse,
        "n": n,
    }


def append(manifest_path, record: dict) -> None:
    """Append one record as a JSON line (creating the file/dir if needed)."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def _parse_date(value):
    """Parse a YYYY-MM-DD (or full ISO) string to a datetime, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", ""))


def _record_valid_dt(rec):
    raw = rec.get("valid_dt") or rec.get("ts_utc")
    try:
        return datetime.fromisoformat(raw) if raw else None
    except (TypeError, ValueError):
        return None


def read(
    manifest_path,
    *,
    model=None,
    var=None,
    verif=None,
    mode=None,
    since=None,
    until=None,
    limit=None,
) -> list[dict]:
    """Read records matching the filters, newest first.

    *since*/*until* (YYYY-MM-DD or ISO) filter on the comparison's ``valid_dt``.
    *limit* caps the number returned after sorting.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return []

    since_dt = _parse_date(since)
    until_dt = _parse_date(until)

    out = []
    with open(manifest_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip a torn/partial line rather than crash
            if model and rec.get("model") != model:
                continue
            if var and rec.get("var") != var:
                continue
            if verif and rec.get("verif") != verif:
                continue
            if mode and rec.get("mode") != mode:
                continue
            if since_dt or until_dt:
                vdt = _record_valid_dt(rec)
                if vdt is None:
                    continue
                if since_dt and vdt < since_dt:
                    continue
                if until_dt and vdt > until_dt:
                    continue
            out.append(rec)

    out.sort(key=lambda r: r.get("ts_utc") or "", reverse=True)
    if limit is not None:
        out = out[: int(limit)]
    return out
