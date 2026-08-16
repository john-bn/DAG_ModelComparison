"""Rolling real-time target selection.

A scheduled (cron) run has no human to type a date/init/forecast hour, so it
must decide *what* to verify from the wall clock. We anchor on the most recent
analysis time that should already be published:

    valid_dt = floor_to_hour(now_utc) - lag_hours

then ask the model registry which forecast cycles actually cover that valid
time (reusing :func:`comparator.normalize.find_runs_for_valid_time`, so we never
request a (cycle, fxx) the model does not produce) and pick the run whose lead
time is closest to the desired ``lead_hours``.

``now_utc`` is passed in by the caller (``datetime.now(timezone.utc)``) so this
module stays deterministic and unit-testable.
"""

from datetime import datetime, timedelta

from comparator import normalize

# Hours to subtract from "now" before picking an analysis time. Covers RTMA/URMA
# publication latency so the analysis file is on disk by the time we ask for it.
DEFAULT_LAG_HOURS = 1

# Forecast lead time (hours) we verify by default. The run closest to this lead
# is chosen among all cycles covering the analysis time.
DEFAULT_LEAD_HOURS = 24


class NoDataYet(Exception):
    """Raised when no model cycle covers the requested rolling analysis time.

    This is an expected, transient condition for a real-time job (e.g. the
    target hour is too fresh), not a hard failure — the CLI treats it as a
    warning with a clean exit so cron does not raise an alarm.
    """


def floor_to_hour(dt: datetime) -> datetime:
    """Truncate a datetime to the top of its hour."""
    return dt.replace(minute=0, second=0, microsecond=0)


def resolve_rolling_target(
    model_key: str,
    now_utc: datetime,
    lag_hours: int = DEFAULT_LAG_HOURS,
    lead_hours: int | None = None,
):
    """Resolve the (cycle_dt, fxx, valid_dt) to verify for a rolling run.

    Parameters
    ----------
    model_key : str
        Registry key (already normalized, e.g. ``"hrrr"``).
    now_utc : datetime
        Current time in UTC (timezone-aware or naive UTC). Injected for
        determinism/testability.
    lag_hours : int
        Data-latency buffer subtracted from the floored current hour.
    lead_hours : int or None
        Preferred forecast lead. Among the cycles that cover *valid_dt*, the one
        whose ``fxx`` is closest to this value is chosen (ties favor the longer
        lead, i.e. the more interesting forecast-vs-analysis comparison). When
        ``None``, :data:`DEFAULT_LEAD_HOURS` is used.

    Returns
    -------
    (cycle_dt, fxx, valid_dt)

    Raises
    ------
    NoDataYet
        If no cycle of *model_key* covers the resolved analysis time.
    """
    valid_dt = floor_to_hour(now_utc) - timedelta(hours=lag_hours)
    # Strip tzinfo so arithmetic/formatting matches the rest of the pipeline,
    # which works in naive-UTC datetimes throughout.
    if valid_dt.tzinfo is not None:
        valid_dt = valid_dt.replace(tzinfo=None)

    runs = normalize.find_runs_for_valid_time(model_key, valid_dt)
    if not runs:
        raise NoDataYet(
            f"No {model_key.upper()} init cycle covers analysis time "
            f"{valid_dt:%Y-%m-%d %H}Z (lag={lag_hours}h)."
        )

    target = DEFAULT_LEAD_HOURS if lead_hours is None else lead_hours
    # Closest lead to target; break ties toward the longer lead.
    cycle_dt, fxx = min(runs, key=lambda cf: (abs(cf[1] - target), -cf[1]))
    return cycle_dt, fxx, valid_dt
