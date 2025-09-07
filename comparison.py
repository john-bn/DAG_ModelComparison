"""Utility functions for computing differences and simple metrics.

These functions work with either :class:`xarray.DataArray` objects or
simple Python sequences.  When xarray is available, alignment and
metadata are preserved.  Otherwise the functions fall back to basic list
operations, allowing the utilities to be tested without heavy
dependencies.
"""
from typing import Sequence, Union, Iterable

try:  # pragma: no cover - optional dependency
    import xarray as xr  # type: ignore
    _HAS_XARRAY = True
except Exception:  # pragma: no cover - executed when xarray missing
    xr = None
    _HAS_XARRAY = False

Number = Union[int, float]


def difference(model: Sequence[Number], analysis: Sequence[Number]):
    """Return model minus analysis.

    Parameters
    ----------
    model, analysis:
        Either ``xarray.DataArray`` objects or numeric sequences.  When
        xarray objects are provided they will be aligned by coordinate
        before subtraction; otherwise element-wise subtraction is
        performed.
    """
    if _HAS_XARRAY and isinstance(model, xr.DataArray) and isinstance(analysis, xr.DataArray):
        return model - analysis
    return [m - a for m, a in zip(model, analysis)]


def mean_bias(diff: Sequence[Number]):
    """Compute the mean of *diff*.

    Works for ``xarray.DataArray`` or any iterable of numbers.
    """
    if _HAS_XARRAY and isinstance(diff, xr.DataArray):
        return float(diff.mean().values)
    total = 0.0
    n = 0
    for val in diff:
        total += val
        n += 1
    return total / n if n else float('nan')


def mean_absolute_error(diff: Sequence[Number]):
    """Compute the mean absolute error of *diff*."""
    if _HAS_XARRAY and isinstance(diff, xr.DataArray):
        return float(abs(diff).mean().values)
    total = 0.0
    n = 0
    for val in diff:
        total += abs(val)
        n += 1
    return total / n if n else float('nan')


def evaluate(model: Sequence[Number], analysis: Sequence[Number]):
    """Convenience wrapper returning difference, bias and MAE.

    Returns a tuple ``(diff, metrics)`` where ``metrics`` is a
    dictionary containing ``bias`` and ``mae``.
    """
    diff = difference(model, analysis)
    metrics = {
        "bias": mean_bias(diff),
        "mae": mean_absolute_error(diff),
    }
    return diff, metrics
