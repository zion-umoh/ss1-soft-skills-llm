"""Model-agnostic metrics for participant-held-out regression evaluation."""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Metric inputs must be a non-empty finite vector.")
    return array


def pearson(prediction: Sequence[float], target: Sequence[float]) -> float | None:
    pred = _finite(prediction)
    truth = _finite(target)
    if len(pred) != len(truth) or len(pred) < 2 or np.std(pred) == 0 or np.std(truth) == 0:
        return None
    return float(np.corrcoef(pred, truth)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(prediction: Sequence[float], target: Sequence[float]) -> float | None:
    pred = _finite(prediction)
    truth = _finite(target)
    if len(pred) != len(truth):
        raise ValueError("Metric vectors must have equal length.")
    return pearson(_rank(pred), _rank(truth))


def ranking_accuracy(prediction: Sequence[float], target: Sequence[float]) -> float | None:
    """Pairwise ranking agreement, with ties receiving half credit."""
    pred = _finite(prediction)
    truth = _finite(target)
    if len(pred) != len(truth) or len(pred) < 2:
        return None
    numerator = 0.0
    denominator = 0
    for left in range(len(pred)):
        for right in range(left + 1, len(pred)):
            pred_delta = pred[left] - pred[right]
            truth_delta = truth[left] - truth[right]
            if pred_delta == 0 or truth_delta == 0:
                numerator += 0.5
            elif pred_delta * truth_delta > 0:
                numerator += 1.0
            denominator += 1
    return float(numerator / denominator) if denominator else None


def standardized_errors(prediction: Sequence[float], target: Sequence[float]) -> dict[str, float | None]:
    pred = _finite(prediction)
    truth = _finite(target)
    if len(pred) != len(truth):
        raise ValueError("Metric vectors must have equal length.")
    pred_std = float(np.std(pred))
    truth_std = float(np.std(truth))
    if pred_std == 0 or truth_std == 0:
        return {"mae": None, "rmse": None}
    error = (pred - np.mean(pred)) / pred_std - (truth - np.mean(truth)) / truth_std
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2)))}


def calibration(prediction: Sequence[float], target: Sequence[float]) -> dict[str, float | None]:
    """Return the descriptive linear calibration slope/intercept and R²."""
    pred = _finite(prediction)
    truth = _finite(target)
    if len(pred) != len(truth) or len(pred) < 2 or np.var(pred) == 0:
        return {"slope": None, "intercept": None, "r2": None}
    slope = float(np.cov(pred, truth, ddof=0)[0, 1] / np.var(pred))
    intercept = float(np.mean(truth) - slope * np.mean(pred))
    fitted = intercept + slope * pred
    residual = float(np.sum((truth - fitted) ** 2))
    total = float(np.sum((truth - np.mean(truth)) ** 2))
    return {"slope": slope, "intercept": intercept, "r2": None if total == 0 else float(1.0 - residual / total)}


def bootstrap_ci(
    prediction: Sequence[float],
    target: Sequence[float],
    metric: Callable[[Sequence[float], Sequence[float]], float | None],
    *,
    seed: int = 42,
    resamples: int = 1000,
) -> tuple[float, float] | None:
    pred = _finite(prediction)
    truth = _finite(target)
    if len(pred) != len(truth) or resamples < 1:
        raise ValueError("Bootstrap inputs must have equal length and positive resamples.")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        indices = rng.integers(0, len(pred), len(pred))
        value = metric(pred[indices].tolist(), truth[indices].tolist())
        if value is not None and math.isfinite(value):
            values.append(float(value))
    if not values:
        return None
    lower, upper = np.percentile(np.asarray(values), [2.5, 97.5])
    return float(lower), float(upper)
