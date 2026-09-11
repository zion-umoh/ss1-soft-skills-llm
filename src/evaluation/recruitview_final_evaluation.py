"""Participant-grouped evaluation helpers for the final RecruitView runs.

This module intentionally contains protocol utilities only.  The final model
is selected and frozen in the later implementation batches; no post-hoc
multi-target evaluator or previously viewed test split is retained here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import KFold


class FinalEvaluationError(ValueError):
    """Raised when a grouped evaluation contract cannot be satisfied."""


def _group_splits(groups: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return complete participant-disjoint folds over row indices."""
    values = np.asarray(groups, dtype=object)
    unique = np.asarray(sorted(set(values.tolist())), dtype=object)
    if folds < 2:
        raise FinalEvaluationError("folds must be at least two")
    if len(unique) < folds:
        raise FinalEvaluationError(f"Need at least {folds} participants, found {len(unique)}.")
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for train_group_indices, test_group_indices in splitter.split(unique):
        train_groups = set(unique[train_group_indices].tolist())
        test_groups = set(unique[test_group_indices].tolist())
        train_indices = np.asarray([i for i, group in enumerate(values) if group in train_groups], dtype=int)
        test_indices = np.asarray([i for i, group in enumerate(values) if group in test_groups], dtype=int)
        if train_groups & test_groups:
            raise FinalEvaluationError("Participant leakage detected while constructing grouped folds.")
        result.append((train_indices, test_indices))
    if sorted(index for _, test in result for index in test.tolist()) != list(range(len(values))):
        raise FinalEvaluationError("Grouped folds do not cover every row exactly once.")
    return result


def assert_group_disjoint(train_groups: Sequence[str], test_groups: Sequence[str]) -> None:
    """Raise if any participant identifier appears in both partitions."""
    overlap = set(train_groups) & set(test_groups)
    if overlap:
        raise FinalEvaluationError(f"Participant leakage detected: {sorted(overlap)!r}")
