from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.recruitview_final_evaluation import _group_splits


class RecruitViewFinalEvaluationTests(unittest.TestCase):
    def test_group_splits_are_complete_and_participant_disjoint(self) -> None:
        groups = np.asarray(["p1", "p1", "p2", "p3", "p3", "p4", "p5"], dtype=object)
        seen: list[int] = []
        for train_indices, test_indices in _group_splits(groups, folds=3, seed=42):
            self.assertFalse(set(groups[train_indices]) & set(groups[test_indices]))
            seen.extend(test_indices.tolist())
        self.assertEqual(sorted(seen), list(range(len(groups))))


if __name__ == "__main__":
    unittest.main()
