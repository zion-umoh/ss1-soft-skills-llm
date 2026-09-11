import unittest

import numpy as np

from src.evaluation.close_recruitview_study import clustered_intervals, partial_rank_correlations


class ClosingEvaluationTests(unittest.TestCase):
    def test_paired_identical_predictions_have_zero_difference(self):
        rng = np.random.default_rng(7)
        y, p = rng.normal(size=(60, 6)), rng.normal(size=(60, 6))
        result = clustered_intervals(y, {"llm_fused": p, "length_only": p.copy()}, np.repeat(np.arange(15), 4), 25)
        for interval in result["primary_minus_baseline"]["length_only"].values():
            np.testing.assert_array_equal(interval, [0, 0])

    def test_partial_rank_removes_an_exact_control_predictor(self):
        rng = np.random.default_rng(8)
        control = np.arange(60, dtype=float)[:, None]
        y = rng.normal(size=(60, 6))
        p = np.repeat(control, 6, axis=1)
        self.assertTrue(np.isnan(partial_rank_correlations(y, p, control)).all())
