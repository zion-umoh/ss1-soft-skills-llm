from __future__ import annotations

import unittest

from src.evaluation.metrics import calibration, ranking_accuracy, spearman


class EvaluationMetricsTests(unittest.TestCase):
    def test_ranking_and_calibration(self) -> None:
        self.assertEqual(ranking_accuracy([1, 2, 3], [10, 20, 30]), 1.0)
        result = calibration([1, 2, 3], [2, 4, 6])
        self.assertAlmostEqual(result["slope"], 2.0)
        self.assertAlmostEqual(result["intercept"], 0.0)

    def test_spearman_orders_values(self) -> None:
        self.assertEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)


if __name__ == "__main__":
    unittest.main()
