import unittest

from src.models.evaluate_model2_selected import choose_candidate


class Model2SelectionTests(unittest.TestCase):
    def test_selection_uses_mae_then_rmse(self) -> None:
        linear = {"validation_metrics": {"mae": 0.2, "rmse": 0.4, "r2": 0.5}}
        nonlinear = {"validation_metrics": {"mae": 0.2, "rmse": 0.3, "r2": 0.1}}
        selected, _ = choose_candidate(linear, nonlinear)
        self.assertEqual(selected, "extra_trees")
