import unittest

from src.models.train_model2_improved import SEED, build_model


class Model2ImprovedTests(unittest.TestCase):
    def test_model_configuration_is_fixed_and_reproducible(self) -> None:
        model = build_model()
        self.assertEqual(model.random_state, SEED)
        self.assertEqual(model.n_estimators, 500)
        self.assertEqual(model.min_samples_leaf, 3)
