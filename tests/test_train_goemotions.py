import unittest

import numpy as np

from src.models.train_goemotions import EMOTIONS, calibrate_f1_thresholds, metrics, threshold_predictions


class GoEmotionsModelTests(unittest.TestCase):
    def test_thresholds_are_validation_derived_and_binary_predictions_match_shape(self) -> None:
        probabilities = np.full((4, len(EMOTIONS)), 0.1, dtype=float)
        labels = np.zeros((4, len(EMOTIONS)), dtype=np.int8)
        probabilities[:, 0] = [0.9, 0.8, 0.2, 0.1]
        labels[:, 0] = [1, 1, 0, 0]
        thresholds = calibrate_f1_thresholds(labels, probabilities)
        predictions = threshold_predictions(probabilities, thresholds)
        self.assertEqual(thresholds.shape, (len(EMOTIONS),))
        self.assertEqual(predictions.shape, labels.shape)
        self.assertTrue(np.isin(predictions, [0, 1]).all())
        self.assertGreaterEqual(thresholds[0], 0.1)
        self.assertLessEqual(thresholds[0], 0.9)

    def test_metrics_reports_each_emotion(self) -> None:
        labels = np.eye(len(EMOTIONS), dtype=np.int8)
        result = metrics(labels, labels.copy())
        self.assertEqual(result["macro"]["f1"], 1.0)
        self.assertEqual(result["per_label"]["admiration"]["support"], 1)
