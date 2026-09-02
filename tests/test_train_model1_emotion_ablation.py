import unittest

import numpy as np

from src.models.train_model1_emotion_ablation import TRAITS, metrics


class Model1EmotionAblationTests(unittest.TestCase):
    def test_metrics_reports_thresholded_and_ranking_measures(self) -> None:
        labels = np.asarray([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=np.int8)
        probabilities = np.asarray([[0.1] * len(TRAITS), [0.9] * len(TRAITS)])
        result = metrics(labels, probabilities)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["macro_balanced_accuracy"], 1.0)
        self.assertEqual(result["macro_auroc"], 1.0)
