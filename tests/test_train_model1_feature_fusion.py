from __future__ import annotations

import unittest

import numpy as np

from src.models.train_model1_feature_fusion import _select_blend_weight


class Model1FeatureFusionTests(unittest.TestCase):
    def test_late_fusion_selects_the_better_validation_modality(self) -> None:
        target = np.arange(6, dtype=float)[:, None]
        text_prediction = target[::-1]
        audio_prediction = target.copy()

        weight, candidates = _select_blend_weight(text_prediction, audio_prediction, target, ("target",))

        self.assertLessEqual(weight, 0.45)
        self.assertEqual(len(candidates), 21)


if __name__ == "__main__":
    unittest.main()
