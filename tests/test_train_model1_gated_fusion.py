import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import spearmanr

from src.models.train_model1_gated_fusion import (
    Config, GatedFusion, correlations, fit_neural, group_bootstrap,
    predict_artifact, split_indices, training_loss,
)


class GatedFusionTests(unittest.TestCase):
    def test_participant_overlap_rejected(self):
        rows = [{"record_id": str(i), "participant_id": p, "split": s}
                for i, (p, s) in enumerate((("a", "train"), ("b", "validation"), ("a", "test")))]
        with self.assertRaisesRegex(ValueError, "leakage"):
            split_indices(rows)

    def test_gate_weights_are_convex_and_gradients_finite(self):
        model = GatedFusion(10, 12, Config())
        prediction, gates = model(torch.randn(8, 10), torch.randn(8, 12))
        torch.testing.assert_close(gates.sum(dim=1), torch.ones(8))
        self.assertTrue(bool((gates >= 0).all()))
        loss = training_loss(prediction, torch.randn(8, 6), "gated_huber_rank")
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_huber_limits_outlier_gradient(self):
        pred = torch.tensor([[100.0]], requires_grad=True)
        training_loss(pred, torch.zeros_like(pred), "gated_huber").backward()
        self.assertAlmostEqual(pred.grad.item(), 1.0)

    def test_preprocessing_excludes_evaluation_and_artifact_reloads(self):
        rng = np.random.default_rng(11)
        text, audio, y = rng.normal(size=(20, 10)), rng.normal(size=(20, 12)), rng.normal(size=(20, 6))
        train, evaluation = np.arange(15), np.arange(15, 20)
        text[evaluation] += 1000
        config = Config(checkpoints=(3,))
        results, scalers = fit_neural(text, audio, y, train, evaluation, "gated_huber", config, 17, (3,))
        np.testing.assert_allclose(scalers[0].mean_, text[train].mean(axis=0))
        # Changing held-out labels cannot change training or predictions.
        changed_y = y.copy()
        changed_y[evaluation] += 1e6
        second, _ = fit_neural(text, audio, changed_y, train, evaluation, "gated_huber", config, 17, (3,))
        np.testing.assert_array_equal(results[3]["prediction"], second[3]["prediction"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump({"config": asdict(config), "variant": "gated_huber",
                         "members": [{"state": results[3]["state"], "scalers": scalers}]}, path)
            np.testing.assert_allclose(predict_artifact(path, text[evaluation], audio[evaluation]), results[3]["prediction"])

    def test_tied_rank_metrics_and_paired_bootstrap(self):
        rng = np.random.default_rng(13)
        y = rng.integers(0, 5, size=(30, 6)).astype(float)
        pred = rng.normal(size=(30, 6))
        np.testing.assert_allclose(correlations(y, pred), [spearmanr(y[:, i], pred[:, i]).statistic for i in range(6)])
        ci = group_bootstrap(y, pred, pred, np.repeat(np.arange(10), 3), 20)
        np.testing.assert_array_equal(ci["big_five_delta_vs_ridge"], [0, 0])
        np.testing.assert_array_equal(ci["speaking_delta_vs_ridge"], [0, 0])
