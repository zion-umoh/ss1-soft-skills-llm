import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.models.train_model2_benchmark import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    TrainingError,
    load_split,
    regression_metrics,
)


class Model2BenchmarkTests(unittest.TestCase):
    def test_regression_metrics_are_per_target_and_perfect_when_predictions_match(self) -> None:
        targets = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]])
        result = regression_metrics(targets, targets.copy())
        self.assertEqual(result["mae"], 0.0)
        self.assertEqual(result["rmse"], 0.0)
        self.assertEqual(result["per_target"][TARGET_COLUMNS[0]]["mae"], 0.0)

    def test_load_split_rejects_mixed_split_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            path.write_text(
                ",".join(["record_id", "split", *FEATURE_COLUMNS, *TARGET_COLUMNS]) + "\n"
                + ",".join(["row_1", "test", *(["0.5"] * len(FEATURE_COLUMNS)), *(["3.0"] * len(TARGET_COLUMNS))])
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(TrainingError):
                load_split(path, "train")
