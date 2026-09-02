import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.models.train_model1_local_llm_stacker import TRAITS, TrainingError, load_local_scores


class LocalModel1StackerTests(unittest.TestCase):
    def test_scores_load_in_requested_id_order_and_scale_to_zero_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            fields = ["record_id", "split"] + [f"score_{trait}" for trait in TRAITS] + ["label_cEXT"]
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"record_id": "second", "split": "train", **{f"score_{trait}": "5" for trait in TRAITS}, "label_cEXT": "1"})
                writer.writerow({"record_id": "first", "split": "train", **{f"score_{trait}": "10" for trait in TRAITS}, "label_cEXT": "0"})
            scores = load_local_scores(path, ["first", "second"], "train")
        self.assertTrue(np.allclose(scores[0], 1.0))
        self.assertTrue(np.allclose(scores[1], 0.5))

    def test_out_of_range_scores_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            fields = ["record_id", "split"] + [f"score_{trait}" for trait in TRAITS]
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"record_id": "one", "split": "train", **{f"score_{trait}": "11" for trait in TRAITS}})
            with self.assertRaises(TrainingError):
                load_local_scores(path, ["one"], "train")


if __name__ == "__main__":
    unittest.main()
