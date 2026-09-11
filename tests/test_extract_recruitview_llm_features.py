from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data.extract_recruitview_llm_features import (
    FEATURE_COLUMNS,
    _validate_features,
    run,
)


class RecruitViewLLMFeatureTests(unittest.TestCase):
    def test_feature_schema_accepts_only_bounded_integer_scores(self) -> None:
        value = {column: 2 for column in FEATURE_COLUMNS}
        self.assertEqual(_validate_features(value), value)
        with self.assertRaises(ValueError):
            _validate_features({**value, FEATURE_COLUMNS[0]: 6})

    def test_dry_run_writes_a_resumable_manifest_without_api_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared.csv"
            with prepared.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["record_id", "participant_id", "question_id", "question", "text_model", "split"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "record_id": "r1",
                        "participant_id": "p1",
                        "question_id": "1",
                        "question": "Introduce yourself",
                        "text_model": "I enjoy solving engineering problems.",
                        "split": "train",
                    }
                )
            progress = root / "progress.json"
            result = run(
                prepared,
                root / "features.jsonl",
                root / "features.csv",
                progress,
                root / "failures.jsonl",
                dry_run=True,
            )
            self.assertEqual(result["status"], "dry_run")
            self.assertEqual(result["remaining_rows"], 1)
            self.assertEqual(json.loads(progress.read_text())["model"], "gpt-5.6-luna")


if __name__ == "__main__":
    unittest.main()
