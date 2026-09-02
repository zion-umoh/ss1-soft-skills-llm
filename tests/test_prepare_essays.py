import csv
import tempfile
import unittest
from pathlib import Path

from src.data.prepare_essays import PreparationError, prepare


class PrepareEssaysTests(unittest.TestCase):
    def write_input(self, path: Path, rows: list[list[str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["#AUTHID", "TEXT", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"])
            writer.writerows(rows)

    def test_prepares_leakage_safe_processed_splits(self) -> None:
        rows = [
            ["author-a", "Same text for duplicate checking.", 1, 0, 1, 0, 1],
            ["author-b", "Same text for duplicate checking.", 0, 1, 0, 1, 0],
            ["author-c", "Unique text one.", 1, 1, 0, 0, 1],
            ["author-d", "Unique text two.", 0, 0, 1, 1, 0],
            ["author-e", "Unique text three.", 1, 0, 0, 1, 1],
            ["author-f", "   ", 0, 1, 1, 0, 0],
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "essays.csv"
            self.write_input(source, rows)
            original_bytes = source.read_bytes()
            summary = prepare(
                source,
                root / "interim",
                root / "processed",
                root / "report.md",
                root / "summary.json",
                train_ratio=0.5,
                validation_ratio=0.25,
                test_ratio=0.25,
            )
            self.assertEqual(summary["row_counts"]["raw_rows"], 6)
            self.assertEqual(summary["row_counts"]["eligible_rows"], 5)
            self.assertEqual(summary["row_counts"]["excluded_rows"], 1)
            self.assertEqual(summary["text_quality"]["duplicate_model_text_records"], 1)
            self.assertEqual(source.read_bytes(), original_bytes)

            duplicate_splits = []
            for split in ("train", "validation", "test"):
                with (root / "processed" / f"essays_{split}.csv").open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    self.assertNotIn("#AUTHID", reader.fieldnames)
                    self.assertNotIn("TEXT", reader.fieldnames)
                    duplicate_splits.extend(
                        split for row in reader if row["text_model"] == "Same text for duplicate checking."
                    )
            self.assertEqual(len(duplicate_splits), 2)
            self.assertEqual(len(set(duplicate_splits)), 1)
            self.assertTrue((root / "report.md").is_file())

    def test_rejects_non_binary_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "essays.csv"
            self.write_input(source, [["author-a", "Some text", "yes", 0, 1, 0, 1]])
            with self.assertRaises(PreparationError):
                prepare(
                    source,
                    root / "interim",
                    root / "processed",
                    root / "report.md",
                    root / "summary.json",
                )


if __name__ == "__main__":
    unittest.main()
