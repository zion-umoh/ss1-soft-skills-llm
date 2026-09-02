import csv
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from src.data.prepare_bfi2_bessi import BESSI_COLUMNS, BFI2_COLUMNS, PreparationError, prepare


class PrepareBfi2BessiTests(unittest.TestCase):
    def write_workbook(self, path: Path, invalid_score: bool = False) -> None:
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "College student sample"
        headers = ["Case", *BFI2_COLUMNS, *BESSI_COLUMNS]
        worksheet.append(headers)
        worksheet.append([1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        worksheet.append([1, 5 if not invalid_score else 6, 4, 3, 2, 1, 5, 4, 3, 2, 1])
        worksheet.append([2, None, None, None, None, None, 3, 3, 3, 3, 3])
        workbook.save(path)

    def test_standardises_scores_and_preserves_raw_workbook(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "sample.xlsx"
            self.write_workbook(source)
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
            self.assertEqual(summary["row_counts"]["raw_rows"], 3)
            self.assertEqual(summary["row_counts"]["eligible_rows"], 2)
            self.assertEqual(summary["row_counts"]["excluded_rows"], 1)
            self.assertFalse(summary["participant_linkage"]["case_identifier_is_unique"])
            self.assertEqual(source.read_bytes(), original_bytes)

            processed_rows = []
            for split in ("train", "validation", "test"):
                with (root / "processed" / f"bfi2_bessi_{split}.csv").open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    self.assertNotIn("source_case", reader.fieldnames)
                    processed_rows.extend(reader)
            self.assertEqual(len(processed_rows), 2)
            self.assertTrue(all(0 <= float(row["big_five_extraversion"]) <= 1 for row in processed_rows))
            self.assertTrue(all(1 <= float(row["bessi_self_management"]) <= 5 for row in processed_rows))
            self.assertTrue((root / "report.md").is_file())

    def test_rejects_scores_outside_documented_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "sample.xlsx"
            self.write_workbook(source, invalid_score=True)
            with self.assertRaises(PreparationError):
                prepare(
                    source,
                    Path(temporary_directory) / "interim",
                    Path(temporary_directory) / "processed",
                    Path(temporary_directory) / "report.md",
                    Path(temporary_directory) / "summary.json",
                )


if __name__ == "__main__":
    unittest.main()
