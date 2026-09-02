import csv
import tempfile
import unittest
from pathlib import Path

from src.data.prepare_goemotions import EMOTION_COLUMNS, METADATA_COLUMNS, PreparationError, prepare


class PrepareGoEmotionsTests(unittest.TestCase):
    def make_row(self, comment_id: str, text: str, rater_id: str, label: str) -> dict[str, str]:
        row = {column: "" for column in (*METADATA_COLUMNS, *EMOTION_COLUMNS)}
        row.update(
            {
                "text": text,
                "id": comment_id,
                "author": f"author-{comment_id}",
                "subreddit": "example",
                "link_id": f"link-{comment_id}",
                "parent_id": f"parent-{comment_id}",
                "created_utc": "0",
                "rater_id": rater_id,
                "example_very_unclear": "False",
            }
        )
        for emotion in EMOTION_COLUMNS:
            row[emotion] = "1" if emotion == label else "0"
        return row

    def write_raw(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*METADATA_COLUMNS, *EMOTION_COLUMNS])
            writer.writeheader()
            writer.writerows(rows)

    def write_split(self, path: Path, text: str, label_index: int, comment_id: str) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream, delimiter="\t").writerow([text, str(label_index), comment_id])

    def test_aggregates_and_validates_official_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_dir = root / "raw"
            official_dir = raw_dir / "official_splits"
            raw_dir.mkdir()
            official_dir.mkdir()
            rows = []
            fixtures = [
                ("comment-1", "Admiration text.", "admiration"),
                ("comment-2", "Anger text.", "anger"),
                ("comment-3", "Neutral text.", "neutral"),
            ]
            for comment_id, text, label in fixtures:
                rows.extend(self.make_row(comment_id, text, str(rater), label) for rater in range(3))
            source = raw_dir / "goemotions_1.csv"
            self.write_raw(source, rows)
            original_bytes = source.read_bytes()
            self.write_split(official_dir / "train.tsv", fixtures[0][1], EMOTION_COLUMNS.index("admiration"), fixtures[0][0])
            self.write_split(official_dir / "validation.tsv", fixtures[1][1], EMOTION_COLUMNS.index("anger"), fixtures[1][0])
            self.write_split(official_dir / "test.tsv", fixtures[2][1], EMOTION_COLUMNS.index("neutral"), fixtures[2][0])

            summary = prepare(
                raw_dir,
                official_dir,
                root / "interim",
                root / "processed",
                root / "report.md",
                root / "summary.json",
                train_ratio=0.34,
                validation_ratio=0.33,
                test_ratio=0.33,
            )
            self.assertEqual(summary["row_counts"]["raw_annotation_rows"], 9)
            self.assertEqual(summary["row_counts"]["agreement_filtered_comments"], 3)
            self.assertEqual(source.read_bytes(), original_bytes)
            processed_rows = []
            for split in ("train", "validation", "test"):
                with (root / "processed" / f"goemotions_{split}.csv").open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    self.assertNotIn("author", reader.fieldnames)
                    self.assertNotIn("id", reader.fieldnames)
                    processed_rows.extend(reader)
            self.assertEqual(len(processed_rows), 3)
            self.assertEqual(sum(int(row["admiration"]) for row in processed_rows), 1)

    def test_rejects_official_label_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_dir = root / "raw"
            official_dir = raw_dir / "official_splits"
            raw_dir.mkdir()
            official_dir.mkdir()
            self.write_raw(
                raw_dir / "goemotions_1.csv",
                [self.make_row("comment-1", "Text.", str(rater), "admiration") for rater in range(3)],
            )
            self.write_split(official_dir / "train.tsv", "Text.", EMOTION_COLUMNS.index("anger"), "comment-1")
            self.write_split(official_dir / "validation.tsv", "Other.", EMOTION_COLUMNS.index("anger"), "comment-2")
            self.write_split(official_dir / "test.tsv", "Else.", EMOTION_COLUMNS.index("anger"), "comment-3")
            with self.assertRaises(PreparationError):
                prepare(
                    raw_dir,
                    official_dir,
                    root / "interim",
                    root / "processed",
                    root / "report.md",
                    root / "summary.json",
                )

    def test_custom_split_keeps_exact_duplicate_text_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_dir = root / "raw"
            official_dir = raw_dir / "official_splits"
            raw_dir.mkdir()
            official_dir.mkdir()
            fixtures = [
                ("comment-1", "Repeated text.", "admiration"),
                ("comment-2", "Repeated text.", "admiration"),
                ("comment-3", "Text three.", "anger"),
                ("comment-4", "Text four.", "approval"),
                ("comment-5", "Text five.", "neutral"),
                ("comment-6", "Text six.", "joy"),
            ]
            raw_rows = []
            for comment_id, text, label in fixtures:
                raw_rows.extend(self.make_row(comment_id, text, str(rater), label) for rater in range(3))
            self.write_raw(raw_dir / "goemotions_1.csv", raw_rows)
            split_fixtures = {"train": fixtures[:2], "validation": fixtures[2:4], "test": fixtures[4:]}
            for split, fixture_rows in split_fixtures.items():
                with (official_dir / {"train": "train.tsv", "validation": "validation.tsv", "test": "test.tsv"}[split]).open(
                    "w", encoding="utf-8", newline=""
                ) as stream:
                    writer = csv.writer(stream, delimiter="\t")
                    for comment_id, text, label in fixture_rows:
                        writer.writerow([text, str(EMOTION_COLUMNS.index(label)), comment_id])
            prepare(
                raw_dir,
                official_dir,
                root / "interim",
                root / "processed",
                root / "report.md",
                root / "summary.json",
            )
            with (root / "interim" / "goemotions_comment_audit.csv").open(newline="") as stream:
                repeated = [
                    row for row in csv.DictReader(stream) if row["text_model"] == "Repeated text."
                ]
            self.assertEqual(len(repeated), 2)
            self.assertEqual({row["split"] for row in repeated}, {repeated[0]["split"]})
