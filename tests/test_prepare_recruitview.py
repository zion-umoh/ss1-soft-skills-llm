import json
import tempfile
import unittest
from pathlib import Path

from src.data.prepare_recruitview import (
    OUTPUT_COLUMNS,
    RecruitViewPreparationError,
    assign_group_safe_splits,
    load_records,
    prepare,
)


def source_row(identifier: str, participant: str, transcript: str = "A useful answer.") -> dict[str, object]:
    row: dict[str, object] = {
        "id": identifier,
        "file_name": f"videos/{identifier}.mp4",
        "duration": "short",
        "question_id": "1",
        "question": "Introduce yourself",
        "video_quality": "High",
        "user_no": participant,
        "transcript": transcript,
        "gemini_summary": "ignored",
    }
    for field in (
        "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
        "speaking_skills",
    ):
        row[field] = 0.25
    return row


class RecruitViewPreparationTests(unittest.TestCase):
    def write_jsonl(self, directory: str, rows: list[dict[str, object]]) -> Path:
        path = Path(directory) / "metadata.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        return path

    def test_load_excludes_empty_transcript_and_keeps_numeric_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, [source_row("1", "a"), source_row("2", "b", "")])
            records, exclusions, summary = load_records(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(exclusions), 1)
        self.assertEqual(summary["participants"], 1)
        self.assertEqual(records[0].traits[0], 0.25)
        self.assertEqual(records[0].speaking_skills, 0.25)

    def test_duplicate_source_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, [source_row("1", "a"), source_row("1", "b")])
            with self.assertRaisesRegex(RecruitViewPreparationError, "duplicate id"):
                load_records(path)

    def test_participant_never_spans_splits_and_assignment_is_deterministic(self) -> None:
        rows = [source_row(str(index), f"participant-{index}") for index in range(20)]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, rows)
            records, _, _ = load_records(path)
        first = assign_group_safe_splits(records)
        second = assign_group_safe_splits(records)
        self.assertEqual([record.split for record in first], [record.split for record in second])
        by_participant: dict[str, set[str]] = {}
        for record in first:
            by_participant.setdefault(record.participant_id, set()).add(record.split)
        self.assertTrue(all(len(splits) == 1 for splits in by_participant.values()))

    def test_prepare_writes_model_ready_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_jsonl(directory, [source_row("1", "a")])
            output_dir = root / "processed"
            paths = prepare(path, output_dir)
            header = paths["prepared"].read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(tuple(header.split(",")), OUTPUT_COLUMNS)
        self.assertNotIn("gemini_summary", header)


if __name__ == "__main__":
    unittest.main()
