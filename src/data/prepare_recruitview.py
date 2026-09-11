"""Prepare the gated RecruitView metadata for feature-based evaluation.

The source metadata contains one row per recorded answer and multiple answers
per participant.  This script keeps every participant in exactly one split so
that an identity cannot leak between evaluation partitions.  The transcript
and active labels are retained; raw media and generated summaries are not.

The raw JSONL file is never modified.  Derived files are written below an
ignored data directory by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping


TRAIT_COLUMNS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)
SOFT_SKILL_COLUMN = "speaking_skills"
REQUIRED_COLUMNS = (
    "id",
    "file_name",
    "question_id",
    "question",
    "user_no",
    "transcript",
    *TRAIT_COLUMNS,
    SOFT_SKILL_COLUMN,
)
OUTPUT_COLUMNS = (
    "record_id",
    "participant_id",
    "question_id",
    "question",
    "media_path",
    "text_model",
    *TRAIT_COLUMNS,
    SOFT_SKILL_COLUMN,
    "split",
)
SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


class RecruitViewPreparationError(ValueError):
    """Raised when RecruitView metadata violates the preparation contract."""


@dataclass(frozen=True)
class RecruitViewRecord:
    source_row_number: int
    source_id: str
    participant_id: str
    question_id: str
    question: str
    media_path: str
    text_model: str
    traits: tuple[float, ...]
    speaking_skills: float
    split: str = ""

    def output_row(self) -> dict[str, str]:
        values: dict[str, str] = {
            "record_id": "recruitview_" + hashlib.sha256(self.source_id.encode("utf-8")).hexdigest()[:16],
            "participant_id": self.participant_id,
            "question_id": self.question_id,
            "question": self.question,
            "media_path": self.media_path,
            "text_model": self.text_model,
            "split": self.split,
        }
        values.update({column: f"{value:.10g}" for column, value in zip(TRAIT_COLUMNS, self.traits, strict=True)})
        values[SOFT_SKILL_COLUMN] = f"{self.speaking_skills:.10g}"
        return values


def clean_text(value: object) -> str:
    """Conservatively normalise transcript text without changing its meaning."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFC", value)
    text = "".join(
        " " if unicodedata.category(character) == "Cc" and character not in "\n\r\t" else character
        for character in text
    )
    return re.sub(r"\s+", " ", text).strip()


def _required_text(row: Mapping[str, object], field: str, row_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RecruitViewPreparationError(f"Row {row_number}: {field} must be a non-empty string.")
    return value.strip()


def _finite_number(row: Mapping[str, object], field: str, row_number: int) -> float:
    value = row.get(field)
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise RecruitViewPreparationError(f"Row {row_number}: {field} must be numeric.") from error
    if not math.isfinite(number):
        raise RecruitViewPreparationError(f"Row {row_number}: {field} must be finite.")
    return number


def load_records(input_path: Path) -> tuple[list[RecruitViewRecord], list[dict[str, str]], dict[str, object]]:
    """Load and validate JSONL rows, excluding only empty transcripts."""
    records: list[RecruitViewRecord] = []
    exclusions: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    try:
        stream = input_path.open("r", encoding="utf-8")
    except OSError as error:
        raise RecruitViewPreparationError(f"Cannot open RecruitView metadata: {input_path}") from error

    with stream:
        for row_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RecruitViewPreparationError(f"Row {row_number}: invalid JSON.") from error
            if not isinstance(row, Mapping):
                raise RecruitViewPreparationError(f"Row {row_number}: expected a JSON object.")
            missing = [field for field in REQUIRED_COLUMNS if field not in row]
            if missing:
                raise RecruitViewPreparationError(f"Row {row_number}: missing required fields {missing}.")
            source_id = _required_text(row, "id", row_number)
            if source_id in seen_ids:
                raise RecruitViewPreparationError(f"Row {row_number}: duplicate id {source_id!r}.")
            seen_ids.add(source_id)
            participant_id = _required_text(row, "user_no", row_number)
            question_id = _required_text(row, "question_id", row_number)
            question = _required_text(row, "question", row_number)
            media_path = _required_text(row, "file_name", row_number)
            text_model = clean_text(row.get("transcript"))
            if not text_model:
                exclusions.append(
                    {
                        "source_row_number": str(row_number),
                        "source_id": source_id,
                        "participant_id": participant_id,
                        "exclusion_reason": "empty_transcript_after_normalisation",
                    }
                )
                continue
            traits = tuple(_finite_number(row, field, row_number) for field in TRAIT_COLUMNS)
            speaking_skills = _finite_number(row, SOFT_SKILL_COLUMN, row_number)
            records.append(
                RecruitViewRecord(
                    source_row_number=row_number,
                    source_id=source_id,
                    participant_id=participant_id,
                    question_id=question_id,
                    question=question,
                    media_path=media_path,
                    text_model=text_model,
                    traits=traits,
                    speaking_skills=speaking_skills,
                )
            )

    if not records:
        raise RecruitViewPreparationError("No eligible RecruitView records remain.")
    participants = {record.participant_id for record in records}
    summary = {
        "source_path": str(input_path),
        "source_rows_read": len(records) + len(exclusions),
        "eligible_records": len(records),
        "excluded_records": len(exclusions),
        "participants": len(participants),
        "questions": len({record.question_id for record in records}),
        "required_columns": list(REQUIRED_COLUMNS),
        "model_input": "transcript plus later engineered audio columns; generated summaries and raw media metadata excluded",
    }
    return records, exclusions, summary


def assign_group_safe_splits(records: Iterable[RecruitViewRecord]) -> list[RecruitViewRecord]:
    """Assign deterministic 70/15/15 splits while keeping participants intact."""
    records = list(records)
    participant_ids = sorted({record.participant_id for record in records})
    ranked = sorted(
        participant_ids,
        key=lambda participant: hashlib.sha256(f"recruitview-split:{participant}".encode("utf-8")).hexdigest(),
    )
    total = len(ranked)
    train_end = round(total * SPLIT_RATIOS["train"])
    validation_end = train_end + round(total * SPLIT_RATIOS["validation"])
    split_by_participant = {
        participant: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, participant in enumerate(ranked)
    }
    return [
        RecruitViewRecord(
            source_row_number=record.source_row_number,
            source_id=record.source_id,
            participant_id=record.participant_id,
            question_id=record.question_id,
            question=record.question,
            media_path=record.media_path,
            text_model=record.text_model,
            traits=record.traits,
            speaking_skills=record.speaking_skills,
            split=split_by_participant[record.participant_id],
        )
        for record in records
    ]


def write_prepared(
    records: Iterable[RecruitViewRecord],
    exclusions: Iterable[dict[str, str]],
    summary: Mapping[str, object],
    output_dir: Path,
) -> dict[str, Path]:
    """Write model-ready data, exclusions, and a non-sensitive audit summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    output_path = output_dir / "recruitview_prepared.csv"
    exclusions_path = output_dir / "recruitview_exclusions.csv"
    summary_path = output_dir / "recruitview_preparation_summary.json"
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(record.output_row() for record in records)
    with exclusions_path.open("w", encoding="utf-8", newline="") as stream:
        fields = ("source_row_number", "source_id", "participant_id", "exclusion_reason")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(exclusions)
    split_summary = {
        split: {
            "records": sum(record.split == split for record in records),
            "participants": len({record.participant_id for record in records if record.split == split}),
        }
        for split in SPLITS
    }
    payload = {
        **dict(summary),
        "created_at": datetime.now(UTC).isoformat(),
        "split_strategy": "stable hash ranking by participant_id; no participant appears in multiple splits",
        "splits": split_summary,
        "output_columns": list(OUTPUT_COLUMNS),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"prepared": output_path, "exclusions": exclusions_path, "summary": summary_path}


def prepare(input_path: Path, output_dir: Path) -> dict[str, Path]:
    records, exclusions, summary = load_records(input_path)
    return write_prepared(assign_group_safe_splits(records), exclusions, summary, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/recruitview/metadata.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/recruitview"))
    args = parser.parse_args()
    try:
        paths = prepare(args.input, args.output_dir)
    except RecruitViewPreparationError as error:
        raise SystemExit(str(error)) from error
    print(f"Prepared RecruitView metadata: {paths['prepared']}")
    print(f"Excluded-record audit: {paths['exclusions']}")
    print(f"Summary: {paths['summary']}")


if __name__ == "__main__":
    main()
