"""Prepare the Essays dataset for the Model 1 Big Five experiment.

The script leaves the raw CSV untouched and produces audit, exclusion,
model-ready split, summary and Markdown quality-report artefacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


TRAIT_COLUMNS = ("cEXT", "cNEU", "cAGR", "cCON", "cOPN")
REQUIRED_COLUMNS = ("#AUTHID", "TEXT", *TRAIT_COLUMNS)
SPLITS = ("train", "validation", "test")
PROCESSED_COLUMNS = ("record_id", "text_model", *TRAIT_COLUMNS, "split")
SHORT_TEXT_TOKEN_THRESHOLD = 50


class PreparationError(ValueError):
    """Raised when raw data does not meet the preparation contract."""


@dataclass
class EssayRecord:
    source_row_number: int
    author_id: str
    text_raw: str
    text_model: str
    labels: tuple[int, int, int, int, int]
    char_count: int
    token_count: int
    replacement_character_count: int
    control_characters_replaced: int
    record_id: str
    split_group_id: str = ""
    split: str = ""


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(text: str) -> tuple[str, int]:
    """Apply reversible-in-spirit, conservative model-text normalisation."""
    text = unicodedata.normalize("NFC", text)
    replaced_controls = sum(
        1
        for character in text
        if unicodedata.category(character) == "Cc" and character not in "\n\r\t"
    )
    text = "".join(
        " "
        if unicodedata.category(character) == "Cc" and character not in "\n\r\t"
        else character
        for character in text
    )
    return re.sub(r"\s+", " ", text).strip(), replaced_controls


def token_count(text: str) -> int:
    return len(re.findall(r"\b\w+(?:['’]\w+)?\b", text, flags=re.UNICODE))


def parse_binary(value: str | None, column: str, source_row_number: int) -> int:
    normalised = (value or "").strip()
    if normalised not in {"0", "1"}:
        raise PreparationError(
            f"Row {source_row_number}: {column} must be 0 or 1, got {value!r}."
        )
    return int(normalised)


def read_essays(input_path: Path) -> tuple[list[EssayRecord], list[dict[str, str]], list[str]]:
    csv.field_size_limit(sys.maxsize)
    records: list[EssayRecord] = []
    exclusions: list[dict[str, str]] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in header]
        unexpected_columns = [column for column in header if column not in REQUIRED_COLUMNS]
        if missing_columns or unexpected_columns:
            raise PreparationError(
                "Essays schema mismatch. "
                f"Missing: {missing_columns or 'none'}; unexpected: {unexpected_columns or 'none'}."
            )

        for source_row_number, row in enumerate(reader, start=2):
            author_id = (row["#AUTHID"] or "").strip()
            if not author_id:
                raise PreparationError(f"Row {source_row_number}: #AUTHID is empty.")
            labels = tuple(
                parse_binary(row[column], column, source_row_number) for column in TRAIT_COLUMNS
            )
            text_raw = row["TEXT"] or ""
            text_model, replaced_controls = clean_text(text_raw)
            record_id = "essay_" + hashlib.sha256(
                f"essays:{source_row_number}".encode("utf-8")
            ).hexdigest()[:16]
            if not text_model:
                exclusions.append(
                    {
                        "source_row_number": str(source_row_number),
                        "author_id": author_id,
                        "exclusion_reason": "empty_text_after_conservative_normalisation",
                    }
                )
                continue
            records.append(
                EssayRecord(
                    source_row_number=source_row_number,
                    author_id=author_id,
                    text_raw=text_raw,
                    text_model=text_model,
                    labels=labels,
                    char_count=len(text_model),
                    token_count=token_count(text_model),
                    replacement_character_count=text_raw.count("\ufffd"),
                    control_characters_replaced=replaced_controls,
                    record_id=record_id,
                )
            )
    if not records:
        raise PreparationError("No eligible Essay records remain after text validation.")
    return records, exclusions, header


def build_groups(records: list[EssayRecord]) -> list[list[EssayRecord]]:
    """Group records by author and exact model text to prevent split leakage."""
    union_find = UnionFind(len(records))
    by_author: dict[str, int] = {}
    by_text: dict[str, int] = {}
    for index, record in enumerate(records):
        if record.author_id in by_author:
            union_find.union(index, by_author[record.author_id])
        else:
            by_author[record.author_id] = index
        if record.text_model in by_text:
            union_find.union(index, by_text[record.text_model])
        else:
            by_text[record.text_model] = index

    groups: dict[int, list[EssayRecord]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[union_find.find(index)].append(record)

    result = list(groups.values())
    for group in result:
        fingerprint = "|".join(sorted(record.record_id for record in group))
        group_id = "group_" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
        for record in group:
            record.split_group_id = group_id
    return result


def target_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: total * ratios[split] for split in SPLITS}
    counts = {split: math.floor(raw[split]) for split in SPLITS}
    remainder = total - sum(counts.values())
    for split in sorted(SPLITS, key=lambda item: (raw[item] - counts[item], item), reverse=True)[:remainder]:
        counts[split] += 1
    return counts


def stratified_group_split(
    groups: list[list[EssayRecord]], ratios: dict[str, float], seed: int
) -> dict[str, list[EssayRecord]]:
    """Assign leakage-safe groups while approximately preserving five label rates."""
    total_rows = sum(len(group) for group in groups)
    row_targets = target_counts(total_rows, ratios)
    all_positive_counts = [
        sum(record.labels[trait_index] for group in groups for record in group)
        for trait_index in range(len(TRAIT_COLUMNS))
    ]
    label_targets = {
        split: [all_positive_counts[index] * ratios[split] for index in range(len(TRAIT_COLUMNS))]
        for split in SPLITS
    }
    prevalence = [count / total_rows for count in all_positive_counts]
    rng = random.Random(seed)

    ranked_groups = []
    for group in groups:
        positive_counts = [sum(record.labels[index] for record in group) for index in range(5)]
        rarity = sum(
            count / prevalence[index] for index, count in enumerate(positive_counts) if prevalence[index]
        )
        ranked_groups.append((rarity, len(group), rng.random(), positive_counts, group))
    ranked_groups.sort(key=lambda item: (-item[0], -item[1], item[2]))

    assignments: dict[str, list[EssayRecord]] = {split: [] for split in SPLITS}
    assigned_rows = {split: 0 for split in SPLITS}
    assigned_labels = {split: [0] * len(TRAIT_COLUMNS) for split in SPLITS}

    for _, group_size, _, group_labels, group in ranked_groups:
        candidates = [
            split for split in SPLITS if assigned_rows[split] + group_size <= row_targets[split]
        ]
        if not candidates:
            candidates = list(SPLITS)

        def score(split: str) -> tuple[float, float, float]:
            label_need = sum(
                max(label_targets[split][index] - assigned_labels[split][index], 0)
                / max(label_targets[split][index], 1)
                * group_labels[index]
                for index in range(len(TRAIT_COLUMNS))
            )
            remaining_capacity = (row_targets[split] - assigned_rows[split]) / max(row_targets[split], 1)
            overflow = max(assigned_rows[split] + group_size - row_targets[split], 0)
            return (label_need, remaining_capacity, -overflow)

        best_score = max(score(split) for split in candidates)
        best_splits = [split for split in candidates if score(split) == best_score]
        selected = rng.choice(best_splits)
        assignments[selected].extend(group)
        assigned_rows[selected] += group_size
        assigned_labels[selected] = [
            assigned_labels[selected][index] + group_labels[index]
            for index in range(len(TRAIT_COLUMNS))
        ]
        for record in group:
            record.split = selected

    if any(not assignments[split] for split in SPLITS):
        raise PreparationError("A split is empty; adjust ratios or review grouping.")
    return assignments


def mean_and_median(values: Iterable[int]) -> dict[str, float | int]:
    values = list(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
    }


def pearson(first: list[int], second: list[int]) -> float | None:
    first_mean, second_mean = statistics.mean(first), statistics.mean(second)
    numerator = sum((x - first_mean) * (y - second_mean) for x, y in zip(first, second))
    denominator = math.sqrt(
        sum((x - first_mean) ** 2 for x in first) * sum((y - second_mean) ** 2 for y in second)
    )
    return None if denominator == 0 else round(numerator / denominator, 4)


def format_correlation(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def make_summary(
    records: list[EssayRecord],
    exclusions: list[dict[str, str]],
    header: list[str],
    input_path: Path,
    groups: list[list[EssayRecord]],
    assignments: dict[str, list[EssayRecord]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, object]:
    model_text_counts = Counter(record.text_model for record in records)
    raw_text_counts = Counter(record.text_raw for record in records)
    label_counts = {
        trait: sum(record.labels[index] for record in records)
        for index, trait in enumerate(TRAIT_COLUMNS)
    }
    per_split = {}
    for split, split_records in assignments.items():
        per_split[split] = {
            "row_count": len(split_records),
            "positive_label_counts": {
                trait: sum(record.labels[index] for record in split_records)
                for index, trait in enumerate(TRAIT_COLUMNS)
            },
        }
    correlations = {
        first: {
            second: pearson(
                [record.labels[first_index] for record in records],
                [record.labels[second_index] for record in records],
            )
            for second_index, second in enumerate(TRAIT_COLUMNS)
        }
        for first_index, first in enumerate(TRAIT_COLUMNS)
    }
    return {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "dataset": "essays",
        "source_file": input_path.as_posix(),
        "source_sha256": sha256(input_path),
        "schema": {"columns": header, "validated_required_columns": list(REQUIRED_COLUMNS)},
        "preparation_policy": {
            "text_cleaning": "NFC Unicode normalisation, C0-control replacement, whitespace collapse; casing and punctuation retained.",
            "exclusion_rule": "Exclude only records with empty text after conservative normalisation.",
            "split_grouping": "#AUTHID plus exact text_model duplicates, preventing author and exact-text leakage.",
            "split_ratios": ratios,
            "random_seed": seed,
        },
        "row_counts": {
            "raw_rows": len(records) + len(exclusions),
            "eligible_rows": len(records),
            "excluded_rows": len(exclusions),
            "unique_authors": len({record.author_id for record in records}),
            "leakage_safe_groups": len(groups),
        },
        "text_quality": {
            "character_count": mean_and_median(record.char_count for record in records),
            "token_count": mean_and_median(record.token_count for record in records),
            "short_text_token_threshold": SHORT_TEXT_TOKEN_THRESHOLD,
            "records_below_short_text_threshold": sum(
                record.token_count < SHORT_TEXT_TOKEN_THRESHOLD for record in records
            ),
            "records_with_replacement_character": sum(
                record.replacement_character_count > 0 for record in records
            ),
            "replacement_character_total": sum(
                record.replacement_character_count for record in records
            ),
            "control_characters_replaced": sum(
                record.control_characters_replaced for record in records
            ),
            "duplicate_raw_text_records": sum(count - 1 for count in raw_text_counts.values() if count > 1),
            "duplicate_model_text_records": sum(
                count - 1 for count in model_text_counts.values() if count > 1
            ),
        },
        "labels": {
            "positive_counts": label_counts,
            "positive_rates": {
                trait: round(count / len(records), 4) for trait, count in label_counts.items()
            },
            "pearson_correlation": correlations,
        },
        "splits": per_split,
    }


def write_csv(path: Path, fieldnames: tuple[str, ...] | list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def record_to_audit_row(record: EssayRecord) -> dict[str, object]:
    row: dict[str, object] = {
        "source_row_number": record.source_row_number,
        "author_id": record.author_id,
        "split_group_id": record.split_group_id,
        "record_id": record.record_id,
        "text_raw": record.text_raw,
        "text_model": record.text_model,
        "char_count": record.char_count,
        "token_count": record.token_count,
        "replacement_character_count": record.replacement_character_count,
        "control_characters_replaced": record.control_characters_replaced,
        "split": record.split,
    }
    row.update(dict(zip(TRAIT_COLUMNS, record.labels)))
    return row


def record_to_processed_row(record: EssayRecord) -> dict[str, object]:
    row: dict[str, object] = {"record_id": record.record_id, "text_model": record.text_model}
    row.update(dict(zip(TRAIT_COLUMNS, record.labels)))
    row["split"] = record.split
    return row


def render_report(summary: dict[str, object]) -> str:
    row_counts = summary["row_counts"]
    text_quality = summary["text_quality"]
    labels = summary["labels"]
    policy = summary["preparation_policy"]
    lines = [
        "# Essays Dataset Quality Report",
        "",
        "## Preparation decision",
        "",
        f"- Raw source: `{summary['source_file']}`",
        f"- SHA-256: `{summary['source_sha256']}`",
        f"- Random seed: `{policy['random_seed']}`",
        f"- Split ratios: `{policy['split_ratios']}`",
        f"- Text handling: {policy['text_cleaning']}",
        f"- Exclusion rule: {policy['exclusion_rule']}",
        f"- Leakage control: {policy['split_grouping']}",
        "",
        "## Dataset profile",
        "",
        f"- Raw rows: {row_counts['raw_rows']}",
        f"- Eligible rows: {row_counts['eligible_rows']}",
        f"- Excluded rows: {row_counts['excluded_rows']}",
        f"- Unique authors: {row_counts['unique_authors']}",
        f"- Leakage-safe groups: {row_counts['leakage_safe_groups']}",
        f"- Exact duplicate raw texts beyond the first occurrence: {text_quality['duplicate_raw_text_records']}",
        f"- Exact duplicate cleaned texts beyond the first occurrence: {text_quality['duplicate_model_text_records']}",
        f"- Records containing U+FFFD replacement characters: {text_quality['records_with_replacement_character']}",
        f"- Control characters replaced in model text: {text_quality['control_characters_replaced']}",
        f"- Records below {text_quality['short_text_token_threshold']} tokens (flagged, not excluded): {text_quality['records_below_short_text_threshold']}",
        "",
        "## Text length",
        "",
        "| Measure | Min | Median | Mean | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Characters | {text_quality['character_count']['min']} | {text_quality['character_count']['median']} | {text_quality['character_count']['mean']} | {text_quality['character_count']['max']} |",
        f"| Tokens | {text_quality['token_count']['min']} | {text_quality['token_count']['median']} | {text_quality['token_count']['mean']} | {text_quality['token_count']['max']} |",
        "",
        "## Big Five label prevalence",
        "",
        "| Trait | Positive count | Positive rate |",
        "| --- | ---: | ---: |",
    ]
    for trait, count in labels["positive_counts"].items():
        lines.append(f"| {trait} | {count} | {labels['positive_rates'][trait]:.2%} |")
    lines.extend(
        [
            "",
            "## Big Five label correlations",
            "",
            "| Trait | cEXT | cNEU | cAGR | cCON | cOPN |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for trait in TRAIT_COLUMNS:
        lines.append(
            f"| {trait} | "
            + " | ".join(
                format_correlation(labels["pearson_correlation"][trait][other])
                for other in TRAIT_COLUMNS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Fixed splits",
            "",
            "| Split | Rows | cEXT | cNEU | cAGR | cCON | cOPN |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in SPLITS:
        split_summary = summary["splits"][split]
        counts = split_summary["positive_label_counts"]
        lines.append(
            f"| {split} | {split_summary['row_count']} | "
            + " | ".join(str(counts[trait]) for trait in TRAIT_COLUMNS)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Next-step guardrails",
            "",
            "- Use only the processed split files for Model 1 training and evaluation.",
            "- Do not use `author_id`, `source_row_number`, `split_group_id` or `text_raw` as model features.",
            "- Treat the test split as held out until the final evaluation.",
            "- Verify the Essays dataset source and licence in the provenance register before analysis or redistribution.",
            "",
        ]
    )
    return "\n".join(lines)


def prepare(
    input_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    report_path: Path,
    summary_path: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, object]:
    ratios = {"train": train_ratio, "validation": validation_ratio, "test": test_ratio}
    if any(value <= 0 for value in ratios.values()) or not math.isclose(sum(ratios.values()), 1.0):
        raise PreparationError("Split ratios must be positive and sum to 1.0.")
    records, exclusions, header = read_essays(input_path)
    groups = build_groups(records)
    assignments = stratified_group_split(groups, ratios, seed)
    summary = make_summary(records, exclusions, header, input_path, groups, assignments, ratios, seed)

    audit_columns = (
        "source_row_number",
        "author_id",
        "split_group_id",
        "record_id",
        "text_raw",
        "text_model",
        "char_count",
        "token_count",
        "replacement_character_count",
        "control_characters_replaced",
        *TRAIT_COLUMNS,
        "split",
    )
    write_csv(interim_dir / "essays_cleaned_audit.csv", audit_columns, (record_to_audit_row(record) for record in records))
    write_csv(
        interim_dir / "essays_exclusions.csv",
        ("source_row_number", "author_id", "exclusion_reason"),
        exclusions,
    )
    for split in SPLITS:
        write_csv(
            processed_dir / f"essays_{split}.csv",
            PROCESSED_COLUMNS,
            (record_to_processed_row(record) for record in assignments[split]),
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/essays/essay_training.csv"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim/essays"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/model1"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data-quality/essays-data-quality-report.md")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("data/metadata/essays-preparation-summary.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare(
        input_path=args.input,
        interim_dir=args.interim_dir,
        processed_dir=args.processed_dir,
        report_path=args.report,
        summary_path=args.summary,
        seed=args.seed,
    )
    print(
        f"Prepared {summary['row_counts']['eligible_rows']} Essays records across "
        f"{', '.join(f'{split}={summary['splits'][split]['row_count']}' for split in SPLITS)}."
    )


if __name__ == "__main__":
    main()
