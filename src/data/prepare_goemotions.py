"""Prepare agreement-filtered GoEmotions data for Model 1 emotion features.

The script aggregates the raw rater-level CSV shards with the published
two-rater agreement rule, verifies the result against Google Research's
published split files, and writes model-ready multi-label CSVs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


EMOTION_COLUMNS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
)
COMMENT_METADATA_COLUMNS = (
    "text", "id", "author", "subreddit", "link_id", "parent_id", "created_utc",
)
METADATA_COLUMNS = (
    "text", "id", "author", "subreddit", "link_id", "parent_id", "created_utc", "rater_id",
    "example_very_unclear",
)
REQUIRED_COLUMNS = (*METADATA_COLUMNS, *EMOTION_COLUMNS)
SPLITS = ("train", "validation", "test")
OFFICIAL_FILENAMES = {"train": "train.tsv", "validation": "validation.tsv", "test": "test.tsv"}
PROCESSED_COLUMNS = ("record_id", "text_model", *EMOTION_COLUMNS, "split")
SHORT_TEXT_TOKEN_THRESHOLD = 3


class PreparationError(ValueError):
    """Raised when raw data or official split files violate the contract."""


@dataclass
class CommentAggregate:
    comment_id: str
    text_raw: str
    link_id: str
    metadata_signature: tuple[str, ...]
    source_shards: set[str] = field(default_factory=set)
    annotation_count: int = 0
    unclear_annotation_count: int = 0
    label_votes: list[int] = field(default_factory=lambda: [0] * len(EMOTION_COLUMNS))
    metadata_conflict_count: int = 0
    unlabeled_clear_annotation_count: int = 0
    unclear_labeled_annotation_count: int = 0
    text_model: str = ""
    token_count: int = 0
    replacement_character_count: int = 0
    control_characters_replaced: int = 0
    labels: tuple[int, ...] = ()
    official_split: str = ""
    split_group_id: str = ""
    split: str = ""
    record_id: str = ""


def parse_boolean(value: str, field_name: str, source: str) -> bool:
    normalised = value.strip().lower()
    if normalised == "true":
        return True
    if normalised == "false":
        return False
    raise PreparationError(f"{source}: {field_name} must be True or False, got {value!r}.")


def parse_binary(value: str, field_name: str, source: str) -> int:
    normalised = value.strip()
    if normalised not in {"0", "1"}:
        raise PreparationError(f"{source}: {field_name} must be 0 or 1, got {value!r}.")
    return int(normalised)


def clean_text(text: str) -> tuple[str, int]:
    text = unicodedata.normalize("NFC", text)
    replaced_controls = sum(
        1
        for character in text
        if unicodedata.category(character) == "Cc" and character not in "\n\r\t"
    )
    text = "".join(
        " " if unicodedata.category(character) == "Cc" and character not in "\n\r\t" else character
        for character in text
    )
    return re.sub(r"\s+", " ", text).strip(), replaced_controls


def count_tokens(text: str) -> int:
    return len(re.findall(r"\b\w+(?:['’]\w+)?\b", text, flags=re.UNICODE))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw_annotations(raw_dir: Path) -> tuple[dict[str, CommentAggregate], dict[str, int], int]:
    csv.field_size_limit(sys.maxsize)
    source_paths = sorted(raw_dir.glob("goemotions_*.csv"))
    if not source_paths:
        raise PreparationError(f"No GoEmotions CSV shards found in {raw_dir}.")
    comments: dict[str, CommentAggregate] = {}
    shard_rows: dict[str, int] = {}
    annotation_rows = 0

    for path in source_paths:
        row_count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames or []
            if header != list(REQUIRED_COLUMNS):
                missing = [column for column in REQUIRED_COLUMNS if column not in header]
                unexpected = [column for column in header if column not in REQUIRED_COLUMNS]
                raise PreparationError(
                    f"{path}: schema mismatch; missing={missing or 'none'}, unexpected={unexpected or 'none'}."
                )
            for row_number, row in enumerate(reader, start=2):
                row_count += 1
                annotation_rows += 1
                source = f"{path.name} row {row_number}"
                comment_id = (row["id"] or "").strip()
                text_raw = row["text"] or ""
                if not comment_id or not text_raw:
                    raise PreparationError(f"{source}: id and text must both be non-empty.")
                labels = [parse_binary(row[label], label, source) for label in EMOTION_COLUMNS]
                very_unclear = parse_boolean(row["example_very_unclear"], "example_very_unclear", source)
                metadata_signature = tuple((row[column] or "") for column in COMMENT_METADATA_COLUMNS)
                aggregate = comments.get(comment_id)
                if aggregate is None:
                    text_model, replaced_controls = clean_text(text_raw)
                    aggregate = CommentAggregate(
                        comment_id=comment_id,
                        text_raw=text_raw,
                        link_id=row["link_id"] or "",
                        metadata_signature=metadata_signature,
                        text_model=text_model,
                        token_count=count_tokens(text_model),
                        replacement_character_count=text_raw.count("\ufffd"),
                        control_characters_replaced=replaced_controls,
                    )
                    comments[comment_id] = aggregate
                elif aggregate.metadata_signature != metadata_signature:
                    aggregate.metadata_conflict_count += 1
                aggregate.source_shards.add(path.name)
                aggregate.annotation_count += 1
                aggregate.unclear_annotation_count += int(very_unclear)
                aggregate.unlabeled_clear_annotation_count += int(not very_unclear and not any(labels))
                aggregate.unclear_labeled_annotation_count += int(very_unclear and any(labels))
                aggregate.label_votes = [
                    total + label for total, label in zip(aggregate.label_votes, labels)
                ]
        shard_rows[path.name] = row_count
    return comments, shard_rows, annotation_rows


def apply_agreement_filter(
    comments: dict[str, CommentAggregate], minimum_votes: int = 2
) -> tuple[list[CommentAggregate], list[dict[str, object]]]:
    included: list[CommentAggregate] = []
    exclusions: list[dict[str, object]] = []
    for aggregate in comments.values():
        aggregate.labels = tuple(int(votes >= minimum_votes) for votes in aggregate.label_votes)
        aggregate.record_id = "goemotion_" + hashlib.sha256(
            f"goemotions:{aggregate.comment_id}".encode("utf-8")
        ).hexdigest()[:16]
        if not aggregate.text_model:
            exclusions.append({"comment_id": aggregate.comment_id, "exclusion_reason": "empty_text_after_normalisation"})
        elif not any(aggregate.labels):
            exclusions.append({"comment_id": aggregate.comment_id, "exclusion_reason": "no_two_rater_label_agreement"})
        else:
            included.append(aggregate)
    return included, exclusions


def read_official_splits(official_splits_dir: Path) -> dict[str, dict[str, tuple[str, tuple[int, ...]]]]:
    official: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {}
    seen_ids: set[str] = set()
    for split, filename in OFFICIAL_FILENAMES.items():
        path = official_splits_dir / filename
        if not path.is_file():
            raise PreparationError(f"Official split reference is missing: {path}")
        split_rows: dict[str, tuple[str, tuple[int, ...]]] = {}
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, delimiter="\t")
            for row_number, row in enumerate(reader, start=1):
                if len(row) != 3:
                    raise PreparationError(f"{path} row {row_number}: expected three TSV columns.")
                text, label_ids, comment_id = row
                if comment_id in seen_ids:
                    raise PreparationError(f"Official split files repeat comment id {comment_id!r}.")
                label_vector = [0] * len(EMOTION_COLUMNS)
                for label_id in label_ids.split(","):
                    index = int(label_id)
                    if index < 0 or index >= len(EMOTION_COLUMNS):
                        raise PreparationError(f"{path} row {row_number}: invalid emotion id {index}.")
                    label_vector[index] = 1
                split_rows[comment_id] = (text, tuple(label_vector))
                seen_ids.add(comment_id)
        official[split] = split_rows
    return official


def verify_and_assign_official_splits(
    included: list[CommentAggregate], official: dict[str, dict[str, tuple[str, tuple[int, ...]]]]
) -> int:
    prepared_by_id = {aggregate.comment_id: aggregate for aggregate in included}
    official_ids = {comment_id for rows in official.values() for comment_id in rows}
    missing_from_prepared = official_ids - prepared_by_id.keys()
    extra_in_prepared = prepared_by_id.keys() - official_ids
    if missing_from_prepared or extra_in_prepared:
        raise PreparationError(
            "Agreement-filtered comments do not match the official split IDs: "
            f"missing={len(missing_from_prepared)}, extra={len(extra_in_prepared)}."
        )
    for split, split_rows in official.items():
        for comment_id, (official_text, official_labels) in split_rows.items():
            aggregate = prepared_by_id[comment_id]
            if aggregate.labels != official_labels:
                raise PreparationError(f"Label aggregation mismatch for GoEmotions comment {comment_id!r}.")
            if clean_text(official_text)[0] != aggregate.text_model:
                raise PreparationError(f"Text mismatch for GoEmotions comment {comment_id!r}.")
            aggregate.official_split = split

    duplicate_splits: dict[str, set[str]] = defaultdict(set)
    for aggregate in included:
        duplicate_splits[aggregate.text_model].add(aggregate.official_split)
    collisions = sum(1 for splits in duplicate_splits.values() if len(splits) > 1)
    return collisions


def target_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: total * ratios[split] for split in SPLITS}
    counts = {split: math.floor(raw[split]) for split in SPLITS}
    remainder = total - sum(counts.values())
    for split in sorted(SPLITS, key=lambda item: (raw[item] - counts[item], item), reverse=True)[:remainder]:
        counts[split] += 1
    return counts


def assign_leakage_safe_splits(
    included: list[CommentAggregate], ratios: dict[str, float], seed: int
) -> int:
    """Assign exact-text groups while approximately preserving 28 label rates."""
    groups_by_text: dict[str, list[CommentAggregate]] = defaultdict(list)
    for aggregate in included:
        groups_by_text[aggregate.text_model].append(aggregate)
    groups = list(groups_by_text.values())
    for group in groups:
        fingerprint = "|".join(sorted(item.record_id for item in group))
        group_id = "group_" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
        for item in group:
            item.split_group_id = group_id

    row_targets = target_counts(len(included), ratios)
    total_positive_counts = [
        sum(item.labels[index] for item in included) for index in range(len(EMOTION_COLUMNS))
    ]
    label_targets = {
        split: [count * ratios[split] for count in total_positive_counts] for split in SPLITS
    }
    prevalence = [count / len(included) for count in total_positive_counts]
    import random

    rng = random.Random(seed)
    ranked_groups = []
    for group in groups:
        group_labels = [sum(item.labels[index] for item in group) for index in range(len(EMOTION_COLUMNS))]
        rarity = sum(
            count / prevalence[index] for index, count in enumerate(group_labels) if prevalence[index]
        )
        ranked_groups.append((rarity, len(group), rng.random(), group_labels, group))
    ranked_groups.sort(key=lambda item: (-item[0], -item[1], item[2]))

    assigned_rows = {split: 0 for split in SPLITS}
    assigned_labels = {split: [0] * len(EMOTION_COLUMNS) for split in SPLITS}
    for _, group_size, _, group_labels, group in ranked_groups:
        candidates = [
            split for split in SPLITS if assigned_rows[split] + group_size <= row_targets[split]
        ] or list(SPLITS)

        def score(split: str) -> tuple[float, float, float]:
            label_need = sum(
                max(label_targets[split][index] - assigned_labels[split][index], 0)
                / max(label_targets[split][index], 1)
                * group_labels[index]
                for index in range(len(EMOTION_COLUMNS))
            )
            remaining_capacity = (row_targets[split] - assigned_rows[split]) / max(row_targets[split], 1)
            overflow = max(assigned_rows[split] + group_size - row_targets[split], 0)
            return label_need, remaining_capacity, -overflow

        best_score = max(score(split) for split in candidates)
        selected = rng.choice([split for split in candidates if score(split) == best_score])
        assigned_rows[selected] += group_size
        assigned_labels[selected] = [
            assigned_labels[selected][index] + group_labels[index]
            for index in range(len(EMOTION_COLUMNS))
        ]
        for item in group:
            item.split = selected
    if any(not any(item.split == split for item in included) for split in SPLITS):
        raise PreparationError("A custom GoEmotions split is empty; adjust split ratios.")
    return len(groups)


def length_summary(values: Iterable[int]) -> dict[str, float | int]:
    values = list(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
    }


def make_summary(
    included: list[CommentAggregate],
    exclusions: list[dict[str, object]],
    comments: dict[str, CommentAggregate],
    shard_rows: dict[str, int],
    annotation_rows: int,
    raw_dir: Path,
    official_collisions: int,
    leakage_safe_groups: int,
    split_ratios: dict[str, float],
    seed: int,
) -> dict[str, object]:
    split_records = {split: [record for record in included if record.split == split] for split in SPLITS}
    label_counts = {
        label: sum(record.labels[index] for record in included)
        for index, label in enumerate(EMOTION_COLUMNS)
    }
    ratings_per_comment = [aggregate.annotation_count for aggregate in comments.values()]
    return {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "dataset": "goemotions",
        "source_files": {
            path.name: {"sha256": sha256(path), "row_count": shard_rows[path.name]}
            for path in sorted(raw_dir.glob("goemotions_*.csv"))
        },
        "aggregation_policy": {
            "unit": "comment ID",
            "rule": "Set each emotion to 1 when at least two raters selected it.",
            "minimum_rater_votes": 2,
            "validation": "Exact comment, text and label agreement with the official Google Research train/dev/test split files.",
            "text_cleaning": "NFC Unicode normalisation, C0-control replacement and whitespace collapse; casing and punctuation retained.",
        },
        "split_policy": {
            "rule": "Custom fixed split grouped by exact text_model, preventing exact-text leakage.",
            "split_ratios": split_ratios,
            "random_seed": seed,
            "official_split_exact_text_collisions": official_collisions,
        },
        "row_counts": {
            "raw_annotation_rows": annotation_rows,
            "unique_comments": len(comments),
            "agreement_filtered_comments": len(included),
            "excluded_comments": len(exclusions),
            "leakage_safe_groups": leakage_safe_groups,
            "custom_split_rows": {split: len(split_records[split]) for split in SPLITS},
        },
        "annotation_quality": {
            "ratings_per_comment": length_summary(ratings_per_comment),
            "very_unclear_annotations": sum(item.unclear_annotation_count for item in comments.values()),
            "unclear_annotations_with_labels": sum(item.unclear_labeled_annotation_count for item in comments.values()),
            "clear_annotations_without_labels": sum(item.unlabeled_clear_annotation_count for item in comments.values()),
            "comments_with_metadata_conflicts": sum(item.metadata_conflict_count > 0 for item in comments.values()),
        },
        "text_quality": {
            "token_count": length_summary(item.token_count for item in included),
            "records_below_short_text_threshold": sum(
                item.token_count < SHORT_TEXT_TOKEN_THRESHOLD for item in included
            ),
            "short_text_token_threshold": SHORT_TEXT_TOKEN_THRESHOLD,
            "records_with_replacement_character": sum(
                item.replacement_character_count > 0 for item in included
            ),
            "replacement_character_total": sum(item.replacement_character_count for item in included),
            "control_characters_replaced": sum(item.control_characters_replaced for item in included),
            "multi_label_records": sum(sum(item.labels) > 1 for item in included),
        },
        "labels": {
            "positive_counts": label_counts,
            "positive_rates": {
                label: round(count / len(included), 5) for label, count in label_counts.items()
            },
        },
        "splits": {
            split: {
                "row_count": len(split_records[split]),
                "positive_label_counts": {
                    label: sum(record.labels[index] for record in split_records[split])
                    for index, label in enumerate(EMOTION_COLUMNS)
                },
            }
            for split in SPLITS
        },
    }


def write_csv(path: Path, fieldnames: tuple[str, ...] | list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def processed_row(aggregate: CommentAggregate) -> dict[str, object]:
    row: dict[str, object] = {
        "record_id": aggregate.record_id,
        "text_model": aggregate.text_model,
    }
    row.update(dict(zip(EMOTION_COLUMNS, aggregate.labels)))
    row["split"] = aggregate.split
    return row


def audit_row(aggregate: CommentAggregate) -> dict[str, object]:
    row: dict[str, object] = {
        "comment_id": aggregate.comment_id,
        "record_id": aggregate.record_id,
        "source_shards": ";".join(sorted(aggregate.source_shards)),
        "annotation_count": aggregate.annotation_count,
        "unclear_annotation_count": aggregate.unclear_annotation_count,
        "metadata_conflict_count": aggregate.metadata_conflict_count,
        "text_raw": aggregate.text_raw,
        "text_model": aggregate.text_model,
        "token_count": aggregate.token_count,
        "replacement_character_count": aggregate.replacement_character_count,
        "control_characters_replaced": aggregate.control_characters_replaced,
        "official_split": aggregate.official_split,
        "split_group_id": aggregate.split_group_id,
        "split": aggregate.split,
    }
    row.update({f"votes_{label}": aggregate.label_votes[index] for index, label in enumerate(EMOTION_COLUMNS)})
    row.update(dict(zip(EMOTION_COLUMNS, aggregate.labels)))
    return row


def render_report(summary: dict[str, object]) -> str:
    rows = summary["row_counts"]
    annotations = summary["annotation_quality"]
    text_quality = summary["text_quality"]
    policy = summary["aggregation_policy"]
    split_policy = summary["split_policy"]
    lines = [
        "# GoEmotions Dataset Quality Report",
        "",
        "## Preparation decision",
        "",
        f"- Unit of analysis: {policy['unit']}",
        f"- Aggregation: {policy['rule']}",
        f"- Validation: {policy['validation']}",
        f"- Text handling: {policy['text_cleaning']}",
        f"- Model split: {split_policy['rule']}",
        f"- Split ratios: {split_policy['split_ratios']} (seed {split_policy['random_seed']})",
        f"- Exact-text collisions in the published split references: {split_policy['official_split_exact_text_collisions']}",
        "",
        "## Dataset profile",
        "",
        f"- Raw annotation rows: {rows['raw_annotation_rows']}",
        f"- Unique comments: {rows['unique_comments']}",
        f"- Agreement-filtered comments: {rows['agreement_filtered_comments']}",
        f"- Excluded comments: {rows['excluded_comments']}",
        f"- Leakage-safe exact-text groups: {rows['leakage_safe_groups']}",
        f"- Ratings per comment: min {annotations['ratings_per_comment']['min']}, median {annotations['ratings_per_comment']['median']}, mean {annotations['ratings_per_comment']['mean']}, max {annotations['ratings_per_comment']['max']}",
        f"- Very-unclear annotations: {annotations['very_unclear_annotations']}",
        f"- Clear annotations with no selected label: {annotations['clear_annotations_without_labels']}",
        f"- Unclear annotations that nevertheless selected a label: {annotations['unclear_annotations_with_labels']}",
        f"- Comments with inconsistent metadata across annotations: {annotations['comments_with_metadata_conflicts']}",
        "",
        "## Text and label quality",
        "",
        f"- Token count: min {text_quality['token_count']['min']}, median {text_quality['token_count']['median']}, mean {text_quality['token_count']['mean']}, max {text_quality['token_count']['max']}",
        f"- Records below {text_quality['short_text_token_threshold']} tokens (flagged, not excluded): {text_quality['records_below_short_text_threshold']}",
        f"- Records containing U+FFFD replacement characters: {text_quality['records_with_replacement_character']}",
        f"- Multi-label records: {text_quality['multi_label_records']}",
        "",
            "## Leakage-safe fixed splits",
        "",
        "| Split | Rows |",
        "| --- | ---: |",
    ]
    for split in SPLITS:
        lines.append(f"| {split} | {rows['custom_split_rows'][split]} |")
    lines.extend(
        [
            "",
            "## Agreement-filtered label prevalence",
            "",
            "| Emotion | Positive count | Positive rate |",
            "| --- | ---: | ---: |",
        ]
    )
    for label, count in summary["labels"]["positive_counts"].items():
        lines.append(f"| {label} | {count} | {summary['labels']['positive_rates'][label]:.2%} |")
    lines.extend(
        [
            "",
            "## Next-step guardrails",
            "",
            "- Train emotion models only on the processed split files; author, subreddit, thread, timestamp and source comment IDs are excluded.",
            "- Choose model thresholds using validation only; keep the test split held out for final emotion-model evaluation.",
            "- Use predicted emotion probabilities—not gold GoEmotions labels—as features when applying the emotion model to Essays.",
            "- GoEmotions captures emotion labels, not emotion regulation; define any regulation feature separately.",
            "- Reddit-derived data is not representative of all populations or writing contexts; report this limitation.",
            "",
        ]
    )
    return "\n".join(lines)


def prepare(
    raw_dir: Path,
    official_splits_dir: Path,
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
    comments, shard_rows, annotation_rows = read_raw_annotations(raw_dir)
    included, exclusions = apply_agreement_filter(comments)
    official = read_official_splits(official_splits_dir)
    official_collisions = verify_and_assign_official_splits(included, official)
    leakage_safe_groups = assign_leakage_safe_splits(included, ratios, seed)
    summary = make_summary(
        included, exclusions, comments, shard_rows, annotation_rows, raw_dir,
        official_collisions, leakage_safe_groups, ratios, seed,
    )

    audit_columns = (
        "comment_id", "record_id", "source_shards", "annotation_count", "unclear_annotation_count",
        "metadata_conflict_count", "text_raw", "text_model", "token_count", "replacement_character_count",
        "control_characters_replaced", *(f"votes_{label}" for label in EMOTION_COLUMNS),
        *EMOTION_COLUMNS, "official_split", "split_group_id", "split",
    )
    write_csv(interim_dir / "goemotions_comment_audit.csv", audit_columns, (audit_row(item) for item in included))
    write_csv(interim_dir / "goemotions_exclusions.csv", ("comment_id", "exclusion_reason"), exclusions)
    for split in SPLITS:
        write_csv(
            processed_dir / f"goemotions_{split}.csv",
            PROCESSED_COLUMNS,
            (processed_row(item) for item in included if item.split == split),
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/goemotions"))
    parser.add_argument(
        "--official-splits-dir", type=Path, default=Path("data/raw/goemotions/official_splits")
    )
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim/goemotions"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/model1"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data-quality/goemotions-data-quality-report.md")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("data/metadata/goemotions-preparation-summary.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare(
        args.raw_dir,
        args.official_splits_dir,
        args.interim_dir,
        args.processed_dir,
        args.report,
        args.summary,
        args.seed,
    )
    split_counts = summary["row_counts"]["custom_split_rows"]
    print(
        f"Prepared {summary['row_counts']['agreement_filtered_comments']} GoEmotions comments across "
        f"train={split_counts['train']}, validation={split_counts['validation']}, test={split_counts['test']}."
    )


if __name__ == "__main__":
    main()
