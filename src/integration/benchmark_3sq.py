"""Benchmark the assessment output against paired 3SQ participant data.

The 3SQ is an independent self-report comparator, not a replacement for the
runtime models. This module requires participant-level paired rows and a
predeclared holdout split; it never treats synthetic fixture responses as
benchmark ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np


MODEL_SKILLS = (
    "self_management",
    "social_engagement",
    "cooperation",
    "emotional_resilience",
    "innovation",
)
THREE_SQ_DIMENSIONS = (
    "self_confidence",
    "curiosity",
    "resilience",
    "openness",
    "collaboration",
    "empathy",
    "leadership",
    "commitment",
    "autonomy",
    "problem_solving",
)
MODEL_COLUMNS = tuple(f"model_{skill}" for skill in MODEL_SKILLS)
THREE_SQ_COLUMNS = tuple(f"three_sq_{dimension}" for dimension in THREE_SQ_DIMENSIONS)
PAIRED_FIELDS = ("participant_id", "split", *MODEL_COLUMNS, *THREE_SQ_COLUMNS)
THREE_SQ_REFERENCE_URL = "https://doi.org/10.7358/ecps-2026-033-mera"

CROSSWALK: dict[str, tuple[str, ...]] = {
    "self_management": ("commitment", "autonomy"),
    "social_engagement": ("openness", "empathy", "leadership"),
    "cooperation": ("collaboration",),
    "emotional_resilience": ("resilience",),
    "innovation": ("curiosity", "problem_solving"),
}
EXPLORATORY_SKILLS = {"social_engagement"}


class Benchmark3SQError(ValueError):
    """Raised when paired benchmark data or its analysis is invalid."""


@dataclass(frozen=True)
class PairedParticipant:
    participant_id: str
    model_scores: tuple[float, float, float, float, float]
    three_sq_scores: tuple[float, float, float, float, float, float, float, float, float, float]

    def model_mapping(self) -> dict[str, float]:
        return dict(zip(MODEL_SKILLS, self.model_scores, strict=True))

    def three_sq_mapping(self) -> dict[str, float]:
        return dict(zip(THREE_SQ_DIMENSIONS, self.three_sq_scores, strict=True))


def _number(value: str, field: str, lower: float, upper: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise Benchmark3SQError(f"{field} must be numeric.") from error
    if not isfinite(result) or not lower <= result <= upper:
        raise Benchmark3SQError(f"{field} must be finite and between {lower:g} and {upper:g}.")
    return result


def load_paired_dataset(path: Path, *, split: str = "test") -> tuple[PairedParticipant, ...]:
    """Load participant-level model/3SQ pairs from one locked split."""
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise Benchmark3SQError(f"Cannot open paired benchmark data {path}.") from error
    with stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != PAIRED_FIELDS:
            raise Benchmark3SQError(f"{path} must contain exactly these columns: {', '.join(PAIRED_FIELDS)}.")
        rows = list(reader)
    if not rows:
        raise Benchmark3SQError("The paired benchmark dataset contains no rows.")
    participants: list[PairedParticipant] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        participant_id = row["participant_id"].strip()
        if not participant_id:
            raise Benchmark3SQError(f"Row {row_number} has an empty participant_id.")
        if participant_id in seen:
            raise Benchmark3SQError(f"Participant {participant_id!r} appears more than once.")
        seen.add(participant_id)
        if row["split"] != split:
            raise Benchmark3SQError(f"Row {row_number} is not in the requested {split!r} split.")
        try:
            model_scores = tuple(_number(row[column], column, 1.0, 5.0) for column in MODEL_COLUMNS)
            three_sq_scores = tuple(_number(row[column], column, 1.0, 5.0) for column in THREE_SQ_COLUMNS)
        except Benchmark3SQError as error:
            raise Benchmark3SQError(f"Invalid paired benchmark row {row_number}: {error}") from error
        participants.append(PairedParticipant(participant_id, model_scores, three_sq_scores))
    if len(participants) < 3:
        raise Benchmark3SQError("At least three paired participants are required for benchmark metrics.")
    return tuple(participants)


def crosswalk_scores(participant: PairedParticipant) -> dict[str, float]:
    """Aggregate mapped 3SQ dimensions into the five comparison outcomes."""
    dimensions = participant.three_sq_mapping()
    return {skill: float(np.mean([dimensions[dimension] for dimension in mapped])) for skill, mapped in CROSSWALK.items()}


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _pearson(model: np.ndarray, reference: np.ndarray) -> float | None:
    centered_model = model - np.mean(model)
    centered_reference = reference - np.mean(reference)
    denominator = sqrt(float(np.sum(centered_model**2) * np.sum(centered_reference**2)))
    return None if denominator == 0 else float(np.sum(centered_model * centered_reference) / denominator)


def _ccc(model: np.ndarray, reference: np.ndarray) -> float | None:
    model_mean = float(np.mean(model))
    reference_mean = float(np.mean(reference))
    covariance = float(np.mean((model - model_mean) * (reference - reference_mean)))
    denominator = float(np.var(model) + np.var(reference) + (model_mean - reference_mean) ** 2)
    return None if denominator == 0 else 2.0 * covariance / denominator


def _metric_values(model: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    return {
        "pearson_r": _pearson(model, reference),
        "spearman_rho": _pearson(_rank(model), _rank(reference)),
        "mae": float(np.mean(np.abs(model - reference))),
        "rmse": float(np.sqrt(np.mean((model - reference) ** 2))),
        "bias_model_minus_3sq": float(np.mean(model - reference)),
        "concordance_correlation": _ccc(model, reference),
    }


def _bootstrap_intervals(
    model: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> dict[str, list[float] | None]:
    if resamples < 100:
        raise Benchmark3SQError("bootstrap resamples must be at least 100.")
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {name: [] for name in _metric_values(model, reference)}
    for _ in range(resamples):
        indices = rng.integers(0, len(model), size=len(model))
        metrics = _metric_values(model[indices], reference[indices])
        for name, value in metrics.items():
            if value is not None and isfinite(value):
                values[name].append(value)
    return {
        name: [round(float(np.percentile(samples, 2.5)), 4), round(float(np.percentile(samples, 97.5)), 4)] if samples else None
        for name, samples in values.items()
    }


def benchmark_metrics(
    participants: Iterable[PairedParticipant], *, seed: int = 42, bootstrap_resamples: int = 1000
) -> dict[str, object]:
    """Compute paired association, error, agreement, and bootstrap intervals."""
    rows = tuple(participants)
    if len(rows) < 3:
        raise Benchmark3SQError("At least three paired participants are required for benchmark metrics.")
    model_by_skill = {skill: np.asarray([row.model_mapping()[skill] for row in rows], dtype=float) for skill in MODEL_SKILLS}
    reference_by_skill = {skill: np.asarray([crosswalk_scores(row)[skill] for row in rows], dtype=float) for skill in MODEL_SKILLS}
    result: dict[str, object] = {}
    for index, skill in enumerate(MODEL_SKILLS):
        estimates = _metric_values(model_by_skill[skill], reference_by_skill[skill])
        intervals = _bootstrap_intervals(model_by_skill[skill], reference_by_skill[skill], seed=seed + index, resamples=bootstrap_resamples)
        result[skill] = {
            "model_column": MODEL_COLUMNS[index],
            "reference_dimensions": list(CROSSWALK[skill]),
            "interpretation": "exploratory crosswalk" if skill in EXPLORATORY_SKILLS else "predeclared crosswalk",
            "metrics": {
                name: None if value is None else round(float(value), 4) for name, value in estimates.items()
            },
            "bootstrap_ci_95": intervals,
        }
    return result


def build_report(
    participants: tuple[PairedParticipant, ...], *, seed: int = 42, bootstrap_resamples: int = 1000
) -> dict[str, object]:
    return {
        "status": "completed",
        "benchmark": {
            "name": "3SQ higher-education self-report comparator",
            "paper": "Rubat du Mérac, Botta and Lupo (2026), Extending the Validation of the 3SQ to Higher Education",
            "reference_url": THREE_SQ_REFERENCE_URL,
            "dimensions": list(THREE_SQ_DIMENSIONS),
        },
        "protocol": {
            "unit_of_analysis": "participant",
            "paired_participants": len(participants),
            "synthetic_ground_truth": False,
            "bootstrap_seed": seed,
            "bootstrap_resamples": bootstrap_resamples,
            "held_out_participant_split_required": True,
        },
        "crosswalk": {skill: list(dimensions) for skill, dimensions in CROSSWALK.items()},
        "metrics": benchmark_metrics(participants, seed=seed, bootstrap_resamples=bootstrap_resamples),
        "limitations": [
            "3SQ is a self-report comparator, not an observed-behaviour gold standard.",
            "The crosswalk is construct-level and is not an item-level equivalence claim.",
            "Social engagement uses an exploratory mapping to openness, empathy, and leadership.",
            "Synthetic fixture responses must not be used as benchmark ground truth.",
        ],
    }


def render_report(report: dict[str, object]) -> str:
    lines = [
        "# 3SQ Benchmark Report",
        "",
        "This compares the runtime soft-skill outputs with an independent 3SQ self-report measure on paired participants.",
        "It is not evidence of observed-behaviour validity.",
        "",
        f"Paired participants: **{report['protocol']['paired_participants']}**",
        "",
        "| Skill | 3SQ dimensions | Pearson r | Spearman rho | MAE | RMSE | CCC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for skill, details in report["metrics"].items():
        metrics = details["metrics"]
        dimensions = ", ".join(details["reference_dimensions"])
        lines.append(
            f"| {skill} | {dimensions} | {metrics['pearson_r']} | {metrics['spearman_rho']} | "
            f"{metrics['mae']} | {metrics['rmse']} | {metrics['concordance_correlation']} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def run_benchmark(
    paired_data: Path,
    *,
    split: str = "test",
    seed: int = 42,
    bootstrap_resamples: int = 1000,
    output_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, object]:
    participants = load_paired_dataset(paired_data, split=split)
    report = build_report(participants, seed=seed, bootstrap_resamples=bootstrap_resamples)
    report["protocol"]["source_path"] = str(paired_data)
    report["protocol"]["split"] = split
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = run_benchmark(
        args.paired_data,
        split=args.split,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
        output_path=args.output,
        report_path=args.report,
    )
    if args.output or args.report:
        print(f"3SQ benchmark completed for {result['protocol']['paired_participants']} paired participants.")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
