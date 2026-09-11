"""Apply the locked Model 2 mapping to held-out RecruitView predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from src.evaluation.metrics import spearman
from src.models.train_model2_benchmark import FEATURE_COLUMNS, TARGET_COLUMNS, load_split


RECRUITVIEW_TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
MODEL2_ORDER = ("extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness")
MODEL2_TRAIT_KEYS = {
    "extraversion": "extraversion",
    "agreeableness": "agreeableness",
    "conscientiousness": "conscientiousness",
    "neuroticism": "negative_emotionality",
    "openness": "open_mindedness",
}


def run(model1_predictions: Path, prepared_recruitview: Path, model2_data_dir: Path, model2_artifact: Path, output: Path, report_path: Path) -> dict[str, object]:
    with model1_predictions.open("r", encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    with prepared_recruitview.open("r", encoding="utf-8", newline="") as stream:
        prepared = {row["record_id"]: row for row in csv.DictReader(stream)}
    if not predictions:
        raise ValueError("Model 1 prediction file is empty.")
    _, bfi_train, _ = load_split(model2_data_dir / "bfi2_bessi_train.csv", "train")
    model2 = joblib.load(model2_artifact)["model"]
    recruitview_train = np.asarray(
        [[float(row[column]) for column in RECRUITVIEW_TRAITS] for row in prepared.values() if row["split"] == "train"],
        dtype=float,
    )
    recruitview_mean = recruitview_train.mean(axis=0)
    recruitview_std = np.maximum(recruitview_train.std(axis=0), 1e-8)
    bfi_mean = bfi_train.mean(axis=0)
    bfi_std = np.maximum(bfi_train.std(axis=0), 1e-8)
    trait_index = {trait: index for index, trait in enumerate(RECRUITVIEW_TRAITS)}
    bfi_index = {column.removeprefix("big_five_"): index for index, column in enumerate(FEATURE_COLUMNS)}
    mapped = []
    speaking = []
    for row in predictions:
        predicted = np.asarray([float(row[f"prediction_{trait}"]) for trait in RECRUITVIEW_TRAITS], dtype=float)
        standardized = (predicted - recruitview_mean) / recruitview_std
        mapped_row = np.asarray(
            [
                bfi_mean[bfi_index[MODEL2_TRAIT_KEYS[trait]]] + standardized[trait_index[trait]] * bfi_std[bfi_index[MODEL2_TRAIT_KEYS[trait]]]
                for trait in MODEL2_ORDER
            ],
            dtype=float,
        )
        mapped.append(np.clip(mapped_row, 0.0, 1.0))
        speaking.append(float(row["observed_speaking_skills"]))
    mapped_array = np.asarray(mapped)
    bessi_predictions = np.asarray(model2.predict(mapped_array), dtype=float)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["record_id", "participant_id", "split", *[f"mapped_{column}" for column in FEATURE_COLUMNS], *[f"estimated_{column}" for column in TARGET_COLUMNS], "observed_speaking_skills"]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(predictions):
            values = {"record_id": row["record_id"], "participant_id": row["participant_id"], "split": row["split"], "observed_speaking_skills": speaking[index]}
            values.update({f"mapped_{column}": mapped_array[index, j] for j, column in enumerate(FEATURE_COLUMNS)})
            values.update({f"estimated_{column}": bessi_predictions[index, j] for j, column in enumerate(TARGET_COLUMNS)})
            writer.writerow(values)
    proxy = {
        target: spearman(bessi_predictions[:, index], speaking)
        for index, target in enumerate(TARGET_COLUMNS)
    }
    report = {
        "status": "completed",
        "input_model1_predictions": str(model1_predictions),
        "model2_artifact": str(model2_artifact),
        "rows": len(predictions),
        "trait_mapping": "standardise RecruitView training-label predictions, then map to BFI-2 training mean/std and clip to 0–1",
        "estimated_targets": list(TARGET_COLUMNS),
        "speaking_skills_proxy_spearman": proxy,
        "interpretation_boundary": "These are BESSI estimates generated from predicted Big Five traits. RecruitView has no BESSI ground truth; correlations with speaking_skills are proxy/convergent evidence only.",
        "output": str(output),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model1-predictions", type=Path, default=Path("outputs/benchmark/recruitview-feature-fusion/predictions.csv"))
    parser.add_argument("--prepared-recruitview", type=Path, default=Path("data/processed/recruitview/recruitview_prepared.csv"))
    parser.add_argument("--model2-data-dir", type=Path, default=Path("data/processed/model2"))
    parser.add_argument("--model2-artifact", type=Path, default=Path("outputs/models/model2_linear_benchmark.joblib"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark/recruitview-bessi/estimates.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/benchmark/recruitview-bessi-integration.json"))
    args = parser.parse_args()
    result = run(args.model1_predictions, args.prepared_recruitview, args.model2_data_dir, args.model2_artifact, args.output, args.report)
    print(json.dumps({"status": result["status"], "rows": result["rows"]}, indent=2))


if __name__ == "__main__":
    main()
