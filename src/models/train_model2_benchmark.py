"""Train the Model 2 transparent multi-output linear-regression benchmark.

The benchmark is fit on the fixed BFI-2/BESSI training split and is evaluated
on validation data only.  The held-out test set is deliberately excluded until
the Week 4 model-selection decision has been frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_COLUMNS = (
    "big_five_extraversion",
    "big_five_agreeableness",
    "big_five_conscientiousness",
    "big_five_negative_emotionality",
    "big_five_open_mindedness",
)
TARGET_COLUMNS = (
    "bessi_self_management",
    "bessi_social_engagement",
    "bessi_cooperation",
    "bessi_emotional_resilience",
    "bessi_innovation",
)
RECORD_ID_COLUMN = "record_id"


class TrainingError(ValueError):
    """Raised when a prepared Model 2 split violates the training contract."""


def load_split(path: Path, expected_split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load exactly the model-ready columns from one fixed Model 2 split."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {RECORD_ID_COLUMN, "split", *FEATURE_COLUMNS, *TARGET_COLUMNS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise TrainingError(f"{path} does not contain the expected Model 2 schema.")
        rows = list(reader)
    if not rows:
        raise TrainingError(f"{path} has no rows.")
    if any(row["split"] != expected_split for row in rows):
        raise TrainingError(f"{path} contains rows outside the {expected_split!r} split.")
    try:
        features = np.asarray([[float(row[column]) for column in FEATURE_COLUMNS] for row in rows], dtype=float)
        targets = np.asarray([[float(row[column]) for column in TARGET_COLUMNS] for row in rows], dtype=float)
    except ValueError as error:
        raise TrainingError(f"{path} contains a non-numeric model value.") from error
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise TrainingError(f"{path} contains a non-finite model value.")
    return [row[RECORD_ID_COLUMN] for row in rows], features, targets


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    """Return overall and per-domain MAE, RMSE and R-squared."""
    per_target: dict[str, dict[str, float]] = {}
    for index, target in enumerate(TARGET_COLUMNS):
        observed = targets[:, index]
        predicted = predictions[:, index]
        per_target[target] = {
            "mae": round(float(mean_absolute_error(observed, predicted)), 4),
            "rmse": round(float(mean_squared_error(observed, predicted) ** 0.5), 4),
            "r2": round(float(r2_score(observed, predicted)), 4),
        }
    return {
        "mae": round(float(mean_absolute_error(targets, predictions)), 4),
        "rmse": round(float(mean_squared_error(targets, predictions) ** 0.5), 4),
        "r2": round(float(r2_score(targets, predictions, multioutput="uniform_average")), 4),
        "per_target": per_target,
    }


def write_predictions(
    path: Path, record_ids: list[str], targets: np.ndarray, predictions: np.ndarray, split: str = "validation"
) -> None:
    fields = [RECORD_ID_COLUMN, "split"]
    fields += [f"observed_{column}" for column in TARGET_COLUMNS]
    fields += [f"prediction_{column}" for column in TARGET_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row_index, record_id in enumerate(record_ids):
            row: dict[str, object] = {RECORD_ID_COLUMN: record_id, "split": split}
            row.update({f"observed_{target}": f"{targets[row_index, index]:.8f}" for index, target in enumerate(TARGET_COLUMNS)})
            row.update({f"prediction_{target}": f"{predictions[row_index, index]:.8f}" for index, target in enumerate(TARGET_COLUMNS)})
            writer.writerow(row)


def render_report(metadata: dict[str, object]) -> str:
    metrics = metadata["validation_metrics"]
    lines = [
        "# Model 2 Linear-Regression Benchmark Report",
        "",
        "## Research framing",
        "",
        "This is the project's transparent trait-to-skill estimation baseline, rather than a reproduction of an identical published model.",
        "",
        "## Protocol",
        "",
        "- Inputs: the five prepared Big Five domain scores only.",
        "- Outputs: the five prepared BESSI domain scores jointly.",
        "- Model: ordinary least squares multi-output linear regression.",
        "- Fitting: fixed training split only.",
        "- Evaluation below: fixed validation split only. The held-out test split has not been read or used.",
        "- Excluded: audit IDs, demographics, `Case`, source-row data, and raw questionnaire items.",
        "",
        "## Validation results",
        "",
        "| Aggregate MAE | Aggregate RMSE | Mean R² |",
        "| ---: | ---: | ---: |",
        f"| {metrics['mae']:.4f} | {metrics['rmse']:.4f} | {metrics['r2']:.4f} |",
        "",
        "| BESSI domain | MAE | RMSE | R² |",
        "| --- | ---: | ---: | ---: |",
    ]
    for target in TARGET_COLUMNS:
        result = metrics["per_target"][target]
        lines.append(f"| {target} | {result['mae']:.4f} | {result['rmse']:.4f} | {result['r2']:.4f} |")
    lines.extend(
        [
            "",
            "## Limitation",
            "",
            "The source `Case` field is non-unique, so the fixed split is row-level. These results must not be claimed as participant-independent performance.",
            "",
        ]
    )
    return "\n".join(lines)


def train_benchmark(
    data_dir: Path, model_path: Path, prediction_path: Path, report_path: Path, metadata_path: Path
) -> dict[str, object]:
    train_ids, train_features, train_targets = load_split(data_dir / "bfi2_bessi_train.csv", "train")
    validation_ids, validation_features, validation_targets = load_split(
        data_dir / "bfi2_bessi_validation.csv", "validation"
    )
    model = LinearRegression()
    model.fit(train_features, train_targets)
    validation_predictions = model.predict(validation_features)
    validation = regression_metrics(validation_targets, validation_predictions)
    metadata: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "benchmark": "transparent multi-output ordinary least squares linear regression",
        "feature_columns": list(FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "training_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "test_rows_used": 0,
        "model_settings": {"fit_intercept": model.fit_intercept, "positive": model.positive},
        "validation_metrics": validation,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, model_path)
    write_predictions(prediction_path, validation_ids, validation_targets, validation_predictions)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(metadata), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/model2"))
    parser.add_argument("--model", type=Path, default=Path("outputs/models/model2_linear_benchmark.joblib"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/model-evaluation/model2_linear_validation_predictions.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/model-evaluation/model2-linear-benchmark-report.md"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/model2-linear-benchmark.json"))
    args = parser.parse_args()
    metadata = train_benchmark(args.data_dir, args.model, args.predictions, args.report, args.metadata)
    results = metadata["validation_metrics"]
    print(f"Trained Model 2 benchmark on {metadata['training_rows']} rows; validation MAE={results['mae']:.4f}, RMSE={results['rmse']:.4f}, R²={results['r2']:.4f}.")


if __name__ == "__main__":
    main()
