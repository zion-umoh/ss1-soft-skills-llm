"""Train the Model 2 nonlinear multi-output comparison model.

This is the predeclared stronger comparison to the transparent linear baseline:
an Extra Trees ensemble can model nonlinear and interaction effects between the
five Big Five inputs while predicting the five BESSI domains jointly.  It uses
the same fixed training and validation data; test remains untouched.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

try:  # Supports both `python -m src.models...` and direct script execution.
    from src.models.train_model2_benchmark import (
        FEATURE_COLUMNS,
        TARGET_COLUMNS,
        load_split,
        regression_metrics,
        write_predictions,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by the Makefile command.
    from train_model2_benchmark import (
        FEATURE_COLUMNS,
        TARGET_COLUMNS,
        load_split,
        regression_metrics,
        write_predictions,
    )


SEED = 42


def build_model() -> ExtraTreesRegressor:
    """Return a fixed, reproducible nonlinear model without validation tuning."""
    return ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=1.0,
        random_state=SEED,
        n_jobs=-1,
    )


def render_report(metadata: dict[str, object]) -> str:
    results = metadata["validation_metrics"]
    lines = [
        "# Model 2 Nonlinear Multi-Output Comparison Report",
        "",
        "## Protocol",
        "",
        "- Inputs and outputs: the same five prepared Big Five and five BESSI-domain columns as the linear benchmark.",
        "- Model: Extra Trees multi-output regression with a fixed seed and predeclared settings.",
        "- Fitting: fixed training split only; no validation hyperparameter tuning was performed.",
        "- Evaluation below: fixed validation split only. The held-out test split has not been read or used.",
        "",
        "## Validation results",
        "",
        "| Aggregate MAE | Aggregate RMSE | Mean R² |",
        "| ---: | ---: | ---: |",
        f"| {results['mae']:.4f} | {results['rmse']:.4f} | {results['r2']:.4f} |",
        "",
        "| BESSI domain | MAE | RMSE | R² |",
        "| --- | ---: | ---: | ---: |",
    ]
    for target in TARGET_COLUMNS:
        result = results["per_target"][target]
        lines.append(f"| {target} | {result['mae']:.4f} | {result['rmse']:.4f} | {result['r2']:.4f} |")
    lines.extend(["", "## Limitation", "", "The source `Case` field is non-unique, so the fixed split is row-level. These results are not participant-independent performance.", ""])
    return "\n".join(lines)


def train_improved(
    data_dir: Path, model_path: Path, prediction_path: Path, report_path: Path, metadata_path: Path
) -> dict[str, object]:
    _, train_features, train_targets = load_split(data_dir / "bfi2_bessi_train.csv", "train")
    validation_ids, validation_features, validation_targets = load_split(
        data_dir / "bfi2_bessi_validation.csv", "validation"
    )
    model = build_model()
    model.fit(train_features, train_targets)
    validation_predictions = model.predict(validation_features)
    metadata: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model": "Extra Trees multi-output regression",
        "feature_columns": list(FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "training_rows": len(train_features),
        "validation_rows": len(validation_ids),
        "test_rows_used": 0,
        "model_settings": model.get_params(),
        "validation_metrics": regression_metrics(validation_targets, validation_predictions),
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
    parser.add_argument("--model", type=Path, default=Path("outputs/models/model2_extra_trees.joblib"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/model-evaluation/model2_extra_trees_validation_predictions.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/model-evaluation/model2-extra-trees-report.md"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/model2-extra-trees.json"))
    args = parser.parse_args()
    metadata = train_improved(args.data_dir, args.model, args.predictions, args.report, args.metadata)
    metrics = metadata["validation_metrics"]
    print(f"Trained Model 2 nonlinear model; validation MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}.")


if __name__ == "__main__":
    main()
