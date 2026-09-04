"""Measure whether observed BFI-2 facets improve BESSI-domain prediction.

This is deliberately an upper-bound diagnostic, not a replacement for the
selected Model 2 pipeline. The essay model currently supplies five Big Five
domains, whereas this experiment receives all 15 observed BFI-2 facets.
Regularisation strength is selected only within the training split.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.data.prepare_bfi2_bessi import BFI2_FACET_COLUMNS
    from src.models.train_model2_benchmark import TARGET_COLUMNS, load_split, regression_metrics, write_predictions
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from prepare_bfi2_bessi import BFI2_FACET_COLUMNS
    from train_model2_benchmark import TARGET_COLUMNS, load_split, regression_metrics, write_predictions


FEATURE_COLUMNS = tuple(BFI2_FACET_COLUMNS.values())
ALPHAS = tuple(float(value) for value in np.logspace(-3, 3, 13))
CV_FOLDS = 5


class FacetTrainingError(ValueError):
    """Raised when facet and baseline datasets are not comparable."""


def build_model() -> Pipeline:
    """Create the one predeclared facet model with train-only CV regularisation."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS, cv=CV_FOLDS, scoring="neg_mean_absolute_error")),
        ]
    )


def require_matching_rows(
    facet_ids: list[str], baseline_ids: list[str], split: str
) -> None:
    """Guarantee that the diagnostic uses the same source rows as Model 2."""
    if facet_ids != baseline_ids:
        raise FacetTrainingError(
            f"Facet and baseline {split} records differ; the upper-bound comparison would be invalid."
        )


def render_report(metadata: dict[str, object]) -> str:
    results = metadata["validation_metrics"]
    baseline = metadata["baseline_validation_metrics"]
    lines = [
        "# Model 2 BFI-2 Facet Upper-Bound Diagnostic",
        "",
        "## Purpose and scope",
        "",
        "This evaluates whether **observed** BFI-2 facets contain useful information beyond the five domain averages when estimating BESSI domains.",
        "It is not deployable in the present essay-to-Big-Five-to-BESSI pipeline: Model 1 supplies five domain estimates, not fifteen facet estimates.",
        "",
        "## Protocol",
        "",
        "- Inputs: 15 supplied BFI-2 facet scores, transformed from 1--5 to 0--1 during preparation.",
        "- Outputs: five supplied BESSI domain scores on the original 1--5 scale.",
        "- Candidate: standardized multi-output Ridge regression.",
        "- Regularisation: alpha selected by five-fold cross-validation on the training split only.",
        "- Comparison: the existing five-domain ordinary-least-squares benchmark on exactly the same validation records.",
        "- No test rows are read or used by this experiment.",
        "",
        "## Validation comparison",
        "",
        "| Version | Aggregate MAE | Aggregate RMSE | Mean R² |",
        "| --- | ---: | ---: | ---: |",
        f"| Five-domain linear baseline | {baseline['mae']:.4f} | {baseline['rmse']:.4f} | {baseline['r2']:.4f} |",
        f"| Observed-facet Ridge upper bound | {results['mae']:.4f} | {results['rmse']:.4f} | {results['r2']:.4f} |",
        "",
        f"Ridge alpha selected inside training data: `{metadata['selected_alpha']}`.",
        "",
        f"Decision: **{metadata['selection_decision']}**",
        "",
        "| BESSI domain | Facet MAE | Facet RMSE | Facet R² |",
        "| --- | ---: | ---: | ---: |",
    ]
    for target in TARGET_COLUMNS:
        result = results["per_target"][target]
        lines.append(f"| {target} | {result['mae']:.4f} | {result['rmse']:.4f} | {result['r2']:.4f} |")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- A gain would show that facets are valuable information, not that the current production pipeline has improved.",
            "- A facet model can only become end-to-end deployable after a separately validated Model 1 facet-prediction stage is available.",
            "- The validation split was used in previous Model 2 selection work; this is exploratory evidence, not a new final generalisation claim.",
            "- Source `Case` is non-unique, so all splits are row-level rather than participant-independent.",
            "",
        ]
    )
    return "\n".join(lines)


def train_facets(
    facet_data_dir: Path,
    baseline_data_dir: Path,
    baseline_metadata_path: Path,
    model_path: Path,
    prediction_path: Path,
    report_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    facet_train_ids, facet_train_features, train_targets = load_split(
        facet_data_dir / "bfi2_bessi_train.csv", "train", FEATURE_COLUMNS
    )
    baseline_train_ids, _, baseline_train_targets = load_split(
        baseline_data_dir / "bfi2_bessi_train.csv", "train"
    )
    require_matching_rows(facet_train_ids, baseline_train_ids, "training")
    if not np.array_equal(train_targets, baseline_train_targets):
        raise FacetTrainingError("Facet and baseline training targets differ.")

    validation_ids, validation_features, validation_targets = load_split(
        facet_data_dir / "bfi2_bessi_validation.csv", "validation", FEATURE_COLUMNS
    )
    baseline_validation_ids, _, baseline_validation_targets = load_split(
        baseline_data_dir / "bfi2_bessi_validation.csv", "validation"
    )
    require_matching_rows(validation_ids, baseline_validation_ids, "validation")
    if not np.array_equal(validation_targets, baseline_validation_targets):
        raise FacetTrainingError("Facet and baseline validation targets differ.")

    baseline_metadata = json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
    model = build_model()
    model.fit(facet_train_features, train_targets)
    validation_predictions = model.predict(validation_features)
    validation_metrics = regression_metrics(validation_targets, validation_predictions)
    selected_alpha = float(model.named_steps["ridge"].alpha_)
    baseline_metrics = baseline_metadata["validation_metrics"]
    selection_decision = (
        "not selected — aggregate validation MAE did not improve"
        if validation_metrics["mae"] >= baseline_metrics["mae"]
        else "diagnostic signal only — lower MAE requires a separate deployable facet-prediction stage"
    )
    metadata: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "experiment_type": "observed-facet upper-bound diagnostic; not deployable with five-domain Model 1 outputs",
        "feature_columns": list(FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "training_rows": len(facet_train_ids),
        "validation_rows": len(validation_ids),
        "test_rows_used": 0,
        "model": "standardized multi-output Ridge regression",
        "alpha_candidates": list(ALPHAS),
        "cv_folds": CV_FOLDS,
        "selected_alpha": selected_alpha,
        "validation_metrics": validation_metrics,
        "baseline_validation_metrics": baseline_metrics,
        "mae_change_vs_baseline": round(validation_metrics["mae"] - baseline_metrics["mae"], 4),
        "rmse_change_vs_baseline": round(validation_metrics["rmse"] - baseline_metrics["rmse"], 4),
        "r2_change_vs_baseline": round(validation_metrics["r2"] - baseline_metrics["r2"], 4),
        "selection_decision": selection_decision,
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
    parser.add_argument("--facet-data-dir", type=Path, default=Path("data/processed/model2_facets"))
    parser.add_argument("--baseline-data-dir", type=Path, default=Path("data/processed/model2"))
    parser.add_argument("--baseline-metadata", type=Path, default=Path("data/metadata/model2-linear-benchmark.json"))
    parser.add_argument("--model", type=Path, default=Path("outputs/models/model2_facets_upper_bound.joblib"))
    parser.add_argument(
        "--predictions", type=Path, default=Path("outputs/model-evaluation/model2_facets_validation_predictions.csv")
    )
    parser.add_argument("--report", type=Path, default=Path("reports/model-evaluation/model2-facets-upper-bound-report.md"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/model2-facets-upper-bound.json"))
    args = parser.parse_args()
    metadata = train_facets(
        args.facet_data_dir,
        args.baseline_data_dir,
        args.baseline_metadata,
        args.model,
        args.predictions,
        args.report,
        args.metadata,
    )
    results = metadata["validation_metrics"]
    print(
        f"Trained Model 2 facet upper bound on {metadata['training_rows']} rows; "
        f"validation MAE={results['mae']:.4f}, RMSE={results['rmse']:.4f}, R²={results['r2']:.4f}."
    )


if __name__ == "__main__":
    main()
