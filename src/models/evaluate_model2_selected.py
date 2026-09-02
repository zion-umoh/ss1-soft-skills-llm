"""Freeze the Model 2 validation selection and evaluate it once on held-out test.

This command reads the two validation reports, selects the lower-MAE candidate
(then lower RMSE if tied), saves a compact comparison, and only evaluates the
chosen already-trained model when explicitly confirmed.  It refuses to replace
an existing test prediction file.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib

try:
    from src.models.train_model2_benchmark import load_split, regression_metrics, write_predictions
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    from train_model2_benchmark import load_split, regression_metrics, write_predictions


def choose_candidate(linear: dict[str, object], nonlinear: dict[str, object]) -> tuple[str, dict[str, object]]:
    candidates = (("linear", linear), ("extra_trees", nonlinear))
    return min(candidates, key=lambda item: (item[1]["validation_metrics"]["mae"], item[1]["validation_metrics"]["rmse"]))


def render_selection_report(selected: str, linear: dict[str, object], nonlinear: dict[str, object], test: dict[str, object] | None) -> str:
    lines = [
        "# Model 2 Benchmark-versus-Improved Comparison",
        "",
        "## Validation-only selection",
        "",
        "| Version | Validation MAE | Validation RMSE | Validation mean R² |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, candidate in (("Linear benchmark", linear), ("Extra Trees improvement", nonlinear)):
        metrics = candidate["validation_metrics"]
        lines.append(f"| {name} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | {metrics['r2']:.4f} |")
    lines.extend(["", f"Selected version: **{selected}**, using validation MAE (then RMSE on a tie).", ""])
    if test is None:
        lines.extend(["The held-out test split has not been evaluated.", ""])
    else:
        lines.extend(
            [
                "## One-time held-out test evaluation",
                "",
                "| Test MAE | Test RMSE | Test mean R² |",
                "| ---: | ---: | ---: |",
                f"| {test['mae']:.4f} | {test['rmse']:.4f} | {test['r2']:.4f} |",
                "",
                "The Model 2 split remains row-level because source `Case` is non-unique; results are not participant-independent estimates.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/model2"))
    parser.add_argument("--linear-metadata", type=Path, default=Path("data/metadata/model2-linear-benchmark.json"))
    parser.add_argument("--extra-trees-metadata", type=Path, default=Path("data/metadata/model2-extra-trees.json"))
    parser.add_argument("--linear-model", type=Path, default=Path("outputs/models/model2_linear_benchmark.joblib"))
    parser.add_argument("--extra-trees-model", type=Path, default=Path("outputs/models/model2_extra_trees.joblib"))
    parser.add_argument("--test-predictions", type=Path, default=Path("outputs/model-evaluation/model2_selected_test_predictions.csv"))
    parser.add_argument("--selection-metadata", type=Path, default=Path("data/metadata/model2-selection.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/model-evaluation/model2-comparison-report.md"))
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args()
    linear = json.loads(args.linear_metadata.read_text(encoding="utf-8"))
    nonlinear = json.loads(args.extra_trees_metadata.read_text(encoding="utf-8"))
    selected, selected_metadata = choose_candidate(linear, nonlinear)
    if not args.confirm_final_test:
        raise SystemExit(f"Selected {selected}; rerun with --confirm-final-test to evaluate held-out test once.")
    if args.test_predictions.exists():
        raise SystemExit(f"Refusing to overwrite existing held-out test predictions: {args.test_predictions}")
    _, features, targets = load_split(args.data_dir / "bfi2_bessi_test.csv", "test")
    record_ids, _, _ = load_split(args.data_dir / "bfi2_bessi_test.csv", "test")
    model_file = args.linear_model if selected == "linear" else args.extra_trees_model
    model = joblib.load(model_file)["model"]
    predictions = model.predict(features)
    test_metrics = regression_metrics(targets, predictions)
    write_predictions(args.test_predictions, record_ids, targets, predictions, split="test")
    selection_metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "selection_split": "validation",
        "selection_rule": "lower aggregate MAE; lower aggregate RMSE breaks ties",
        "selected_version": selected,
        "selected_validation_metrics": selected_metadata["validation_metrics"],
        "test_rows_used_once": len(record_ids),
        "test_metrics": test_metrics,
    }
    args.selection_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.selection_metadata.write_text(json.dumps(selection_metadata, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_selection_report(selected, linear, nonlinear, test_metrics), encoding="utf-8")
    print(f"Selected {selected}; held-out test MAE={test_metrics['mae']:.4f}, RMSE={test_metrics['rmse']:.4f}, R²={test_metrics['r2']:.4f}.")


if __name__ == "__main__":
    main()
