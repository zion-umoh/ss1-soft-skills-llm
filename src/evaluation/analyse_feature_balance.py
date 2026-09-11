"""Audit whether one engineered feature block dominates Model 1.

This is a post-selection diagnostic. It does not change the locked model or
readjust any hyperparameters. Importance is measured on the fixed validation
split using a train-only Ridge refit with the already-selected alpha.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from src.data.extract_recruitview_llm_features import FEATURE_COLUMNS as LLM_FEATURE_COLUMNS
from src.evaluation.metrics import spearman
from src.models.model1_protocol import TRAIT_COLUMNS


AUDIO_GROUPS = {
    "audio_timing": (
        "audio_duration_seconds",
        "audio_pause_count",
        "audio_pause_fraction",
        "audio_speaking_time_ratio",
        "audio_words_per_second",
    ),
    "audio_pitch_energy": (
        "audio_rms_mean",
        "audio_rms_std",
        "audio_pitch_mean_hz",
        "audio_pitch_std_hz",
    ),
    "audio_spectral": (
        "audio_zero_crossing_rate",
        "audio_spectral_centroid_mean_hz",
        "audio_spectral_centroid_std_hz",
    ),
}
LLM_GROUPS = {
    "llm_affect": ("emotional_expressiveness", "positive_affect", "negative_affect", "interpersonal_warmth"),
    "llm_social_agency": ("social_orientation", "assertiveness"),
    "llm_cognition_communication": ("certainty", "cognitive_complexity", "response_elaboration", "question_relevance"),
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def _matrix(path: Path, source_rows: list[dict[str, str]], columns: tuple[str, ...]) -> np.ndarray:
    rows = _read(path)
    by_id = {row["record_id"]: row for row in rows}
    if set(by_id) != {row["record_id"] for row in source_rows}:
        raise ValueError(f"{path} does not cover exactly the prepared rows.")
    values = np.asarray([[float(by_id[row["record_id"]][column]) for column in columns] for row in source_rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in {path}")
    return values


def _score(target: np.ndarray, prediction: np.ndarray) -> float:
    values = [spearman(target[:, index], prediction[:, index]) for index in range(target.shape[1])]
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else 0.0


def _stats(values: np.ndarray) -> dict[str, float | int]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p99": float(np.quantile(values, 0.99)),
        "unique_values": int(len(np.unique(values))),
        "zero_fraction": float(np.mean(values == 0)),
    }


def _group_importance(model: Ridge, X: np.ndarray, y: np.ndarray, feature_names: list[str], groups: dict[str, tuple[str, ...]], repeats: int, seed: int) -> tuple[float, dict[str, dict[str, float]]]:
    baseline = _score(y, model.predict(X))
    rng = np.random.default_rng(seed)
    index = {name: i for i, name in enumerate(feature_names)}
    result: dict[str, dict[str, float]] = {}
    for group, columns in groups.items():
        indices = [index[column] for column in columns]
        drops = []
        for _ in range(repeats):
            permuted = X.copy()
            order = rng.permutation(len(X))
            permuted[:, indices] = X[order][:, indices]
            drops.append(baseline - _score(y, model.predict(permuted)))
        result[group] = {"mean_score_drop": float(np.mean(drops)), "std_score_drop": float(np.std(drops))}
    return baseline, result


def run(
    prepared_path: Path,
    audio_path: Path,
    llm_path: Path,
    speech_path: Path,
    artifact_path: Path,
    model_report_path: Path,
    output_json: Path,
    output_markdown: Path,
    repeats: int = 20,
) -> dict[str, object]:
    prepared = _read(prepared_path)
    audio_columns = tuple(_read(audio_path)[0].keys())[1:]
    audio = _matrix(audio_path, prepared, audio_columns)
    llm = _matrix(llm_path, prepared, LLM_FEATURE_COLUMNS)
    speech_columns = tuple(_read(speech_path)[0].keys())[1:]
    speech = _matrix(speech_path, prepared, speech_columns)
    artifact = joblib.load(artifact_path)
    model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
    selected = model_report["protocol"]["primary_variant"]
    if selected not in {"llm_fused", "llm_speech_fused"}:
        raise ValueError(f"Feature-balance analysis expects a fused LLM model, got {selected!r}.")
    train = np.asarray([i for i, row in enumerate(prepared) if row["split"] == "train"], dtype=int)
    validation = np.asarray([i for i, row in enumerate(prepared) if row["split"] == "validation"], dtype=int)
    test = np.asarray([i for i, row in enumerate(prepared) if row["split"] == "test"], dtype=int)
    llm_scaled = artifact["llm_scaler"].transform(llm)
    audio_scaled = artifact["audio_scaler"].transform(audio)
    speech_scaled = artifact["speech_scaler"].transform(speech)
    if selected == "llm_speech_fused":
        feature_names = [*LLM_FEATURE_COLUMNS, *audio_columns, *speech_columns]
        X = np.hstack((llm_scaled, audio_scaled, speech_scaled))
    else:
        feature_names = [*LLM_FEATURE_COLUMNS, *audio_columns]
        X = np.hstack((llm_scaled, audio_scaled))
    targets = np.asarray([[float(row[column]) for column in (*TRAIT_COLUMNS, "speaking_skills")] for row in prepared], dtype=float)
    alpha = model_report["variants"][selected]["big_five"]["selected_alpha"]
    model = Ridge(alpha=alpha).fit(X[train], targets[train, :5])
    target_names = list(TRAIT_COLUMNS)
    coefficient_rows = []
    for target_index, target_name in enumerate(target_names):
        coefficients = model.coef_[target_index]
        order = np.argsort(-np.abs(coefficients))
        coefficient_rows.append(
            {
                "target": target_name,
                "ranked_features": [
                    {"feature": feature_names[index], "coefficient": float(coefficients[index]), "absolute_coefficient": float(abs(coefficients[index]))}
                    for index in order
                ],
            }
        )
    all_groups = {**LLM_GROUPS, **AUDIO_GROUPS}
    if selected == "llm_speech_fused":
        all_groups["speech_embedding"] = speech_columns
    group_coefficients = {}
    for group, columns in all_groups.items():
        indices = [feature_names.index(column) for column in columns]
        group_coefficients[group] = {
            "absolute_coefficient_sum": float(np.abs(model.coef_[:, indices]).sum()),
            "features": list(columns),
        }
    baseline, group_permutation = _group_importance(model, X[validation], targets[validation, :5], feature_names, all_groups, repeats, 42)
    baseline_test = _score(targets[test, :5], model.predict(X[test]))
    group_ablation = {}
    for group, columns in all_groups.items():
        removed = {feature_names.index(column) for column in columns}
        keep = [index for index in range(len(feature_names)) if index not in removed]
        ablated_model = Ridge(alpha=alpha).fit(X[train][:, keep], targets[train, :5])
        validation_score = _score(targets[validation, :5], ablated_model.predict(X[validation][:, keep]))
        test_score = _score(targets[test, :5], ablated_model.predict(X[test][:, keep]))
        group_ablation[group] = {
            "validation_macro_spearman_without_group": validation_score,
            "validation_change_from_full": validation_score - baseline,
            "test_macro_spearman_without_group": test_score,
            "test_change_from_full_train_only": test_score - baseline_test,
        }
    raw_features = np.hstack((llm, audio, speech)) if selected == "llm_speech_fused" else np.hstack((llm, audio))
    feature_stats = {name: _stats(values) for name, values in zip(feature_names, raw_features.T)}
    z = (raw_features - np.mean(raw_features[train], axis=0)) / np.maximum(np.std(raw_features[train], axis=0), 1e-8)
    correlations = []
    correlation_matrix = np.corrcoef(z, rowvar=False)
    rows_i, columns_j = np.where(np.triu(np.abs(correlation_matrix) >= 0.70, k=1))
    for i, j in zip(rows_i, columns_j, strict=True):
        correlations.append({"feature_a": feature_names[int(i)], "feature_b": feature_names[int(j)], "pearson_r": float(correlation_matrix[i, j])})
    correlations.sort(key=lambda item: -abs(item["pearson_r"]))
    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "completed",
        "selected_variant": selected,
        "validation_rows": len(validation),
        "permutation_repeats": repeats,
        "feature_columns": feature_names,
        "feature_stats": feature_stats,
        "high_correlations_abs_ge_0.70": correlations,
        "validation_big_five_baseline_macro_spearman": baseline,
        "test_big_five_baseline_macro_spearman_train_only": baseline_test,
        "group_absolute_coefficients": group_coefficients,
        "group_validation_permutation_importance": group_permutation,
        "group_ablation_fixed_alpha": group_ablation,
        "per_target_standardized_coefficients": coefficient_rows,
        "interpretation": "Exploratory post-selection diagnostics. These results identify concentration and redundancy; they do not retune the selected model or establish causal feature effects.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Model 1 feature-balance analysis",
        "",
        "This is an exploratory post-selection diagnostic using a train-only Ridge refit and the fixed validation split. It does not retune the final model.",
        "",
        f"Selected model: **{selected}**; validation rows: **{len(validation)}**; baseline Big Five macro Spearman: **{baseline:.4f}**.",
        "",
        "## Feature blocks",
        "",
        "| Block | Absolute coefficient sum | Validation permutation score drop |",
        "| --- | ---: | ---: |",
    ]
    for group in all_groups:
        lines.append(f"| {group} | {group_coefficients[group]['absolute_coefficient_sum']:.4f} | {group_permutation[group]['mean_score_drop']:+.4f} ± {group_permutation[group]['std_score_drop']:.4f} |")
    lines.extend(["", "## Leave-one-block-out sensitivity", "", "Each row refits the same train-only Ridge protocol after removing one block; alpha remains fixed at the selected value. The test column is diagnostic only and was not used for selection.", "", "| Omitted block | Validation macro Spearman | Change | Train-only test macro Spearman | Change |", "| --- | ---: | ---: | ---: | ---: |"])
    for group, values in group_ablation.items():
        lines.append(f"| {group} | {values['validation_macro_spearman_without_group']:.4f} | {values['validation_change_from_full']:+.4f} | {values['test_macro_spearman_without_group']:.4f} | {values['test_change_from_full_train_only']:+.4f} |")
    lines.extend(["", "## High feature correlations", "", "| Feature A | Feature B | Pearson r |", "| --- | --- | ---: |"])
    for pair in correlations[:20]:
        lines.append(f"| {pair['feature_a']} | {pair['feature_b']} | {pair['pearson_r']:+.3f} |")
    if not correlations:
        lines.append("| None above | 0.70 absolute correlation | — |")
    lines.extend(["", "## Interpretation boundary", "", "Large coefficients or permutation drops indicate predictive concentration in this fitted model, not causal importance. In particular, duration and speaking-rate features must be interpreted as potential response-length confounds.", ""])
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/processed/recruitview/recruitview_prepared.csv"))
    parser.add_argument("--audio", type=Path, default=Path("data/processed/recruitview/recruitview_audio_features.csv"))
    parser.add_argument("--llm", type=Path, default=Path("data/processed/recruitview/recruitview_llm_features.csv"))
    parser.add_argument("--speech", type=Path, default=Path("data/processed/recruitview/recruitview_speech_embeddings.csv"))
    parser.add_argument("--artifact", type=Path, default=Path("outputs/models/model1_feature_fusion.joblib"))
    parser.add_argument("--model-report", type=Path, default=Path("reports/benchmark/recruitview-feature-fusion.json"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/benchmark/recruitview-feature-balance.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("reports/benchmark/recruitview-feature-balance.md"))
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    result = run(args.prepared, args.audio, args.llm, args.speech, args.artifact, args.model_report, args.output_json, args.output_markdown, args.repeats)
    print(json.dumps({"status": result["status"], "selected_variant": result["selected_variant"]}, indent=2))


if __name__ == "__main__":
    main()
