"""Train the transformer, LLM-text, engineered-audio and speech-embedding comparisons.

This command is deliberately limited to the active dissertation targets: the
five RecruitView Big Five labels and the separate direct ``speaking_skills``
benchmark. Audio enters through extracted numeric characteristics and a
pretrained WavLM embedding table; the LLM enters only through its cached
structured text-feature table.
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
from sklearn.preprocessing import StandardScaler

from src.data.extract_recruitview_llm_features import FEATURE_COLUMNS as LLM_FEATURE_COLUMNS
from src.evaluation.metrics import spearman
from src.models.model1_protocol import DIRECT_TARGET_COLUMNS, PROTOCOL, SPEAKING_SKILLS_COLUMN, TRAIT_COLUMNS


ALPHAS = (0.1, 1.0, 10.0, 100.0)
BLEND_WEIGHTS = tuple(np.round(np.linspace(0.0, 1.0, 21), 2))
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class FeatureFusionError(ValueError):
    """Raised when prepared feature tables violate the Model 1 contract."""


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise FeatureFusionError(f"No rows found in {path}")
    return rows


def _load_numeric_table(
    path: Path,
    rows: list[dict[str, str]],
    feature_columns: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    table_rows = _read(path)
    by_id = {row["record_id"]: row for row in table_rows}
    if set(by_id) != {row["record_id"] for row in rows}:
        raise FeatureFusionError(f"Features in {path} must cover exactly the prepared records.")
    columns = feature_columns or tuple(column for column in table_rows[0] if column != "record_id")
    if not set(columns).issubset(table_rows[0]):
        raise FeatureFusionError(f"Features in {path} are missing required columns {sorted(set(columns) - set(table_rows[0]))}.")
    try:
        values = np.asarray([[float(by_id[row["record_id"]][column]) for column in columns] for row in rows], dtype=float)
    except (KeyError, ValueError) as error:
        raise FeatureFusionError(f"Features in {path} must be numeric and complete.") from error
    if not np.isfinite(values).all():
        raise FeatureFusionError(f"Features in {path} must be finite.")
    return values, columns


def _load_tables(
    prepared_path: Path,
    audio_path: Path,
    llm_features_path: Path,
    speech_embeddings_path: Path,
) -> tuple[list[dict[str, str]], np.ndarray, tuple[str, ...], np.ndarray, tuple[str, ...], np.ndarray, tuple[str, ...]]:
    rows = _read(prepared_path)
    if len({row["record_id"] for row in rows}) != len(rows):
        raise FeatureFusionError("Prepared RecruitView record IDs must be unique.")
    audio, audio_columns = _load_numeric_table(audio_path, rows)
    llm_features, llm_columns = _load_numeric_table(llm_features_path, rows, LLM_FEATURE_COLUMNS)
    speech_embeddings, speech_columns = _load_numeric_table(speech_embeddings_path, rows)
    return rows, audio, audio_columns, llm_features, llm_columns, speech_embeddings, speech_columns


def _encode(rows: list[dict[str, str]], model_name: str, batch_size: int, device: str | None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    texts = [f"QUESTION: {row['question']} RESPONSE: {row['text_model']}" for row in rows]
    model = SentenceTransformer(model_name, device=device)
    return np.asarray(
        model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True),
        dtype=float,
    )


def _split(rows: list[dict[str, str]], split: str) -> np.ndarray:
    return np.asarray([index for index, row in enumerate(rows) if row["split"] == split], dtype=int)


def _targets(rows: list[dict[str, str]], columns: tuple[str, ...]) -> np.ndarray:
    values = np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise FeatureFusionError("Target values must be finite.")
    return values


def _length_features(rows: list[dict[str, str]]) -> np.ndarray:
    """Create the predeclared response-length confound baseline."""
    values = []
    for row in rows:
        text = row["text_model"]
        values.append((np.log1p(len(text.split())), np.log1p(len(text)), np.log1p(text.count(".") + text.count("!") + text.count("?"))))
    return np.asarray(values, dtype=float)


def _scale(block: np.ndarray, train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler().fit(block[train])
    return scaler.transform(block[train]), scaler.transform(block[other]), scaler


def _metrics(target: np.ndarray, prediction: np.ndarray, columns: tuple[str, ...]) -> dict[str, object]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if target.ndim == 1:
        target = target[:, None]
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    per_target = {column: {"spearman": spearman(target[:, index], prediction[:, index])} for index, column in enumerate(columns)}
    values = [float(value["spearman"]) for value in per_target.values() if value["spearman"] is not None]
    return {"macro_spearman": float(np.mean(values)) if values else None, "per_target": per_target}


def _select_alpha(features: np.ndarray, target: np.ndarray, train: np.ndarray, validation: np.ndarray, columns: tuple[str, ...]) -> tuple[float, dict[str, float]]:
    candidates: dict[str, float] = {}
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(features[train], target[train])
        candidates[str(alpha)] = _metrics(target[validation], model.predict(features[validation]), columns)["macro_spearman"]
    selected = max(ALPHAS, key=lambda alpha: (candidates[str(alpha)], -alpha))
    return selected, candidates


def _fit_final(features: np.ndarray, target: np.ndarray, train: np.ndarray, validation: np.ndarray, test: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, Ridge]:
    """Return honest validation predictions and a final train+validation model."""
    validation_model = Ridge(alpha=alpha).fit(features[train], target[train])
    validation_prediction = validation_model.predict(features[validation])
    fit_indices = np.concatenate((train, validation))
    model = Ridge(alpha=alpha).fit(features[fit_indices], target[fit_indices])
    return validation_prediction, model.predict(features[test]), model


def _select_blend_weight(
    text_prediction: np.ndarray,
    audio_prediction: np.ndarray,
    target: np.ndarray,
    columns: tuple[str, ...],
) -> tuple[float, dict[str, float | None]]:
    """Select the text contribution for late fusion on validation predictions."""
    candidates: dict[str, float | None] = {}
    for text_weight in BLEND_WEIGHTS:
        blended = text_weight * text_prediction + (1.0 - text_weight) * audio_prediction
        candidates[f"{text_weight:.2f}"] = _metrics(target, blended, columns)["macro_spearman"]
    selected = max(
        BLEND_WEIGHTS,
        key=lambda weight: (
            -np.inf if candidates[f"{weight:.2f}"] is None else candidates[f"{weight:.2f}"],
            -abs(weight - 0.5),
        ),
    )
    return float(selected), candidates


def run(
    prepared_path: Path,
    audio_path: Path,
    llm_features_path: Path,
    artifact_path: Path,
    predictions_path: Path,
    report_path: Path,
    speech_embeddings_path: Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    device: str | None = None,
) -> dict[str, object]:
    rows, audio, audio_columns, llm_features, llm_columns, speech_embeddings, speech_columns = _load_tables(prepared_path, audio_path, llm_features_path, speech_embeddings_path)
    train, validation, test = (_split(rows, split) for split in ("train", "validation", "test"))
    if any(len(indices) == 0 for indices in (train, validation, test)):
        raise FeatureFusionError("Every participant-grouped split must contain rows.")
    groups = [set(rows[index]["participant_id"] for index in indices) for indices in (train, validation, test)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise FeatureFusionError("Participant leakage detected between fixed splits.")

    embeddings = _encode(rows, embedding_model, batch_size, device)
    scaled_text_train, scaled_text_other, text_scaler = _scale(embeddings, train, np.arange(len(rows)))
    scaled_audio_train, scaled_audio_other, audio_scaler = _scale(audio, train, np.arange(len(rows)))
    scaled_llm_train, scaled_llm_other, llm_scaler = _scale(llm_features, train, np.arange(len(rows)))
    scaled_speech_train, scaled_speech_other, speech_scaler = _scale(speech_embeddings, train, np.arange(len(rows)))
    text_features = scaled_text_other
    audio_features = scaled_audio_other
    llm_text_features = scaled_llm_other
    speech_features = scaled_speech_other
    fused_features = np.hstack((text_features, audio_features))
    llm_fused_features = np.hstack((llm_text_features, audio_features))
    llm_speech_features = np.hstack((llm_text_features, speech_features))
    llm_speech_fused_features = np.hstack((llm_text_features, audio_features, speech_features))

    targets = _targets(rows, DIRECT_TARGET_COLUMNS)
    length = _length_features(rows)
    _, length_features, length_scaler = _scale(length, train, np.arange(len(rows)))
    variants = {
        "length_only": length_features,
        "text_only": text_features,
        "audio_only": audio_features,
        "fused": fused_features,
        "llm_text_only": llm_text_features,
        "llm_fused": llm_fused_features,
        "speech_only": speech_features,
        "llm_speech": llm_speech_features,
        "llm_speech_fused": llm_speech_fused_features,
    }
    results: dict[str, object] = {}
    final_models: dict[str, object] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for variant, features in variants.items():
        alpha_traits, candidate_traits = _select_alpha(features, targets[:, :5], train, validation, TRAIT_COLUMNS)
        trait_validation, trait_test, trait_model = _fit_final(features, targets[:, :5], train, validation, test, alpha_traits)
        alpha_speaking, candidate_speaking = _select_alpha(features, targets[:, 5:], train, validation, (SPEAKING_SKILLS_COLUMN,))
        speaking_validation, speaking_test, speaking_model = _fit_final(features, targets[:, 5:], train, validation, test, alpha_speaking)
        speaking_validation = np.asarray(speaking_validation)[:, None] if np.asarray(speaking_validation).ndim == 1 else speaking_validation
        speaking_test = np.asarray(speaking_test)[:, None] if np.asarray(speaking_test).ndim == 1 else speaking_test
        results[variant] = {
            "big_five": {"selected_alpha": alpha_traits, "validation_candidates": candidate_traits, "validation_metrics": _metrics(targets[validation, :5], trait_validation, TRAIT_COLUMNS), "test_metrics": _metrics(targets[test, :5], trait_test, TRAIT_COLUMNS)},
            "speaking_skills": {"selected_alpha": alpha_speaking, "validation_candidates": candidate_speaking, "validation_metrics": _metrics(targets[validation, 5:], speaking_validation, (SPEAKING_SKILLS_COLUMN,)), "test_metrics": _metrics(targets[test, 5:], speaking_test, (SPEAKING_SKILLS_COLUMN,))},
        }
        final_models[variant] = {"big_five": trait_model, "speaking_skills": speaking_model}
        validation_predictions[variant] = np.hstack((trait_validation, speaking_validation))
        test_predictions[variant] = np.hstack((trait_test, speaking_test))

    # Late fusion combines predictions from independently trained text and audio
    # models. Each blend weight is selected on validation predictions only.
    fusion_metadata: dict[str, object] = {}

    def add_late_fusion(label: str, text_variant: str) -> None:
        big_five_text_weight, big_five_blend_candidates = _select_blend_weight(
            validation_predictions[text_variant][:, :5],
            validation_predictions["audio_only"][:, :5],
            targets[validation, :5],
            TRAIT_COLUMNS,
        )
        speaking_text_weight, speaking_blend_candidates = _select_blend_weight(
            validation_predictions[text_variant][:, 5:],
            validation_predictions["audio_only"][:, 5:],
            targets[validation, 5:],
            (SPEAKING_SKILLS_COLUMN,),
        )
        late_validation = np.hstack(
            (
                big_five_text_weight * validation_predictions[text_variant][:, :5]
                + (1.0 - big_five_text_weight) * validation_predictions["audio_only"][:, :5],
                speaking_text_weight * validation_predictions[text_variant][:, 5:]
                + (1.0 - speaking_text_weight) * validation_predictions["audio_only"][:, 5:],
            )
        )
        late_test = np.hstack(
            (
                big_five_text_weight * test_predictions[text_variant][:, :5]
                + (1.0 - big_five_text_weight) * test_predictions["audio_only"][:, :5],
                speaking_text_weight * test_predictions[text_variant][:, 5:]
                + (1.0 - speaking_text_weight) * test_predictions["audio_only"][:, 5:],
            )
        )
        results[label] = {
            "big_five": {
                "text_weight": big_five_text_weight,
                "audio_weight": 1.0 - big_five_text_weight,
                "validation_candidates": big_five_blend_candidates,
                "validation_metrics": _metrics(targets[validation, :5], late_validation[:, :5], TRAIT_COLUMNS),
                "test_metrics": _metrics(targets[test, :5], late_test[:, :5], TRAIT_COLUMNS),
            },
            "speaking_skills": {
                "text_weight": speaking_text_weight,
                "audio_weight": 1.0 - speaking_text_weight,
                "validation_candidates": speaking_blend_candidates,
                "validation_metrics": _metrics(targets[validation, 5:], late_validation[:, 5:], (SPEAKING_SKILLS_COLUMN,)),
                "test_metrics": _metrics(targets[test, 5:], late_test[:, 5:], (SPEAKING_SKILLS_COLUMN,)),
            },
        }
        final_models[label] = {
            "text_variant": text_variant,
            "audio_variant": "audio_only",
            "big_five_text_weight": big_five_text_weight,
            "speaking_skills_text_weight": speaking_text_weight,
        }
        test_predictions[label] = late_test
        fusion_metadata[label] = final_models[label]

    add_late_fusion("late_fusion", "text_only")
    add_late_fusion("llm_late_fusion", "llm_text_only")

    train_mean = targets[train].mean(axis=0)
    validation_mean = np.broadcast_to(train_mean, (len(validation), len(DIRECT_TARGET_COLUMNS)))
    test_mean = np.broadcast_to(train_mean, (len(test), len(DIRECT_TARGET_COLUMNS)))
    results["mean_baseline"] = {
        "big_five": {"validation_metrics": _metrics(targets[validation, :5], validation_mean[:, :5], TRAIT_COLUMNS), "test_metrics": _metrics(targets[test, :5], test_mean[:, :5], TRAIT_COLUMNS)},
        "speaking_skills": {"validation_metrics": _metrics(targets[validation, 5:], validation_mean[:, 5:], (SPEAKING_SKILLS_COLUMN,)), "test_metrics": _metrics(targets[test, 5:], test_mean[:, 5:], (SPEAKING_SKILLS_COLUMN,))},
    }

    # Selection is made on validation, never on the final test set.
    candidate_variants = (*variants.keys(), "late_fusion", "llm_late_fusion")
    selected_variant = max(
        candidate_variants,
        key=lambda variant: (results[variant]["big_five"]["validation_metrics"]["macro_spearman"], variant == "late_fusion"),
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": final_models,
            "text_scaler": text_scaler,
            "audio_scaler": audio_scaler,
            "llm_scaler": llm_scaler,
            "speech_scaler": speech_scaler,
            "length_scaler": length_scaler,
            "embedding_model": embedding_model,
            "audio_feature_columns": list(audio_columns),
            "llm_feature_columns": list(llm_columns),
            "speech_embedding_columns": list(speech_columns),
            "late_fusion": {"blend_grid": list(BLEND_WEIGHTS), "variants": fusion_metadata},
            "protocol": PROTOCOL,
        },
        artifact_path,
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["record_id", "participant_id", "split", *[f"observed_{column}" for column in DIRECT_TARGET_COLUMNS], *[f"prediction_{column}" for column in DIRECT_TARGET_COLUMNS]]
    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row_index, original_index in enumerate(test):
            row = {"record_id": rows[original_index]["record_id"], "participant_id": rows[original_index]["participant_id"], "split": "test"}
            row.update({f"observed_{column}": targets[original_index, index] for index, column in enumerate(DIRECT_TARGET_COLUMNS)})
            row.update({f"prediction_{column}": test_predictions[selected_variant][row_index, index] for index, column in enumerate(DIRECT_TARGET_COLUMNS)})
            writer.writerow(row)

    report: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "completed",
        "protocol": {
            "raw_audio_input": False,
            "embedding_model": embedding_model,
            "llm_feature_extractor": "gpt-5.6-luna structured observable-text features",
            "llm_feature_columns": list(llm_columns),
            "speech_embedding_model": "microsoft/wavlm-base-plus",
            "speech_embedding_columns": list(speech_columns),
            "primary_variant": selected_variant,
            "primary_metric": PROTOCOL.primary_metric,
            "fusion_method": "participant-disjoint Ridge comparison with direct concatenation and validation-tuned late fusion",
            "participant_disjoint": True,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
        "variants": results,
        "artifact": str(artifact_path),
        "predictions": str(predictions_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    def show(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    lines = [
        "# RecruitView Model 1 feature comparison",
        "",
        "Raw audio was decoded into numerical characteristics and pretrained speech embeddings; no waveform was passed directly to a Ridge predictor.",
        "The transcript representation is a frozen `sentence-transformers/all-MiniLM-L6-v2` embedding.",
        "The LLM text representation is a cached, structured `gpt-5.6-luna` extraction of observable linguistic and affective features; it was not asked to predict the targets directly.",
        "",
        "| Variant | Validation Big Five Spearman | Test Big Five Spearman | Validation speaking_skills Spearman | Test speaking_skills Spearman |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for variant in (*candidate_variants, "mean_baseline"):
        value = results[variant]
        big_five = value["big_five"]
        speaking = value["speaking_skills"]
        lines.append(
            f"| {variant} | {show(big_five['validation_metrics']['macro_spearman'])} | {show(big_five['test_metrics']['macro_spearman'])} | {show(speaking['validation_metrics']['macro_spearman'])} | {show(speaking['test_metrics']['macro_spearman'])} |"
        )
    lines.extend(
        [
            "",
            f"Selected Big Five variant using validation only: **{selected_variant}**.",
            "The held-out test values above were generated once after that selection and must not be used to tune the model.",
            "",
            "This is a same-dataset RecruitView benchmark. It is not direct BESSI validation; the final published CRMF comparison is a separate later batch.",
            "",
        ]
    )
    report_path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/processed/recruitview/recruitview_prepared.csv"))
    parser.add_argument("--audio", type=Path, default=Path("data/processed/recruitview/recruitview_audio_features.csv"))
    parser.add_argument("--llm-features", type=Path, default=Path("data/processed/recruitview/recruitview_llm_features.csv"))
    parser.add_argument("--speech-embeddings", type=Path, default=Path("data/processed/recruitview/recruitview_speech_embeddings.csv"))
    parser.add_argument("--artifact", type=Path, default=Path("outputs/models/model1_feature_fusion.joblib"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/benchmark/recruitview-feature-fusion/predictions.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/benchmark/recruitview-feature-fusion.json"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    result = run(args.prepared, args.audio, args.llm_features, args.artifact, args.predictions, args.report, args.speech_embeddings, args.embedding_model, args.batch_size, args.device)
    print(json.dumps({"status": result["status"], "primary_variant": result["protocol"]["primary_variant"]}, indent=2))


if __name__ == "__main__":
    main()
