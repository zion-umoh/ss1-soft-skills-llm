"""Train a label-supervised Model 1 stacker over local-LLM scores and local features.

The local LLM is used only as a fixed, label-free feature extractor.  A
regularised one-vs-rest logistic-regression stacker is fitted on the fixed
Essays train split and compared on validation only.  Test data are never read.
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

try:
    from src.models.train_model1_emotion_ablation import (
        RECORD_ID_COLUMN,
        TRAITS,
        TrainingError,
        load_emotions,
        load_essays,
        load_piastra_metrics,
        metrics,
        write_predictions,
    )
    from src.models.train_model1_embedding_ablation import MODEL_NAME, WORDS_PER_CHUNK, build_classifier, embed_documents
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    from train_model1_emotion_ablation import (
        RECORD_ID_COLUMN,
        TRAITS,
        TrainingError,
        load_emotions,
        load_essays,
        load_piastra_metrics,
        metrics,
        write_predictions,
    )
    from train_model1_embedding_ablation import MODEL_NAME, WORDS_PER_CHUNK, build_classifier, embed_documents


def load_local_scores(path: Path, record_ids: list[str], expected_split: str) -> np.ndarray:
    """Load exactly five score columns, preventing labels from becoming features."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {RECORD_ID_COLUMN, "split", *[f"score_{trait}" for trait in TRAITS]}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise TrainingError(f"{path} does not contain local LLM trait scores.")
        rows = list(reader)
    by_id = {row[RECORD_ID_COLUMN]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(record_ids):
        raise TrainingError(f"{path} does not align one-to-one with the requested Essays split.")
    if any(row["split"] != expected_split for row in rows):
        raise TrainingError(f"{path} contains a row outside {expected_split!r}.")
    try:
        scores = np.asarray([[float(by_id[record_id][f"score_{trait}"]) for trait in TRAITS] for record_id in record_ids])
    except ValueError as error:
        raise TrainingError(f"{path} contains a non-numeric local LLM score.") from error
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 10).any():
        raise TrainingError(f"{path} contains a score outside 0--10.")
    return scores / 10.0


def render_report(metadata: dict[str, object]) -> str:
    lines = [
        "# Model 1 Local-LLM Score Stacker Report",
        "",
        "## Protocol",
        "",
        "- Local LLM trait scores are generated independently and label-free from `text_model` on the same local model for train and validation.",
        "- The stacker is a standardised, class-balanced one-vs-rest logistic-regression model fitted only to Essays training labels.",
        "- Candidates successively add local MiniLM semantic embeddings and 28 precomputed GoEmotions probabilities; each is an ablation, not a hidden change in classifier capacity.",
        "- Test data are not read. Selection is validation macro AUROC, then macro F1 on a tie.",
        "",
        "## Validation comparison",
        "",
        "| Version | Macro AUROC | Macro F1 | Macro balanced accuracy | Exact-match accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, candidate in metadata["candidates"].items():
        result = candidate["validation_metrics"]
        lines.append(f"| {name} | {result['macro_auroc']:.4f} | {result['macro_f1']:.4f} | {result['macro_balanced_accuracy']:.4f} | {result['exact_match_accuracy']:.4f} |")
    lines.extend([
        "",
        f"Provisional validation selection: **{metadata['selected_candidate']}**.",
        "",
        "The zero-shot local score row tests the local LLM directly. The three stacked rows test whether supervised calibration and complementary local representations provide enough new signal to exceed the paper-based GPT benchmark. It is not a claim that the local model itself is stronger than GPT.",
        "",
    ])
    return "\n".join(lines)


def train_stacker(
    data_dir: Path, output_dir: Path, report_path: Path, metadata_path: Path,
    train_scores_path: Path, validation_scores_path: Path, piastra_predictions: Path,
    model_name: str, batch_size: int, device: str | None,
) -> dict[str, object]:
    train_ids, train_texts, train_labels = load_essays(data_dir / "essays_train.csv", "train")
    validation_ids, validation_texts, validation_labels = load_essays(data_dir / "essays_validation.csv", "validation")
    train_scores = load_local_scores(train_scores_path, train_ids, "train")
    validation_scores = load_local_scores(validation_scores_path, validation_ids, "validation")
    emotion_columns, train_emotions = load_emotions(data_dir / "essays_emotion_features_train.csv", train_ids, "train")
    validation_emotion_columns, validation_emotions = load_emotions(
        data_dir / "essays_emotion_features_validation.csv", validation_ids, "validation"
    )
    if emotion_columns != validation_emotion_columns:
        raise TrainingError("GoEmotions feature columns differ between train and validation.")
    train_embeddings, train_chunks, resolved_device = embed_documents(train_texts, model_name, batch_size, device)
    validation_embeddings, validation_chunks, validation_device = embed_documents(validation_texts, model_name, batch_size, device)
    if resolved_device != validation_device:
        raise RuntimeError("Embedding device changed between train and validation encoding.")
    candidates = {
        "local_llm_zero_shot": None,
        "local_llm_score_stacker": (train_scores, validation_scores),
        "local_llm_scores_plus_semantic_embeddings": (
            np.hstack((train_scores, train_embeddings)), np.hstack((validation_scores, validation_embeddings)),
        ),
        "local_llm_scores_plus_semantic_embeddings_plus_goemotions": (
            np.hstack((train_scores, train_embeddings, train_emotions)),
            np.hstack((validation_scores, validation_embeddings, validation_emotions)),
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_candidates: dict[str, object] = {
        "local_llm_zero_shot": {
            "feature_count": len(TRAITS),
            "validation_metrics": metrics(validation_labels, validation_scores),
            "model_settings": {"type": "label-free local LLM scores; no supervised fit"},
        }
    }
    for name, matrices in candidates.items():
        if matrices is None:
            continue
        train_matrix, validation_matrix = matrices
        classifier = build_classifier()
        classifier.fit(train_matrix, train_labels)
        probabilities = classifier.predict_proba(validation_matrix)
        item = {
            "feature_count": int(train_matrix.shape[1]),
            "validation_metrics": metrics(validation_labels, probabilities),
            "model_settings": {"classifier": "StandardScaler + class-balanced LogisticRegression", "C": 0.5, "features": name},
        }
        metadata_candidates[name] = item
        joblib.dump(
            {"classifier": classifier, "candidate": name, "metadata": item, "embedding_model": model_name},
            output_dir / f"{name}.joblib",
        )
        write_predictions(output_dir / f"{name}_validation_predictions.csv", validation_ids, validation_labels, probabilities)
    metadata_candidates["paper_zero_shot_piastra"] = load_piastra_metrics(piastra_predictions, validation_ids, validation_labels)
    selected_candidate = max(
        metadata_candidates,
        key=lambda name: (
            metadata_candidates[name]["validation_metrics"]["macro_auroc"],
            metadata_candidates[name]["validation_metrics"]["macro_f1"],
        ),
    )
    metadata: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "training_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "test_rows_used": 0,
        "local_llm_feature_count": len(TRAITS),
        "embedding_model": model_name,
        "embedding_device": resolved_device,
        "words_per_chunk": WORDS_PER_CHUNK,
        "batch_size": batch_size,
        "training_chunks": train_chunks,
        "validation_chunks": validation_chunks,
        "emotion_feature_columns": emotion_columns,
        "candidates": metadata_candidates,
        "selected_candidate": selected_candidate,
        "selection_rule": "higher macro AUROC, then higher macro F1",
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(metadata), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/model1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models/model1_local_llm_stacker"))
    parser.add_argument("--report", type=Path, default=Path("reports/model-evaluation/model1-local-llm-stacker-report.md"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/model1-local-llm-stacker.json"))
    parser.add_argument("--train-scores", type=Path, default=Path("outputs/model-evaluation/model1_local_llm_train_scores.csv"))
    parser.add_argument("--validation-scores", type=Path, default=Path("outputs/model-evaluation/model1_local_llm_validation_scores.csv"))
    parser.add_argument("--piastra-predictions", type=Path, default=Path("outputs/model-evaluation/model1_piastra_validation_predictions.csv"))
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    metadata = train_stacker(
        args.data_dir, args.output_dir, args.report, args.metadata, args.train_scores,
        args.validation_scores, args.piastra_predictions, args.model_name, args.batch_size, args.device,
    )
    print(f"Model 1 local-LLM stacker complete; selected {metadata['selected_candidate']} using validation only.")


if __name__ == "__main__":
    main()
