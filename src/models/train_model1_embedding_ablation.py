"""Compare sentence-embedding Model 1 variants using train and validation only.

Long Essays are split into conservative word chunks, embedded independently with
all-MiniLM-L6-v2, then mean-pooled to one document embedding.  The study then
compares semantic embeddings alone and embeddings plus GoEmotions probabilities
against the completed zero-shot paper benchmark.  Test data are not read.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.models.train_model1_emotion_ablation import (
        SEED, load_emotions, load_essays, load_piastra_metrics, metrics, write_predictions,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    from train_model1_emotion_ablation import (
        SEED, load_emotions, load_essays, load_piastra_metrics, metrics, write_predictions,
    )


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
WORDS_PER_CHUNK = 180


def chunk_text(text: str, words_per_chunk: int = WORDS_PER_CHUNK) -> list[str]:
    """Break long documents into bounded units for mean-pooled embedding."""
    if words_per_chunk < 1:
        raise ValueError("words_per_chunk must be positive.")
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[start:start + words_per_chunk]) for start in range(0, len(words), words_per_chunk)]


def embed_documents(texts: list[str], model_name: str, batch_size: int, device: str | None) -> tuple[np.ndarray, int, str]:
    """Return L2-normalised mean-pooled document embeddings and provenance."""
    from sentence_transformers import SentenceTransformer

    document_chunk_counts = []
    chunks = []
    for text in texts:
        document_chunks = chunk_text(text)
        document_chunk_counts.append(len(document_chunks))
        chunks.extend(document_chunks)
    model = SentenceTransformer(model_name, device=device)
    chunk_embeddings = model.encode(
        chunks, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    )
    embeddings = []
    offset = 0
    for count in document_chunk_counts:
        document_embedding = chunk_embeddings[offset:offset + count].mean(axis=0)
        norm = np.linalg.norm(document_embedding)
        embeddings.append(document_embedding / norm if norm else document_embedding)
        offset += count
    return np.asarray(embeddings, dtype=np.float32), len(chunks), str(model.device)


def build_classifier() -> OneVsRestClassifier:
    return OneVsRestClassifier(
        make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=2_000, random_state=SEED),
        ),
        n_jobs=-1,
    )


def render_report(metadata: dict[str, object]) -> str:
    lines = [
        "# Model 1 Semantic-Embedding Ablation Report",
        "",
        "## Protocol",
        "",
        f"- Semantic model: `{metadata['embedding_model']}`, applied to bounded text chunks and mean-pooled to document embeddings.",
        "- Supervised candidates use the same standardised one-vs-rest logistic-regression classifier.",
        "- GoEmotions augmentation appends the same 28 probabilities used in the prior ablation.",
        "- Train and validation splits only; the Essays test split has not been read.",
        "- Selection criterion: highest validation macro AUROC, then macro F1 on a tie.",
        "",
        "## Validation comparison",
        "",
        "| Version | Macro AUROC | Macro F1 | Macro balanced accuracy | Exact-match accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, candidate in metadata["candidates"].items():
        result = candidate["validation_metrics"]
        lines.append(f"| {name} | {result['macro_auroc']:.4f} | {result['macro_f1']:.4f} | {result['macro_balanced_accuracy']:.4f} | {result['exact_match_accuracy']:.4f} |")
    lines.extend(
        [
            "",
            f"Provisional validation selection: **{metadata['selected_candidate']}**.",
            "",
            "## Interpretation boundary",
            "",
            "The semantic model was pretrained elsewhere and is not fine-tuned on the Essays labels. GoEmotions features represent emotion expressed in text, rather than a direct measure of emotion regulation. The paper-based zero-shot benchmark remains the reference; any supervised candidate is a distinct improvement family.",
            "",
        ]
    )
    return "\n".join(lines)


def train_embedding_ablation(
    data_dir: Path, output_dir: Path, report_path: Path, metadata_path: Path,
    piastra_predictions: Path, model_name: str, batch_size: int, device: str | None,
) -> dict[str, object]:
    train_ids, train_texts, train_labels = load_essays(data_dir / "essays_train.csv", "train")
    validation_ids, validation_texts, validation_labels = load_essays(data_dir / "essays_validation.csv", "validation")
    emotion_columns, train_emotions = load_emotions(data_dir / "essays_emotion_features_train.csv", train_ids, "train")
    validation_emotion_columns, validation_emotions = load_emotions(data_dir / "essays_emotion_features_validation.csv", validation_ids, "validation")
    if emotion_columns != validation_emotion_columns:
        raise ValueError("GoEmotions columns differ between train and validation.")
    train_embeddings, train_chunks, resolved_device = embed_documents(train_texts, model_name, batch_size, device)
    validation_embeddings, validation_chunks, validation_device = embed_documents(validation_texts, model_name, batch_size, device)
    if validation_device != resolved_device:
        raise RuntimeError("Embedding device changed between training and validation encoding.")
    candidates = {
        "semantic_embeddings_only": train_embeddings,
        "semantic_embeddings_plus_goemotions": np.hstack((train_embeddings, train_emotions)),
    }
    validation_features = {
        "semantic_embeddings_only": validation_embeddings,
        "semantic_embeddings_plus_goemotions": np.hstack((validation_embeddings, validation_emotions)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_metadata: dict[str, object] = {}
    for name, train_features in candidates.items():
        classifier = build_classifier()
        classifier.fit(train_features, train_labels)
        probabilities = classifier.predict_proba(validation_features[name])
        item = {
            "feature_count": int(train_features.shape[1]),
            "validation_metrics": metrics(validation_labels, probabilities),
            "model_settings": {"classifier": "StandardScaler + class-balanced LogisticRegression", "C": 0.5, "random_seed": SEED},
        }
        candidate_metadata[name] = item
        joblib.dump({"classifier": classifier, "candidate": name, "metadata": item}, output_dir / f"{name}.joblib")
        write_predictions(output_dir / f"{name}_validation_predictions.csv", validation_ids, validation_labels, probabilities)
    candidate_metadata["paper_zero_shot_piastra"] = load_piastra_metrics(piastra_predictions, validation_ids, validation_labels)
    selected_candidate = max(
        candidate_metadata,
        key=lambda name: (candidate_metadata[name]["validation_metrics"]["macro_auroc"], candidate_metadata[name]["validation_metrics"]["macro_f1"]),
    )
    metadata: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "embedding_model": model_name,
        "embedding_device": resolved_device,
        "words_per_chunk": WORDS_PER_CHUNK,
        "batch_size": batch_size,
        "training_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "training_chunks": train_chunks,
        "validation_chunks": validation_chunks,
        "test_rows_used": 0,
        "emotion_feature_columns": emotion_columns,
        "candidates": candidate_metadata,
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models/model1_embedding_ablation"))
    parser.add_argument("--report", type=Path, default=Path("reports/model-evaluation/model1-semantic-embedding-ablation-report.md"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/model1-semantic-embedding-ablation.json"))
    parser.add_argument("--piastra-predictions", type=Path, default=Path("outputs/model-evaluation/model1_piastra_validation_predictions.csv"))
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="SentenceTransformer device such as cpu or mps; default chooses automatically.")
    args = parser.parse_args()
    metadata = train_embedding_ablation(
        args.data_dir, args.output_dir, args.report, args.metadata, args.piastra_predictions,
        args.model_name, args.batch_size, args.device,
    )
    print(f"Model 1 embedding ablation complete; selected {metadata['selected_candidate']} using validation only.")


if __name__ == "__main__":
    main()
