"""Train a text-only GoEmotions baseline and create Essays emotion features.

The model is fit only on the GoEmotions training split. Per-label probability
thresholds are selected only on validation data and then used unchanged for the
held-out test evaluation and Essays feature generation.
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier


EMOTIONS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
)
TEXT_COLUMN = "text_model"
RECORD_ID_COLUMN = "record_id"
SEED = 42


class TrainingError(ValueError):
    """Raised when model inputs do not meet the training contract."""


def load_labeled_data(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {RECORD_ID_COLUMN, TEXT_COLUMN, *EMOTIONS, "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise TrainingError(f"{path} does not contain the expected GoEmotions model schema.")
        rows = list(reader)
    if not rows:
        raise TrainingError(f"{path} has no rows.")
    ids = [row[RECORD_ID_COLUMN] for row in rows]
    texts = [row[TEXT_COLUMN] for row in rows]
    labels = np.asarray([[int(row[emotion]) for emotion in EMOTIONS] for row in rows], dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise TrainingError(f"{path} contains non-binary emotion labels.")
    return ids, texts, labels


def load_essay_texts(path: Path) -> tuple[list[str], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {RECORD_ID_COLUMN, TEXT_COLUMN, "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise TrainingError(f"{path} does not contain the expected Essays model schema.")
        rows = list(reader)
    return [row[RECORD_ID_COLUMN] for row in rows], [row[TEXT_COLUMN] for row in rows]


def build_model() -> tuple[TfidfVectorizer, OneVsRestClassifier]:
    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    classifier = OneVsRestClassifier(
        SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=1_000,
            tol=1e-3,
            class_weight="balanced",
            average=True,
            random_state=SEED,
        ),
        n_jobs=-1,
    )
    return vectorizer, classifier


def calibrate_f1_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Choose one F1-maximising threshold per label using validation data only."""
    thresholds = np.full(len(EMOTIONS), 0.5, dtype=np.float64)
    for index in range(len(EMOTIONS)):
        if not y_true[:, index].any():
            continue
        precision, recall, candidates = precision_recall_curve(y_true[:, index], probabilities[:, index])
        if not len(candidates):
            continue
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        best = np.flatnonzero(f1 == np.nanmax(f1))
        if not len(best):
            continue
        # Resolve F1 ties toward the conventional 0.5 threshold.
        chosen = min(best, key=lambda item: abs(candidates[item] - 0.5))
        thresholds[index] = candidates[chosen]
    return thresholds


def threshold_predictions(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (probabilities >= thresholds.reshape(1, -1)).astype(np.int8)


def metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, average=None, zero_division=0
    )
    macro = precision_recall_fscore_support(y_true, predictions, average="macro", zero_division=0)
    micro = precision_recall_fscore_support(y_true, predictions, average="micro", zero_division=0)
    return {
        "macro": {"precision": round(float(macro[0]), 4), "recall": round(float(macro[1]), 4), "f1": round(float(macro[2]), 4)},
        "micro": {"precision": round(float(micro[0]), 4), "recall": round(float(micro[1]), 4), "f1": round(float(micro[2]), 4)},
        "per_label": {
            emotion: {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
            for index, emotion in enumerate(EMOTIONS)
        },
    }


def write_probability_audit(
    path: Path,
    record_ids: list[str],
    split: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> None:
    fields = [RECORD_ID_COLUMN, "split"]
    fields += [f"label_{emotion}" for emotion in EMOTIONS]
    fields += [f"probability_{emotion}" for emotion in EMOTIONS]
    fields += [f"prediction_{emotion}" for emotion in EMOTIONS]
    rows = []
    for row_index, record_id in enumerate(record_ids):
        row: dict[str, object] = {RECORD_ID_COLUMN: record_id, "split": split}
        row.update({f"label_{emotion}": int(labels[row_index, index]) for index, emotion in enumerate(EMOTIONS)})
        row.update({f"probability_{emotion}": f"{probabilities[row_index, index]:.8f}" for index, emotion in enumerate(EMOTIONS)})
        row.update({f"prediction_{emotion}": int(predictions[row_index, index]) for index, emotion in enumerate(EMOTIONS)})
        rows.append(row)
    write_csv(path, fields, rows)


def write_essay_features(
    path: Path, record_ids: list[str], probabilities: np.ndarray, split: str
) -> None:
    fields = [RECORD_ID_COLUMN, "split", *[f"emotion_probability_{emotion}" for emotion in EMOTIONS]]
    rows = []
    for row_index, record_id in enumerate(record_ids):
        row: dict[str, object] = {RECORD_ID_COLUMN: record_id, "split": split}
        row.update(
            {
                f"emotion_probability_{emotion}": f"{probabilities[row_index, index]:.8f}"
                for index, emotion in enumerate(EMOTIONS)
            }
        )
        rows.append(row)
    write_csv(path, fields, rows)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_report(metadata: dict[str, object]) -> str:
    validation = metadata["validation_metrics"]
    test = metadata["test_metrics"]
    lines = [
        "# GoEmotions Text-Only Model Report",
        "",
        "## Protocol",
        "",
        "- Training data: custom leakage-safe GoEmotions training split only.",
        "- Input: `text_model` only; no Reddit metadata or source identifiers.",
        "- Model: word and bigram TF-IDF with class-balanced one-vs-rest logistic SGD classifiers.",
        "- Thresholds: one F1-maximising threshold per label, selected on validation only.",
        "- Test set: evaluated once using frozen validation thresholds.",
        "- Essays features: probability outputs from this frozen GoEmotions-trained model; no Big Five labels were used.",
        "",
        "## Aggregate results",
        "",
        "| Split | Macro F1 | Micro F1 | Macro precision | Macro recall |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Validation | {validation['macro']['f1']:.4f} | {validation['micro']['f1']:.4f} | {validation['macro']['precision']:.4f} | {validation['macro']['recall']:.4f} |",
        f"| Test | {test['macro']['f1']:.4f} | {test['micro']['f1']:.4f} | {test['macro']['precision']:.4f} | {test['macro']['recall']:.4f} |",
        "",
        "## Per-label test results and validation thresholds",
        "",
        "| Emotion | Threshold | Support | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, emotion in enumerate(EMOTIONS):
        result = test["per_label"][emotion]
        lines.append(
            f"| {emotion} | {metadata['thresholds'][emotion]:.4f} | {result['support']} | "
            f"{result['precision']:.4f} | {result['recall']:.4f} | {result['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Use in Model 1",
            "",
            "Join each `essays_emotion_features_<split>.csv` file to its matching Essays split by `record_id`.",
            "Use probability columns as continuous features. Do not use thresholded GoEmotions predictions as personality targets.",
            "",
        ]
    )
    return "\n".join(lines)


def train_and_generate(
    goemotions_dir: Path,
    essays_dir: Path,
    model_path: Path,
    evaluation_dir: Path,
    report_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    go_paths = {split: goemotions_dir / f"goemotions_{split}.csv" for split in ("train", "validation", "test")}
    train_ids, train_texts, train_labels = load_labeled_data(go_paths["train"])
    validation_ids, validation_texts, validation_labels = load_labeled_data(go_paths["validation"])
    test_ids, test_texts, test_labels = load_labeled_data(go_paths["test"])

    vectorizer, classifier = build_model()
    train_matrix = vectorizer.fit_transform(train_texts)
    classifier.fit(train_matrix, train_labels)
    validation_probabilities = classifier.predict_proba(vectorizer.transform(validation_texts))
    thresholds = calibrate_f1_thresholds(validation_labels, validation_probabilities)
    validation_predictions = threshold_predictions(validation_probabilities, thresholds)
    test_probabilities = classifier.predict_proba(vectorizer.transform(test_texts))
    test_predictions = threshold_predictions(test_probabilities, thresholds)

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    write_probability_audit(
        evaluation_dir / "goemotions_validation_predictions.csv",
        validation_ids, "validation", validation_labels, validation_probabilities, validation_predictions,
    )
    write_probability_audit(
        evaluation_dir / "goemotions_test_predictions.csv",
        test_ids, "test", test_labels, test_probabilities, test_predictions,
    )
    essay_feature_files: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        essay_ids, essay_texts = load_essay_texts(essays_dir / f"essays_{split}.csv")
        essay_probabilities = classifier.predict_proba(vectorizer.transform(essay_texts))
        feature_path = essays_dir / f"essays_emotion_features_{split}.csv"
        write_essay_features(feature_path, essay_ids, essay_probabilities, split)
        essay_feature_files[split] = feature_path.as_posix()

    metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model": {
            "type": "TF-IDF word/bigram + class-balanced OneVsRest SGD log-loss",
            "random_seed": SEED,
            "vocabulary_size": len(vectorizer.vocabulary_),
            "scikit_learn_version": sklearn.__version__,
            "python_version": platform.python_version(),
        },
        "training_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "test_rows": len(test_ids),
        "thresholds": {emotion: round(float(thresholds[index]), 8) for index, emotion in enumerate(EMOTIONS)},
        "validation_metrics": metrics(validation_labels, validation_predictions),
        "test_metrics": metrics(test_labels, test_predictions),
        "essay_feature_files": essay_feature_files,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier, "thresholds": thresholds, "emotions": EMOTIONS, "metadata": metadata}, model_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(metadata), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goemotions-dir", type=Path, default=Path("data/processed/model1"))
    parser.add_argument("--essays-dir", type=Path, default=Path("data/processed/model1"))
    parser.add_argument("--model", type=Path, default=Path("outputs/models/goemotions_tfidf_sgd.joblib"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("outputs/model-evaluation"))
    parser.add_argument("--report", type=Path, default=Path("reports/model-evaluation/goemotions-text-model-report.md"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/goemotions-emotion-model.json"))
    args = parser.parse_args()
    metadata = train_and_generate(
        args.goemotions_dir, args.essays_dir, args.model, args.evaluation_dir, args.report, args.metadata
    )
    print(
        f"Trained GoEmotions model on {metadata['training_rows']} rows; "
        f"test macro F1={metadata['test_metrics']['macro']['f1']:.4f}, "
        f"micro F1={metadata['test_metrics']['micro']['f1']:.4f}."
    )


if __name__ == "__main__":
    main()
