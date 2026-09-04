"""Compare supervised Essays text and GoEmotions feature variants on validation.

The Piastra & Catellani zero-shot LLM result remains the paper benchmark.  This
module is a controlled improvement experiment: it holds the supervised model
constant and compares text-only, GoEmotions-only, and text-plus-GoEmotions
features using only fixed train and validation splits.  It never reads test.
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
import scipy.sparse
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier


TRAITS = ("cEXT", "cNEU", "cAGR", "cCON", "cOPN")
RECORD_ID_COLUMN = "record_id"
TEXT_COLUMN = "text_model"
SEED = 42


class TrainingError(ValueError):
    """Raised when fixed Essays or emotion-feature files violate the contract."""


def load_essays(path: Path, expected_split: str) -> tuple[list[str], list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {RECORD_ID_COLUMN, TEXT_COLUMN, "split", *TRAITS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise TrainingError(f"{path} does not contain the expected Essays schema.")
        rows = list(reader)
    if not rows or any(row["split"] != expected_split for row in rows):
        raise TrainingError(f"{path} is empty or contains rows outside {expected_split!r}.")
    labels = np.asarray([[int(row[trait]) for trait in TRAITS] for row in rows], dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise TrainingError(f"{path} contains non-binary trait labels.")
    return [row[RECORD_ID_COLUMN] for row in rows], [row[TEXT_COLUMN] for row in rows], labels


def load_emotions(path: Path, record_ids: list[str], expected_split: str) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise TrainingError(f"{path} has no header.")
        emotion_columns = [column for column in reader.fieldnames if column.startswith("emotion_probability_")]
        required = {RECORD_ID_COLUMN, "split", *emotion_columns}
        if len(emotion_columns) != 28 or not required.issubset(reader.fieldnames):
            raise TrainingError(f"{path} does not contain exactly 28 GoEmotions probability columns.")
        rows = list(reader)
    if any(row["split"] != expected_split for row in rows):
        raise TrainingError(f"{path} contains rows outside {expected_split!r}.")
    by_id = {row[RECORD_ID_COLUMN]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(record_ids):
        raise TrainingError(f"{path} does not join one-to-one with its Essays split by record_id.")
    try:
        values = np.asarray([[float(by_id[record_id][column]) for column in emotion_columns] for record_id in record_ids], dtype=np.float32)
    except ValueError as error:
        raise TrainingError(f"{path} contains a non-numeric emotion probability.") from error
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise TrainingError(f"{path} has a probability outside 0--1.")
    return emotion_columns, values


def build_classifier() -> OneVsRestClassifier:
    return OneVsRestClassifier(
        SGDClassifier(
            loss="log_loss", alpha=1e-5, max_iter=1_000, tol=1e-3,
            class_weight="balanced", average=True, random_state=SEED,
        ),
        n_jobs=-1,
    )


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predictions = (probabilities >= 0.5).astype(np.int8)
    _, _, macro_f1, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    per_trait = {}
    for index, trait in enumerate(TRAITS):
        _, _, f1, _ = precision_recall_fscore_support(labels[:, index], predictions[:, index], average="binary", zero_division=0)
        per_trait[trait] = {
            "f1": round(float(f1), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(labels[:, index], predictions[:, index])), 4),
            "auroc": round(float(roc_auc_score(labels[:, index], probabilities[:, index])), 4),
            "positive_support": int(labels[:, index].sum()),
        }
    return {
        "macro_f1": round(float(macro_f1), 4),
        "macro_balanced_accuracy": round(float(np.mean([item["balanced_accuracy"] for item in per_trait.values()])), 4),
        "macro_auroc": round(float(np.mean([item["auroc"] for item in per_trait.values()])), 4),
        "exact_match_accuracy": round(float(np.mean(np.all(labels == predictions, axis=1))), 4),
        "per_trait": per_trait,
    }


def write_predictions(path: Path, record_ids: list[str], labels: np.ndarray, probabilities: np.ndarray) -> None:
    fields = [RECORD_ID_COLUMN, "split"]
    fields += [f"label_{trait}" for trait in TRAITS]
    fields += [f"probability_{trait}" for trait in TRAITS]
    fields += [f"prediction_{trait}" for trait in TRAITS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, record_id in enumerate(record_ids):
            row: dict[str, object] = {RECORD_ID_COLUMN: record_id, "split": "validation"}
            row.update({f"label_{trait}": int(labels[index, trait_index]) for trait_index, trait in enumerate(TRAITS)})
            row.update({f"probability_{trait}": f"{probabilities[index, trait_index]:.8f}" for trait_index, trait in enumerate(TRAITS)})
            row.update({f"prediction_{trait}": int(probabilities[index, trait_index] >= 0.5) for trait_index, trait in enumerate(TRAITS)})
            writer.writerow(row)


def load_piastra_metrics(path: Path, record_ids: list[str], labels: np.ndarray, split: str = "validation") -> dict[str, object]:
    """Score the completed paper benchmark on the requested ranking-evaluation split."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {RECORD_ID_COLUMN, "split", *[f"score_{trait}" for trait in TRAITS]}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise TrainingError(f"{path} does not contain Piastra benchmark scores.")
        rows = list(reader)
    by_id = {row[RECORD_ID_COLUMN]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(record_ids) or any(row["split"] != split for row in rows):
        raise TrainingError(f"{path} does not align one-to-one with Essays {split} IDs.")
    try:
        probabilities = np.asarray([[float(by_id[record_id][f"score_{trait}"]) / 10 for trait in TRAITS] for record_id in record_ids])
    except ValueError as error:
        raise TrainingError(f"{path} contains a non-numeric Piastra score.") from error
    return {"feature_count": None, "validation_metrics": metrics(labels, probabilities), "model_settings": {"type": "paper-based zero-shot LLM benchmark"}}


def render_report(metadata: dict[str, object]) -> str:
    rows = metadata["candidates"]
    lines = [
        "# Model 1 Emotion-Feature Ablation Report",
        "",
        "## Protocol",
        "",
        "- Fixed Essays training and validation splits only; no held-out test data were read.",
        "- Text representation: word and bigram TF-IDF fitted on Essays training text only.",
        "- Emotion representation: 28 continuous GoEmotions probabilities, generated by the separately trained GoEmotions model.",
        "- Classifier: the same class-balanced one-vs-rest logistic SGD classifier in every supervised candidate.",
        "- Binary threshold: fixed at 0.50 for every supervised candidate; no validation threshold tuning.",
        "- Selection criterion: highest validation macro AUROC, then macro F1 on a tie.",
        "",
        "## Validation comparison",
        "",
        "| Version | Macro AUROC | Macro F1 | Macro balanced accuracy | Exact-match accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, candidate in rows.items():
        result = candidate["validation_metrics"]
        lines.append(f"| {name} | {result['macro_auroc']:.4f} | {result['macro_f1']:.4f} | {result['macro_balanced_accuracy']:.4f} | {result['exact_match_accuracy']:.4f} |")
    lines.extend(
        [
            "",
            f"Provisional validation selection: **{metadata['selected_candidate']}**.",
            "",
            "## Interpretation boundary",
            "",
            "GoEmotions probabilities represent emotion expressed in text, not a direct measure of emotion regulation. The Reddit-to-Essays transfer is a domain-shift limitation. The text-only versus fusion comparison isolates the added predictive value of this emotion-feature layer within the supervised model family; it does not claim that emotion features were added to the Piastra & Catellani zero-shot LLM benchmark.",
            "",
        ]
    )
    return "\n".join(lines)


def train_ablation(
    data_dir: Path, output_dir: Path, report_path: Path, metadata_path: Path, piastra_predictions: Path
) -> dict[str, object]:
    train_ids, train_texts, train_labels = load_essays(data_dir / "essays_train.csv", "train")
    validation_ids, validation_texts, validation_labels = load_essays(data_dir / "essays_validation.csv", "validation")
    emotion_columns, train_emotions = load_emotions(data_dir / "essays_emotion_features_train.csv", train_ids, "train")
    validation_emotion_columns, validation_emotions = load_emotions(data_dir / "essays_emotion_features_validation.csv", validation_ids, "validation")
    if emotion_columns != validation_emotion_columns:
        raise TrainingError("GoEmotions feature columns differ between train and validation.")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.995, max_features=100_000, sublinear_tf=True, dtype=np.float32)
    train_text_matrix = vectorizer.fit_transform(train_texts)
    validation_text_matrix = vectorizer.transform(validation_texts)
    train_emotion_matrix = scipy.sparse.csr_matrix(train_emotions)
    validation_emotion_matrix = scipy.sparse.csr_matrix(validation_emotions)
    candidates = {
        "supervised_text_only": (train_text_matrix, validation_text_matrix),
        "goemotions_only": (train_emotion_matrix, validation_emotion_matrix),
        "supervised_text_plus_goemotions": (
            scipy.sparse.hstack((train_text_matrix, train_emotion_matrix), format="csr"),
            scipy.sparse.hstack((validation_text_matrix, validation_emotion_matrix), format="csr"),
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_candidates: dict[str, object] = {}
    for name, (train_matrix, validation_matrix) in candidates.items():
        classifier = build_classifier()
        classifier.fit(train_matrix, train_labels)
        probabilities = classifier.predict_proba(validation_matrix)
        candidate_metadata = {
            "feature_count": int(train_matrix.shape[1]),
            "validation_metrics": metrics(validation_labels, probabilities),
            "model_settings": {
                "base_estimator": "class-balanced SGDClassifier(loss=log_loss, average=True)",
                "alpha": 1e-5,
                "max_iter": 1_000,
                "random_seed": SEED,
            },
        }
        metadata_candidates[name] = candidate_metadata
        joblib.dump({"vectorizer": vectorizer, "classifier": classifier, "candidate": name, "metadata": candidate_metadata}, output_dir / f"{name}.joblib")
        write_predictions(output_dir / f"{name}_validation_predictions.csv", validation_ids, validation_labels, probabilities)
    metadata_candidates["paper_zero_shot_piastra"] = load_piastra_metrics(
        piastra_predictions, validation_ids, validation_labels
    )
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
        "text_features": {"ngram_range": [1, 2], "min_df": 2, "max_df": 0.995, "max_features": 100_000, "vocabulary_size": len(vectorizer.vocabulary_)},
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models/model1_ablation"))
    parser.add_argument("--report", type=Path, default=Path("reports/model-evaluation/model1-emotion-ablation-report.md"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/model1-emotion-ablation.json"))
    parser.add_argument("--piastra-predictions", type=Path, default=Path("outputs/model-evaluation/model1_piastra_validation_predictions.csv"))
    args = parser.parse_args()
    metadata = train_ablation(args.data_dir, args.output_dir, args.report, args.metadata, args.piastra_predictions)
    print(f"Model 1 ablation complete; selected {metadata['selected_candidate']} using validation only.")


if __name__ == "__main__":
    main()
