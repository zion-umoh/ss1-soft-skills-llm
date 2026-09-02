"""Run the paper-based zero-shot LLM benchmark for Essays → Big Five.

This operationalises Piastra and Catellani (2025): each essay is independently
scored with a zero-shot prompt for five 0--10 Big Five estimates and confidence
scores.  The source dataset has binary labels, so this project adaptation uses
a predeclared score threshold of 5.0; it does not tune a threshold on validation
or test labels.  Default execution scores validation only, preserving the
held-out test split for the selected model's single final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


TRAITS = ("cEXT", "cNEU", "cAGR", "cCON", "cOPN")
TRAIT_NAMES = {
    "cEXT": "extraversion",
    "cNEU": "neuroticism",
    "cAGR": "agreeableness",
    "cCON": "conscientiousness",
    "cOPN": "openness",
}
RECORD_ID_COLUMN = "record_id"
TEXT_COLUMN = "text_model"
SCORE_THRESHOLD = 5.0


class ScoringError(ValueError):
    """Raised when a scoring response or prepared split violates the contract."""


def load_env_value(path: Path, name: str) -> str | None:
    """Read one simple dotenv assignment without executing any file content."""
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value or None
    return None


def trusted_ssl_context() -> ssl.SSLContext:
    """Use a present system CA bundle; never disable certificate verification."""
    configured_bundle = os.environ.get("SSL_CERT_FILE")
    candidates = [configured_bundle] if configured_bundle else []
    candidates.append("/etc/ssl/cert.pem")
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ssl.create_default_context(cafile=candidate)
    return ssl.create_default_context()


def load_essays(path: Path, expected_split: str) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {RECORD_ID_COLUMN, TEXT_COLUMN, "split", *TRAITS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ScoringError(f"{path} does not contain the expected Essays model schema.")
        rows = list(reader)
    if not rows:
        raise ScoringError(f"{path} has no rows.")
    if any(row["split"] != expected_split for row in rows):
        raise ScoringError(f"{path} contains rows outside the {expected_split!r} split.")
    records = []
    for row in rows:
        try:
            labels = {trait: int(row[trait]) for trait in TRAITS}
        except ValueError as error:
            raise ScoringError(f"{path} contains a non-binary trait label.") from error
        if any(label not in (0, 1) for label in labels.values()):
            raise ScoringError(f"{path} contains a non-binary trait label.")
        records.append({"record_id": row[RECORD_ID_COLUMN], "text": row[TEXT_COLUMN], "labels": labels})
    return records


def prompt_for(text: str) -> str:
    """Return the zero-shot personality-estimation prompt without target labels."""
    return (
        "Read the following written text and estimate its author's Big Five personality traits. "
        "Provide one numerical estimate from 0 (very low) to 10 (very high), plus a confidence from 0 to 10, "
        "for extraversion, neuroticism, agreeableness, conscientiousness, and openness. "
        "Base your response only on this text; do not infer demographic attributes.\n\n"
        f"TEXT:\n{text}"
    )


def response_schema() -> dict[str, object]:
    properties: dict[str, object] = {}
    for trait in TRAITS:
        properties[f"{trait}_score"] = {"type": "number", "minimum": 0, "maximum": 10}
        properties[f"{trait}_confidence"] = {"type": "number", "minimum": 0, "maximum": 10}
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": list(properties)}


def extract_output_text(response: dict[str, object]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    raise ScoringError("Responses API result did not contain output text.")


def parse_scores(response: dict[str, object]) -> dict[str, float]:
    try:
        payload = json.loads(extract_output_text(response))
    except json.JSONDecodeError as error:
        raise ScoringError("Responses API returned malformed JSON despite the requested schema.") from error
    values: dict[str, float] = {}
    for trait in TRAITS:
        for suffix in ("score", "confidence"):
            field = f"{trait}_{suffix}"
            try:
                value = float(payload[field])
            except (KeyError, TypeError, ValueError) as error:
                raise ScoringError(f"Response omitted numeric field {field!r}.") from error
            if not 0 <= value <= 10:
                raise ScoringError(f"Response field {field!r} was outside 0--10.")
            values[field] = value
    return values


def retry_delay_seconds(error: urllib.error.HTTPError, detail: str, attempt: int) -> float:
    """Respect an API retry hint, with a bounded fallback exponential delay."""
    retry_after = error.headers.get("retry-after") if error.headers else None
    if retry_after:
        try:
            return max(float(retry_after), 0.5)
        except ValueError:
            pass
    match = re.search(r"try again in\s+(\d+)ms", detail, flags=re.IGNORECASE)
    if match:
        return max(int(match.group(1)) / 1_000, 0.5)
    return min(2 ** attempt, 60.0)


def call_responses_api(api_key: str, model: str, prompt: str, max_retries: int = 25) -> dict[str, object]:
    request_body = {
        "model": model,
        "input": prompt,
        "temperature": 0,
        "store": False,
        "text": {"format": {"type": "json_schema", "name": "big_five_scores", "strict": True, "schema": response_schema()}},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120, context=trusted_ssl_context()) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            if error.code == 429 and attempt < max_retries:
                time.sleep(retry_delay_seconds(error, detail, attempt))
                continue
            raise ScoringError(f"Responses API returned HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise ScoringError(f"Responses API request failed: {error.reason}") from error
    raise AssertionError("The retry loop must either return or raise.")


def binary_predictions(scores: np.ndarray) -> np.ndarray:
    return (scores >= SCORE_THRESHOLD).astype(np.int8)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
    return {
        "exact_match_accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_f1": round(float(precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)[2]), 4),
        "per_trait": {
            trait: {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
            for index, trait in enumerate(TRAITS)
        },
    }


def write_predictions(path: Path, rows: list[dict[str, object]], split: str) -> None:
    fields = [RECORD_ID_COLUMN, "split"]
    fields += [f"label_{trait}" for trait in TRAITS]
    fields += [f"score_{trait}" for trait in TRAITS]
    fields += [f"confidence_{trait}" for trait in TRAITS]
    fields += [f"prediction_{trait}" for trait in TRAITS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def prediction_fields() -> list[str]:
    fields = [RECORD_ID_COLUMN, "split"]
    fields += [f"label_{trait}" for trait in TRAITS]
    fields += [f"score_{trait}" for trait in TRAITS]
    fields += [f"confidence_{trait}" for trait in TRAITS]
    fields += [f"prediction_{trait}" for trait in TRAITS]
    return fields


def load_partial_predictions(path: Path, split: str) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != prediction_fields():
            raise ScoringError(f"Partial prediction file {path} does not match the expected schema.")
        rows = list(reader)
    if any(row["split"] != split for row in rows):
        raise ScoringError(f"Partial prediction file {path} contains another split.")
    if len({row[RECORD_ID_COLUMN] for row in rows}) != len(rows):
        raise ScoringError(f"Partial prediction file {path} has duplicate record IDs.")
    return rows


def append_partial_prediction(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=prediction_fields())
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def scored_row(record: dict[str, object], split: str, api_key: str, model: str) -> dict[str, object]:
    result = parse_scores(call_responses_api(api_key, model, prompt_for(str(record["text"]))))
    trait_scores = [result[f"{trait}_score"] for trait in TRAITS]
    predictions = binary_predictions(np.asarray([trait_scores]))[0]
    label_values = [int(record["labels"][trait]) for trait in TRAITS]
    row: dict[str, object] = {RECORD_ID_COLUMN: record["record_id"], "split": split}
    row.update({f"label_{trait}": label_values[index] for index, trait in enumerate(TRAITS)})
    row.update({f"score_{trait}": f"{trait_scores[index]:.4f}" for index, trait in enumerate(TRAITS)})
    row.update({f"confidence_{trait}": f"{result[f'{trait}_confidence']:.4f}" for trait in TRAITS})
    row.update({f"prediction_{trait}": int(predictions[index]) for index, trait in enumerate(TRAITS)})
    return row


def render_report(metadata: dict[str, object]) -> str:
    results = metadata["metrics"]
    lines = [
        "# Model 1 Piastra & Catellani Zero-Shot Benchmark Report",
        "",
        "## Benchmark source and adaptation",
        "",
        "Piastra, M. and Catellani, P. (2025), *On the emergent capabilities of ChatGPT 4 to estimate personality traits*, Frontiers in Artificial Intelligence, 8, 1484260. https://doi.org/10.3389/frai.2025.1484260",
        "",
        "The paper uses a zero-shot GPT-4 assessment of written text and 0--10 trait/confidence scores. This project retains that text-only, independent zero-shot scoring procedure, but evaluates the Essays dataset's five binary labels. A fixed score ≥ 5.0 becomes a positive binary prediction; labels are never sent to the model and no validation/test threshold is tuned.",
        "",
        "## Protocol",
        "",
        "- Input: `text_model` only; no author IDs, demographic information, or GoEmotions features.",
        "- Each text is scored independently using the recorded zero-shot prompt.",
        "- The paper used GPT-4. The exact API model used for this operational replication is recorded in metadata, because archived GPT-4 deployments may not remain callable.",
        "- Evaluation: fixed validation split only. The held-out test split has not been used.",
        "",
        "## Validation results",
        "",
        "| Exact-match accuracy | Macro F1 |",
        "| ---: | ---: |",
        f"| {results['exact_match_accuracy']:.4f} | {results['macro_f1']:.4f} |",
        "",
        "| Trait | Precision | Recall | F1 | Positive support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for trait in TRAITS:
        result = results["per_trait"][trait]
        lines.append(f"| {trait} | {result['precision']:.4f} | {result['recall']:.4f} | {result['f1']:.4f} | {result['support']} |")
    lines.append("")
    return "\n".join(lines)


def score_split(
    essays_path: Path,
    split: str,
    api_key: str,
    model: str,
    prediction_path: Path,
    report_path: Path,
    metadata_path: Path,
    partial_path: Path,
    workers: int,
) -> dict[str, object]:
    records = load_essays(essays_path, split)
    output_rows = load_partial_predictions(partial_path, split)
    completed_ids = {row[RECORD_ID_COLUMN] for row in output_rows}
    pending = [record for record in records if record["record_id"] not in completed_ids]
    if workers < 1:
        raise ScoringError("workers must be at least 1.")
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(scored_row, record, split, api_key, model): record["record_id"] for record in pending}
            for future in as_completed(futures):
                row = future.result()
                append_partial_prediction(partial_path, row)
                output_rows.append(row)
                print(f"Scored {len(output_rows)}/{len(records)} {split} essays.", flush=True)
    if len(output_rows) != len(records):
        raise ScoringError(f"Only {len(output_rows)} of {len(records)} records were scored.")
    labels = [[int(row[f"label_{trait}"]) for trait in TRAITS] for row in output_rows]
    scores = [[float(row[f"score_{trait}"]) for trait in TRAITS] for row in output_rows]
    label_array = np.asarray(labels, dtype=np.int8)
    score_array = np.asarray(scores, dtype=float)
    metric_values = classification_metrics(label_array, binary_predictions(score_array))
    metadata: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "benchmark_paper": "Piastra & Catellani (2025), doi:10.3389/frai.2025.1484260",
        "api_model": model,
        "prompt_mode": "zero-shot independent text scoring",
        "score_scale": "0-10",
        "binary_positive_threshold": SCORE_THRESHOLD,
        "split": split,
        "rows_scored": len(records),
        "metrics": metric_values,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    write_predictions(prediction_path, output_rows, split)
    partial_path.unlink(missing_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(metadata), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--essays", type=Path, default=Path("data/processed/model1/essays_validation.csv"))
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--model", default="gpt-4.1", help="Callable GPT-4-family API model to record in the benchmark metadata.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Local dotenv file; it is never written to outputs.")
    parser.add_argument("--confirm-remote-inference", action="store_true", help="Required acknowledgement before texts are sent to the API.")
    parser.add_argument("--predictions", type=Path, default=Path("outputs/model-evaluation/model1_piastra_validation_predictions.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/model-evaluation/model1-piastra-benchmark-report.md"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/model1-piastra-benchmark.json"))
    parser.add_argument("--partial-predictions", type=Path, default=Path("outputs/model-evaluation/model1_piastra_validation_predictions.partial.csv"))
    parser.add_argument("--workers", type=int, default=3, help="Independent concurrent API calls; partial results are checkpointed.")
    args = parser.parse_args()
    if not args.confirm_remote_inference:
        raise SystemExit("Refusing remote inference without --confirm-remote-inference. This sends Essays text to the selected API model.")
    api_key = os.environ.get(args.api_key_env) or load_env_value(args.env_file, args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key in environment variable {args.api_key_env!r}.")
    metadata = score_split(
        args.essays, args.split, api_key, args.model, args.predictions, args.report, args.metadata,
        args.partial_predictions, args.workers,
    )
    print(f"Scored {metadata['rows_scored']} Essays {args.split} rows; macro F1={metadata['metrics']['macro_f1']:.4f}.")


if __name__ == "__main__":
    main()
