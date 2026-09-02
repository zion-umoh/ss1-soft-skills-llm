"""Score Essays locally with an Ollama-hosted LLM for the Model 1 stacker.

This runner keeps all essay text on the local machine.  It sends one
zero-temperature request per essay to the local Ollama service and requests all
five Big Five scores in a single JSON response.  Train and validation must be
scored by the same local model before the supervised stacker is fitted; test is
intentionally not an accepted default input.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn

try:
    from src.models.score_model1_piastra import (
        RECORD_ID_COLUMN,
        SCORE_THRESHOLD,
        TEXT_COLUMN,
        TRAITS,
        ScoringError,
        binary_predictions,
        classification_metrics,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    from score_model1_piastra import (
        RECORD_ID_COLUMN,
        SCORE_THRESHOLD,
        TEXT_COLUMN,
        TRAITS,
        ScoringError,
        binary_predictions,
        classification_metrics,
    )


DEFAULT_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


def load_essays(path: Path, expected_split: str) -> list[dict[str, object]]:
    """Load only the fixed requested split, retaining labels for later evaluation."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {RECORD_ID_COLUMN, TEXT_COLUMN, "split", *TRAITS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ScoringError(f"{path} does not contain the expected Essays model schema.")
        rows = list(reader)
    if not rows or any(row["split"] != expected_split for row in rows):
        raise ScoringError(f"{path} is empty or contains rows outside {expected_split!r}.")
    records = []
    for row in rows:
        try:
            labels = {trait: int(row[trait]) for trait in TRAITS}
        except ValueError as error:
            raise ScoringError(f"{path} contains a non-binary trait label.") from error
        if any(value not in (0, 1) for value in labels.values()):
            raise ScoringError(f"{path} contains a non-binary trait label.")
        records.append({"record_id": row[RECORD_ID_COLUMN], "text": row[TEXT_COLUMN], "labels": labels})
    return records


def response_schema() -> dict[str, object]:
    """Return Ollama's JSON schema for the five numeric trait scores."""
    properties = {f"{trait}_score": {"type": "number", "minimum": 0, "maximum": 10} for trait in TRAITS}
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": list(properties)}


def prompt_for(text: str) -> str:
    """Create a label-free, compact instruction for one independent essay."""
    return (
        "Estimate the author's Big Five personality from the written text alone. "
        "Score extraversion, neuroticism, agreeableness, conscientiousness, and openness from 0 "
        "(very low) to 10 (very high). Do not infer demographic attributes. "
        "Return only the requested JSON object, with no explanation or reasoning.\n\n"
        f"TEXT:\n{text}"
    )


def parse_scores(response: dict[str, object]) -> dict[str, float]:
    """Validate the local model's structured response before it enters an artifact."""
    try:
        message = response["message"]
        content = message["content"] if isinstance(message, dict) else None
        payload = json.loads(content) if isinstance(content, str) else None
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ScoringError("Ollama returned malformed JSON despite the requested schema.") from error
    values: dict[str, float] = {}
    for trait in TRAITS:
        field = f"{trait}_score"
        try:
            value = float(payload[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ScoringError(f"Ollama response omitted numeric field {field!r}.") from error
        if not 0 <= value <= 10:
            raise ScoringError(f"Ollama response field {field!r} was outside 0--10.")
        values[field] = value
    return values


def call_ollama(url: str, model: str, prompt: str, max_retries: int = 2) -> dict[str, object]:
    """Call localhost only; no credential or external API request is involved."""
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "format": response_schema(),
        "messages": [
            {"role": "system", "content": "Return valid JSON only. Do not expose reasoning."},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0, "num_predict": 80},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            if 500 <= error.code < 600 and attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise ScoringError(f"Local Ollama service returned HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise ScoringError(
                f"Could not reach local Ollama at {url}. Start Ollama and ensure model {model!r} is installed."
            ) from error
    raise AssertionError("The retry loop must either return or raise.")


def prediction_fields() -> list[str]:
    fields = [RECORD_ID_COLUMN, "split"]
    fields += [f"label_{trait}" for trait in TRAITS]
    fields += [f"score_{trait}" for trait in TRAITS]
    fields += [f"prediction_{trait}" for trait in TRAITS]
    return fields


def load_partial_predictions(path: Path, split: str) -> list[dict[str, str]]:
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
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=prediction_fields())
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def scored_row(record: dict[str, object], split: str, model: str, url: str) -> dict[str, object]:
    scores = parse_scores(call_ollama(url, model, prompt_for(str(record["text"]))))
    values = np.asarray([[scores[f"{trait}_score"] for trait in TRAITS]], dtype=float)
    row: dict[str, object] = {RECORD_ID_COLUMN: record["record_id"], "split": split}
    row.update({f"label_{trait}": int(record["labels"][trait]) for trait in TRAITS})
    row.update({f"score_{trait}": f"{scores[f'{trait}_score']:.4f}" for trait in TRAITS})
    row.update({f"prediction_{trait}": int(value) for trait, value in zip(TRAITS, binary_predictions(values)[0], strict=True)})
    return row


def render_report(metadata: dict[str, object]) -> str:
    lines = [
        "# Model 1 Local-LLM Scores Report",
        "",
        "## Protocol",
        "",
        "- Inference ran through the local Ollama service; no external API calls or credentials were used.",
        "- Each essay produces one zero-temperature structured JSON response containing all five 0--10 Big Five scores.",
        "- The prompt does not include labels, demographics, IDs, GoEmotions features, or training examples.",
        "- Results are a feature-generation input to a separately trained, train-only stacker; this runner never fits on validation labels.",
        "",
    ]
    if metadata["metrics"] is not None:
        metrics = metadata["metrics"]
        lines.extend([
            "## Validation results",
            "",
            "| Exact-match accuracy | Macro F1 |",
            "| ---: | ---: |",
            f"| {metrics['exact_match_accuracy']:.4f} | {metrics['macro_f1']:.4f} |",
            "",
        ])
    else:
        lines.extend(["This is a train-only feature artifact; no training-set metrics are used for selection.", ""])
    return "\n".join(lines)


def score_split(
    essays_path: Path, split: str, model: str, url: str, prediction_path: Path,
    report_path: Path, metadata_path: Path, partial_path: Path, limit: int | None,
) -> dict[str, object]:
    records = load_essays(essays_path, split)
    if limit is not None:
        if limit < 1:
            raise ScoringError("limit must be at least 1 when provided.")
        records = records[:limit]
    completed_rows = load_partial_predictions(partial_path, split)
    completed_ids = {row[RECORD_ID_COLUMN] for row in completed_rows}
    expected_ids = {str(record["record_id"]) for record in records}
    if not completed_ids.issubset(expected_ids):
        raise ScoringError("Partial predictions do not match the requested scoring subset.")
    for record in records:
        if str(record["record_id"]) in completed_ids:
            continue
        row = scored_row(record, split, model, url)
        append_partial_prediction(partial_path, row)
        completed_rows.append({key: str(value) for key, value in row.items()})
        print(f"Scored {len(completed_rows)}/{len(records)} {split} essays locally.", flush=True)
    if len(completed_rows) != len(records):
        raise ScoringError(f"Only {len(completed_rows)} of {len(records)} records were scored.")
    labels = np.asarray([[int(row[f"label_{trait}"]) for trait in TRAITS] for row in completed_rows], dtype=np.int8)
    score_values = np.asarray([[float(row[f"score_{trait}"]) for trait in TRAITS] for row in completed_rows], dtype=float)
    selection_metrics = classification_metrics(labels, binary_predictions(score_values)) if split == "validation" else None
    metadata: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "inference_location": "local Ollama service only",
        "ollama_model": model,
        "ollama_url": url,
        "prompt_mode": "zero-shot independent text scoring, JSON schema",
        "score_scale": "0-10",
        "binary_positive_threshold": SCORE_THRESHOLD,
        "split": split,
        "rows_scored": len(records),
        "is_pilot": limit is not None,
        "metrics": selection_metrics,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=prediction_fields())
        writer.writeheader()
        writer.writerows(completed_rows)
    partial_path.unlink(missing_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(metadata), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--essays", type=Path, default=Path("data/processed/model1/essays_validation.csv"))
    parser.add_argument("--split", default="validation", choices=("train", "validation"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--predictions", type=Path, default=Path("outputs/model-evaluation/model1_local_llm_validation_scores.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/model-evaluation/model1-local-llm-scores-report.md"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/model1-local-llm-validation-scores.json"))
    parser.add_argument("--partial-predictions", type=Path, default=Path("outputs/model-evaluation/model1_local_llm_validation_scores.partial.csv"))
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N fixed-split records for a prompt-compliance pilot.")
    args = parser.parse_args()
    metadata = score_split(
        args.essays, args.split, args.model, args.ollama_url, args.predictions, args.report,
        args.metadata, args.partial_predictions, args.limit,
    )
    print(f"Scored {metadata['rows_scored']} Essays {args.split} rows locally with {metadata['ollama_model']}.")


if __name__ == "__main__":
    main()
