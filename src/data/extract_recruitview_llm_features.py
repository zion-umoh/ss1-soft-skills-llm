"""Extract reproducible linguistic and affective features with an OpenAI model.

The model is used as a frozen feature extractor. It receives only the interview
question and transcript and returns a fixed JSON object of observable text
features; it is never shown RecruitView labels and is not asked to predict
personality or BESSI directly.

The extraction is resumable. A successful response is appended to JSONL and the
progress manifest is flushed immediately, so rerunning the command skips rows
that are already complete. The final CSV is rebuilt from the checkpoint after
each run. If a process dies in the narrow window after an API response and
before its checkpoint write, at most that one request may be repeated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MODEL = "gpt-5.6-luna"
PROMPT_VERSION = "recruitview-llm-text-features-v1"
FEATURE_COLUMNS = (
    "emotional_expressiveness",
    "positive_affect",
    "negative_affect",
    "social_orientation",
    "assertiveness",
    "certainty",
    "cognitive_complexity",
    "response_elaboration",
    "interpersonal_warmth",
    "question_relevance",
)
SCORE_MIN = 0
SCORE_MAX = 5

SYSTEM_PROMPT = """You extract observable linguistic and affective characteristics from an interview response.

Return only the required JSON object. Score every feature from 0 to 5 using the
text evidence in the response and the question context. Use 2 when there is
insufficient evidence rather than guessing. These are text characteristics,
not diagnoses or judgments of the person.

Do not infer or output Big Five traits, BESSI skills, hiring suitability,
intelligence, protected characteristics, or any other personality label.

Feature meanings:
- emotional_expressiveness: explicit emotional language and emotional detail
- positive_affect: positive, enthusiastic or appreciative language
- negative_affect: worry, frustration, sadness or other negative-affect language
- social_orientation: references to cooperation, people, relationships or groups
- assertiveness: agency, initiative and confident action language
- certainty: confidence versus hedging or uncertainty in wording
- cognitive_complexity: explanation, comparison, qualification and reasoning detail
- response_elaboration: amount of relevant detail beyond a minimal answer
- interpersonal_warmth: friendly, respectful and considerate language
- question_relevance: how directly the response addresses the question
"""

FEATURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        column: {"type": "integer", "minimum": SCORE_MIN, "maximum": SCORE_MAX}
        for column in FEATURE_COLUMNS
    },
    "required": list(FEATURE_COLUMNS),
    "additionalProperties": False,
}


class LLMFeatureExtractionError(ValueError):
    """Raised when the input, response or checkpoint violates the contract."""


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_env_file(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries without overriding the shell environment."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"record_id", "participant_id", "question_id", "question", "text_model", "split"}
    if not rows or not required.issubset(rows[0]):
        raise LLMFeatureExtractionError(f"Prepared table must contain {sorted(required)}")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise LLMFeatureExtractionError("Prepared record IDs must be unique.")
    return rows


def _prompt_for(row: dict[str, str]) -> str:
    return f"QUESTION:\n{row['question']}\n\nINTERVIEW RESPONSE:\n{row['text_model']}"


def _input_hash(row: dict[str, str]) -> str:
    return _sha256(f"{PROMPT_VERSION}\n{SYSTEM_PROMPT}\n{_prompt_for(row)}")


def _validate_features(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(FEATURE_COLUMNS):
        raise LLMFeatureExtractionError("LLM output did not match the required feature schema.")
    validated: dict[str, int] = {}
    for column in FEATURE_COLUMNS:
        score = value[column]
        if isinstance(score, bool) or not isinstance(score, int) or not SCORE_MIN <= score <= SCORE_MAX:
            raise LLMFeatureExtractionError(f"Feature {column!r} must be an integer from {SCORE_MIN} to {SCORE_MAX}.")
        validated[column] = score
    return validated


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise LLMFeatureExtractionError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise LLMFeatureExtractionError(f"JSONL record at {path}:{line_number} is not an object.")
            values.append(value)
    return values


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], source_rows: list[dict[str, str]]) -> None:
    by_id = {row["record_id"]: row for row in rows}
    fields = ["record_id", "participant_id", "question_id", "split", *FEATURE_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for source in source_rows:
            extracted = by_id.get(source["record_id"])
            if extracted is None:
                continue
            writer.writerow(
                {
                    "record_id": source["record_id"],
                    "participant_id": source["participant_id"],
                    "question_id": source["question_id"],
                    "split": source["split"],
                    **{column: extracted[column] for column in FEATURE_COLUMNS},
                }
            )
    temporary.replace(path)


def _usage_value(usage: Any, name: str) -> int:
    value = getattr(usage, name, 0) if usage is not None else 0
    return int(value or 0)


def _extract_one(client: Any, row: dict[str, str], max_output_tokens: int) -> tuple[dict[str, int], dict[str, int], str]:
    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=_prompt_for(row),
        reasoning={"effort": "none"},
        text={
            "format": {
                "type": "json_schema",
                "name": "recruitview_text_features",
                "strict": True,
                "schema": FEATURE_SCHEMA,
            },
            "verbosity": "low",
        },
        max_output_tokens=max_output_tokens,
        temperature=0,
    )
    features = _validate_features(json.loads(response.output_text))
    usage = {
        "input_tokens": _usage_value(response.usage, "input_tokens"),
        "output_tokens": _usage_value(response.usage, "output_tokens"),
        "total_tokens": _usage_value(response.usage, "total_tokens"),
    }
    response_id = str(getattr(response, "id", ""))
    return features, usage, response_id


def run(
    prepared_path: Path,
    output_jsonl: Path,
    output_csv: Path,
    progress_path: Path,
    failures_path: Path,
    limit: int | None = None,
    max_retries: int = 5,
    max_output_tokens: int = 200,
    workers: int = 4,
    dry_run: bool = False,
) -> dict[str, Any]:
    source_rows = _read_rows(prepared_path)
    existing = _read_jsonl(output_jsonl)
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    contract = {
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": _sha256(SYSTEM_PROMPT),
        "schema_hash": _sha256(json.dumps(FEATURE_SCHEMA, sort_keys=True)),
    }
    if progress and any(progress.get(key) != value for key, value in contract.items()):
        raise LLMFeatureExtractionError("Checkpoint contract differs from the current model or prompt; use a new output directory.")
    completed = {row["record_id"]: row for row in existing}
    if any(row.get("record_id") not in {source["record_id"] for source in source_rows} for row in existing):
        raise LLMFeatureExtractionError("Checkpoint contains a record absent from the prepared table.")
    pending = [row for row in source_rows if row["record_id"] not in completed]
    if limit is not None:
        if limit < 0:
            raise LLMFeatureExtractionError("--limit must be non-negative.")
        pending = pending[:limit]

    manifest: dict[str, Any] = {
        **contract,
        "prepared_path": str(prepared_path),
        "output_jsonl": str(output_jsonl),
        "output_csv": str(output_csv),
        "total_rows": len(source_rows),
        "completed_rows": len(completed),
        "remaining_rows": len(source_rows) - len(completed),
        "unresolved_rows": len(source_rows) - len(completed),
        "feature_columns": list(FEATURE_COLUMNS),
        "score_range": [SCORE_MIN, SCORE_MAX],
        "last_record_id": progress.get("last_record_id"),
        "input_tokens": int(progress.get("input_tokens", 0)),
        "output_tokens": int(progress.get("output_tokens", 0)),
        "total_tokens": int(progress.get("total_tokens", 0)),
        "failed_rows": int(progress.get("failed_rows", 0)),
        "started_at_utc": progress.get("started_at_utc", _now()),
        "updated_at_utc": _now(),
        "status": "dry_run" if dry_run else ("complete" if not pending else "running"),
    }
    _write_json_atomic(progress_path, manifest)
    if dry_run:
        return manifest

    _load_env_file()
    if not os.environ.get("OPENAI_API_KEY"):
        raise LLMFeatureExtractionError("OPENAI_API_KEY is not configured; no paid request was made.")
    from openai import OpenAI

    if workers < 1:
        raise LLMFeatureExtractionError("--workers must be at least 1.")
    clients = threading.local()

    def process(row: dict[str, str]) -> dict[str, Any]:
        if not hasattr(clients, "client"):
            clients.client = OpenAI()
        error_message: str | None = None
        for attempt in range(max_retries + 1):
            try:
                features, usage, response_id = _extract_one(clients.client, row, max_output_tokens)
                record = {
                    "record_id": row["record_id"],
                    "participant_id": row["participant_id"],
                    "question_id": row["question_id"],
                    "split": row["split"],
                    "input_hash": _input_hash(row),
                    "model": MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "response_id": response_id,
                    "extracted_at_utc": _now(),
                    **features,
                }
                return {"record": record, "usage": usage}
            except Exception as error:  # API, transport, or structured-output failure.
                error_message = f"{type(error).__name__}: {error}"
                if attempt >= max_retries:
                    return {
                        "failure": {
                            "record_id": row["record_id"],
                            "input_hash": _input_hash(row),
                            "attempts": attempt + 1,
                            "error": error_message,
                            "failed_at_utc": _now(),
                        }
                    }
                else:
                    time.sleep(min(30.0, 2.0**attempt))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process, row) for row in pending]
        for future in as_completed(futures):
            result = future.result()
            if "record" in result:
                record = result["record"]
                _append_jsonl(output_jsonl, record)
                completed[record["record_id"]] = record
                usage = result["usage"]
                manifest["input_tokens"] += usage["input_tokens"]
                manifest["output_tokens"] += usage["output_tokens"]
                manifest["total_tokens"] += usage["total_tokens"]
                manifest["last_record_id"] = record["record_id"]
            else:
                _append_jsonl(failures_path, result["failure"])
                manifest["failed_rows"] += 1
            manifest["completed_rows"] = len(completed)
            manifest["remaining_rows"] = len(source_rows) - len(completed)
            manifest["unresolved_rows"] = len(source_rows) - len(completed)
            manifest["updated_at_utc"] = _now()
            _write_json_atomic(progress_path, manifest)

    manifest["status"] = "complete" if len(completed) == len(source_rows) else "partial"
    manifest["remaining_rows"] = len(source_rows) - len(completed)
    manifest["unresolved_rows"] = len(source_rows) - len(completed)
    manifest["updated_at_utc"] = _now()
    _write_csv(output_csv, list(completed.values()), source_rows)
    _write_json_atomic(progress_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/processed/recruitview/recruitview_prepared.csv"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/processed/recruitview/recruitview_llm_features.jsonl"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/recruitview/recruitview_llm_features.csv"))
    parser.add_argument("--progress", type=Path, default=Path("data/processed/recruitview/recruitview_llm_features_progress.json"))
    parser.add_argument("--failures", type=Path, default=Path("data/processed/recruitview/recruitview_llm_features_failures.jsonl"))
    parser.add_argument("--limit", type=int, help="Process at most this many pending rows; use for a pilot.")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent API requests; keep bounded to protect rate limits.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and checkpoint plan without calling the API.")
    args = parser.parse_args()
    result = run(
        prepared_path=args.prepared,
        output_jsonl=args.output_jsonl,
        output_csv=args.output_csv,
        progress_path=args.progress,
        failures_path=args.failures,
        limit=args.limit,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
