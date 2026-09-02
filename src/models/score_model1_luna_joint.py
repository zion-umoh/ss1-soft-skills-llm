"""Evaluate one joint, continuous Luna Big Five assessment per validation essay.

This is a no-training, zero-shot correction to the earlier five-specialist
experiment.  One response estimates all traits as continuous 0--10 values;
binary predictions are derived only for secondary threshold metrics.  Scores
are rounded to the requested two-decimal resolution before AUROC calculation
so floating-point noise cannot break genuine ties.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import platform
import sklearn

try:
    from src.models.score_model1_piastra import RECORD_ID_COLUMN, TRAITS, extract_output_text, load_env_value
    from src.models.score_model1_prompt_ensemble_batch import BatchError, openai_client
    from src.models.train_model1_emotion_ablation import load_essays, load_piastra_metrics, metrics
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    from score_model1_piastra import RECORD_ID_COLUMN, TRAITS, extract_output_text, load_env_value
    from score_model1_prompt_ensemble_batch import BatchError, openai_client
    from train_model1_emotion_ablation import load_essays, load_piastra_metrics, metrics


MODEL = "gpt-5.6-luna"
INPUT_PRICE_PER_MILLION = 0.20
OUTPUT_PRICE_PER_MILLION = 1.20
MAX_OUTPUT_TOKENS = 64
PROMPT_TOKEN_ALLOWANCE = 430
SCORE_DECIMALS = 2
_clients = threading.local()


def schema() -> dict[str, object]:
    properties = {f"{trait}_score": {"type": "number", "minimum": 0, "maximum": 10} for trait in TRAITS}
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": list(properties)}


def instructions() -> str:
    return (
        "Assess only the author of the supplied essay. Estimate one continuous Big Five trait level for each trait, "
        "from 0.00 (very low) to 10.00 (very high). Use behavioural linguistic evidence: social energy and "
        "assertiveness for extraversion; worry and emotional volatility for neuroticism; warmth, cooperation and "
        "compassion for agreeableness; planning, responsibility and self-discipline for conscientiousness; and "
        "curiosity, imagination and intellectual or aesthetic exploration for openness. Use intermediate values "
        "when evidence is limited. Do not infer demographics, do not use stereotypes, and do not estimate class "
        "membership, population rank, prevalence, or a binary label. Return only the requested structured scores."
    )


def request_body(text: str) -> dict[str, object]:
    return {
        "model": MODEL,
        "store": False,
        "reasoning": {"effort": "none"},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "instructions": instructions(),
        "input": text,
        "text": {"format": {"type": "json_schema", "name": "big_five_continuous_scores", "strict": True, "schema": schema()}},
    }


def conservative_cost_usd(texts: list[str]) -> float:
    input_tokens = sum(math.ceil(len(text) / 3) for text in texts) + len(texts) * PROMPT_TOKEN_ALLOWANCE
    return input_tokens / 1_000_000 * INPUT_PRICE_PER_MILLION + len(texts) * MAX_OUTPUT_TOKENS / 1_000_000 * OUTPUT_PRICE_PER_MILLION


def call(api_key: str, text: str) -> dict[str, object]:
    client = getattr(_clients, "client", None)
    if client is None:
        client = openai_client(api_key)
        _clients.client = client
    try:
        body = client.responses.create(**request_body(text)).model_dump()
    except Exception as error:
        raise BatchError(f"Luna request failed: {error}") from error
    if body.get("status") != "completed":
        raise BatchError(f"Luna returned status {body.get('status')!r}.")
    return body


def completed_ids(path: Path, expected: set[str]) -> set[str]:
    if not path.is_file():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for number, raw in enumerate(stream, start=1):
            try:
                row = json.loads(raw)
                identifier = row["custom_id"]
                body = row["response"]["body"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise BatchError(f"Invalid joint-Luna checkpoint row {number}.") from error
            if identifier not in expected or identifier in completed or not isinstance(body, dict):
                raise BatchError(f"Invalid joint-Luna checkpoint row {number}.")
            completed.add(identifier)
    return completed


def run(api_key: str, record_ids: list[str], texts: list[str], output: Path, workers: int, max_calls: int | None) -> int:
    if workers < 1:
        raise BatchError("--workers must be at least 1.")
    expected = set(record_ids)
    done = completed_ids(output, expected)
    pending = [(record_id, text) for record_id, text in zip(record_ids, texts, strict=True) if record_id not in done]
    if max_calls is not None:
        if max_calls < 1:
            raise BatchError("--max-calls must be positive.")
        pending = pending[:max_calls]
    if not pending:
        return len(done)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting {len(pending)} joint Luna calls; resuming after {len(done)} completed calls.", flush=True)
    with output.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call, api_key, text): record_id for record_id, text in pending}
        for count, future in enumerate(as_completed(futures), start=len(done) + 1):
            body = future.result()
            stream.write(json.dumps({"custom_id": futures[future], "response": {"status_code": 200, "body": body}}, separators=(",", ":")) + "\n")
            stream.flush()
            if count % 25 == 0 or count == len(expected):
                print(f"Completed {count}/{len(expected)} joint Luna calls.", flush=True)
    return len(done) + len(pending)


def scores_from_output(path: Path, record_ids: list[str]) -> np.ndarray:
    found: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            row = json.loads(raw)
            identifier = row["custom_id"]
            if identifier in found:
                raise BatchError(f"Joint-Luna output repeats {identifier!r}.")
            try:
                payload = json.loads(extract_output_text(row["response"]["body"]))
                values = [float(payload[f"{trait}_score"]) for trait in TRAITS]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise BatchError(f"Joint-Luna output lacks valid scores for {identifier!r}.") from error
            if any(not 0 <= value <= 10 for value in values):
                raise BatchError(f"Joint-Luna score is outside 0--10 for {identifier!r}.")
            found[identifier] = values
    if set(found) != set(record_ids):
        raise BatchError(f"Joint-Luna output is incomplete: {len(found)}/{len(record_ids)} responses.")
    return np.round(np.asarray([found[record_id] for record_id in record_ids], dtype=float), SCORE_DECIMALS)


def write_results(scores: np.ndarray, record_ids: list[str], labels: np.ndarray, predictions: Path, report: Path, metadata: Path, piastra: Path) -> dict[str, object]:
    probabilities = scores / 10.0
    candidate = {
        "feature_count": None,
        "validation_metrics": metrics(labels, probabilities),
        "model_settings": {"type": "zero-shot, untrained joint continuous Luna assessment", "model": MODEL, "target": "continuous 0--10 Big Five trait level", "score_decimals_for_metrics": SCORE_DECIMALS},
    }
    baseline = load_piastra_metrics(piastra, record_ids, labels)
    candidates = {"luna_joint_continuous": candidate, "paper_zero_shot_piastra": baseline}
    selected = max(candidates, key=lambda name: (candidates[name]["validation_metrics"]["macro_auroc"], candidates[name]["validation_metrics"]["macro_f1"]))
    result = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"), "model": MODEL,
        "inference_mode": "normal OpenAI Responses API (store=false)", "split": "validation", "training_rows_used": 0,
        "test_rows_used": 0, "candidates": candidates, "selected_candidate": selected,
        "selection_rule": "higher macro AUROC, then macro F1", "scikit_learn_version": sklearn.__version__, "python_version": platform.python_version(),
    }
    predictions.parent.mkdir(parents=True, exist_ok=True)
    fields = [RECORD_ID_COLUMN, "split"] + [f"label_{trait}" for trait in TRAITS] + [f"score_luna_joint_{trait}" for trait in TRAITS] + [f"prediction_luna_joint_{trait}" for trait in TRAITS]
    with predictions.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, record_id in enumerate(record_ids):
            row = {RECORD_ID_COLUMN: record_id, "split": "validation"}
            row.update({f"label_{trait}": int(labels[index, trait_index]) for trait_index, trait in enumerate(TRAITS)})
            row.update({f"score_luna_joint_{trait}": f"{scores[index, trait_index]:.{SCORE_DECIMALS}f}" for trait_index, trait in enumerate(TRAITS)})
            row.update({f"prediction_luna_joint_{trait}": int(scores[index, trait_index] >= 5.0) for trait_index, trait in enumerate(TRAITS)})
            writer.writerow(row)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Model 1 Luna Joint Continuous Report", "", "- One zero-shot Luna call scores all five continuous 0--10 traits.", "- Text only; no labels are sent to the API, no parameters are trained, and the test split is untouched.", f"- Scores are rounded to {SCORE_DECIMALS} decimals before ranking metrics to preserve genuine ties.", "", "| Candidate | Macro AUROC | Macro F1 | Macro balanced accuracy | Exact-match accuracy |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, value in candidates.items():
        measured = value["validation_metrics"]
        lines.append(f"| {name} | {measured['macro_auroc']:.4f} | {measured['macro_f1']:.4f} | {measured['macro_balanced_accuracy']:.4f} | {measured['exact_match_accuracy']:.4f} |")
    lines.extend(["", f"Provisional validation selection: **{selected}**.", ""])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--essays", type=Path, default=Path("data/processed/model1/essays_validation.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/model-evaluation/model1_luna_joint_validation_output.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/model-evaluation/model1_luna_joint_validation_predictions.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/model-evaluation/model1-luna-joint-report.md"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/model1-luna-joint-validation.json"))
    parser.add_argument("--piastra-predictions", type=Path, default=Path("outputs/model-evaluation/model1_piastra_validation_predictions.csv"))
    parser.add_argument("--env-file", type=Path, default=Path(".env")); parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-estimated-usd", type=float, default=0.20); parser.add_argument("--workers", type=int, default=5); parser.add_argument("--max-calls", type=int)
    parser.add_argument("--confirm-remote-inference", action="store_true")
    args = parser.parse_args()
    if not args.confirm_remote_inference:
        raise SystemExit("Refusing remote inference without --confirm-remote-inference.")
    api_key = os.environ.get(args.api_key_env) or load_env_value(args.env_file, args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key in {args.api_key_env!r}.")
    for variable in ("OPENAI_PROJECT_ID", "OPENAI_ORGANIZATION_ID"):
        value = os.environ.get(variable) or load_env_value(args.env_file, variable)
        if value:
            os.environ[variable] = value
    record_ids, texts, labels = load_essays(args.essays, "validation")
    estimate = conservative_cost_usd(texts)
    if estimate > args.max_estimated_usd:
        raise SystemExit(f"Conservative estimate ${estimate:.4f} exceeds cap ${args.max_estimated_usd:.4f}.")
    complete = run(api_key, record_ids, texts, args.output, args.workers, args.max_calls)
    if complete != len(record_ids):
        print(f"Checkpoint contains {complete}/{len(record_ids)} responses; ready to resume.")
        return
    result = write_results(scores_from_output(args.output, record_ids), record_ids, labels, args.predictions, args.report, args.metadata, args.piastra_predictions)
    print(f"Collected joint Luna validation results; selected {result['selected_candidate']}; conservative estimate ${estimate:.4f}.")


if __name__ == "__main__":
    main()
