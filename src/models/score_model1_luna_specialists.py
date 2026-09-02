"""Run a no-training, Luna-only Big Five specialist assessment on validation.

Each trait is assessed independently as a continuous probability of belonging to
the high half of the prepared Essays label distribution.  The design uses text
only, never sends labels, and does not combine another model's scores.
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
except ModuleNotFoundError:  # pragma: no cover
    from score_model1_piastra import RECORD_ID_COLUMN, TRAITS, extract_output_text, load_env_value
    from score_model1_prompt_ensemble_batch import BatchError, openai_client
    from train_model1_emotion_ablation import load_essays, load_piastra_metrics, metrics


MODEL = "gpt-5.6-luna"
INPUT_PRICE_PER_MILLION = 0.20
OUTPUT_PRICE_PER_MILLION = 1.20
MAX_OUTPUT_TOKENS = 64
PROMPT_TOKEN_ALLOWANCE = 260
_clients = threading.local()

TRAIT_DETAILS = {
    "cEXT": ("extraversion", "social energy, assertiveness, activity, positive engagement, and preference for interpersonal stimulation"),
    "cNEU": ("neuroticism", "emotional volatility, worry, vulnerability to stress, self-consciousness, and persistent negative affect"),
    "cAGR": ("agreeableness", "compassion, cooperation, trust, warmth, forgiveness, and concern for other people"),
    "cCON": ("conscientiousness", "orderliness, responsibility, persistence, self-discipline, deliberation, and goal-directed organisation"),
    "cOPN": ("openness", "intellectual curiosity, imagination, aesthetic interest, preference for novelty, reflection, and openness to ideas"),
}


def custom_id(record_id: str, trait: str) -> str:
    return f"{trait}::{record_id}"


def schema() -> dict[str, object]:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {"high_trait_probability": {"type": "number", "minimum": 0, "maximum": 1}},
        "required": ["high_trait_probability"],
    }


def instructions_for(trait: str) -> str:
    try:
        name, facets = TRAIT_DETAILS[trait]
    except KeyError as error:
        raise BatchError(f"Unknown trait {trait!r}.") from error
    return (
        "You are performing one bounded text classification task. Assess only the author of the supplied essay; "
        "do not infer demographics or use information outside the text. Estimate the probability from 0 to 1 that "
        f"the author belongs to the higher half of this essay population on {name}. Base the estimate on evidence for "
        f"{facets}. Evidence may be weak; use intermediate probabilities when the text is uninformative. Do not use "
        "stereotypes, do not explain, and return only the requested structured value."
    )


def request_body(text: str, trait: str) -> dict[str, object]:
    return {
        "model": MODEL, "store": False, "reasoning": {"effort": "none"}, "max_output_tokens": MAX_OUTPUT_TOKENS,
        "instructions": instructions_for(trait), "input": text,
        "text": {"format": {"type": "json_schema", "name": "high_trait_probability", "strict": True, "schema": schema()}},
    }


def conservative_cost_usd(texts: list[str]) -> float:
    requests = len(texts) * len(TRAITS)
    input_tokens = sum(math.ceil(len(text) / 3) for text in texts) * len(TRAITS) + requests * PROMPT_TOKEN_ALLOWANCE
    return input_tokens / 1_000_000 * INPUT_PRICE_PER_MILLION + requests * MAX_OUTPUT_TOKENS / 1_000_000 * OUTPUT_PRICE_PER_MILLION


def call(api_key: str, text: str, trait: str) -> dict[str, object]:
    client = getattr(_clients, "client", None)
    if client is None:
        client = openai_client(api_key)
        _clients.client = client
    try:
        body = client.responses.create(**request_body(text, trait)).model_dump()
    except Exception as error:
        raise BatchError(f"Luna request failed for {trait}: {error}") from error
    if body.get("status") != "completed":
        raise BatchError(f"Luna returned status {body.get('status')!r} for {trait}.")
    return body


def completed_ids(path: Path, expected: set[str]) -> set[str]:
    if not path.is_file():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as stream:
        for number, raw in enumerate(stream, start=1):
            try:
                line = json.loads(raw); identifier = line["custom_id"]; response = line["response"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise BatchError(f"Invalid Luna checkpoint row {number}.") from error
            if identifier not in expected or identifier in completed:
                raise BatchError(f"Invalid Luna checkpoint ID at row {number}.")
            if not isinstance(response, dict) or response.get("status_code") != 200 or not isinstance(response.get("body"), dict):
                raise BatchError(f"Luna checkpoint has an unsuccessful row {number}.")
            completed.add(identifier)
    return completed


def run(api_key: str, record_ids: list[str], texts: list[str], output: Path, workers: int, max_calls: int | None) -> int:
    if workers < 1:
        raise BatchError("--workers must be at least 1.")
    expected = {custom_id(record_id, trait) for record_id in record_ids for trait in TRAITS}
    done = completed_ids(output, expected)
    pending = [(custom_id(record_id, trait), text, trait) for record_id, text in zip(record_ids, texts, strict=True) for trait in TRAITS if custom_id(record_id, trait) not in done]
    if max_calls is not None:
        if max_calls < 1:
            raise BatchError("--max-calls must be positive.")
        pending = pending[:max_calls]
    if not pending:
        return len(done)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting {len(pending)} Luna specialist calls; resuming after {len(done)} completed calls.", flush=True)
    with output.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call, api_key, text, trait): identifier for identifier, text, trait in pending}
        for count, future in enumerate(as_completed(futures), start=len(done) + 1):
            body = future.result()
            stream.write(json.dumps({"custom_id": futures[future], "response": {"status_code": 200, "body": body}}, separators=(",", ":")) + "\n")
            stream.flush()
            if count % 50 == 0 or count == len(expected):
                print(f"Completed {count}/{len(expected)} Luna specialist calls.", flush=True)
    return len(done) + len(pending)


def scores_from_output(path: Path, record_ids: list[str]) -> np.ndarray:
    expected = {custom_id(record_id, trait) for record_id in record_ids for trait in TRAITS}
    found: dict[str, float] = {}
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            line = json.loads(raw); identifier = line["custom_id"]; body = line["response"]["body"]
            if identifier not in expected or identifier in found:
                raise BatchError("Luna output does not match the validation request set.")
            try:
                value = float(json.loads(extract_output_text(body))["high_trait_probability"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise BatchError(f"Luna output lacks a valid probability for {identifier}.") from error
            if not 0 <= value <= 1:
                raise BatchError(f"Luna probability outside 0--1 for {identifier}.")
            found[identifier] = value
    if set(found) != expected:
        raise BatchError(f"Luna output is incomplete: {len(found)}/{len(expected)} responses.")
    return np.asarray([[found[custom_id(record_id, trait)] for trait in TRAITS] for record_id in record_ids])


def write_results(scores: np.ndarray, record_ids: list[str], labels: np.ndarray, predictions: Path, report: Path, metadata: Path, piastra: Path) -> dict[str, object]:
    candidate = {"feature_count": None, "validation_metrics": metrics(labels, scores), "model_settings": {"type": "zero-shot, untrained Luna trait specialists", "model": MODEL, "target": "probability of high binary trait class"}}
    baseline = load_piastra_metrics(piastra, record_ids, labels)
    candidates = {"luna_trait_specialists": candidate, "paper_zero_shot_piastra": baseline}
    selected = max(candidates, key=lambda name: (candidates[name]["validation_metrics"]["macro_auroc"], candidates[name]["validation_metrics"]["macro_f1"]))
    metadata_value = {"generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"), "model": MODEL, "inference_mode": "normal OpenAI Responses API (store=false)", "split": "validation", "training_rows_used": 0, "test_rows_used": 0, "candidates": candidates, "selected_candidate": selected, "selection_rule": "higher macro AUROC, then macro F1", "scikit_learn_version": sklearn.__version__, "python_version": platform.python_version()}
    predictions.parent.mkdir(parents=True, exist_ok=True)
    fields = [RECORD_ID_COLUMN, "split"] + [f"label_{trait}" for trait in TRAITS] + [f"score_luna_{trait}" for trait in TRAITS]
    with predictions.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for index, record_id in enumerate(record_ids):
            row = {RECORD_ID_COLUMN: record_id, "split": "validation"}; row.update({f"label_{trait}": int(labels[index, trait_index]) for trait_index, trait in enumerate(TRAITS)}); row.update({f"score_luna_{trait}": f"{scores[index, trait_index]:.8f}" for trait_index, trait in enumerate(TRAITS)}); writer.writerow(row)
    metadata.parent.mkdir(parents=True, exist_ok=True); metadata.write_text(json.dumps(metadata_value, indent=2) + "\n", encoding="utf-8")
    lines = ["# Model 1 Luna Trait-Specialist Report", "", "- Five independent, zero-shot Luna specialists score continuous high-class probabilities.", "- Text only; no labels are sent to the API, no model parameters are trained, and the test split is untouched.", "", "| Candidate | Macro AUROC | Macro F1 | Macro balanced accuracy | Exact-match accuracy |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, value in candidates.items():
        result = value["validation_metrics"]; lines.append(f"| {name} | {result['macro_auroc']:.4f} | {result['macro_f1']:.4f} | {result['macro_balanced_accuracy']:.4f} | {result['exact_match_accuracy']:.4f} |")
    lines.extend(["", f"Provisional validation selection: **{selected}**.", ""]); report.parent.mkdir(parents=True, exist_ok=True); report.write_text("\n".join(lines), encoding="utf-8")
    return metadata_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--essays", type=Path, default=Path("data/processed/model1/essays_validation.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/model-evaluation/model1_luna_specialists_validation_output.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/model-evaluation/model1_luna_specialists_validation_predictions.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/model-evaluation/model1-luna-specialists-report.md"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/model1-luna-specialists-validation.json"))
    parser.add_argument("--piastra-predictions", type=Path, default=Path("outputs/model-evaluation/model1_piastra_validation_predictions.csv"))
    parser.add_argument("--env-file", type=Path, default=Path(".env")); parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-estimated-usd", type=float, default=1.50); parser.add_argument("--workers", type=int, default=5); parser.add_argument("--max-calls", type=int)
    parser.add_argument("--confirm-remote-inference", action="store_true")
    args = parser.parse_args()
    if not args.confirm_remote_inference:
        raise SystemExit("Refusing remote inference without --confirm-remote-inference.")
    api_key = os.environ.get(args.api_key_env) or load_env_value(args.env_file, args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key in {args.api_key_env!r}.")
    for variable in ("OPENAI_PROJECT_ID", "OPENAI_ORGANIZATION_ID"):
        value = os.environ.get(variable) or load_env_value(args.env_file, variable)
        if value: os.environ[variable] = value
    record_ids, texts, labels = load_essays(args.essays, "validation")
    estimate = conservative_cost_usd(texts)
    if estimate > args.max_estimated_usd:
        raise SystemExit(f"Conservative estimate ${estimate:.4f} exceeds cap ${args.max_estimated_usd:.4f}.")
    complete = run(api_key, record_ids, texts, args.output, args.workers, args.max_calls)
    expected = len(record_ids) * len(TRAITS)
    if complete != expected:
        print(f"Checkpoint contains {complete}/{expected} responses; ready to resume.")
        return
    result = write_results(scores_from_output(args.output, record_ids), record_ids, labels, args.predictions, args.report, args.metadata, args.piastra_predictions)
    print(f"Collected Luna validation results; selected {result['selected_candidate']}; conservative estimate ${estimate:.4f}.")


if __name__ == "__main__":
    main()
