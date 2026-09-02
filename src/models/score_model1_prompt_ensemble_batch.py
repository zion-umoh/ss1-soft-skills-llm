"""Run a no-training, three-prompt Big Five ensemble through OpenAI.

Only the fixed validation split is accepted by default.  Each prompt is an
independent zero-shot assessment; their unweighted mean is an ensemble, not a
trained stacker.  Either Batch or normal Responses API inference can be used;
both have a local conservative spend cap before any essay text is sent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import ssl
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn

try:
    from src.models.score_model1_piastra import (
        RECORD_ID_COLUMN, SCORE_THRESHOLD, TRAITS, ScoringError, binary_predictions,
        extract_output_text, load_env_value, trusted_ssl_context,
    )
    from src.models.train_model1_emotion_ablation import load_essays, load_piastra_metrics, metrics
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    from score_model1_piastra import (
        RECORD_ID_COLUMN, SCORE_THRESHOLD, TRAITS, ScoringError, binary_predictions,
        extract_output_text, load_env_value, trusted_ssl_context,
    )
    from train_model1_emotion_ablation import load_essays, load_piastra_metrics, metrics


MODEL = "gpt-5-mini-2025-08-07"
INPUT_PRICE_PER_MILLION = 0.25
OUTPUT_PRICE_PER_MILLION = 2.00
DIRECT_INPUT_PRICE_PER_MILLION = 0.50
DIRECT_OUTPUT_PRICE_PER_MILLION = 4.00
# GPT-5 mini requires minimal reasoning; this leaves enough budget for that
# reasoning plus the five-field structured result.
MAX_OUTPUT_TOKENS = 256
PROMPT_TOKEN_ALLOWANCE = 260
VARIANTS = ("direct", "behavioural_rubric", "cautious_assessor")


class BatchError(ValueError):
    """Raised if a batch artifact, API response, or result violates the protocol."""


def response_schema() -> dict[str, object]:
    properties = {f"{trait}_score": {"type": "number", "minimum": 0, "maximum": 10} for trait in TRAITS}
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": list(properties)}


def instructions_for(variant: str) -> str:
    """Return a label-free fixed prompt variant, deliberately predeclared before validation."""
    shared = (
        "Assess only the author of the supplied essay. Do not infer demographic attributes and do not use any "
        "information outside the essay. Produce the requested structured scores without explanation."
    )
    variants = {
        "direct": (
            "Estimate the author's Big Five personality traits from the essay. Rate extraversion, neuroticism, "
            "agreeableness, conscientiousness, and openness on a 0 (very low) to 10 (very high) scale. " + shared
        ),
        "behavioural_rubric": (
            "Estimate the author's Big Five personality traits from behavioural linguistic evidence in the essay: "
            "social engagement for extraversion, worry and emotional volatility for neuroticism, cooperation and "
            "warmth for agreeableness, planning and self-discipline for conscientiousness, and curiosity and "
            "intellectual/aesthetic exploration for openness. Rate each from 0 (very low) to 10 (very high). " + shared
        ),
        "cautious_assessor": (
            "Estimate the author's Big Five personality traits from the essay, but distinguish evidence from guesswork. "
            "Use mid-range scores when evidence for a trait is sparse rather than relying on stereotypes. Rate "
            "extraversion, neuroticism, agreeableness, conscientiousness, and openness from 0 (very low) to 10 "
            "(very high). " + shared
        ),
    }
    try:
        return variants[variant]
    except KeyError as error:
        raise BatchError(f"Unknown prompt variant {variant!r}.") from error


def request_body(text: str, variant: str, model: str = MODEL) -> dict[str, object]:
    return {
        "model": model,
        "store": False,
        # This fixed GPT-5 mini snapshot requires at least minimal reasoning effort.
        # It changes inference only; no parameters are trained or updated.
        "reasoning": {"effort": "minimal"},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "instructions": instructions_for(variant),
        "input": text,
        "text": {"format": {"type": "json_schema", "name": "big_five_scores", "strict": True, "schema": response_schema()}},
    }


def custom_id(record_id: str, variant: str) -> str:
    return f"{variant}::{record_id}"


def request_lines(record_ids: list[str], texts: list[str], model: str = MODEL) -> list[dict[str, object]]:
    if len(record_ids) != len(texts) or len(set(record_ids)) != len(record_ids):
        raise BatchError("Essay IDs and texts must be one-to-one before batch generation.")
    lines = [
        {"custom_id": custom_id(record_id, variant), "method": "POST", "url": "/v1/responses", "body": request_body(text, variant, model)}
        for record_id, text in zip(record_ids, texts, strict=True)
        for variant in VARIANTS
    ]
    if len({line["custom_id"] for line in lines}) != len(lines):
        raise BatchError("Batch request IDs were not unique.")
    return lines


def conservative_cost_usd(
    texts: list[str], input_price_per_million: float = INPUT_PRICE_PER_MILLION,
    output_price_per_million: float = OUTPUT_PRICE_PER_MILLION,
) -> float:
    """Upper-bound cost estimate without installing a tokenizer or making an API request."""
    requests = len(texts) * len(VARIANTS)
    # Three characters/token is deliberately more conservative than typical English BPE tokenisation.
    input_tokens = sum(math.ceil(len(text) / 3) for text in texts) * len(VARIANTS) + requests * PROMPT_TOKEN_ALLOWANCE
    output_tokens = requests * MAX_OUTPUT_TOKENS
    return (input_tokens / 1_000_000 * input_price_per_million) + (output_tokens / 1_000_000 * output_price_per_million)


def direct_conservative_cost_usd(texts: list[str]) -> float:
    """Conservative regular Responses API cost; normal pricing is twice Batch pricing."""
    return conservative_cost_usd(texts, DIRECT_INPUT_PRICE_PER_MILLION, DIRECT_OUTPUT_PRICE_PER_MILLION)


def write_jsonl(path: Path, lines: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for line in lines:
            stream.write(json.dumps(line, separators=(",", ":")) + "\n")


def openai_headers(api_key: str, content_type: str) -> dict[str, str]:
    """Keep file upload and Batch creation explicitly in the same configured project."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": content_type}
    project_id = os.environ.get("OPENAI_PROJECT_ID")
    organization_id = os.environ.get("OPENAI_ORGANIZATION_ID")
    if project_id:
        headers["OpenAI-Project"] = project_id
    if organization_id:
        headers["OpenAI-Organization"] = organization_id
    return headers


def api_json(api_key: str, method: str, url: str, body: dict[str, object] | None = None) -> dict[str, object]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers=openai_headers(api_key, "application/json"), method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=120, context=trusted_ssl_context()) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:800]
        raise BatchError(f"OpenAI API returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise BatchError(f"OpenAI API request failed: {error.reason}") from error


def openai_client(api_key: str):
    """Create the official SDK client with the optional configured project scope."""
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        project=os.environ.get("OPENAI_PROJECT_ID") or None,
        organization=os.environ.get("OPENAI_ORGANIZATION_ID") or None,
        max_retries=2,
        timeout=120,
    )


_thread_clients = threading.local()


def direct_response(api_key: str, text: str, variant: str) -> dict[str, object]:
    """Make one regular, stateless Responses request using a per-worker SDK client."""
    client = getattr(_thread_clients, "client", None)
    if client is None:
        client = openai_client(api_key)
        _thread_clients.client = client
    try:
        response = client.responses.create(**request_body(text, variant))
    except Exception as error:  # SDK retries transient API failures before raising.
        raise BatchError(f"Normal Responses API request failed for {variant!r}: {error}") from error
    body = response.model_dump()
    if body.get("status") != "completed":
        raise BatchError(f"Normal Responses API returned non-completed status {body.get('status')!r} for {variant!r}.")
    return body


def completed_direct_ids(path: Path, expected: set[str]) -> set[str]:
    """Validate the resumable local checkpoint and return its completed request IDs."""
    if not path.is_file():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            try:
                line = json.loads(raw_line)
                identifier = line["custom_id"]
                response = line["response"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise BatchError(f"Direct checkpoint has an invalid row at line {line_number}.") from error
            if not isinstance(identifier, str) or identifier not in expected:
                raise BatchError(f"Direct checkpoint has unexpected custom_id at line {line_number}.")
            if identifier in completed:
                raise BatchError(f"Direct checkpoint repeats {identifier!r}.")
            if not isinstance(response, dict) or response.get("status_code") != 200 or not isinstance(response.get("body"), dict):
                raise BatchError(f"Direct checkpoint has a non-successful response at line {line_number}.")
            completed.add(identifier)
    return completed


def run_direct_responses(
    api_key: str, record_ids: list[str], texts: list[str], output_path: Path, workers: int, max_calls: int | None = None,
) -> int:
    """Run missing validation calls and append each successful result atomically as a JSONL checkpoint."""
    if workers < 1:
        raise BatchError("--workers must be at least 1.")
    expected = {custom_id(record_id, variant) for record_id in record_ids for variant in VARIANTS}
    completed = completed_direct_ids(output_path, expected)
    pending = [
        (custom_id(record_id, variant), text, variant)
        for record_id, text in zip(record_ids, texts, strict=True)
        for variant in VARIANTS
        if custom_id(record_id, variant) not in completed
    ]
    if max_calls is not None:
        if max_calls < 1:
            raise BatchError("--max-calls must be at least 1 when supplied.")
        pending = pending[:max_calls]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not pending:
        print(f"Direct checkpoint already contains all {len(expected)} validation responses.")
        return len(completed)
    print(f"Starting {len(pending)} normal Responses API calls; resuming after {len(completed)} completed calls.")
    with output_path.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(direct_response, api_key, text, variant): identifier
            for identifier, text, variant in pending
        }
        for completed_count, future in enumerate(as_completed(futures), start=len(completed) + 1):
            identifier = futures[future]
            body = future.result()
            stream.write(json.dumps({"custom_id": identifier, "response": {"status_code": 200, "body": body}}, separators=(",", ":")) + "\n")
            stream.flush()
            if completed_count % 25 == 0 or completed_count == len(expected):
                print(f"Completed {completed_count}/{len(expected)} normal Responses API calls.")
    return len(completed) + len(pending)


def upload_batch_file(api_key: str, path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as stream:
            return openai_client(api_key).files.create(file=stream, purpose="batch").model_dump()
    except Exception as error:  # The SDK supplies a typed error hierarchy that may change between releases.
        raise BatchError(f"Official OpenAI SDK file upload failed: {error}") from error


def submit_batch(api_key: str, input_file_id: str) -> dict[str, object]:
    try:
        return openai_client(api_key).batches.create(
            input_file_id=input_file_id, endpoint="/v1/responses", completion_window="24h",
            metadata={"study": "model1-zero-shot-prompt-ensemble", "split": "validation", "training": "false"},
        ).model_dump()
    except Exception as error:  # The SDK supplies a typed error hierarchy that may change between releases.
        raise BatchError(f"Official OpenAI SDK batch creation failed: {error}") from error


def wait_for_processed_file(api_key: str, input_file_id: str, timeout_seconds: int = 120) -> dict[str, object]:
    """Wait until Batch can reliably consume the uploaded JSONL before creating a job."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        file_object = api_json(api_key, "GET", f"https://api.openai.com/v1/files/{input_file_id}")
        status = file_object.get("status")
        if status == "processed":
            return file_object
        if status == "error":
            raise BatchError(f"OpenAI could not process the input batch file: {file_object.get('status_details')}")
        if time.monotonic() >= deadline:
            raise BatchError(f"Timed out waiting for input file {input_file_id!r} to be processed; last status {status!r}.")
        time.sleep(2)


def parse_scores(response_body: dict[str, object]) -> dict[str, float]:
    try:
        payload = json.loads(extract_output_text(response_body))
    except json.JSONDecodeError as error:
        raise BatchError("Batch result contained malformed JSON despite Structured Outputs.") from error
    result = {}
    for trait in TRAITS:
        field = f"{trait}_score"
        try:
            value = float(payload[field])
        except (KeyError, ValueError, TypeError) as error:
            raise BatchError(f"Batch response omitted numeric {field!r}.") from error
        if not 0 <= value <= 10:
            raise BatchError(f"Batch response {field!r} was outside 0--10.")
        result[trait] = value / 10.0
    return result


def parse_output(path: Path, expected_ids: list[str]) -> dict[str, np.ndarray]:
    expected = {custom_id(record_id, variant) for record_id in expected_ids for variant in VARIANTS}
    results: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = json.loads(raw_line)
            identifier = line.get("custom_id")
            response = line.get("response")
            if identifier not in expected:
                raise BatchError("Batch output had an unexpected custom_id.")
            if not isinstance(response, dict) or response.get("status_code") != 200 or not isinstance(response.get("body"), dict):
                raise BatchError(f"Batch request {identifier!r} did not complete successfully.")
            if identifier in results:
                raise BatchError(f"Batch output repeated {identifier!r}.")
            results[str(identifier)] = parse_scores(response["body"])
    if set(results) != expected:
        raise BatchError(f"Batch output is incomplete: received {len(results)} of {len(expected)} expected responses.")
    return {
        variant: np.asarray([[results[custom_id(record_id, variant)][trait] for trait in TRAITS] for record_id in expected_ids])
        for variant in VARIANTS
    }


def write_predictions(path: Path, record_ids: list[str], labels: np.ndarray, candidate_scores: dict[str, np.ndarray]) -> None:
    fields = [RECORD_ID_COLUMN, "split"] + [f"label_{trait}" for trait in TRAITS]
    for candidate in candidate_scores:
        fields += [f"score_{candidate}_{trait}" for trait in TRAITS]
    fields += [f"prediction_ensemble_{trait}" for trait in TRAITS]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, record_id in enumerate(record_ids):
            row: dict[str, object] = {RECORD_ID_COLUMN: record_id, "split": "validation"}
            row.update({f"label_{trait}": int(labels[index, trait_index]) for trait_index, trait in enumerate(TRAITS)})
            for candidate, scores in candidate_scores.items():
                row.update({f"score_{candidate}_{trait}": f"{scores[index, trait_index]:.8f}" for trait_index, trait in enumerate(TRAITS)})
            row.update({f"prediction_ensemble_{trait}": int(score >= SCORE_THRESHOLD / 10) for trait, score in zip(TRAITS, candidate_scores["equal_weight_ensemble"][index], strict=True)})
            writer.writerow(row)


def render_report(metadata: dict[str, object]) -> str:
    lines = [
        "# Model 1 No-Training Prompt-Ensemble Report", "",
        f"- Model: fixed `gpt-5-mini-2025-08-07` snapshot via {metadata['inference_mode']}.",
        "- Three zero-shot prompts were predeclared and evaluated independently; their equal-weight mean is not trained.",
        "- Validation only; no Essays training or test rows were read.",
        "- Selection: macro AUROC, then macro F1 on a tie.", "",
        "| Candidate | Macro AUROC | Macro F1 | Macro balanced accuracy | Exact-match accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, candidate in metadata["candidates"].items():
        value = candidate["validation_metrics"]
        lines.append(f"| {name} | {value['macro_auroc']:.4f} | {value['macro_f1']:.4f} | {value['macro_balanced_accuracy']:.4f} | {value['exact_match_accuracy']:.4f} |")
    lines.extend(["", f"Provisional validation selection: **{metadata['selected_candidate']}**.", ""])
    return "\n".join(lines)


def collect_results(
    output_path: Path, essays_path: Path, predictions_path: Path, report_path: Path, metadata_path: Path,
    piastra_predictions: Path, batch_state_path: Path | None, inference_mode: str = "OpenAI Batch",
) -> dict[str, object]:
    record_ids, _, labels = load_essays(essays_path, "validation")
    per_prompt = parse_output(output_path, record_ids)
    candidate_scores = {**per_prompt, "equal_weight_ensemble": np.mean(np.stack(list(per_prompt.values())), axis=0)}
    candidates = {
        name: {"feature_count": None, "validation_metrics": metrics(labels, scores), "model_settings": {"type": "zero-shot, untrained", "prompt": name}}
        for name, scores in candidate_scores.items()
    }
    candidates["paper_zero_shot_piastra"] = load_piastra_metrics(piastra_predictions, record_ids, labels)
    selected = max(candidates, key=lambda name: (candidates[name]["validation_metrics"]["macro_auroc"], candidates[name]["validation_metrics"]["macro_f1"]))
    state = json.loads(batch_state_path.read_text(encoding="utf-8")) if batch_state_path and batch_state_path.is_file() else {}
    metadata: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"), "model": MODEL,
        "inference_mode": inference_mode, "split": "validation",
        "training_rows_used": 0, "test_rows_used": 0, "variants": list(VARIANTS), "ensemble": "equal untrained mean",
        "candidates": candidates, "selected_candidate": selected, "selection_rule": "higher macro AUROC, then higher macro F1",
        "scikit_learn_version": sklearn.__version__, "python_version": platform.python_version(),
    }
    if batch_state_path:
        metadata["batch_id"] = state.get("batch_id")
        metadata["input_file_id"] = state.get("input_file_id")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(predictions_path, record_ids, labels, candidate_scores)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(metadata), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "retry", "resubmit", "status", "collect", "direct"))
    parser.add_argument("--essays", type=Path, default=Path("data/processed/model1/essays_validation.csv"))
    parser.add_argument("--jsonl", type=Path, default=Path("outputs/model-evaluation/model1_prompt_ensemble_validation_requests.jsonl"))
    parser.add_argument("--batch-state", type=Path, default=Path("data/metadata/model1-prompt-ensemble-validation-batch.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/model-evaluation/model1_prompt_ensemble_validation_batch_output.jsonl"))
    parser.add_argument("--direct-output", type=Path, default=Path("outputs/model-evaluation/model1_prompt_ensemble_validation_direct_output.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/model-evaluation/model1_prompt_ensemble_validation_predictions.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/model-evaluation/model1-prompt-ensemble-report.md"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/model1-prompt-ensemble-validation.json"))
    parser.add_argument("--piastra-predictions", type=Path, default=Path("outputs/model-evaluation/model1_piastra_validation_predictions.csv"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-estimated-usd", type=float)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-calls", type=int, help="Run only this many missing direct calls, then leave a resumable checkpoint.")
    parser.add_argument("--confirm-remote-inference", action="store_true")
    args = parser.parse_args()
    record_ids, texts, _ = load_essays(args.essays, "validation")
    if args.action == "prepare":
        lines = request_lines(record_ids, texts)
        write_jsonl(args.jsonl, lines)
        print(f"Prepared {len(lines)} validation batch requests; conservative estimate ${conservative_cost_usd(texts):.4f}.")
        return
    if args.action == "collect":
        metadata = collect_results(args.output, args.essays, args.predictions, args.report, args.metadata, args.piastra_predictions, args.batch_state)
        print(f"Collected validation results; selected {metadata['selected_candidate']}.")
        return
    api_key = os.environ.get(args.api_key_env) or load_env_value(args.env_file, args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key in {args.api_key_env!r}.")
    for variable in ("OPENAI_PROJECT_ID", "OPENAI_ORGANIZATION_ID"):
        configured = os.environ.get(variable) or load_env_value(args.env_file, variable)
        if configured:
            os.environ[variable] = configured
    if args.action == "status":
        if not args.batch_state.is_file():
            raise SystemExit("No batch state file exists.")
        state = json.loads(args.batch_state.read_text(encoding="utf-8"))
        print(json.dumps(api_json(api_key, "GET", f"https://api.openai.com/v1/batches/{state['batch_id']}"), indent=2))
        return
    if not args.confirm_remote_inference:
        raise SystemExit("Refusing remote inference without --confirm-remote-inference.")
    if args.action == "direct":
        estimate = direct_conservative_cost_usd(texts)
        cap = args.max_estimated_usd if args.max_estimated_usd is not None else 2.00
        if estimate > cap:
            raise SystemExit(f"Conservative normal-API estimate ${estimate:.4f} exceeds cap ${cap:.4f}.")
        completed_count = run_direct_responses(api_key, record_ids, texts, args.direct_output, args.workers, args.max_calls)
        expected_count = len(record_ids) * len(VARIANTS)
        if completed_count != expected_count:
            print(f"Direct checkpoint contains {completed_count}/{expected_count} responses; leaving it ready to resume.")
            return
        metadata = collect_results(
            args.direct_output, args.essays, args.predictions, args.report, args.metadata,
            args.piastra_predictions, None, "normal OpenAI Responses API (store=false)",
        )
        print(
            f"Collected normal Responses API validation results; selected {metadata['selected_candidate']}. "
            f"Conservative preflight estimate ${estimate:.4f} under cap ${cap:.4f}."
        )
        return
    if args.action == "retry":
        if not args.batch_state.is_file():
            raise SystemExit("No failed batch state file exists to retry.")
        state = json.loads(args.batch_state.read_text(encoding="utf-8"))
        failed_batch = api_json(api_key, "GET", f"https://api.openai.com/v1/batches/{state['batch_id']}")
        usage = failed_batch.get("usage")
        if failed_batch.get("status") != "failed" or not isinstance(usage, dict) or usage.get("total_tokens") != 0:
            raise SystemExit("Refusing retry: the recorded batch was not a zero-usage failed submission.")
        wait_for_processed_file(api_key, str(state["input_file_id"]))
        replacement = submit_batch(api_key, str(state["input_file_id"]))
        state["failed_zero_usage_batch_id"] = state["batch_id"]
        state["batch_id"] = replacement["id"]
        state["retried_at_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        args.batch_state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        print(f"Retried the processed input file as Batch {state['batch_id']}; no new file was uploaded.")
        return
    if args.action == "resubmit":
        if not args.batch_state.is_file():
            raise SystemExit("No failed batch state file exists to resubmit.")
        state = json.loads(args.batch_state.read_text(encoding="utf-8"))
        failed_batch = api_json(api_key, "GET", f"https://api.openai.com/v1/batches/{state['batch_id']}")
        usage = failed_batch.get("usage")
        if failed_batch.get("status") != "failed" or not isinstance(usage, dict) or usage.get("total_tokens") != 0:
            raise SystemExit("Refusing resubmit: the recorded batch was not a zero-usage failed submission.")
        estimate = conservative_cost_usd(texts)
        cap = args.max_estimated_usd if args.max_estimated_usd is not None else 0.60
        if estimate > cap:
            raise SystemExit(f"Conservative estimate ${estimate:.4f} exceeds cap ${cap:.4f}.")
        lines = request_lines(record_ids, texts)
        write_jsonl(args.jsonl, lines)
        replacement_file = upload_batch_file(api_key, args.jsonl)
        wait_for_processed_file(api_key, str(replacement_file["id"]))
        replacement_batch = submit_batch(api_key, str(replacement_file["id"]))
        failures = list(state.get("failed_zero_usage_batch_ids", []))
        failures.append(state["batch_id"])
        state.update({
            "failed_zero_usage_batch_ids": failures,
            "batch_id": replacement_batch["id"],
            "input_file_id": replacement_file["id"],
            "resubmitted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        })
        args.batch_state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        print(f"Resubmitted as Batch {state['batch_id']} with a new input file in the explicitly configured project.")
        return
    if args.batch_state.exists():
        raise SystemExit("A batch state file already exists; refusing to submit a duplicate batch.")
    estimate = conservative_cost_usd(texts)
    cap = args.max_estimated_usd if args.max_estimated_usd is not None else 0.60
    if estimate > cap:
        raise SystemExit(f"Conservative estimate ${estimate:.4f} exceeds cap ${cap:.4f}.")
    lines = request_lines(record_ids, texts)
    write_jsonl(args.jsonl, lines)
    file_object = upload_batch_file(api_key, args.jsonl)
    wait_for_processed_file(api_key, str(file_object["id"]))
    batch_object = submit_batch(api_key, str(file_object["id"]))
    state = {"created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"), "batch_id": batch_object["id"], "input_file_id": file_object["id"], "requests": len(lines), "conservative_estimate_usd": round(estimate, 4), "model": MODEL, "split": "validation"}
    args.batch_state.parent.mkdir(parents=True, exist_ok=True)
    args.batch_state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"Submitted Batch {state['batch_id']} with {len(lines)} validation requests; conservative estimate ${estimate:.4f}.")


if __name__ == "__main__":
    main()
