"""Evaluate retrieval-augmented, few-shot Luna scoring on Essays validation.

For every validation essay, three semantically nearest Essays training essays
and their binary labels are provided as in-context references. Retrieval never
uses validation labels; the LLM returns continuous 0--10 Big Five scores in one
joint response. No parameters are trained and the held-out test split is never
read.
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
    from src.models.score_model1_luna_joint import (
        INPUT_PRICE_PER_MILLION, MAX_OUTPUT_TOKENS, MODEL, OUTPUT_PRICE_PER_MILLION, SCORE_DECIMALS,
        schema,
    )
    from src.models.score_model1_piastra import RECORD_ID_COLUMN, TRAITS, extract_output_text, load_env_value
    from src.models.score_model1_prompt_ensemble_batch import BatchError, openai_client
    from src.models.train_model1_embedding_ablation import MODEL_NAME, chunk_text
    from src.models.train_model1_emotion_ablation import load_essays, load_piastra_metrics, metrics
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    from score_model1_luna_joint import INPUT_PRICE_PER_MILLION, MAX_OUTPUT_TOKENS, MODEL, OUTPUT_PRICE_PER_MILLION, SCORE_DECIMALS, schema
    from score_model1_piastra import RECORD_ID_COLUMN, TRAITS, extract_output_text, load_env_value
    from score_model1_prompt_ensemble_batch import BatchError, openai_client
    from train_model1_embedding_ablation import MODEL_NAME, chunk_text
    from train_model1_emotion_ablation import load_essays, load_piastra_metrics, metrics


K_RETRIEVED_EXAMPLES = 3
TRAIT_BALANCED_REFERENCES_PER_SIDE = 1
PROMPT_TOKEN_ALLOWANCE = 570
DEFAULT_REASONING_EFFORT = "none"
_clients = threading.local()


def encode_documents(texts: list[str], model_name: str, batch_size: int, device: str | None) -> np.ndarray:
    """Mean-pool local MiniLM chunks into normalised document embeddings."""
    from sentence_transformers import SentenceTransformer

    counts: list[int] = []
    chunks: list[str] = []
    for text in texts:
        document_chunks = chunk_text(text)
        counts.append(len(document_chunks))
        chunks.extend(document_chunks)
    # Retrieval is intentionally local: never trigger a model download while
    # preparing an API experiment. The project already records this cached model.
    model = SentenceTransformer(model_name, device=device, local_files_only=True)
    embeddings = model.encode(chunks, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    pooled = []
    offset = 0
    for count in counts:
        value = embeddings[offset:offset + count].mean(axis=0)
        norm = np.linalg.norm(value)
        pooled.append(value / norm if norm else value)
        offset += count
    return np.asarray(pooled, dtype=np.float32)


def retrieve_indices(train_ids: list[str], train_embeddings: np.ndarray, validation_embeddings: np.ndarray, k: int) -> list[list[int]]:
    """Return deterministic top-k training neighbours for each validation essay."""
    if k < 1 or k > len(train_ids):
        raise ValueError("k must be between 1 and the number of training rows.")
    if train_embeddings.ndim != 2 or validation_embeddings.ndim != 2 or train_embeddings.shape[1] != validation_embeddings.shape[1]:
        raise ValueError("Train and validation embeddings must have matching two-dimensional shapes.")
    result = []
    for embedding in validation_embeddings:
        similarities = train_embeddings @ embedding
        result.append(sorted(range(len(train_ids)), key=lambda index: (-float(similarities[index]), train_ids[index]))[:k])
    return result


def trait_balanced_retrieve_indices(
    train_ids: list[str], train_embeddings: np.ndarray, evaluation_embeddings: np.ndarray, train_labels: np.ndarray,
    references_per_side: int = TRAIT_BALANCED_REFERENCES_PER_SIDE,
) -> list[list[int]]:
    """Retrieve nearest labelled high and low anchors separately for every trait."""
    if references_per_side < 1:
        raise ValueError("references_per_side must be positive.")
    if train_labels.shape != (len(train_ids), len(TRAITS)):
        raise ValueError("train_labels must align with train_ids and Big Five traits.")
    result = []
    for embedding in evaluation_embeddings:
        similarities = train_embeddings @ embedding
        selected: list[int] = []
        for trait_index in range(len(TRAITS)):
            for label in (1, 0):
                candidates = [index for index in range(len(train_ids)) if int(train_labels[index, trait_index]) == label]
                ranked = sorted(candidates, key=lambda index: (-float(similarities[index]), train_ids[index]))
                selected.extend(ranked[:references_per_side])
        result.append(list(dict.fromkeys(selected)))
    return result


def labels_text(labels: np.ndarray) -> str:
    return "; ".join(f"{trait}={'high' if int(value) else 'low'}" for trait, value in zip(TRAITS, labels, strict=True))


def input_for(target_text: str, examples: list[tuple[str, np.ndarray]]) -> str:
    blocks = []
    for number, (text, labels) in enumerate(examples, start=1):
        blocks.append(f"REFERENCE EXAMPLE {number}\nESSAY:\n{text}\nDATASET LABELS:\n{labels_text(labels)}")
    return "Retrieved training references:\n\n" + "\n\n".join(blocks) + f"\n\nTARGET ESSAY:\n{target_text}"


def instructions() -> str:
    return (
        "Assess the author of the TARGET ESSAY only. The retrieved training examples are labelled examples from this "
        "same dataset; use them only to understand its label convention and calibrate your judgement. Do not copy or "
        "average their labels, and do not report a binary class. Estimate continuous 0.00--10.00 Big Five trait levels "
        "from the target's behavioural linguistic evidence: social energy and assertiveness for extraversion; worry and "
        "emotional volatility for neuroticism; warmth, cooperation and compassion for agreeableness; planning, "
        "responsibility and self-discipline for conscientiousness; and curiosity, imagination and intellectual or "
        "aesthetic exploration for openness. Use intermediate values when evidence is limited. Do not infer demographics, "
        "do not use stereotypes, and return only the requested structured scores."
    )


def request_body(
    text: str,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> dict[str, object]:
    return {
        "model": MODEL, "store": False, "reasoning": {"effort": reasoning_effort}, "max_output_tokens": max_output_tokens,
        "instructions": instructions(), "input": text,
        "text": {"format": {"type": "json_schema", "name": "big_five_continuous_scores", "strict": True, "schema": schema()}},
    }


def conservative_cost_usd(inputs: list[str], max_output_tokens: int = MAX_OUTPUT_TOKENS) -> float:
    input_tokens = sum(math.ceil(len(value) / 3) for value in inputs) + len(inputs) * PROMPT_TOKEN_ALLOWANCE
    return input_tokens / 1_000_000 * INPUT_PRICE_PER_MILLION + len(inputs) * max_output_tokens / 1_000_000 * OUTPUT_PRICE_PER_MILLION


def call(api_key: str, text: str, reasoning_effort: str, max_output_tokens: int) -> dict[str, object]:
    client = getattr(_clients, "client", None)
    if client is None:
        client = openai_client(api_key)
        _clients.client = client
    try:
        body = client.responses.create(**request_body(text, reasoning_effort, max_output_tokens)).model_dump()
    except Exception as error:
        raise BatchError(f"Retrieval-Luna request failed: {error}") from error
    if body.get("status") != "completed":
        raise BatchError(f"Retrieval-Luna returned status {body.get('status')!r}.")
    return body


def completed_ids(path: Path, expected: set[str]) -> set[str]:
    if not path.is_file():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for number, raw in enumerate(stream, start=1):
            try:
                row = json.loads(raw); identifier = row["custom_id"]; body = row["response"]["body"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise BatchError(f"Invalid retrieval-Luna checkpoint row {number}.") from error
            if identifier not in expected or identifier in completed or not isinstance(body, dict):
                raise BatchError(f"Invalid retrieval-Luna checkpoint row {number}.")
            completed.add(identifier)
    return completed


def run(api_key: str, record_ids: list[str], inputs: list[str], output: Path, workers: int, max_calls: int | None, reasoning_effort: str, max_output_tokens: int) -> int:
    if workers < 1:
        raise BatchError("--workers must be at least 1.")
    expected = set(record_ids)
    done = completed_ids(output, expected)
    pending = [(record_id, text) for record_id, text in zip(record_ids, inputs, strict=True) if record_id not in done]
    if max_calls is not None:
        if max_calls < 1:
            raise BatchError("--max-calls must be positive.")
        pending = pending[:max_calls]
    if not pending:
        return len(done)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting {len(pending)} retrieval-Luna calls; resuming after {len(done)} completed calls.", flush=True)
    with output.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call, api_key, text, reasoning_effort, max_output_tokens): record_id for record_id, text in pending}
        for count, future in enumerate(as_completed(futures), start=len(done) + 1):
            stream.write(json.dumps({"custom_id": futures[future], "response": {"status_code": 200, "body": future.result()}}, separators=(",", ":")) + "\n")
            stream.flush()
            if count % 25 == 0 or count == len(expected):
                print(f"Completed {count}/{len(expected)} retrieval-Luna calls.", flush=True)
    return len(done) + len(pending)


def scores_from_output(path: Path, record_ids: list[str]) -> np.ndarray:
    found: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            row = json.loads(raw); identifier = row["custom_id"]
            if identifier in found:
                raise BatchError(f"Retrieval-Luna output repeats {identifier!r}.")
            try:
                payload = json.loads(extract_output_text(row["response"]["body"])); values = [float(payload[f"{trait}_score"]) for trait in TRAITS]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise BatchError(f"Retrieval-Luna output lacks valid scores for {identifier!r}.") from error
            if any(not 0 <= value <= 10 for value in values):
                raise BatchError(f"Retrieval-Luna score is outside 0--10 for {identifier!r}.")
            found[identifier] = values
    if set(found) != set(record_ids):
        raise BatchError(f"Retrieval-Luna output is incomplete: {len(found)}/{len(record_ids)} responses.")
    return np.round(np.asarray([found[record_id] for record_id in record_ids], dtype=float), SCORE_DECIMALS)


def write_retrievals(path: Path, validation_ids: list[str], train_ids: list[str], train_labels: np.ndarray, train_embeddings: np.ndarray, validation_embeddings: np.ndarray, neighbours: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["validation_record_id", "rank", "train_record_id", "cosine_similarity", *[f"label_{trait}" for trait in TRAITS]])
        writer.writeheader()
        for validation_index, indices in enumerate(neighbours):
            for rank, train_index in enumerate(indices, start=1):
                row = {"validation_record_id": validation_ids[validation_index], "rank": rank, "train_record_id": train_ids[train_index], "cosine_similarity": f"{float(validation_embeddings[validation_index] @ train_embeddings[train_index]):.8f}"}
                row.update({f"label_{trait}": int(train_labels[train_index, trait_index]) for trait_index, trait in enumerate(TRAITS)})
                writer.writerow(row)


def write_results(scores: np.ndarray, record_ids: list[str], labels: np.ndarray, predictions: Path, report: Path, metadata: Path, piastra: Path, retrievals: Path, estimate: float, reasoning_effort: str, max_output_tokens: int, experiment_name: str, split: str = "validation", retrieval_mode: str = "semantic_top_k", average_retrieved_examples: float = K_RETRIEVED_EXAMPLES) -> dict[str, object]:
    metric_key = f"{split}_metrics"
    candidate = {"feature_count": None, metric_key: metrics(labels, scores / 10.0), "model_settings": {"type": "retrieval-augmented few-shot, untrained joint continuous Luna assessment", "model": MODEL, "reasoning_effort": reasoning_effort, "max_output_tokens": max_output_tokens, "target": "continuous 0--10 Big Five trait level", "retrieval_mode": retrieval_mode, "average_retrieved_examples": round(average_retrieved_examples, 2), "score_decimals_for_metrics": SCORE_DECIMALS}}
    baseline = load_piastra_metrics(piastra, record_ids, labels, split)
    baseline[metric_key] = baseline.pop("validation_metrics")
    candidates = {experiment_name: candidate, "paper_zero_shot_piastra": baseline}
    selected = max(candidates, key=lambda name: (candidates[name][metric_key]["macro_auroc"], candidates[name][metric_key]["macro_f1"]))
    result = {"generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"), "model": MODEL, "reasoning_effort": reasoning_effort, "max_output_tokens": max_output_tokens, "inference_mode": "normal OpenAI Responses API (store=false)", "split": split, "training_rows_used_as_retrieval_references": 1727, "test_rows_used": len(record_ids) if split == "test" else 0, "embedding_model": MODEL_NAME, "retrieval_mode": retrieval_mode, "average_retrieved_examples_per_evaluation_essay": round(average_retrieved_examples, 2), "retrieval_manifest": str(retrievals), "conservative_estimate_usd": round(estimate, 4), "candidates": candidates, "selected_candidate": selected if split == "validation" else experiment_name, "selection_rule": "higher macro AUROC, then macro F1" if split == "validation" else "model preselected on validation; test metrics reported once without test-based selection", "scikit_learn_version": sklearn.__version__, "python_version": platform.python_version()}
    predictions.parent.mkdir(parents=True, exist_ok=True)
    fields = [RECORD_ID_COLUMN, "split"] + [f"label_{trait}" for trait in TRAITS] + [f"score_luna_retrieval_{trait}" for trait in TRAITS] + [f"prediction_luna_retrieval_{trait}" for trait in TRAITS]
    with predictions.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for index, record_id in enumerate(record_ids):
            row = {RECORD_ID_COLUMN: record_id, "split": split}
            row.update({f"label_{trait}": int(labels[index, trait_index]) for trait_index, trait in enumerate(TRAITS)})
            row.update({f"score_luna_retrieval_{trait}": f"{scores[index, trait_index]:.{SCORE_DECIMALS}f}" for trait_index, trait in enumerate(TRAITS)})
            row.update({f"prediction_luna_retrieval_{trait}": int(scores[index, trait_index] >= 5.0) for trait_index, trait in enumerate(TRAITS)})
            writer.writerow(row)
    metadata.parent.mkdir(parents=True, exist_ok=True); metadata.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    split_note = "test labels are used only after the locked model is scored" if split == "test" else "validation labels are not used to select examples; this run does not read test rows"
    lines = ["# Model 1 Retrieval-Augmented Luna Report", "", f"- One joint Luna call per {split} essay, augmented with {average_retrieved_examples:.2f} training examples on average ({retrieval_mode}).", f"- Reasoning effort: `{reasoning_effort}`; response-token allowance: {max_output_tokens}.", f"- Retrieval uses training text only; {split_note}; no parameters are trained.", f"- Continuous scores are rounded to {SCORE_DECIMALS} decimals before AUROC calculation to preserve genuine ties.", "", "| Candidate | Macro AUROC | Macro F1 | Macro balanced accuracy | Exact-match accuracy |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, value in candidates.items():
        measured = value[metric_key]
        lines.append(f"| {name} | {measured['macro_auroc']:.4f} | {measured['macro_f1']:.4f} | {measured['macro_balanced_accuracy']:.4f} | {measured['exact_match_accuracy']:.4f} |")
    conclusion = f"Provisional validation selection: **{selected}**." if split == "validation" else f"Locked-model test comparison: **{experiment_name}** was selected on validation before this test run."
    lines.extend(["", conclusion, ""])
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/model1"))
    parser.add_argument("--evaluation-file", default="essays_validation.csv")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, default=Path("outputs/model-evaluation/model1_luna_retrieval_validation_output.jsonl"))
    parser.add_argument("--retrievals", type=Path, default=Path("outputs/model-evaluation/model1_luna_retrieval_validation_references.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/model-evaluation/model1_luna_retrieval_validation_predictions.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/model-evaluation/model1-luna-retrieval-report.md"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/model1-luna-retrieval-validation.json"))
    parser.add_argument("--piastra-predictions", type=Path, default=Path("outputs/model-evaluation/model1_piastra_validation_predictions.csv"))
    parser.add_argument("--env-file", type=Path, default=Path(".env")); parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--embedding-model", default=MODEL_NAME); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--device", default=None)
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"), default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--retrieval-mode", choices=("semantic_top_k", "trait_balanced"), default="semantic_top_k")
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--experiment-name", default="luna_retrieval_few_shot")
    parser.add_argument("--estimate-only", action="store_true", help="Compute and print the conservative cost without calling the API.")
    parser.add_argument("--max-estimated-usd", type=float, default=0.50); parser.add_argument("--workers", type=int, default=5); parser.add_argument("--max-calls", type=int)
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
    train_ids, train_texts, train_labels = load_essays(args.data_dir / "essays_train.csv", "train")
    evaluation_ids, evaluation_texts, evaluation_labels = load_essays(args.data_dir / args.evaluation_file, args.split)
    embeddings = encode_documents(train_texts + evaluation_texts, args.embedding_model, args.batch_size, args.device)
    train_embeddings, evaluation_embeddings = embeddings[:len(train_ids)], embeddings[len(train_ids):]
    neighbours = (
        retrieve_indices(train_ids, train_embeddings, evaluation_embeddings, K_RETRIEVED_EXAMPLES)
        if args.retrieval_mode == "semantic_top_k"
        else trait_balanced_retrieve_indices(train_ids, train_embeddings, evaluation_embeddings, train_labels)
    )
    inputs = [input_for(text, [(train_texts[index], train_labels[index]) for index in indices]) for text, indices in zip(evaluation_texts, neighbours, strict=True)]
    if args.max_output_tokens < 1:
        raise SystemExit("--max-output-tokens must be positive.")
    estimate = conservative_cost_usd(inputs, args.max_output_tokens)
    if args.estimate_only:
        print(f"Conservative estimate ${estimate:.4f} for {len(evaluation_ids)} {args.split} essays.")
        return
    if estimate > args.max_estimated_usd:
        raise SystemExit(f"Conservative estimate ${estimate:.4f} exceeds cap ${args.max_estimated_usd:.4f}.")
    write_retrievals(args.retrievals, evaluation_ids, train_ids, train_labels, train_embeddings, evaluation_embeddings, neighbours)
    complete = run(api_key, evaluation_ids, inputs, args.output, args.workers, args.max_calls, args.reasoning_effort, args.max_output_tokens)
    if complete != len(evaluation_ids):
        print(f"Checkpoint contains {complete}/{len(evaluation_ids)} responses; ready to resume.")
        return
    result = write_results(scores_from_output(args.output, evaluation_ids), evaluation_ids, evaluation_labels, args.predictions, args.report, args.metadata, args.piastra_predictions, args.retrievals, estimate, args.reasoning_effort, args.max_output_tokens, args.experiment_name, args.split, args.retrieval_mode, float(np.mean([len(indices) for indices in neighbours])))
    print(f"Collected retrieval-Luna {args.split} results; selected {result['selected_candidate']}; conservative estimate ${estimate:.4f}.")


if __name__ == "__main__":
    main()
