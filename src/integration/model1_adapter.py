"""Adapter for the locked Model 1 retrieval-augmented scorer.

This batch builds request and response boundaries only. It does not make a
remote call. Retrieval references are supplied by the caller so the locked
Model 1 retrieval protocol remains explicit and testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

from src.models.score_model1_luna_retrieval import (
    K_RETRIEVED_EXAMPLES,
    MODEL,
    input_for,
    request_body,
)

from .schemas import ContractError, Model1Scores, model1_scores_from_mapping


LOCKED_REASONING_EFFORT = "medium"
LOCKED_MODEL1_NAME = MODEL
# The locked medium-reasoning evaluation protocol used 512 output tokens.
# The underlying joint scorer's compact default is 64, which can truncate the
# structured response before all five scores are emitted.
LOCKED_MAX_OUTPUT_TOKENS = 512


class Model1AdapterError(ContractError):
    """Raised when a Model 1 request or response violates its contract."""


@dataclass(frozen=True)
class RetrievedReference:
    """One labelled training reference supplied to the locked scorer."""

    text: str
    labels: tuple[int, int, int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise Model1AdapterError("Model 1 retrieval reference text must be non-empty.")
        if len(self.labels) != 5 or any(label not in (0, 1) for label in self.labels):
            raise Model1AdapterError("Each Model 1 retrieval reference must contain five binary labels.")


@dataclass(frozen=True)
class Model1Request:
    """A request payload plus provenance for one assessment."""

    assessment_id: str
    target_text: str
    payload: dict[str, object]
    model_name: str
    reasoning_effort: str
    max_output_tokens: int


@dataclass(frozen=True)
class Model1Result:
    """Parsed Model 1 output linked to its assessment."""

    assessment_id: str
    scores: Model1Scores


def build_request(
    assessment_id: str,
    target_text: str,
    references: Sequence[RetrievedReference],
    *,
    reasoning_effort: str = LOCKED_REASONING_EFFORT,
    max_output_tokens: int = LOCKED_MAX_OUTPUT_TOKENS,
) -> Model1Request:
    """Build the locked retrieval request without performing inference."""
    if not isinstance(assessment_id, str) or not assessment_id.strip():
        raise Model1AdapterError("assessment_id must be a non-empty string.")
    if not isinstance(target_text, str) or not target_text.strip():
        raise Model1AdapterError("Model 1 target text must be non-empty.")
    if len(references) != K_RETRIEVED_EXAMPLES:
        raise Model1AdapterError(f"Locked Model 1 requires exactly {K_RETRIEVED_EXAMPLES} retrieval references.")
    if not isinstance(reasoning_effort, str) or not reasoning_effort.strip():
        raise Model1AdapterError("reasoning_effort must be a non-empty string.")
    if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool) or max_output_tokens < 1:
        raise Model1AdapterError("max_output_tokens must be a positive integer.")
    checked_references = tuple(reference if isinstance(reference, RetrievedReference) else RetrievedReference(reference.text, tuple(reference.labels)) for reference in references)
    examples = [(reference.text, list(reference.labels)) for reference in checked_references]
    retrieval_input = input_for(target_text.strip(), examples)
    payload = request_body(retrieval_input, reasoning_effort=reasoning_effort, max_output_tokens=max_output_tokens)
    return Model1Request(
        assessment_id=assessment_id.strip(),
        target_text=target_text.strip(),
        payload=payload,
        model_name=LOCKED_MODEL1_NAME,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )


def _extract_output_text(response: Mapping[str, object]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, Mapping) and block.get("type") == "output_text" and isinstance(block.get("text"), str) and block["text"].strip():
                    return block["text"]
    raise Model1AdapterError("Model 1 response did not contain output text.")


def parse_response(assessment_id: str, response: Mapping[str, object]) -> Model1Result:
    """Parse one Responses API result into the frozen Model 1 score contract."""
    if not isinstance(response, Mapping):
        raise Model1AdapterError("Model 1 response must be a mapping.")
    status = response.get("status")
    if status is not None and status != "completed":
        raise Model1AdapterError(f"Model 1 response status is {status!r}, not 'completed'.")
    try:
        payload = json.loads(_extract_output_text(response))
    except json.JSONDecodeError as error:
        raise Model1AdapterError("Model 1 output text was not valid JSON.") from error
    if not isinstance(payload, Mapping):
        raise Model1AdapterError("Model 1 JSON output must be an object.")
    try:
        scores = model1_scores_from_mapping(payload, LOCKED_MODEL1_NAME)
    except ContractError as error:
        raise Model1AdapterError(str(error)) from error
    return Model1Result(assessment_id=assessment_id, scores=scores)
