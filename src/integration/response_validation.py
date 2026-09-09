"""Validate runtime assessment responses before any model call.

This module is intentionally limited to the Batch-1 boundary. It loads a
released respondent-facing instrument, validates one assessment's response
records, and builds a neutral response-only text payload for Model 1. It does
not load model artifacts or make network calls.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .schemas import ContractError, InstrumentScenario, ReleasedInstrument, ResponseRecord


RESPONSE_FIELDS = ("assessment_id", "scenario_id", "response_order", "response_text")
DELIVERY_FIELDS = ("scenario_id", "delivery_order", "text")
MAX_RESPONSE_CHARACTERS = 12_000
MAX_AGGREGATED_CHARACTERS = 50_000


class ResponseValidationError(ContractError):
    """Raised when an instrument or response file violates the runtime contract."""


@dataclass(frozen=True)
class ValidatedResponseBatch:
    """A complete, ordered response set for exactly one assessment."""

    assessment_id: str
    instrument: ReleasedInstrument
    responses: tuple[ResponseRecord, ...]

    def model1_text(self) -> str:
        return aggregate_responses(self.responses)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path, expected_fields: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise ResponseValidationError(f"Cannot open input file {path}.") from error
    with stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ResponseValidationError(f"{path} must contain exactly these columns: {', '.join(expected_fields)}.")
        rows = list(reader)
    if not rows:
        raise ResponseValidationError(f"{path} contains no data rows.")
    return rows


def load_released_instrument(delivery_path: Path, metadata_path: Path) -> ReleasedInstrument:
    """Load and verify a released respondent-facing instrument."""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResponseValidationError(f"Cannot read instrument metadata {metadata_path}.") from error
    if metadata.get("status") != "released":
        raise ResponseValidationError("Instrument metadata is not marked released.")
    if metadata.get("meets_release_gate") is not True:
        raise ResponseValidationError("Instrument metadata does not pass the release gate.")
    selected_skills = metadata.get("selected_skills")
    pipeline_version = metadata.get("pipeline_version")
    if not isinstance(selected_skills, list) or not isinstance(pipeline_version, str):
        raise ResponseValidationError("Instrument metadata is missing the release contract.")

    rows = _read_csv(delivery_path, DELIVERY_FIELDS)
    scenarios: list[InstrumentScenario] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            delivery_order = int(row["delivery_order"])
        except (TypeError, ValueError) as error:
            raise ResponseValidationError(f"Instrument row {row_number} has an invalid delivery_order.") from error
        try:
            scenarios.append(InstrumentScenario(row["scenario_id"], delivery_order, row["text"]))
        except ContractError as error:
            raise ResponseValidationError(f"Instrument row {row_number} is invalid: {error}") from error

    digest = _sha256(delivery_path)
    expected_digest = metadata.get("delivery_sha256")
    if expected_digest and digest != expected_digest:
        raise ResponseValidationError("Instrument delivery file does not match its recorded SHA-256 digest.")
    expected_count = metadata.get("selected_scenario_count")
    if expected_count is not None and expected_count != len(scenarios):
        raise ResponseValidationError("Instrument delivery count does not match its metadata.")
    try:
        return ReleasedInstrument(pipeline_version, tuple(selected_skills), tuple(scenarios), digest)
    except ContractError as error:
        raise ResponseValidationError(f"Released instrument is invalid: {error}") from error


def load_response_records(path: Path) -> tuple[ResponseRecord, ...]:
    """Load response rows without logging or including response text in errors."""
    rows = _read_csv(path, RESPONSE_FIELDS)
    records: list[ResponseRecord] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            response_order = int(row["response_order"])
        except (TypeError, ValueError) as error:
            raise ResponseValidationError(f"Response row {row_number} has an invalid response_order.") from error
        try:
            record = ResponseRecord(row["assessment_id"], row["scenario_id"], response_order, row["response_text"])
        except ContractError as error:
            raise ResponseValidationError(f"Response row {row_number} is invalid: {error}") from error
        if len(record.response_text) > MAX_RESPONSE_CHARACTERS:
            raise ResponseValidationError(f"Response row {row_number} exceeds the {MAX_RESPONSE_CHARACTERS}-character limit.")
        records.append(record)
    return tuple(records)


def validate_response_batch(
    instrument: ReleasedInstrument,
    responses: tuple[ResponseRecord, ...],
    expected_assessment_id: str | None = None,
) -> ValidatedResponseBatch:
    """Validate one complete assessment against a released instrument."""
    if not responses:
        raise ResponseValidationError("The response set contains no records.")
    assessment_ids = {record.assessment_id for record in responses}
    if len(assessment_ids) != 1:
        raise ResponseValidationError("A response batch must contain exactly one assessment_id.")
    assessment_id = next(iter(assessment_ids))
    if expected_assessment_id is not None and assessment_id != expected_assessment_id:
        raise ResponseValidationError("The response batch does not match the requested assessment_id.")

    expected_ids = [scenario.scenario_id for scenario in instrument.scenarios]
    actual_ids = [record.scenario_id for record in responses]
    duplicate_ids = sorted({scenario_id for scenario_id in actual_ids if actual_ids.count(scenario_id) > 1})
    if duplicate_ids:
        raise ResponseValidationError(f"Response set contains duplicate scenario IDs: {', '.join(duplicate_ids)}.")
    unknown_ids = sorted(set(actual_ids) - set(expected_ids))
    if unknown_ids:
        raise ResponseValidationError(f"Response set contains unknown scenario IDs: {', '.join(unknown_ids)}.")
    missing_ids = [scenario_id for scenario_id in expected_ids if scenario_id not in set(actual_ids)]
    if missing_ids:
        raise ResponseValidationError(f"Response set is missing scenario IDs: {', '.join(missing_ids)}.")
    if actual_ids != expected_ids:
        raise ResponseValidationError("Responses must be supplied in the released instrument's delivery order.")
    if [record.response_order for record in responses] != list(range(1, len(expected_ids) + 1)):
        raise ResponseValidationError("response_order must be contiguous and match delivery order.")
    aggregated_length = sum(len(record.response_text) for record in responses)
    if aggregated_length > MAX_AGGREGATED_CHARACTERS:
        raise ResponseValidationError(f"Combined response text exceeds the {MAX_AGGREGATED_CHARACTERS}-character limit.")
    return ValidatedResponseBatch(assessment_id, instrument, responses)


def aggregate_responses(responses: tuple[ResponseRecord, ...]) -> str:
    """Build neutral Model 1 input from already validated responses."""
    if not responses:
        raise ResponseValidationError("Cannot aggregate an empty response set.")
    blocks = [f"RESPONSE {index}\n{record.response_text}" for index, record in enumerate(responses, start=1)]
    aggregated = "\n\n".join(blocks)
    if len(aggregated) > MAX_AGGREGATED_CHARACTERS:
        raise ResponseValidationError(f"Combined response text exceeds the {MAX_AGGREGATED_CHARACTERS}-character limit.")
    return aggregated

