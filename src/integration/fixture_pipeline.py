"""Deterministic end-to-end runner for the runtime assessment connection.

The fixture runner exercises the same validation, Model 1 parsing, and Model 2
artifact boundaries used by the live path. Model 1 is represented by a fixed
Responses-API-shaped payload, so this command never makes a network call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .model1_adapter import (
    RetrievedReference,
    build_request,
    parse_response,
)
from .model2_adapter import load_model2_adapter, model1_to_model2_features, select_skill_predictions
from .response_validation import (
    load_released_instrument,
    load_response_records,
    validate_response_batch,
)


class FixturePipelineError(RuntimeError):
    """Raised when the deterministic end-to-end fixture cannot complete."""


FIXTURE_REFERENCES = tuple(
    RetrievedReference(
        text=f"Reference response {index}: I would clarify the goal, gather the relevant facts, and agree a practical next step.",
        labels=(1, 0, 1, 0, 1),
    )
    for index in range(3)
)
FIXTURE_MODEL1_RESPONSE: dict[str, object] = {
    "status": "completed",
    "output_text": json.dumps(
        {
            "cEXT_score": 6.0,
            "cNEU_score": 4.0,
            "cAGR_score": 7.0,
            "cCON_score": 8.0,
            "cOPN_score": 5.0,
        },
        sort_keys=True,
    ),
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _response_payload(response: Mapping[str, object]) -> dict[str, object]:
    """Return a JSON-safe copy of a mocked or captured Model 1 response."""
    try:
        return json.loads(json.dumps(response))
    except (TypeError, ValueError) as error:
        raise FixturePipelineError("The Model 1 fixture response must be JSON-serializable.") from error


def run_fixture_pipeline(
    delivery_path: Path,
    metadata_path: Path,
    response_path: Path,
    model2_path: Path,
    *,
    model1_response: Mapping[str, object] = FIXTURE_MODEL1_RESPONSE,
    output_path: Path | None = None,
    mode: str = "deterministic_fixture",
) -> dict[str, object]:
    """Run one complete assessment using a deterministic Model 1 response."""
    try:
        instrument = load_released_instrument(delivery_path, metadata_path)
        responses = load_response_records(response_path)
        batch = validate_response_batch(instrument, responses)
        request = build_request(batch.assessment_id, batch.model1_text(), FIXTURE_REFERENCES)
        model1_result = parse_response(batch.assessment_id, model1_response)
        if model1_result.assessment_id != batch.assessment_id:
            raise FixturePipelineError("Model 1 result assessment_id does not match the response batch.")
        features = model1_to_model2_features(model1_result.scores)
        model2 = load_model2_adapter(model2_path)
        predictions = model2.predict(features)
        selected_predictions = select_skill_predictions(predictions, instrument.selected_skills)
    except FixturePipelineError:
        raise
    except Exception as error:
        raise FixturePipelineError(f"Fixture pipeline failed: {error}") from error

    model1_input = batch.model1_text()
    report: dict[str, object] = {
        "status": "completed",
        "mode": mode,
        "assessment_id": batch.assessment_id,
        "inputs": {
            "instrument_pipeline_version": instrument.pipeline_version,
            "selected_skills": list(instrument.selected_skills),
            "scenario_count": len(instrument.scenarios),
            "responses": [
                {
                    "response_order": record.response_order,
                    "scenario_id": record.scenario_id,
                    "response_text": record.response_text,
                }
                for record in batch.responses
            ],
            "model1_input": model1_input,
        },
        "model1": {
            "model_name": model1_result.scores.model_name,
            "scores": model1_result.scores.as_code_mapping(),
            "request": {
                "reasoning_effort": request.reasoning_effort,
                "max_output_tokens": request.max_output_tokens,
                "target_text_sha256": _sha256_text(request.target_text),
            },
        },
        "model2": {
            "artifact_path": str(model2.artifact_path),
            "artifact_sha256": model2.artifact_sha256,
            "features": features.as_mapping(),
            "predictions": predictions.as_mapping(),
        },
        "respondent_output": {
            "selected_skills": list(instrument.selected_skills),
            "predictions": selected_predictions,
        },
        "provenance": {
            "instrument_delivery": str(delivery_path),
            "instrument_metadata": str(metadata_path),
            "instrument_delivery_sha256": instrument.delivery_sha256,
            "responses": str(response_path),
            "model1_response": _response_payload(model1_response),
            "model2_feature_columns": list(features.as_mapping()),
        },
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _default_paths(root: Path, fixture: str) -> tuple[Path, Path, Path]:
    fixture_root = root / "tests"
    if fixture == "one":
        stem = "fixture_one_skill"
    else:
        stem = "fixture_three_skill"
    return (
        fixture_root / f"{stem}_delivery.csv",
        fixture_root / f"{stem}_metadata.json",
        fixture_root / f"{stem}_responses.csv",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic assessment integration fixture.")
    parser.add_argument("--fixture", choices=("one", "three"), default="three")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model2", type=Path, default=Path("outputs/models/model2_linear_benchmark.joblib"))
    parser.add_argument("--output", type=Path, default=None, help="Optional path for the JSON assessment report.")
    args = parser.parse_args()
    delivery, metadata, responses = _default_paths(args.root, args.fixture)
    report = run_fixture_pipeline(delivery, metadata, responses, args.model2, output_path=args.output)
    if args.output:
        print(
            f"Fixture assessment completed for {len(report['respondent_output']['selected_skills'])} selected skills; "
            f"report written to {args.output}."
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
