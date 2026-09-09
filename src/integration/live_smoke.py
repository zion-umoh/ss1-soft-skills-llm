"""Controlled live Model 1 -> Model 2 smoke test.

This module sends exactly one approved assessment to the locked Model 1
Responses endpoint, then reuses the deterministic runtime boundaries to parse
Model 1 and score the locked Model 2 artifact. It is a connectivity smoke test,
not a predictive-validity evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from .fixture_pipeline import FIXTURE_REFERENCES, _default_paths, run_fixture_pipeline
from .model1_adapter import LOCKED_MODEL1_NAME, RetrievedReference, build_request
from .model2_adapter import load_model2_adapter
from .response_validation import load_released_instrument, load_response_records, validate_response_batch
from src.models.score_model1_piastra import load_env_value


MAX_LIVE_REQUESTS = 1


class LiveSmokeError(RuntimeError):
    """Raised when the controlled live smoke test cannot complete."""


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported API response value: {type(value).__name__}")


def _response_mapping(response: object) -> dict[str, object]:
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    if not isinstance(response, Mapping):
        raise LiveSmokeError("The live Model 1 response was not an object.")
    try:
        return _json_safe(response)  # type: ignore[return-value]
    except TypeError as error:
        raise LiveSmokeError("The live Model 1 response was not JSON-serializable.") from error


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as error:
        raise LiveSmokeError(f"Could not write smoke-test artifact {path}.") from error


def _openai_client(api_key: str):
    """Build a single-request SDK client with retries disabled."""
    try:
        from openai import OpenAI
    except ImportError as error:  # pragma: no cover - dependency is project-managed.
        raise LiveSmokeError("The OpenAI Python SDK is not installed in the project environment.") from error
    return OpenAI(
        api_key=api_key,
        project=os.environ.get("OPENAI_PROJECT_ID") or None,
        organization=os.environ.get("OPENAI_ORGANIZATION_ID") or None,
        max_retries=0,
        timeout=120,
    )


def run_live_smoke(
    delivery_path: Path,
    metadata_path: Path,
    response_path: Path,
    model2_path: Path,
    *,
    api_key: str,
    output_dir: Path,
    confirm: bool = False,
    client: object | None = None,
    references: Sequence[RetrievedReference] = FIXTURE_REFERENCES,
) -> dict[str, object]:
    """Run one confirmed live Model 1 request and the locked Model 2 path."""
    if not confirm:
        raise LiveSmokeError("Refusing remote inference without explicit confirmation.")
    if not isinstance(api_key, str) or not api_key.strip():
        raise LiveSmokeError("A non-empty OpenAI API key is required for the live smoke test.")
    if MAX_LIVE_REQUESTS != 1:
        raise LiveSmokeError("The live smoke-test request budget must remain exactly one request.")

    request_count = 0
    raw_path = output_dir / "model1-raw-response.json"
    report_path = output_dir / "live-assessment-report.json"
    metadata_out = output_dir / "live-smoke-metadata.json"
    failure_path = output_dir / "live-smoke-failure.json"
    try:
        existing_artifacts = [path for path in (raw_path, report_path, metadata_out, failure_path) if path.exists()]
        if existing_artifacts:
            names = ", ".join(path.name for path in existing_artifacts)
            raise LiveSmokeError(f"Output directory already contains smoke artifacts ({names}); choose a new output directory.")
        instrument = load_released_instrument(delivery_path, metadata_path)
        responses = load_response_records(response_path)
        batch = validate_response_batch(instrument, responses)
        request = build_request(batch.assessment_id, batch.model1_text(), references)
        api_client = client if client is not None else _openai_client(api_key)
        responses_api = getattr(api_client, "responses", None)
        create = getattr(responses_api, "create", None)
        if not callable(create):
            raise LiveSmokeError("The API client does not expose responses.create.")
        request_count += 1
        raw_response = _response_mapping(create(**request.payload))
        _write_json(raw_path, raw_response)

        report = run_fixture_pipeline(
            delivery_path,
            metadata_path,
            response_path,
            model2_path,
            model1_response=raw_response,
            output_path=report_path,
            mode="live_smoke",
        )
        metadata: dict[str, object] = {
            "status": "completed",
            "assessment_id": batch.assessment_id,
            "request_count": request_count,
            "request_budget": MAX_LIVE_REQUESTS,
            "model1_model": LOCKED_MODEL1_NAME,
            "raw_response_path": str(raw_path),
            "assessment_report_path": str(report_path),
            "model2_artifact": str(load_model2_adapter(model2_path).artifact_path),
            "interpretation": "Connectivity and contract smoke test only; not predictive-validity evidence.",
        }
        _write_json(metadata_out, metadata)
        return {"metadata": metadata, "report": report}
    except LiveSmokeError as error:
        failure = {
            "status": "failed",
            "request_count": request_count,
            "request_budget": MAX_LIVE_REQUESTS,
            "error": str(error),
            "interpretation": "No completed assessment report was released.",
        }
        try:
            _write_json(failure_path, failure)
        except LiveSmokeError:
            pass
        raise
    except Exception as error:
        failure = {
            "status": "failed",
            "request_count": request_count,
            "request_budget": MAX_LIVE_REQUESTS,
            "error": str(error),
            "interpretation": "No completed assessment report was released.",
        }
        try:
            _write_json(failure_path, failure)
        except LiveSmokeError:
            pass
        raise LiveSmokeError(f"Live smoke test failed: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=("one", "three"), default="three")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model2", type=Path, default=Path("outputs/models/model2_linear_benchmark.joblib"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/integration/live-smoke"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--confirm-remote-inference", action="store_true")
    args = parser.parse_args()
    if not args.confirm_remote_inference:
        raise SystemExit("Refusing remote inference without --confirm-remote-inference.")
    answer = input(
        f"This will send one response batch to {LOCKED_MODEL1_NAME} and may use exactly one request. "
        "Type RUN to continue, or press Enter to cancel: "
    )
    if answer.strip() != "RUN":
        raise SystemExit("Live smoke test cancelled.")
    api_key = os.environ.get(args.api_key_env) or load_env_value(args.env_file, args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key in {args.api_key_env!r}.")
    delivery, metadata, responses = _default_paths(args.root, args.fixture)
    try:
        result = run_live_smoke(
            delivery,
            metadata,
            responses,
            args.model2,
            api_key=api_key,
            output_dir=args.output_dir,
            confirm=True,
        )
    except LiveSmokeError as error:
        raise SystemExit(str(error)) from error
    print(f"Live smoke test completed; report written to {result['metadata']['assessment_report_path']}.")


if __name__ == "__main__":
    main()
