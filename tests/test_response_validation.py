import csv
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.integration.response_validation import (
    DELIVERY_FIELDS,
    MAX_RESPONSE_CHARACTERS,
    RESPONSE_FIELDS,
    ResponseValidationError,
    aggregate_responses,
    load_released_instrument,
    load_response_records,
    validate_response_batch,
)
from src.integration.schemas import InstrumentScenario, ReleasedInstrument, ResponseRecord


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ResponseValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instrument = ReleasedInstrument(
            "4.0.0",
            ("cooperation", "innovation"),
            (
                InstrumentScenario("scenario-1", 1, "First question.", "cooperation"),
                InstrumentScenario("scenario-2", 2, "Second question.", "innovation"),
            ),
        )
        self.responses = (
            ResponseRecord("assessment-1", "scenario-1", 1, "I would clarify the competing needs."),
            ResponseRecord("assessment-1", "scenario-2", 2, "I would test a practical alternative."),
        )

    def test_complete_batch_validates_and_aggregates_without_skill_metadata(self) -> None:
        batch = validate_response_batch(self.instrument, self.responses)
        text = batch.model1_text()
        self.assertEqual(batch.assessment_id, "assessment-1")
        self.assertIn("RESPONSE 1", text)
        self.assertIn("RESPONSE 2", text)
        self.assertNotIn("scenario-1", text)
        self.assertNotIn("cooperation", text)
        self.assertNotIn("innovation", text)

    def test_missing_duplicate_unknown_and_out_of_order_responses_fail(self) -> None:
        cases = (
            (self.responses[:1], "missing scenario IDs"),
            ((self.responses[0], self.responses[0]), "duplicate scenario IDs"),
            ((self.responses[0], ResponseRecord("assessment-1", "unknown", 2, "Answer.")), "unknown scenario IDs"),
            ((self.responses[1], self.responses[0]), "delivery order"),
        )
        for records, expected in cases:
            with self.subTest(expected=expected), self.assertRaisesRegex(ResponseValidationError, expected):
                validate_response_batch(self.instrument, records)

    def test_mixed_assessment_ids_and_invalid_order_fail(self) -> None:
        mixed = (self.responses[0], ResponseRecord("assessment-2", "scenario-2", 2, "Answer."))
        with self.assertRaisesRegex(ResponseValidationError, "exactly one assessment_id"):
            validate_response_batch(self.instrument, mixed)
        bad_order = (self.responses[0], ResponseRecord("assessment-1", "scenario-2", 3, "Answer."))
        with self.assertRaisesRegex(ResponseValidationError, "contiguous"):
            validate_response_batch(self.instrument, bad_order)

    def test_response_loader_rejects_empty_and_oversized_text_without_echoing_content(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "responses.csv"
            write_csv(path, RESPONSE_FIELDS, [{"assessment_id": "a", "scenario_id": "s", "response_order": 1, "response_text": "  "}])
            with self.assertRaisesRegex(ResponseValidationError, "response_text") as caught:
                load_response_records(path)
            self.assertNotIn("  ", str(caught.exception))
            write_csv(path, RESPONSE_FIELDS, [{"assessment_id": "a", "scenario_id": "s", "response_order": 1, "response_text": "x" * (MAX_RESPONSE_CHARACTERS + 1)}])
            with self.assertRaisesRegex(ResponseValidationError, "character limit"):
                load_response_records(path)

    def test_released_instrument_loader_verifies_metadata_and_digest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            delivery = root / "delivery.csv"
            write_csv(delivery, DELIVERY_FIELDS, [
                {"scenario_id": "scenario-1", "delivery_order": 1, "text": "First question."},
                {"scenario_id": "scenario-2", "delivery_order": 2, "text": "Second question."},
            ])
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({
                "status": "released",
                "meets_release_gate": True,
                "pipeline_version": "4.0.0",
                "selected_skills": ["cooperation", "innovation"],
                "selected_scenario_count": 2,
                "delivery_sha256": hashlib.sha256(delivery.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            loaded = load_released_instrument(delivery, metadata)
            self.assertEqual(loaded.pipeline_version, "4.0.0")
            self.assertEqual([item.scenario_id for item in loaded.scenarios], ["scenario-1", "scenario-2"])
            delivery.write_text(delivery.read_text(encoding="utf-8").replace("First", "Changed"), encoding="utf-8")
            with self.assertRaisesRegex(ResponseValidationError, "SHA-256"):
                load_released_instrument(delivery, metadata)


if __name__ == "__main__":
    unittest.main()
