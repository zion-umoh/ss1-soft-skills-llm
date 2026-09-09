import json
import unittest

from src.integration.model1_adapter import (
    K_RETRIEVED_EXAMPLES,
    LOCKED_MAX_OUTPUT_TOKENS,
    LOCKED_MODEL1_NAME,
    LOCKED_REASONING_EFFORT,
    Model1AdapterError,
    RetrievedReference,
    build_request,
    parse_response,
)
from src.models.score_model1_luna_retrieval import MODEL


class Model1AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.references = tuple(
            RetrievedReference(f"Reference essay {index}.", (1, 0, 1, 0, 1))
            for index in range(K_RETRIEVED_EXAMPLES)
        )

    def test_build_request_uses_locked_retrieval_contract(self) -> None:
        request = build_request("assessment-1", "I would clarify the issue first.", self.references)
        self.assertEqual(LOCKED_MODEL1_NAME, MODEL)
        self.assertEqual(request.model_name, MODEL)
        self.assertEqual(request.reasoning_effort, LOCKED_REASONING_EFFORT)
        self.assertEqual(request.max_output_tokens, LOCKED_MAX_OUTPUT_TOKENS)
        self.assertEqual(request.max_output_tokens, 512)
        self.assertEqual(request.payload["model"], MODEL)
        self.assertFalse(request.payload["store"])
        self.assertEqual(request.payload["reasoning"], {"effort": "medium"})
        self.assertIn("TARGET ESSAY:", request.payload["input"])
        self.assertIn("I would clarify the issue first.", request.payload["input"])
        self.assertNotIn("cooperation", request.payload["input"])

    def test_request_requires_exactly_three_binary_references(self) -> None:
        with self.assertRaisesRegex(Model1AdapterError, "exactly 3"):
            build_request("assessment-1", "Response text.", self.references[:2])
        with self.assertRaisesRegex(Model1AdapterError, "five binary"):
            RetrievedReference("Reference.", (1, 0, 1, 0, 2))

    def test_parse_response_accepts_structured_output_text(self) -> None:
        payload = {f"{trait}_score": index * 2.5 for index, trait in enumerate(("cEXT", "cNEU", "cAGR", "cCON", "cOPN"))}
        result = parse_response("assessment-1", {"status": "completed", "output_text": json.dumps(payload)})
        self.assertEqual(result.assessment_id, "assessment-1")
        self.assertEqual(result.scores.scores[-1], 10.0)
        self.assertEqual(result.scores.model_name, MODEL)

    def test_parse_response_accepts_responses_api_output_blocks(self) -> None:
        payload = {f"{trait}_score": 5 for trait in ("cEXT", "cNEU", "cAGR", "cCON", "cOPN")}
        result = parse_response("assessment-1", {"status": "completed", "output": [{"content": [{"type": "output_text", "text": json.dumps(payload)}]}]})
        self.assertEqual(result.scores.scores, (5.0, 5.0, 5.0, 5.0, 5.0))

    def test_parse_response_rejects_failed_malformed_and_out_of_range_outputs(self) -> None:
        with self.assertRaisesRegex(Model1AdapterError, "not 'completed'"):
            parse_response("assessment-1", {"status": "failed"})
        with self.assertRaisesRegex(Model1AdapterError, "valid JSON"):
            parse_response("assessment-1", {"status": "completed", "output_text": "not json"})
        payload = {f"{trait}_score": 5 for trait in ("cEXT", "cNEU", "cAGR", "cCON", "cOPN")}
        payload["cOPN_score"] = 11
        with self.assertRaisesRegex(Model1AdapterError, "between 0 and 10"):
            parse_response("assessment-1", {"status": "completed", "output_text": json.dumps(payload)})


if __name__ == "__main__":
    unittest.main()
