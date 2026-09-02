import json
import unittest

from src.models.score_model1_prompt_ensemble_batch import (
    MAX_OUTPUT_TOKENS, VARIANTS, conservative_cost_usd, custom_id, direct_conservative_cost_usd, request_body,
    request_lines, response_schema,
)


class PromptEnsembleBatchTests(unittest.TestCase):
    def test_builds_one_distinct_request_per_prompt_and_record(self) -> None:
        lines = request_lines(["first", "second"], ["one", "two"])
        self.assertEqual(len(lines), 2 * len(VARIANTS))
        self.assertEqual(len({line["custom_id"] for line in lines}), len(lines))
        self.assertEqual(lines[0]["url"], "/v1/responses")

    def test_requests_are_zero_shot_structured_and_have_no_labels(self) -> None:
        body = request_body("Example essay.", "behavioural_rubric")
        self.assertEqual(body["reasoning"]["effort"], "minimal")
        self.assertEqual(body["max_output_tokens"], MAX_OUTPUT_TOKENS)
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertNotIn("cEXT", body["instructions"])
        self.assertEqual(response_schema()["required"], ["cEXT_score", "cNEU_score", "cAGR_score", "cCON_score", "cOPN_score"])

    def test_conservative_estimate_grows_with_text_and_prompt_count(self) -> None:
        self.assertGreater(conservative_cost_usd(["a" * 3_000]), conservative_cost_usd(["a" * 300]))
        self.assertEqual(custom_id("record-1", "direct"), "direct::record-1")

    def test_normal_api_estimate_is_twice_batch_pricing(self) -> None:
        texts = ["a" * 1_500, "b" * 1_000]
        self.assertEqual(direct_conservative_cost_usd(texts), conservative_cost_usd(texts) * 2)


if __name__ == "__main__":
    unittest.main()
