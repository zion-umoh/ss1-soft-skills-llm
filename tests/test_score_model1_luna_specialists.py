import unittest

from src.models.score_model1_luna_specialists import (
    MAX_OUTPUT_TOKENS, TRAIT_DETAILS, conservative_cost_usd, custom_id, instructions_for, request_body, schema,
)


class LunaSpecialistTests(unittest.TestCase):
    def test_each_trait_has_a_distinct_label_free_request(self) -> None:
        self.assertEqual(custom_id("essay-1", "cEXT"), "cEXT::essay-1")
        self.assertEqual(set(TRAIT_DETAILS), {"cEXT", "cNEU", "cAGR", "cCON", "cOPN"})
        prompt = instructions_for("cCON")
        self.assertIn("higher half", prompt)
        self.assertNotIn("label_cCON", prompt)

    def test_uses_luna_none_reasoning_and_strict_probability_schema(self) -> None:
        body = request_body("An essay.", "cOPN")
        self.assertEqual(body["model"], "gpt-5.6-luna")
        self.assertEqual(body["reasoning"]["effort"], "none")
        self.assertEqual(body["max_output_tokens"], MAX_OUTPUT_TOKENS)
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(schema()["required"], ["high_trait_probability"])

    def test_estimate_grows_with_text(self) -> None:
        self.assertGreater(conservative_cost_usd(["x" * 3000]), conservative_cost_usd(["x" * 300]))


if __name__ == "__main__":
    unittest.main()
