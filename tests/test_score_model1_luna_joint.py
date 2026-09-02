import unittest

from src.models.score_model1_luna_joint import MAX_OUTPUT_TOKENS, conservative_cost_usd, instructions, request_body, schema


class LunaJointTests(unittest.TestCase):
    def test_joint_prompt_is_continuous_and_label_free(self) -> None:
        prompt = instructions()
        self.assertIn("continuous Big Five trait level", prompt)
        self.assertIn("not estimate class membership", prompt)
        self.assertNotIn("higher half", prompt)

    def test_request_uses_luna_none_reasoning_and_all_trait_schema(self) -> None:
        body = request_body("An essay.")
        self.assertEqual(body["model"], "gpt-5.6-luna")
        self.assertEqual(body["reasoning"]["effort"], "none")
        self.assertEqual(body["max_output_tokens"], MAX_OUTPUT_TOKENS)
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(set(schema()["required"]), {"cEXT_score", "cNEU_score", "cAGR_score", "cCON_score", "cOPN_score"})

    def test_estimate_grows_with_text(self) -> None:
        self.assertGreater(conservative_cost_usd(["x" * 3000]), conservative_cost_usd(["x" * 300]))


if __name__ == "__main__":
    unittest.main()
