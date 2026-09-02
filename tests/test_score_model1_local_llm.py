import json
import unittest

import numpy as np

from src.models.score_model1_local_llm import (
    TRAITS,
    parse_scores,
    prediction_fields,
    prompt_for,
    response_schema,
)


class LocalModel1ScorerTests(unittest.TestCase):
    def test_schema_and_parser_require_five_bounded_scores(self) -> None:
        payload = {f"{trait}_score": 6.0 for trait in TRAITS}
        parsed = parse_scores({"message": {"content": json.dumps(payload)}})
        self.assertEqual(parsed["cEXT_score"], 6.0)
        self.assertEqual(response_schema()["required"], list(payload))

    def test_prompt_is_label_free_and_requests_no_reasoning(self) -> None:
        prompt = prompt_for("A short essay.")
        self.assertIn("A short essay.", prompt)
        self.assertIn("no explanation or reasoning", prompt)
        self.assertNotIn("cEXT", prompt)

    def test_prediction_schema_has_exactly_one_score_per_trait(self) -> None:
        fields = prediction_fields()
        self.assertEqual([field for field in fields if field.startswith("score_")], [f"score_{trait}" for trait in TRAITS])

    def test_out_of_range_score_is_rejected(self) -> None:
        payload = {f"{trait}_score": 6.0 for trait in TRAITS}
        payload["cOPN_score"] = 10.1
        with self.assertRaises(ValueError):
            parse_scores({"message": {"content": json.dumps(payload)}})


if __name__ == "__main__":
    unittest.main()
