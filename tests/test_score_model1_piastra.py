import tempfile
import unittest
from pathlib import Path
from email.message import Message
from urllib.error import HTTPError

import numpy as np

from src.models.score_model1_piastra import (
    TRAITS,
    binary_predictions,
    load_env_value,
    load_partial_predictions,
    parse_scores,
    prompt_for,
    response_schema,
    retry_delay_seconds,
    trusted_ssl_context,
)


class Model1PiastraTests(unittest.TestCase):
    def test_binary_conversion_uses_the_predeclared_midpoint(self) -> None:
        scores = np.full((1, len(TRAITS)), 4.99)
        scores[0, 0] = 5.0
        predictions = binary_predictions(scores)
        self.assertEqual(predictions.tolist(), [[1, 0, 0, 0, 0]])

    def test_scores_are_parsed_and_range_checked(self) -> None:
        fields = {}
        for trait in TRAITS:
            fields[f"{trait}_score"] = 6.0
            fields[f"{trait}_confidence"] = 7.0
        parsed = parse_scores({"output_text": __import__("json").dumps(fields)})
        self.assertEqual(parsed["cEXT_score"], 6.0)
        self.assertEqual(response_schema()["required"], list(fields))

    def test_prompt_contains_text_but_no_dataset_label_names(self) -> None:
        prompt = prompt_for("A short essay.")
        self.assertIn("A short essay.", prompt)
        self.assertNotIn("cEXT", prompt)

    def test_env_value_reads_assignment_without_executing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("export OPENAI_API_KEY='test-value'\nUNRELATED=ignored\n", encoding="utf-8")
            self.assertEqual(load_env_value(path, "OPENAI_API_KEY"), "test-value")

    def test_ssl_context_requires_certificates(self) -> None:
        self.assertEqual(trusted_ssl_context().verify_mode.name, "CERT_REQUIRED")

    def test_missing_partial_predictions_load_as_empty(self) -> None:
        self.assertEqual(load_partial_predictions(Path("/private/tmp/does-not-exist-model1.csv"), "validation"), [])

    def test_rate_limit_delay_uses_api_millisecond_hint(self) -> None:
        headers = Message()
        error = HTTPError("https://example.test", 429, "rate limit", headers, None)
        self.assertEqual(retry_delay_seconds(error, "Try again in 600ms.", 0), 0.6)
