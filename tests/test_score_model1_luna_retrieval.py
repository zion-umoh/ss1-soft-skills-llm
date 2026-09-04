import unittest

import numpy as np

from src.models.score_model1_luna_retrieval import (
    K_RETRIEVED_EXAMPLES, conservative_cost_usd, input_for, instructions, request_body, retrieve_indices,
    trait_balanced_retrieve_indices,
)


class LunaRetrievalTests(unittest.TestCase):
    def test_retrieval_is_top_k_and_deterministic(self) -> None:
        train_ids = ["z", "a", "b"]
        train = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
        validation = np.asarray([[1.0, 0.0]], dtype=np.float32)
        self.assertEqual(retrieve_indices(train_ids, train, validation, 2), [[0, 1]])

    def test_examples_include_training_labels_but_not_target_labels(self) -> None:
        rendered = input_for("target text", [("reference text", np.asarray([1, 0, 1, 0, 1]))])
        self.assertIn("cEXT=high", rendered)
        self.assertIn("TARGET ESSAY:\ntarget text", rendered)

    def test_trait_balanced_retrieval_has_a_nearest_high_and_low_anchor_per_trait(self) -> None:
        train_ids = ["a", "b", "c", "d"]
        train = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
        labels = np.asarray([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 1, 0, 0, 1], [0, 0, 1, 1, 0]])
        selected = trait_balanced_retrieve_indices(train_ids, train, np.asarray([[1.0, 0.0]], dtype=np.float32), labels)
        self.assertEqual(selected, [[0, 1]])

    def test_request_is_joint_continuous_luna(self) -> None:
        self.assertIn("do not report a binary class", instructions())
        body = request_body("input")
        self.assertEqual(body["model"], "gpt-5.6-luna")
        self.assertEqual(body["reasoning"]["effort"], "none")
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(K_RETRIEVED_EXAMPLES, 3)

    def test_request_can_use_medium_reasoning_with_a_larger_allowance(self) -> None:
        body = request_body("input", "medium", 512)
        self.assertEqual(body["reasoning"]["effort"], "medium")
        self.assertEqual(body["max_output_tokens"], 512)

    def test_estimate_grows_with_input(self) -> None:
        self.assertGreater(conservative_cost_usd(["x" * 3000]), conservative_cost_usd(["x" * 300]))


if __name__ == "__main__":
    unittest.main()
