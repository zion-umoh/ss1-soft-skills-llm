import unittest

from src.models.train_model1_embedding_ablation import chunk_text


class Model1EmbeddingAblationTests(unittest.TestCase):
    def test_chunk_text_preserves_order_and_respects_limit(self) -> None:
        chunks = chunk_text("one two three four five", words_per_chunk=2)
        self.assertEqual(chunks, ["one two", "three four", "five"])

    def test_chunk_text_represents_empty_document(self) -> None:
        self.assertEqual(chunk_text(""), [""])
