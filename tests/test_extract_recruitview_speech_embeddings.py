from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.extract_recruitview_speech_embeddings import _atomic_write, _load_checkpoint


class SpeechEmbeddingCheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.jsonl"
            _atomic_write(path, json.dumps({"record_id": "r1", "embedding": [0.1, 0.2]}) + "\n")
            self.assertEqual(_load_checkpoint(path), {"r1": [0.1, 0.2]})


if __name__ == "__main__":
    unittest.main()
