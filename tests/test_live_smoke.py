import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.integration.fixture_pipeline import FIXTURE_MODEL1_RESPONSE
from src.integration.live_smoke import LiveSmokeError, run_live_smoke


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests"
MODEL2 = ROOT / "outputs/models/model2_linear_benchmark.joblib"


class FakeResponses:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls = 0
        self.payloads: list[dict[str, object]] = []

    def create(self, **payload: object) -> object:
        self.calls += 1
        self.payloads.append(payload)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.responses = FakeResponses(response, error)


class LiveSmokeTests(unittest.TestCase):
    def paths(self) -> tuple[Path, Path, Path]:
        return (
            FIXTURES / "fixture_one_skill_delivery.csv",
            FIXTURES / "fixture_one_skill_metadata.json",
            FIXTURES / "fixture_one_skill_responses.csv",
        )

    def test_confirmation_gate_prevents_remote_call(self) -> None:
        client = FakeClient(FIXTURE_MODEL1_RESPONSE)
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(LiveSmokeError, "explicit confirmation"):
                run_live_smoke(*self.paths(), MODEL2, api_key="test-key", output_dir=Path(directory), client=client)
        self.assertEqual(client.responses.calls, 0)

    def test_success_uses_one_request_and_saves_raw_and_derived_artifacts(self) -> None:
        client = FakeClient(FIXTURE_MODEL1_RESPONSE)
        with TemporaryDirectory() as directory:
            result = run_live_smoke(*self.paths(), MODEL2, api_key="test-key", output_dir=Path(directory), confirm=True, client=client)
            self.assertEqual(client.responses.calls, 1)
            self.assertEqual(result["metadata"]["request_count"], 1)
            self.assertEqual(result["report"]["mode"], "live_smoke")
            self.assertTrue((Path(directory) / "model1-raw-response.json").is_file())
            self.assertTrue((Path(directory) / "live-assessment-report.json").is_file())
            metadata = json.loads((Path(directory) / "live-smoke-metadata.json").read_text(encoding="utf-8"))
            self.assertIn("Connectivity and contract smoke test only", metadata["interpretation"])
            self.assertEqual(client.responses.payloads[0]["model"], result["report"]["model1"]["model_name"])

    def test_remote_failure_writes_failure_metadata_without_report(self) -> None:
        client = FakeClient(error=RuntimeError("network unavailable"))
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(LiveSmokeError, "Live smoke test failed"):
                run_live_smoke(*self.paths(), MODEL2, api_key="test-key", output_dir=Path(directory), confirm=True, client=client)
            failure_path = Path(directory) / "live-smoke-failure.json"
            self.assertTrue(failure_path.is_file())
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertEqual(failure["request_count"], 1)
            self.assertFalse((Path(directory) / "live-assessment-report.json").exists())

    def test_invalid_model1_response_does_not_release_report(self) -> None:
        client = FakeClient({"status": "completed", "output_text": "{}"})
        with TemporaryDirectory() as directory:
            with self.assertRaises(LiveSmokeError):
                run_live_smoke(*self.paths(), MODEL2, api_key="test-key", output_dir=Path(directory), confirm=True, client=client)
            self.assertTrue((Path(directory) / "model1-raw-response.json").is_file())
            self.assertFalse((Path(directory) / "live-assessment-report.json").exists())

    def test_existing_artifacts_are_rejected_before_remote_call(self) -> None:
        client = FakeClient(FIXTURE_MODEL1_RESPONSE)
        with TemporaryDirectory() as directory:
            (Path(directory) / "live-assessment-report.json").write_text("previous", encoding="utf-8")
            with self.assertRaisesRegex(LiveSmokeError, "already contains"):
                run_live_smoke(*self.paths(), MODEL2, api_key="test-key", output_dir=Path(directory), confirm=True, client=client)
        self.assertEqual(client.responses.calls, 0)


if __name__ == "__main__":
    unittest.main()
