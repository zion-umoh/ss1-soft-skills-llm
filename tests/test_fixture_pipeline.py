import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.integration.fixture_pipeline import FixturePipelineError, run_fixture_pipeline


ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = ROOT / "tests"
MODEL2 = ROOT / "outputs/models/model2_linear_benchmark.joblib"


class FixturePipelineTests(unittest.TestCase):
    def run_fixture(self, fixture: str) -> dict[str, object]:
        return run_fixture_pipeline(
            FIXTURE_ROOT / f"fixture_{fixture}_skill_delivery.csv",
            FIXTURE_ROOT / f"fixture_{fixture}_skill_metadata.json",
            FIXTURE_ROOT / f"fixture_{fixture}_skill_responses.csv",
            MODEL2,
        )

    def test_one_skill_assessment_completes(self) -> None:
        report = self.run_fixture("one")
        self.assertEqual(report["status"], "completed")
        output = report["respondent_output"]
        self.assertEqual(output["selected_skills"], ["self_management"])
        self.assertEqual(tuple(output["predictions"]), ("self_management",))

    def test_three_skill_assessment_completes_and_model1_input_is_neutral(self) -> None:
        report = self.run_fixture("three")
        self.assertEqual(report["inputs"]["scenario_count"], 6)
        self.assertEqual(
            report["respondent_output"]["selected_skills"],
            ["self_management", "cooperation", "innovation"],
        )
        model1_input = report["inputs"]["model1_input"]
        for hidden_skill in ("self_management", "cooperation", "innovation"):
            self.assertNotIn(hidden_skill, model1_input)
        self.assertEqual(len(report["model2"]["predictions"]), 5)

    def test_repeated_runs_are_structurally_reproducible(self) -> None:
        first = self.run_fixture("three")
        second = self.run_fixture("three")
        self.assertEqual(first, second)

    def test_input_failure_is_propagated_without_partial_report(self) -> None:
        with TemporaryDirectory() as directory:
            response_path = Path(directory) / "invalid.csv"
            response_path.write_text(
                (FIXTURE_ROOT / "fixture_one_skill_responses.csv").read_text(encoding="utf-8").splitlines()[0] + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(FixturePipelineError):
                run_fixture_pipeline(
                    FIXTURE_ROOT / "fixture_one_skill_delivery.csv",
                    FIXTURE_ROOT / "fixture_one_skill_metadata.json",
                    response_path,
                    MODEL2,
                )

    def test_model1_failure_is_propagated(self) -> None:
        with self.assertRaises(FixturePipelineError):
            run_fixture_pipeline(
                FIXTURE_ROOT / "fixture_one_skill_delivery.csv",
                FIXTURE_ROOT / "fixture_one_skill_metadata.json",
                FIXTURE_ROOT / "fixture_one_skill_responses.csv",
                MODEL2,
                model1_response={"status": "completed", "output_text": "{}"},
            )

    def test_model2_failure_is_propagated(self) -> None:
        with self.assertRaises(FixturePipelineError):
            self.run_fixture_with_model2(Path("does-not-exist.joblib"))

    def run_fixture_with_model2(self, model2_path: Path) -> dict[str, object]:
        return run_fixture_pipeline(
            FIXTURE_ROOT / "fixture_one_skill_delivery.csv",
            FIXTURE_ROOT / "fixture_one_skill_metadata.json",
            FIXTURE_ROOT / "fixture_one_skill_responses.csv",
            model2_path,
        )

    def test_report_can_be_written_as_json(self) -> None:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "report.json"
            report = run_fixture_pipeline(
                FIXTURE_ROOT / "fixture_one_skill_delivery.csv",
                FIXTURE_ROOT / "fixture_one_skill_metadata.json",
                FIXTURE_ROOT / "fixture_one_skill_responses.csv",
                MODEL2,
                output_path=output_path,
            )
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), report)


if __name__ == "__main__":
    unittest.main()
