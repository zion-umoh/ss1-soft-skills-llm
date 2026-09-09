import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.integration.benchmark_3sq import (
    CROSSWALK,
    MODEL_COLUMNS,
    PAIRED_FIELDS,
    Benchmark3SQError,
    benchmark_metrics,
    crosswalk_scores,
    load_paired_dataset,
    run_benchmark,
)


def write_dataset(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAIRED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def paired_row(index: int) -> dict[str, object]:
    base = 2.0 + index * 0.3
    model = {
        "self_management": base,
        "social_engagement": base + 0.1,
        "cooperation": base + 0.2,
        "emotional_resilience": base + 0.3,
        "innovation": base + 0.4,
    }
    dimensions = {
        "self_confidence": 3.0,
        "curiosity": model["innovation"],
        "resilience": model["emotional_resilience"],
        "openness": model["social_engagement"],
        "collaboration": model["cooperation"],
        "empathy": model["social_engagement"],
        "leadership": model["social_engagement"],
        "commitment": model["self_management"],
        "autonomy": model["self_management"],
        "problem_solving": model["innovation"],
    }
    return {
        "participant_id": f"participant-{index}",
        "split": "test",
        **{f"model_{skill}": value for skill, value in model.items()},
        **{f"three_sq_{dimension}": value for dimension, value in dimensions.items()},
    }


class Benchmark3SQTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [paired_row(index) for index in range(8)]

    def test_contract_requires_participant_level_pairs_and_exact_fields(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "paired.csv"
            write_dataset(path, self.rows)
            participants = load_paired_dataset(path)
            self.assertEqual(len(participants), 8)
            self.assertEqual(tuple(f"model_{skill}" for skill in participants[0].model_mapping()), MODEL_COLUMNS)
            with self.assertRaisesRegex(Benchmark3SQError, "more than once"):
                write_dataset(path, self.rows + [self.rows[0]])
                load_paired_dataset(path)

    def test_crosswalk_maps_five_skills_to_declared_3sq_dimensions(self) -> None:
        scores = crosswalk_scores(load_paired_dataset_from_rows(self.rows)[0])
        self.assertEqual(set(scores), set(CROSSWALK))
        self.assertEqual(CROSSWALK["cooperation"], ("collaboration",))
        self.assertAlmostEqual(scores["self_management"], self.rows[0]["model_self_management"])

    def test_metrics_include_association_error_agreement_and_confidence_intervals(self) -> None:
        metrics = benchmark_metrics(load_paired_dataset_from_rows(self.rows), bootstrap_resamples=100)
        self.assertAlmostEqual(metrics["cooperation"]["metrics"]["pearson_r"], 1.0)
        self.assertAlmostEqual(metrics["cooperation"]["metrics"]["mae"], 0.0)
        self.assertAlmostEqual(metrics["cooperation"]["metrics"]["concordance_correlation"], 1.0)
        self.assertEqual(len(metrics["cooperation"]["bootstrap_ci_95"]["pearson_r"]), 2)
        self.assertEqual(metrics["social_engagement"]["interpretation"], "exploratory crosswalk")

    def test_runner_writes_json_and_markdown_report(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paired = root / "paired.csv"
            output = root / "benchmark.json"
            report = root / "benchmark.md"
            write_dataset(paired, self.rows)
            result = run_benchmark(paired, bootstrap_resamples=100, output_path=output, report_path=report)
            self.assertEqual(result["protocol"]["synthetic_ground_truth"], False)
            self.assertTrue(output.is_file())
            self.assertTrue(report.is_file())
            self.assertIn("3SQ Benchmark Report", report.read_text(encoding="utf-8"))

    def test_wrong_split_and_out_of_scale_values_fail(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "paired.csv"
            wrong_split = list(self.rows)
            wrong_split[0] = {**wrong_split[0], "split": "train"}
            write_dataset(path, wrong_split)
            with self.assertRaisesRegex(Benchmark3SQError, "requested 'test' split"):
                load_paired_dataset(path)
            out_of_scale = list(self.rows)
            out_of_scale[0] = {**out_of_scale[0], "model_cooperation": 6}
            write_dataset(path, out_of_scale)
            with self.assertRaisesRegex(Benchmark3SQError, "between 1 and 5"):
                load_paired_dataset(path)


def load_paired_dataset_from_rows(rows: list[dict[str, object]]):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "paired.csv"
        write_dataset(path, rows)
        return load_paired_dataset(path)


if __name__ == "__main__":
    unittest.main()
