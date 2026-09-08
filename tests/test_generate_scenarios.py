import io
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

from src.scenarios.generate_scenarios import (
    MAX_SELECTED_SKILLS, SCENARIOS_PER_SKILL, build_instrument, evaluate_candidate,
    SKILL_CATALOG, generate_candidates, main, parse_selected_skills, revise_candidate,
    select_skills_interactively, write_delivery_instrument,
)


class ScenarioInstrumentPipelineTests(unittest.TestCase):
    def test_multiselect_is_validated_and_burden_capped(self) -> None:
        self.assertEqual(parse_selected_skills("cooperation,innovation"), ("cooperation", "innovation"))
        with self.assertRaises(ValueError):
            parse_selected_skills("cooperation,cooperation")
        with self.assertRaises(ValueError):
            parse_selected_skills("self_management,social_engagement,cooperation,innovation")
        self.assertEqual(MAX_SELECTED_SKILLS, 3)

    def test_selected_instrument_is_balanced_reproducible_and_release_ready(self) -> None:
        skills = parse_selected_skills("self_management,cooperation,innovation")
        first, metadata = build_instrument(skills)
        second, repeated_metadata = build_instrument(skills)
        self.assertEqual(first, second)
        self.assertEqual(metadata["release_gate"], repeated_metadata["release_gate"])
        self.assertTrue(metadata["meets_release_gate"])
        self.assertEqual(len(first), len(skills) * SCENARIOS_PER_SKILL)
        self.assertEqual({candidate.skill_id for candidate in first}, set(skills))
        self.assertTrue(all(evaluate_candidate(candidate)["passes_threshold"] for candidate in first))

    def test_every_supported_selection_passes_the_release_gate(self) -> None:
        skill_ids = tuple(SKILL_CATALOG)
        for size in range(1, MAX_SELECTED_SKILLS + 1):
            for selected in combinations(skill_ids, size):
                candidates, metadata = build_instrument(selected)
                self.assertTrue(metadata["meets_release_gate"], selected)
                self.assertEqual(len(candidates), size * SCENARIOS_PER_SKILL)

    def test_weak_candidate_fails_predefined_rubric(self) -> None:
        candidate = generate_candidates(parse_selected_skills("cooperation"))[0]
        weak = replace(candidate, text="A vague group situation.")
        result = evaluate_candidate(weak)
        self.assertFalse(result["passes_threshold"])
        self.assertLess(result["total_score"], 12)
        self.assertTrue(evaluate_candidate(revise_candidate(weak))["passes_threshold"])

    def test_delivery_file_excludes_skill_map_and_audit_fields(self) -> None:
        candidates, _ = build_instrument(parse_selected_skills("cooperation"))
        with TemporaryDirectory() as directory:
            output = Path(directory) / "delivery.csv"
            write_delivery_instrument(output, candidates)
            header = output.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header, "scenario_id,delivery_order,text")

    def test_console_selection_uses_curated_catalogue(self) -> None:
        answers = iter(["1,3"])
        selected = select_skills_interactively(read=lambda _: next(answers), write=lambda _: None)
        self.assertEqual(selected, ("self_management", "cooperation"))

    def test_interactive_mode_does_not_require_a_model_or_api_key(self) -> None:
        answers = iter(["1,3"])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            main(
                ["--interactive", "--delivery", str(root / "delivery.csv"), "--audit-bank", str(root / "audit.csv"), "--metadata", str(root / "metadata.json"), "--report", str(root / "report.md")],
                read=lambda _: next(answers),
                write=lambda _: None,
            )
            self.assertTrue((root / "delivery.csv").is_file())

    def test_skills_are_required_for_noninteractive_cli(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main([])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
