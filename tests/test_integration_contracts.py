import unittest

from src.integration.schemas import (
    MODEL1_TRAIT_CODES,
    MODEL2_FEATURE_COLUMNS,
    MODEL2_TARGET_COLUMNS,
    ContractError,
    InstrumentScenario,
    Model1Scores,
    Model2Features,
    Model2Predictions,
    ReleasedInstrument,
    ResponseRecord,
    model1_scores_from_mapping,
)


class IntegrationContractTests(unittest.TestCase):
    def test_released_instrument_requires_contiguous_delivery_order(self) -> None:
        scenarios = (
            InstrumentScenario("scenario-1", 1, "First question."),
            InstrumentScenario("scenario-2", 2, "Second question."),
        )
        instrument = ReleasedInstrument("4.0.0", ("cooperation",), scenarios)
        self.assertEqual(instrument.scenarios[0].delivery_order, 1)
        with self.assertRaises(ContractError):
            ReleasedInstrument("4.0.0", ("cooperation",), (InstrumentScenario("scenario-1", 2, "Question."),))

    def test_response_contract_requires_non_empty_text(self) -> None:
        response = ResponseRecord("assessment-1", "scenario-1", 1, "I would speak with the team first.")
        self.assertEqual(response.assessment_id, "assessment-1")
        with self.assertRaises(ContractError):
            ResponseRecord("assessment-1", "scenario-1", 1, "  ")

    def test_model1_contract_uses_native_zero_to_ten_scores(self) -> None:
        scores = model1_scores_from_mapping({f"{trait}_score": index * 2.5 for index, trait in enumerate(MODEL1_TRAIT_CODES)}, "locked-model1")
        self.assertEqual(tuple(scores.as_code_mapping()), MODEL1_TRAIT_CODES)
        self.assertEqual(scores.scores[-1], 10.0)
        with self.assertRaises(ContractError):
            Model1Scores((0, 0, 0, 0, 11), "locked-model1")

    def test_model2_contract_preserves_declared_feature_and_target_order(self) -> None:
        features = Model2Features((0.1, 0.2, 0.3, 0.4, 0.5))
        self.assertEqual(tuple(features.as_mapping()), MODEL2_FEATURE_COLUMNS)
        predictions = Model2Predictions((1, 2, 3, 4, 5), "model2.joblib")
        self.assertEqual(tuple(predictions.as_mapping()), MODEL2_TARGET_COLUMNS)
        with self.assertRaises(ContractError):
            Model2Features((0.1, 0.2, 0.3, 0.4, 1.1))

    def test_target_skill_is_internal_metadata_not_required_for_delivery(self) -> None:
        scenario = InstrumentScenario("scenario-1", 1, "Question text.")
        self.assertIsNone(scenario.target_skill)
        mapped = InstrumentScenario("scenario-1", 1, "Question text.", target_skill="cooperation")
        self.assertEqual(mapped.target_skill, "cooperation")


if __name__ == "__main__":
    unittest.main()
