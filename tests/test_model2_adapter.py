import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from src.integration.model1_adapter import LOCKED_MODEL1_NAME
from src.integration.model2_adapter import (
    MODEL2_FEATURE_COLUMNS,
    MODEL2_TARGET_COLUMNS,
    Model2AdapterError,
    load_model2_adapter,
    model1_to_model2_features,
    select_skill_predictions,
)
from src.integration.schemas import Model1Scores, Model2Predictions


class BadModel:
    n_features_in_ = 5

    def predict(self, _features):
        return [[1, 2]]


def write_artifact(path: Path, feature_columns=MODEL2_FEATURE_COLUMNS, target_columns=MODEL2_TARGET_COLUMNS) -> None:
    model = LinearRegression().fit(
        np.asarray([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=float),
        np.asarray([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=float),
    )
    joblib.dump({"model": model, "metadata": {"feature_columns": list(feature_columns), "target_columns": list(target_columns)}}, path)


class Model2AdapterTests(unittest.TestCase):
    def test_reorders_and_normalizes_model1_scores(self) -> None:
        scores = Model1Scores((1, 2, 3, 4, 5), LOCKED_MODEL1_NAME)
        features = model1_to_model2_features(scores)
        self.assertEqual(features.values, (0.1, 0.3, 0.4, 0.2, 0.5))
        self.assertEqual(tuple(features.as_mapping()), MODEL2_FEATURE_COLUMNS)

    def test_loads_artifact_checks_schema_and_predicts(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model2.joblib"
            write_artifact(path)
            adapter = load_model2_adapter(path)
            predictions = adapter.predict(model1_to_model2_features(Model1Scores((5, 5, 5, 5, 5), LOCKED_MODEL1_NAME)))
            self.assertEqual(predictions.model_artifact, str(path))
            self.assertEqual(len(predictions.values), 5)
            self.assertTrue(all(np.isfinite(predictions.values)))
            self.assertEqual(tuple(predictions.as_mapping()), MODEL2_TARGET_COLUMNS)

    def test_rejects_incompatible_artifact_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model2.joblib"
            write_artifact(path, feature_columns=("wrong",) * 5)
            with self.assertRaisesRegex(Model2AdapterError, "feature columns"):
                load_model2_adapter(path)

    def test_filters_only_selected_skills(self) -> None:
        predictions = Model2Predictions((1, 2, 3, 4, 5), "model2.joblib")
        selected = select_skill_predictions(predictions, ("cooperation", "innovation"))
        self.assertEqual(selected, {"cooperation": 3.0, "innovation": 5.0})
        with self.assertRaisesRegex(Model2AdapterError, "Unknown selected skill"):
            select_skill_predictions(predictions, ("not_a_skill",))

    def test_prediction_shape_is_checked(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model2.joblib"
            joblib.dump({"model": BadModel(), "metadata": {"feature_columns": list(MODEL2_FEATURE_COLUMNS), "target_columns": list(MODEL2_TARGET_COLUMNS)}}, path)
            adapter = load_model2_adapter(path)
            with self.assertRaisesRegex(Model2AdapterError, "shape"):
                adapter.predict(model1_to_model2_features(Model1Scores((5, 5, 5, 5, 5), LOCKED_MODEL1_NAME)))


if __name__ == "__main__":
    unittest.main()
