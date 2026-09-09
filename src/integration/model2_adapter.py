"""Adapter from locked Model 1 scores to the locked Model 2 artifact."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np

from .schemas import (
    MODEL2_FEATURE_COLUMNS,
    MODEL2_TARGET_COLUMNS,
    ContractError,
    Model1Scores,
    Model2Features,
    Model2Predictions,
)


SKILL_TO_TARGET = {
    "self_management": "bessi_self_management",
    "social_engagement": "bessi_social_engagement",
    "cooperation": "bessi_cooperation",
    "emotional_resilience": "bessi_emotional_resilience",
    "innovation": "bessi_innovation",
}


class Model2AdapterError(ContractError):
    """Raised when the Model 2 artifact or prediction violates its contract."""


@dataclass(frozen=True)
class Model2Adapter:
    """Loaded, schema-checked Model 2 artifact."""

    model: object
    artifact_path: Path
    metadata: Mapping[str, object]
    artifact_sha256: str

    def predict(self, features: Model2Features) -> Model2Predictions:
        """Predict all five BESSI domains for one validated feature vector."""
        try:
            raw = np.asarray(self.model.predict(np.asarray([features.values], dtype=float)), dtype=float)
        except Exception as error:
            raise Model2AdapterError("Model 2 prediction failed.") from error
        if raw.shape != (1, len(MODEL2_TARGET_COLUMNS)):
            raise Model2AdapterError("Model 2 returned an unexpected prediction shape.")
        try:
            predictions = Model2Predictions(tuple(float(value) for value in raw[0]), str(self.artifact_path))
        except (TypeError, ValueError, ContractError) as error:
            raise Model2AdapterError("Model 2 returned invalid predictions.") from error
        return predictions


def model1_to_model2_features(scores: Model1Scores) -> Model2Features:
    """Reorder Model 1 traits and convert native 0--10 scores to 0--1."""
    by_code = scores.as_code_mapping()
    values = (
        by_code["cEXT"] / 10.0,
        by_code["cAGR"] / 10.0,
        by_code["cCON"] / 10.0,
        by_code["cNEU"] / 10.0,
        by_code["cOPN"] / 10.0,
    )
    return Model2Features(values)


def select_skill_predictions(predictions: Model2Predictions, selected_skills: tuple[str, ...]) -> dict[str, float]:
    """Return only the skills requested by the released instrument."""
    unknown = sorted(set(selected_skills) - set(SKILL_TO_TARGET))
    if unknown:
        raise Model2AdapterError(f"Unknown selected skill(s): {', '.join(unknown)}.")
    values = predictions.as_mapping()
    return {skill: values[SKILL_TO_TARGET[skill]] for skill in selected_skills}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model2_adapter(path: Path) -> Model2Adapter:
    """Load and verify the selected linear Model 2 artifact."""
    try:
        bundle = joblib.load(path)
    except Exception as error:
        raise Model2AdapterError(f"Cannot load Model 2 artifact {path}.") from error
    if not isinstance(bundle, Mapping) or "model" not in bundle or "metadata" not in bundle:
        raise Model2AdapterError("Model 2 artifact must contain model and metadata entries.")
    model = bundle["model"]
    metadata = bundle["metadata"]
    if not isinstance(metadata, Mapping):
        raise Model2AdapterError("Model 2 artifact metadata must be an object.")
    if tuple(metadata.get("feature_columns", ())) != MODEL2_FEATURE_COLUMNS:
        raise Model2AdapterError("Model 2 artifact feature columns do not match the integration contract.")
    if tuple(metadata.get("target_columns", ())) != MODEL2_TARGET_COLUMNS:
        raise Model2AdapterError("Model 2 artifact target columns do not match the integration contract.")
    if not callable(getattr(model, "predict", None)):
        raise Model2AdapterError("Model 2 artifact does not expose a predict method.")
    if getattr(model, "n_features_in_", len(MODEL2_FEATURE_COLUMNS)) != len(MODEL2_FEATURE_COLUMNS):
        raise Model2AdapterError("Model 2 artifact expects an incompatible number of features.")
    try:
        digest = _sha256(path)
    except OSError as error:
        raise Model2AdapterError(f"Cannot hash Model 2 artifact {path}.") from error
    return Model2Adapter(model, path, dict(metadata), digest)
