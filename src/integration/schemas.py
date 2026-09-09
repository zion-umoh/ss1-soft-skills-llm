"""Batch-0 data contracts for the runtime assessment pipeline.

These contracts deliberately contain no model calls or file I/O. They define
the stable boundary between the released scenario instrument, respondent
responses, Model 1, and Model 2. Runtime loading and validation are added in
later implementation batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


MODEL1_TRAIT_CODES = ("cEXT", "cNEU", "cAGR", "cCON", "cOPN")
MODEL1_TRAIT_NAMES = {
    "cEXT": "extraversion",
    "cNEU": "negative_emotionality",
    "cAGR": "agreeableness",
    "cCON": "conscientiousness",
    "cOPN": "open_mindedness",
}
MODEL2_FEATURE_COLUMNS = (
    "big_five_extraversion",
    "big_five_agreeableness",
    "big_five_conscientiousness",
    "big_five_negative_emotionality",
    "big_five_open_mindedness",
)
MODEL2_TARGET_COLUMNS = (
    "bessi_self_management",
    "bessi_social_engagement",
    "bessi_cooperation",
    "bessi_emotional_resilience",
    "bessi_innovation",
)


class ContractError(ValueError):
    """Raised when an integration object violates its data contract."""


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string.")
    return value.strip()


def _finite_number(value: object, field: str, lower: float, upper: float) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{field} must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{field} must be numeric.") from error
    if not isfinite(number) or not lower <= number <= upper:
        raise ContractError(f"{field} must be finite and between {lower:g} and {upper:g}.")
    return number


@dataclass(frozen=True)
class InstrumentScenario:
    """One released respondent-facing scenario.

    ``target_skill`` is optional because it belongs to the internal audit
    mapping, not to the respondent-facing delivery file or Model 1 input.
    """

    scenario_id: str
    delivery_order: int
    text: str
    target_skill: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_id", _required_text(self.scenario_id, "scenario_id"))
        if not isinstance(self.delivery_order, int) or isinstance(self.delivery_order, bool) or self.delivery_order < 1:
            raise ContractError("delivery_order must be a positive integer.")
        object.__setattr__(self, "text", _required_text(self.text, "scenario text"))
        if self.target_skill is not None:
            object.__setattr__(self, "target_skill", _required_text(self.target_skill, "target_skill"))


@dataclass(frozen=True)
class ReleasedInstrument:
    """Versioned instrument contract consumed by the runtime pipeline."""

    pipeline_version: str
    selected_skills: tuple[str, ...]
    scenarios: tuple[InstrumentScenario, ...]
    delivery_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pipeline_version", _required_text(self.pipeline_version, "pipeline_version"))
        skills = tuple(_required_text(skill, "selected skill") for skill in self.selected_skills)
        if not skills or len(skills) != len(set(skills)):
            raise ContractError("selected_skills must contain unique, non-empty skill IDs.")
        object.__setattr__(self, "selected_skills", skills)
        scenarios = tuple(self.scenarios)
        if not scenarios:
            raise ContractError("scenarios must contain at least one released scenario.")
        if any(not isinstance(scenario, InstrumentScenario) for scenario in scenarios):
            raise ContractError("scenarios must contain InstrumentScenario objects.")
        identifiers = [scenario.scenario_id for scenario in scenarios]
        orders = [scenario.delivery_order for scenario in scenarios]
        if len(identifiers) != len(set(identifiers)):
            raise ContractError("scenario IDs must be unique.")
        if sorted(orders) != list(range(1, len(orders) + 1)):
            raise ContractError("delivery_order must be contiguous starting at 1.")
        object.__setattr__(self, "scenarios", scenarios)
        if self.delivery_sha256 is not None:
            digest = _required_text(self.delivery_sha256, "delivery_sha256")
            if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
                raise ContractError("delivery_sha256 must be a 64-character hexadecimal digest.")
            object.__setattr__(self, "delivery_sha256", digest.lower())


@dataclass(frozen=True)
class ResponseRecord:
    """One respondent answer linked to one released scenario."""

    assessment_id: str
    scenario_id: str
    response_order: int
    response_text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "assessment_id", _required_text(self.assessment_id, "assessment_id"))
        object.__setattr__(self, "scenario_id", _required_text(self.scenario_id, "scenario_id"))
        if not isinstance(self.response_order, int) or isinstance(self.response_order, bool) or self.response_order < 1:
            raise ContractError("response_order must be a positive integer.")
        object.__setattr__(self, "response_text", _required_text(self.response_text, "response_text"))


@dataclass(frozen=True)
class Model1Scores:
    """The locked Model 1 continuous Big Five output contract.

    Scores are kept in the model's native 0--10 scale. The Model 2 adapter is
    responsible for the explicit 0--10 to 0--1 conversion.
    """

    scores: tuple[float, float, float, float, float]
    model_name: str
    confidence: tuple[float, float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if len(self.scores) != len(MODEL1_TRAIT_CODES):
            raise ContractError("Model 1 must return exactly five trait scores.")
        normalized = tuple(_finite_number(value, f"Model 1 {trait} score", 0.0, 10.0) for trait, value in zip(MODEL1_TRAIT_CODES, self.scores, strict=True))
        object.__setattr__(self, "scores", normalized)
        object.__setattr__(self, "model_name", _required_text(self.model_name, "model_name"))
        if self.confidence is not None:
            if len(self.confidence) != len(MODEL1_TRAIT_CODES):
                raise ContractError("Model 1 confidence must contain exactly five values.")
            object.__setattr__(self, "confidence", tuple(_finite_number(value, f"Model 1 {trait} confidence", 0.0, 10.0) for trait, value in zip(MODEL1_TRAIT_CODES, self.confidence, strict=True)))

    def as_code_mapping(self) -> dict[str, float]:
        """Return scores keyed in the locked Model 1 trait order."""
        return dict(zip(MODEL1_TRAIT_CODES, self.scores, strict=True))


@dataclass(frozen=True)
class Model2Features:
    """Model 2's normalized five-feature input contract."""

    values: tuple[float, float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.values) != len(MODEL2_FEATURE_COLUMNS):
            raise ContractError("Model 2 must receive exactly five Big Five features.")
        object.__setattr__(self, "values", tuple(_finite_number(value, f"Model 2 {column}", 0.0, 1.0) for column, value in zip(MODEL2_FEATURE_COLUMNS, self.values, strict=True)))

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(MODEL2_FEATURE_COLUMNS, self.values, strict=True))


@dataclass(frozen=True)
class Model2Predictions:
    """Model 2's five BESSI prediction output contract."""

    values: tuple[float, float, float, float, float]
    model_artifact: str

    def __post_init__(self) -> None:
        if len(self.values) != len(MODEL2_TARGET_COLUMNS):
            raise ContractError("Model 2 must return exactly five BESSI predictions.")
        object.__setattr__(self, "values", tuple(_finite_number(value, f"Model 2 {column}", float("-inf"), float("inf")) for column, value in zip(MODEL2_TARGET_COLUMNS, self.values, strict=True)))
        object.__setattr__(self, "model_artifact", _required_text(self.model_artifact, "model_artifact"))

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(MODEL2_TARGET_COLUMNS, self.values, strict=True))


def model1_scores_from_mapping(payload: Mapping[str, object], model_name: str) -> Model1Scores:
    """Parse a locked Model 1 payload without performing any inference."""
    values = []
    for trait in MODEL1_TRAIT_CODES:
        key = f"{trait}_score"
        if key not in payload:
            raise ContractError(f"Model 1 output is missing {key!r}.")
        values.append(payload[key])
    return Model1Scores(tuple(values), model_name=model_name)

