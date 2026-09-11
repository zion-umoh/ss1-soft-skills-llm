"""Shared contract for the RecruitView Model 1 experiments.

The actual feature extraction and model-selection commands are added after the
data audit.  Keeping the target and feature contract here prevents the old
multi-outcome experiments from reappearing in later batches.
"""

from __future__ import annotations

from dataclasses import dataclass


TRAIT_COLUMNS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)
SPEAKING_SKILLS_COLUMN = "speaking_skills"
DIRECT_TARGET_COLUMNS = (*TRAIT_COLUMNS, SPEAKING_SKILLS_COLUMN)


@dataclass(frozen=True)
class Model1Protocol:
    """Frozen study choices shared by feature extraction and evaluation."""

    primary_metric: str = "participant-held-out Spearman correlation"
    transcript_representation: str = "frozen pretrained transformer embedding"
    audio_representation: str = "engineered vocal-delivery columns"
    raw_audio_input: bool = False
    participant_grouped_splits: bool = True


PROTOCOL = Model1Protocol()
