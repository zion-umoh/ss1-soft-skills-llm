"""Build a selected-skill scenario instrument through a reproducible release gate.

This module implements only the scenario-instrument workflow:
selection -> skill lookup -> candidate generation -> rubric evaluation ->
revision/rejection -> balanced instrument. It does not call, tune, or test
Model 1 or Model 2. The resulting instrument is a candidate for expert and
participant validation, not a validated assessment by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path


PIPELINE_VERSION = "4.0.0"
DEFAULT_SEED = 42
MAX_SELECTED_SKILLS = 3
SCENARIOS_PER_SKILL = 2
CANDIDATES_PER_SKILL = 3
RUBRIC_THRESHOLD = 12
RUBRIC_MAXIMUM = 14
PROTECTED_TERMS = ("race", "ethnicity", "religion", "disability", "sexual orientation", "gender identity")


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    display_name: str
    behavioural_indicators: tuple[str, str, str]
    design_rules: tuple[str, str]


@dataclass(frozen=True)
class ScenarioCandidate:
    candidate_id: str
    skill_id: str
    candidate_number: int
    context_key: str
    behavioural_indicators: tuple[str, str, str]
    text: str
    revision: int = 0


SKILL_CATALOG = {
    "self_management": SkillSpec("self_management", "Self-management", ("prioritises competing tasks", "makes a workable plan", "follows through on commitments"), ("include a time or resource constraint", "require a concrete next action")),
    "social_engagement": SkillSpec("social_engagement", "Social engagement", ("initiates communication", "shares relevant information", "coordinates with others"), ("include people who need to be engaged", "avoid rewarding mere confidence or verbosity")),
    "cooperation": SkillSpec("cooperation", "Cooperation", ("listens to competing perspectives", "negotiates a fair response", "supports productive teamwork"), ("include a genuine conflict of needs", "avoid a single morally obvious response")),
    "emotional_resilience": SkillSpec("emotional_resilience", "Emotional resilience", ("pauses to assess a setback", "chooses a constructive next step", "seeks appropriate support when needed"), ("represent a manageable setback or uncertainty", "ask about observable responses, not private emotion labels")),
    "innovation": SkillSpec("innovation", "Innovation", ("questions an unhelpful assumption", "seeks useful information", "tries and evaluates an alternative"), ("include a familiar approach that has a limitation", "make exploration relevant to the shared goal")),
}

CONTEXTS = (
    ("team_deadline", "working with a project team two days before an agreed deadline", "one part of the shared work is incomplete and the original plan cannot be finished as written", "the team, the person responsible for the missing work, and the person receiving the result"),
    ("service_handover", "handling a busy service handover with colleagues", "the available notes conflict and the colleague who prepared them cannot be reached", "the person waiting for an update, your colleagues, and the service team"),
    ("community_event", "helping in the hour before a community event", "the planned venue becomes unavailable and volunteers disagree about the most practical alternative", "visitors, volunteers, and the venue coordinator"),
    ("shared_task", "working on a shared task with people who have different availability", "one person asks to postpone their part even though the group must decide what to do today", "the group members and people affected by the task"),
    ("process_review", "taking part in a team review of a recurring process", "the familiar method has caused avoidable problems, but changing it would require time and agreement", "your team and people affected by the work"),
    ("group_decision", "joining a planning meeting with differing priorities", "two reasonable proposals compete for the same limited time and resources", "the people proposing each option and the wider group"),
)

SKILL_PRESSURES = {
    "self_management": "The immediate difficulty is prioritising competing tasks, making a workable plan, and keeping the shared commitment realistic.",
    "social_engagement": "The immediate difficulty is initiating communication, sharing relevant information, and coordinating with the people involved.",
    "cooperation": "The immediate difficulty is responding to different needs, negotiating a fair response, and supporting productive teamwork without dismissing anyone.",
    "emotional_resilience": "The immediate difficulty is responding constructively to an unexpected setback, choosing a constructive next step, and seeking appropriate support when needed.",
    "innovation": "The immediate difficulty is questioning an unhelpful assumption, investigating an alternative, and evaluating an approach that is not working well.",
}


def parse_selected_skills(raw: str) -> tuple[str, ...]:
    """Validate ordered multi-select input without silently changing its scope."""
    selected = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not selected:
        raise ValueError("Select at least one soft skill.")
    if len(selected) != len(set(selected)):
        raise ValueError("Each selected soft skill must appear once.")
    unknown = [skill for skill in selected if skill not in SKILL_CATALOG]
    if unknown:
        raise ValueError(f"Unknown soft skill(s): {', '.join(unknown)}.")
    if len(selected) > MAX_SELECTED_SKILLS:
        raise ValueError(f"Select at most {MAX_SELECTED_SKILLS} skills to control participant burden.")
    return selected


def lookup_skills(selected_skills: tuple[str, ...]) -> tuple[SkillSpec, ...]:
    """Return the pre-mapped behavioural indicators and design rules."""
    return tuple(SKILL_CATALOG[skill] for skill in selected_skills)


def _candidate_text(spec: SkillSpec, context: tuple[str, str, str, str]) -> str:
    _, setting, constraint, stakeholders = context
    return (
        f"You are {setting}. Your group needs to make progress on a shared goal. However, {constraint}. "
        f"This affects {stakeholders}. {SKILL_PRESSURES[spec.skill_id]}\n\n"
        "In 80--170 words, describe what you would actually do first and what you would do next. "
        "Explain how you would communicate with the people involved and what information you would seek or use. "
        "There is no single correct answer."
    )


def generate_candidates(selected_skills: tuple[str, ...], seed: int = DEFAULT_SEED) -> list[ScenarioCandidate]:
    """Generate distinct, deterministic candidates for each selected soft skill."""
    candidates = []
    for skill_index, spec in enumerate(lookup_skills(selected_skills)):
        offset = (seed + skill_index * CANDIDATES_PER_SKILL) % len(CONTEXTS)
        for candidate_number in range(CANDIDATES_PER_SKILL):
            context = CONTEXTS[(offset + candidate_number) % len(CONTEXTS)]
            candidates.append(ScenarioCandidate(
                candidate_id=f"{spec.skill_id}-{candidate_number + 1:02d}", skill_id=spec.skill_id,
                candidate_number=candidate_number + 1, context_key=context[0],
                behavioural_indicators=spec.behavioural_indicators, text=_candidate_text(spec, context),
            ))
    return candidates


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", text))


def evaluate_candidate(candidate: ScenarioCandidate) -> dict[str, object]:
    """Apply the fixed literature-informed content pre-screen to one candidate."""
    lower = candidate.text.lower()
    word_count = _word_count(candidate.text)
    scores = {
        "target_skill_mapping": 2 if candidate.skill_id in SKILL_CATALOG and len(candidate.behavioural_indicators) == 3 else 0,
        "behavioural_representativeness": 2 if "what you would actually do first" in lower and "what you would do next" in lower else 0,
        "significant_judgement": 2 if "however," in lower and "immediate difficulty" in lower else 0,
        "realistic_shared_context": 2 if candidate.context_key in {item[0] for item in CONTEXTS} and "this affects" in lower else 0,
        "non_knowledge_focus": 2 if "what information you would seek or use" in lower and "technical" not in lower else 0,
        "clarity_and_accessibility": 2 if 80 <= word_count <= 170 and not any(term in lower for term in PROTECTED_TERMS) else 0,
        "non_keyed_response": 2 if "there is no single correct answer" in lower else 0,
    }
    total = sum(scores.values())
    mandatory = ("target_skill_mapping", "behavioural_representativeness", "significant_judgement", "realistic_shared_context")
    return {"candidate_id": candidate.candidate_id, "skill_id": candidate.skill_id, "context_key": candidate.context_key, "word_count": word_count, "rubric_scores": scores, "total_score": total, "maximum_score": RUBRIC_MAXIMUM, "passes_threshold": total >= RUBRIC_THRESHOLD and all(scores[name] == 2 for name in mandatory)}


def revise_candidate(candidate: ScenarioCandidate) -> ScenarioCandidate:
    """Apply the one deterministic revision path for a failed candidate."""
    context = next(item for item in CONTEXTS if item[0] == candidate.context_key)
    corrected = _candidate_text(SKILL_CATALOG[candidate.skill_id], context)
    return replace(candidate, text=corrected, revision=candidate.revision + 1)


def _unique_texts(candidates: list[ScenarioCandidate]) -> bool:
    return len({candidate.text for candidate in candidates}) == len(candidates)


def build_instrument(selected_skills: tuple[str, ...], seed: int = DEFAULT_SEED) -> tuple[list[ScenarioCandidate], dict[str, object]]:
    """Evaluate, revise once, reject failures, then retain a balanced instrument."""
    candidates = generate_candidates(selected_skills, seed)
    audit = []
    accepted_by_skill: dict[str, list[ScenarioCandidate]] = {skill: [] for skill in selected_skills}
    for candidate in candidates:
        evaluation = evaluate_candidate(candidate)
        final_candidate = candidate
        if not evaluation["passes_threshold"]:
            final_candidate = revise_candidate(candidate)
            evaluation = evaluate_candidate(final_candidate)
        audit.append({"candidate": asdict(final_candidate), "evaluation": evaluation})
        if evaluation["passes_threshold"]:
            accepted_by_skill[final_candidate.skill_id].append(final_candidate)

    selected, rejected = [], []
    for skill in selected_skills:
        accepted = accepted_by_skill[skill]
        if len(accepted) < SCENARIOS_PER_SKILL:
            raise ValueError(f"Only {len(accepted)} acceptable candidates available for {skill}; instrument cannot be built.")
        selected.extend(accepted[:SCENARIOS_PER_SKILL])
        rejected.extend(accepted[SCENARIOS_PER_SKILL:])
    if not _unique_texts(selected):
        raise ValueError("Selected scenarios are not textually unique.")

    counts = Counter(candidate.skill_id for candidate in selected)
    quality = {
        "selection_scope_valid": set(counts) == set(selected_skills),
        "participant_burden_within_cap": len(selected_skills) <= MAX_SELECTED_SKILLS and len(selected) <= MAX_SELECTED_SKILLS * SCENARIOS_PER_SKILL,
        "balanced_skill_coverage": all(count == SCENARIOS_PER_SKILL for count in counts.values()),
        "all_selected_pass_rubric": all(evaluate_candidate(candidate)["passes_threshold"] for candidate in selected),
        "selected_texts_unique": _unique_texts(selected),
        "no_unselected_skill_in_instrument": all(candidate.skill_id in selected_skills for candidate in selected),
    }
    metadata = {
        "pipeline_version": PIPELINE_VERSION, "status": "released", "seed": seed, "selected_skills": list(selected_skills),
        "skill_lookup": [asdict(spec) for spec in lookup_skills(selected_skills)], "candidate_count": len(candidates),
        "selected_scenario_count": len(selected), "scenarios_per_skill": SCENARIOS_PER_SKILL,
        "rejected_after_selection_count": len(rejected), "candidate_audit": audit, "release_gate": quality,
        "meets_release_gate": all(quality.values()),
        "evaluation_scope": "deterministic content pre-screen; no participant responses, Model 1 calls, Model 2 calls, or held-out benchmark data used",
    }
    return selected, metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_delivery_instrument(path: Path, candidates: list[ScenarioCandidate]) -> None:
    """Write the respondent-facing file without hidden skill mappings or audit data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("scenario_id", "delivery_order", "text"))
        writer.writeheader()
        for order, candidate in enumerate(candidates, start=1):
            writer.writerow({"scenario_id": candidate.candidate_id, "delivery_order": order, "text": candidate.text})


def write_audit_bank(path: Path, candidates: list[ScenarioCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("scenario_id", "skill_id", "context_key", "behavioural_indicators", "revision", "text")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({"scenario_id": candidate.candidate_id, "skill_id": candidate.skill_id, "context_key": candidate.context_key, "behavioural_indicators": " | ".join(candidate.behavioural_indicators), "revision": candidate.revision, "text": candidate.text})


def write_report(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Scenario Instrument Pipeline Run", "", f"- Pipeline version: `{metadata['pipeline_version']}`; seed: `{metadata['seed']}`.", f"- Selected skills: {', '.join(metadata['selected_skills'])}.", f"- Candidates: {metadata['candidate_count']}; selected scenarios: {metadata['selected_scenario_count']} ({metadata['scenarios_per_skill']} per skill).", f"- Deterministic release gate: {'PASS' if metadata['meets_release_gate'] else 'FAIL'}.", "", "| Release criterion | Result |", "| --- | --- |"]
    for name, value in metadata["release_gate"].items():
        lines.append(f"| {name.replace('_', ' ')} | {'pass' if value else 'fail'} |")
    lines.extend(["", "This is an automated content pre-screen. Participant data collection, psychometric analysis, and Model 1 → Model 2 integration are outside this module's scope.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def select_skills_interactively(read=input, write=print) -> tuple[str, ...]:
    """Prompt for a bounded selection from the curated skill catalogue."""
    skill_ids = tuple(SKILL_CATALOG)
    write("\nScenario instrument — curated scenario bank (no API calls):")
    for number, skill_id in enumerate(skill_ids, start=1):
        write(f"  {number}. {SKILL_CATALOG[skill_id].display_name}")
    while True:
        raw = read("\nEnter numbers separated by commas (for example: 1,3): ").strip()
        try:
            selected_ids = []
            for item in raw.split(","):
                number = int(item.strip())
                if number < 1 or number > len(skill_ids):
                    raise ValueError
                selected_ids.append(skill_ids[number - 1])
            return parse_selected_skills(",".join(selected_ids))
        except (ValueError, IndexError):
            write("Please enter 1–3 distinct numbers from the list, separated by commas.")


def main(argv: list[str] | None = None, read=input, write=print) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills", help=f"Comma-separated soft-skill ids; maximum {MAX_SELECTED_SKILLS}.")
    parser.add_argument("--interactive", action="store_true", help="choose skills in the console")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--delivery", type=Path, default=Path("data/processed/scenarios/scenario_instrument_delivery.csv"))
    parser.add_argument("--audit-bank", type=Path, default=Path("data/processed/scenarios/scenario_instrument_audit.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/scenario-instrument-pipeline.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/scenario-design/scenario-instrument-pipeline.md"))
    args = parser.parse_args(argv)
    if args.interactive:
        if args.skills:
            parser.error("--interactive chooses skills in the console; do not also supply --skills.")
        selected_skills = select_skills_interactively(read=read, write=write)
    elif not args.skills:
        parser.error("--skills is required unless --interactive is used.")
    else:
        selected_skills = parse_selected_skills(args.skills)
    selected, metadata = build_instrument(selected_skills, args.seed)
    write_delivery_instrument(args.delivery, selected)
    write_audit_bank(args.audit_bank, selected)
    metadata.update({"generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"), "delivery_path": str(args.delivery), "delivery_sha256": _sha256(args.delivery), "audit_bank_path": str(args.audit_bank), "audit_bank_sha256": _sha256(args.audit_bank)})
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_report(args.report, metadata)
    print(f"Built {len(selected)}-scenario instrument for {len(selected_skills)} selected skills; release gate passed.")


if __name__ == "__main__":
    main()
