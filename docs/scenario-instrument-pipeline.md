# Scenario Instrument Pipeline

## Workflow

```text
User selects 1–3 soft skills
  → system looks up mapped behavioural indicators and design rules
  → curated scenario bank creates three candidates per skill
  → literature-based rubric evaluates each candidate
  → weak candidates receive one deterministic revision; remaining failures are rejected
  → two passing, unique scenarios per skill are combined into the instrument
```

Each scenario has one target skill. The audit CSV retains the selected scenarios' skill mapping, behavioural indicators, context, and revision number; the metadata JSON retains every candidate's rubric result and the release decision. The respondent-facing delivery file contains only scenario text and delivery order. `emotional_resilience` remains a selectable BESSI outcome, but there is no separate emotion-design dimension and no request for private emotion labels.

## Why the curated path is production

The live LLM generator was tested in four controlled runs. Its wording varied enough that a deterministic literature rubric could not reliably distinguish acceptable paraphrases from weak candidates without either false rejections or an additional evaluator model. Because scenario generation is not the important model stage, the live layer was removed from the production path.

The curated bank is deterministic, reproducible, inspectable, and has no API cost. It keeps the full pipeline mechanics in place: user selection, specification lookup, candidate generation, rubric evaluation, revision/rejection, burden capping, balanced selection, respondent/audit separation, and release gating. This is an instrument-construction pre-screen, not psychometric validation.

## Release gate

Each candidate receives 0–2 on seven criteria: target-skill mapping, behavioural representativeness, significant judgement, realistic shared context, non-knowledge focus, clarity/accessibility, and non-keyed response. Release requires at least 12/14 and full marks on the first four mandatory criteria. The instrument must also contain exactly two unique passing scenarios for each selected skill, stay within the three-skill/two-scenario burden cap, and contain no protected-characteristic cue.

The rubric is a design-time content pre-screen informed by situational-judgement-test construction literature. Empirical validation is outside this module's scope; this pipeline makes no psychometric validity claim.

## Console use

From the repository root:

```bash
make scenarios
```

The console displays the skill catalogue, accepts a numeric multi-select, and builds the instrument locally. It makes no API calls and needs no API key.

For a fixed three-skill demo:

```bash
make generate-scenarios-demo
```

Demo outputs are prefixed with `demo_` and do not overwrite live-named audit files. The respondent-facing file is `data/processed/scenarios/demo_scenario_instrument_delivery.csv`; the audit and metadata files contain the release evidence.

## Scope boundary

This module ends at automated content pre-screening and release. It does not collect participant responses, perform psychometric analysis, or connect scenario responses to the locked Model 1 and Model 2 stages.

## Planned external benchmark

The primary external benchmark candidate is Rubat du Mérac, Botta and Lupo (2026), *Extending the Validation of the 3SQ to Higher Education: Factor Structure and Reliability in a Large Italian University Sample* ([paper and full text](https://www.ledonline.it/index.php/ECPS-Journal/article/view/7879), [DOI](https://doi.org/10.7358/ecps-2026-033-mera)). The Soft Skills Self-evaluation Questionnaire (3SQ) uses 41 five-point Likert items and reports ten soft-skill dimensions: self-confidence, curiosity, resilience, openness, collaboration, empathy, leadership, commitment, autonomy, and problem-solving. The study collected data from 2,130 university students in 2024–2025 and tested the established ten-factor structure with confirmatory factor analysis.

This is an external outcome benchmark, not a requirement that the comparator use the same input or algorithm. The planned comparison is within-participant: the same participants complete the released scenario instrument and the 3SQ, then our predicted scores are compared with the mapped 3SQ scores. The initial crosswalk is self-management → commitment/autonomy, cooperation → collaboration, emotional resilience → resilience, innovation → curiosity/problem-solving, and social engagement → openness/empathy/leadership. The last mapping is exploratory and must not be treated as a direct scale equivalence without justification.

The comparison will use per-skill rank/linear association, error metrics, agreement estimates, and confidence intervals. Synthetic response fixtures may test the software connection but cannot serve as benchmark ground truth. Because the 3SQ measures self-perceived skills, agreement with it is external criterion/convergent evidence rather than direct observation of behaviour; language, population, and administration differences must be reported.
