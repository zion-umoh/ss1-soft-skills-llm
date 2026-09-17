# Interview Features → Big Five → BESSI Skills

MSc Advanced R&D dissertation project.

The compact study is closed for model expansion. Start with the [plain-English dissertation handoff](docs/dissertation-handoff.md) and [final evaluation](reports/benchmark/recruitview-final-review.md) for the conclusion, evidence, and remaining submission tasks.

## Repository map

- `src/data/` — dataset preparation and cached feature extraction.
- `src/models/` — Model 1 interview predictor and Model 2 Big Five-to-BESSI mapping.
- `src/evaluation/` — diagnostics, paper comparison, integration and closing analysis.
- `data/metadata/` — provenance, schemas and split/audit records.
- `data/processed/` — reproducible derived tables (raw data remains in `data/raw/`).
- `outputs/benchmark/` — retained predictions, estimates and exploratory gated-fusion checkpoints.
- `reports/benchmark/` — final evidence and benchmark reports.
- `docs/` — ethics, governance, implementation plan and dissertation handoff.
- `outputs/poster/` — final poster PDF and editable PowerPoint.
- `archive/legacy/` — ignored earlier experiments kept only for recovery.

## Research question

Can transcript representations and engineered vocal-delivery features from interview responses predict Big Five personality traits, and can those predicted traits be used to estimate related behavioural, emotional and social skills represented by BESSI?

## Active pipeline

```text
RecruitView interview
  → transcript representation + engineered audio-characteristic columns
  → modern supervised Big Five predictor
  → predicted Big Five profile
  → BFI-2/BESSI mapping model
  → estimated BESSI skill domains
```

Raw audio is not passed directly to the predictors. The audio is processed into documented columns such as speech rate, pause behaviour, pitch and energy statistics, duration and voice-quality summaries.

There is also a separate direct benchmark:

```text
RecruitView features → direct speaking_skills predictor
                  → final comparison with the published RecruitView method
```

The direct target is `speaking_skills`. RecruitView does not contain BESSI labels, so it cannot directly validate the BESSI outputs.

## Data

| Dataset | Purpose |
| --- | --- |
| RecruitView | Interview transcript/audio characteristics and Big Five plus `speaking_skills` labels |
| BFI-2/BESSI | Train and evaluate the Big Five → BESSI mapping |

RecruitView evaluations use participant-disjoint splits. Model 2 uses a row-level split because the source identifier is non-unique; its results are exploratory. The original RecruitView test has already been inspected, so subsequent extensions are labelled exploratory and select settings using development participants only.

## Model plan

Model 1 compares the frozen transformer baseline with an explicit OpenAI LLM-assisted text-feature extractor and engineered audio columns. Text-only, LLM-text-only, engineered-audio-only, simple-concatenation and validation-tuned late-fusion variants are compared. The LLM returns structured observable linguistic and affective features; it is not asked to output Big Five or BESSI labels directly. Late fusion trains the text and audio regressors separately and blends their predictions using validation data only.

The extended audio experiment adds a pretrained WavLM speech embedding. It is extracted from five-second audio chunks with a resumable checkpoint, then compared as a speech-only branch and as a leakage-safe LLM-text + engineered-audio + speech-embedding fusion branch. The waveform itself is never passed to the Ridge predictor.

Model 2 is a separately trained multi-output mapping from the five Big Five traits to five BESSI domains. Its outputs are estimates, not deterministic conversions or psychological diagnoses.

The small gated-fusion follow-up reuses the 10 cached LLM text features and 12 engineered audio features. Separate 16-unit projections feed either fixed or learned weighting and a small shared prediction head. It compares squared error, Huber loss, and Huber plus a training-pair ranking loss over three participant-grouped development folds and three fixed random seeds. Model choice is saved before scoring the historical test. No new API calls are needed.

The development-selected gated model scored 0.4592 Big Five / 0.4551 speaking correlation on the historical test, against 0.4519 / 0.4859 for a matched Ridge refit. Participant-bootstrap intervals do not establish an improvement. The original primary pipeline is retained. See [the follow-up report](reports/benchmark/recruitview-gated-fusion.md). Artifacts and resumable fold checkpoints are saved separately under `outputs/benchmark/recruitview-gated-fusion/`; a changed protocol or input requires a new output directory.

## Implementation order

The complete ordered plan is in [docs/recruitview-bessi-implementation-plan.md](docs/recruitview-bessi-implementation-plan.md). Batch 0 leaves one active RecruitView → Big Five → BESSI pipeline before new modelling begins.

## Useful commands

```bash
hf download AI4A-lab/RecruitView --repo-type dataset --local-dir data/raw/recruitview --include 'videos/*.mp4'
make audit-data
make prepare-recruitview
make extract-recruitview-audio
make extract-recruitview-llm-features
make extract-recruitview-speech-embeddings
make train-model1-feature-fusion
make train-model1-gated-fusion
make close-recruitview-study
make analyse-recruitview-feature-balance
make integrate-recruitview-bessi
make compare-recruitview-paper
make prepare-bfi2-bessi
make train-model2-benchmark
make train-model2-improved
make evaluate-model2-selected
make test
```

## Responsible-use boundary

This is a research prototype. It must not be used for automated hiring, diagnosis, ranking people, or claims that a short interview objectively measures a person’s true soft skills.
