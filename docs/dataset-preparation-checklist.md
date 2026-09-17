# Dataset preparation checklist

## Rules for every dataset

- Keep `data/raw/` unchanged.
- Record source, licence, version, download date, schema, row count and checksum.
- Keep identifiers for auditing and participant grouping only; never use them as model features.
- Use versioned scripts for every transformation.
- Freeze participant-disjoint splits before model comparison.
- Save a data-quality report before modelling.

## RecruitView

**Purpose:** interview feature extraction, Big Five prediction and the direct `speaking_skills` benchmark.

- [x] Verify access terms and dataset version.
- [x] Record the Hugging Face commit hash and download the MP4 media into `data/raw/recruitview/videos/`.
- [x] Validate transcript, audio path, participant ID, question ID and target columns.
- [x] Retain the five Big Five labels and `speaking_skills` only for the active study.
- [x] Audit missing media, empty transcripts, duplicate records and repeated participants.
- [x] Create participant-grouped train/validation/test manifests.
- [x] Create frozen transcript representations.
- [x] Extract audio characteristics into documented feature columns; do not pass raw audio to the predictor.
- [x] Save feature tables, column definitions and split metadata.

## BFI-2/BESSI

**Purpose:** train and evaluate the Big Five → BESSI mapping.

- [x] Identify the five Big Five input columns and five BESSI domain targets.
- [x] Verify linkage, duplicate rows and the available participant identifier.
- [x] Document score direction, ranges and missing-data handling.
- [x] Assess participant-disjoint splitting; the non-unique `Case` field means the final Model 2 result is explicitly row-level and exploratory.
- [x] Save model-ready tables and distributions before training.

## Evaluation controls

- [x] Establish mean and response-length baselines.
- [x] Select the original model only on training/validation data.
- [x] Keep the original test split fixed; label later checks on that historical test as exploratory.
- [x] Report per-target and macro Spearman, MAE/RMSE and uncertainty intervals.
- [x] Run length-only, text-only, engineered-audio-only, simple-concatenation and validation-tuned late-fusion ablations.
- [x] Run the published RecruitView comparison after the primary model was frozen.
