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

- [ ] Verify access terms and dataset version.
- [ ] Record the Hugging Face commit hash and download the MP4 media into `data/raw/recruitview/videos/`.
- [ ] Validate transcript, audio path, participant ID, question ID and target columns.
- [ ] Retain the five Big Five labels and `speaking_skills` only for the active study.
- [ ] Audit missing media, empty transcripts, duplicate records and repeated participants.
- [ ] Create participant-grouped train/validation/test manifests.
- [ ] Create frozen transcript representations.
- [ ] Extract audio characteristics into documented feature columns; do not pass raw audio to the predictor.
- [ ] Save feature tables, column definitions and split metadata.

## BFI-2/BESSI

**Purpose:** train and evaluate the Big Five → BESSI mapping.

- [x] Identify the five Big Five input columns and five BESSI domain targets.
- [ ] Verify linkage, duplicate rows and the available participant identifier.
- [ ] Document score direction, ranges and missing-data handling.
- [ ] Create participant-disjoint splits where the source identifier permits it.
- [ ] Save model-ready tables and distributions before training.

## Evaluation controls

- [ ] Establish mean and response-length baselines.
- [ ] Select models only on training/validation data.
- [ ] Keep the final test set locked.
- [ ] Report per-target and macro Spearman, MAE/RMSE and uncertainty intervals.
- [ ] Run length-only, text-only, engineered-audio-only, simple-concatenation and validation-tuned late-fusion ablations.
- [ ] Run the published RecruitView comparison last, after our model is frozen.
