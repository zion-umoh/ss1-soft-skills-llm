# Dataset Preparation Checklist

This is the working checklist for preparing the datasets used by the SS1 research pipeline. It is a planning reference and should be updated when preprocessing decisions are finalised.

## Preparation rules for every dataset

- [x] Keep files in `data/raw/` unchanged.
- [ ] Record source, licence, download date, version, schema, row count and checksum.
- [x] Document the random seed and split policy.
- [x] Retain author and record identifiers for audit only; do not use them as model features.
- [x] Keep a reproducible script or notebook for every transformation.
- [x] Produce a concise data-quality report before modelling.

## 1. Essays dataset — start here

**Purpose:** the primary Model 1 dataset: text paired with binary Big Five labels.

Expected columns: `#AUTHID`, `TEXT`, `cEXT`, `cNEU`, `cAGR`, `cCON`, `cOPN`.

- [x] Confirm the expected schema and validate that each trait label is binary.
- [x] Parse the CSV safely and preserve the original text as `text_raw`.
- [x] Create a conservative model-input text field, `text_model`; do not overwrite raw text.
- [x] Audit missing text, duplicate records, duplicate authors, extremely short responses, malformed characters and encoding issues.
- [x] Summarise text length, trait prevalence and relationships among the five labels.
- [x] Create fixed train, validation and test splits, grouped by `#AUTHID` and balanced across trait labels where feasible.
- [x] Save model-ready split files with an anonymous row ID, `text_model`, the five labels and `split`.
- [x] Save an Essays data-quality report.

**Deliverable:** `data/processed/model1/essays_{train,validation,test}` and an Essays quality report.

## 2. GoEmotions dataset — prepare next

**Purpose:** train or support the emotion-feature layer for Model 1. It is not the personality target dataset.

The downloaded shards contain text, metadata and 27 emotion labels plus `neutral`.

- [x] Combine `goemotions_1.csv`, `goemotions_2.csv` and `goemotions_3.csv` after confirming their schemas match.
- [x] Preserve source-shard information for auditing.
- [x] Retain text, comment ID, annotation/ambiguity fields and emotion labels for preparation.
- [x] Exclude author, subreddit, link, parent and timestamp fields from model input features.
- [x] Audit duplicate comment IDs, repeated annotations, `example_very_unclear`, label prevalence and class imbalance.
- [x] Decide and document how repeated annotations become one example-level multi-label target.
- [x] Create fixed train, validation and test splits by comment or thread, avoiding duplicate text across splits.
- [x] Train the emotion model using text only; calibrate thresholds on the validation set.
- [x] Save predicted emotion probabilities as features for the Essays Big Five model.

**Training note:** the 28 labels are substantially imbalanced. In particular, grief, pride and relief are rare. Use an appropriate imbalance-aware multi-label loss or class weighting, inspect per-label validation metrics, and choose prediction thresholds on validation data only—not the held-out test set.

**Important:** GoEmotions labels emotions, not emotion regulation. Any emotion-regulation feature must have its own documented theoretical basis and operational definition.

## 3. Big Five and BESSI dataset — Model 2

**Purpose:** train the multi-output model that maps Big Five inputs to selected BESSI skill outcomes.

- [x] Define the selected BESSI skills before preparation: the five supplied BESSI domain scales.
- [x] Document score direction, scale ranges, missing-data rules and any reverse scoring.
- [ ] Check participant linkage and ensure Big Five inputs and BESSI targets are correctly matched. `Case` is reused across distinct rows, so a unique participant ID is still required for participant-level claims.
- [x] Create fixed, documented train, validation and test splits (row-level pending a unique participant ID).
- [x] Produce a model-ready table with Big Five inputs, selected BESSI targets and `split`.

## 4. Integrated assessment pipeline

- [ ] Use the validated scenario instrument to collect a scenario text response.
- [ ] Apply the same locked feature-extraction process used by Model 1.
- [ ] Infer the Big Five profile.
- [ ] Pass the inferred profile to the Big Five-to-BESSI model.
- [ ] Report predicted skills together with limitations and appropriate-use boundaries.

## Evaluation checklist

- [ ] Keep model selection separate from final held-out evaluation.
- [ ] Establish baseline models for both stages.
- [ ] Run ablation tests, particularly for semantic, emotion and regulation features.
- [ ] Report stage-specific results as well as end-to-end findings.
- [ ] Check for leakage at every data transformation and split boundary.
