# Personality Traits and Emotion Recognition for Soft-Skill Prediction Using LLMs

MSc Advanced R&D Project — **SS1**

## Overview

This project investigates whether a written response to a scenario can be used to infer a person's Big Five personality traits and, subsequently, predict selected behavioural, emotional and social skills.

The project uses only existing public or anonymised secondary datasets. It does not recruit participants or collect new personal data.

## Research pipeline

1. **Scenario design** — generate and evaluate scenario-based questions that elicit behavioural and emotional information.
2. **Model 1: text to Big Five** — predict Big Five personality traits from written text using semantic, emotion and emotion-regulation features.
3. **Model 2: Big Five to BESSI skills** — predict selected BESSI soft-skill outcomes from Big Five personality inputs.
4. **Integration and evaluation** — connect the two models and report performance, limitations and appropriate use boundaries.

## Datasets

| Dataset | Purpose | Pipeline stage |
| --- | --- | --- |
| Essays | Written text with Big Five labels | Model 1 primary dataset |
| GoEmotions | Text labelled with 27 emotions plus neutral | Model 1 emotion-recognition support |
| BFI-2 | Big Five questionnaire responses | Model 2 input |
| BESSI | Behavioural, emotional and social skill measures | Model 2 target |
| Measure What Matters | Situational-judgement-test reference material | Scenario design and evaluation |

RecruitView is an optional external benchmark only if access is granted; it is not part of the core pipeline.

## Model 1: current scope

**Goal:** build an emotion-aware Big Five prediction model from written text.

**Inputs**

- Essay text
- Linguistic/semantic features
- Emotion and emotion-regulation features

**Outputs**

- Predictions for Openness, Conscientiousness, Extraversion, Agreeableness and Neuroticism

**Evaluation principles**

- Fixed, documented train/validation/test splits
- No data leakage between splits
- Reproducible preprocessing and random seeds
- Baseline models and ablation tests
- Held-out evaluation with stage-specific reporting

## Repository structure

```text
data/          Raw, interim and processed datasets
notebooks/     Exploratory analysis and experiments
src/           Reusable data, feature, model and evaluation code
outputs/       Figures, tables and saved models
docs/          Methodology and scenario-design materials
tests/         Validation tests for reusable code
```

Raw datasets are retained unchanged in `data/raw/`. Cleaned and model-ready files are written to `data/processed/`.

## Status

Week 4 benchmarking is complete for both model stages. Model 1's locked Luna retrieval method matched the adapted Piastra-style zero-shot benchmark on test AUROC while improving fixed-threshold F1; Model 2 selected the linear benchmark over Extra Trees and has one-time held-out test results. See [Model 1 comparison](reports/model-evaluation/model1-comparison-report.md), [Model 2 comparison](reports/model-evaluation/model2-comparison-report.md), and the detailed [Model 1 benchmarking record](docs/model1-benchmarking-record.md).

## Data preparation commands

Run these from the repository root:

```bash
# Update the inventory of immutable raw CSV files.
make audit-data

# Monitor raw CSV files while working; the inventory refreshes after changes.
make watch-data

# Prepare the Essays audit artefacts and leakage-safe Model 1 splits.
make prepare-essays

# Prepare agreement-filtered GoEmotions artefacts and leakage-safe Model 1 splits.
make prepare-goemotions

# Create the project environment once before training models.
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Train the text-only GoEmotions model and write Essays emotion features.
make train-goemotions

# Run preparation-pipeline tests.
make test
```

Generated data-quality reports and derived datasets are reproducible local artefacts; the source files and scripts remain in the repository.

## Expected deliverables

- Validated scenario-design module
- Model 1 personality-prediction results
- Model 2 soft-skill-prediction results
- Integrated pipeline and final evaluation
- Dissertation, code and results package
