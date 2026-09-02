# TEMP — W4 Model Benchmark To-Do

**Week:** 31 August–6 September 2026  
**Delete this file:** when every item below is complete and the results are recorded in the permanent model reports.

## Model 1 — Essays text → Big Five

- [x] Record the selected 2025 benchmark paper: Piastra & Catellani (2025), *On the emergent capabilities of ChatGPT 4 to estimate personality traits*.
- [x] Document the adaptation: predict the five binary Essays labels, rather than the paper's continuous scores.
- [x] Implement the paper-based text-only benchmark on the fixed Essays splits.
- [x] Save validation predictions, metrics, model settings and a concise benchmark report.
- [x] Implement the improvement: combine text features with the existing 28 GoEmotions probability features.
- [x] Select the better Model 1 version using validation results only.
- [ ] Evaluate the selected version once on the held-out Essays test split.

## Model 2 — Big Five → BESSI soft-skill estimation

- [x] Record the research framing: this is the project's trait-to-skill estimation contribution, not a reproduction of an identical published model.
- [x] Implement the transparent baseline: five Big Five inputs to five BESSI-domain outputs using linear regression.
- [x] Save validation predictions, MAE/RMSE/R², model settings and a concise benchmark report.
- [x] Implement one stronger multi-output model.
- [x] Select the better Model 2 version using validation results only.
- [x] Evaluate the selected version once on the held-out BESSI test split.

## Shared completion rules

- [ ] Use only the prepared fixed splits; do not use audit IDs, demographics or raw questionnaire-item columns as features.
- [ ] Keep model selection separate from the held-out test evaluation.
- [ ] Create a compact benchmark-versus-improved comparison table for each model.
- [ ] Record limitations, especially Model 2's row-level split and non-unique source `Case` field.
- [ ] Move permanent findings into `reports/model-evaluation/`, then delete this temporary file.
