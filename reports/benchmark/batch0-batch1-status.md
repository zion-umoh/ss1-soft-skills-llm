# Batch 0–1 status

This file retains the implementation history. The current closing status is in `docs/dissertation-handoff.md`, with the final checks in `reports/benchmark/recruitview-final-review.md`. Model expansion is finished; final manuscript/poster formatting and FigJam text-fit cleanup remain.

## Completed

- Removed the obsolete scenario, 3SQ, Essays, GoEmotions, legacy LLM and fixture-pipeline implementation branches from the active source tree.
- Kept the raw source directories unchanged for provenance; they are not active model inputs.
- Reduced the active RecruitView target contract to the five Big Five labels plus `speaking_skills`.
- Added participant-safe preparation with a stable 70/15/15 split.
- Added raw JSONL auditing and RecruitView provenance metadata.
- Added deterministic MP4-to-column audio extraction. Raw waveforms are not written to the model table.
- Added a checkpointed OpenAI LLM text-feature extractor using `gpt-5.6-luna`; the complete structured feature table covers all 1,998 usable RecruitView rows.
- Added the feature-fusion Model 1 command for MiniLM text, LLM text features, engineered-audio-only, simple-concatenation and validation-tuned late-fusion comparisons.
- Added a resumable pretrained WavLM speech-embedding branch and evaluated speech-only, LLM+speech and LLM+engineered-audio+speech variants.
- Prepared Model 2 and trained its validation-only linear benchmark and Extra Trees comparison.
- Completed the full audio feature table and the first frozen-transformer comparison.

## Current data counts

- RecruitView metadata rows: 2,011.
- RecruitView usable transcript rows: 1,998.
- RecruitView participants: 331.
- RecruitView split: 1,403 train / 303 validation / 292 test rows.
- Model 2 prepared rows: 313 (219 train / 47 validation / 47 test).
- RecruitView audio feature rows: 1,998; 12 finite columns; no missing media.
- RecruitView speech-embedding rows: 1,998; 768-dimensional WavLM vectors; no unresolved extraction failures.
- LLM text-feature rows: 1,998; 10 bounded 0–5 linguistic/affective columns; 1,132,004 tokens recorded; no unresolved rows.
- Model 1 validation-selected Big Five variant: LLM + audio concatenation (macro Spearman 0.4010 on validation; 0.4518 on test). Audio-only remains strongest on the untouched Big Five test set at 0.4645; the LLM + audio variant is retained because it was selected by validation and materially improves over the MiniLM text baseline.
- Direct `speaking_skills` test Spearman for the selected LLM + audio variant: 0.4859; LLM/audio late fusion is 0.4931 and audio-only is 0.4825.
- The WavLM extension did not improve held-out results: LLM + engineered audio + speech embeddings scored 0.3585 validation / 0.3869 test for Big Five and 0.4419 / 0.3609 for `speaking_skills`; therefore the predeclared LLM + engineered-audio variant remains primary.
- Existing one-time Model 2 final test is preserved: linear mapping selected on validation; test MAE 0.3360, RMSE 0.4055, mean R² 0.5041.
- End-to-end estimates were generated for the 292 held-out RecruitView responses; their association with `speaking_skills` is recorded as proxy evidence only (BESSI has no RecruitView ground truth).

## Controls

- All RecruitView splits are participant-disjoint.
- `overall_performance`, confidence, facial-expression, interview-score and answer-score labels are excluded from the active targets.
- The RecruitView final test split is not used for model selection.
- Model 2's one-time held-out test evaluation is preserved and recorded separately.

## Final evaluation status

- Completed the small gated-fusion follow-up (Batch 4c): fixed/gated fusion, MSE/Huber/ranking ablations, three participant-grouped development folds, three seeds, and paired participant-bootstrap intervals. Development selected gated Huber + ranking. Historical test Big Five rose slightly to 0.4592 against a matched Ridge 0.4519, but speaking fell to 0.4551 against 0.4859; intervals do not establish improvement. Original primary artifacts remain unchanged. See `recruitview-gated-fusion.md`.
- Clarification for follow-up work: the historical test has now been inspected across multiple experiments and diagnostics. New results on it are exploratory, even when tuning is confined to development data. The earlier "read once" statements describe the original selection intention and must not be extended to the entire research history.

- The LLM-text and late-fusion refinements were evaluated using validation-only selection; the final test was read once after selection.
- Model 2 was integrated with the held-out RecruitView predictions and the published RecruitView comparison was regenerated.
- LLM + audio improves the MiniLM text baseline and direct `speaking_skills`; audio-only remains the strongest Big Five test variant, so no further test-set tuning is justified.
- Feature-balance diagnostics show timing features have the largest permutation importance, while LLM cognition/communication features provide additional validation signal. The exact pause-fraction/speaking-time complement is documented as a redundancy, not silently tuned away.
