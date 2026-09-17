# RecruitView → Big Five → BESSI implementation plan

## Purpose

**Closing status:** Core computational batches are complete with the recorded limitations. Batch 10 now includes repeated grouped development sensitivity, participant-bootstrap intervals, length controls, duration/question checks and error examples. Demographic analysis is unavailable in the prepared RecruitView data. Batch 11 has a methods/results/limitations handoff and poster-ready copy; final institutional formatting and the full manuscript remain. See [dissertation-handoff.md](dissertation-handoff.md). No further model expansion is planned.

This is the implementation order for the revised dissertation design. It keeps the direct benchmark separate from the BESSI mapping and removes the former scenario-based assessment path:

1. **Main dissertation pipeline:** transcript plus engineered audio-characteristic columns → predicted Big Five → estimated BESSI skill domains.
2. **Final direct benchmark:** the same engineered features → `speaking_skills`, compared with the published RecruitView method after our model is frozen.

The RecruitView performance labels are not BESSI labels. The first path is the two-stage BESSI estimation study; the second is a same-dataset direct speaking-skills benchmark.

## Final research question

Can transcript and vocal-delivery features from interview responses predict Big Five personality traits, and can those predicted traits be used to estimate related behavioural, emotional and social skills represented by BESSI?

## Scope decisions to freeze before coding

- RecruitView is the external interview dataset and contains transcript, audio/video and continuous labels for Big Five plus interview-performance dimensions.
- The only direct soft-skill target is `speaking_skills`. Overall performance, confidence, facial expression, interview score and answer score are not dissertation targets.
- The published RecruitView CRMF results are the same-dataset reference point. They are not treated as BESSI validation.
- BESSI mapping is trained and evaluated on the paired BFI-2/BESSI dataset.
- All splits are participant-disjoint. Rows from one participant must never occur in both training and test sets.
- Raw audio is never passed directly into the predictor. Audio is processed into documented numerical characteristics, which become ordinary feature columns.
- The core predictor is a modern supervised feature-fusion model: a pretrained transformer produces transcript representations, engineered audio columns are concatenated, and a supervised regression head predicts the five Big Five traits. A generative LLM is optional as a secondary baseline, not required as the main model.
- Model outputs are estimates for research, not hiring decisions, diagnosis or objective measurement of a person.
- The former scenario-generation, scenario-alignment and 3SQ pathway is out of scope and must not remain in the active implementation.

## Batch 0 — Rebaseline and clean the codebase

**Dependency:** none.

**Goal:** remove the obsolete project architecture before implementing the pivot. This batch is a codebase change, not just documentation.

**Tasks**

- Create a keep-list for the new core: RecruitView preparation and evaluation, RecruitView feature extraction/model code, BFI-2/BESSI preparation and Model 2 mapping, shared metrics, tests for retained code, documentation and reproducibility metadata.
- Remove or archive the former scenario-generation and scenario-alignment code, scenario fixtures, scenario reports, scenario metadata, 3SQ pathway and scenario-only integration adapters/tests.
- Remove or archive obsolete Essays/GoEmotions/Piastra/Luna/prompt-ensemble/local-LLM experiments that are not used by the RecruitView model. Preserve final reports only in a clearly labelled legacy archive if needed for dissertation history.
- Remove or archive obsolete live-smoke, fixture-pipeline and response-validation code if it exists only to serve the discarded scenario runtime.
- Edit `README.md`, `Makefile`, dataset checklists, documentation indexes and test discovery so they describe only the RecruitView → Big Five → BESSI study.
- Edit imports, paths, configuration and requirements exactly to match the retained code; remove dependencies used only by deleted experiments.
- Run an import/reference scan so no active file points to a removed module, scenario path or obsolete output.
- Record the research question, targets, modalities, metrics and participant-level split rule after the cleanup.
- Decide whether the core Model 1 input is transcript-only, audio-only, or transcript + audio. The recommended primary comparison is all three: text-only, audio-only and combined.
- Define the main claim as estimation/prediction, not conversion or direct measurement.
- Record the two evaluation tracks and the fact that RecruitView has no direct BESSI labels.
- Add the RecruitView paper and BESSI papers to the dissertation bibliography.

**Exit criteria**

- The repository has one active pipeline and no dead scenario commands or imports.
- `make test` (or the reduced retained test suite) runs without references to removed modules.
- The README and Makefile expose only commands needed for the revised study.
- A one-page protocol exists before any new model is trained.
- No target, split or metric is changed after looking at final test results.

## Batch 1 — Audit and prepare RecruitView

**Dependency:** Batch 0.

**Existing entry point:** `make prepare-recruitview`.

**Tasks**

- Verify access and record the dataset version, licence and research-use restrictions.
- Preserve the raw metadata and media without modification.
- Build a model-ready table with participant ID, question ID, transcript path, audio path and the five Big Five labels plus `speaking_skills`.
- Check missing media, empty transcripts, duplicate clips and repeated participants.
- Produce a participant-level data summary and a reproducible exclusion report.

**Exit criteria**

- Every usable row has a stable participant ID and target values.
- The prepared table can be recreated from the raw download.
- No generated summaries or target labels are passed into Model 1 inputs.

## Batch 2 — Create leakage-safe splits and engineered feature columns

**Dependency:** Batch 1.

**Tasks**

- Create fixed participant-grouped train, validation and test manifests.
- Keep the final test manifest locked and write its hash to metadata.
- Extract transcript features: a frozen pretrained transformer embedding plus response length and basic lexical statistics.
- Extract audio characteristics into columns: duration, speech rate, pause duration/count, pitch statistics, energy statistics, speaking-time ratio and documented voice-quality summaries.
- Do not feed waveform/audio files directly to Model 1 or the direct `speaking_skills` model.
- Keep feature extraction deterministic and save a feature table with version metadata and column definitions.
- Add a length-only baseline because response length can be a strong confound.

**Exit criteria**

- No participant crosses a split.
- Transcript, engineered-audio and combined feature tables have identical row IDs.
- A second run reproduces the same features and split manifests.

## Batch 3 — Establish internal baselines and model-selection gates

**Dependency:** Batch 2.

**Goal:** create honest reference points before selecting a modern predictor. These are progress checks, not pass/fail claims about human ability.

**Tasks**

- Implement mean-prediction, length-only, transcript-only and engineered-audio-only baselines for Big Five and `speaking_skills`.
- Predefine the primary metric as participant-held-out Spearman correlation, with MAE/RMSE as secondary metrics.
- Use validation results to select the model; do not use the final test set for model choice.
- Treat a baseline failure as a debugging or negative-result signal, not a reason to invent a required score threshold.

**Exit criteria**

- Baseline tables and locked validation rules exist for both Big Five and `speaking_skills`.
- The protocol states that moving forward requires a valid, reproducible evaluation—not a preselected accuracy number.

## Batch 4 — Build and evaluate Model 1: engineered interview features → Big Five

**Dependency:** Batch 3.

**Tasks**

- Train separate transcript-only, engineered-audio-only and simple-concatenation fused Big Five regressors.
- Add a predeclared late-fusion variant: train the text and audio regressors independently, then choose a text/audio prediction blend on validation data only.
- Use a modern pretrained transformer encoder for transcript representations and a supervised regression head over each feature representation.
- Keep the transformer frozen for the first experiment; fine-tuning is an explicitly labelled secondary experiment if compute and time allow.
- Do not use raw audio as a model input and do not make a generative LLM the only evidence for the result.
- Predict all five continuous traits supplied by RecruitView.
- Use participant-grouped validation for model selection and reserve the test set for one final run.
- Select the primary variant using validation Big Five macro-Spearman; report the direct `speaking_skills` result separately.
- Report per-trait and macro Spearman correlation, MAE/RMSE and confidence intervals.
- Run ablations for engineered audio characteristics, transcript representation and response length.

**Exit criteria**

- One locked Model 1 artefact and one final report exist.
- The report identifies which traits are predictable and which are weak.
- The model version and feature schema are recorded for downstream use.

## Batch 4a — OpenAI LLM-assisted text features

**Dependency:** Batch 3 and the prepared RecruitView transcript table.

**Tasks**

- Use the cost-sensitive `gpt-5.6-luna` API model as a frozen, structured text-feature extractor.
- Send one question-plus-transcript request per response; do not send raw audio or video.
- Require a strict JSON schema of observable linguistic and affective features; do not ask the LLM to output Big Five or BESSI labels directly.
- Save every response incrementally to JSONL, maintain a progress manifest, retry transient failures, and resume by `record_id` after interruption.
- Run a small pilot and inspect feature validity before processing the remaining rows.
- Record model ID, prompt version, schema hash, response IDs and token usage for reproducibility and cost accounting.

**Exit criteria**

- A complete, validated LLM feature table exists, or outstanding failures are explicitly reported.
- The extraction can be resumed without reprocessing completed records.
- The LLM feature schema is frozen before downstream model comparison.

## Batch 4b — Pretrained speech-embedding fusion sensitivity

**Dependency:** Batch 4 and the completed engineered-audio table.

**Tasks**

- Extract fixed-size WavLM embeddings from five-second audio chunks with an append-only checkpoint and no raw-waveform output.
- Compare speech-only, LLM-text + speech, and LLM-text + engineered-audio + speech variants using the same participant-disjoint splits.
- Fit scalers and Ridge models on training participants only; select alpha and the primary variant on validation data only.
- Report the speech branch as a controlled representation experiment, not as evidence that the waveform itself is a personality measurement.
- Run the feature-balance and leave-one-block-out diagnostics after selection.

**Exit criteria**

- A complete speech-embedding table and extraction summary exist, or failures are explicitly reported.
- The extended comparison is reproducible from the saved checkpoint and feature table.
- The original `llm_fused` result remains available as the pre-extension baseline.

## Batch 4c — Small gated-fusion follow-up (completed)

**Dependency:** cached Batch 4a text features and Batch 2 audio features.

This extension was proposed after the original test results had been inspected. It is exploratory; the historical test cannot be described as newly untouched. No new media features or API calls are required.

- Reuse 10 structured LLM columns and 12 engineered audio columns.
- Project each block into 16 units, combine through fixed or learned response-dependent weights, and predict the five traits plus speaking skills with a shared small network.
- Compare fixed fusion with squared error, gated fusion with squared error, gated Huber, and gated Huber plus a ranking term. Ranking pairs use training label order, not the original annotators' comparison records.
- Select epoch count and model by mean Big Five correlation over three participant-grouped development folds, averaging three fixed seeds. Fit scalers inside each fold and compare a Ridge reference under the same splits.
- Save configuration, input/source hashes, fold checkpoints, and selection before historical test scoring. Save all model variants and test predictions separately from the original primary artifact.
- Quantify paired score differences with 1,000 participant-bootstrap resamples and verify saved-model predictions after reload.

**Outcome:** gated Huber plus ranking won development selection (Big Five 0.3842 versus Ridge 0.3601). Historical test Big Five was 0.4592 versus 0.4519; speaking was 0.4551 versus 0.4859. Both paired difference intervals include zero. The result does not establish an overall improvement or close the published CRMF gap. Preserve the original primary pipeline and report this as a completed exploratory extension.

**Outputs:** `reports/benchmark/recruitview-gated-fusion.md`, its JSON detail, and `outputs/benchmark/recruitview-gated-fusion/`. Run with `make train-model1-gated-fusion`.

## Batch 5 — Prepare the BFI-2/BESSI mapping dataset

**Dependency:** Batch 0.

**Existing entry point:** `make prepare-bfi2-bessi`.

**Tasks**

- Verify the five BFI-2 input columns and five BESSI domain targets.
- Confirm that the rows represent matched observations and document the available participant identifier.
- Standardise inputs using training data only.
- Check whether repeated `Case` values represent distinct observations; do not make participant-level claims without a valid unique identifier.
- Create participant-disjoint train/validation/test splits where possible.

**Exit criteria**

- A model-ready paired table and split manifest exist.
- The limitation around the current source identifier is documented before final modelling.

## Batch 6 — Train and evaluate Model 2: Big Five → BESSI

**Dependency:** Batch 5.

**Tasks**

- Fit a regularised linear multi-output baseline first.
- Compare one pre-declared alternative, such as Extra Trees, using validation data only.
- Evaluate the selected model on the untouched test set with per-skill and macro R², MAE and RMSE.
- Add bootstrap confidence intervals and a mean-prediction baseline.
- Run coefficient/sign checks against the expected mapping: conscientiousness/self-management, agreeableness/cooperation, extraversion/social engagement, emotional stability/emotional resilience and openness/innovation.
- State that the mapping is empirically informed and probabilistic, not a deterministic conversion.

**Exit criteria**

- A locked Model 2 artefact and report exist.
- Any optimistic row-level split is clearly labelled exploratory; participant-level results are used for the main claim when available.

## Batch 7 — Integrate the complete pipeline

**Dependency:** Batches 4 and 6.

**Tasks**

- Pass Model 1’s five predicted traits into the locked Model 2 input adapter.
- Check trait order, scale, missing-value handling and version compatibility.
- Produce five estimated BESSI domains for each held-out RecruitView response or participant.
- Preserve both intermediate Big Five predictions and final BESSI estimates for auditability.
- Do not compare these BESSI estimates directly with RecruitView performance labels as if they were the same construct.

**Exit criteria**

- One deterministic command produces an end-to-end output table.
- Each output records dataset version, model versions, feature version and split membership.

## Batch 8 — End-to-end validity and proxy analysis

**Dependency:** Batch 7.

**Tasks**

- Test whether estimated BESSI scores are associated with RecruitView interview-performance labels.
- Report this as convergent/proxy evidence only.
- Compare the two-stage estimates with the direct `speaking_skills` analysis from the final benchmark batch only as a construct/proxy analysis.
- Analyse whether the two-stage mapping is associated with interview performance without treating the targets as interchangeable.
- Avoid claiming criterion validity for BESSI without BESSI-labelled RecruitView participants.

**Exit criteria**

- The final report separates direct target validation, BESSI mapping validation and proxy association.
- No result is presented as evidence of hiring suitability.

## Batch 9 — Final locked comparison with published RecruitView work

**Dependency:** Batches 3, 4, 6, 7 and 8.

**Goal:** perform the paper comparison last, after our features, model choice and mapping are frozen.

**Tasks**

- Run the frozen feature-fusion model directly on RecruitView `speaking_skills`.
- Run the frozen feature-fusion model on RecruitView Big Five targets for the second same-dataset comparison.
- Compare both result sets with the published RecruitView CRMF results using the same target names and metrics where possible.
- If the published split or modality differs, label the comparison partial or contextual rather than calling it a reproduction.
- Do not tune our model after seeing the final published-reference comparison.

**Exit criteria**

- The final table contains our result, the published reference and the exact comparison conditions.
- The direct benchmark is described as interview-performance prediction, not direct BESSI validation.

## Batch 10 — Robustness, ethics and final analysis

**Dependency:** Batches 4, 6, 8 and 9.

**Tasks**

- Repeat the main analyses across participant-grouped folds or bootstrap resamples.
- Report uncertainty, missing-data exclusions and sensitivity to response length.
- Check performance by question, response duration and available demographic groups without making unsupported subgroup claims.
- Include error examples and explain failure modes in plain English.
- State limitations: indirect BESSI validation, self/perceived ratings, domain shift, small participant count and possible label bias.
- Add data governance, fairness and responsible-use boundaries.

**Exit criteria**

- The final results table is frozen.
- Every headline number has a script, input manifest and report path.

## Batch 11 — Poster and dissertation package

**Dependency:** Batch 10.

**Poster sections**

1. Problem and research question.
2. RecruitView data and BFI-2/BESSI data.
3. Updated pipeline diagram.
4. Big Five prediction results.
5. Big Five→BESSI results.
6. Final `speaking_skills` benchmark against CRMF.
7. Limitations and ethical boundaries.
8. Main conclusion and future work.

**Final deliverables**

- Reproducible code and environment file.
- Raw-data access and licence record.
- Prepared-data and split manifests.
- Locked Model 1 and Model 2 artefacts.
- Final Big Five and `speaking_skills` benchmark report.
- BESSI mapping report.
- End-to-end proxy report.
- Poster, dissertation methods chapter and limitations section.

## Definition of success

The project is dissertation-level if it demonstrates a reproducible participant-held-out interview benchmark, a separately validated Big Five→BESSI mapping, a clearly labelled end-to-end estimate, and honest limits on what RecruitView can validate. The contribution is the evidence-based pipeline and evaluation design, not a claim that a short interview objectively measures a person’s true soft skills.

## Core references to record

- Gupta, A. K., Sheth, F., Shaikh, H., Kumar, D., Puniya, A., Panwar, D., Chaurasia, S., & Mathur, P. (2025). *RecruitView: A Multimodal Dataset for Predicting Personality and Interview Performance for Human Resources Applications*. arXiv:2512.00450.
- Soto et al. (2022), *An integrative framework for conceptualizing and assessing social, emotional, and behavioral skills: The BESSI*, DOI 10.1037/pspp0000401; data documentation by Sewell et al. (2022), DOI 10.1016/j.dib.2022.107792. The college workbook is OSF 7cktv, version 1, modified 12 August 2021, with an exact matching local SHA-256. Dataset redistribution licensing remains unspecified; see [source verification](bessi-source-verification.md).
