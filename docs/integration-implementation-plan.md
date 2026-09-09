# Full Assessment Pipeline: Staged Implementation Plan

## Purpose

This plan separates the ancillary question-design workflow from the runtime assessment workflow and the external benchmark. The work is intentionally delivered in small batches so each interface can be reviewed and tested before the next layer is added.

The locked Model 1 and Model 2 implementations remain unchanged. New code wraps their existing contracts rather than changing their training or evaluation procedures.

## Target architecture

```text
Ancillary design pipeline
  select skills → generate candidates → rubric → revise/reject → release instrument

Runtime assessment pipeline
  released instrument + responses
    → response validation and neutral aggregation
    → locked Model 1 (response text → Big Five estimates)
    → trait adapter (names, order, scale)
    → locked Model 2 (Big Five → five BESSI predictions)
    → selected-skill report

External benchmark
  same participants complete the 3SQ
    → 3SQ scores and documented crosswalk
    → comparison with our selected-skill predictions
```

## Batch 0 — Freeze scope and contracts

**Status: complete.** Contracts are implemented in `src/integration/schemas.py` and covered by `tests/test_integration_contracts.py`.

### Goal

Write the schemas and invariants before implementing the connection.

### Deliverables

- Instrument input contract: released instrument version, scenario IDs, delivery order, and target-skill metadata.
- Response contract: `assessment_id`, `response_order`, `scenario_id`, and `response_text`.
- Model 1 output contract: five trait estimates; confidence is optional because the locked Luna scorer currently exposes scores only.
- Model 2 input/output contract: five normalized Big Five features and five BESSI predictions.
- Assessment report contract with model versions, instrument version, hashes, and processing status.
- Explicit rule that skill labels and rubric metadata are never passed into Model 1.

### Exit criteria

- Schemas are documented and tested against representative examples.
- The Model 1 trait order and Model 2 feature order are written down explicitly.
- The `0–10` to `0–1` conversion is recorded as an integration assumption.

## Batch 1 — Runtime data validation

**Status: complete.** The validator and neutral response aggregator are implemented in `src/integration/response_validation.py` and covered by `tests/test_response_validation.py`.

### Goal

Build a safe input layer without calling either model.

### Deliverables

- Response loader and validator.
- Checks for missing, duplicate, unknown, or out-of-order scenarios.
- Checks for empty and excessively large response text.
- Neutral response aggregation in delivery order.
- Assessment-level error messages that identify the invalid field without exposing response contents.

### Tests

- Complete response set.
- Missing response.
- Duplicate response.
- Unknown scenario ID.
- Empty response.
- Multiple assessment IDs mixed in one file.

### Exit criteria

- Invalid input fails before any model call.
- Valid input produces one deterministic Model 1 text payload per assessment.

## Batch 2 — Model 1 adapter

**Status: complete (offline adapter).** The request contract and parser are implemented in `src/integration/model1_adapter.py` and covered by `tests/test_model1_adapter.py`; the remote smoke test remains intentionally deferred to Batch 5.

### Goal

Connect validated response text to the locked Model 1 scorer through a narrow adapter.

### Deliverables

- Reusable Model 1 inference interface.
- Response payload construction that excludes question text, skill IDs, rubric scores, and participant metadata.
- Strict parsing and range validation for all five trait scores and confidence values.
- Model 1 version and request metadata in the assessment record.

### Tests

- Adapter tests using a mocked Model 1 response.
- Malformed response handling.
- Missing trait handling.
- Out-of-range score handling.
- Leakage test confirming no target-skill metadata enters the prompt.

### Exit criteria

- The adapter can be tested without a network call.
- The controlled live smoke test is deferred to Batch 5.

## Batch 3 — Model 2 adapter

**Status: complete.** The adapter and artifact checks are implemented in `src/integration/model2_adapter.py` and covered by `tests/test_model2_adapter.py`.

### Goal

Pass Model 1 estimates into the locked Model 2 artifact without changing Model 2 training code.

### Deliverables

- Trait rename and reorder adapter.
- Explicit `0–10` to `0–1` conversion.
- Model 2 artifact loader and compatibility checks.
- Prediction output for all five BESSI domains.
- Selection layer that reports only the skills requested by the released instrument.

### Tests

- Feature order test.
- Scale conversion test.
- Model artifact schema test.
- Prediction shape and finite-value test.
- Selected-skill filtering test.

### Exit criteria

- A fixed Model 1 output produces a reproducible Model 2 input vector.
- Model 2 returns five valid predictions and the report contains the requested subset.

## Batch 4 — End-to-end fixture pipeline

**Status: complete.** The deterministic runner, committed fixtures, report schema, and console target are implemented in `src/integration/fixture_pipeline.py`, `tests/fixture_*`, and `tests/test_fixture_pipeline.py`.

### Goal

Run the complete connection using deterministic test-user responses.

### Deliverables

- Committed fixture response files for one- and three-skill instruments.
- End-to-end runner using mocked Model 1 and the locked Model 2 artifact.
- Assessment report containing inputs, intermediate traits, selected predictions, and provenance.
- Make target or console command for the fixture run.

### Tests

- Complete one-assessment run.
- Complete multi-skill run.
- Failure propagation from each layer.
- Repeated-run structural reproducibility.
- Confirmation that respondent-facing output does not contain hidden skill mappings.

### Exit criteria

- A fixture assessment can be run from the console with one command.
- The generated report is complete and traceable.
- Full existing test suite remains green.

## Batch 5 — Live Model 1 smoke path

**Status: complete.** The controlled runner and fake-client tests are implemented in `src/integration/live_smoke.py` and `tests/test_live_smoke.py`. A confirmed one-request run completed through Model 1 and the locked Model 2 artifact after correcting the output-token budget to 512.

### Goal

Verify the real Model 1 → Model 2 connection separately from the deterministic tests.

### Deliverables

- Explicit confirmation prompt before remote Model 1 inference.
- Request budget and failure handling.
- Saved raw Model 1 response metadata and derived assessment output.
- Clear distinction between smoke-test success and predictive validity.

### Exit criteria

- One approved fixture assessment completes through the live Model 1 and locked Model 2.
- No scenario-generation LLM is introduced.
- A failed remote call does not produce a partial released assessment.

## Batch 6 — 3SQ benchmark pathway

**Status: implemented and contract-tested.** The paired-data loader, construct crosswalk, metrics, bootstrap intervals, and report runner are implemented in `src/integration/benchmark_3sq.py` and covered by `tests/test_benchmark_3sq.py`. Real participant data are still required to produce an empirical benchmark result.

### Goal

Add the independent external outcome comparison after the runtime pipeline is stable.

### Reference method

Rubat du Mérac, Botta and Lupo (2026), *Extending the Validation of the 3SQ to Higher Education: Factor Structure and Reliability in a Large Italian University Sample*.

The 3SQ has ten dimensions: self-confidence, curiosity, resilience, openness, collaboration, empathy, leadership, commitment, autonomy, and problem-solving.

### Deliverables

- 3SQ response schema and scoring implementation.
- Documented crosswalk to our five skills.
- Paired-participant dataset contract.
- Benchmark runner and report.
- Per-skill association, error, agreement, and confidence-interval calculations.

### Crosswalk

- Self-management → commitment and autonomy.
- Cooperation → collaboration.
- Emotional resilience → resilience.
- Innovation → curiosity and problem-solving.
- Social engagement → openness, empathy, and leadership; exploratory only until justified.

### Exit criteria

- Benchmark comparisons use the same participants and a held-out participant-level test set.
- Synthetic response fixtures are not used as benchmark ground truth.
- Results clearly distinguish external agreement from observed-behaviour validity.

## Batch 7 — Final review and release

**Status: review complete; conditional release.** The offline checks and confirmed live smoke pass. A real 3SQ paired-participant dataset is still required before claiming empirical benchmark evidence.

### Goal

Review the complete implementation without changing locked model results.

### Checks

- Full test suite.
- Contract and leakage checks.
- Fixture end-to-end run.
- Live smoke-test audit.
- Benchmark report schema.
- Documentation and command examples.
- Git diff and stale-file review.

### Release criteria

- Every batch exit criterion is met.
- Runtime reports include instrument, model, and assessment provenance.
- Failures are explicit and no incomplete assessment is released.
- Claims are limited to what the available benchmark data support.

## Scope boundary

Question generation remains an ancillary, versioned design pipeline. Runtime integration does not regenerate questions. The 3SQ comparison is an external benchmark study and requires paired participant data; it is not replaced by synthetic test responses.
