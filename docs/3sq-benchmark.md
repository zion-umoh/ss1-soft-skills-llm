# 3SQ benchmark pathway

The benchmark compares this project's five selected soft-skill outputs with the
independent Soft Skills Self-evaluation Questionnaire (3SQ) described by
Rubat du Mérac, Botta and Lupo (2026): [paper](https://doi.org/10.7358/ecps-2026-033-mera).
The paper validates a ten-dimension, 41-item, five-point self-report measure in
higher education. It is an external self-report comparator, not an observed
behaviour gold standard.

## Paired-data contract

Provide a participant-level CSV with one row per participant in the held-out
`test` split. The exact columns are:

```text
participant_id,split,
model_self_management,model_social_engagement,model_cooperation,
model_emotional_resilience,model_innovation,
three_sq_self_confidence,three_sq_curiosity,three_sq_resilience,
three_sq_openness,three_sq_collaboration,three_sq_empathy,
three_sq_leadership,three_sq_commitment,three_sq_autonomy,
three_sq_problem_solving
```

All model and 3SQ values must be participant-level scores on the same 1--5
scale. IDs must be unique, and the runner refuses rows outside the requested
holdout split.

## Crosswalk

The comparison is predeclared at the construct level:

| Project skill | 3SQ dimensions | Status |
| --- | --- | --- |
| Self-management | Commitment, autonomy | Predeclared |
| Social engagement | Openness, empathy, leadership | Exploratory |
| Cooperation | Collaboration | Predeclared |
| Emotional resilience | Resilience | Predeclared |
| Innovation | Curiosity, problem-solving | Predeclared |

Mapped dimensions are averaged within participant before comparison. The
runner reports Pearson and Spearman association, MAE/RMSE and bias, concordance
correlation, and deterministic-seed 95% bootstrap intervals.

## Run

```bash
.venv/bin/python -m src.integration.benchmark_3sq \
  --paired-data path/to/paired_3sq_test.csv \
  --output outputs/benchmark/3sq.json \
  --report reports/benchmark/3sq.md
```

Synthetic fixture responses are deliberately excluded from this benchmark
pathway. They can test code contracts, but they are not benchmark ground truth.

The selected 2026 paper provides aggregate psychometric results rather than a
public participant-level paired dataset. Until the same participants' 3SQ
responses are available, the runner can validate the benchmark contract but
cannot produce a defensible empirical comparison score.
