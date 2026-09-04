# Model 1 Benchmarking Record

## Purpose

Model 1 predicts five binary Big Five labels from Essays text: extraversion (`cEXT`), neuroticism (`cNEU`), agreeableness (`cAGR`), conscientiousness (`cCON`), and openness (`cOPN`). This record documents the relevant benchmarking decisions, experiments, and final interpretation for Week 4.

The selection metric was **macro AUROC**: it measures whether a method ranks high-labelled essays above low-labelled essays across all five traits without relying on a particular binary threshold. Macro F1 and balanced accuracy are secondary operational metrics after applying the predeclared score threshold of 5.0/10.0.

## Data and evaluation protocol

| Split | Essays | Use |
| --- | ---: | --- |
| Train | 1,727 | Retrieval references and any supervised baselines |
| Validation | 370 | Development comparisons and model selection |
| Test | 370 | One locked final comparison |

- Retrieval always searched training essays only. Validation or test labels were never used to retrieve examples or included in an API prompt.
- Scores were continuous 0--10 values, rounded to two decimals before AUROC computation so equal scores remain genuine ties.
- Remote calls used the OpenAI Responses API with `store=false`.
- The original Batch route was abandoned after repeated file-access failures. Normal checkpointed Responses API calls were used instead.

## Benchmark reference and adaptation

The reference paper is Piastra and Catellani (2025), *On the emergent capabilities of ChatGPT 4 to estimate personality traits*, [Frontiers in Artificial Intelligence](https://doi.org/10.3389/frai.2025.1484260).

This project is a **paper-inspired operational benchmark**, not an exact replication:

- The paper assesses continuous questionnaire trait scores using GPT-4-era models and repeated zero-shot assessments.
- Essays supplies cohort-derived binary labels. The paper itself notes that such labels are difficult to reproduce strictly zero-shot because the model cannot know the cohort standardisation.
- The project benchmark therefore used independent text-only zero-shot `gpt-4.1` assessments, continuous 0--10 scores, and a fixed 5.0 threshold for binary secondary metrics.

This distinction must be retained in any report: results compare against the project’s adapted Piastra-style baseline, not against the paper’s published numerical results.

## Relevant development results

All values below are validation macro AUROC unless stated otherwise.

| Method | Design decision tested | AUROC | Outcome |
| --- | --- | ---: | --- |
| Piastra-style baseline | Independent, text-only zero-shot assessment | 0.6312 | Reference benchmark |
| TF-IDF + GoEmotions | Add 28 emotion probabilities to a supervised text classifier | 0.5899 | Did not help |
| MiniLM embeddings + GoEmotions | Semantic embedding alternative | 0.5828 | Did not help |
| GPT-5 mini prompt ensemble | Three zero-shot prompts, no training | about 0.629 | Close, but did not beat baseline reproducibly |
| Luna trait specialists | Five separate trait calls | 0.6226 | Worse and unnecessarily call-heavy |
| Luna joint zero-shot | One call, all traits, continuous scores, no retrieval | 0.6313 | Effectively tied baseline |
| Luna semantic retrieval, no reasoning | Three nearest labelled training essays | 0.6293 | Better binary calibration, not ranking |
| **Luna semantic retrieval, medium reasoning** | Three nearest labelled training essays; one joint call | **0.6351** | Selected before test |
| Luna trait-balanced retrieval, medium reasoning | Nearest high and low labelled anchors per trait; 3.89 examples on average | 0.6398 | Exploratory only; run after test was opened |

### What these comparisons established

1. More calls were not inherently better: one joint all-trait call was more economical and at least as effective as five specialist calls.
2. Continuous scoring was preferable to direct binary output. It preserves ranking information for AUROC; a fixed threshold can still produce binary predictions later.
3. Generic semantic retrieval improved the fixed-cutoff metrics. Increasing Luna reasoning to `medium` produced the first clear validation AUROC point lead.
4. Trait-balanced retrieval provided a plausible further improvement because it gives the model comparable high and low anchors for each trait, rather than merely topically similar essays. Its validation AUROC lead over the Piastra-style baseline was +0.0086, but its bootstrap interval still crossed zero (approximately -0.0068 to +0.0245).

## Locked final model and one-time test evaluation

The model selected before opening test labels was:

- `gpt-5.6-luna`
- one joint call returning five continuous 0--10 scores
- `reasoning.effort="medium"`
- three MiniLM semantic-nearest labelled training examples
- no model-parameter training

It was compared once with a newly scored Piastra-style `gpt-4.1` baseline on the same 370 test essays.

| Test metric | Locked Luna retrieval | Piastra-style baseline | Difference |
| --- | ---: | ---: | ---: |
| Macro AUROC | **0.6559** | 0.6542 | +0.0017 |
| Macro F1 | **0.6606** | 0.6106 | +0.0500 |
| Macro balanced accuracy | **0.5948** | 0.5596 | +0.0352 |
| Exact-match accuracy | **0.0649** | 0.0432 | +0.0217 |

### Final interpretation

- The locked Luna method **substantially improves fixed binary classification utility** (F1 and balanced accuracy).
- Its macro-AUROC point estimate is slightly higher, but the paired 1,000-resample bootstrap interval for the AUROC difference crosses zero (approximately -0.0166 to +0.0186). Therefore it should be described as **matching, not conclusively outperforming**, the adapted Piastra-style benchmark on ranking.
- The trait-balanced retrieval result must not replace this final result: it was developed after the test split had been examined. It is a documented exploratory finding and a future-work direction, not a valid new test winner.
- No further Model 1 optimisation should use these validation/test splits. A future improvement study needs a new independent holdout set or a predeclared repeated cross-validation design.

## Cost-relevant record

| Completed Luna run | Actual API cost (USD) |
| --- | ---: |
| Five trait specialists | 0.3703 |
| Joint zero-shot | 0.0960 |
| Semantic retrieval, no reasoning | 0.3076 |
| Semantic retrieval, medium reasoning — validation | 0.3771 |
| Semantic retrieval, medium reasoning — test | 0.3790 |
| Trait-balanced retrieval, medium reasoning — exploratory validation | 0.4368 |

The successful approaches used normal Responses API calls with checkpoint files, avoiding repeated charges after interruptions.

## Evidence artefacts

- [Piastra-style validation report](../reports/model-evaluation/model1-piastra-benchmark-report.md)
- [Locked Luna validation report](../reports/model-evaluation/model1-luna-retrieval-medium-report.md)
- [Locked Luna test report](../reports/model-evaluation/model1-luna-retrieval-medium-test-report.md)
- [Piastra-style test report](../reports/model-evaluation/model1-piastra-test-benchmark-report.md)
- [Trait-balanced exploratory validation report](../reports/model-evaluation/model1-luna-trait-balanced-validation-report.md)
- [Locked test metadata](../data/metadata/model1-luna-retrieval-medium-test.json)
- [Trait-balanced validation metadata](../data/metadata/model1-luna-trait-balanced-validation.json)

## Report-ready conclusion

On the Essays binary-label task, retrieval-augmented Luna scoring with medium reasoning improved threshold-based classification metrics over a paper-inspired zero-shot GPT-4.1 baseline, while producing statistically inconclusive but slightly higher macro AUROC. The result supports retrieval-augmented LLM assessment as a useful practical improvement, but does not justify a strong claim that it reliably surpasses the adapted zero-shot benchmark on ranking.
