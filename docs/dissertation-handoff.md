# Dissertation handoff: final compact study

Existing dissertation title: **Personality Traits and Emotion Recognition to Predict Engineering Students’ Soft Skills using applications of LLMs**.

## The project in plain English

The project tests whether interview wording and measurable vocal delivery can predict personality ratings and speaking skills. An LLM extracts ten structured text features, including emotional-expression cues. Twelve audio measurements describe delivery. A small trained regression model combines those features. A separate model maps predicted personality traits to estimated BESSI skill domains.

The final system uses transcript and audio characteristics, with no visual features. The evidence supports a compact exploratory research prototype. It does not establish a person's true personality or directly validate their BESSI skills.

## Methods copy

RecruitView supplied 2,011 interview records. After transcript exclusions, 1,998 responses from 331 participants were retained. The fixed participant-disjoint split contained 1,403 training, 303 validation and 292 test responses. All responses from a participant stayed in one split. The test portion contained 49 participants.

The cached LLM extraction produced ten linguistic and affective features per response. The predictor also received twelve numerical audio characteristics covering timing, pitch, energy and spectral properties. Raw video imagery was not used. Feature scaling used training data, and the original model was chosen using validation Big Five correlation. The selected Ridge variant combined LLM and audio features. Speaking skills was a separate supervised output, rather than a BESSI label.

Comparisons included length-only, transcript embeddings, audio-only, LLM-only, combined features and late fusion. Additional WavLM speech embeddings did not improve results. A small gated neural model with robust and ranking losses also failed to establish an overall improvement, so the original primary model was retained.

Model 2 used the paired BFI-2/BESSI college-student workbook. Of 322 source rows, 313 were prepared (219 train, 47 validation, 47 test). Linear regression outperformed Extra Trees on validation and was evaluated on the saved test split. Its source identifier was non-unique; report its split as row-level and exploratory. The source workbook is now verified against the authors' OSF file by checksum.

For integration, RecruitView predictions are aligned to the training distribution of BFI-2 before applying Model 2. This statistical scale adjustment is an assumption; it does not establish equivalence between interview impressions and questionnaire measurements.

## Main results

| Interview predictor | Big Five mean Spearman | Speaking Spearman |
| --- | ---: | ---: |
| Length-only baseline | 0.4284 | 0.4685 |
| Engineered audio only | 0.4645 | 0.4825 |
| Selected LLM + audio | 0.4518 | 0.4859 |
| Gated Huber + ranking follow-up | 0.4592 | 0.4551 |
| Published RecruitView CRMF reference | 0.5233 | 0.5947 |

The paper's Big Five mean is calculated from its five published trait scores, not its 12-target overall average. The paper uses text, audio and video with a different split and training protocol. Its numbers provide context, not a controlled superiority test. The gated model is an exploratory follow-up, not a newly untouched test result.

Model 2's saved test mean R² was **0.5041**, MAE **0.3360**, and RMSE **0.4055**. Its row-bootstrap 95% interval for mean R² was **0.3197–0.6028**, conditional on the unverified independence of rows. These statistics do not describe the accuracy of the complete interview-to-BESSI pipeline.

The integrated pipeline generated BESSI estimates for 292 RecruitView test responses. Their correlations with speaking ratings were about **0.45–0.48**. These are exploratory associations between related constructs; RecruitView has no BESSI ground truth.

## What the final checks add

The final review reproduced the saved primary predictions and compared them with the length and audio baselines. It used 1,000 paired bootstrap resamples of whole test participants, plus five development folds repeated across three fixed seeds. The development analysis is a sensitivity check using already selected settings, not fresh nested validation.

Across those development folds, the primary model averaged **0.3652** for Big Five versus **0.3476** for length-only; speaking averaged **0.4316** versus **0.4160**. The mean differences favoured the richer features. However, on the historical test, the primary-minus-length 95% intervals included zero: **−0.0189 to +0.0667** for Big Five and **−0.0365 to +0.0810** for speaking. An overall improvement over response length is therefore not established.

After descriptive adjustment for word count and duration, remaining correlations were approximately **0.11–0.30**, with speaking at **0.20**. This suggests some additional association while emphasising the importance of response length. It does not identify causal effects.

Performance varied across duration and question groups. Only two question groups met the reporting threshold of ten participants, so question-level conclusions are limited. The largest recorded errors involved extreme ratings while predictions stayed nearer the middle of the scale. That describes a failure mode; it does not show that the human labels are wrong. No demographic fairness conclusion is possible from the prepared RecruitView table, which lacks verified demographic fields.

## Suggested conclusion

“This study developed a compact LLM-assisted pipeline using interview text and measurable vocal characteristics to predict perceived personality traits and speaking skills without visual inputs. The results showed moderate predictive associations. Response length provided a strong baseline, and the additional benefit of richer features remained uncertain. More complex feature representations and neural fusion did not consistently improve performance. A separately evaluated Big Five-to-BESSI mapping enabled exploratory skill estimates, but these require direct validation in the target population.”

## Poster-ready wording

- **Aim:** Investigate whether LLM-derived text cues and vocal characteristics can predict personality ratings and support soft-skill estimation.
- **Data:** RecruitView: 1,998 responses, 331 participants. Separate paired Big Five/BESSI data: 313 usable rows.
- **Approach:** Extract text and audio features; predict Big Five and speaking ratings; use a separately trained mapping to estimate BESSI domains.
- **Result:** Selected interview model: Big Five correlation 0.452; speaking correlation 0.486. Separate BESSI mapping: mean R² 0.504 on a row-level test split.
- **Finding:** Compact features performed reasonably; greater complexity did not consistently help. Added value over response length remains uncertain.
- **Limitations:** Different protocol from the published paper; previously viewed test results; indirect BESSI validation; Model 2 identity uncertainty; no independently validated emotion-recognition output or engineering-specific generalisation claim.
- **Future work:** Direct BESSI-labelled interviews, independent engineering-student evaluation, stronger participant identification, and a controlled visual-feature comparison.

## Evidence and reproducibility

- [Final diagnostics](../reports/benchmark/recruitview-final-review.md): uncertainty, length checks, repeated folds, question/duration groups, error examples and Model 2 intervals.
- [Original model comparison](../reports/benchmark/recruitview-feature-fusion.md).
- [Gated-fusion follow-up](../reports/benchmark/recruitview-gated-fusion.md).
- [Paper comparison](../reports/benchmark/recruitview-paper-comparison.md).
- [BESSI source verification](bessi-source-verification.md).
- [Existing simplified FigJam](https://www.figma.com/board/e0LtwOih5qMTa21kYEhQZ4).

Run `make close-recruitview-study` to regenerate the closing analysis and `make test` for the retained test suite. The closing JSON records input and primary-artifact hashes. Raw data and fitted models were preserved. All follow-up scores on the historical test must be described as exploratory; repeated checks cannot make it untouched again.

The current FigJam board is the source of truth for the four-step pipeline and project-status diagram. Older poster-build files and board snapshots are preserved outside the active tree under `archive/legacy/`.

## Completion boundary

The computational evaluation and the methods/results/limitations copy above are complete within the recorded scope. Final institutional poster layout, the full dissertation manuscript, supervisor review and submission are still required. Dataset redistribution terms must be checked before sharing the BESSI workbook; both OSF project and parent returned an unspecified licence. The current emotional features are LLM-derived cues, not an independently benchmarked emotion recogniser. Keep the existing title, but make this operational scope explicit in the dissertation.

## Core references

1. Gupta, A. K., et al. (2025). *RecruitView: A Multimodal Dataset for Predicting Personality and Interview Performance for Human Resources Applications*. [arXiv:2512.00450](https://arxiv.org/abs/2512.00450). Dataset and multimodal reference benchmark.
2. Soto, C. J., Napolitano, C. M., Sewell, M. N., Yoon, H. J., & Roberts, B. W. (2022). *An integrative framework for conceptualizing and assessing social, emotional, and behavioral skills: The BESSI*. Journal of Personality and Social Psychology, 123(1), 192–222. [DOI](https://doi.org/10.1037/pspp0000401). Conceptual and empirical basis for studying related personality and skill measures; it does not establish deterministic conversion.
3. Sewell et al. (2022). *Survey data of social, emotional, and behavioral skills among seven independent samples*. Data in Brief, 40, 107792. [DOI](https://doi.org/10.1016/j.dib.2022.107792). Data documentation and the link to the paired [college-student workbook](https://osf.io/7cktv/).
