# Model 1 feature-balance analysis

This is an exploratory post-selection diagnostic using a train-only Ridge refit and the fixed validation split. It does not retune the final model.

Selected model: **llm_fused**; validation rows: **303**; baseline Big Five macro Spearman: **0.4010**.

## Feature blocks

| Block | Absolute coefficient sum | Validation permutation score drop |
| --- | ---: | ---: |
| llm_affect | 0.7524 | +0.0534 ± 0.0151 |
| llm_social_agency | 0.3994 | +0.0135 ± 0.0065 |
| llm_cognition_communication | 0.7542 | +0.0828 ± 0.0244 |
| audio_timing | 1.4842 | +0.1630 ± 0.0276 |
| audio_pitch_energy | 0.4470 | +0.0014 ± 0.0032 |
| audio_spectral | 0.4551 | +0.0121 ± 0.0068 |

## Leave-one-block-out sensitivity

Each row refits the same train-only Ridge protocol after removing one block; alpha remains fixed at the selected value. The test column is diagnostic only and was not used for selection.

| Omitted block | Validation macro Spearman | Change | Train-only test macro Spearman | Change |
| --- | ---: | ---: | ---: | ---: |
| llm_affect | 0.3999 | -0.0011 | 0.4672 | +0.0139 |
| llm_social_agency | 0.4017 | +0.0007 | 0.4581 | +0.0048 |
| llm_cognition_communication | 0.3819 | -0.0191 | 0.4498 | -0.0035 |
| audio_timing | 0.3963 | -0.0047 | 0.4269 | -0.0264 |
| audio_pitch_energy | 0.4031 | +0.0021 | 0.4545 | +0.0012 |
| audio_spectral | 0.3983 | -0.0027 | 0.4452 | -0.0081 |

## High feature correlations

| Feature A | Feature B | Pearson r |
| --- | --- | ---: |
| audio_pause_fraction | audio_speaking_time_ratio | -1.000 |
| audio_rms_mean | audio_rms_std | +0.939 |
| audio_zero_crossing_rate | audio_spectral_centroid_mean_hz | +0.868 |
| cognitive_complexity | response_elaboration | +0.806 |
| audio_duration_seconds | audio_pause_count | +0.804 |
| response_elaboration | audio_duration_seconds | +0.782 |

## Interpretation boundary

Large coefficients or permutation drops indicate predictive concentration in this fitted model, not causal importance. In particular, duration and speaking-rate features must be interpreted as potential response-length confounds.
