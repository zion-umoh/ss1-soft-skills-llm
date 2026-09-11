# Final evaluation: compact RecruitView study

Model expansion is closed. These analyses reuse existing data and saved predictions; no primary model is changed.

## Historical test results

292 responses from 49 participants. Scores are correlations, not accuracy percentages.

| Model | Big Five average | Speaking skills |
| --- | ---: | ---: |
| length_only | 0.4284 | 0.4685 |
| audio_only | 0.4645 | 0.4825 |
| llm_fused | 0.4518 | 0.4859 |

## Does the full feature set add more than response length?

Paired 95% intervals resample whole participants (1,000 draws). An interval containing zero does not establish an improvement. They are descriptive after prior test inspection.

| Primary model minus baseline | Big Five difference interval | Speaking difference interval |
| --- | ---: | ---: |
| length_only | [-0.0189, +0.0667] | [-0.0365, +0.0810] |
| audio_only | [-0.0499, +0.0280] | [-0.0499, +0.0636] |

## Primary per-target results

| Target | Spearman | 95% participant CI | MAE | RMSE |
| --- | ---: | --- | ---: | ---: |
| openness | 0.5099 | [+0.3552, +0.6335] | 0.6410 | 0.9976 |
| conscientiousness | 0.4638 | [+0.3190, +0.5786] | 0.5710 | 0.7917 |
| extraversion | 0.4833 | [+0.3291, +0.6094] | 0.6715 | 0.9919 |
| agreeableness | 0.4833 | [+0.3352, +0.5797] | 0.6513 | 1.0454 |
| neuroticism | 0.3187 | [+0.1670, +0.4414] | 0.3288 | 0.4240 |
| speaking_skills | 0.4859 | [+0.3469, +0.6099] | 0.7426 | 1.2065 |

## Stability across development participants

Five participant-grouped folds repeated with three fixed seeds. Previously selected alphas are held fixed; these folds provide sensitivity checks, not fresh unbiased validation. Fold SD is descriptive and is not a confidence interval.

| Model | Mean Big Five (fold SD) | Mean speaking (fold SD) |
| --- | ---: | ---: |
| length_only | 0.3476 (0.0524) | 0.4160 (0.0544) |
| audio_only | 0.3408 (0.0368) | 0.3960 (0.0420) |
| llm_fused | 0.3652 (0.0370) | 0.4316 (0.0467) |

## Response length and duration

Descriptive partial rank correlations remove linear associations with ranked word count and duration from both labels and predictions. They do not remove every length effect or prove causation.

- openness: 0.2073
- conscientiousness: 0.1057
- extraversion: 0.2990
- agreeableness: 0.1742
- neuroticism: 0.1588
- speaking_skills: 0.2002

Duration groups use training-only boundaries of 19.86 and 36.41 seconds. Question/duration subgroup scores are descriptive and are suppressed below 10 participants. No demographic attributes are inferred.

| Subgroup | Rows | Participants | Big Five | Speaking |
| --- | ---: | ---: | ---: | ---: |
| duration_short | 88 | 27 | 0.3202 | 0.2884 |
| duration_medium | 90 | 32 | 0.3042 | 0.2227 |
| duration_long | 114 | 29 | 0.2280 | 0.2873 |
| question_1 | 34 | 31 | 0.7057 | 0.8109 |
| question_76 | 24 | 21 | 0.4798 | 0.5965 |

## Error examples

Five largest mean Big Five errors, scaled using development-label standard deviations. IDs support local audit; transcripts and identifying details are omitted. Large signed errors show the direction of the mismatch, not its psychological cause.

| Record | Question | Seconds | Largest-error trait | Observed | Predicted | Error (SD) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| recruitview_c211bfaf37edd35f | 68 | 30.7 | conscientiousness | -5.877 | -0.317 | +6.37 |
| recruitview_0c1c43111448b131 | 33 | 29.4 | extraversion | -6.305 | -0.103 | +5.73 |
| recruitview_051c2e380d07844f | 8 | 29.2 | agreeableness | 6.978 | 0.059 | -5.79 |
| recruitview_12d9634a066ef75f | 11 | 5.1 | agreeableness | 7.064 | 0.797 | -5.25 |
| recruitview_c5080dbc42e822a3 | 56 | 42.7 | extraversion | -5.211 | -0.537 | +4.32 |

## Model 2 uncertainty

Computed from the 47 saved test rows; no model refit. Row bootstrap assumes independent rows, which the source identifier cannot verify. These intervals are exploratory.

- mae: 0.3360; 95% interval [+0.2978, +0.3770].
- rmse: 0.4055; 95% interval [+0.3625, +0.4519].
- mean_r2: 0.5041; 95% interval [+0.3197, +0.6028].

## Boundaries

- Historical test previously inspected; descriptive results cannot restore independent confirmation.
- Repeated development folds use previously selected alphas; sensitivity analysis, not nested unbiased model selection.
- Length adjustment is descriptive and cannot establish causal skill measurement.
- Model 2 intervals assume independent rows, an assumption not verified from the non-unique Case field.
- RecruitView contains perceived interview ratings, not measured BESSI skills.

Full fold membership, subgroup suppression, input hashes and machine-readable metrics are in the companion JSON. Run `make close-recruitview-study` to regenerate.
