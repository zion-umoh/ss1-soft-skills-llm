# Final RecruitView comparison with published CRMF

This is a contextual benchmark, not an exact reproduction: the paper uses multimodal CRMF with VideoMAE + Wav2Vec2 and its own split, while this study uses the fixed participant-disjoint split, engineered audio columns and an LLM-assisted structured text representation.

## Big Five (Spearman ρ)

| Target | Our selected variant | Published CRMF | Difference |
| --- | ---: | ---: | ---: |
| openness | 0.5099 | 0.6384 | -0.1285 |
| conscientiousness | 0.4638 | 0.5572 | -0.0934 |
| extraversion | 0.4833 | 0.5681 | -0.0848 |
| agreeableness | 0.4833 | 0.5927 | -0.1094 |
| neuroticism | 0.3187 | 0.2603 | +0.0584 |

## Direct `speaking_skills` benchmark

| Our selected variant | Published CRMF | Difference |
| ---: | ---: | ---: |
| 0.4859 | 0.5947 | -0.1088 |

The published values are not a pass/fail threshold. They show the context in which this smaller, engineered-feature experiment should be interpreted.
