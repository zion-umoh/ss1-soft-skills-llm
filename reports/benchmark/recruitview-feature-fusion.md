# RecruitView Model 1 feature comparison

Raw audio was decoded into numerical characteristics and pretrained speech embeddings; no waveform was passed directly to a Ridge predictor.
The transcript representation is a frozen `sentence-transformers/all-MiniLM-L6-v2` embedding.
The LLM text representation is a cached, structured `gpt-5.6-luna` extraction of observable linguistic and affective features; it was not asked to predict the targets directly.

| Variant | Validation Big Five Spearman | Test Big Five Spearman | Validation speaking_skills Spearman | Test speaking_skills Spearman |
| --- | ---: | ---: | ---: | ---: |
| length_only | 0.3353 | 0.4284 | 0.4207 | 0.4685 |
| text_only | 0.2340 | 0.2666 | 0.2776 | 0.3275 |
| audio_only | 0.3395 | 0.4645 | 0.4030 | 0.4825 |
| fused | 0.2651 | 0.3165 | 0.2966 | 0.3469 |
| llm_text_only | 0.3641 | 0.3945 | 0.4234 | 0.4237 |
| llm_fused | 0.4010 | 0.4518 | 0.4685 | 0.4859 |
| speech_only | 0.3408 | 0.3595 | 0.4143 | 0.3126 |
| llm_speech | 0.3536 | 0.3790 | 0.4324 | 0.3416 |
| llm_speech_fused | 0.3585 | 0.3869 | 0.4419 | 0.3609 |
| late_fusion | 0.3460 | 0.4537 | 0.4143 | 0.4774 |
| llm_late_fusion | 0.4007 | 0.4616 | 0.4628 | 0.4931 |
| mean_baseline | n/a | n/a | n/a | n/a |

Selected Big Five variant using validation only: **llm_fused**.
The held-out test values above were generated once after that selection and must not be used to tune the model.

This is a same-dataset RecruitView benchmark. It is not direct BESSI validation; the final published CRMF comparison is a separate later batch.
