# Small gated-fusion follow-up

Exploratory extension after viewing the original test results. This is not a fresh confirmatory test or a CRMF reproduction.

Inputs: 10 cached LLM text columns + 12 engineered audio columns. No new API calls or media extraction.
Development: 1706 responses / 282 participants; 3 grouped folds. Each neural prediction averages 3 fixed seeds.
Preprocessing is fitted separately inside each training fold. All six outputs share a small network. Huber uses training-standardised targets; ranking uses training-label order, not original human pairwise judgments.
Epoch and model selection maximise mean fold Big Five Spearman. Speaking skills is secondary and never chooses the winning variant. Decisions are saved before historical test scoring.

| Variant | Epoch / alpha | Development Big Five (fold SD) | Development speaking | Historical test Big Five | Historical test speaking |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_mse | 150 | 0.3721 (0.0614) | 0.4308 | 0.4644 | 0.4770 |
| gated_mse | 100 | 0.3821 (0.0554) | 0.4423 | 0.4624 | 0.4633 |
| gated_huber | 150 | 0.3839 (0.0569) | 0.4517 | 0.4591 | 0.4557 |
| gated_huber_rank | 150 | 0.3842 (0.0570) | 0.4517 | 0.4592 | 0.4551 |
| ridge | 100.0 | 0.3601 (0.0599) | 0.4209 | 0.4519 | 0.4859 |

Development winner: **gated_huber_rank:150**. Existing primary Model 1 and downstream BESSI artifacts are preserved.

## Paired participant-bootstrap uncertainty on the historical test

Intervals below are score differences against the Ridge refit using this same development protocol. They do not account for prior researcher exposure to test results.

| Variant | Big Five difference, 95% CI | Speaking difference, 95% CI |
| --- | ---: | ---: |
| fixed_mse | [-0.0077, +0.0325] | [-0.0383, +0.0188] |
| gated_mse | [-0.0189, +0.0425] | [-0.0585, +0.0145] |
| gated_huber | [-0.0272, +0.0396] | [-0.0678, +0.0073] |
| gated_huber_rank | [-0.0275, +0.0398] | [-0.0679, +0.0059] |

## Interpretation

The gate learns response-dependent weights; these weights are not calibrated reliability estimates or causal explanations. Comparisons isolate fixed versus learned fusion under MSE, then Huber and the additional ranking term. All settings were fixed before running this follow-up.
The original Ridge report uses a different selection procedure. Use the matched Ridge refit here to assess this extension; retain the original results as the historical reference.
No claim of better true-personality measurement, BESSI validation, or superiority to published CRMF follows from these correlations. A fresh external dataset would be needed for new independent confirmation.
