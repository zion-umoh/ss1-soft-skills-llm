# Live pipeline probe results

Date: 2026-09-09

These are four independent live probes through the released three-skill
instrument, the locked Model 1 request, and the local Model 2 artifact. Each
probe used one Model 1 request and completed with a released runtime report.
The inputs are functional probes, not labelled participants, so they show
pipeline behaviour rather than predictive accuracy.

| Input profile | Model 1 `(cEXT, cNEU, cAGR, cCON, cOPN)` | Self-management | Cooperation | Innovation |
| --- | --- | ---: | ---: | ---: |
| Planner | `(6.5, 2.5, 6.5, 9.0, 5.5)` | 4.0991 | 3.6636 | 3.0919 |
| Collaborator | `(6.5, 2.5, 8.5, 7.5, 6.5)` | 3.9008 | 4.1820 | 3.4036 |
| Innovator | `(5.0, 3.0, 5.5, 8.5, 8.0)` | 4.0146 | 3.4142 | 3.6435 |
| Overwhelmed/reactive | `(3.5, 6.5, 3.0, 1.5, 2.5)` | 2.1691 | 2.4465 | 2.0824 |

The outputs show the intended directional behaviour: the collaborator profile
has the highest cooperation prediction, the innovator profile has the highest
innovation prediction, and the overwhelmed/reactive profile is lowest across
all three selected skills. This is a sanity check, not evidence of validity.

## Review-fix note

The first live attempt returned an incomplete Model 1 response because the
adapter inherited a 64-token compact default. The locked medium-reasoning
protocol was corrected to 512 output tokens, after which the live smoke test
and all 90 tests passed.

## 3SQ benchmark status

The chosen 3SQ paper reports sample-level psychometric validation, not a
public participant-level paired prediction file. Therefore no honest
correlation, error, or agreement score against that paper can be computed yet.
The benchmark runner is ready for a held-out CSV containing the same
participants' 3SQ scores and our predictions; synthetic probes are explicitly
excluded as benchmark ground truth.
