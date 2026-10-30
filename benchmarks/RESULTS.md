# Initial benchmark results

Kernel success now means a correct, source-compliant **measured speedup strictly greater than 1.01× versus baseline**. This is a retrospective rescore requested after the original publication. The same sources, timings and timestamps are used; the original actors pursued larger historical-reference targets. No new model or GPU runs were performed.

**7 of 15 kernel cells exceed the measured cutoff; 7 also have a one-sided 95% timing lower bound above it.** All cutoff-crossing measurements also have timing lower bounds above 1.01×. The prefill measurements do not qualify. Two GPT-5.6 sources are AST-identical to the baseline; their small measured deviations are controls, not evidence of algorithmic progress.

One valid trial per model/task, high reasoning, with a 30-minute cap. Shared queue waits, infrastructure replacements and different cohort concurrency limit model comparisons.

![Progress matrix](charts/recovery-matrix.png)

## Kernel speedup and research time

Time is the earliest in-budget checkpoint containing the exact final source that was evaluated. Earlier different source versions were not exhaustively evaluated, so this is **time to the measured final artifact**, not proof of the earliest possible speedup. Total trial duration and exact seconds remain in the JSON/CSV.

| Task | Model | Measured speedup | Time to measured artifact | >1.01× | 95% lower bound > cutoff | Source note |
|---|---|---:|---:|---|---|---|
| Attention decode | GPT-6.1 Sol | 0.646674× | 22m 01.34s | No | No | Reviewed source |
| Attention decode | GPT-6 Astra | 0.841710× | 23m 38.49s | No | No | Reviewed source |
| Attention decode | GPT-5.6 Sol | — | — | No | — | Compilation failure |
| Dense projections | GPT-6.1 Sol | 1.127900× | 27m 36.72s | Yes | Yes | Reviewed source |
| Dense projections | GPT-6 Astra | 1.088004× | 18m 03.22s | Yes | Yes | Reviewed source |
| Dense projections | GPT-5.6 Sol | 0.515899× | 16m 14.39s | No | No | Reviewed source |
| Decoder layer | GPT-6.1 Sol | 1.044181× | 23m 55.46s | Yes | Yes | Reviewed source |
| Decoder layer | GPT-6 Astra | 1.098858× | 22m 22.85s | Yes | Yes | Reviewed source |
| Decoder layer | GPT-5.6 Sol | 1.002899× | 12m 35.59s | No | No | Unchanged baseline AST |
| Routed experts | GPT-6.1 Sol | 1.424259× | 22m 46.48s | Yes | Yes | Reviewed source |
| Routed experts | GPT-6 Astra | 1.266741× | 26m 54.08s | Yes | Yes | Reviewed source |
| Routed experts | GPT-5.6 Sol | 1.105026× | 11m 42.67s | Yes | Yes | Reviewed source |
| Prefill layer | GPT-6.1 Sol | 1.007282× | 20m 31.97s | No | No | Reviewed source |
| Prefill layer | GPT-6 Astra | 1.001426× | 15m 55.36s | No | No | Reviewed source |
| Prefill layer | GPT-5.6 Sol | 1.001131× | 12m 55.68s | No | No | Unchanged baseline AST |

![Kernel measurements](charts/kernel-performance.png)

The old “below target” label meant below our calibrated historical references (about 1.39×–2.45×), not necessarily slower than baseline. No candidate recovered those original reference targets. They remain a secondary metric in the JSON; the [original results](data/results-historical-target-v1.json), [original scoring policy](gpu/scoring-policy-historical-v1.json) and [calibration traces](data/calibrated-targets.json) are preserved.

These are local H200 measurements, not official OKC leaderboard scores. The requested raw cutoff does not establish that tiny gains are real; timing support and unchanged-code controls are reported separately.

## Integer multiplication

| Model | Outcome | Time to admitted witness |
|---|---|---:|
| GPT-6.1 Sol | pass | 11m 38.49s |
| GPT-6 Astra | pass | 13m 51.80s |
| GPT-5.6 Sol | right_censored_no_submission | — |

Integer admissions are exact finite conditional witnesses, with the trusted reference independently replayed through all eight mandatory stages. They do not prove inherited all-size compiler or analytic interfaces. GPT-5.6 reached the cap without a sealed submission.

![Integer recovery](charts/integer-recovery.png)

## Additional mathematical trials

| Task | Model | Evidence | Outcome |
|---|---|---|---|
| quasi-rh-nested-levels | GPT-6.1 Sol | independent_exact_frozen_verifier | pass |
| quasi-rh-nested-levels | GPT-6 Astra | independent_exact_frozen_verifier | pass |
| quasi-rh-nested-levels | GPT-5.6 Sol | independent_exact_frozen_verifier | pass |
| van-der-waerden-balance-dilation | GPT-6.1 Sol | submitted_proof_unadmitted | pending_review |
| van-der-waerden-balance-dilation | GPT-6 Astra | submitted_proof_unadmitted | pending_review |
| van-der-waerden-balance-dilation | GPT-5.6 Sol | submitted_proof_unadmitted | pending_review |
| unit-distance-grid | GPT-6.1 Sol | conditional_argument_review_plus_reference_algebra_checks | pass |
| unit-distance-grid | GPT-6 Astra | conditional_argument_review_plus_reference_algebra_checks | pass |
| unit-distance-grid | GPT-5.6 Sol | conditional_argument_review_plus_reference_algebra_checks | pass |
| sidorenko-residual | GPT-6.1 Sol | activation_geometry_dependency_unresolved | unscored_pending_source_review |
| sidorenko-residual | GPT-6 Astra | activation_geometry_dependency_unresolved | unscored_pending_source_review |
| sidorenko-residual | GPT-5.6 Sol | activation_geometry_dependency_unresolved | unscored_pending_source_review |

Quasi-RH admits exact arithmetic under H1–H5; unit-distance admits a conditional argument after non-formal curator review plus exact reference algebra checks. Neither is a complete formal theorem or a new headline exponent. Van der Waerden proof review and Sidorenko activation-geometry provenance remain unresolved and unscored.

## Reproduce and inspect

[Setup](README.md) · [Protocol](PROTOCOL.md) · [JSON](data/results.json) · [CSV](data/results.csv) · [Cohort metadata](data/cohort.json) · [Chart source](tools/chart.py) · [Table source](tools/render_results.py) · [GPU deletion accounting](data/gpu-lifecycle.json)

Sealed kernel sources, two-bank measurements, mathematical proof hashes and integer witnesses are published. Private optimized reference sources and raw credential/client logs are excluded. Research isolation cannot eliminate historical training contamination. Four infrastructure-invalid attempts were excluded and replaced; these remain documented in the data.
