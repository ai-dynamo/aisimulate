<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GBDT forward-pass accuracy and coverage

Use this review with the
[online GBDT tutorial](../../../docs/perf-model/fpm-self-service/online-gbdt.md)
to evaluate cold start, fallback and online adaptation on a specific deployment.
Read prediction error together with coverage: a low error over available
predictions can hide a long initialization period or a workload class that
never becomes ready.

This page describes the evidence needed to assess the five data scenarios. It
does not supply a benchmark result for a particular model or qualify a
deployment. Published results must identify the evaluated configuration and
provide enough source information to reproduce its predictions and feedback
schedule.

## Evidence sources

The public [FPM Scenario Comparison](https://ai-dynamo.org/aisimulate/fpm-accuracy/scenario-comparison.html)
compares the five scenarios. Select a configuration, training share and method,
then inspect its coverage, recipe controls and native diagnostics. The
[dashboard documentation](../../../pages/fpm-accuracy/README.md#scenario-comparison)
explains the published protocol and the evidence attached to each evaluation.
Results depend on the selected evaluation; an unavailable scenario supplies no
accuracy evidence for that configuration.

The [evaluator documentation](../../../scripts/fpm_accuracy/README.md#scenario-comparison)
describes model construction and the predict–score–update lifecycle. Its
[execution instructions](../../../pages/fpm-accuracy/README.md#local-checks-and-smoke-evaluation)
explain the inputs for a complete campaign or a local smoke run. A smoke run
checks only its selected configurations.

For a review, retain the evaluation identity and its dataset identity with the
results. Record the complete model configuration and input hashes so a later
review can distinguish changed data from changed prediction behavior. The
dashboard's source links and qualification records provide this information
for published campaigns; a local campaign must retain equivalent records.

## Five data scenarios

The three sources are **A: self-benchmark**, **B: End-to-end offline profiling**,
and **C: online observations**. Self-benchmarking controls the measured batch shapes directly.
End-to-end offline profiling captures the batches produced by a serving
workload; online observations arrive during later operation.

| Scenario | Initial state and feedback | Evidence to inspect |
| --- | --- | --- |
| A-only | Fit from A and freeze the fit. Eligible missing mixed stores may use the original-A interpolation policy. | Self-benchmark identity, held-out prediction coverage and the estimator used for each workload class. |
| B-only | Fit on a declared subset of B; score the held-out subset without updates. The public comparison distributes training positions within each workload class. | Training share, split membership and any workload class without enough rows to fit. |
| C-only | Begin without a seed. Admit labels causally and construct each worker/workload store when it has enough observations. | Time and admitted-label count to the first prediction, including classes that never become ready. |
| A+C | Start from A and its qualified fallback; use C to update existing forests or construct missing ones. | Fallback use, first-forest construction and subsequent paired comparisons used to select an estimator. |
| B+C | Fit a chronological prefix of B; score the remaining observations before admitting their labels as C. | Prefix boundary, suffix coverage and feedback timing. This scenario has no original-A fallback unless that is explicitly added as a different policy. |

The evaluator replays recorded captures to simulate online feedback. In a live
B+C deployment, C comes from subsequent serving observations; it need not be
the suffix of the offline capture. Keep that distinction with the reported
protocol.

A result labeled GBDT can include interpolation for eligible missing mixed
stores in A and A+C. Describe such a result as the combined policy and report
its fallback use. The [mixed-workload fallback reference](../../../docs/perf-model/fpm_ml/mixed-workload-fallback.md)
defines source eligibility and the independently managed mixed classes.

Compare scenarios over the same capture population when assessing the benefit
of online feedback. A different training split or set of successful predictions
can change MAPE even when the underlying predictor is unchanged.

## Reference and experiment settings

Tutorial settings demonstrate a lifecycle; an evaluation recipe specifies the
experiment. Record the resolved settings instead of assuming that defaults
reproduce a published result.

| Control | Record for each evaluation |
| --- | --- |
| Initial fit | Feature preset, tree count, leaf limit, learning rate and minimum rows per worker/workload store. |
| Feedback admission | Event-time source, interval, first tick and treatment of gaps or equal timestamps. State whether bootstrap follows the same schedule. |
| Online updates | Retained window, historical weight, update interval and any limits on structural changes. |
| Fallback comparison | Paired-loss definition, smoothing factor, minimum comparisons, switching margin, persistence and reentry policy. |
| Calibration | History capacity, readiness threshold, correction bounds and workload-domain limits for each expert. |

The public strict-C protocol admits the first scored observation at or after
each five-second event-time tick, including during cold start. The first tick
is five seconds after the first scored timestamp. Crossing several empty ticks
admits that observation once; intervening observations do not become training
data. A+C and B+C follow the same schedule after initialization. These are
evaluation choices, not a requirement on every application's feedback rate.

The tutorial also explains a consecutive-observation bootstrap. Such a run
can initialize earlier and needs a distinct protocol label; its coverage cannot
be substituted for strict-C coverage. In either schedule, predict and score an
event before using its label.

Initial fitting labels are not paired error comparisons: the fitted forest
did not exist when those predictions were attempted. Track readiness separately
for each worker/workload store, since sparse classes can remain uninitialized
after other stores have warmed up. The
[causal-selection reference](../../../docs/perf-model/fpm_ml/online-gbdt-and-causal-selection.md#hybrid-calibration-with-a-causal-selector)
explains the comparison state and switching rule.

## Forward-pass accuracy and coverage

Let $Q$ contain the evaluation's eligible queries and let $S\subseteq Q$ contain
queries with successful predictions. For measured duration $y_i>0$ and
prediction $\widehat\tau_i$, report

$$
\operatorname{coverage}=100\frac{|S|}{|Q|},\qquad
\operatorname{MAPE}=\frac{100}{|S|}\sum_{i\in S}
\frac{|\widehat\tau_i-y_i|}{y_i}.
$$

Coverage retains cold starts and unavailable predictions in its denominator.
MAPE describes successful predictions only. Report an undefined metric when
its denominator is zero. Retain the integer counts as well as percentages:
coverage rounded to 100% can still include misses. Source exclusions, such as
invalid timing observations, need their own counts and reasons.

For A+C, let $F\subseteq S$ be the queries answered by fallback. Report
$100|F|/|S|$ as the fallback share of successful predictions, alongside
$100(|Q|-|S|)/|Q|$ as the unavailable share of eligible queries. These different
denominators should appear in the report. Break down the counts by workload
class so a well-covered decode stream does not hide an unavailable mixed class.

Assess switching over time as well as over the whole capture. Record the first
successful GBDT prediction, the first eligible paired comparison and each route
change. Where both experts predict before the label arrives, their errors can
be compared on those same observations. A lower full-capture error for the
combined policy alone does not establish that the warmed-up GBDT is better.

State the aggregation rule across captures and repeated runs. An unweighted
mean of capture MAPEs differs from a mean weighted by successful predictions.
Repeated execution measures run-to-run variability; testing on additional
workloads assesses generalization. Workloads used to choose thresholds or
other tuning parameters should be identified as tuning data.

The measured timer must also match the intended prediction target. A CUDA
model-step duration and a host inter-output interval measure different work.
Retain the timing boundary with the deployment identity, and report missing
timing metadata as a qualification gap.

## Documentation validation

Synthetic examples can check construction, feedback ordering, routing and
checkpoint restoration. They do not measure deployment accuracy. Record the
installed runtime and the observed behavior when running the tutorial, then
evaluate real captures with matched identities and timing boundaries.

A review should make untested behavior explicit. For example, successful
direct predictions leave native Replay integration unverified until the same
prepared model runs there. A checkpoint round trip should also check the
restored routing and update behavior needed by the application.

## Apply the results to a deployment

Select `fpm_learned` explicitly for these workflows. The application collects
observations and fits the initial model, then controls feedback admission and
model replacement. AISimulate predicts and updates an already constructed
model through its supported APIs. A general application fallback wrapper stays
outside native Replay and Recommend; native consumers require a prepared model
with their expected workload coverage and resource capacity.

| Check | Evidence needed |
| --- | --- |
| Forward-pass accuracy | Predictions paired with measured iteration durations, with timing boundaries and deployment identity matched. Report coverage, initialization duration and fallback use alongside error. |
| Native Replay/Recommend integration | Successful construction and execution for the intended backend, worker role and attention-DP configuration, including required stores and capacity information. |
| End-to-end serving accuracy | Matched measured-serving and Replay results for TTFT, TPOT, throughput and the relevant latency distributions. |

These checks answer different questions. A forward-pass result does not
establish the other two results for the same configuration. Follow the
[serving-validation procedure](../../../docs/perf-model/fpm-self-service/implementation.md#validate-collection-and-serving-accuracy)
for the measured comparison, and record unsupported or untested configurations
alongside the accepted ones.

Native mixed fallback also has a separate persistence constraint: a portable
snapshot cannot contain the complete mixed policy. The
[tutorial's persistence guidance](../../../docs/perf-model/fpm-self-service/online-gbdt.md)
describes the supported checkpoint routes and the original inputs needed for
continuation.

Keep the last accepted model with its configuration and source identity. To
resume learning after restoring it, also restore the matching feedback cursor
and buffers. Decide explicitly which later observations to replay so rollback
does not silently duplicate or discard admitted measurements.
