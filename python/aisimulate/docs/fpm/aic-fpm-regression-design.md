# FPM regression: worker ownership and workload buckets

The regression model learns forward-pass latency from earlier observations.
This design separates those observations at two levels: each worker has its
own predictor, and each Aggregated predictor has a separate fit for each
workload kind. A Decode-only iteration therefore uses a fit learned from
earlier Decode-only iterations on the same worker. A fit is the learned
relationship between workload features and latency.

This document explains the AISim bucket change relative to revision
`99d6acb722bf75e2b16119c59c79e1bad73b4efd`. It also describes the caller contract
implemented by the companion FPM Gym change. The current validation uses
offline replay. Live Planner integration remains a separate task.

Here, regression means learning from `ForwardPassMetrics` observations. The
separately collected `fpm_forward_perf` database belongs to the existing
interpolation path.

## 1. Why change the design?

Two sources of mixing can reduce fitting accuracy:

1. **Different workers share a predictor.** Every observation sent to that
   instance updates the same learned state, even if its `worker_id` differs.
2. **Different workloads share a fit within one worker.** An Aggregated worker
   can alternate among Decode-only, Prefill-only, and combined work. These
   workloads can have different relationships between the existing features
   and latency.

Worker ownership addresses the first issue in the caller. Workload buckets
address the second issue inside AISim. Both are needed for the offline design.
Changing the bucket layout alone does not isolate workers if the caller still
sends several workers to one instance.

### Before and after in AISim

| Aspect | Before this PR | With this PR |
|---|---|---|
| Predictor role | Fixed at construction: Prefill, Decode, or Aggregated | Same fixed role |
| Prefill predictor | One regression bucket | One `pure_prefill` bucket |
| Decode predictor | One regression bucket | One `pure_decode` bucket |
| Aggregated predictor | One bucket shared by all workloads | Four buckets, selected from the full iteration |
| Observation limit | One limit for the predictor's single bucket | The same limit applies independently to each bucket |
| Readiness | The single fit determines readiness | Each bucket has its own readiness; summary readiness means any bucket has a fit |
| Diagnostics | Predictor-level summary | Existing summary plus per-bucket diagnostics |
| Worker ownership | Caller responsibility | Caller responsibility; the companion Gym change supplies a worker map |

The required role argument, two regression features, feature weights, fitter,
and retention algorithm already existed at the baseline. This PR changes how
observations are partitioned among fits.

## 2. Ownership and terminology

The terms below describe different parts of the design:

- **Predictor:** one `ForwardPassPerfModel` instance, with a fixed regression
  role and its own learned state.
- **Iteration:** the metrics for one worker's forward pass across its
  attention data-parallel (DP) ranks. Each rank reports its scheduled work.
- **Workload bucket:** the observations and fit for one workload kind within
  that predictor.
- **Retention cell:** a region of the feature space inside a bucket, used to
  decide which observations to keep. A cell does not have its own fit.

The two existing features summarize attention work and total feed-forward or
mixture-of-experts (FFN/MoE) token work. The reference sections give their exact
formulas and explain how each rank contributes.

### Responsibilities

| Component | Responsibility |
|---|---|
| Gym caller | Infer each worker's offline role, create one predictor per `worker_id`, and preserve replay order |
| AISim predictor | Validate the iteration, extract features, select the workload bucket, predict, and tune |
| Workload bucket | Retain its own observations, rebuild its fit, and track its readiness |

For example, the caller could own the following predictors:

| Worker | Fixed role | Buckets owned by its predictor |
|---|---|---|
| A | Prefill | `pure_prefill` |
| B | Decode | `pure_decode` |
| C | Aggregated | All four workload buckets |
| D | Aggregated | A separate set of all four workload buckets |

Workers C and D use the same implementation but have independent observations,
fits, retention state, and readiness. The caller selects the predictor before
calling the usual prediction or tuning method.

AISim accepts `worker_id` in FPM identity metadata, but that field does not
select or validate an instance or bucket inside AISim. To obtain worker
isolation, the caller must consistently send one worker's observations to its
own instance.

### Offline role inference in the companion Gym change

Before replay, Gym examines eligible observations from each worker across all
supplied active ranks:

| Work present in the worker's complete observed history | Fixed regression role |
|---|---|
| Prefill only | Prefill |
| Decode only | Decode |
| Both phases, either together or in different iterations | Aggregated |

A worker that starts with Decode-only iterations and later performs Prefill
work is therefore Aggregated for the entire replay. Its individual iterations
can still route to different buckets. Role inference uses scheduled workload
fields and does not read observed latency. It is an offline policy that uses
the complete workload inventory; it does not infer the configured deployment
role from a live prefix of the trace.

## 3. How AISim selects a workload bucket

Classification uses scheduled work. A rank has Prefill activity when
`sum_prefill_tokens > 0` and Decode activity when `num_decode_requests > 0`.
A rank with neither is idle. Cached Prefill KV by itself does not create
Prefill activity. Existing metric validation runs before routing.

Prefill and Decode predictors always use their single bucket, after the existing
role-compatibility checks. For an Aggregated predictor, AISim examines every
active rank in the iteration:

| Bucket label | Selection rule |
|---|---|
| `pure_decode` | Every active rank performs Decode only |
| `contains_locally_mixed` | At least one rank itself performs both Prefill and Decode |
| `cross_rank_aggregated` | There are Prefill-only and Decode-only ranks, and no rank performs both phases |
| `pure_prefill` | Every active rank performs Prefill only |

Locally mixed work takes precedence over cross-rank separation. Idle ranks do
not affect the decision. AISim computes the label in the same regression
feature-extraction function used by both prediction and tuning.

### Routing examples for an Aggregated predictor

In this table, `P` means Prefill only, `D` means Decode only, `P+D` means both
phases on the same rank, and `idle` means no scheduled work.

| Rank 0 | Rank 1 | Rank 2 | Selected bucket |
|---|---|---|---|
| D | D | idle | `pure_decode` |
| P | P | idle | `pure_prefill` |
| D | P | idle | `cross_rank_aggregated` |
| P+D | D | P | `contains_locally_mixed` |
| idle | P+D | idle | `contains_locally_mixed` |
| idle | idle | idle | No bucket; predict zero and retain no observation |

For example, suppose rank 0 has a large Decode batch and rank 1 has a small
Prefill batch. The iteration is `cross_rank_aggregated`, even if rank 0 would
be chosen as a representative rank by another estimator. Looking at only that
rank would miss the Prefill work. Reordering the ranks does not change the
regression bucket label.

All four Aggregated buckets use the existing Aggregated feature formula. The
bucket label chooses which observations train a fit; it does not change the
predictor's role or introduce another feature formula.

## 4. Prediction and tuning for one iteration

The caller keeps the original observation order. For each eligible, nonempty
iteration:

1. Select the predictor using the worker's identity.
2. Ask it to predict from scheduled workload fields. AISim validates the
   metrics, computes the existing two features, and selects the workload bucket.
3. If validation succeeds and the selected bucket can produce a finite
   estimate, return that latency. If validation succeeds but no estimate is
   available, return `None` in Python. Invalid metrics or a role mismatch raise
   an error.
4. Record the outcome and score an available estimate against observed latency.
5. Tune the same worker's predictor with the observation. AISim uses the same
   classifier, retains the sample in the selected bucket, and refits that bucket.

The current iteration's latency is introduced at the tuning step. Prediction
does not consume it. An unavailable prediction can still be followed by a
tuning update, allowing the selected bucket to warm up.

### Warm-up is per bucket

For a fresh bucket with the default minimum of five observations, the first
five nonempty queries are unavailable when each query precedes one tuning
update. The sixth query can use a fit if those five earlier observations were
accepted for tuning and support a usable fit.

Five observations are necessary but do not guarantee a usable fit. For
example, observations with identical feature values do not provide the
variation needed to learn a latency response. A fit can also become unavailable
after an update or eviction if the retained data no longer supports it.
Even a ready fit can return unavailable if a particular query produces a
nonfinite calculation.

An Aggregated predictor never uses one bucket's fit to answer a query for
another bucket. Suppose its state is:

| Bucket | Retained observations | Has a usable fit? |
|---|---:|---|
| `pure_decode` | 20 | Yes |
| `contains_locally_mixed` | 0 | No |
| `cross_rank_aggregated` | 0 | No |
| `pure_prefill` | 3 | No |

The predictor has 23 retained observations and reports summary readiness as
`ready`. A valid Decode-only query can use the Decode fit. A Prefill-only query
still returns `None`, because its selected bucket has only three observations.

## 5. Capacity and retention cells

The option values keep their existing defaults. Their scope changes for an
Aggregated regression predictor:

| Option | Default | Meaning in each workload bucket |
|---|---:|---|
| `max_observations` | 64 | Maximum number of retained observations |
| `min_observations` | 5 | Minimum retained observations before a fit can be ready |
| `bucket_count` | 16 | Internal retention grid: four cells on each of the two feature axes |

The resulting capacity is:

| Predictor role | Allocated buckets | Maximum retained observations at defaults |
|---|---:|---:|
| Prefill | 1 | 64 |
| Decode | 1 | 64 |
| Aggregated | 4 | 256 |

Capacity stays with its bucket. If an Aggregated worker only produces
Decode-only work, its Decode bucket still retains at most 64 observations.
The unused buckets do not lend it their capacity.

Each bucket has one fit over all of its retained observations. Its 16 retention
cells divide the existing two-dimensional feature space to guide sample
retention. They do not mean 16 fits or a fixed allowance of four observations
per cell.

The existing retention algorithm expands its grid bounds as observations
arrive and removes an observation from a largest cell when the bucket exceeds
its capacity. It does not simply keep the latest 64 observations. Hash-map
ordering can affect eviction ties and sample order, so repeated runs may keep
different samples. This PR preserves that behavior.

## 6. Diagnostics and API compatibility

The existing API names use `store`; they refer to the workload buckets
described here.

The new `regression_store_diagnostics()` method reports every allocated bucket,
including buckets that have never received an observation:

| Field in each entry | Meaning |
|---|---|
| `workload_kind` | Bucket label, such as `pure_decode` |
| `ready` | Whether that bucket currently has a fit |
| `retained_observations` | Number of observations currently kept in that bucket, after eviction |

Prefill and Decode return one entry. Aggregated returns four entries in this
order: `pure_decode`, `contains_locally_mixed`, `cross_rank_aggregated`, and
`pure_prefill`. Native models return an empty list.

The public Rust method returns a vector of
`ForwardPassRegressionStoreDiagnostics` records. PyO3 serializes the records
as JSON, and the Python facade returns a list of dictionaries:

```python
from aisimulate_core.sdk import RustForwardPassPerfModel

model = RustForwardPassPerfModel.from_regression(
    "aggregated",
    {"max_observations": 64, "min_observations": 5, "bucket_count": 16},
)

summary = model.diagnostics()
buckets = model.regression_store_diagnostics()
```

Immediately after construction, `buckets` contains four entries with
`ready=False` and `retained_observations=0`.

### Summary readiness and query readiness answer different questions

| Diagnostic | Question it answers |
|---|---|
| Summary `retained_observations` | How many observations are kept across all buckets? |
| Summary `readiness == "ready"` | Does at least one bucket currently have a fit? |
| A bucket's `ready` field | Does this particular workload bucket have a fit? |
| The prediction result | Could the selected bucket produce an estimate for this query? |

The existing summary keeps its field names. When no bucket is ready, explicit
regression reports `insufficient_data`; regression fallback carrying a native
failure warning reports `unsupported_config`. These existing labels describe
the predictor summary. Gym uses the selected bucket's diagnostics for its
per-prediction readiness metadata.

Existing constructor signatures and prediction return types are unchanged by
this PR. The role argument and regression-weight options were already required
or supported at the implementation baseline. No package or telemetry-schema
version change is introduced here.

## 7. Native interpolation and correction

The four-bucket classifier is specific to regression. The native path keeps
its existing behavior:

| Native component | Behavior retained |
|---|---|
| Interpolation | Existing performance-database lookup and interpolation |
| Workload classification | Prefill, Decode, and Mixed categories with existing representative-rank features |
| Online correction | Three existing fixed-grid correction buckets, correction bounds, and readiness rules |
| Native constructors | Existing configuration, validation, and estimation behavior |
| `best_available()` | Use native estimation when native construction succeeds; use the new regression buckets when it falls back |

The companion Gym change applies the worker map only to Regression. Op-based
and FPM-based predictors retain their existing construction and evaluation
behavior. Compatibility tests cover both normal prediction and error handling.

## 8. Role and observation contract

Rust uses `ForwardPassWorkerType::{Prefill, Decode, Aggregated}`. Python accepts
exactly `"prefill"`, `"decode"`, and `"aggregated"`.

- A Prefill predictor rejects scheduled Decode requests on any rank.
- A Decode predictor rejects fresh Prefill tokens on any rank.
- An Aggregated predictor accepts all four nonempty workload compositions.
- Existing metric validation rejects inconsistent counts, such as Decode KV
  tokens without a Decode request.
- A valid rank list with no scheduled work predicts `0.0` and adds no sample.
  An empty rank list is invalid. An empty outer batch passed to tuning is a
  no-op.

Tuning uses the maximum finite, positive `wall_time` across all supplied ranks,
including idle ranks, and converts seconds to milliseconds. An observation
without a usable target is skipped. A target that overflows during conversion
is not retained. Prediction ignores `wall_time` and queued request fields.

The caller supplies complete, unique, iteration-aligned rank sets and correct
worker identities. This PR does not add expected-rank-count or worker-lifecycle
enforcement inside AISim. Gym retains its existing parser exclusions.

The following signatures already exist and remain available:

```text
Rust:   from_regression(worker_type, options)
        best_available(config, worker_type, options)
        best_available_with_roots(config, worker_type, options, systems_root)
Python: from_regression(worker_type, options=None)
        best_available(config, worker_type, options=None)
```

`from_native` and `from_native_with_roots` remain role-free. Each regression
instance keeps its role for its lifetime. A caller that needs another role
creates a new predictor.

## 9. Validation and limits

The implementation tests cover full-rank classification, rank-order
invariance, idle ranks, locally mixed precedence, independent fitting and
eviction, capacity limits, role guards, and diagnostics through the compiled
Python extension. Native interpolation and correction are checked separately
to confirm that their behavior remains unchanged.

### Hand-derived routing and prediction oracle

The Rust test
`regression_bucket_predictions_match_hand_derived_equal_feature_oracle`
gives all four workload buckets identical feature coordinates but different
synthetic observed latencies. This makes accidental sharing observable: a shared
fit cannot return four different answers for the same coordinates.

For pure Prefill work with unit weights and one request, the existing feature
definition reduces to `A = (P + 1) * H + P * (P + 1) / 2` and `T = P`. The test
constructs the same `(A, T)` for each workload:

| Workload | Scheduled work |
|---|---|
| Pure Decode | `B = T`, `K = A` |
| Locally mixed | One rank: `P = 1`, `H = 0`, `B = T - 1`, `K = A - 1` |
| Cross-rank aggregated | Prefill rank: `P = 1`, `H = 0`; Decode rank: `B = T - 1`, `K = A` |
| Pure Prefill | `(P, H)` values `(2, 0)`, `(3, 0)`, `(2, 1)`, `(4, 0)`, `(3, 1)` |

The five training coordinates are `(3, 2)`, `(6, 3)`, `(6, 2)`, `(10, 4)`,
and `(10, 3)`. Literal targets follow `y = c + 2A + 3T`, with intercepts
`c = 10, 20, 30, 40` milliseconds in the table's order. At the query
`(A, T) = (14, 3)`, the independently calculated answers are **47, 57, 67,
and 77 ms**. The previous pooled fit would average these equally represented
intercepts and return **62 ms** for every workload. This is a mathematical
before/after anchor, not a benchmark measurement.

The test uses literal labels and expected answers; it does not call the
production feature extractor to calculate them. It also checks that each
bucket remains unavailable until its own five observations have arrived,
and that training another bucket cannot change an already fitted answer.

### Offline benchmark evidence and limits

Joint offline validation with the Gym caller used seven configurations and
five fresh processes for each configuration. The 35 runs evaluated 5,751,130
iterations with zero prediction or tuning errors. The 1,150,226 distinct
iterations matched the saved input inventory before accuracy was compared.

The results broadly reproduce the earlier `mixed_role_sensitivity64`
experiment, with some median differences. MiniMax M3 mean absolute percentage
error (MAPE) was 2.016% compared with 1.967% in the saved runs;
the repeat ranges overlap. H200 TP4 coverage remained variable within its
historical range. These results support the partitioning design but do not
establish an accuracy guarantee for a new workload. The unchanged retention
and numerical behavior can still affect accuracy and later readiness.

Live Planner role forwarding, worker lifecycle handling, collection/grouping
changes, alternative bucket splits, and changes to the fitting or retention
algorithms need separate work. The sections below describe the existing
mathematics used inside every bucket.

## Reference: symbols and features

For attention-DP rank $d$, define:

| Symbol | `ForwardPassMetrics` quantity | Meaning |
|---|---|---|
| $P_d$ | `sum_prefill_tokens` | newly computed Prefill tokens |
| $H_d$ | `sum_prefill_kv_tokens` | previously cached Prefill KV tokens |
| $N_d$ | `num_prefill_requests` | scheduled Prefill requests |
| $B_d$ | `num_decode_requests` | scheduled Decode requests |
| $K_d$ | `sum_decode_kv_tokens` | Decode KV tokens read |

$H_d$ and $P_d$ are sums over the Prefill requests scheduled on rank $d$;
$K_d$ is a sum over that rank's Decode requests. They are not per-request
lengths. $N_d$ and $B_d$ are the corresponding per-rank request counts.

The telemetry schema permits fully cached metadata with $P_d=0$ and
$H_d>0$. Cached Prefill KV creates attention work only when fresh Prefill
work exists:

$$
\widetilde H_d=H_d\,\mathbf{1}[P_d>0].
$$

Per-request token lengths are not available, so Prefill attention pairs use a
balanced-request approximation. Under that approximation, each of the $N_d$
requests has cached length $H_d/N_d$ and newly computed length $P_d/N_d$.
For $P_d>0$, metric validation guarantees $N_d>0$, and the total
attention-pair estimate for rank $d$ is

$$
\begin{aligned}
Q_d
&=N_d\left[
\left(\dfrac{H_d}{N_d}\right)\left(\dfrac{P_d}{N_d}\right)
+\dfrac{(P_d/N_d)(P_d/N_d+1)}{2}
\right]\\
&=\dfrac{H_dP_d}{N_d}+\dfrac{P_d^2}{2N_d}+\dfrac{P_d}{2}.
\end{aligned}
$$

When $P_d=0$, define $Q_d=0$. Thus $Q_d$ already includes the factor $N_d$:
it estimates total Prefill attention-pair work on the rank, not work for one
request. Callers must not multiply it by $N_d$ again. $N_d$ is the number of
Prefill requests on rank $d$, not the attention-DP size; multiple ranks are
reduced only by the role-specific maximum and sum below.

All roles share the axis order

$$
x=[\text{critical attention},\ \text{global FFN/MoE}].
$$

With $\alpha$ the KV-attention weight, $\beta$ the Prefill
attention-pair weight, and $\gamma$ the tokenwise FFN/MoE weight, the exact
features are:

$$
x_P=
\left[
\max_d\left(\alpha\widetilde H_d+\beta Q_d\right),
\ \gamma\sum_d P_d
\right],
$$

$$
x_D=
\left[
\alpha\max_d K_d,
\ \gamma\sum_d B_d
\right],
$$

$$
x_A=
\left[
\max_d\left(\alpha(\widetilde H_d+K_d)+\beta Q_d\right),
\ \gamma\sum_d(P_d+B_d)
\right].
$$

The reductions for one and multiple attention-DP ranks are shown below.
Unsubscripted symbols in the `attention_dp = 1` column refer to the sole rank.

| Worker | `attention_dp = 1` | `attention_dp > 1` |
|---|---|---|
| Prefill | $x[0]=\alpha\widetilde H+\beta Q$<br>$x[1]=\gamma P$ | $x[0]=\max_d(\alpha\widetilde H_d+\beta Q_d)$<br>$x[1]=\gamma\sum_d P_d$ |
| Decode | $x[0]=\alpha K$<br>$x[1]=\gamma B$ | $x[0]=\alpha\max_d K_d$<br>$x[1]=\gamma\sum_d B_d$ |
| Aggregated | $x[0]=\alpha(\widetilde H+K)+\beta Q$<br>$x[1]=\gamma(P+B)$ | $x[0]=\max_d[\alpha(\widetilde H_d+K_d)+\beta Q_d]$<br>$x[1]=\gamma\sum_d(P_d+B_d)$ |

The regression remains two-dimensional at every attention-DP size. Critical
attention uses a maximum across ranks, while global FFN/MoE work uses a sum.

The Aggregated maximum is taken after composing the entire rank-local
attention score; the implementation must not combine independent maxima from
different ranks. Counters are converted to `f64` before multiplication or
cross-rank summation, and derived features must be finite and nonnegative.

The options are construction-time knobs and all default to `1.0`:

| Formula | `ForwardPassPerfOptions` field |
|---|---|
| $\alpha$ | `regression_attention_kv_weight` |
| $\beta$ | `regression_prefill_attention_pair_weight` |
| $\gamma$ | `regression_ffn_token_weight` |

- $\alpha$ scales KV-token-related attention work for every role:
  $\widetilde H$ for Prefill, $K$ for Decode, and $\widetilde H+K$ for
  Aggregated. It is not a Prefill-only weight.
- $\beta$ scales only the Prefill attention-pair estimate $Q$. It does not
  represent Decode attention.
- $\gamma$ scales global tokenwise FFN/MoE work: $P$ for Prefill, $B$ for
  Decode, and $P+B$ for Aggregated.

They must be finite and strictly positive when regression is constructed. They
are ignored by a successful native-only model. Changing a weight requires
constructing a new model, because it changes the feature and retention-cell
coordinates. To preserve learned history, replay the original per-rank FPM
observations into the new model.

The ergonomic Python facade accepts ordinary Python floats and marshals the
three weight fields on a shallow copy of the caller's options dictionary.
Finite values remain JSON numbers; nonfinite values use the exact valid-JSON
string sentinels `"NaN"`, `"Infinity"`, and `"-Infinity"`. JSON-oriented raw
PyO3 callers may send those same sentinels. Rust deserialization maps only
those exact strings back to their corresponding `f64` values; unknown strings
and other value types are invalid. This transport does not relax validation:
`from_native` and a successful native `best_available` ignore all three
weights, whereas `from_regression` and a fallback `best_available` reject a
decoded nonfinite value with the corresponding field-specific error.

## Reference: retention and fit pipeline

Each accepted observation keeps the raw two-dimensional feature vector and
observed milliseconds. Retention uses separate bucket coordinates:

$$
b_j=\log(1+x_j).
$$

The existing dynamic two-dimensional grid consumes these continuous `f64`
coordinates unchanged. Its bounds expand and trigger rebucketing, never
shrink, and its fattest-cell eviction policy still enforces the per-bucket sample
cap. With the default `bucket_count=16`, the grid is $4\times4$. Buckets
choose which observations survive; they are not local predictors and are not
queried during estimation.

After insertion and eviction, the fit is rebuilt from the retained **raw**
features. For each axis, population mean and standard deviation are computed
with stable Welford accumulation:

$$
\mu_j=\frac1n\sum_i x_{ij},\qquad
\sigma_j=\sqrt{\frac1n\sum_i(x_{ij}-\mu_j)^2},\qquad
z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}.
$$

An axis is inactive when

$$
\sigma_j\le 10^{-12}\max(1,|\mu_j|).
$$

Its standardized value and coefficient are zero. If both axes are inactive,
the regression remains unready. Otherwise, the existing nonnegative
active-set linear regression is fitted on standardized features with a free
intercept and its existing slope-only regularized fallback. A fit retains its
own means, scales, active-axis flags, standardized coefficients, and intercept;
prediction transforms raw features with that same snapshot. Inputs must be
exactly two finite, nonnegative values. Extrapolation and the positive nonidle
prediction floor of `1e-6` milliseconds remain supported.

With the default five-observation minimum, the fitted model also needs a
positive coefficient on at least one active feature axis. An intercept-only
fit remains unready. The existing special handling for an explicitly configured
lower minimum is preserved when there are too few observations to identify
the active slopes.

Each workload bucket runs this pipeline independently. Refitting one bucket
leaves the other buckets and other predictor instances unchanged.
