<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Recursive regression in the forward-pass performance model

AISimulate updates centered sufficient statistics when a retained observation
is inserted or evicted. By default it also solves the standardized, nonnegative
linear regression after each accepted update transaction. Opt-in lazy updates
defer coefficient publication while retaining every accepted statistical update.
The usual fitting path therefore does not scan the retained observations.
Numerically ambiguous cases use one full rebuild of statistics and batch
coefficients on the exact retained data.

The default linear model has two features and one fitted plane per workload
store. Recursive updates make maintaining that plane cheaper; sample selection
and the plane's ability to describe a workload remain separate concerns. Periodic statistics
rebuilding is disabled by default (`None` in Rust/Python, `null` in JSON).
Numerical recovery and conservative batch fallbacks remain enabled.
The optional `fit.linear` controls support one to six fitted features, signed
slopes, and lazy updates. Independently, `sampling.axes` and `bins_per_axis`
configure one to six retention dimensions. The defaults remain two A/M axes,
a 4×4 grid, capacity 64 per store, nonnegative fitting, and eager updates.

The optional `fit.kind: spline` model learns two or three knots per feature axis.
It uses the same feature extraction, workload stores, and retained observations.
The two-feature equations below describe the default linear model; the next
section explains the spline's additional state.

## Learned-knot spline regression

The canonical configuration selects `fpm_regression.fit.kind: spline` and places
its controls in `fpm_regression.fit.spline`. Rust supplies two knots per axis and
adaptive searches with `window: 16`, `trigger: 8`, `tolerance: 0.05`,
`absolute_tolerance_ms: 1.0`, and `cooldown: 64`. A periodic alternative is
`search: {kind: periodic, step: 64}`. See the
[configuration examples and validation rules](online-regression.md#selecting-linear-or-spline-regression).

The fit is a sum of two continuous piecewise-linear functions plus a free
intercept. Nonnegative segment slopes allow the slope to decrease across a knot
while keeping each feature's contribution monotone. There are no products of
the two feature axes. Constant axes and axes with few distinct retained values
use fewer knots. A knot search optimizes positions on the retained data; it does
not require physical workload boundaries in advance.

Each store owns one sampler shared by its spline and linear fallback. Both fits
consume the same insertion and the actual eviction. Between knot searches,
the spline updates sufficient statistics for its fixed basis and refits its
coefficients. A search changes that basis and rebuilds its statistics from the
retained samples. Numerical recovery and batch fallback remain available; knot
searches are counted separately from fixed-basis rebuilds and batch fallbacks.
Relocating knots resets the spline's fixed-basis mutation clock, while the shared
linear fit keeps its existing statistics rebuild clock.

Startup requires `max(32, min_observations)` accepted observations, with at least
`min_observations` still retained. The default linear fit can serve after five
observations when it has a usable load signal. With default minimums, periodic
searches happen at accepted counts 32, 64, 128, and onward. Adaptive searches
after the initial one require the cooldown and enough excessive errors in the
latest window; they cannot occur before accepted count 96 with the defaults.
Search clocks count accepted observations per store, unlike `rebuild_interval`,
which counts insertion and eviction separately. Queries and rejected targets
advance neither clock.

An adaptive error compares the spline's raw, unclamped prediction before update
with the new observed latency. It is excessive if its magnitude is greater than
`max(absolute_tolerance_ms, tolerance * observed_ms)`. The monitor uses the spline
prediction even outside the retained domain and clears its window after every
search. This prevents the serving guard from hiding a spline's poor extrapolation
from the search policy. A full window is not required, and the observation that
satisfies the cooldown need not itself have an excessive error if the rolling
count still meets the trigger.

Serving first requires an available prediction from the shared linear fit for
the query. It then uses the spline within the current retained raw-feature
bounding box and the linear fit outside it. An unready spline also falls back
to that linear prediction. If the linear prediction is unavailable, neither
path serves a prediction, including inside retained bounds. The store's `ready`
flag requires a usable linear fit; `spline.ready` describes only the spline
component and can be true while the store is unready. Neither path borrows
another workload store's training data. Both paths retain the positive latency
floor. Per-store diagnostics expose spline initialization, readiness, accepted
count, knot searches, last search count, numerical rebuilds, and batch fallbacks;
the linear-only diagnostic shape is unchanged. Configuration serialization
records controls, not trained state.

## What one store learns

By default, an observation contains two finite, nonnegative raw features and
a positive, finite observed latency in milliseconds:

```math
x = [x_0, x_1], \qquad y = \text{observed latency in ms}.
```

`RegressionIterationFeatures` computes $x_0$ as critical attention work: a
maximum over attention-DP ranks after composing each rank's attention score.
It computes $x_1$ as global FFN/MoE work: a weighted sum of fresh prefill tokens
and/or decode requests across ranks, according to the worker's role. In
particular, the aggregated role uses the sum of prefill tokens **plus decode
requests**. Feature weights are fixed at construction. The
[feature reference](online-regression.md#reference-symbols-and-features)
gives the full formulas.

Tuning obtains $y$ from the maximum finite, positive rank wall time, converted
from seconds to milliseconds. Prediction uses scheduled workload fields and
does not consume the observed wall time.

Each logical store owns its retained samples, statistics, fit, and rebuild
clock:

| Worker role | Independent regression stores |
|---|---|
| Prefill | `pure_prefill` |
| Decode | `pure_decode` |
| Aggregated | `pure_decode`, `contains_locally_mixed`, `cross_rank_aggregated`, `pure_prefill` |

A query selects its workload store. Training another store does not change
its answer. The default capacity of 64 applies to **each store**; the default
four-by-four retention grid inside a store does not create 16 fitted planes.

`fit.linear.feature_axes` selects the fitted coordinates independently from
`sampling.axes`. Both lists must contain one to six distinct supported axes;
each sampling axis has its own positive bin count. All retention coordinates
use `log1p` of the selected feature value, including when that feature itself
contains a logarithm. This transformation does not alter the fitted feature.
The catalog and a complete lazy-update example are in the
[canonical API reference](online-regression.md#linear-features-and-lazy-coefficient-updates).
Features requiring aligned scheduled-request lengths fail if those inputs are
absent; missing extrema or cross moments are not inferred from aggregate counters.
Spline fitting and its retention grid remain restricted to the default A/M axes.

## The fitted plane and standardization

For the current $n$ retained observations, the population mean and scale of
feature $j$ are

```math
\mu_j = \frac{1}{n}\sum_i x_{ij}, \qquad
s_j = \sqrt{\frac{1}{n}\sum_i (x_{ij}-\mu_j)^2}.
```

The denominator is $n$, not $n-1$. A feature is active only when its scale is
finite and

```math
s_j > 10^{-12}\max(1, |\mu_j|).
```

An active feature is transformed to $`u_j=(x_j-\mu_j)/s_j`$. An inactive feature
has transformed value zero and fitted coefficient zero. The target stays in
milliseconds; it is not divided by a target standard deviation.

The fitted prediction is

```math
\widehat y = a + b_0 u_0 + b_1 u_1, \qquad b_j \ge 0.
```

The intercept $a$ is free, including negative values. The slopes are
constrained to be nonnegative by default; `fit.linear.non_negative: false`
permits signed slopes. For active axes, the corresponding raw plane
has coefficients

```math
\theta_j = b_j/s_j, \qquad
\theta_{\mathrm{intercept}} = a - \sum_{j\text{ active}} \theta_j\mu_j.
```

Thus nonnegative standardized slopes also mean nonnegative raw slopes.
The stored intercept $a$ is the value at the retained feature means, not
necessarily the raw plane's value at the origin. Each fit keeps the means,
scales, and active-axis flags that produced its coefficients, so prediction
uses a consistent snapshot.

## Statistics that support insertion and eviction

Combine the two raw features and target into a three-dimensional vector:

```math
z_i = [x_{i0}, x_{i1}, y_i]^\mathsf T.
```

`RecursiveFit` maintains a count $n$, mean vector $m$, and symmetric centered
scatter matrix $C$:

```math
m = \frac{1}{n}\sum_i z_i, \qquad
C = \sum_i (z_i-m)(z_i-m)^\mathsf T
  = \begin{bmatrix} C_{xx} & C_{xy} \\ C_{xy}^\mathsf T & C_{yy}\end{bmatrix}.
```

Here $`C_{xx}`$ is two-by-two, $`C_{xy}`$ has two entries, and $`C_{yy}`$ is a scalar.
They contain everything needed to standardize the features, build the normal
equations, and score a candidate's residual error. Keeping centered scatter
avoids routinely subtracting two large raw quantities such as
$`\sum x_i^2-n\mu^2`$ to recover a small variance. Floating-point roundoff is
still possible, especially during removal.

### Adding one observation

Let $z$ be the incoming observation and $`\delta=z-m`$ before insertion. Then

```math
\begin{aligned}
n' &= n+1,\\
m' &= m + \frac{\delta}{n+1},\\
C' &= C + \frac{n}{n+1}\delta\delta^\mathsf T.
\end{aligned}
```

To see the scatter update, recenter the old points at $m'$. Their centered
deviations sum to zero, leaving $`C+n(m-m')(m-m')^\mathsf T`$. Adding the new
point's deviation $`(z-m')(z-m')^\mathsf T`$ gives the factor $n/(n+1)$ above.
The first insertion initializes $m=z$ and $C=0$.

The code uses the equivalent Welford expression
$`\delta(z-m')^\mathsf T`$. For off-diagonal entries it averages the two
coordinate orders and mirrors the result to retain symmetry.

### Removing one retained observation

For $`n>1`$, let $z$ be the actual evicted observation and $`\delta=z-m`$ before
removal. Inverting the insertion identity gives

```math
\begin{aligned}
n' &= n-1,\\
m' &= m - \frac{\delta}{n-1},\\
C' &= C - \frac{n}{n-1}\delta\delta^\mathsf T.
\end{aligned}
```

Removing the last observation clears the count, mean, and scatter. A full
store performs insertion followed by removal; the removal formula uses the
mean **after insertion**. The sampler returns the actual evicted value, so
the update does not assume that the globally oldest observation was removed.

For example, start with $(x_0,x_1,y)=(0,0,1)$ and $(2,0,5)$:

```math
n=2,\quad m=[1,0,3]^\mathsf T,\quad
C=\begin{bmatrix}2&0&4\\0&0&0\\4&0&8\end{bmatrix}.
```

Inserting $(4,0,9)$ gives $`m'=[2,0,5]^\mathsf T`$ and
$`C'=\left[\begin{smallmatrix}8&0&16\\0&0&0\\16&0&32\end{smallmatrix}\right]`$.
Removing $(0,0,1)$ then gives mean $`[3,0,7]^\mathsf T`$ and the original
scatter matrix. The remaining points have feature scale $s_0=1$, so their
line is $7+2(x_0-3)=1+2x_0$. This illustrates the statistics; with only two
retained points, the default minimum of five still prevents a ready fit.

## Solving the constrained fit from the statistics

For active axes, let $`D=\mathrm{diag}(s_j)`$ and define

```math
G=D^{-1}C_{xx}D^{-1}, \qquad h=D^{-1}C_{xy}.
```

These are the standardized feature Gram matrix and feature-target cross
products. Centered features have zero sum, so the intercept separates from
the slopes. For a selected subset of fitted axes $S$, the normal equations are

```math
\begin{bmatrix}n&0\\0&G_{SS}\end{bmatrix}
\begin{bmatrix}a\\b_S\end{bmatrix}
=
\begin{bmatrix}n m_y\\h_S\end{bmatrix}.
```

In exact arithmetic, the free intercept is $a=m_y$. Axes outside $S$ have
coefficient zero. With two active features the implementation considers four
faces of the nonnegative constraint: no slopes, only the first slope, only
the second slope, and both slopes. With one active feature there are two.
It solves each candidate, discards candidates with negative fitted slopes,
and chooses the smallest residual sum of squares (SSE). It does not simply
clip a negative unconstrained coefficient to zero: the remaining coefficients
must be refitted on that face.

For a hand-derived example, take all nine combinations $`x_0,x_1\in\{0,1,2\}`$
and labels $y=20-2x_0+3x_1$. The independent feature columns make the
nonnegative optimum $`\widehat y=18+3x_1`$: the forbidden negative term is
replaced by its mean $-2$. This plane predicts 30 at $(100,4)$, regardless of
the first coordinate. A production test anchors this behavior independently
of the batch comparator.

### Ridge is a singular-solve fallback

Each candidate first attempts an unregularized solve. Only a failed solve
retries with a penalty on its fitted slopes:

```math
\left(H+\lambda\,\mathrm{diag}(0,1,\ldots,1)\right)c=r,
\qquad
\lambda=\texttt{singular\_ridge\_scale}\,
\max\!\left(1,\sum_k |H_{kk}|\right).
```

$H$ is that candidate's normal-equation matrix, including the intercept
entry $n$. `singular_ridge_scale` defaults to $`10^{-9}`$. The intercept remains
unpenalized, and the penalty applies in standardized feature coordinates.
The configured scale also reaches every batch fallback.

This preserves the existing algorithm: some candidates can use ridge and
others can use ordinary least squares, and all are ranked by **unpenalized
SSE**. It is not a single always-regularized ridge objective. The small linear
solver treats a pivot below $`10^{-12}`$ in absolute magnitude as singular.

### Scoring without another pass over the observations

For a candidate $a,b$, with omitted slopes filled with zeros, expansion of
$`\sum_i(y_i-a-b^\mathsf T u_i)^2`$ yields

```math
\mathrm{SSE}
=C_{yy}-2b^\mathsf T h+b^\mathsf T G b+n(a-m_y)^2.
```

The centered cross terms disappear because $`\sum_i u_i=0`$ and
$`\sum_i(y_i-m_y)=0`$. This avoids a residual scan for each face. Its terms can
nearly cancel for a good fit, so the implementation checks negative/nonfinite
scores and close candidate scores before trusting the selected face.

## Numerical recovery, readiness, and prediction

The recurrence and fresh batch fitting are equivalent in exact arithmetic;
floating-point accumulation orders differ. The implementation retains the
original batch fitter to handle decisions that are sensitive to that
difference, including:

- feature spread close to the active-axis threshold;
- nearly collinear active features, including $`1-\rho^2\le10^{-8}`$;
- fitted slopes close to the zero constraint boundary;
- failed or nonfinite solves, invalid SSE, and numerically tied candidate SSE.

All three linear rebuild reasons—periodic maintenance, numerical recovery,
and a conservative batch fallback—use the same **full rebuild**. It
reaccumulates the means and scatter and recomputes batch coefficients from the
same retained rows, scores direct residuals using the configured ridge, and
resets the mutation clock. It never publishes a fresh batch fit while retaining
the old incremental statistics. For `fit.kind: linear`, a finite, identifiable
candidate with all feature slopes zero is rejected while preserving the previous
serving snapshot, including its coefficients and normalization. The statistics
are still rebuilt and the mutation clock still resets. Without a previous fit,
the store remains unready. Other unusable batch results clear readiness. This
exception does not change spline fitting or its linear fallback.

Recovery rebuilds run when the statistics become nonfinite, a scatter diagonal
becomes negative, or a
downdate removes almost all previously represented variance. These guards
remain active when periodic rebuilding is disabled. They reduce numerical
risk but do not constitute a bound on roundoff for every possible stream.

A new fit requires enough retained observations (five by default)
and at least one varying feature. The default nonnegative fit also requires a
positive slope, except for the existing low-observation case $`n\le d`$, where
$d$ is the number of varying axes. This count is not the rank of the feature
matrix. With the default minimum, an intercept-only candidate cannot become the
serving model, but rejection preserves any previous linear serving model. Signed
fits allow negative slopes; identifiable all-zero candidates are still rejected.
A store can still lose readiness after an eviction if another fit requirement
fails, such as having no varying features.

For a valid nonempty query, a ready store transforms features using its fit
snapshot. If transformation and evaluation are finite, it returns the prediction
floored at $`10^{-6}`$ ms; otherwise it returns no prediction. The floor
is applied after fitting; training SSE uses raw predictions. A valid iteration
with no scheduled work returns zero through the outer model and adds no
observation. A cold workload store returns no prediction even if another store
is ready.

## Lazy coefficient updates

`fit.linear.update_policy` is `always` by default. The optional
`error_threshold` policy monitors the raw, unclipped prediction before the
new target enters retention or statistics. For positive observed latency $y$,
an error is excessive only when

```math
|\widehat y-y| > \max(\text{absolute\_tolerance\_ms},\;\text{relative\_tolerance}\,y).
```

Equality is acceptable. The monitor retains the most recent `window` accepted
observation flags with finite prior predictions and requests a fit when its excessive count is at least
`trigger` and at least `cooldown` accepted observations have passed since the
last successful fit. It need not fill the whole window; the current error may
be acceptable while older excessive flags still meet the trigger. Prediction
calls and rejected observations do not advance these counts.

The first `startup_observations` accepted rows are eager, with a default of 10.
An unready or numerically unusable model bypasses the lazy gate and retries.
Successful fits clear the monitor; failed or rejected fits do not. Retaining the
previous snapshot after an all-zero candidate does not reset the error window or
cooldown, so subsequent observations can trigger another attempt. A periodic
rebuild or numerical recovery also overrides deferral. These operations do not have a
fixed 256-row minimum gap, and there is no separate batch-fallback interval.

Retention, actual evictions, and centered statistics advance on every accepted
row. A skipped solve leaves coefficients, means, scales, and active-axis flags
together as one prediction snapshot. The eager default avoids the extra
monitoring prediction. Lazy gating reduces optional solve work; it does not
remove retention or maintenance costs or promise an error bound.

## Configuring periodic rebuilding

The canonical field is
`estimator_config.fpm_regression.fit.rebuild_interval`. Rust owns its default
and validation:

| Setting | Behavior |
|---|---|
| Omitted, Rust/Python `None`, or JSON `null` | No scheduled full rebuild |
| Positive integer $I$ | Rebuild after at least $I$ retained-sample mutations |
| Zero, negative, noninteger, or boolean | Field-specific configuration error |

An accepted insertion counts as one mutation and an actual eviction counts
as another. The check runs once after the complete transaction. A full-store
replacement therefore contributes two mutations; it can cross an odd interval
by one before the clock resets to zero. Interval 1 produces one rebuild per
accepted transaction, including when that transaction inserts and evicts.
Rejected input and spatial rebucketing do not advance the clock. Every full
rebuild, including a conservative batch fallback, resets it. The clock
saturates instead of overflowing when left running indefinitely.

For an explicitly configured interval of 4096 and capacity 64, with no earlier
recovery or batch fallback, the first periodic rebuild is after 2080 accepted inputs:
64 initial insertions plus 2016 insert/evict pairs. Later rebuilds occur every
2048 accepted inputs while the store stays full. This schedule is per store,
not per prediction query or elapsed time.

Construct regression through the same public API as other estimators:

```python
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

config = ForwardPassPerfModelConfig(
    model="Qwen/Qwen3-32B",
    system="h200_sxm",
    backend="vllm",
    worker_type="aggregated",
    estimation_mode="fpm_regression",
    estimator_config={
        "fpm_regression": {
            "sampling": {"bins_per_axis": [4, 4], "max_observations": 64},
            "fit": {"rebuild_interval": None},
        },
    },
)
model = RustForwardPassPerfModel.best_available(config)
```

The explicit `None` documents the choice; omitting the field has the same
default behavior. Set it to `4096`, for example, to enable periodic rebuilding
on a newly constructed model. JSON uses `null` rather than Python's `None`.
In Rust, set the same typed field before construction:

```rust
use aisimulate_core::{
    BackendKind, EstimationMode, ForwardPassPerfModel, ForwardPassPerfModelConfig,
    ForwardPassWorkerType,
};

let mut config = ForwardPassPerfModelConfig::new(
    "Qwen/Qwen3-32B", "h200_sxm", BackendKind::Vllm, ForwardPassWorkerType::Decode,
);
config.estimation_mode = EstimationMode::FpmRegression;
config.estimator_config.fpm_regression.fit.rebuild_interval = Some(4096);
let model = ForwardPassPerfModel::best_available(config)?;
```

The setting is fixed at construction and preserved in resolved configuration
and provenance. Reloading a saved explicit interval keeps that interval;
changing the default does not overwrite it. There is no separate flat legacy
option for this control. See the [canonical API](../configuration.md#choosing-a-forward-pass-api)
for selection, diagnostics, migration, and saved-configuration behavior.

## Sample retention and total tuning cost

The sampler uses $`\log(1+x_j)`$ of its selected retention coordinates to choose
cells. The independent fit uses its selected feature values. Grid bounds expand
with incoming observations and do not
shrink on eviction. If the store exceeds its capacity, it evicts the oldest
observation in a most-populated cell. Hash-map ordering can affect tied
choices. Each retained observation has weight one; an evicted observation
has weight zero. This is spatially balanced retention, not a FIFO window or
exponential forgetting.

Let $M$ be retained capacity, $B$ the bucket-map storage scanned to select an
eviction. With two fixed features:

| Work in a tuning update | Cost |
|---|---|
| Feature validation and reduction | Linear in the supplied rank count |
| Centered insertion/removal and small constrained fit | Constant in $M$ on the healthy path |
| Finding a fattest retention cell | Scan of the bucket map, $O(B)$ |
| Removing the oldest row from that cell's `VecDeque` | $O(1)$ |
| Rebucketing after dynamic bounds expand | $O(M)$ |
| Full statistics and batch-coefficient rebuild | $O(M)$ |
| Stored observations | $O(M)$, plus constant-sized fitting state |

The table holds the feature count fixed. Linear fitting supports up to six
features: statistics scale quadratically in feature count, and nonnegative
fitting examines up to 64 constraint faces. Signed fitting uses one face.
Request-level features scan the supplied request lists once per observation;
scalar-only configurations do not require those lists.

The healthy fitter requests retained rows lazily, so it avoids both a scan
and a temporary observation copy. Periodic rebuilding, when enabled, adds
an amortized term proportional to $M/I$ for interval $I$; a full store uses
approximately two mutations per accepted input. Recovery and batch-fallback
frequency depend on the data.

Consequently, increasing capacity can make fitting cheaper than repeated
batch refits while still increasing memory, eviction, rebucketing, and
recovery work. It also changes the training sample distribution. One plane
per workload store can average over a curved or changing latency surface;
retaining more of the landscape does not turn it into a local tangent model
or a nonlinear model. Capacity remains an accuracy and retention choice as
well as a resource choice.

## Relationship to classic inverse-matrix RLS

For fixed features $`\phi=[1,x_0,x_1]^\mathsf T`$, ordinary RLS often maintains
the inverse normal matrix $`P=(\sum_i\phi_i\phi_i^\mathsf T)^{-1}`$. For one
insertion, its rank-one update is

```math
P'=P-\frac{P\phi\phi^\mathsf T P}{1+\phi^\mathsf T P\phi}.
```

That is useful when the feature coordinates and fitting objective stay fixed.
Our implementation uses recursive **sufficient statistics** instead. Means
and scales change with the retained observations, active axes can change,
the nonnegative slope face can change, and ridge is conditional on a failed
candidate solve. Recomputing the small constrained solve from updated scatter
preserves those behaviors directly, without maintaining and transforming an
inverse for each possible constraint face. It still performs recursive
least-squares estimation; it does not use a forgetting factor or the classic
inverse-matrix coefficient update.

### Linear features and lazy coefficient updates

Configure linear fits under `estimator_config.fpm_regression.fit.linear`.
`feature_axes` defaults to `[attention, moe]`, `non_negative` defaults to `true`,
and `update_policy` defaults to `{kind: always}`. Setting `non_negative: false`
allows signed slopes; the intercept is always unconstrained. Fitting and
retention may select different ordered lists of one to six distinct axes.
The supported names are `attention`, `moe`, `n`, `E`, `P`, `maxE`, `maxP`,
`minP`, `P2`, `F`, `nE`, `logF`, `meanE`, `meanP`, `cvE2`, `cvP2`, `logN`,
`n2`, and `logP`. Features use scheduled work only. Request-list features require
the corresponding aligned request lengths; unavailable input is rejected, not
reconstructed from aggregate counts. These controls do not change the workload
store selected from all active attention-DP ranks.

`attention`, `moe`, `n`, `logN`, and `n2` need only the existing scheduled scalar
counters. Other axes require both optional `scheduled_requests.extend_lengths`
and `scheduled_requests.past_kv_lengths`, each an array of unsigned 64-bit
integers. Both arrays must have one entry per scheduled request and identical
lengths. Their sums may differ from aggregate token counters because backends
can use different counting conventions, such as padded prefill tokens. Omitted
arrays do not add null fields to existing serialized metrics. Existing Gym inputs without
these lists can evaluate the scalar axes; they cannot qualify list-derived ones.
Prediction needs the request lists only when fitted axes use them. Tuning also
requires them when retention axes use request-level features.

This example uses three retention dimensions and a different three-feature fit:

```yaml
estimator_config:
  fpm_regression:
    sampling:
      axes: [attention, moe, n]
      bins_per_axis: [2, 4, 2]
      max_observations: 128
    fit:
      linear:
        feature_axes: [attention, moe, logN]
        non_negative: false
        update_policy:
          kind: error_threshold
          relative_tolerance: 0.05
          absolute_tolerance_ms: 0.1
          window: 8
          trigger: 2
          cooldown: 4
          startup_observations: 10
```

Lazy updating is opt-in. Every accepted observation still updates retention and
centered statistics. Before admitting it, the model compares its **raw, unclipped
prior prediction** with the positive measured latency `y`. An error is excessive
only when `abs(prediction - y) > max(absolute_tolerance_ms, relative_tolerance * y)`;
equality does not trigger. The rolling monitor counts the latest `window`
accepted observations with finite prior predictions. A fit is requested when at least `trigger` flags are
excessive and at least `cooldown` accepted observations have passed since the
last successful fit. A full window is unnecessary, and an observation with a
small error can satisfy the cooldown while earlier excessive flags remain.

For `fit.kind: linear`, a finite, identifiable candidate whose feature weights
are all zero is rejected without replacing the previous serving snapshot. Its
coefficients and normalization remain together; a store with no previous fit
stays unready. This safeguard applies to eager and lazy updates, including full
rebuilds. Other failures, such as insufficient data or an unavailable numerical
solution, still clear the serving fit. The default nonnegative constraint and
its existing underdetermined-fit exception are unchanged. Signed fits may use
negative weights, but an identifiable all-zero result is still rejected.
Spline fitting and its linear fallback retain their existing behavior.

The first `startup_observations` accepted rows are eager (default 10). An unready
or unusable model keeps trying to fit. A successful fit clears the monitor;
a failed or rejected fit does not. Periodic full rebuilds and numerical recovery
override lazy deferral. Between fits, coefficients and the feature means/scales used
with them remain one prediction snapshot. Eager defaults do not collect this
monitor or compute its extra prediction. Lazy thresholds are not a guarantee
on future prediction error.

Both tolerances must be finite and nonnegative. Window, trigger, cooldown, and
startup count must be positive integers, with `trigger <= window`. Unknown axes,
duplicate axes, invalid grid shapes, and incompatible policy fields fail before
estimator selection. `fit.linear` is rejected with `fit.kind: spline`;
the spline fit and retention axes remain `[attention, moe]`.

## Selecting linear or spline regression

```python
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

config = ForwardPassPerfModelConfig(
    model="Qwen/Qwen3-32B", system="h200_sxm", backend="vllm",
    worker_type="decode", estimation_mode="fpm_regression",
    estimator_config={"fpm_regression": {"fit": {"kind": "spline"}}},
)
model = RustForwardPassPerfModel.best_available(config)
```

The `linear` alias normalizes to `standardized_nnls`. Spline controls belong
under `fit.spline`: `knots_per_axis` is 2 or 3; `search` is either
`{kind: periodic, step: 64}` or the adaptive policy described above.
Periodic `step` and adaptive `window`, `trigger`, and `cooldown` must be positive,
with `trigger <= window`; adaptive `tolerance` is positive finite and
`absolute_tolerance_ms` nonnegative finite. Policies cannot mix fields.
Spline requires `sampling.max_observations >= 32`, default A/M fit and retention
axes, and no `fit.linear` block. Evaluate fit choice against deployment-specific
observations; a nonlinear fit does not guarantee lower prediction error.

Configuration serialization saves controls, not observations, knots, or fitted
state. Reconstructing a saved configuration creates a cold model. Offline CLI,
Sweeper, and Replay reject an untrained regression even when its configuration
parses successfully.

## Implementation

The authoritative implementation is
[`fpm/`](../../../crates/core/src/perfmodel/fpm/); the
[Python wrapper](../../../python/aisimulate/src/aisimulate_core/sdk/rust_engine_step.py)
normalizes telemetry and calls the same native estimator. Queued requests do
not supply scheduled work, and timing observations must belong to the same
worker/deployment identity. Regression cannot discover a changed backend,
precision, graph-capture policy, or topology from two aggregate load features.

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
$`H_d>0`$. Cached Prefill KV creates attention work only when fresh Prefill
work exists:

```math
\widetilde H_d=H_d\,\mathbf{1}[P_d>0].
```

The default aggregate A/M feature path estimates Prefill attention pairs using a
balanced-request approximation. Under that approximation, each of the $N_d$
requests has cached length $H_d/N_d$ and newly computed length $P_d/N_d$.
For $`P_d>0`$, metric validation guarantees $`N_d>0`$, and the total
attention-pair estimate for rank $d$ is

```math
\begin{aligned}
Q_d
&=N_d\left[
\left(\dfrac{H_d}{N_d}\right)\left(\dfrac{P_d}{N_d}\right)
+\dfrac{(P_d/N_d)(P_d/N_d+1)}{2}
\right]\\
&=\dfrac{H_dP_d}{N_d}+\dfrac{P_d^2}{2N_d}+\dfrac{P_d}{2}.
\end{aligned}
```

When $P_d=0$, define $Q_d=0$. Thus $Q_d$ already includes the factor $N_d$:
it estimates total Prefill attention-pair work on the rank, not work for one
request. Callers must not multiply it by $N_d$ again. $N_d$ is the number of
Prefill requests on rank $d$, not the attention-DP size; multiple ranks are
reduced only by the role-specific maximum and sum below.

All roles share the axis order

```math
x=[\text{critical attention},\ \text{global FFN/MoE}].
```

With $`\alpha`$ the KV-attention weight, $`\beta`$ the Prefill
attention-pair weight, and $`\gamma`$ the tokenwise FFN/MoE weight, the exact
features are:

```math
x_P=
\left[
\max_d\left(\alpha\widetilde H_d+\beta Q_d\right),
\ \gamma\sum_d P_d
\right],
```

```math
x_D=
\left[
\alpha\max_d K_d,
\ \gamma\sum_d B_d
\right],
```

```math
x_A=
\left[
\max_d\left(\alpha(\widetilde H_d+K_d)+\beta Q_d\right),
\ \gamma\sum_d(P_d+B_d)
\right].
```

The reductions for one and multiple attention-DP ranks are shown below.
Unsubscripted symbols in the `attention_dp = 1` column refer to the sole rank.

| Worker | `attention_dp = 1` | `attention_dp > 1` |
|---|---|---|
| Prefill | $`x[0]=\alpha\widetilde H+\beta Q`$<br>$`x[1]=\gamma P`$ | $`x[0]=\max_d(\alpha\widetilde H_d+\beta Q_d)`$<br>$`x[1]=\gamma\sum_d P_d`$ |
| Decode | $`x[0]=\alpha K`$<br>$`x[1]=\gamma B`$ | $`x[0]=\alpha\max_d K_d`$<br>$`x[1]=\gamma\sum_d B_d`$ |
| Aggregated | $`x[0]=\alpha(\widetilde H+K)+\beta Q`$<br>$`x[1]=\gamma(P+B)`$ | $`x[0]=\max_d[\alpha(\widetilde H_d+K_d)+\beta Q_d]`$<br>$`x[1]=\gamma\sum_d(P_d+B_d)`$ |

The default regression remains two-dimensional at every attention-DP size. Critical
attention uses a maximum across ranks, while global FFN/MoE work uses a sum.

The Aggregated maximum is taken after composing the entire rank-local
attention score; the implementation must not combine independent maxima from
different ranks. Counters are converted to `f64` before multiplication or
cross-rank summation, and derived features must be finite and nonnegative.

The options are construction-time knobs and all default to `1.0`:

| Formula | `estimator_config.features` field |
|---|---|
| $`\alpha`$ | `attention_kv_weight` |
| $`\beta`$ | `prefill_attention_pair_weight` |
| $`\gamma`$ | `ffn_token_weight` |

- $`\alpha`$ scales KV-token-related attention work for every role:
  $`\widetilde H`$ for Prefill, $K$ for Decode, and $`\widetilde H+K`$ for
  Aggregated. It is not a Prefill-only weight.
- $`\beta`$ scales only the Prefill attention-pair estimate $Q$. It does not
  represent Decode attention.
- $`\gamma`$ scales global tokenwise FFN/MoE work: $P$ for Prefill, $B$ for
  Decode, and $P+B$ for Aggregated.

They must be finite and strictly positive when regression is constructed. They
are ignored by a successful native-only model. Changing a weight requires
constructing a new model, because it changes the feature and retention-cell
coordinates. To preserve learned history, replay the original per-rank FPM
observations into the new model.

New configuration uses the nested feature-weight names above. The legacy
EngineConfig/options migration adapter retains the old flat names and exact
nonfinite JSON sentinels. These weights are ignored by a selected native model;
regression construction rejects nonpositive or nonfinite weights.


## Workload routing and worker ownership

Each estimator instance belongs to one worker and fixed deployment identity.
The caller must route observations to that instance; a `worker_id` metadata field
does not select another instance internally. Predict before tuning on the new
observation when evaluating online accuracy.

For an aggregated predictor, classify all active attention-DP ranks: any locally
mixed rank selects `contains_locally_mixed`; otherwise separate prefill-only and
decode-only ranks select `cross_rank_aggregated`; a single active phase selects
its pure store. Idle ranks do not change the label. Cached prefill KV without
fresh prefill tokens does not create prefill activity. Summary readiness can
mean that one store is ready while a query to another store still returns `None`.
