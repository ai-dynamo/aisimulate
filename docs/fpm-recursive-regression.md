<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Recursive regression in the forward-pass performance model

AISimulate updates its regression fit when a retained observation is inserted
or evicted. It maintains small, centered sufficient statistics and solves the
existing standardized, nonnegative linear regression from those statistics.
The usual fitting path therefore does not scan the retained observations.
Numerically ambiguous cases still use a fresh batch fit on the exact retained
data.

The model has two features and one fitted plane per workload store. Recursive
updates make maintaining that plane cheaper; sample selection and the plane's
ability to describe a workload remain separate concerns. Periodic statistics
rebuilding is disabled by default (`None` in Rust/Python, `null` in JSON).
Numerical recovery and conservative batch fallbacks remain enabled.

## What one store learns

An observation contains two finite, nonnegative raw features and a positive,
finite observed latency in milliseconds:

$$
x = [x_0, x_1], \qquad y = \text{observed latency in ms}.
$$

`RegressionIterationFeatures` computes $x_0$ as critical attention work: a
maximum over attention-DP ranks after composing each rank's attention score.
It computes $x_1$ as global FFN/MoE work: a weighted sum of fresh prefill tokens
and/or decode requests across ranks, according to the worker's role. In
particular, the aggregated role uses the sum of prefill tokens **plus decode
requests**. Feature weights are fixed at construction. The
[feature reference](../python/aisimulate/docs/fpm/aic-fpm-regression-design.md#reference-symbols-and-features)
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

## The fitted plane and standardization

For the current $n$ retained observations, the population mean and scale of
feature $j$ are

$$
\mu_j = \frac{1}{n}\sum_i x_{ij}, \qquad
s_j = \sqrt{\frac{1}{n}\sum_i (x_{ij}-\mu_j)^2}.
$$

The denominator is $n$, not $n-1$. A feature is active only when its scale is
finite and

$$
s_j > 10^{-12}\max(1, |\mu_j|).
$$

An active feature is transformed to $u_j=(x_j-\mu_j)/s_j$. An inactive feature
has transformed value zero and fitted coefficient zero. The target stays in
milliseconds; it is not divided by a target standard deviation.

The fitted prediction is

$$
\widehat y = a + b_0 u_0 + b_1 u_1, \qquad b_j \ge 0.
$$

The intercept $a$ is free, including negative values. The slopes are
constrained to be nonnegative. For active axes, the corresponding raw plane
has coefficients

$$
\theta_j = b_j/s_j, \qquad
\theta_{\mathrm{intercept}} = a - \sum_{j\text{ active}} \theta_j\mu_j.
$$

Thus nonnegative standardized slopes also mean nonnegative raw slopes.
The stored intercept $a$ is the value at the retained feature means, not
necessarily the raw plane's value at the origin. Each fit keeps the means,
scales, and active-axis flags that produced its coefficients, so prediction
uses a consistent snapshot.

## Statistics that support insertion and eviction

Combine the two raw features and target into a three-dimensional vector:

$$
z_i = [x_{i0}, x_{i1}, y_i]^\mathsf T.
$$

`RecursiveFit` maintains a count $n$, mean vector $m$, and symmetric centered
scatter matrix $C$:

$$
m = \frac{1}{n}\sum_i z_i, \qquad
C = \sum_i (z_i-m)(z_i-m)^\mathsf T
  = \begin{bmatrix} C_{xx} & C_{xy} \\ C_{xy}^\mathsf T & C_{yy}\end{bmatrix}.
$$

Here $C_{xx}$ is two-by-two, $C_{xy}$ has two entries, and $C_{yy}$ is a scalar.
They contain everything needed to standardize the features, build the normal
equations, and score a candidate's residual error. Keeping centered scatter
avoids routinely subtracting two large raw quantities such as
$\sum x_i^2-n\mu^2$ to recover a small variance. Floating-point roundoff is
still possible, especially during removal.

### Adding one observation

Let $z$ be the incoming observation and $\delta=z-m$ before insertion. Then

$$
\begin{aligned}
n' &= n+1,\\
m' &= m + \frac{\delta}{n+1},\\
C' &= C + \frac{n}{n+1}\delta\delta^\mathsf T.
\end{aligned}
$$

To see the scatter update, recenter the old points at $m'$. Their centered
deviations sum to zero, leaving $C+n(m-m')(m-m')^\mathsf T$. Adding the new
point's deviation $(z-m')(z-m')^\mathsf T$ gives the factor $n/(n+1)$ above.
The first insertion initializes $m=z$ and $C=0$.

The code uses the equivalent Welford expression
$\delta(z-m')^\mathsf T$. For off-diagonal entries it averages the two
coordinate orders and mirrors the result to retain symmetry.

### Removing one retained observation

For $n>1$, let $z$ be the actual evicted observation and $\delta=z-m$ before
removal. Inverting the insertion identity gives

$$
\begin{aligned}
n' &= n-1,\\
m' &= m - \frac{\delta}{n-1},\\
C' &= C - \frac{n}{n-1}\delta\delta^\mathsf T.
\end{aligned}
$$

Removing the last observation clears the count, mean, and scatter. A full
store performs insertion followed by removal; the removal formula uses the
mean **after insertion**. The sampler returns the actual evicted value, so
the update does not assume that the globally oldest observation was removed.

For example, start with $(x_0,x_1,y)=(0,0,1)$ and $(2,0,5)$:

$$
n=2,\quad m=[1,0,3]^\mathsf T,\quad
C=\begin{bmatrix}2&0&4\\0&0&0\\4&0&8\end{bmatrix}.
$$

Inserting $(4,0,9)$ gives $m'=[2,0,5]^\mathsf T$ and
$C'=\left[\begin{smallmatrix}8&0&16\\0&0&0\\16&0&32\end{smallmatrix}\right]$.
Removing $(0,0,1)$ then gives mean $[3,0,7]^\mathsf T$ and the original
scatter matrix. The remaining points have feature scale $s_0=1$, so their
line is $7+2(x_0-3)=1+2x_0$. This illustrates the statistics; with only two
retained points, the default minimum of five still prevents a ready fit.

## Solving the constrained fit from the statistics

For active axes, let $D=\operatorname{diag}(s_j)$ and define

$$
G=D^{-1}C_{xx}D^{-1}, \qquad h=D^{-1}C_{xy}.
$$

These are the standardized feature Gram matrix and feature-target cross
products. Centered features have zero sum, so the intercept separates from
the slopes. For a selected subset of fitted axes $S$, the normal equations are

$$
\begin{bmatrix}n&0\\0&G_{SS}\end{bmatrix}
\begin{bmatrix}a\\b_S\end{bmatrix}
=
\begin{bmatrix}n m_y\\h_S\end{bmatrix}.
$$

In exact arithmetic, the free intercept is $a=m_y$. Axes outside $S$ have
coefficient zero. With two active features the implementation considers four
faces of the nonnegative constraint: no slopes, only the first slope, only
the second slope, and both slopes. With one active feature there are two.
It solves each candidate, discards candidates with negative fitted slopes,
and chooses the smallest residual sum of squares (SSE). It does not simply
clip a negative unconstrained coefficient to zero: the remaining coefficients
must be refitted on that face.

For a hand-derived example, take all nine combinations $x_0,x_1\in\{0,1,2\}$
and labels $y=20-2x_0+3x_1$. The independent feature columns make the
nonnegative optimum $\widehat y=18+3x_1$: the forbidden negative term is
replaced by its mean $-2$. This plane predicts 30 at $(100,4)$, regardless of
the first coordinate. A production test anchors this behavior independently
of the batch comparator.

### Ridge is a singular-solve fallback

Each candidate first attempts an unregularized solve. Only a failed solve
retries with a penalty on its fitted slopes:

$$
\left(H+\lambda\operatorname{diag}(0,1,\ldots,1)\right)c=r,
\qquad
\lambda=\texttt{singular\_ridge\_scale}\,
\max\!\left(1,\sum_k |H_{kk}|\right).
$$

$H$ is that candidate's normal-equation matrix, including the intercept
entry $n$. `singular_ridge_scale` defaults to $10^{-9}$. The intercept remains
unpenalized, and the penalty applies in standardized feature coordinates.
The configured scale also reaches every batch fallback.

This preserves the existing algorithm: some candidates can use ridge and
others can use ordinary least squares, and all are ranked by **unpenalized
SSE**. It is not a single always-regularized ridge objective. The small linear
solver treats a pivot below $10^{-12}$ in absolute magnitude as singular.

### Scoring without another pass over the observations

For a candidate $a,b$, with omitted slopes filled with zeros, expansion of
$\sum_i(y_i-a-b^\mathsf T u_i)^2$ yields

$$
\operatorname{SSE}
=C_{yy}-2b^\mathsf T h+b^\mathsf T G b+n(a-m_y)^2.
$$

The centered cross terms disappear because $\sum_i u_i=0$ and
$\sum_i(y_i-m_y)=0$. This avoids a residual scan for each face. Its terms can
nearly cancel for a good fit, so the implementation checks negative/nonfinite
scores and close candidate scores before trusting the selected face.

## Numerical recovery, readiness, and prediction

The recurrence and fresh batch fitting are equivalent in exact arithmetic;
floating-point accumulation orders differ. The implementation retains the
original batch fitter to handle decisions that are sensitive to that
difference, including:

- feature spread close to the active-axis threshold;
- nearly collinear active features, including $1-\rho^2\le10^{-8}$;
- fitted slopes close to the zero constraint boundary;
- failed or nonfinite solves, invalid SSE, and numerically tied candidate SSE.

The batch fallback rebuilds standardized observations from the retained raw
rows and scores direct residuals, using the same ridge configuration. It
returns a fit for the current update without replacing the incremental
statistics or resetting their mutation clock.

A **statistics rebuild** is different: it reaccumulates the means and scatter
from the retained raw rows and resets the clock. Recovery rebuilds run when
the statistics become nonfinite, a scatter diagonal becomes negative, or a
downdate removes almost all previously represented variance. These guards
remain active when periodic rebuilding is disabled. They reduce numerical
risk but do not constitute a bound on roundoff for every possible stream.

A store is ready only if it has enough retained observations (five by default)
and at least one varying feature. After selecting a fit, it also requires a
positive slope, except for the existing low-observation case $n\le d$, where
$d$ is the number of varying axes. This count is not the rank of the feature
matrix. With the default minimum, an intercept-only solution is unready. A
store can lose readiness after an eviction.

For a valid nonempty query, a ready store transforms features using its fit
snapshot. If transformation and evaluation are finite, it returns the prediction
floored at $10^{-6}$ ms; otherwise it returns no prediction. The floor
is applied after fitting; training SSE uses raw predictions. A valid iteration
with no scheduled work returns zero through the outer model and adds no
observation. A cold workload store returns no prediction even if another store
is ready.

## Configuring periodic rebuilding

The canonical field is
`estimator_config.fpm_regression.fit.rebuild_interval`. Rust owns its default
and validation:

| Setting | Behavior |
|---|---|
| Omitted, Rust/Python `None`, or JSON `null` | No periodic statistics rebuild |
| Positive integer $I$ | Rebuild after at least $I$ retained-sample mutations |
| Zero, negative, noninteger, or boolean | Field-specific configuration error |

An accepted insertion counts as one mutation and an actual eviction counts
as another. The check runs once after the complete transaction. A full-store
replacement therefore contributes two mutations; it can cross an odd interval
by one before the clock resets to zero. Interval 1 produces one rebuild per
accepted transaction, including when that transaction inserts and evicts.
Rejected input and spatial rebucketing do not advance the clock. Each actual
statistics rebuild resets it; an ordinary batch fallback does not. The clock
saturates instead of overflowing when left running indefinitely.

For an explicitly configured interval of 4096 and capacity 64, with no earlier
recovery rebuild, the first periodic rebuild is after 2080 accepted inputs:
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
option for this control. See the [canonical API](core-api.md#choosing-a-forward-pass-api)
for selection, diagnostics, migration, and saved-configuration behavior.

## Sample retention and total tuning cost

The sampler uses $\log(1+x_j)$ coordinates to choose retention cells. Fitting
uses the raw $x_j$. Grid bounds expand with incoming observations and do not
shrink on eviction. If the store exceeds its capacity, it evicts the oldest
observation in a most-populated cell. Hash-map ordering can affect tied
choices. Each retained observation has weight one; an evicted observation
has weight zero. This is spatially balanced retention, not a FIFO window or
exponential forgetting.

Let $M$ be retained capacity, $B$ the bucket-map storage scanned to select an
eviction, and $k$ the population of the selected cell. With two fixed features:

| Work in a tuning update | Cost |
|---|---|
| Feature validation and reduction | Linear in the supplied rank count |
| Centered insertion/removal and small constrained fit | Constant in $M$ on the healthy path |
| Finding a fattest retention cell | Scan of the bucket map, $O(B)$ |
| Removing the oldest row from that cell's vector | $O(k)$, up to $O(M)$ |
| Rebucketing after dynamic bounds expand | $O(M)$ |
| Statistics rebuild or batch-fit fallback | $O(M)$ |
| Stored observations | $O(M)$, plus constant-sized fitting state |

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

For fixed features $\phi=[1,x_0,x_1]^\mathsf T$, ordinary RLS often maintains
the inverse normal matrix $P=(\sum_i\phi_i\phi_i^\mathsf T)^{-1}$. For one
insertion, its rank-one update is

$$
P'=P-\frac{P\phi\phi^\mathsf T P}{1+\phi^\mathsf T P\phi}.
$$

That is useful when the feature coordinates and fitting objective stay fixed.
Our implementation uses recursive **sufficient statistics** instead. Means
and scales change with the retained observations, active axes can change,
the nonnegative slope face can change, and ridge is conditional on a failed
candidate solve. Recomputing the small constrained solve from updated scatter
preserves those behaviors directly, without maintaining and transforming an
inverse for each possible constraint face. It still performs recursive
least-squares estimation; it does not use a forgetting factor or the classic
inverse-matrix coefficient update.

## Speed measurements

The following CPU measurements time one **retained-sample update**:
`BucketedRegression::add_observation` from already computed raw features.
This includes validation, log-grid coordinates, retention/eviction, recursive
statistics, and fitting, including any recovery or batch fallback the data
triggers. It excludes rank-level feature extraction, Python/JSON overhead,
prediction queries, and GPU execution. It is not an end-to-end serving speedup.

The baseline uses the original full-bucket update from `8e763d4` and the same
sampler and batch-fitting arithmetic. The recursive path uses the production
fitter with `rebuild_interval=None`, including all numerical guards. Both
paths use a four-by-four retention grid, minimum five observations, and
`singular_ridge_scale=1e-9`.

Times are medians of 21 paired trials; brackets show the minimum–maximum
trial range. Speedup is the batch median divided by the recursive median.

| Input data | Capacity | Batch update (µs) | Recursive update (µs) | Speedup |
|---|---:|---:|---:|---:|
| Toy | 64 | 7.078 [7.006–7.136] | 0.812 [0.788–0.823] | 8.7× |
| Toy | 8,192 | 736.844 [717.487–750.102] | 1.120 [1.034–1.257] | 657.9× |
| H200 / MiniMax-M2.7 / prefill | 64 | 7.093 [6.995–7.299] | 0.818 [0.781–0.840] | 8.7× |
| H200 / MiniMax-M2.7 / prefill | 8,192 | 738.917 [710.320–774.915] | 1.770 [1.706–1.889] | 417.6× |
| B200 / MiniMax-M3 NVFP4 / decode | 64 | 6.909 [6.815–7.027] | 0.807 [0.780–0.831] | 8.6× |
| B200 / MiniMax-M3 NVFP4 / decode | 8,192 | 737.600 [720.991–754.895] | 2.288 [2.195–2.434] | 322.3× |

Outside the timed region, **6,150 fits and 50,835,900 raw/floored
prediction comparisons passed**, with identical retained rows and no readiness,
active-feature, or slope-constraint failures. The audit covers the warm state
and every update in the 1,024-row tail for each case. It checks retained points
and five boundary/extrapolation queries, including 1.5 times the feature maxima.
Maximum absolute prediction difference was **2.724e-10 ms**,
within `1e-8 + 1e-8 * abs(batch prediction)` ms. Maximum scaled raw-coefficient
and training-SSE differences were 1.065e-12 and
1.549e-14, respectively, using `abs(batch-recursive) / max(1, abs(batch))`.
This tests recovery of the batch fit on these rows, not an unbounded-stream
roundoff guarantee.

Measured on 2026-09-18 on an Apple M5 Pro, macOS 26.6.2, arm64,
18 logical CPUs, with Rust 1.97.1. The isolated benchmark compiles the production
regression and sampling modules with `opt-level=3`, `target-cpu=native`, and
`codegen-units=1`; timing is single-threaded. The recursive arithmetic is
unchanged from `14ac37e4`.
These are fresh measurements of the production path with the disabled setting,
not the earlier prototype timings.

Each case warms the model with 8,192 observations, then measures a distinct
1,024-observation tail. Each method starts from a clone of the same retained
state, preserving identical sampling decisions. There are 21 paired trials,
alternating which method runs first, after one discarded trial per method.
Each trial repeats that tail from the same warm state 16 times at capacity 64
or four times at capacity 8,192.
Cloning, setup, and destruction are outside the timer; every updated fit is
passed to `black_box`. This measures repeated warm windows, not continuous
long-stream behavior. Reported ranges describe timing variation for the fixed
retained state; new processes can resolve sampling ties differently.

Inputs are shuffled with seed `0xA15120260917` for this CPU comparison:

- **Toy:** independent log-uniform features and
  $y=\max(0.01,3+3\times10^{-5}x_0+0.025x_1+\epsilon)$ ms, with
  $\epsilon\sim N(0,0.03^2)$ and generator seed `0xA15170`.
- **H200 prefill:** MiniMax-M2.7, vLLM 0.25.1, TP4/DP1; 9,505 measured
  offline shape-grid rows. Feature coordinates use the balanced-request proxy.
  Source `fpm_forward_perf.parquet` at AISimulate revision
  `b790939feb2fc681c5f1c3b55c23e4388a56dd6e`, collector cell
  `fpm-574efc8a51db0a0b`.
- **B200 decode:** MiniMax-M3 NVFP4, vLLM 0.28.0, TP4/DP1; the 310,492
  pure-decode rows from recorded serving iterations in
  `nvidia/aisimulate-fpm-dataset` revision
  `5487a4599a7fbc012c07bcd3699754bdf4a8bef7`.

Each case uses 9,216 unique source row IDs for warmup and its measured tail.
The shuffle and repeated trials are a timing protocol; they do not measure
chronological adaptation or held-out prediction accuracy.

## Source map and validation anchors

| Source | Responsibility |
|---|---|
| [model.rs](../crates/core/src/perfmodel/fpm/model.rs) | `RegressionIterationFeatures`, workload routing, independent `RegressionStores`, observation targets |
| [samples.rs](../crates/core/src/perfmodel/fpm/samples.rs) | Retention grid and `SampleInsertion` carrying the actual eviction |
| [regression.rs](../crates/core/src/perfmodel/fpm/regression.rs) | `BucketedRegression`, original batch fit, small linear solvers, prediction floor |
| [regression/recursive.rs](../crates/core/src/perfmodel/fpm/regression/recursive.rs) | Centered updates, constrained fit from statistics, guards, rebuild clock |
| [estimator.rs](../crates/core/src/perfmodel/fpm/estimator.rs) | Rust-owned `RegressionFitConfig` defaults and validation |

Tests compare recursive and batch fits on identical retained rows, including
evictions, changing scales, inactive axes, and nondefault ridge strengths.
Independent plane and nonnegative-boundary examples check expected numerical
answers. Scheduling tests cover custom intervals, disabled periodic rebuilding,
recovery, clone independence, and counter saturation. These checks complement
measured workload validation; they do not establish a universal bound on
long-run floating-point drift.
