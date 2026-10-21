<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Online binned GBDT and causal selection: mathematics

This document defines the online update equations and hyperparameter semantics
for checkpoint-backed learned forward-pass models and the causal selector
that combines online GBDT with PR 493 hybrid calibration. The original learned-model
design and evaluation for [PR 280](https://github.com/ai-dynamo/aisimulate/pull/280)
remain in [design.md](design.md).

Construct the model through `RustForwardPassPerfModel.best_available(config)`
with explicit `estimation_mode: fpm_learned`. Frozen adaptation preserves the
forest and can apply hybrid calibration; online adaptation updates the binned
trees from feedback. The combined selector keeps both experts independent
and selects using their earlier errors. All three use complete iteration
telemetry. The current engine replay, engine CLI and Sweeper phase interfaces
cannot supply those metrics and reject learned execution.

For setup and usage, see the [learned-model workflow](learned-forward-pass-model.md).
The [configuration reference](configuration.md#learned-forest-and-online-fitting-controls)
provides field names, defaults and valid ranges alongside these equations.

## Online binned tree adaptation

Online adaptation starts from a fitted forest and its training checkpoint. It
preserves the ordered feature ABI, fitted bin thresholds, baseline, tree count
and each tree's reserved node capacity. Feedback updates leaf values and,
optionally, the splits within those trees. The ordinary inference artifact
contains enough information to predict, but not enough to continue fitting:
training rows, fitted bins and fitting settings belong to the checkpoint.

The canonical control is `estimator_config.fpm_learned.adaptation: online`.
This standalone online estimator requires `correction.enabled: false`; its
reported latency is the adaptive forest's prediction. Frozen adaptation remains
the separate option for applying hybrid calibration to an unchanged forest.
Combined `causal_selector` adaptation uses these same online fitting rules for
its adaptive expert and separate hybrid calibration for its frozen expert.
The [configuration table](configuration.md#learned-forest-and-online-fitting-controls)
defines every online parameter, its default, units and valid range.

### Target, derivatives and leaf values

Let $y_i>0$ be an observed latency in milliseconds. A `log_ms` forest learns
$z_i=\log y_i$ and predicts

$$
F(x)=b+\sum_{t=1}^{T} f_t(x),\qquad
\widehat y(x)=\max(10^{-6},\exp F(x)).
$$

For an `ms` artifact, use $z_i=y_i$ and the identity output transform.
Inference applies the same $10^{-6}$-millisecond floor after that transform.
The floor is an inference safeguard; fitting derivatives use the unclipped
forest prefix in target space. Online fitting supports squared error.
Squared log error measures multiplicative
errors symmetrically in log space; it is not the same objective as MAPE.
The unweighted derivative for observation $i$ at the prefix before tree $t$ is
$g_i=F_{<t}(x_i)-z_i$, with Hessian $h_i=1$. For row weights $w_i$, a region has

$$
G=\sum_i w_i g_i,\qquad H=\sum_i w_i.
$$

With checkpoint learning rate $\eta$ and L2 penalty $\lambda$, its fitted leaf
contribution is

$$
v^*=-\eta\frac{G}{H+\lambda}.
$$

The stored contribution already includes the learning rate; prediction must
not multiply it by $\eta$ again. The separate `update_step` $\rho$ controls
replacement of an existing leaf:

$$
v_{\mathrm{new}}=v_{\mathrm{old}}+\rho(v^*-v_{\mathrm{old}}).
$$

The default $\rho=1$ publishes the fitted value. Smaller values are supported
only for `leaf_only`, where old and new leaf regions are identical. An empty
region has fitted value zero; a partial update moves its old value toward zero.
Neither update mode changes the forest baseline.
An optional `mutable_tree_tail` freezes the earlier trees; those trees still
contribute to every later tree's prefix prediction.

### Historical evidence and recent feedback

Each store retains immutable training history and a bounded FIFO of online
observations. `recent_window` limits the latter, not the former. An optional
`recent_max_age_observations` also removes recent rows whose global accepted
observation IDs have expired. Its inclusive age limit counts observations
across the model's stores; it is not a wall-clock timeout. Observation cadence
is controlled by the caller. Expiry is evaluated when the store fits a batch
or the caller invokes `flush_updates`, relative to the latest accepted
observation across all stores. An explicit flush also removes expired fitted
rows from stores without pending feedback. If only never-fitted pending rows
expire, they are discarded without publishing a new forest. Prediction queries
and checkpoint saves do not expire rows or publish another forest.

With $H_0$ original historical observations and $R$ retained recent observations,
unit row weights are the default. An explicit `history_weight` $\beta$ instead
sets the historical fraction of total loss mass after a support ramp. For ramp
length $K$, define

$$
s=\begin{cases}
1,&K=0,\\
\min(R/K,1),&K>0.
\end{cases}
\qquad
q_R=(1-s)\frac{R}{H_0+R}+s(1-\beta).
$$

The weights of each historical and recent observation are then

$$
w_H=\frac{(H_0+R)(1-q_R)}{H_0},\qquad
w_R=\frac{(H_0+R)q_R}{R}.
$$

Total weight remains $H_0+R$. If either population is empty, the implementation
uses unit weights. Before the ramp completes, weights interpolate from the
population-proportional balance toward the requested mass. The ramp uses
retained support $R$, so expiry can reduce that support. If $K$ exceeds the
recent-window capacity, the ramp remains partial and the final requested mass
is never reached at that capacity. A setting of
`history_weight: 0.1` does not retain only 10% of the historical rows: it retains
them all and changes their contribution to the fitting objective. Preserving
total weight also matters when $\lambda>0$, because regularization competes
with the absolute Hessian mass.

### Complete-bin grouping of immutable history

For feature $j$, let its fixed thresholds define bin index $b_j(x_j)$; equality
belongs to the lower bin, matching the tree's `<=` comparison. NaN has a separate
missing bin. A historical group is identified by the complete vector
$(b_1(x_1),\ldots,b_d(x_d))$, including missing bins. Every split allowed by the
checkpoint routes all members of a group identically.

With `history_bin_grouping: true`, store one representative feature vector,
logical multiplicity $c_g$ and transformed-target sum $S_g=\sum_{i\in g}z_i$
per group. At a common prefix prediction $p_g$, its unweighted contributions
are exactly

$$
G_g=c_g p_g-S_g,\qquad H_g=c_g.
$$

Applying $w_H$ to these quantities gives the historical group contribution to
any leaf or split. This follows by expanding
$\frac12\sum_{i\in g}(p_g-z_i)^2$ and dropping the term independent of $p_g$.
Replacing a group by a unit-weight mean target would change the objective and
minimum leaf support; the implementation preserves both target sums and counts.
`min_samples_leaf` is checked against original logical counts, independently of
loss weighting. Original observation IDs and their identity digest remain
available for checkpoint validation and diagnostics.

Grouping leaves the seed predictor unchanged. The number of stored historical
representatives is at most the original history size, with savings when many
rows share all fitted bins. It preserves the mathematical objective, but does
not promise bit-identical floating-point sums: representative accumulation can
change summation order. Missing values and every selected feature must be part
of the grouping key for this equivalence to hold.

### Split gain and bounded structural changes

For a proposed partition into left and right children, the regularized
second-order gain is

$$
\Delta=\frac12\left[
\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}
-\frac{G_P^2}{H_P+\lambda}\right].
$$

Children must meet the checkpoint's minimum logical leaf count and have positive
weighted Hessian mass. Histogram prefix sums evaluate numeric thresholds;
missing values are considered on either side. The missing-only partition is
also a candidate, including for features with no ordinary numeric thresholds.

`split_sampling_rate` chooses a deterministic subset of fitted numeric
thresholds per feature. The existing split remains eligible even when not drawn
in that subset. The RNG depends on `seed`, tree/node identity and rebuilding
generation. Candidate ties retain the incumbent when possible, with stable
ordering for other ties.

`split_rank_tolerance` $\tau$ governs when an existing valid split must change.
If $N$ candidates are valid, its acceptable rank is

$$
r_{\max}=\max(1,\lceil\tau N\rceil),\qquad
r_{\mathrm{current}}=1+\#\{c:\Delta_c>\Delta_{\mathrm{current}}\}.
$$

Gain comparisons use a numerical tolerance. The region is rebuilt when the
incumbent's rank exceeds $r_{\max}$; an invalid child population bypasses that
rank guard. Thus $\tau=0$ requires a best-ranked incumbent, whereas increasing
$\tau$ permits some lower-ranked splits to remain. This is a rank criterion,
not a relative improvement threshold. A terminal region may split when a
positive-gain candidate and reserved capacity permit it.

Structural updates reuse the affected subtree's original node arena and obey
the checkpoint depth limit. They can change split feature, threshold and
missing direction, or collapse an invalid partition, without adding trees or
expanding the seed tree's capacity. `leaf_only` keeps all splits unchanged.

### Batches, derivative refresh and publication

A valid feedback row receives an observation ID and enters its store's pending
batch. Reaching `update_every` starts fitting; `flush_updates()` processes a
smaller pending batch explicitly. Recent FIFO membership is selected from the
existing recent rows and pending rows, so a batch larger than the window keeps
only its newest eligible rows. Pending rows are not a second training history.
Prediction queries only read the last successfully published forest and never
consume labels or trigger fitting.

In eager mode, the fitter visits trees in order. It recomputes derivatives for
all retained historical representatives and recent rows using the already
updated preceding trees, then updates the current tree. Reusing derivatives
from the old complete forest would implement a different update.

Lazy mode retains cached derivative sums and histograms. It subtracts the
stored derivative contribution of each evicted recent row and adds derivatives
for incoming rows. Previously retained derivatives stay at their earlier
values, so lazy fitting is an approximation between refreshes. If a structural
change is required, the lazy attempt rolls back and performs a full eager fit.
An eager refresh also occurs at `eager_refresh_interval` fitting batches;
setting it to 1 makes every batch eager.

A fitting attempt stages the candidate forest and population changes before
publication. Invalid or nonfinite state does not publish a partial candidate;
pending rows remain available for explicit retry. A failed pending batch must
be retried rather than allowing unlimited further admissions. This transaction
is the fitting batch, not a promise that an arbitrary list of caller updates
across multiple stores is one transaction.

### Persistence and host cost

A seed checkpoint records the exact fitted observations, bin mapper and fitting
settings; an online checkpoint additionally records mutable trees, retained and
pending rows, derivative caches, counters and sampling state. Loading checks
identity and file integrity before accepting the state. Resume with the same
model/hardware/backend/topology/worker identity and fitting controls. Saving
does not implicitly flush; pending rows must survive restart. Exporting only
the inference JSON preserves predictions at that instant, not the information
needed for subsequent online continuation.

Ordinary prediction remains a feature build and tree traversal; it does not
scan training history or recent feedback. Updates pay for retained rows,
histograms and any rebuilt regions. Grouping reduces historical traversal from
original rows to complete-bin representatives, while eager refresh still visits
the whole retained fitting population. With `mutable_tree_tail`, eager fitting
stages and publishes replacement training state only for that tail; frozen
histograms, split candidates and region arrays retain their existing allocations.
The baseline-plus-frozen-tree prefix for each immutable historical row is cached
in the original accumulation order. Eager updates copy these cached values and
traverse the frozen trees only for recent rows. The cache uses eight bytes per
stored historical row when a frozen prefix exists; unrestricted updates allocate
no cache. Grouping and checkpoint loading rebuild this derived state without
changing the checkpoint format. The candidate forest still copies the small
prediction arrays for validation before publication.
Lazy fitting trades some derivative freshness for cheaper insertions and
evictions; it does not eliminate periodic
full work. Track original history count, stored representative count, recent
and pending counts, update/fallback counts and retained bytes separately in
`learned_online_diagnostics()`. Host update cost and predicted serving latency
are different quantities; measure preparation, queries, fitting and memory at
the intended history size and feedback cadence.

## Hybrid calibration with a causal selector

`adaptation: causal_selector` keeps the calibrated forest and online forest as
independent experts. Calibration can adjust the scale of a trained response
surface while preserving its structure; tree adaptation can change that
surface. Their errors need not improve together. The selector compares their
recent observed errors in the workload region where each comparison applies.
It makes no claim that either expert dominates across deployments or that
online selection improves every sequence.

### Experts and independent state

Let $w$ identify a model instance owned by one worker, $s$ a learned forest
store, and $q$ a calibration query kind. The selector key is

$$
k=(w,s,q).
$$

The forest store $s$ is selected from the complete rank list and is one of
`pure_prefill`, `pure_decode`, `contains_locally_mixed`, or
`cross_rank_aggregated`. The query kind $q$ is prefill, decode or mixed,
selected by the existing correction feature extractor's critical rank.
It uses prefill tokens for a prefill query, decode requests and decode KV
tokens for decode, and prefill tokens and decode KV tokens for mixed. The
critical rank maximizes the sum of those coordinates. These query coordinates
are separate from the learned forest's full feature vector $x$.

There are up to 12 store/query keys per worker instance. Some combinations
cannot occur for a particular worker role, but no key borrows another key's
error evidence. Each forest store also owns its own three correction stores.
For example, a cross-rank aggregated iteration can select a prefill critical
rank; its evidence must not warm the pure-prefill forest's selector or
calibrator. The online fitter still pools all admitted query kinds assigned
to the same forest store. A `worker_id` field in telemetry does not multiplex
independent models; callers own one instance per worker.

For forest features $x$ and calibration coordinates $u$, define the expert
predictions immediately before an observation's label is admitted:

$$
C_k(x,u)=c_k(u)F_{0,s}(x),\qquad A_s(x)=F_{\mathrm{online},s}(x).
$$

Here $F_{0,s}$ is the immutable seed predictor in milliseconds, including its
output transform and prediction floor; $c_k$ is the PR493 hybrid calibration
factor; and $F_{\mathrm{online},s}$ is the most recently published adaptive predictor.
Both forests start from the same training checkpoint. Only the adaptive forest
changes its trees. Calibration receives ratios against $F_{0,s}$ for its
entire lifetime, so publishing an adaptive forest cannot invalidate or reset
those ratios. The adaptive prediction has no correction multiplier.

### The stable expert's hybrid factor

For admitted latency $y>0$, calibration stores the absolute ratio

$$
r=\operatorname{clip}_{[l,h]}\left(\frac{y}{F_{0,s}(x)}\right),
$$

where the default lower and upper factor bounds are $l=0.5$ and $h=2.0$.
A disabled bound leaves that side unclamped. The denominator is always the
uncorrected frozen prediction, never the selected prediction or a previously
corrected value.

Let $D_q$ be the fixed grid's configured domain for query kind $q$. Its upper
limits default to 8,192 prefill tokens, 512 decode requests and 2,000,000 decode
KV tokens, using the axes above; all endpoints are included. The hybrid factor
is

$$
c_k(u)=\begin{cases}
c^{\mathrm{fixed}}_k(u),&u\in D_q,\\
c^{\mathrm{dynamic}}_k(u),&u\notin D_q.
\end{cases}
$$

The fixed factor is the median ratio in the query's raw-coordinate bucket once
the fixed store has `min_observations` retained samples. It is 1 for an empty
bucket or an unready store. This neutral in-domain result is preserved even
if the dynamic policy has enough samples. The hybrid boundary never depends
on selector errors, store readiness or expanding dynamic bounds.

The dynamic policy bins $\log(1+u)$ with bounds expanded by admitted feedback.
Let $m_G$ be the median ratio in its recent FIFO, or 1 while that FIFO has fewer
than `min_observations` samples. When the primary set is ready and the query
has a populated local bucket containing $n_b$ ratios with median $m_b$, its
factor is

$$
c^{\mathrm{dynamic}}_k(u)=\operatorname{clip}_{[l,h]}
\exp\left[\frac{n_b}{n_b+20}\log m_b+
\frac{20}{n_b+20}\log m_G\right].
$$

Otherwise it uses $m_G$, including outside the current dynamic bounds. The
constant 20 is the calibration shrinkage support, not the selector warmup or
the online history ramp. Both fixed and dynamic policies receive the same
frozen-base ratio; the fixed policy rejects samples outside $D_q$.
The dynamic policy also rejects a ratio that overflows or underflows before
clipping. Offering one admitted label to both policies therefore does not imply
that both retain a sample; their existing PR493 admission rules remain in force.

For correction capacity $M$ and readiness threshold $m$, each policy has its
own $M$-slot budget. Within the dynamic budget, the recent FIFO has
$\max(\lfloor M/4\rfloor,m)$ slots and the primary set has the remainder.
Primary retention balances occupied buckets; dynamic samples expire after
$2M$ admissions to that calibration key. The two dynamic sets can contain
the same observation. Correction capacity, online recent-window capacity,
and selector comparison count are three different quantities.

Canonical `correction.sampling.bins_per_axis: [2, 2]` means four total buckets:
four prefill intervals or a 2-by-2 decode/mixed grid. This is the legacy
`bucket_count: 4` setting, sometimes abbreviated `b4`; `[4, 4]` means 16 total
buckets and is the API default. An explicit `M=32`, `b4`, `[0.5, 2.0]` profile
changes calibration retention and geometry without changing the online loss
or selector equations. See the [configuration example](learned-forward-pass-model.md#hybrid-calibration-with-a-causal-selector)
for its exact canonical fields.

For this profile, $M=32$ and `min_observations=5` allocate eight recent FIFO
slots and 24 primary slots within dynamic calibration, plus a separate
32-slot fixed budget for each store/query key. `b4` uses the 2-by-2 grid above.
`W64` retains up to 64 online rows per forest store; the selector's
$N_0=10$ warmup instead counts comparisons per store/query key. With
$\alpha=0.025$, an old error contribution halves in about 27.38 comparisons
of that key. Online `history_weight=0.1` assigns 90% of loss mass to recent
rows after the 32-row retained-support ramp completes; all historical rows
remain represented, rather than retaining only 10% of them.

### Error evidence and switching

Index only valid admitted comparisons of key $k$ by $n=1,2,\ldots$. For expert
$e\in\{C,A\}$, use its prediction before feedback to compute fractional APE:

$$
\ell_{k,n}^{e}=\left|\frac{\widehat y_{k,n}^{e}}{y_{k,n}}-1\right|.
$$

The loss is not clipped; 0.1 means 10% error. Predictions and labels must be
finite and strictly positive, and the loss and updated averages must remain
finite. Invalid or overflowing comparisons are rejected before admission.
Calibration bounds constrain ratios, not this selector loss. Online fitting
continues to minimize weighted squared error in the checkpoint target space;
it does not fit APE.

With `alpha` $\alpha$, each key and expert maintains

$$
E_{k,1}^{e}=\ell_{k,1}^{e},\qquad
E_{k,n}^{e}=(1-\alpha)E_{k,n-1}^{e}+\alpha\ell_{k,n}^{e}
\quad(n>1).
$$

The first comparison initializes the average directly, avoiding a zero-loss
startup prior. Larger $\alpha$ follows new errors more quickly. For
$0<\alpha<1$, an old contribution halves every
$\log(1/2)/\log(1-\alpha)$ comparisons of that key; $\alpha=1$ uses only the
latest loss. Queries, unadmitted labels and comparisons belonging to other
keys do not advance this clock. There is no wall-clock decay or separate
freshness reset for a rarely observed key.

Let $a_{k,n-1}$ be the active expert before comparison $n$, initially $C$,
and let $b$ be the other expert. After updating both averages and incrementing
the key's count, the challenger qualifies exactly when

$$
Q_{k,n}=\left[n\ge N_0\right]\;\land\;
\left[E_{k,n}^{b}<(1-\delta)E_{k,n}^{a_{k,n-1}}\right],
$$

where $N_0$ is `warmup` and $\delta$ is `margin`. The test is symmetric: after
online becomes active, the calibrated expert must satisfy the same relative
advantage to switch back. Equality does not qualify, even when the margin is
zero. If both errors are zero, the active expert remains selected.

For `persistence` $P$, increment a key's streak when $Q_{k,n}$ holds and reset
it to zero otherwise. Once the streak reaches $P$, set $a_{k,n}=b$ and reset
the streak. Otherwise retain the active expert. A streak is consecutive in
that key's comparisons, so unrelated labels neither extend nor interrupt it.
The resulting choice serves subsequent queries of the key. The current
observation's prediction used $a_{k,n-1}$ and cannot benefit from its own label.

Defaults are $\alpha=0.025$, $N_0=10$, $\delta=0.025$ and $P=1$. The online
profile name `W64` means `recent_window: 64`; its FIFO capacity is independent
of the selector's `warmup: 10` comparisons per key. For example,
with active EWMA 0.20, a margin of 0.025 requires challenger error strictly
below 0.195. This is a 2.5% relative improvement over active error, not a
2.5-percentage-point reduction in MAPE. Warmup limits sparse early decisions;
margin requires an advantage; persistence requires repeated evidence. These
controls do not establish a statistical confidence guarantee.

### Admission, failures and supported lifecycle

For each chronological iteration, the caller first estimates latency, then
supplies the complete observation to `tune_with_fpms` before any later update.
Tuning computes both expert predictions from the state before feedback and
stages the selector transition. A successful online admission then offers the
label to calibration and commits selector evidence once, regardless of which
expert is active. Thus an inactive adaptive expert keeps learning, and an
inactive stable expert keeps calibrating. Feedback from future or unadmitted
observations has no role in the gate.

Online admission and forest publication are separate events. If a label enters
the pending batch but fitting fails, it is still offered to calibration once,
and selector evidence advances once using the pre-feedback predictions. The
published adaptive forest remains unchanged. `flush_updates()` retries the pending fitting work
without repeating that label's calibration or selector comparison. An
observation rejected before admission changes neither history. Re-submitting
an already admitted label would be another observation, not a retry protocol.

The API computes comparison predictions during tuning; it does not retain
per-query tickets. Delayed or reordered feedback therefore does not reproduce
the original predictions when intervening updates occurred. Use synchronous
chronological estimate/tune pairs and separate instances for separate workers.
The prediction method itself is read-only with respect to these histories.

Combined construction accepts a training seed and starts fresh stable
calibration, online recent state and selector evidence. A standalone online
runtime checkpoint does not contain the combined histories and is rejected.
`save_online_checkpoint()` and `export_learned_artifact()` reject combined
models until a format can preserve both experts, every correction store and
every selector key together. `learned_selector_diagnostics()` exposes selection
state; `learned_online_diagnostics()` exposes the adaptive fitting state. Neither
diagnostic snapshot is a resumable model.

The selector adds constant-size error state per key, while the stable forest
and its correction stores add their own memory and update costs. Feedback
needs both expert predictions as well as calibration and online fitting work.
Measure these costs separately from serving-latency accuracy. Hyperparameter
selection and evaluation must use disjoint chronological data; evaluating all
labels is useful for reporting, but only the declared admitted labels may
update either expert or the selector.

## Algorithm provenance

The online updater is inspired by Lin, Chung, Lao and Zhao,
[Online Gradient Boosting Decision Tree: In-Place Updates for Efficient Adding/Deleting Data](https://arxiv.org/abs/2502.01634v1).
The Rust implementation and independent reference tests were written for
AISimulate; no implementation code from the paper's repository was imported.

The fixed seed bins, baseline and tree capacity, weighted historical/recent
populations, scalar squared loss, transactional publication and periodic eager
refresh are this implementation's choices. The paper's theoretical guarantees
and accuracy results are not claimed for these adaptations. The
[source-import record](../../../benchmarks/evidence/fpm_online_publication_v1/README.md)
identifies the local implementation snapshot and its provenance.
