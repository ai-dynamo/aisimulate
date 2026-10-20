<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Online binned GBDT: mathematics

This document defines the online update equations and hyperparameter semantics
for checkpoint-backed learned forward-pass models. The original learned-model
design and evaluation for [PR 280](https://github.com/ai-dynamo/aisimulate/pull/280)
remain in [design.md](design.md).

Construct the model through `RustForwardPassPerfModel.best_available(config)`
with explicit `estimation_mode: fpm_learned`. Frozen adaptation preserves the
forest and can apply hybrid calibration; online adaptation updates the binned
trees from feedback. Both use complete iteration telemetry. The current engine
replay, engine CLI and Sweeper phase interfaces cannot supply those metrics and
reject learned execution.

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
is controlled by the caller. Expiry is evaluated when the store fits a batch,
relative to that batch's newest observation ID; a prediction query does not
expire rows or publish another forest.

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
the whole retained fitting population. Lazy fitting trades some derivative
freshness for cheaper insertions and evictions; it does not eliminate periodic
full work. Track original history count, stored representative count, recent
and pending counts, update/fallback counts and retained bytes separately in
`learned_online_diagnostics()`. Host update cost and predicted serving latency
are different quantities; measure preparation, queries, fitting and memory at
the intended history size and feedback cadence.

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
