<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Learned forward-pass model configuration

This page documents `estimator_config.fpm_learned` and
`estimator_config.selector`, and the optional
`estimator_config.temporary_mixed_composition` policy for the canonical performance-model API.
Model, hardware, backend, topology, worker identity and shared correction controls are described
in the [performance-model configuration reference](../configuration.md).
For equations and hyperparameter semantics, see the
[mathematical explanation](online-gbdt-and-causal-selection.md); for setup,
feedback and checkpoint examples, see the
[learned-model workflow](learned-forward-pass-model.md).

## Selection and supported execution

The learned forest is selected only with explicit `estimation_mode: fpm_learned`;
it is outside the automatic search order. Its supported execution surface is the
standalone canonical model with complete per-iteration `ForwardPassMetrics`
for estimation and feedback. Engine replay, engine CLI prediction and
recommendation, and Sweeper timing price each pass of a learned role with one
such query over the worker's attention-DP ranks, filled in replay's telemetry
convention and carrying `extend_lengths` / `past_kv_lengths` and the length
variances when the artifact reads request, slot or variance features (see
[engine replay timing](../configuration.md#engine-replay-timing)). Replay does
not tune the model.
Supply exactly one of
`estimator_config.fpm_learned.artifact_path` and `checkpoint_path`. Frozen
adaptation accepts either source; online adaptation is the default and requires
a training or runtime checkpoint. Omitted `correction.enabled` resolves to
`false` for standalone online learned mode; explicitly enabling it is invalid. The combined
`causal_selector` adaptation requires a training seed checkpoint and enabled
hybrid correction. Learned mode
requires `fallback_policy: deny`; `allow` and the legacy `regression` policy
are invalid configurations. Learned source open, checksum, parsing and identity
errors fail construction. Learned mode also rejects active speculative decoding
(`nextn != 0` or `speculation.kind` other than `none`). An explicit
`speculation: {kind: none}` is accepted and normalized to no speculation;
it must not contain parameters or a draft checkpoint.

## Estimator namespaces

- `fpm_learned`: an explicit forest source, `adaptation` (`online` by default),
  and Rust-owned online fitting controls. Frozen forests support the existing
  correction policies. Online forests consume feedback to update their trees and
  require correction to be disabled. `causal_selector` retains a frozen hybrid
  expert alongside the online forest and selects between their predictions.
  See the field tables below.
- `selector`: `alpha` (0.4), `warmup` (5), `margin` (0.05), and
  `persistence` (1). These controls select the expert in `causal_selector`
  adaptation; they do not change either expert's fitting objective.
- `temporary_mixed_composition`: absent by default. Explicitly enables original-A
  capability checks and compatible FPM interpolation for unsupported mixed classes in
  A-only or A+C evaluation. Its calibration state is independent of the learned
  forests and selector. See the policy table below.

## Learned forest and online fitting controls

All fields below belong to `estimator_config.fpm_learned`. The defaults describe
API behavior, not an accuracy recommendation. Omitted fields in canonical
JSON/YAML select the standalone or `causal_selector` defaults listed below;
explicit values remain overrides. Configuration normalization and validation
happen in Rust; Python passes the complete configuration through.

Typed Rust configurations already contain field values. Changing `estimation_mode`
or `adaptation` retains those values; a typed online config must explicitly set
`estimator_config.correction.enabled = false`. Deserialize a sparse canonical
config to apply omission-based defaults. Explicit correction choices are never
rewritten during validation.

When every learned control has its default value, serialization omits the
`fpm_learned` block to preserve existing non-learned configurations. Any
nondefault control preserves the complete block, even without a source path.

| Field | Default | Meaning and constraints |
| --- | --- | --- |
| `artifact_path` | `null` | Path to the inference JSON artifact. Explicit learned mode requires exactly one source path. An artifact alone supports frozen adaptation. Empty paths are invalid. |
| `checkpoint_path` | `null` | Path to a seed training checkpoint or saved online checkpoint. Contains the forest, fitted bins, fitting metadata and required training state. Required for online adaptation and the causal selector; the selector accepts training seeds only. Mutually exclusive with `artifact_path`. |
| `source_sha256` | `null` | Optional 64-character hexadecimal digest of the artifact file or checkpoint manifest. Rust checks and fills this during normalization, pinning the source in the resolved configuration. Remove the old digest when intentionally changing source paths. |
| `adaptation` | `online` | `frozen` preserves tree values and topology; `online` updates the forest with correction disabled; `causal_selector` chooses between a frozen hybrid-calibrated forest and an independent online forest. The selector requires enabled hybrid correction and a training checkpoint. Set `frozen` explicitly when loading an inference artifact. |
| `update_mode` | `structural` | With online adaptation, `leaf_only` refits existing leaves; `structural` can rebuild subtrees within each seed tree's existing capacity. Neither adds trees nor relearns feature-bin thresholds. |
| `recent_window` | `64`; selector: `8` | Positive maximum number of retained online observations per learned store. FIFO eviction removes the oldest recent observations; immutable training history is additional to this budget. |
| `update_every` | `1`; selector: `4` | Positive number of pending observations in a store that triggers a fitting batch. Predictions use the last published forest until fitting succeeds; `flush_updates()` processes a partial batch explicitly. Counts admitted observations, independently of the caller's feedback schedule. |
| `derivative_refresh` | `lazy` | `eager` recomputes derivatives on the full retained fitting population. `lazy` updates cached statistics for insertions and evictions, with eager fallback when structure must change. |
| `eager_refresh_interval` | `32` | Positive maximum fitting-batch interval between full derivative refreshes in lazy mode. Counts batches, not predictions or elapsed seconds. An interval of 1 refreshes every batch. |
| `split_sampling_rate` | `0.5` | Finite fraction in `(0, 1]` of fitted numeric thresholds considered per feature for structural split search. The incumbent and missing-only boundary remain eligible. This is candidate sampling, not observation sampling. |
| `split_rank_tolerance` | `0.01` | Finite fraction in `[0, 1]` defining the acceptable rank of the incumbent split among valid candidates. Zero requires a best-ranked split; larger values can retain a lower-ranked incumbent. Invalid child populations always trigger rebuilding. |
| `seed` | `0` | Unsigned integer seed for deterministic split-candidate sampling. It does not change the offline artifact's fitted thresholds or training random state. |
| `history_weight` | `0.1`; selector: `0.25` | Optional finite historical loss-mass fraction in `[0, 1]`. `null` gives every logical observation unit weight and defaults an omitted ramp to `0`; an explicit positive ramp is invalid. A number balances total historical and recent mass independently of their population sizes after the ramp. |
| `history_weight_ramp_observations` | `32`; selector: `8` | Number of retained recent observations needed to reach the configured history/recent mass balance. Zero applies it immediately; a positive value requires `history_weight`. This is retained support, not lifetime admissions. A ramp longer than `recent_window` is valid but cannot reach the final mass balance while the window remains smaller. |
| `history_bin_grouping` | `true` | Compress immutable history by complete fitted-feature-bin vectors, preserving target sums and original multiplicities. Changes storage and fitting work, not the intended weighted objective. |
| `mutable_tree_tail` | `null` | Optional positive number of final seed trees allowed to change. `null` permits all trees. A value above the forest size also permits all trees; the prefix remains frozen. |
| `update_step` | `1.0` | Finite leaf-value displacement fraction in `(0, 1]`. Values below 1 interpolate from the existing leaf toward its fitted value and require `update_mode: leaf_only`. This differs from the seed learning rate. |
| `recent_max_age_observations` | `null` | Optional positive maximum age of a recent row in the model's global accepted-observation sequence, inclusive. Batch fitting and explicit `flush_updates()` use the latest admission across every store, including when a store has no pending rows. `null` disables age expiry. This is an observation count, not time; history is never expired. |

The seed checkpoint owns the feature order, numeric bin thresholds, target
transform, baseline, tree count, per-tree capacity, learning rate, L2 penalty,
minimum leaf support and maximum depth. They are verified when loading the
checkpoint rather than independently overridden by online configuration.
Original observation multiplicities govern minimum leaf support even when
history is grouped. See [the mathematical contract](online-gbdt-and-causal-selection.md#online-binned-tree-adaptation)
for group weights, split ranks and derivative refresh.

Normalization makes source paths absolute and records their content digest.
Checkpoint loading verifies the normalized deployment identity; an inference
artifact verifies it when `metadata.identity` is present. Legacy artifacts
without this metadata remain supported, with `identity_verified: false` in
`learned_online_diagnostics()`. Every source must still match `worker_type`.
The content digest pins source bytes; it does not establish deployment identity
for an artifact that lacks identity metadata.

For example, this standalone default profile keeps 64 recent observations and
fits after each admitted label. It retains complete historical evidence while
assigning it 10% of total loss mass once 32 recent observations are retained.
These settings are not a universal optimum.

```yaml
estimation_mode: fpm_learned
fallback_policy: deny
worker_type: decode
estimator_config:
  correction:
    enabled: false
  fpm_learned:
    checkpoint_path: models/decode_online_seed
    adaptation: online
    update_mode: structural
    recent_window: 64
    update_every: 1
    history_weight: 0.1
    history_weight_ramp_observations: 32
    history_bin_grouping: true
    derivative_refresh: lazy
    eager_refresh_interval: 32
    split_sampling_rate: 0.5
    split_rank_tolerance: 0.01
```

This fragment belongs inside the complete canonical configuration, alongside
model, hardware, backend, topology and worker identity. One model instance owns
one worker's mutable state. Reusing immutable training history between instances
does not share their online observations or fitted trees. Querying a missing
learned store returns `None`; loading a different store or choosing another
estimator is not a query-time fallback.

`learned_online_diagnostics()` reports online fitting state and resource counts.
For standalone frozen or online adaptation, `export_learned_artifact()` produces
an inference artifact containing only the current forest and artifact metadata.
It excludes calibration feedback and correction stores: reloading an export
from a calibrated frozen model starts with empty correction state, so its
predictions can differ from the exporting model's calibrated predictions.
For online adaptation, the export contains the published forest but excludes
fitting state and pending observations. `save_online_checkpoint(path)` preserves
that online continuation state in a new checkpoint directory without implicitly
flushing pending observations. Resume through the same canonical constructor with matching
identity and fitting controls. The combined selector rejects both export
methods because neither format preserves all its state.

For `causal_selector`, `learned_expert_artifacts()` inspects both forests without
fitting pending observations or changing calibration or selection. The Python
SDK returns a dictionary with exactly two keys, `stable` and `online`; the raw
native binding returns the same object as a JSON string. Each value is a
version-1 `aic_fpm_learned_forward_perf` inference artifact: `stable` contains
the frozen seed forest, and `online` contains the currently published adaptive
forest, including any appended mixed store. These artifacts exclude calibration,
pending feedback and selector evidence. They cannot resume the selector or
reproduce its complete predictions, including an interpolation fallback. Other
adaptation modes reject this inspection method.

[State snapshots](../configuration.md#state-snapshots) save every adaptation,
including the combined selector, with calibration and pending rows, unless
`temporary_mixed_composition` is enabled. See
[the training and resume workflow](learned-forward-pass-model.md#online-tree-adaptation)
for source export, observation ordering and checkpoint differences.

## Causal expert selector controls

Set `estimator_config.fpm_learned.adaptation: causal_selector` with explicit
`estimation_mode: fpm_learned`. Construction requires a training seed checkpoint,
`correction.enabled: true`, and `correction.mode: hybrid` (`auto` resolves to
`hybrid`). The online controls above govern the adaptive expert; correction
controls govern the stable expert. The stable forest never changes its leaves
or splits, and the online forest receives no correction multiplier.

With only `adaptation: causal_selector` and a source checkpoint supplied, omitted
controls resolve to the W8 online recipe above, selector settings below, and
hybrid correction with `min_observations: 5`, `sampling.max_observations: 64`,
`sampling.bins_per_axis: [4, 4]`, and factor bounds `[0.5, 2.0]`.
Omitted correction is enabled for this mode. Standalone hybrid calibration
retains these same correction defaults and uses explicit `adaptation: frozen`.

Correction uses the [shared correction controls](../configuration.md#estimator-controls).
The combined selector owns the fixed and dynamic sampling budgets independently
in each learned forest store. It additionally separates correction histories by
learned forest store; the coordinates and PR 493 hybrid policy are unchanged.

All fields below belong to `estimator_config.selector`:

Nondefault selector settings require both `estimation_mode: fpm_learned` and
`adaptation: causal_selector`; other combinations fail validation. Default
settings are accepted in every mode and omitted from serialized configurations
so ordinary configurations remain readable by versions without the selector.
Changing any selector field preserves the complete selector block on round trip.

| Field | Default | Meaning and constraints |
| --- | --- | --- |
| `alpha` | `0.4` | Finite EWMA update fraction in `(0, 1]` for each expert's admitted fractional absolute percentage error. The first comparison initializes the averages directly. Larger values give recent comparisons more weight; this is separate from tree learning rate and calibration shrinkage. |
| `warmup` | `5` | Positive number of valid comparisons required for a store/query key before it can switch experts. The stable expert starts active. The comparison that reaches this count can begin a qualifying streak. |
| `margin` | `0.05` | Finite relative error advantage in `[0, 1)` required to challenge the active expert: `challenger_error < (1 - margin) * active_error`. This is a fraction of active error, not percentage points of MAPE. The test is symmetric in both directions and strict at the boundary. |
| `persistence` | `1` | Positive number of consecutive qualifying comparisons of the same store/query key required to switch. A tie, insufficient advantage, or switch resets the streak. Other keys do not advance it. |

Each model instance belongs to one worker. Its selector evidence is separated
by the four learned forest stores and the three correction query kinds, for up
to 12 keys. A key's count, error averages, active expert and streak use only
its admitted labels. Calibration is also separated by forest store and query
kind. Tree feedback remains shared across query kinds within the same forest
store, as in standalone online adaptation. Both experts receive feedback even
while inactive.

Use a synchronous chronological sequence: estimate one iteration, then admit
its complete FPM observation before processing the next. Tuning evaluates both
experts before that label changes their state; a switch affects later queries.
The interface does not retain a prediction ticket for delayed or reordered
feedback. `learned_selector_diagnostics()` exposes the key-specific evidence.
See [the selector workflow](learned-forward-pass-model.md#hybrid-calibration-with-a-causal-selector)
and [its equations](online-gbdt-and-causal-selection.md#hybrid-calibration-with-a-causal-selector).

## Temporary mixed composition for self-benchmark seeds

This opt-in policy covers the period when the original self-benchmark (A) has
no usable training store for mixed workloads. It requires explicit learned mode,
an `aggregated` worker and qualified original-A inputs. It applies only to
GBDT A-only and A+C, frozen hybrid calibration in A+C, and the causal selector
in A+C. B, B+C, C-only and regression keep their existing behavior. Omitting the policy preserves ordinary learned
execution.

See the [mixed-workload fallback design](mixed-workload-fallback.md) for the
population matrix, eligibility, calibration geometry, causal feedback ordering,
quality checking and persistence boundaries.

All fields below belong to `estimator_config.temporary_mixed_composition`:

| Field | Default | Meaning and constraints |
| --- | --- | --- |
| `scenario` | Required | `a` means original A with no feedback; `a_c` admits chronological feedback after scoring. The caller supplies this collection scope explicitly. |
| `mode` | Required | `compose` is valid only with `scenario: a` and frozen learned adaptation. `compose_calibrated` is valid only with `scenario: a_c`; it independently calibrates interpolation, with adaptation-specific routing described below. |
| `original_a.manifest_path` | Required | Path to the immutable original-A manifest described below. Normalization makes it absolute. |
| `original_a.source_sha256` | `null` | Optional SHA-256 of the manifest bytes. Normalization verifies or fills it; a stale digest is an error. This is the manifest hash, separate from the hashes of its source data and seed artifacts. |
| `calibration.sampling.bins_per_axis` | `[4, 4]` | Positive two-dimensional grid shape over native attention and MoE scores. Each missing mixed class owns independent fixed and dynamic stores. |
| `calibration.sampling.max_observations` | `64` | Sample-slot budget per policy and mixed class. Hybrid keeps independent fixed and dynamic budgets, up to twice this number of slots per class. |
| `calibration.min_observations` | `5` | Positive support threshold used by hybrid correction. A-only composition never admits feedback. |
| `calibration.factor_bounds.min` | `0.5` | Finite positive lower multiplier bound, no greater than 1; `null` removes the floor. |
| `calibration.factor_bounds.max` | `2.0` | Finite upper multiplier bound, no less than 1; `null` removes the ceiling. |
| `quality_gate` | Absent | Optional continuous quality check for standalone online GBDT in A+C. Frozen hybrid and selector reject this field. |
| `quality_gate.alpha` | `0.05` | EWMA weight in `(0, 1]` for paired pre-update fractional absolute percentage errors. |
| `quality_gate.min_comparisons` | `5` | Positive number of paired admitted labels after the mixed forest becomes ready. Training labels do not count. |

For example, add this sibling of `fpm_learned` to a complete A+C learned
configuration with its ordinary artifact or training checkpoint:

```yaml
estimation_mode: fpm_learned
fallback_policy: deny
worker_type: aggregated
estimator_config:
  temporary_mixed_composition:
    scenario: a_c
    mode: compose_calibrated
    original_a:
      manifest_path: models/original-a.json
    calibration:
      sampling:
        bins_per_axis: [4, 4]
        max_observations: 64
      min_observations: 5
      factor_bounds: {min: 0.5, max: 2.0}
```

The manifest has `schema_version: 1`, `source_kind: self_benchmark`, canonical
deployment `identity`, a positive `min_store_rows`, `accepted_iterations`,
`seed_artifacts` containing paths and SHA-256 digests, and source/grouping
`provenance`. `accepted_iterations` preserves complete attention-DP groups
from A, including eligible groups in classes below the fitting threshold.
When only an aggregate latency was measured, preserve that label on its
recorded carrier rank and use the existing zero sentinel on other ranks;
do not fabricate individual rank measurements. The SDK's
`fpm_learned.write_original_a_manifest(...)` uses the trainer's eligibility rules
and canonical identity resolver and refuses to overwrite an existing manifest.
The source population must be original A, before any C admission. Inference
seed artifacts must contain matching `metadata.identity` and per-class
`metadata.train_rows`; their union must cover every sufficiently populated
original-A class. For seed `model.json` files in a training checkpoint, the
exporter also records `checkpoint_manifest_sha256` to bind later online
continuations to their original A anchor. Malformed, changed or incompatible
inputs fail construction.

Source hashes verify byte identity; they do not prove that arbitrary authored
groups were derived from those bytes. The caller must use a trusted exporter
that preserves accepted original-A grouping and labels. A separately supplied interpolation table is permitted when its deployment
identity matches; its own hashes and resolved method are pinned separately.

At construction, Rust independently checks `contains_locally_mixed` (at least
one rank contains prefill and decode) and `cross_rank_aggregated` (prefill and
decode occur only on separate pure ranks). A class with enough accepted A
groups and a usable original trained store follows the ordinary GBDT or
selector path. Missing or insufficient A data makes that class eligible for
interpolation. Pure prefill and pure decode always retain their existing paths.

The interpolation prior uses the configured FPM interpolation method and existing
rank composition, with correction disabled on that prior. Canonical `auto`
selection resolves to SOL for registered models; direct interpolation requires
a qualified `fpm_profile`. The resolved method and exact timing-source hashes
are pinned in diagnostics and saved state. The policy does not fabricate a
profile or change methods after a query fails. It does not infer per-rank timing
from balanced A labels. A typed lookup outside the selected method's supported
domain returns an unavailable prediction; other configuration or data errors
remain errors. A-only returns the uncalibrated composition. A+C applies existing
hybrid correction using native attention/MoE coordinates, with fixed upper bounds
computed once from the accepted A population. Dynamic bounds evolve only from
admitted feedback. The two mixed classes never share interpolation corrections.
For standalone A+C GBDT, adding `quality_gate: {alpha: 0.05, min_comparisons: 1}`
keeps calibrated interpolation selected until a ready GBDT has equal or lower
EWMA error on subsequent paired feedback. Comparisons use both predictions before
learning the admitted label, independently by forest store and critical-rank
query type. Both experts keep learning while unselected. The check continues
and can return to interpolation; it cannot guarantee future prediction accuracy.
If the quality gate is omitted, an available ordinary prediction takes priority.

For standalone hybrid, use frozen learned adaptation with ordinary
`correction.enabled: true` and `correction.mode: hybrid`. A-supported stores
keep their frozen forests and ordinary calibration. Missing mixed stores use
interpolation multiplied by independent mixed calibration. Before its support
threshold, the multiplier is one. No mixed forest is trained from C and no
switch occurs after 20 labels. Configure this correction through the policy's
`calibration` block, separately from ordinary correction. For example,
`min_observations: 1`, a `[4, 4]` grid, `max_observations: 128` and factor bounds
`[0.5, 2.0]` start bounded calibration with the first admitted mixed label.

For A+C selector, calibrated interpolation replaces the frozen
GBDT with hybrid correction for each eligible mixed class. It supplies predictions
before the online GBDT is ready. Once ready, the existing causal selector compares
these two experts using subsequent paired feedback. Its configured warmup, margin,
and persistence still apply; readiness does not force an immediate switch to GBDT.
If only one expert can predict a query, that expert is used. Typed prior support
misses can therefore use a ready online GBDT; hard configuration, data, and
prediction errors remain errors. Selection never uses the current query's label.

The original-A eligibility map remains fixed for the model's lifetime. C
observations continue through ordinary learning and independently update the
fallback correction against the uncorrected interpolation prior, even after an
ordinary prediction becomes available. For online GBDT and selector, callers
can train a missing mixed store from at least 20 causally admitted C observations.
The twentieth query is scored before fitting; no training label is replayed as
selector comparison evidence. Subsequent labels update the online learner and,
when both experts predicted successfully before that update, selector losses.
Interpolation calibration continues independently, including after the online
expert becomes selected. Pure and original-A-supported classes retain their
existing selector experts and bootstrap behavior.

Prepare a standard single-class training checkpoint, then invoke the
canonical model's `append_mixed_learned_checkpoint(path)`. This operation is
restricted to eligible, absent mixed classes in A+C online GBDT or a causal selector. It
validates checkpoint hashes, identity, features, target and class ownership before
atomically adding only the online forest and learner state. It preserves existing
forests, calibration and selector histories. It neither creates a frozen expert
from C nor rewrites original-A evidence. Duplicate or incompatible checkpoints
fail without changing the live model. This scoped operation does not provide a
general append API for other scenarios.
If a fallback prior lookup fails during feedback, valid ordinary learning still
proceeds and the lookup error is returned; that call may therefore have updated
the ordinary history. A failed ordinary fit likewise follows its existing
admission/retry contract. Inspect the separate counters before retrying a failed
feedback call. Unsupported prior observations do not enter its correction state.
Callers own feedback sampling and cold-start training; submit each admitted
label once. Appending a trained store never replays its training labels into
calibration or quality evidence.

When a later self-benchmark
collects a class, retrain its seed and construct a new model with a new original-A
manifest: that class automatically returns to its ordinary path if its A store
is usable. The other mixed class is checked independently. Once both classes
are supported, the policy creates no interpolation prior or calibration stores;
it can then be removed from configuration.

`mixed_composition_route(metrics_by_rank)` inspects the selected route without
admitting a label; resolving selector availability can query interpolation.
`estimate_forward_pass_with_route(metrics_by_rank)` returns the prediction and
its actual route together through one native dispatch, avoiding a second prior
query. Its immutable
`fallback_eligible` field is separate from `route` and `owned`, which identify
whether this query selects ordinary execution or interpolation. Its
`calibration_enabled` field describes the independent fallback correction state,
including when ordinary execution is selected. Selector route metadata also
identifies the selected expert, critical-rank query type and online readiness;
its paired-loss counters belong to that store/query-type key. Availability of an expert
not queried is unknown rather than an asserted successful prediction.

When a mixed query has neither prior coverage nor an available learned expert,
`estimate_forward_pass_detailed(metrics_by_rank)` returns `latency_ms: null`
with an `unavailable_reason` string explaining the prior's support miss. The
field is omitted when there is no such reason. Prediction-details source/time
diagnostics preserve it; invalid configuration or corrupt data still raises an
error.

`mixed_composition_diagnostics()` reports
the eligibility reasons, selection policy, accepted A counts, pinned sources, score bounds and retained
calibration state. With the policy enabled, `diagnostics()` includes the same
value in its optional `temporary_mixed_composition` field; that field is omitted
when the policy is disabled. Both `readiness_report()` and
`regression_store_diagnostics()` count an eligible mixed interpolation route as
ready, including before a mixed forest exists or calibration has enough samples.
This reports estimator availability, not query coverage: an unsupported query
can still return no estimate. `mixed_composition_state()` and
`restore_mixed_composition_state(snapshot)` save and restore only the mixed
calibration and standalone quality state, with identity, source and policy checks. They do not save the
ordinary experts, selector histories or the caller's admission cursor; preserve
those separately for a complete resume. Standalone online learned
`save_online_checkpoint()` includes mixed state automatically. Portable
`save_state_snapshot()` and `estimator_config.state_snapshot` reject this policy;
their format does not package the mixed prior and its source/state dependencies.
A full mixed-selector checkpoint remains unsupported. A runtime checkpoint containing mixed
state cannot be loaded with frozen adaptation, even if the new configuration
omits the policy. Continue it online, or use the original training seed and
restore the mixed snapshot explicitly; this prevents silently losing calibration.
The policy uses version `original_a_mixed_experts_attention_moe_v3`. Snapshots from
the previous forced-interpolation and availability-priority policies are rejected
because they do not bind the new selector expert roles. Reconstruct the histories from their
original sources and causally admitted observations when migrating an old run.
