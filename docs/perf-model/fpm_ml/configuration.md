<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Learned forward-pass model configuration

This page documents `estimator_config.fpm_learned` and
`estimator_config.selector` for the canonical performance-model API.
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
for estimation and feedback. Current engine replay, engine CLI prediction and
Sweeper timing callers provide phase aggregates through a `native_engine`
interface. They cannot supply the learned model's full iteration/request
features. Python prediction and Sweeper configurations reject learned mode during
validation; native replay rejects it before loading the learned source.
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
[State snapshots](../configuration.md#state-snapshots) save every adaptation,
including the combined selector, with calibration and pending rows. See
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
