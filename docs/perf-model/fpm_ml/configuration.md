<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Learned forward-pass model configuration

This page documents `estimator_config.fpm_learned` for the canonical
performance-model API.
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
features and reject learned execution at construction. Parsing or retaining a
learned configuration in those callers does not make that execution supported.
No missing features are fabricated. Supply exactly one of
`estimator_config.fpm_learned.artifact_path` and `checkpoint_path`. Frozen
adaptation accepts either source; online adaptation requires a training or runtime
checkpoint and `estimator_config.correction.enabled: false`. Learned source
open, checksum, parsing and identity errors fail construction even with
`fallback_policy: allow`; they do not select a different estimator. Learned
mode also rejects speculative decoding (`nextn != 0` or a supplied
`speculation` block).

## Estimator namespaces

- `fpm_learned`: an explicit forest source, `adaptation` (`frozen` by default),
  and Rust-owned online fitting controls. Frozen forests support the existing
  correction policies. Online forests consume feedback to update their trees and
  require correction to be disabled. See the field table below.

## Learned forest and online fitting controls

All fields below belong to `estimator_config.fpm_learned`. The defaults describe
API behavior, not an accuracy recommendation. Configuration normalization and
validation happen in Rust; Python passes the complete configuration through.

| Field | Default | Meaning and constraints |
| --- | --- | --- |
| `artifact_path` | `null` | Path to the inference JSON artifact. Explicit learned mode requires exactly one source path. An artifact alone supports frozen adaptation. Empty paths are invalid. |
| `checkpoint_path` | `null` | Path to a seed training checkpoint or saved online checkpoint. Contains the forest, fitted bins, fitting metadata and required training state. Required for online adaptation; mutually exclusive with `artifact_path`. |
| `source_sha256` | `null` | Optional 64-character hexadecimal digest of the artifact file or checkpoint manifest. Rust checks and fills this during normalization, pinning the source in the resolved configuration. Remove the old digest when intentionally changing source paths. |
| `adaptation` | `frozen` | `frozen` preserves tree values and topology; `online` updates the forest from admitted feedback. Online adaptation requires `correction.enabled: false`. |
| `update_mode` | `structural` | With online adaptation, `leaf_only` refits existing leaves; `structural` can rebuild subtrees within each seed tree's existing capacity. Neither adds trees nor relearns feature-bin thresholds. |
| `recent_window` | `2048` | Positive maximum number of retained online observations per learned store. FIFO eviction removes the oldest recent observations; immutable training history is additional to this budget. |
| `update_every` | `64` | Positive number of pending observations in a store that triggers a fitting batch. Predictions use the last published forest until fitting succeeds; `flush_updates()` processes a partial batch explicitly. |
| `derivative_refresh` | `eager` | `eager` recomputes derivatives on the full retained fitting population. `lazy` updates cached statistics for insertions and evictions, with eager fallback when structure must change. |
| `eager_refresh_interval` | `32` | Positive maximum fitting-batch interval between full derivative refreshes in lazy mode. Counts batches, not predictions or elapsed seconds. An interval of 1 refreshes every batch. |
| `split_sampling_rate` | `1.0` | Finite fraction in `(0, 1]` of fitted numeric thresholds considered per feature for structural split search. The incumbent and missing-only boundary remain eligible. This is candidate sampling, not observation sampling. |
| `split_rank_tolerance` | `0.0` | Finite fraction in `[0, 1]` defining the acceptable rank of the incumbent split among valid candidates. Zero requires a best-ranked split; larger values can retain a lower-ranked incumbent. Invalid child populations always trigger rebuilding. |
| `seed` | `0` | Unsigned integer seed for deterministic split-candidate sampling. It does not change the offline artifact's fitted thresholds or training random state. |
| `history_weight` | `null` | Optional finite historical loss-mass fraction in `[0, 1]`. `null` gives every logical observation unit weight. A number balances total historical and recent mass independently of their population sizes after the ramp. |
| `history_weight_ramp_observations` | `0` | Number of retained recent observations needed to reach the configured history/recent mass balance. Zero applies it immediately; a positive value requires `history_weight`. This is retained support, not lifetime admissions. A ramp longer than `recent_window` is valid but cannot reach the final mass balance while the window remains smaller. |
| `history_bin_grouping` | `false` | Compress immutable history by complete fitted-feature-bin vectors, preserving target sums and original multiplicities. Changes storage and fitting work, not the intended weighted objective. |
| `mutable_tree_tail` | `null` | Optional positive number of final seed trees allowed to change. `null` permits all trees. A value above the forest size also permits all trees; the prefix remains frozen. |
| `update_step` | `1.0` | Finite leaf-value displacement fraction in `(0, 1]`. Values below 1 interpolate from the existing leaf toward its fitted value and require `update_mode: leaf_only`. This differs from the seed learning rate. |
| `recent_max_age_observations` | `null` | Optional positive maximum age of a recent row in the model's global accepted-observation sequence, inclusive. `null` disables age expiry. This is an observation count, not time; history is never expired. |

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

For example, this explicit profile keeps 64 recent observations and fits after
each admitted label. It retains complete historical evidence while assigning it
10% of total loss mass once 32 recent observations are retained. These settings
illustrate the controls; they are not a universal optimum.

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
`export_learned_artifact()` produces an inference snapshot; it does not preserve
online continuation. `save_online_checkpoint(path)` writes a new checkpoint
directory without implicitly flushing pending observations. Resume through the
same canonical constructor with matching identity and fitting controls. See
[the training and resume workflow](learned-forward-pass-model.md#online-tree-adaptation)
for source export, observation ordering and checkpoint differences.
