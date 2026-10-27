<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance-model configuration

## Choosing a forward-pass API

Use `RustForwardPassPerfModel.best_available(config)` from Python or
`ForwardPassPerfModel::best_available(config)` from Rust. The canonical
`ForwardPassPerfModelConfig` owns model, hardware, backend, topology, data
policies, a required immutable `worker_type`, and the complete nested
`estimator_config`. Worker roles are `prefill`, `decode`, and `aggregated`.
Topology carries two optional context-parallel knobs next to `tp`, `pp`,
`attention_dp`, `moe_tp_size`, and `moe_ep_size`: `cp_size` (prefill context
parallelism, SGLang `--attn-cp-size` / vLLM `-pcp`; extra attention ranks, so
`tp * attention_dp * cp_size == moe_tp_size * moe_ep_size`) and `dcp`
(decode context parallelism, vLLM `-dcp` / SGLang `--dcp-size`; stripes the
decode KV cache across the existing TP ranks and adds no GPUs). Both default to
one and stay out of the serialized identity when unset.

```python
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

config = ForwardPassPerfModelConfig(
    model="Qwen/Qwen3-32B",
    system="h200_sxm",
    backend="vllm",
    worker_type="aggregated",
    tp=2,
    estimation_mode="auto",
    fallback_policy="deny",
    estimator_config={
        "features": {"attention_kv_weight": 1.0},
        "fpm_regression": {
            "sampling": {"bins_per_axis": [4, 16], "max_observations": 128},
            "min_observations": 5,
            # Opt into periodic rebuilding; omission leaves it disabled.
            "fit": {"rebuild_interval": 4096},
        },
        "correction": {"enabled": True},
    },
)
model = RustForwardPassPerfModel.best_available(config)
print(model.diagnostics()["provenance"])
```

### Engine identity controls

Canonical JSON/YAML requires string values for text fields. Numeric-looking YAML
identities must be quoted to preserve their exact spelling:

| Field | Default | Input requirement |
| --- | --- | --- |
| `model` | Required | String model name or path; write a numeric identifier as `model: "123"`. |
| `backend_version` | `null` | Optional string version; write `backend_version: "0.10"` to preserve the trailing zero. |

This is an intentional compatibility change for direct Rust YAML parsing:
earlier parsing accepted numeric scalars for string fields, while canonical
normalization now rejects them. Quote numeric, Boolean, or null-like values
when they represent text. An actual `backend_version: null` still means unset.

The canonical configuration also carries quantization overrides and `attention_backend`, `moe_backend`, `moe_kernel_source` (default absent), `enable_eplb` (default `false`), and `wideep_num_slots` (default absent). These controls reach model construction, KV memory sizing, and replay provenance. EPLB/slots and nondefault MoE backend or kernel-source selection require an MoE model. Collected FPM interpolation cannot represent EPLB, slots, MoE backend, or kernel-source overrides; it rejects an explicit incompatible request and is skipped during automatic selection for those identities.

`encoder_parallel` (default absent) compiles a VL model's vision tower in the given layout, `tp` (sharded over the tensor-parallel group) or `dp` (replicated, images split across ranks), so the estimator can price encoder calls. It requires a model with a vision encoder, is omitted from the serialized payload when unset, and is part of the estimator identity, cache keys and provenance. Llama 4 rejects `dp`.

`moe_kernel_source` selects an exact, nonblank collected `kernel_source` label for fused MoE compute. It is distinct from the existing `moe_backend` graph/backend control; source labels are not backend aliases and are preserved without trimming. `None` keeps the existing default source-selection policy, including eligible low-latency NVFP4 selection. `SILICON` reads only the requested source's table; `EMPIRICAL` derives its estimate from that same source; `HYBRID` may fall back to empirical estimation within that source, but does not substitute a different source. Missing source data remains an error. An explicit `moe_torch_flow_min_latency` requires gated NVFP4 and at most 128 tokens after attention-DP gathering. Pure-roofline `SOL` remains table-independent and does not claim measured support for the requested source.

`moe_perf.parquet` may include Boolean selection metadata `default_eligible`. An absent column preserves legacy automatic selection; when present, every value must be a non-null Boolean. A `false` row requires a nonblank string `kernel_source`, preserved exactly, and is available only through that named source. It cannot enter automatic standard or low-latency grids, including empirical cross-shape and cross-quant reference selection. Among eligible rows, existing source priority and first-row precedence remain unchanged. Default table views, coverage, and readiness use those same eligible grids; raw Parquet enumeration retains all measured rows. Malformed eligibility metadata is an invalid-data error, never a missing-data fallback. The flag is not a measurement identity dimension. Collector finalization preserves it during merges, including existing annotations when a legacy recollection omits the column; genuinely new legacy keys remain eligible.

An explicit source is rejected for dense graphs, MegaMoE modules, large-EP expert-compute graphs, and any constructed timing phase with no compatible fused MoE operator. It is also incompatible with whole-forward FPM, including the legacy Task `forward_model='fpm'` rewrite. Invalid graph/source combinations fail as invalid configuration rather than triggering estimator fallback. An untrained `fpm_regression` model remains not-ready; retaining a source in its configuration is not evidence of source-specific prediction support.

AFD regular companions currently reject exact-source requests through their legacy estimator and fixed timing paths. The external-FPM companion forwards the source to canonical validation, which rejects the incompatible FPM request. These controls describe standalone AISimulate behavior, not downstream Dynamo planner integration.

`nextn` remains compute-side identity. Expected accepted draft tokens are a
simulator workload assumption, supplied separately by the unified CLI as
`engine.nextn_accepted`; they do not tune the estimator.

### Selection and fallback

`estimation_mode` defaults to `auto`; `fallback_policy` defaults to `deny`.
Auto always searches `op_level`, `fpm_interpolation`, then `fpm_regression`,
including when fallback is denied. For an explicit choice of one of those three
modes, deny permits only that estimator; allow tries the requested estimator
first, then the remaining estimators in the same global priority order. Each
native mode tries the ordered `systems_paths` before moving to another estimator.
Omitted roots preserve configured SDK discovery (or the systems-path environment
override when the SDK uses its packaged default); an explicit `default` entry
selects the packaged root.
The resolved paths are shared by preflight, construction and capacity estimation.
Selection occurs
at construction; queries do not silently switch estimators on a data-domain error.

`fpm_learned` requires explicit selection and `fallback_policy: deny`; it is
never an automatic or fallback candidate. Supply exactly one of
`estimator_config.fpm_learned.artifact_path` and `checkpoint_path`. Online
adaptation is the default; set `adaptation: frozen` explicitly for a fixed forest.
Frozen adaptation accepts either source, online adaptation requires a training or
runtime checkpoint, and `causal_selector` requires a training seed checkpoint.
Learned models support the standalone canonical API with complete per-iteration
`ForwardPassMetrics` for prediction and feedback. Replay, engine CLI prediction
and recommendation, and Sweeper timing use phase aggregates that cannot supply
these features, so they reject learned mode before loading its source. See
[learned configuration](fpm_ml/configuration.md#selection-and-supported-execution)
for source identity checks, adaptation controls and correction requirements.

Invalid caller configuration does not trigger fallback. A constructed regression
may be unready: nonempty queries return `None` until their selected workload
store has a usable fit. Offline prediction/recommendation reject an untrained
regression instead of fabricating a latency. The current aggregated regression
retains its four workload stores; dedicated roles retain one each.

The returned provenance records the requested and selected modes, failed
selection attempts, effective backend version, data policy, selected root,
and complete estimator configuration. Its resolved config pins the selected
mode with deny so saved replay input repeats that selection.

### Diagnostic source values

`model.diagnostics()["source"]` identifies the selected model and its current
calibration state. The Rust API serializes the same values that the Python SDK
returns:

| Value | Meaning |
| --- | --- |
| `aic` | Native op-level or FPM interpolation model with no ready correction store. |
| `aic_with_correction` | Native model with at least one ready correction store. |
| `fallback_regression` | Regression model, including an explicitly selected `fpm_regression`; the name does not imply that fallback occurred or that the model is ready. |
| `learned` | Frozen learned forest with correction disabled or no ready correction store. |
| `learned_with_correction` | Frozen learned forest with correction enabled and at least one ready correction store. |
| `learned_online` | Standalone online learned forest, including before its first update. |
| `learned_causal_selector` | Combined causal selector, regardless of which expert serves a query. |

These values describe the model, not coverage or correction availability for
every query. Consult `readiness` and the store diagnostics for those details.
Use `learned_selector_diagnostics()` to inspect per-key expert selection;
switching experts does not change the combined model's `source`.

### Speculative decoding configuration

Use the same `best_available(config)` constructor and returned model for
speculative iteration costs. The canonical `speculation` field carries the
existing SDK scheme identity:

| Field | Default | Meaning |
| --- | --- | --- |
| `speculation` | `null` | Optional speculative scheme; omit it for the existing autoregressive or legacy `nextn` configuration. |
| `speculation.kind` | Required | Registered scheme name: `none`, `mtp`, `ngram`, `eagle3`, `dflash`, `dspark`, or `draft_model`. A valid `none` configuration normalizes to disabled speculation. Scheme, model, and backend capability checks still apply. |
| `speculation.params` | `{}` | Scheme-specific parameters, validated by the existing scheme implementation. Unknown parameters are rejected. |
| `speculation.draft_model_path` | `null` | Draft checkpoint path or model ID. Required and verified for standalone `draft_model`, which builds its graph from that checkpoint. |
| `speculation.draft_config` | `null` | Parsed draft checkpoint configuration retained as identity evidence. Learned schemes can reuse resolved geometry; this field does not replace `draft_model_path` for standalone `draft_model`. |

For example, Python/JSON prompt lookup retains its existing shape:
`{"kind": "ngram", "params": {"num_speculative_tokens": 2}}`. MTP uses
`{"kind": "mtp", "params": {"depth": 2}}`. Legacy `nextn` input remains
supported; do not combine a nonzero `nextn` with `speculation`. The generic cost
API uses each scheme's parameter validation rather than a shared 1–5 draft-token
limit. The existing n-gram contract retains vLLM and 1–5 draft tokens. Replay's
backend, scheme, and scheduling limits remain separate.

Rust callers now construct the public struct instead of the former
`ForwardPassSpeculationConfig::Ngram` enum variant:

```rust
use aisimulate_core::{
    BackendKind, DatabaseMode, EstimationMode, ForwardPassPerfModel,
    ForwardPassPerfModelConfig, ForwardPassSpeculationConfig, ForwardPassWorkerType,
};

let mut config = ForwardPassPerfModelConfig::new(
    "Qwen/Qwen3-8B", "h200_sxm", BackendKind::Vllm, ForwardPassWorkerType::Decode,
);
config.estimation_mode = EstimationMode::OpLevel;
config.database_mode = DatabaseMode::Sol;
config.speculation = Some(ForwardPassSpeculationConfig {
    kind: "ngram".into(),
    params: [("num_speculative_tokens".into(), 2.into())].into_iter().collect(),
    draft_model_path: None,
    draft_config: None,
});
let model = ForwardPassPerfModel::best_available(config)?;
```

This is a Rust source migration; existing n-gram JSON and legacy `nextn`
configuration remain readable. OP timing and analytical SOL reuse the same
speculative graph: choose `estimation_mode="op_level"` and set `database_mode`
to the desired database policy, or `"SOL"` for analytical timing. SOL is not a
new estimator mode, and individual operations may lack SOL support. FPM's
`estimator_config.fpm_interpolation.method="sol"` instead controls interpolation
of measured whole-forward data; it does not select analytical-only timing.

Query conventions remain unchanged. For OP/SOL,
`estimate_forward_pass_time_ms()` consumes already-packed `ForwardPassMetrics`
decode counters: for `B` real requests, verification width `v`, and per-request
KV length `s`, use `num_decode_requests = B*v` and
`sum_decode_kv_tokens = B*v*s`. Their ratio remains `s`; the query does not apply
the width again. `predict_decode_latency_total()` wraps the same counters.
`static_phase_latency()` instead takes the real request batch and applies the
configured verification width internally. This OP telemetry convention is
distinct from the proposed speculative FPM mapping, which remains unsupported.

The canonical constructor currently supports explicit speculative schemes with
OP/SOL timing. FPM interpolation and regression are unsupported candidates for
these schemes. An explicit FPM request with `fallback_policy="deny"` fails;
`auto` or an explicit mode with `fallback_policy="allow"` can select a supported
OP candidate while preserving the complete speculative identity. They do not
drop speculation to obtain an autoregressive estimate. FPM profiles remain
autoregressive-only. Lower-level FPM hybrid code does not imply support through
the canonical ForwardPassMetrics query API.

Acceptance assumptions and scheduler seeds belong to the prediction/Replay
layer, not this cost configuration. The returned model continues to estimate
iteration latency; its constructor and query signatures are unchanged. For
prediction and AgentX, the [registered-scheme Replay input](../replay/engine/speculation.md#registered-schemes)
adds acceptance and a seed while retaining this cost identity. That path uses
the existing linear 1–5-token executor and requires fixed KV capacity; broader
cost-model support does not imply broader scheduler support.

### Estimator controls

For supported execution paths, `estimator_config` is passed intact through the
Python facade, CLI, Sweeper, and Replay, then parsed and validated in Rust.
Learned mode remains limited to the standalone API described above. Unknown
fields report their nested paths. The supported namespaces are:

- `op_level.execution_profile`: optional pinned external operation graph, with
  an absolute `path` and lowercase `sha256`. See
  [external execution profiles](methods/op-level.md#external-execution-profiles)
  for admission, data isolation and query limits.
- `features`: `attention_kv_weight`, `prefill_attention_pair_weight`, and
  `ffn_token_weight`, each defaulting to 1.0. These currently affect regression
  only; positive finite values are required when regression is constructed.
- `fpm_regression`: independent `sampling`, `min_observations` (5), and `fit`.
  The default fit kind is `standardized_nnls` (also accepted as `linear`),
  with a free intercept and nonnegative slopes. `spline` selects an additive
  piecewise-linear fit with learned knots and nonnegative segment slopes.
  Optional `fit.linear` controls fitted axes, signed slopes, and lazy updates;
  omission preserves the existing linear behavior and serialized defaults.
  `singular_ridge_scale` defaults to `1e-9` and applies to the shared linear
  fit only when retrying a singular equation. `rebuild_interval` defaults to JSON `null` / Python
  `None`, disabling periodic rebuilding. A positive integer opts into
  rebuilding after that many retained-sample mutations.
- `fpm_learned`: explicit forest or checkpoint source, adaptation mode, and
  online fitting controls for standalone learned models. See the
  [learned configuration field tables](fpm_ml/configuration.md).
- `selector`: expert selection controls for learned
  `adaptation: causal_selector`, documented on the same learned configuration page.
- `state_snapshot`: restores a model saved by `save_state_snapshot`,
  including its tuned state. See [state snapshots](#state-snapshots).
- `correction`: `enabled` (false for standalone online learned mode, true otherwise),
  `mode` (`auto`), independent `sampling`, `min_observations`
  (5), `factor_bounds` (min 0.5, max 2.0), and the existing `max_num_tokens`
  (8192), `max_batch_size` (512), and `max_kv_tokens` (2000000) fixed-policy ranges.
- `fpm_interpolation.method`: `auto` (default), `sol`, or `direct`. Rust selects
  SOL for a registered architecture, or direct interpolation for a verified
  architecture without a registered class when a profile is supplied. Without
  a profile, auto retains SOL. Explicit SOL requires a registered analytical
  model; direct requires a profile and also supports registered architectures.
- `fpm_interpolation.collect_coverage`: `false` (default). Opt in to bounded
  evidence from actual direct-FPM lookups, as described below. It requires
  explicit `estimation_mode: fpm_interpolation`, resolved `method: direct`, and
  `fallback_policy: deny`. It is omitted from normalized serialization when false.

The top-level `fpm_profile` contains the complete profile dictionary: pinned
model revision, architecture, context length, expert count, deployment precision
and topology, cache geometry, memory evidence, and provenance. A profile requires
an explicit literal `backend_version` that matches its selected deployment;
slot aliases and omitted versions are rejected. This profile schema does not
declare recorded DCP, so combining `fpm_profile` with an explicit `dcp` is
rejected; measured DCP profiles without `fpm_profile` retain their existing route.
Profile/schema and precision conflicts fail before estimator fallback. Omitted precision fields are filled
from the profile and preserved in the resolved canonical configuration.

Each deployment may declare `worker_type: prefill`, `decode`, or `aggregated`.
The canonical construction request selects only the matching role's precision,
scheduler envelope, cache geometry and memory evidence. Separate P/D deployments
may share hardware and topology while retaining different resources. Historical
deployments without this field keep their shared-resource behavior and omit the
field on serialization; they cannot coexist with role-specific deployments at
the same hardware/runtime/topology identity. The Python memory adapters accept
`worker_type` to select the same role, defaulting to `aggregated` for existing
callers. An absent or different explicit role is an error, not a fallback to
another role's capacity.

Runtime normalization and construction verify the profile architecture against
checkpoint `config.json` from the local model path or pinned remote revision
before interpolation selection. Missing or malformed architecture metadata and
mismatched declarations fail explicitly. This check reads configuration only:
no weights or analytical graph are required. An architecture with verified
metadata remains valid for direct interpolation without a registered analytical
class. Schema-only profile and application configuration parsing remain lightweight.

For measured-only timing, pass `estimation_mode="fpm_interpolation"`,
`fallback_policy="deny"`, and
`estimator_config={"fpm_interpolation": {"method": "direct"}}` together with
`fpm_profile`. Direct interpolation requires `database_mode="SILICON"`, emits
whole-forward native operations without SOL operations, and never constructs
an analytical graph. Profile resource estimates and memory planning do not
require timing data or a native timing model.

Direct readiness requires genuine measurements for the request's `worker_type`:
prefill requires prefill rows, decode requires decode rows, and aggregated
requires both phases. Querying an absent phase still fails explicitly. Construction
continues to later systems roots when a required phase is unavailable. Query
coverage still needs an exact point or supported interpolation. Cross-KV prefill
uses the nearest same-batch lower and upper KV curves that both cover the
requested token count, without a KV distance limit. SOL's site-distance guard
does not apply to this direct bracket. See the [self-service coverage rules](fpm-self-service/implementation.md#choose-the-model-execution-route).

The returned provenance pins both the selected estimation mode and interpolation
method, alongside the complete normalized profile. Reusing its `config` keeps
that selection across serialization and replay. Later timing coverage errors
never switch estimator or interpolation method. A registered model's graph
construction failure does not change SOL to direct; top-level fallback still
follows the configured estimator ordering and policy.

- `fpm_interpolation`: `text_only` (false) permits text prefill/decode profiles
  for multimodal architectures while retaining encoder weights. It does not
  supply encoder timing. `unrecorded_quant_modes` (empty) may contain `fmha`
  and/or `comm` to match an explicitly unrecorded precision field in a profile.
  The corresponding top-level quant mode must remain unset. This selects null
  profile values exactly; it does not make precision matching a wildcard.
  `fpm_parquet_path` selects an external FPM parquet with its same-stem metadata
  sidecar. Unknown fields are rejected.

Sampling defaults to `bins_per_axis: [4, 4]` and `max_observations: 64` per
logical store. Rectangular grids were already supported. Regression's
`sampling.axes` now selects one to six distinct coordinates, defaulting to
`[attention, moe]`; `bins_per_axis` must have the same length, contain positive
integers, and have a representable product. The dimension is the number of
selected axes, independent of how many features the fit uses. Regression uses
dynamic `log1p` retention coordinates and fits standardized feature values. Correction's
one-dimensional prefill grid uses the product of the two axis counts. Its coordinate
transform and bounds depend on the policy below. Balanced retention evicts the oldest
sample from the most populated cell when the store exceeds its budget.

| Correction field | Default | Behavior |
| --- | --- | --- |
| `estimator_config.correction.enabled` | `false` for standalone online learned mode; `true` otherwise | Standalone online learned mode requires disabled correction; the combined selector requires enabled correction. Explicit values override the default and are validated for the selected mode. |
| `estimator_config.correction.mode` | `auto` | `auto` keeps fixed correction for native AIC estimates and selects `hybrid` for learned forests. `hybrid` uses the fixed grid inside its configured limits and dynamic local correction outside. `fixed` and `dynamic_local` explicitly select a single policy. |
| `estimator_config.correction.sampling.max_observations` | `64` | Sample-slot budget per workload kind and policy. Hybrid retains independent fixed and dynamic budgets, up to `2 * M` slots per kind (128 by default). Dynamic and hybrid updates take `O(M log M)` time and `O(M)` temporary storage; measure update cost before increasing capacity. |

The fixed policy uses raw workload coordinates and the configured fixed limits.
The dynamic local policy expands `log1p` bounds as valid feedback arrives, balances
local samples across the grid, and blends local median ratios with a recent-global
median. Empty cells and out-of-range queries use that recent-global fallback.
At the default capacity of 64, it reserves 48 local and 16 recent slots and expires
samples after 128 admissions. Bounds and cached factors change only on feedback.
The hybrid policy feeds both stores the same observations and uncorrected base
predictions; the fixed store continues rejecting out-of-grid observations.
Selection depends only on the query coordinates and the configured fixed limits,
including their upper endpoints. An empty or unready in-grid cell keeps the fixed
factor of 1.0, even if the dynamic store is ready. Dynamic geometry never changes
the fixed grid. Native AIC can also select `hybrid` explicitly.
See [learned-model calibration](fpm_ml/learned-forward-pass-model.md#online-correction)
for the readiness thresholds, shrinkage, and custom-capacity behavior. The existing
learned-model SDK adapter accepts the equivalent legacy option `correction_mode`;
canonical native configurations use `estimator_config.correction.mode`.

Correction explicitly reports `feature_space: legacy_workload` for all policies
and retains separate prefill/decode/mixed stores. Native AIC's default policy
and median-ratio calculation remain unchanged. A shared role-based correction space requires separate
accuracy validation and is not accepted as a configuration value in this release.

Use `regression_store_diagnostics()` for per-store counts/readiness. Summary
readiness means at least one store has a usable linear fit, including when
spline fitting is selected; another cold store can still return `None`.
Readiness does not guarantee coverage for every query: spline predictions require
an available linear prediction for the same query. `tune_with_fpms()` preserves
the established FPM observation contract. Native construction still uses Python
model compilation; estimator selection, regression, correction, and latency
computation are owned by Rust.

For fit configuration see [Online regression](methods/online-regression.md).
For direct measurement coverage see [Whole-forward FPM](methods/whole-forward.md).

### State snapshots

Configuration describes a cold model. To keep what a model has learned from
`tune_with_fpms()`, save it with `model.save_state_snapshot(path)` (Rust:
`ForwardPassPerfModel::save_state_snapshot(path, &StateSnapshotOptions)`).
Saving atomically writes a new directory and returns the pinned canonical
configuration, which is also written to `path/config.json`. Restore through
the same constructor: `RustForwardPassPerfModel.best_available(state_snapshot=path)`
reads that file, and passing the returned configuration is equivalent. There is no
separate loader; `best_available` restores whenever
`estimator_config.state_snapshot` is present.

```python
config = model.save_state_snapshot("/tmp/decode-snapshot")
frozen = RustForwardPassPerfModel.best_available(state_snapshot="/tmp/decode-snapshot")
config["estimator_config"]["state_snapshot"]["on_tune"] = "continue"
resumed = RustForwardPassPerfModel.best_available(config)
```

| Field | Default | Behavior |
| --- | --- | --- |
| `estimator_config.state_snapshot.path` | required | Snapshot directory written by `save_state_snapshot`. Normalization makes it absolute against the working directory and removes `.` and `..` lexically; symbolic links are kept, so a stable mount alias stays valid. |
| `estimator_config.state_snapshot.sha256` | `null` | SHA-256 of `path/manifest.json`. Normalization fills it; a supplied value that differs is an error. |
| `estimator_config.state_snapshot.on_tune` | `reject` | `reject` makes `tune_with_fpms()` and `flush_updates()` return an error. `continue` resumes tuning from the saved state exactly as the saving model would have continued. |

A snapshot covers every estimator kind. The returned configuration is the
model's resolved provenance configuration plus `state_snapshot`; files the
snapshot owns are redirected into it:

| Saved model | Snapshot contents |
| --- | --- |
| Native `op_level` or `fpm_interpolation` | Correction stores (fixed, dynamic local or hybrid). The engine is recompiled from the configuration; the manifest records the compiled engine-specification digest from construction and a performance-data fingerprint. A configured `op_level.execution_profile` file is copied. |
| `fpm_regression` | Every workload store: retained samples, fits, incremental statistics, lazy-update and spline search clocks. |
| `fpm_learned`, `frozen` | The forest (an artifact's exact bytes, or the checkpoint's forest exported unchanged) and correction stores. |
| `fpm_learned`, `online` | An online runtime checkpoint with the current forest, retained and pending rows and fitting state. |
| `fpm_learned`, `causal_selector` | The frozen expert, the adaptive expert's runtime checkpoint, every calibration store and all twelve selector evidence keys. |
| `from_learned` model | The forest, its constructor options and correction stores. |

Saving does not fit pending online rows or change the model. Every staged
file and directory is synced to disk before the staging directory is renamed
into place, and the parent directory after. A restored model
predicts bit-identically, reports the same readiness, store and correction
diagnostics, and after the same later feedback holds byte-identical state,
including in another process. Readiness follows the restored state: a cold
regression saved before it fit is restored cold. The restored provenance
describes the snapshot construction, so it records the explicit mode, no
selection failures and no fallback warning; a not-ready regression that was
originally selected by fallback therefore reports `insufficient_data`
instead of `unsupported_config`. Direct-FPM query coverage counters and two
online learner diagnostics are not model state: the coverage counters restart
empty, `training_state_bytes` reflects the restored allocation, and
`history_grouping_ms` is the time this process spent grouping history rows.

Restoration requires an explicit `estimation_mode` and `fallback_policy: deny`
and fails, without selecting another estimator, when:

- the manifest digest, a pinned file's size or digest, or the snapshot format
  differs: snapshots load only in the same `aisimulate-core` version with the
  same state layout and engine-specification schema;
- the configuration, compared field by field after normalization, differs
  from the saved identity (the error names the first differing field);
- a native engine compiles to a different specification, its performance
  data no longer matches the fingerprint (checked before any of it is loaded),
  or this process loaded that data before it changed;
- saved state is truncated, corrupt or internally inconsistent, for example a
  sample stored under the wrong bucket, a dynamic correction row outside its
  bounds, spline knots that do not fit the basis, or store kinds that do not
  match the worker role; or
- the restored diagnostics differ from those recorded when saving.

Native snapshots reference performance data by content fingerprint: every
file directly under the systems root, the system's `data_dir` resolved as the
loader resolves it (every directory and file below it, or its absence when an
external parquet needs no operation tables), and an external
`fpm_parquet_path` with its `.metadata.json` sidecar. Source resolution
depends on which files and directories exist, not only on those read so far,
so additions, removals and new empty version directories invalidate the
fingerprint too. Like the resolvers, the fingerprint follows symbolic links,
records an unresolvable link without reading it, and skips dot-prefixed and
non-UTF-8 names. A native model fingerprints its data when the process first
loads that systems root; later constructions reuse that result.
`save_state_snapshot` fingerprints the data again and fails if it changed
since it was loaded; restart the process to save a model of the current data. Pass
`include_perf_db=True` (Rust: `StateSnapshotOptions { include_perf_db: true }`)
to copy exactly the fingerprinted entries into the snapshot, pointing
`systems_paths` (and `fpm_parquet_path`) at the copy. A `data_dir` outside
the systems root can be referenced but not packed. A packed snapshot holds a
copy of the system's whole data directory, so its size follows that data.
Learned forests and online checkpoints are always copied. The model
configuration is not: native restoration reads `model` again, from a
configuration bundled with the package, a local directory or Hugging Face, so
a packed native snapshot still needs it, and restoration fails if it compiles
a different engine specification from the saved one.

Snapshot configurations pin absolute paths. Restoration reads the pinned
`state_snapshot.path`, so `best_available(state_snapshot=path)` raises
`ValueError` unless `path` is the directory the snapshot was saved to (compared with
`os.path.samefile`) rather than silently restoring the original. A
configuration is not compared with that location. Edited to name a copy, a
snapshot with a deployment identity restores the copy after verifying it
against its manifest, but its other pinned paths (packed performance data, an
execution profile, learned artifacts and checkpoints) belong to that identity
and must keep naming the original directory. A `from_learned` snapshot derives
its artifact path from `state_snapshot.path`, so it rejects the edit unless
`fpm_learned.artifact_path` names the copy too. Save a new snapshot instead of
moving one. Snapshot
I/O failures raise an `OSError` subclass (for example `FileNotFoundError` for
a missing snapshot) through both `state_snapshot=` and a configuration, never the
skippable `PerfDataNotAvailableError`. A manifest from another release
reports its versions even when it has fields this release does not know.
Online learner state, including the causal selector's adaptive expert, is
always saved in full, so its size grows with the forest and its retained
training history. Native restoration recompiles through Python like any
native construction. A `from_learned`
model has no deployment identity: its snapshot configuration carries the
placeholder identity `model: from_learned`, `system: from_learned`,
`backend: vllm`, restoration checks only its estimator controls and files, and
`deployment_topology()` returns `None`.

Gates for replaying a saved model read the snapshot without loading it.
`RustForwardPassPerfModel.inspect_state_snapshot(path)` returns the snapshot
kind, versions, pinned identity, deployment topology and the readiness report
recorded at save time. `model.readiness_report()` lists, for every logical
workload store of the worker role, whether its iterations receive an estimate
and why not (`insufficient_data`, `no_usable_fit` or `untrained_forest_store`),
plus every correction store's readiness; native estimators are always ready.
`model.deployment_topology()` reports the identity's worker role, parallelism,
speculation, KV block size and fixed correction limits.
`RustForwardPassPerfModel.config_engine_launch_mismatches(engine_args, state_snapshot=path)`
and `model.engine_launch_mismatches(engine_args)` apply the engine launch rules
(for example `dp_size` against `attention_dp`) to the canonical launch arguments
with the snapshot as the timing model and return every rule that fails; an
empty list means the launch accepts the combination. Launch arguments that are
invalid on their own, such as `dp_size: 0` or `gpu_memory_utilization: 2.0`,
raise with the launch's own message instead. The Rust equivalent is
`EngineLaunchConfig::timing_config_mismatches`.

## Unified CLI timing fields

The `engine` block of `predict` and `recommend` YAML is documented in
[Replay engine](../replay/engine/README.md). The per-role `timing` fields below
select the forward-pass timing provider. SDK constructor names differ (for
example, `parallelism.tensor` becomes `tp`). A Default Range of `x` is
concrete-only.

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `engine.estimation_mode` | `auto` | `x` | `-` | `auto`, `op_level`, `fpm_interpolation`, or `fpm_regression`. `fpm_learned` is rejected by prediction and recommendation configuration validation. |
| `engine.workers.<role>.timing.estimation_mode` | `null` | `x` | `-` | Optional role override using the same four modes; omission inherits the global selection. `fpm_learned` is rejected here too. |
| `engine.workers.<role>.timing.type` | `default` | `x` | `-` | `default`, `fixed`, or `polynomial`. |
| `engine.workers.<role>.timing.prefill_ms` | `null` | `x` | `-` | Nonnegative and required for `fixed` timing. |
| `engine.workers.<role>.timing.decode_ms` | `null` | `x` | `-` | Nonnegative and required for `fixed` timing. |
| `engine.workers.<role>.timing.forward_model` | `op_level` | `x` | `-` | `op_level` or `fpm`; `default` timing only. `fpm` replays whole-forward (FPM) latency measured for the role's exact model, hardware, backend version, parallel shape and quantization, and fails closed when no such cell exists. |
| `engine.workers.<role>.timing.fpm_parquet_path` | `null` | `x` | `-` | External FPM parquet for `forward_model: fpm`; the adjacent same-stem `.metadata.json` sidecar is required. Relative paths are anchored to the working directory when the engine is constructed. Preserved per role in recommendations, candidate YAML, and regular prefill/decode companions in AFD+PD. |

The supported `estimation_mode`, `fallback_policy`, `estimator_config`, and
`fpm_profile` controls pass through the CLI into the same native constructor.
The CLI and Sweeper reject `fpm_learned` because their phase timing interface
lacks complete iteration features; the same restriction applies to global and
role-specific settings. The generated configuration schemas advertise only the
four supported modes above.
A role's `timing.estimator_config` replaces the global dictionary for that role;
include all required controls in that override. Legacy `timing.forward_model: fpm`
and `timing.fpm_parquet_path` normalize to FPM interpolation and its external pair.
The mode does not fall back to op-level on a missing measured cell.

## Flat engine precision and execution controls

These additional public YAML fields are concrete controls:

| Field | Meaning |
| --- | --- |
| `engine.gemm_quant_mode`, `moe_quant_mode`, `fmha_quant_mode`, `comm_quant_mode` | Explicit precision overrides; omitted values follow model/profile normalization. |
| `engine.kvcache_quant_mode` | KV storage precision: `bfloat16`, `fp8`, or `int8`. Omission follows model/profile normalization and can select FP8 for quantized checkpoints. |
| `engine.attention_backend`, `moe_backend`, `moe_kernel_source` | Runtime graph/backend and exact collected source selectors, subject to canonical validation. |
| `engine.enable_eplb`, `wideep_num_slots` | Expert load balancing and redundant expert slots; require compatible MoE execution. |
| `engine.nextn`, `nextn_accepted` | Compute-side MTP identity and separate expected acceptance assumption; both are required for explicit MTP. |
| `engine.enable_chunked_prefill` | Override chunked prefill; omission preserves the backend default. |

These controls require default timing on language roles. AFD and encoder-pool
configurations of either mode reject the unsupported controls. Use `--stack engine`
and the supported model/backend; an older external adapter can reject them at
capability validation before Replay. Expected acceptance is workload behavior,
not an estimator-fitting observation. See [Replay features](../replay/features.md)
for execution and cache combinations.

FP8 tensor-core compute is rejected during estimator construction on pre-SM89
GPUs, including `a100_sxm`. Choose a BF16 checkpoint on A100, or an FP8-capable
GPU with matching backend timing data. This checks compute precision, not weight
storage or KV dtype: weight-only formats that compute in BF16 remain distinct.
FPM also checks the preserved GEMM/MoE identity before selecting a timing cell,
including profiles without an original op-level graph.
Missing hardware metadata still follows the system-spec/data diagnostics.
Serving runtimes may execute FP8 checkpoints through weight-only kernels such as
Marlin on these GPUs; AISimulate does not model that path.

### KV precision and token capacity

An explicit `engine.kvcache_quant_mode` or worker timing override takes precedence,
subject to model/backend and profile validation. An FPM profile supplies its
recorded `kv_cache_dtype`. Otherwise, the first matching rule below applies:

| Checkpoint metadata | Inferred KV precision |
|---|---|
| `kv_cache_quant_algo` is not missing/null, `fp8`, `bfloat16`, or `"none"` (for example, `int8`) | Rejected during checkpoint normalization |
| DeepSeek V4/V4.1 architecture | FP8 |
| `kv_cache_quant_algo: fp8` | FP8 |
| `kv_cache_quant_algo: "none"` | BF16 |
| KV declaration missing/null or `bfloat16`; weight `quant_algo` is `fp8`, `fp8_block`, or `nvfp4` | FP8 |
| Other supported configurations | BF16 |

These are AISimulate policies, not serving-engine defaults. `torch_dtype: bfloat16`
does not prevent FP8 KV inference. For regular language roles using default timing,
pass `--set engine.kvcache_quant_mode=bfloat16` or set it in YAML to require BF16.
The explicit engine field also accepts `int8`, subject to model/backend validation;
this does not make an `int8` checkpoint KV declaration valid.
Fixed/polynomial timing, EPD, and AFD currently reject this engine override.

Ordinary `predict` output shows each known role's effective KV dtype and source.
JSON stdout and `prediction.json` summaries expose `kv_cache_precision.<role>`
with `dtype` and `source`: `user_override`, `checkpoint` (a matching explicit KV
declaration), `fpm_profile`, or `aisimulate_policy`. `--detail memory` also includes
`kvcache_quant_mode` beside capacity. Naive fallback and roles without a capacity
estimate omit this disclosure. Fixed/polynomial timing roles currently size capacity
at compile time and do not report the resolved precision through this path.

FP8 uses one byte per KV element; BF16 uses two. At the same byte budget, linear
FP8 KV holds twice as many tokens. This does not imply twice the throughput.

## Migrating saved configuration

Use `ForwardPassPerfModelConfig.from_legacy_engine_config(old_config,
worker_type, old_options, allow_regression=False)` to convert a saved flat
EngineConfig and tuning options. It pins the old explicit native mode instead
of changing it to auto. Set `allow_regression=True` only for an old caller that
allowed direct regression fallback; the migration preserves that two-mode
order rather than adding interpolation. Legacy `forward_model: fpm` maps to
`fpm_interpolation`, and `fallback_policy: error` maps to deny. The deprecated
`regression` policy remains readable for these saved direct-fallback requests.
Legacy `extra.fpm_profile` and `extra.fpm_interpolation` migrate to the full
canonical profile and nested interpolation method; newly exported configuration
uses only the canonical fields.

Migration merges saved estimator controls with explicitly supplied legacy
options before applying ordinary defaults. It preserves disjoint settings and
accepts agreeing overlaps; contradictory explicit values report the canonical
setting's path. The same rule applies to legacy and canonical interpolation
methods, including an explicit `auto`. An omitted field does not override a
saved value. `ForwardPassPerfOptions.to_dict()` serializes only arguments
explicitly supplied to that legacy options object, including explicit defaults.

Previously saved CLI timing with `forward_model` retains explicit selection
and deny. Newly authored requests without a selection use auto. `ForwardPassPerfOptions`
is retained as a legacy migration value type; new construction has one complete
config and no separate options argument. The raw PyO3 class also exposes
`normalize_config` and migration helpers for JSON-oriented consumers.

## Decode context parallelism

Op-level DCP declarations are model- and backend-specific: DeepSeek-V3-class
MLA supports vLLM and SGLang, DeepSeek-V3.2/GLM-5 DSA supports vLLM only,
and Kimi-K3 supports SGLang only. Dense/Qwen-MoE GQA on vLLM also needs enough
attention-TP ranks per KV head. These declarations do not bypass precision,
attention-kernel, communication-style or topology checks. Unsupported requests
fail validation; they are not silently priced as DCP1.

These are op-level modeling capabilities. Exact measured whole-forward DCP
profiles have a separate [coverage contract](methods/whole-forward.md#dcp-self-benchmark-profiles);
profiling one backend does not establish op-level or serving support in another.
