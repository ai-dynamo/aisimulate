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
from aisim_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

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

### Per-iteration queries

`estimate_forward_pass_time_ms(metrics_by_rank)` prices one iteration from its
per-attention-DP-rank `ForwardPassMetrics` list. Every estimator handles that
list itself: native models price each rank and take the maximum, correction
uses the heaviest rank, and regression and learned models use joint cross-rank
features. A rank is prefill work only when it schedules fresh prefill tokens
(`sum_prefill_tokens > 0`) and decode work when it schedules decode requests.

`estimate_forward_pass_detailed(metrics_by_rank)` returns the same latency with
its composition. Native estimators record one `ranks` entry per input rank, with
its uncorrected `latency_ms` (an iteration that schedules no work on any rank
returns no entries); `max_rank`, `native_latency_ms` and
`correction_factor` explain the reduction. Whole-model FPM ranks carry their
direct lookups in `queries` (see [Direct FPM query evidence](api/python.md#direct-fpm-query-evidence)).
Operation-graph ranks (op-level models and DeepSeek-V4.1 stage graphs) carry
`operations` instead: name-folded entries with `name`, `latency_ms`,
`energy_wms`, `source` and `details` (SOL comparison or its unavailable reason,
and executed measurement substitutions), the same type as
[static phase diagnostics](api/python.md#static-phase-diagnostics). A
prefill-only or decode-only rank reports the operations of that phase at the
rank's shape. A mixed rank folds every operation that its composition prices:
the combined non-attention pass, context attention at the prefill shape and
decode attention at the decode shape. A stage-graph rank reports the stage
children that it prices. Each rank's entries sum to its latency, before online
correction; reporting them does not change any estimate. The field is additive,
omitted when empty, and absent for whole-model FPM. Learned and regression
estimators have no rank entries.

### Engine identity controls

Canonical JSON/YAML requires string values for text fields. Numeric-looking YAML
identities must be quoted to preserve their exact spelling:

| Field | Default | Input requirement |
| --- | --- | --- |
| `model` | Required | String model name or path; write a numeric identifier as `model: "123"`. |
| `backend_version` | `null` | Optional string version; write `backend_version: "0.10"` to preserve the trailing zero. |
| `system`, `backend_version`, `tp`, `pp`, `attention_dp`, `moe_tp_size`, `moe_ep_size`, `gemm_quant_mode`, `moe_quant_mode`, `fmha_quant_mode`, `kvcache_quant_mode`, `comm_quant_mode`, `attention_backend` | As above | With `estimation_mode` `fpm_regression` or `fpm_learned`, each may be the literal `"unknown"` for a fact the saver did not know. Every other mode, a `fallback_policy` other than `deny` and an `fpm_profile` reject it. The integer fields accept only the string spelling. |
| `deployment_limits` | absent | Optional mapping of the calibrated deployment's `num_gpus` (GPUs of one worker, `tp * pp * attention_dp`), `max_num_batched_tokens`, `max_num_seqs`, `max_kv_tokens` (KV capacity of one attention-DP rank, in tokens) and `context_length`. Each is a positive integer or `"unknown"`; an omitted entry is `"unknown"`. Absent, it is omitted from the serialized configuration. Accepted in every mode. |

This is an intentional compatibility change for direct Rust YAML parsing:
earlier parsing accepted numeric scalars for string fields, while canonical
normalization now rejects them. Quote numeric, Boolean, or null-like values
when they represent text. An actual `backend_version: null` still means unset.

Regression and learned models read no performance data, so their identity only
records the deployment they were calibrated on. A producer such as the Dynamo
planner, which may not know that deployment's hardware, versions, parallelism or
precision, records `"unknown"` instead of a guess. In memory an unknown integer
holds `UNKNOWN_SIZE` (`u32::MAX`, Rust `aisim_core::UNKNOWN_SIZE`), which
serializes back to `"unknown"`, so the configuration round-trips exactly;
`unknown_identity_facts()` lists them. Unknown widths are not checked against
each other (`tp * attention_dp * cp_size == moe_tp_size * moe_ep_size`). An
unknown `tp`, `pp` or `attention_dp` neither conflicts with nor derives an
engine launch's `tensor_parallel_size` and `dp_size`, and replay takes the
engine's own topology for it. `deployment_limits` is identity too: estimates
never read it and snapshot restoration compares it exactly. Replay and
`predict` read only a known `max_kv_tokens`, as the KV capacity of a regression
or learned role without explicit capacity (see
[engine replay timing](#engine-replay-timing)); they do not check the worker's
shape or scheduler limits against the other limits. Recommendation of such a
snapshot role also requires its GPUs per replica to equal a known `num_gpus`
and caps its `max_batched_tokens` and `max_sequences` domains at
`max_num_batched_tokens` and `max_num_seqs` (see
[snapshot roles](#recommending-with-a-calibrated-snapshot)). Nothing reads
`context_length`.

The canonical configuration carries the following AFD identity fields. Ordinary
agg configurations omit them; AFD derives pool identities through the same
`best_available` construction interface.

| Field | Default | Behavior |
|---|---|---|
| `afd_routing_location` | absent | AFD passes `a` to both pools. An explicit `f` fails before estimator/data preparation. |
| `afd_pool_role` | absent | `a` emits attention, shared and dense computation; `f` emits routed computation. Used for selective pool compilation and retained in canonical/cache identity. |

Selective pool compilation requires SGLang, PP1/CP1/DCP1 and op-level estimation.
Every AFD pool carries its role and a typed source layer plan. See
[AFD execution plans](../sweeper/afd-topology.md#performance-model) for ownership
and [AFD configuration](../replay/engine/analytical.md#afd) for public YAML.

The canonical configuration also carries quantization overrides and `attention_backend`, `moe_backend`, `moe_kernel_source` (default absent), `enable_eplb` (default `false`), and `wideep_num_slots` (default absent). These controls reach model construction, KV memory sizing, and replay provenance. EPLB/slots and nondefault MoE backend or kernel-source selection require an MoE model. Collected FPM interpolation cannot represent EPLB, slots, MoE backend, or kernel-source overrides; it rejects an explicit incompatible request and is skipped during automatic selection for those identities.

`encoder_parallel` (default absent) compiles a VL model's vision tower in the given layout, `tp` (sharded over the tensor-parallel group) or `dp` (replicated, images split across ranks), so the estimator can price encoder calls. It requires a model with a vision encoder, is omitted from the serialized payload when unset, and is part of the estimator identity, cache keys and provenance. Llama 4 rejects `dp`.

`moe_kernel_source` selects an exact, nonblank collected `kernel_source` label for fused MoE compute. It is distinct from the existing `moe_backend` graph/backend control; source labels are not backend aliases and are preserved without trimming. `None` keeps the existing default source-selection policy, including eligible low-latency NVFP4 selection. `SILICON` reads only the requested source's table; `EMPIRICAL` derives its estimate from that same source; `HYBRID` may fall back to empirical estimation within that source, but does not substitute a different source. Missing source data remains an error. An explicit `moe_torch_flow_min_latency` requires gated NVFP4 and at most 128 tokens after attention-DP gathering. Pure-roofline `SOL` remains table-independent and does not claim measured support for the requested source.

`moe_perf.parquet` may include Boolean selection metadata `default_eligible`. An absent column preserves legacy automatic selection; when present, every value must be a non-null Boolean. A `false` row requires a nonblank string `kernel_source`, preserved exactly, and is available only through that named source. It cannot enter automatic standard or low-latency grids, including empirical cross-shape and cross-quant reference selection. Among eligible rows, existing source priority and first-row precedence remain unchanged. Default table views, coverage, and readiness use those same eligible grids; raw Parquet enumeration retains all measured rows. Malformed eligibility metadata is an invalid-data error, never a missing-data fallback. The flag is not a measurement identity dimension. Collector finalization preserves it during merges, including existing annotations when a legacy recollection omits the column; genuinely new legacy keys remain eligible.

An explicit source is rejected for dense graphs, MegaMoE modules, large-EP expert-compute graphs, and any non-AFD timing phase with no compatible fused MoE operator. In AFD, the A pool retains the selector in its canonical configuration but emits no routed MoE operation; the F pool applies it to routed MoE operations. It is also incompatible with whole-forward FPM, including the legacy Task `forward_model='fpm'` rewrite. Invalid graph/source combinations fail as invalid configuration rather than triggering estimator fallback. An untrained `fpm_regression` model remains not-ready; retaining a source in its configuration is not evidence of source-specific prediction support.

Ordinary prefill/decode companions of AFD roles use the canonical timing provider
and preserve exact-source requests under the same eligibility rules. Fixed or
polynomial timing and whole-forward FPM reject incompatible source selectors.
These controls describe standalone AISim behavior, not downstream Dynamo
planner integration.

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
`ForwardPassMetrics` for prediction and feedback. Engine replay, engine CLI
prediction and recommendation, and Sweeper timing price every pass of a learned
role with one iteration query, adding the per-request lengths an artifact
reads (see [engine replay timing](#engine-replay-timing)). See
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
use aisim_core::{
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
configured verification width internally. Here `s` is the attention length
including the current token: static decode evaluates one step at
`input_tokens + 1`, so `static_phase_latency(batch_size=B, input_tokens=s-1,
output_tokens=2, prefill=False)` prices the same step as the packed counters.
This OP telemetry convention is distinct from the [STP FPM speculative approximation](methods/whole-forward.md#estimate-speculative-decode-from-an-stp-fpm):
that per-query `draft_length` option takes logical decode request counts and
total past KV, widens the former and preserves the latter.

The canonical constructor currently supports explicit speculative schemes with
OP/SOL timing. FPM interpolation and regression are unsupported candidates for
these schemes. An explicit FPM request with `fallback_policy="deny"` fails;
`auto` or an explicit mode with `fallback_policy="allow"` can select a supported
OP candidate while preserving the complete speculative identity. They do not
drop speculation to obtain an autoregressive estimate. Separately, an ordinary
STP FPM can estimate pure speculative decode via the query-only `draft_length`
option without setting `speculation` or `nextn`. Explicit measured
[speculative-round profiles](methods/whole-forward.md#recorded-speculative-round-identity)
have their own identity contract.

The two FPM speculative paths keep separate query rules:

- **Path 1** scales an ordinary STP FPM estimate by the empirical
  `draft_length` formula and accepts decode-only ranks. As in every rank
  dispatch, a rank is prefill work only when it schedules fresh prefill tokens
  (`sum_prefill_tokens > 0`). A fully prefix-cached prefill record is therefore
  decode-only.
- **Path 2**, a measured end-to-end SD-round profile, rejects `draft_length`
  because its measurements already include speculative decoding: "a measured
  speculative-round profile already includes speculative decoding; omit
  draft_length". Without the option it prices one attention-DP rank. A mixed
  rank costs its measured prefill plus a full measured decode round, not a
  marginal decode share, because the measured rounds have no pass baseline.

The legacy hybrid route, `forward_model="fpm"` with materialized OP draft
operations, is deprecated and emits a construction warning. Existing hybrid
calculations remain available during migration, and pure OP speculative decoding
remains supported. The [migration guidance](methods/whole-forward.md#hybrid-fpmop-deprecation-and-migration)
describes Option 1 (an ordinary non-SD FPM queried with `draft_length`) and
Option 2 (a measured complete SD-round profile). Transitional behavior preserves
applicable OP draft estimates; Option 1 is currently an explicit query fallback,
not an automatic conversion by `best_available` or Replay. Lower-level hybrid
support does not imply support through the canonical ForwardPassMetrics API.
Removal depends on replacement readiness for affected callers and workloads;
there is no fixed removal date.

Acceptance assumptions and scheduler seeds belong to the prediction/Replay
layer, not this cost configuration. The returned model continues to estimate
iteration latency; the existing calls remain supported. For
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
  Optional `fit.linear` controls fitted axes, signed slopes, and lazy updates,
  or opts into the `fpm5` aggregate feature set described below;
  omission preserves the existing linear behavior and serialized defaults.
  `singular_ridge_scale` defaults to `1e-9` and applies to the shared linear
  fit only when retrying a singular equation; the `fpm5` fit always applies it
  to its slopes. `rebuild_interval` defaults to JSON `null` / Python
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
- `temporary_mixed_composition`: absent by default. An explicit learned-model
  policy for original self-benchmark A/A+C workloads. It pins missing mixed
  classes to the configured FPM interpolation method and optionally independent hybrid correction;
  it does not change automatic estimator selection. See the
  [mixed policy configuration](fpm_ml/configuration.md#temporary-mixed-composition-for-self-benchmark-seeds).

| Temporary mixed policy field | Default | Behavior |
| --- | --- | --- |
| `estimator_config.temporary_mixed_composition` | Absent | Opt-in original-A fallback eligibility for aggregated GBDT A/A+C, frozen hybrid A+C, and selector A+C. Requires `scenario`, `mode` and `original_a.manifest_path`. A+C GBDT can require a continuous quality check before selecting ordinary predictions. Frozen hybrid calibrates interpolation for missing mixed stores. For eligible selector classes, calibrated interpolation replaces the frozen/calibrated expert and competes causally with online GBDT after its C bootstrap. |
| `estimator_config.temporary_mixed_composition.original_a.source_sha256` | `null` | Pins the original-A manifest; normalization verifies or fills its digest. C updates cannot change its capability decisions. |
| `estimator_config.temporary_mixed_composition.calibration` | Hybrid; `[4, 4]` bins, 64 slots per policy/class, 5 observations, factors `[0.5, 2.0]` | Independent attention/MoE correction for missing mixed classes in A+C. Its fixed score bounds come from original A and do not reuse ordinary correction geometry. |

The optional `temporary_mixed_composition.quality_gate` checks eligible A+C
mixed classes on standalone online GBDT. Selector retains its existing causal
gate; frozen hybrid retains calibrated interpolation. Configuring this quality
gate on either of those adaptations is an error.

| Mixed quality gate field | Default | Behavior |
| --- | --- | --- |
| `estimator_config.temporary_mixed_composition.quality_gate` | Absent | Preserve existing availability-first GBDT and causal selector behavior. An object enables the check for A+C only. |
| `estimator_config.temporary_mixed_composition.quality_gate.alpha` | `0.05` | EWMA weight in `(0, 1]` for paired, unbounded fractional absolute percentage errors. The first pair initializes each mean. |
| `estimator_config.temporary_mixed_composition.quality_gate.min_comparisons` | `5` | Positive number of future paired admitted observations required before GBDT may qualify. Bootstrap training rows do not contribute. |

For each forest store and critical-rank query type, interpolation stays selected
until GBDT is ready and its EWMA error is less than or equal to interpolation's.
The check continues after selection and returns to interpolation when its error
is lower. Set `min_comparisons: 1` to qualify on the first subsequent paired label.
Predictions for each comparison precede both expert updates; the resulting choice
applies to future queries. The fallback prediction includes its current independent
calibration. Both experts continue learning even while unselected. This checks
past performance and cannot guarantee lower error on future labels. A typed
interpolation coverage miss can use ready GBDT without advancing paired evidence;
route metadata marks this coverage exception. Hard prior errors remain errors.
C-only has no original-A interpolation table and is unaffected.

`append_mixed_learned_checkpoint` supports online GBDT and selector, preserving
calibration and quality evidence as a missing class becomes ready. Mixed snapshots
include the quality gate and reject mismatched or incomplete saved state. They
must be restored together with the matching ordinary learner and admission cursor;
a mixed snapshot alone is not a full model checkpoint.

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

| Regression linear fit field | Default | Behavior |
| --- | --- | --- |
| `estimator_config.fpm_regression.fit.linear` | Absent | Absent keeps the existing attention/MoE fit with nonnegative slopes and eager updates. Rejected with `fit.kind: spline`. |
| `estimator_config.fpm_regression.fit.linear.feature_axes` | `[attention, moe]` | One to six distinct fitted coordinates, independent of `sampling.axes`. |
| `estimator_config.fpm_regression.fit.linear.non_negative` | `true` | `false` permits signed slopes; the intercept is always free. |
| `estimator_config.fpm_regression.fit.linear.update_policy` | `{kind: always}` | `error_threshold` defers coefficient publication until recent prediction errors exceed its tolerances. |
| `estimator_config.fpm_regression.fit.linear.feature_set` | Absent | `fpm5` fits the 17 cross-rank aggregates of the learned `fpm5` preset instead of `feature_axes`, which must keep its default. Requires `sampling.max_observations >= 18` and a positive `fit.singular_ridge_scale`. Omitted from serialization when absent. |
| `estimator_config.fpm_regression.fit.linear.max_kv_splits` | Absent | KV splits the decode-attention kernel gives a request that runs alone: the GPU's SM count over the kernel's CTAs per request. Positive, and required when `feature_axes` or `sampling.axes` selects `longest_split`. Omitted from serialization when absent. See [engine-constant axes](methods/online-regression.md#engine-constant-axes). |
| `estimator_config.fpm_regression.fit.linear.max_cudagraph_capture_size` | Absent | The most scheduled tokens a rank runs in a CUDA graph, as vLLM's knob of this name. Positive, and required when `feature_axes` or `sampling.axes` selects `eager`. Omitted from serialization when absent. |

`feature_set: fpm5` fits the same named aggregates as the learned
`--features fpm5` preset: active-rank, request and token counts and sums,
per-rank maxima, means, the balanced-partition prefill attention proxy, and
their `log1p` forms. It needs only the scalar scheduled counters, not request
lists. Each fit refits the retained rows of its store with nonnegative least
squares (or signed least squares) on standardized features, always adding
`singular_ridge_scale` to the slopes. Constant features and exact duplicates,
such as `max_rank_*` with one attention-DP rank, are inactive. A store becomes
ready only with at least `max(min_observations, active features + 1)` retained
observations; until then the readiness report gives `insufficient_data` below
`min_observations` and `no_usable_fit` above it. Once ready, a store keeps
serving its last fit when a refit cannot produce a usable one, for example when
multi-rank traffic first activates more aggregates than it has retained rows.
`sampling.axes` still selects
retention coordinates, and `update_policy` and `non_negative` keep their
meaning. `rebuild_interval` must stay `null` because every fit already uses the
exact retained rows. See [online regression](methods/online-regression.md#fpm5-aggregate-features).

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
reads that file with `state_snapshot.path` naming `path`, wherever the
directory now is, and passing the returned configuration is equivalent while
the snapshot stays where it was saved. There is no separate loader;
`best_available` restores whenever `estimator_config.state_snapshot` is present.

```python
config = model.save_state_snapshot("/tmp/decode-snapshot")
frozen = RustForwardPassPerfModel.best_available(state_snapshot="/tmp/decode-snapshot")
config["estimator_config"]["state_snapshot"]["on_tune"] = "continue"
resumed = RustForwardPassPerfModel.best_available(config)
```

| Field | Default | Behavior |
| --- | --- | --- |
| `estimator_config.state_snapshot.path` | required | Snapshot directory written by `save_state_snapshot`, at its save location or wherever it was copied or moved. Normalization makes it absolute against the working directory and removes `.` and `..` lexically; symbolic links are kept. A path written in [`ais_perf_model`](#unified-perf-model-input) is made absolute before that, by the rule of its [relative paths](#relative-paths), which follows the links in the folders of the path before `..` is applied. The inputs the snapshot owns are read from this directory (see [relocation](#relocating-a-snapshot)). |
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

Models with `estimator_config.temporary_mixed_composition` reject portable
snapshot saving and restoration because this format does not package their
interpolation prior and additional source/state dependencies. See the
[mixed-policy persistence contract](fpm_ml/mixed-workload-fallback.md#diagnostics-persistence-and-compatibility)
for online checkpoint and partial mixed-state alternatives.

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
  differs: snapshots load only in the same `aisim-core` version with the
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
loads that systems root; later constructions reuse that result. Performance
tables that another engine in the process parsed before then are not shared
with the model, so it reads the fingerprinted data; restoration does the same
after its check. Only tables an earlier restoration in the process loaded from
the same data stay shared, and only while no fingerprinted file or directory
has been written since, even back to its original bytes, and no symbolic link
followed on the way to the data, including links inside link targets, has been
retargeted: tables parse on first query, so they could otherwise hold rows from
data changed in between.
`save_state_snapshot` fingerprints the data again and fails if it changed
since it was loaded; restart the process to save a model of the current data. Pass
`include_perf_db=True` (Rust: `StateSnapshotOptions { include_perf_db: true }`)
to copy exactly the fingerprinted entries into the snapshot, pointing
`systems_paths` (and `fpm_parquet_path`) at the copy. The hardware YAML is
copied unchanged, so packing requires its `data_dir` to be a relative path of
plain directory names; a `data_dir` with `..` components, an absolute path or
one outside the systems root can be referenced but not packed. A packed snapshot holds a
copy of the system's whole data directory, so its size follows that data.
Learned forests and online checkpoints are always copied. The model
configuration is not: native restoration reads `model` again, from a
configuration bundled with the package, a local directory or Hugging Face, so
a packed native snapshot still needs it, and restoration fails if it compiles
a different engine specification from the saved one.

#### Relocating a snapshot

A snapshot directory can be copied or moved, for example to another host or
mount point, and restored after the original is deleted. The manifest names
the inputs a snapshot owns (packed performance data under `perf_db/`, a
copied execution profile, learned artifacts and online checkpoints) by their
fixed paths inside the snapshot, and construction reads them from the
directory that `state_snapshot.path` names, whatever paths the rest of the
configuration gives for them. Restore a moved snapshot with
`best_available(state_snapshot=new_path)`, or set
`estimator_config.state_snapshot.path` in a saved configuration;
`config.json` and the returned configuration record the absolute paths at the
save location. The manifest pins the content of every owned input, so a copy
is verified exactly like the original.

Where external performance data is found is not identity. A snapshot saved
without `include_perf_db` reads its data through `systems_paths` (and
`fpm_parquet_path`), which may name any root with the same content: the
fingerprint decides. `best_available(state_snapshot=path)` keeps the roots
recorded at save time, so on another host either save with
`include_perf_db=True` or pass a configuration with `systems_paths` replaced.
Regression and learned estimators read no performance data, so their
`systems_paths` are not compared. Every other field, such as the model,
hardware, backend version, parallelism, `kv_block_size`, quantization and the
estimator controls, must still equal the saved identity. A snapshot is
self-contained only when saved with `include_perf_db=True`: re-saving a model
restored from a packed snapshot without it references that snapshot's packed
data as external data.

The manifest schema version is 2. Version 1 manifests, written before
snapshots were relocatable (for example by nightly
`0.13.0.dev202610280000000086`), recorded owned inputs as absolute paths below
the save location. A snapshot restores only in the aisim-core version
that wrote it, so a version 1 snapshot from a published nightly cannot be
relocated: restore it in place with the build that wrote it, or tune and save
a new snapshot with a current build, which rejects the old one with an error
naming the writer's version and manifest schema version. A current build still
reads a version 1 manifest that carries its own version string, such as one
from an unstamped local `0.13.0` build, and relocates it by recovering the save
location from the owned inputs' fixed relative paths. Newer schema versions
are rejected with the versions this build reads.

#### Errors, replay and inspection

Snapshot I/O failures raise an `OSError` subclass (for example `FileNotFoundError` for
a missing snapshot or a missing learned artifact or checkpoint inside it)
through both `state_snapshot=` and a configuration, never the skippable
`PerfDataNotAvailableError`. A manifest from another release
reports its versions even when it has fields this release does not know.
Online learner state, including the causal selector's adaptive expert, is
always saved in full, so its size grows with the forest and its retained
training history. Native restoration recompiles through Python like any
native construction. A `from_learned`
model has no deployment identity: its snapshot configuration carries the
placeholder identity `model: from_learned`, `system: from_learned`,
`backend: vllm`, restoration checks only its estimator controls and files, and
`deployment_topology()` returns `None`.

Engine replay restores a snapshot named by its AIC timing configuration like
any other `best_available` call. Its saved online correction applies through
[engine replay timing](#engine-replay-timing) as in any other estimate, the
same with and without `capture_performance_diagnostics`. Operation evidence
and its energy scale with the correction, so reported power is unchanged.

Gates for replaying a saved model read the snapshot without loading it.
`RustForwardPassPerfModel.inspect_state_snapshot(path)` returns the snapshot
kind, versions, pinned identity, deployment topology and the readiness report
recorded at save time. `model.readiness_report()` lists, for every logical
workload store of the worker role, whether an estimator route is available
and why not (`insufficient_data`, `no_usable_fit` or `untrained_forest_store`),
plus every correction store's readiness; native estimators are always ready.
Eligible mixed interpolation routes count as ready even when their learned
forest store is absent, consistently with `regression_store_diagnostics()`.
Readiness does not guarantee coverage for every query: an interpolation route
can still return no estimate outside its support. Portable snapshots reject
the temporary mixed policy, so snapshot inspection does not report mixed
fallback readiness.
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

### Engine replay timing

Engine replay prices each forward pass of a worker with one
`estimate_forward_pass_time_ms()` query, or `estimate_forward_pass_detailed()`
when it needs rank evidence, holding one `ForwardPassMetrics` per attention-DP
rank, idle ranks included. Every rank of the worker ends at the shared time
from its own compute start. An online correction applies as in any other
estimate: the correction of the heaviest rank's workload multiplies the
slowest rank's native time. A one-rank pass that computes only prefill or only
decode takes that phase's correction; a vLLM or TensorRT-LLM rank that computes
both runs as one forward and takes the correction of that mixed iteration.

Replay prices a pass this way only while `decode_speedup_ratio` is 1, its
default and the only value MTP (`nextn > 0`) accepts: the shared time then
takes `speedup_ratio` whatever work it holds. With another decode speedup,
each rank prices its own phases, prefill at `speedup_ratio` and decode at
`speedup_ratio * decode_speedup_ratio`, and the pass ends with the slowest
rank, exactly as before the canonical query. A pass on which no rank computes
model work is never sent to the model and its forward takes no time, so a
scheduler round without model work, such as an SGLang prefill-decode interval
round, keeps its zero duration.

Replay fills each rank's metrics in the selected estimator's convention.
Prefill always passes the scheduled totals. Whole-model FPM reads decode past
KV, excluding each request's current token and clamped to the scheduler's KV
bound. Operation-level timing reads packed decode counters: `B*v` requests at
`s` tokens each, where `s` is the mean context including the current token and
`v` the configured verification width. A tuned operation-level model therefore
looks up a decode pass's online correction at `(B*v, B*v*s)`, where per-rank
phase timing looked it up at `B` requests and their past KV tokens.

Against the former per-rank phase timing, the canonical query changes these
results: an operation-level vLLM or TensorRT-LLM pass that computes prefill and
decode is the model's mix step instead of a prefill pass followed by a separate
decode pass; operation-level prefill reads the per-request means `new / B` and
`prefix / B` instead of `(new + prefix) / B - prefix / B`; a tuned
operation-level model corrects decode at the packed coordinates above; with
attention DP, the correction comes from the heaviest rank and applies to the
slowest rank's time; and a worker's pass ends at the latest rank compute start
plus the shared time. A one-rank pure prefill or pure decode pass whose prefix
tokens divide evenly across its requests keeps its previous latency and
evidence on whole-model FPM, and on an operation-level model without online
corrections.

A measured SD-round profile prices exactly one rank. A rank that computes
prefill and decode costs the measured prefill plus a full decode round. Without
online corrections that is the total of the former separate prefill and decode
passes; a tuned model applies the correction of that mixed iteration to the
whole forward, where the separate passes each took their own phase's
correction.

Operation-level power evidence comes from the same detailed estimate: the
`operations` of every rank with work, stretched by the pass's correction like
its latency, so a pass's energy sums over its ranks as it did when each rank
priced its own phases. A rank's evidence is the work that the query priced, so
a mixed rank reports its mix step's operations rather than separate static
prefill and decode phases. A rank with one kind of work is recorded with that
phase. A mixed rank is attributed by mix-step pass: context-attention
operations are prefill, decode-attention operations decode, and each operation
of the shared non-attention pass splits by the rank's token share,
`ctx_tokens / (ctx_tokens + gen_tokens)` prefill, with `ctx_tokens` the
scheduled prefill tokens and `gen_tokens` the packed decode requests. When name
folding would merge operations of different passes, as for GLM-5.3 attention,
or the graph composes a mixed rank otherwise, as DeepSeek-V4.1 stages do, the
whole rank splits by that token share. A split scales an operation's latency,
energy and SOL alike, so its power and the rank's totals are unchanged.
Operation details such as SOL are kept only while
`capture_performance_diagnostics` is set, as for phase evidence.

Captured whole-model FPM evidence records the estimate of every rank with
work, at its own queries and native latency under the pass's correction, as
each rank recorded its own estimate under per-rank phase timing; a rank that
computes prefill and decode is reported with decode.

A native estimator keeps per-rank phase timing, each rank pricing its own
phases and the pass ending with the slowest rank, for a `decode_speedup_ratio`
other than 1, bounded decoder replay, v1 execution profiles, hybrid speculative
FPM (operation-level draft tails, or `nextn > 0` without a measured SD-round
profile), a measured SD-round profile with attention DP above one, a scheduler
whose speculative depth differs from the model's, an operation graph whose
measured scopes cannot compose a mix step when a rank computes prefill and
decode, and totals beyond the query's 32-bit counters. Lookup coverage
collection counts every pass, so replay does not reuse a priced iteration while
it is enabled.

Regression and learned estimators (`fpm_regression`, and `fpm_learned` with
any adaptation) have no native engine. Replay prices every pass of their role
with the iteration query and never asks them for a separate phase, so totals
beyond the 32-bit counters are an error. Each rank's metrics follow the
`ForwardPassMetrics` telemetry such models are fitted to, which Dynamo's vLLM
worker reports: prefill passes its scheduled totals, and decode KV is the
requests' past KV, the sum of their computed tokens excluding each request's
current token. That is the whole-model FPM convention, the context lengths
minus one token per decode request, but without the scheduler's KV clamp. When
the model reads per-request detail (a regression request feature axis, or a
learned `req_*`, `slot*` or length-variance feature), each rank also carries
`extend_lengths` and `past_kv_lengths` (each prefill request's new tokens and
prefix, each decode request's verification width and past KV) and the population
variances of the prefill requests' full prompt lengths, as the telemetry
reports them whatever chunk of a prompt the pass computes, and of the decode
requests' past KV; those passes are not reused from the iteration cache. Each
rank's metrics carry its position in the group as `dp_rank`.

A vLLM regression fitted at `nextn > 0` gives a valid estimate only when its
training timings cover complete MTP rounds, drafting and verification, as a
measured speculative-round profile's do; `nextn` records the depth, not how
the timings were obtained. Replay times such a role only at that depth:
the snapshot pins `nextn`, and the engine's `aic_nextn` must equal it. Each
decode request counts once in the totals with its past KV, as in the
telemetry; only its `extend_lengths` entry carries the tokens it verifies: one
plus the drafts the scheduler gave it, fewer than `nextn` when the token budget
or the context limit cuts the round short. No `draft_length` scaling applies.

Construction checks that the role's workload stores are ready: pure prefill
for a prefill worker, pure decode for a decode worker, and pure prefill and
pure decode for an aggregated worker. An aggregated vLLM or TensorRT-LLM worker
also needs locally mixed, since its scheduler computes chunked prefill beside
decode in one rank pass; an SGLang rank runs a prefill batch alone (mixed chunk
is not modeled), so it does not. Every aggregated worker whose engine runs
attention DP above one also needs cross-rank aggregated. A regression store is
ready with a usable fit and a learned store when the forest has it. It rejects
a rank hosting the vision encoder, a speculation scheme, `nextn > 0` for a
learned model or a non-vLLM regression (SGLang's telemetry reports an MTP
decode request's extend as 1), bounded decoder replay, and an automatic selection that
would fall back to an untrained regression. Replay also rejects such a role
whose engine sets a `decode_speedup_ratio` other than 1: replay then prices
each rank's phases, which these estimators never do. A pass on which no rank
computes model work takes no time without a query, as for every estimator; so
does an SGLang decode step whose page reservation failed, which a native
estimator is still asked to price at no charge. These estimators are
latency-only: they record no operation evidence, so replay reports
`power_w: null` as for other latency-only timing.

Without explicit KV capacity, a regression or learned role takes
`deployment_limits.max_kv_tokens` in whole scheduler blocks. Without a
recorded capacity the AIC memory estimate needs the complete identity, so an
identity with unknown facts must then set `num_gpu_blocks`. A `tp` or
`attention_dp` recorded as `"unknown"` takes the engine's, and a `pp` recorded
so takes the engine's single pipeline stage, including for `kv_cache.kvcr`.
Replay reads no other `deployment_limits` entry.

## Unified perf-model input

`ais_perf_model` is one value that describes the perf model of each worker
type. You write it once and use it everywhere: in the `aisim perf-model`
tools, as `engine.ais_perf_model` in `predict` and `recommend` YAML, and as
`search_space.ais_perf_model` in a Sweeper search.

Dynamo components that use the [loader](api/python.md#ais-perf-model-loader),
`aisim_core.ais_perf_model`, read the same value. A Dynamo release whose
components do not use it may read a different `ais_perf_model` mapping; see
the documentation of that release.

```yaml
# perf-model.yaml: one entry per worker type
- worker_type: prefill
  from: {dataset: {root: /opt/aisim-data, manifest: metadata/prefill-manifest.json}}
- worker_type: decode
  model: Qwen/Qwen3-32B
  system: h200_sxm
  backend: vllm
```

Check a file before you use it. `check` builds every entry in it:

```bash
aisim perf-model check ./perf-model.yaml
```

A consumer builds only the entry for its own worker type, so run `check` in
CI. See [perf-model tools](../reference/cli.md#perf-model) for `check`,
`migrate` and `fetch`.

### Entries

The value is one perf-model config, or a list with one config per worker type.
Each config is an entry.

| Key | Meaning |
|---|---|
| `worker_type` | `prefill`, `decode`, or `aggregated`. Required in a list, and each worker type may appear once. One config alone may leave it out; the consumer then fills in its own worker type. |
| `from` | Optional. Starts the entry from a template file, a dataset, or a state snapshot. See [Start an entry with `from`](#start-an-entry-with-from). |
| Any other key | A `ForwardPassPerfModelConfig` field, such as `model`, `system`, `backend`, `tp`, `estimation_mode`, or `estimator_config`. |

- A key written as `null` counts as not written. So does an empty
  `systems_paths` or an empty `estimator_config`.
- A `null` inside `estimator_config` is a value. `transfer_policy: []` is a
  value too.
- An entry with only `worker_type` is valid.
- An empty config, an empty list, a repeated key, and an unknown or repeated
  worker type are errors.
- A key written twice is an error at every depth of the value, in these
  places:
  - a perf-model file, a template file named by `from`, and inline JSON;
  - the value written under `engine.ais_perf_model` or
    `search_space.ais_perf_model` in a `predict`, `recommend` or search YAML
    file, when `aisim predict`, `aisim recommend`,
    `CorePredictionConfig.from_yaml`, `CoreRecommendationConfig.from_yaml` or
    `SmartSearchConfig.from_yaml` reads the file.

  A configuration file reports it like this:
  `engine.ais_perf_model in predict.yaml: the key 'estimation_mode' is written twice (lines 9 and 11); write it once`.
  `aisim predict` and `aisim recommend` then exit with code `2`
  before they touch the output directory. A key that a YAML merge (`<<`)
  brings in does not repeat a key written next to it.
- A key written twice is not checked in a value given with `--set`. The YAML
  reader keeps the last value there. It does the same for a key repeated
  anywhere else in a configuration file, such as an old estimator field. When
  `ais_perf_model` itself is written twice in its section, the value that is
  kept is the one that is checked. `aisim perf-model migrate` reads a
  configuration file without this check. Python data passed to
  `model_validate` cannot hold a key twice.
- When a consumer needs a worker type the value does not have, the error lists
  the worker types the value does have.

### File or inline JSON

A value given as text, to `aisim perf-model check` or to the
[loader](api/python.md#ais-perf-model-loader), is read like this:

| Value | Read as |
|---|---|
| Starts with `{` or `[` | Inline JSON. |
| Anything else | A file path. A `.json` file is read as JSON, any other file as YAML. |

```bash
aisim perf-model check \
  '{"model": "Qwen/Qwen3-32B", "system": "h200_sxm", "backend": "vllm"}' \
  --worker-type aggregated
```

Inline YAML is not supported. For a file whose name starts with `{` or `[`,
write `./NAME`.

In `predict`, `recommend` and search YAML, write the content directly under
the key. A string there is an error. To reuse a file, write `from: <file>`.

### Start an entry with `from`

```yaml
- worker_type: decode
  from: ./base.yaml          # a template file
  tp: 4                      # the entry wins over the template
```

| `from` | The entry starts from |
|---|---|
| `./base.yaml` | A template file. Keys in the entry win over keys in the file. `estimator_config` is merged mapping by mapping; lists and scalars are replaced. When the file is a list, the entry takes the config with its own `worker_type`. A template cannot use `from` itself. |
| `{dataset: {root, manifest}}` | A [dataset](methods/op-level.md#shared-dataset-layout-and-local-preparation) on local disk. `manifest` is a path relative to `root`. The dataset is prepared once in a workspace folder and reused after that. |
| `{state_snapshot: {path}}` | A saved perf-model state; see [State snapshots](#state-snapshots). Optional `sha256` must match the saved state. Optional `on_tune: continue` lets the restored model keep tuning; without it the saved `reject` stays. |

What you write wins over a template file. Both win over a consumer's
defaults. A consumer adds what it observes only when the entry leaves it out.
It never overrides: a written value that differs from the observed one is an
error that names the field and both values.

A dataset and a state snapshot are fixed. They record the model, hardware and
settings they were built for. Other keys in such an entry are only compared
with the saved values:

- A different value is an error that names the field and both values.
- A key that the source does not record cannot be added.
- `systems_paths` next to a state snapshot is allowed. It says where the data
  is, not what the model is.

### Relative paths

| Where the value is written | Relative paths start at |
|---|---|
| A file: a perf-model file, a template file, or a `predict`, `recommend` or search configuration file | That file's folder |
| Inline JSON, a `--set` override, or Python data passed to `model_validate` | The current working directory |

`manifest` in a dataset `from` always starts at `root`.

The rule covers `from`, the items of `systems_paths` (the word `default` is
kept), and these `estimator_config` fields: `state_snapshot.path`,
`fpm_learned.artifact_path`, `fpm_learned.checkpoint_path`,
`temporary_mixed_composition.original_a.manifest_path`,
`op_level.execution_profile.path` and `fpm_interpolation.fpm_parquet_path`.

Symbolic links are treated the way the system treats them when it opens a
path:

- Links in the folders of a path are followed before `..` is applied. So
  `link/..` is the parent of the folder that the link names.
- A file reached through a linked folder starts its relative paths at the
  real folder.
- The last name of a path is kept as written. A file that is itself a link
  starts its relative paths at the folder of the link, not at the folder of
  the file that the link names. In the value that the loader returns, a data
  folder or a snapshot that is a link keeps its name. The `systems_paths`
  field of a `predict`, `recommend` or search configuration then stores the
  real folder, as it does for the old field.

This differs from the old estimator fields. Their relative paths start at the
working directory. `aisim perf-model migrate` rewrites them to start at
the folder of the converted file. When `migrate` is given the file through a
symbolic link in another folder, as the configuration file or as `FILE`, it
writes them as absolute paths; see
[`perf-model migrate`](../reference/cli.md#perf-model-migrate).

### In predict, recommend and search configurations

```yaml
engine:
  mode: disaggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  workers:
    prefill: {}
    decode: {}
  ais_perf_model:
    - worker_type: prefill
      estimation_mode: fpm_interpolation
      fallback_policy: deny
      systems_paths: [./fpm-data, default]
    - worker_type: decode
      estimation_mode: op_level
```

In this release, `engine.ais_perf_model` and `search_space.ais_perf_model` are
another spelling of the existing estimator fields. Validation rewrites the
entries into those fields. Saved configurations, candidate YAML and
`model_dump()` output show the existing fields and leave `ais_perf_model` out.

An entry has two kinds of keys here:

| Kind | Keys | What happens |
|---|---|---|
| Estimator keys | `estimation_mode`, `fallback_policy`, `estimator_config`, `systems_paths`, `database_mode`, `transfer_policy`, `fpm_profile` | They set the estimator. They are written into the existing fields. |
| Identity keys | Every other key, such as `model`, `system`, `backend`, `tp` or `kv_block_size` | They are only checked. The engine section, or the search space, still states them. A different value is an error. |

#### Worker types

The configuration decides which worker types are active:

| Configuration | Active worker types |
|---|---|
| Aggregated mode | `aggregated` |
| Disaggregated mode | `prefill` and `decode` |
| Several modes in `recommend` or in a search | The worker types of every listed mode |
| Mode left out | `predict`: `aggregated`. `recommend` and search: all three. |

- One config without `worker_type` applies to every active worker type.
- A list, or one config that writes `worker_type`, needs an entry for every
  active worker type and none for an inactive one.

#### Estimator keys

Entries that agree are the engine-level spelling, entries that differ the
per-worker spelling. The rule is applied to each key on its own:

- Every active entry gives the key, and all give the same value: the value is
  written to `engine.<key>`.
- Otherwise it is written to `engine.workers.<role>.timing.<key>` for each
  entry that gives the key.

The example above is the same as `estimation_mode: fpm_interpolation`,
`fallback_policy: deny` and `systems_paths` under
`engine.workers.prefill.timing`, and `estimation_mode: op_level` under
`engine.workers.decode.timing`. A single config without `worker_type` is the
same as the `engine.<key>` fields.

- `fpm_profile` is engine-wide. All active entries must give the same value.
  An entry that leaves it out next to one that gives it is an error too.
- In a search, the engine level is the top-level `SearchSpace` field and the
  per-worker level is `role_estimator_controls`; see
  [Sweeper Python SDK](../sweeper/sdk.md#ais-perf-model).

#### Identity keys

Identity keys use the perf-model field names, such as `tp`, not
`parallelism.tensor`.

| Command | Keys an entry may state | Compared with |
|---|---|---|
| `predict` | Any `ForwardPassPerfModelConfig` field except `afd_routing_location` and `afd_pool_role` | The value the engine section gives that worker, or the field's default when the engine section does not set it. |
| `recommend` | `model`, `system`, `backend`, `backend_version`, `nextn`, and the engine-wide precision and model controls: the five `*_quant_mode` fields, `attention_backend`, `moe_backend`, `moe_kernel_source`, `enable_eplb` and `wideep_num_slots` | `engine.model`; `engine.workers.<role>.hardware`, else `engine.hardware`; `engine.backend`; the `engine` field of the same name, or its default. |
| Search | `model`, `system`, `backend` | `model_name`; `prefill_hardware_sku` or `decode_hardware_sku` when set, else `hardware_sku`; `backend`. |

- `predict` compares `backend_version` as written. An entry that states it
  needs the same text in `engine.backend_version`.
- In `predict`, a key that is not a `ForwardPassPerfModelConfig` field is an
  error when the configuration is validated:
  `engine.ais_perf_model entry for aggregated: estimaton_mode: not a perf model field (did you mean estimation_mode?). Correct the key or drop it from the entry, or from the template file when 'from: <file>' brings it in.`
- In `predict`, an entry that sets `afd_routing_location` or `afd_pool_role`
  is an error at the same point. Only the pools of an AFD worker have these
  fields:
  `engine.ais_perf_model entry for aggregated: afd_pool_role is set, and ais_perf_model does not cover AFD yet. Drop the key from the entry, or from the template file when 'from: <file>' brings it in.`
- `predict` compares the values of identity keys later, when it builds the
  workers of a run. `CorePredictionConfig.model_validate` alone does not
  report a different value. With `--overwrite`, the known outputs are already
  removed by then; see
  [Existing Output Directories](../reference/cli.md#existing-output-directories).
- `recommend` and search need one stated value to compare with. An entry that
  states `system` is an error with `hardware: auto`. An entry that states
  `backend` is an error unless the configuration names exactly one backend.
- A boolean and a number are different values. `tp: true` does not restate
  `tp: 1`, and `enable_eplb: 0` does not restate `enable_eplb: false`:
  `engine.ais_perf_model entry for aggregated does not match the engine section: tp: entry has true, the engine section gives 1. Fix the engine section or drop the key from the entry.`
- In `recommend`, a snapshot entry that states `backend_version`, `nextn` or
  an engine-wide control needs that key set in `engine`. The default is not
  used for it.
- Any other identity key, such as `tp`, is an error in `recommend` and in a
  search. They choose those values for each candidate. Drop the key from the
  entry, or from the template file when `from: <file>` brings it in. A `null`
  in the entry does not remove a template key.

#### State snapshot entries

```yaml
engine:
  mode: disaggregated
  model: Qwen/Qwen3-32B
  hardware: h200_sxm
  workers:
    prefill: {}
    decode: {}
  ais_perf_model:
    - worker_type: prefill
    - worker_type: decode
      from: {state_snapshot: {path: /mnt/planner/decode-snapshot}}
```

A state snapshot belongs to one worker. A snapshot entry is an entry with
`from: {state_snapshot: ...}`, or one that writes
`estimator_config.state_snapshot`. When any active entry is a snapshot entry,
every estimator key is written at the per-worker level, even when the entries
agree. `fpm_profile` stays engine-wide.

- One config without `worker_type` can be a snapshot entry only when exactly
  one worker type is active.
- Name the snapshot once. `from.state_snapshot` together with
  `estimator_config.state_snapshot` is an error.
- The snapshot is then judged like the per-worker field; see
  [Recommending with a calibrated snapshot](#recommending-with-a-calibrated-snapshot).
  Its error messages name
  `engine.workers.<role>.timing.estimator_config.state_snapshot`.

#### Overrides with `--set`

```bash
aisim predict -c prediction.yaml \
  --set engine.ais_perf_model.estimation_mode=op_level
```

- `--set engine.ais_perf_model.<key>=VALUE` changes one config, or creates it.
- A list can only be replaced as a whole.
- `--set engine.ais_perf_model=null` removes the key.
- A relative path given with `--set` starts at the working directory.

### Old and new fields together

State a setting with the old field or with the new key, not with both. A
written old estimator field next to `ais_perf_model` is an error:

```text
engine.ais_perf_model cannot be combined with old estimator fields that say something else: engine.estimation_mode. Remove the old fields.
```

One case is allowed. An old field may hold exactly the value that the entries
produce at the same level. This keeps a saved configuration valid when you add
the key back to it.

`transfer_policy` is compared by the transfer kinds it names, not by spelling.
`engine.transfer_policy: [xshape]` next to an entry with
`transfer_policy: xshape` is the same value. So is a preset name next to the
list of kinds it stands for. A text is read as a preset name, a kind name, or
several of them separated by commas, in any letter case. A list must hold kind
names, written exactly: `[balanced]` and `[XSHAPE]` are not lists of kinds. A
value that does not work on its own, such as `other`, and a list that holds
anything but kind names, is the same value only as a value written
identically. The configuration keeps the spelling of the entry. A search space
follows the same rule; see
[Sweeper Python SDK](../sweeper/sdk.md#ais-perf-model).

An old field that repeats the key still prints the deprecation warning. The
warning then says to remove the old field; see
[Old estimator fields](#old-estimator-fields).

### Limits in this release

- **Dataset entries.** `from: {dataset: ...}` works in
  `aisim perf-model check` and in Python code that calls the
  [loader](api/python.md#ais-perf-model-loader). `predict` and `recommend`
  configurations reject it:
  `dataset entries are not supported in predict and recommend configs yet; use 'aisim perf-model check' or a Dynamo component`.
  A search configuration rejects it with the same message for search configs.
  The message names a Dynamo component: a Dynamo component that uses the
  loader and gives it a workspace folder can take a dataset entry.
- **AFD, encoder pools, fixed and polynomial timing.** These keep the old
  fields. A configuration with an AFD worker
  (`engine.workers.<role>.type: afd`), with `engine.workers.encoder`, or with
  `timing.type: fixed` or `polynomial` on an active worker rejects
  `ais_perf_model`. Its old fields do not print the deprecation warning. The
  same holds for a search with `afd_roles`, with `encoder`, or with a
  `<role>_timing_model`.
- **Snapshot entries in `predict`.** The entry only re-spells the per-worker
  field. `predict` still needs every saved estimator control written out, and
  `engine.backend_version` equal to the saved one. `recommend` takes them from
  the snapshot.
- **Identity keys in `recommend`.** Only the keys in the table above can be
  checked. Every other identity key is an error.
- **Generated files.** Candidate files written by `recommend`
  (`recommendations/NNNN.yaml`) and replay configurations written by
  `aisim onboard` still use the old fields. Reading one back prints the
  deprecation warning. Do not migrate a replay configuration written by
  `aisim onboard`. For such a file the warning ends with
  `('aisim onboard' wrote this file and rejects it once it is changed, so leave it as it is)`
  in place of the `migrate` command. This holds for the `predict.yaml` that
  `onboard` writes next to its `validation.json`, and for the
  `matched/predict.yaml` that serving validation writes in the folder of its
  `recipe.json`, while the file is unchanged and that report is in place. A
  copy in another folder, or a file that was changed, gets the `migrate`
  command.
- **Simulated router.** The key sets the simulated engines. A `router`
  section is read only by the router adapter of a serving-stack integration;
  the `engine` stack rejects a configuration that has one. The adapter defines
  the keys of the section, and AISim hands the section to it without
  reading them.

### Old estimator fields

The estimator fields under `engine` and in the search space are deprecated.
They work as before in this release. The next release removes them. Use
`engine.ais_perf_model` or `search_space.ais_perf_model`.

Loading a configuration file that writes an old field prints one warning that
lists every old field:

```text
ais-perf-model: engine.estimation_mode, engine.workers.decode.timing.systems_paths are deprecated and will be removed in the next release; use engine.ais_perf_model (run 'aisim perf-model migrate prediction.yaml' to convert the file)
```

The command in the warning shows the conversion and writes nothing. Preview
the conversion, then write it:

```bash
# Preview: print the new block and the fields it removes.
aisim perf-model migrate prediction.yaml
# Write: convert the file in place.
aisim perf-model migrate prediction.yaml --output prediction.yaml
```

The preview ends with
`Nothing was written. Pass --output FILE to write the converted config.` The
second command ends with
`Wrote prediction.yaml (comments and formatting of the original file are not kept).`
The converted file loads without the warning. To keep the original file, name
another file after `--output`. See
[`perf-model migrate`](../reference/cli.md#perf-model-migrate).

| Old field | In an entry |
|---|---|
| `engine.estimation_mode`, `fallback_policy`, `estimator_config`, `systems_paths`, `database_mode`, `transfer_policy`, `fpm_profile` | The same key, in one config or in every entry. |
| `engine.workers.<role>.timing.estimation_mode`, `fallback_policy`, `estimator_config`, `systems_paths`, `database_mode`, `transfer_policy` | The same key, in the entry for that worker type. |
| `engine.systems_path: P` | `systems_paths: [P]` |
| `engine.workers.<role>.timing.forward_model: op_level` | `estimation_mode: op_level` with `fallback_policy: deny` |
| `engine.workers.<role>.timing.forward_model: fpm` | `estimation_mode: fpm_interpolation` with `fallback_policy: deny` |
| `engine.workers.<role>.timing.fpm_parquet_path: P` | `estimator_config.fpm_interpolation.fpm_parquet_path: P` |
| Search space, top level: `estimation_mode`, `fallback_policy`, `estimator_config`, `systems_paths`, `database_mode`, `transfer_policy`, `fpm_profile` | The same key, in one config or in every entry. |
| Search space, top level: `systems_path: P` | `systems_paths: [P]` |
| Search space: `role_estimator_controls.<role>.<key>` | The same key, in the entry for that worker type. The `agg` role is the `aggregated` worker type. |
| Search space: `<role>_forward_model: op_level` | `estimation_mode: op_level`, in the entry for that worker type. No `fallback_policy` is added. |
| Search space: `<role>_forward_model: fpm` | `estimation_mode: fpm_interpolation`, in the entry for that worker type. No `fallback_policy` is added. |
| Search space: `<role>_fpm_parquet_path: P` | `estimator_config.fpm_interpolation.fpm_parquet_path: P`, in the entry for that worker type. |

The names of the old search fields are not keys of an entry. An entry that
writes `<role>_forward_model`, `<role>_fpm_parquet_path` or
`role_estimator_controls` is an error such as
`search_space.ais_perf_model entry for aggregated: agg_forward_model, agg_fpm_parquet_path cannot be checked in a search config.`
[Sweeper Python SDK](../sweeper/sdk.md#replacing-the-per-role-fpm-fields)
shows the old and the new spelling of one search side by side.

- With one old field the warning says `is deprecated`. A path with a space or
  a character such as `&` is quoted for the shell, as in
  `migrate 'my configs/prediction.yaml'`.
- A field counts as written when it is present, not `null`, and differs from
  its default. A saved configuration that only lists defaults does not warn.
  `systems_path`, `forward_model` and `fpm_parquet_path` count whenever they
  are set.
- One search field counts also when it is `null`:
  `role_estimator_controls.<role>.transfer_policy`. A `null` there replaces
  the top-level policy for that role with every transfer kind; see
  [Sweeper Python SDK](../sweeper/sdk.md#ais-perf-model).
- `aisim predict` and `aisim recommend` print the warning. So do
  `CorePredictionConfig.from_yaml`, `CoreRecommendationConfig.from_yaml` and
  `SmartSearchConfig.from_yaml`. Validating Python data with `model_validate`
  does not warn.
- When every old field comes from `--set` and the file has none, the warning
  ends with `(set engine.ais_perf_model.<key> with --set instead)` in place of
  the `migrate` command. When the file also has an old field, the warning
  names every old field and ends with the `migrate` command.
- When the configuration also sets `ais_perf_model`, the warning ends with
  `(it is already set, so remove the old fields)`. When the file holds the
  key and old fields, and `--set engine.ais_perf_model=null` removes the key
  for the run, the ending is
  `(the file already sets it, so remove the old fields from the file)`.
  `migrate` does not merge old fields into a key that is already set.
  To set the key again in a run that has `--set engine.ais_perf_model=null`,
  write it as one mapping
  (`--set 'engine.ais_perf_model={transfer_policy: xshape}'`) or drop the
  null override. `--set engine.ais_perf_model.<key>` after the null override
  is an error, also when the warning of that run ends with the `--set` hint
  above.
- For a replay configuration that `aisim onboard` wrote and that is
  unchanged, the warning ends with
  `('aisim onboard' wrote this file and rejects it once it is changed, so leave it as it is)`;
  see [Limits in this release](#limits-in-this-release).
- The warning category is `AisPerfModelDeprecationWarning`, a `FutureWarning`
  in `aisim_core.ais_perf_model`. With `PYTHONWARNINGS=error` the command
  fails with exit code `1`.
- `timing.type`, `timing.prefill_ms` and `timing.decode_ms` are not
  deprecated.

## Unified CLI timing fields

The `engine` block of `predict` and `recommend` YAML is documented in
[Replay engine](../replay/engine/README.md). The per-role `timing` fields below
select the forward-pass timing provider. SDK constructor names differ (for
example, `parallelism.tensor` becomes `tp`). A Default Range of `x` is
concrete-only.

The estimator fields in this table are deprecated, and their rows say so. They
work as before in this release. The next release removes them.
`engine.ais_perf_model` replaces them; see
[Unified perf-model input](#unified-perf-model-input) and its
[list of old fields](#old-estimator-fields). `timing.type`,
`timing.prefill_ms` and `timing.decode_ms` are not deprecated.

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `engine.estimation_mode` | `auto` | `x` | `-` | Deprecated; use [`engine.ais_perf_model`](#unified-perf-model-input). `auto`, `op_level`, `fpm_interpolation`, `fpm_regression`, or `fpm_learned`. Regression and learned roles are priced per iteration (see [engine replay timing](#engine-replay-timing)). |
| `engine.workers.<role>.timing.estimation_mode` | `null` | `x` | `-` | Deprecated; use [`engine.ais_perf_model`](#unified-perf-model-input). Optional role override using the same five modes; omission inherits the global selection. |
| `engine.workers.<role>.timing.type` | `default` | `x` | `-` | `default`, `fixed`, or `polynomial`. |
| `engine.workers.<role>.timing.prefill_ms` | `null` | `x` | `-` | Nonnegative and required for `fixed` timing. |
| `engine.workers.<role>.timing.decode_ms` | `null` | `x` | `-` | Nonnegative and required for `fixed` timing. |
| `engine.workers.<role>.timing.forward_model` | `op_level` | `x` | `-` | Deprecated; use [`engine.ais_perf_model`](#unified-perf-model-input). `op_level` or `fpm`; `default` timing only. `fpm` replays whole-forward (FPM) latency measured for the role's exact model, hardware, backend version, parallel shape and quantization, and fails closed when no such cell exists. |
| `engine.workers.<role>.timing.fpm_parquet_path` | `null` | `x` | `-` | Deprecated; use [`engine.ais_perf_model`](#unified-perf-model-input). External FPM parquet for `forward_model: fpm`; the adjacent same-stem `.metadata.json` sidecar is required. Relative paths are anchored to the working directory when the engine is constructed. Preserved per role in recommendations, candidate YAML, and ordinary prefill/decode companions of AFD roles. |
| `engine.workers.<role>.timing.estimator_config.state_snapshot` | `null` | `x` | `-` | Deprecated; use [`engine.ais_perf_model`](#unified-perf-model-input). `recommend`: `{path, sha256}` of a calibrated snapshot; the role takes its estimator and engine identity from it. See [snapshot roles](#recommending-with-a-calibrated-snapshot). `predict` requires the complete saved configuration, as for any `state_snapshot`. |

AFD replay prices every decode pass at its actual batch, context length and
MTP verification width. The former `estimator_config.afd.frozen_decode_identity`
control and saved pricing identity have been removed. Configurations and saved
ReplaySpecs containing them are rejected for every value, including `false` and
`null`; regenerate them with the current compiler. Static forward queries remain
available for configuration-sized estimates.

The supported `estimation_mode`, `fallback_policy`, `estimator_config`, and
`fpm_profile` controls pass through the CLI into the same native constructor.
The generated configuration schemas advertise the five modes above; a learned
source is read when the canonical model is constructed, not during configuration
validation.
A role's `timing.estimator_config` replaces the global dictionary for that role;
include all required controls in that override. Legacy `timing.forward_model: fpm`
and `timing.fpm_parquet_path` normalize to FPM interpolation and its external pair.
The mode does not fall back to op-level on a missing measured cell.

### Recommending with a calibrated snapshot

A `recommend` role can be timed by a model calibrated online, such as a
Dynamo planner's, by naming only its [state snapshot](#state-snapshots). Native
`op_level` and `fpm_interpolation` snapshots follow the rules below;
[regression and learned snapshots](#regression-and-learned-snapshots) add
their own:

```yaml
engine:
  mode: disaggregated
  model: Qwen/Qwen3-32B
  hardware: h200_sxm
  workers:
    prefill: {}
    decode:
      timing:
        estimator_config:
          state_snapshot:
            path: /mnt/planner/decode-snapshot  # sha256 optional
```

This per-worker field is deprecated. The same role is written as a
[snapshot entry](#state-snapshot-entries) of `engine.ais_perf_model`, and the
rules below apply to it unchanged. They are about the `engine` section. A
snapshot entry may also restate `backend_version`, `nextn` or an engine-wide
control as an [identity key](#identity-keys), but only when `engine` sets it.

A snapshot restores exactly one engine identity, so the role is pinned to it.
Before the search starts, the backend and backend version, the role's tensor,
pipeline, attention-data and MoE parallelism, its KV block size, the
engine-wide quantization and model controls and the MTP depth (`nextn`) are
taken from the snapshot; omit them, or give values that include the saved
ones. A conflicting explicit value fails configuration validation and names
the snapshot's value; an explicit default such as `enable_eplb: false` or
`moe_backend: default` conflicts with a snapshot saved with EPLB or another
MoE backend. Replicas, scheduler limits (`max_batched_tokens`,
`max_sequences`), KV memory, prefix caching and the other roles are still
searched. Scheduler limits are not capped at the snapshot's correction grid:
outside the observed region a fixed correction applies factor 1.0 and a
dynamic or hybrid correction a constant recent ratio. The engine-wide values
also apply to roles without a snapshot, and two snapshot roles must agree on
them. A snapshot with `nextn > 0` needs `engine.nextn` set to the saved depth
and `engine.nextn_accepted`, which a snapshot does not record.

The role takes every estimator control from the snapshot, so `timing` may
name only the snapshot, plus `estimation_mode` equal to the saved mode,
`fallback_policy: deny` and, for a snapshot saved without `include_perf_db`,
`systems_paths` naming one root that holds the referenced data. A packed
snapshot also serves as the role's systems root for enumeration and capacity.
Each candidate restores the snapshot through `best_available`, which compares
its complete engine identity with the saved one and rejects any difference;
the sweeper restores each role's snapshot once per run, and replay once per
candidate. The recommended deployment's prediction YAML keeps the restored
configuration, so `predict` replays the same calibrated model. For a role whose
timing names a snapshot, `predict` takes `decoder_replay`,
`enable_shared_layer`, `strict_provenance` and `fpm_fmha_quant_mode` from the
snapshot unless the engine sets them, and reads packed data from the snapshot
wherever `systems_paths` points.

Configuration validation rejects snapshots replay cannot use: a `from_learned`
model, which records no identity; a native snapshot without an enabled, ready
correction store for the iterations replay prices for the role (prefill, decode,
or any of prefill, decode and mixed for an aggregated worker); a snapshot
without `kv_block_size`; one saved for another worker role, model or hardware;
and snapshots with an FPM profile, speculation, pipeline parallelism (engine
replay runs only `pipeline: 1`) or context parallelism.
`engine.estimator_config.state_snapshot` is rejected: a snapshot belongs to one
role.

#### Regression and learned snapshots

A Dynamo planner without an `ais_perf_model` role configuration saves an
`fpm_regression` snapshot, and learned (`fpm_learned`) snapshots work the same
way. Replay prices every pass of such a role with the saved model (see
[engine replay timing](#engine-replay-timing)), so the snapshot needs a ready
workload store for each kind of iteration the role computes: pure prefill for a
prefill worker, pure decode for a decode worker, and pure prefill and pure
decode for an aggregated worker, plus locally mixed for an aggregated vLLM or
TensorRT-LLM worker and cross-rank aggregated when the role runs attention DP
above one. Its identity facts may be
[`"unknown"`](#engine-identity-controls); the planner should record what it
does not know that way rather than defaults such as `system: unprofiled` or
`tp: 1`:

```yaml
# Saved identity (excerpt)
estimation_mode: fpm_regression
system: unknown
backend_version: unknown
tp: unknown
pp: unknown
attention_dp: unknown
kv_block_size: 16
deployment_limits:
  num_gpus: 1
  max_num_batched_tokens: 8192
  max_num_seqs: 256
  max_kv_tokens: 16384
  context_length: unknown
```

Known identity facts are pinned exactly as for a native snapshot, and a
conflicting value names the saved one. Facts the snapshot does not know are
handled as follows:

| Saved identity | Recommendation |
| --- | --- |
| `system: unknown` | The role runs on `engine.hardware` (or its own `hardware`) without a conflict. |
| `backend_version`, a quantization mode or `attention_backend` `unknown` | Not pinned engine-wide; the caller's value applies to the other roles and the role's estimator identity stays `"unknown"`. |
| `tp`, `pp`, `attention_dp`, `moe_tp_size` or `moe_ep_size` `unknown` | The role needs one explicit value for each unknown width, such as `parallelism: {preset: false, tensor: 1, pipeline: 1, attention_data: 1}`; known widths are still pinned. `tp * pp * attention_dp` must equal `deployment_limits.num_gpus` when known. |
| `deployment_limits.max_kv_tokens` | Required. The role's KV capacity is fixed to that many tokens in whole `kv_block_size` blocks and KV memory is not searched; a different fixed capacity or a `memory_fraction` search is a conflict. |
| `deployment_limits.max_num_batched_tokens`, `max_num_seqs` | Required. The role's `max_batched_tokens` and `max_sequences` domains keep only values at most the recorded limit; an empty domain is an error. |
| `nextn > 0` | For a vLLM regression, pinned as for a native snapshot: `engine.nextn` takes the saved depth, and `engine.nextn_accepted` is required. Rejected for other backends. |

A snapshot whose `backend_version` is unset (`null`) is rejected: save the
deployed version or `"unknown"`. `deployment_limits.num_gpus` and
`context_length` may stay unknown, and `context_length` is not read. A native
snapshot keeps the rules above this section and does not apply its
`deployment_limits`. The exported prediction YAML names the snapshot with the
role's explicit shape and fixed KV capacity, but no systems root, since these
models read no performance data, and no quantization mode the snapshot records
as unknown. `predict` spells the unknown facts as the snapshot does, so it
replays the same model. A `backend_version` alias such as `current` resolves
from the roles that know their version, or on the role's hardware when none
does. Candidate provenance reports the hardware the role ran on when its
identity's `system` is unknown.

## Flat engine precision and execution controls

These additional public YAML fields are concrete controls:

| Field | Meaning |
| --- | --- |
| `engine.gemm_quant_mode`, `moe_quant_mode`, `fmha_quant_mode`, `comm_quant_mode` | Explicit precision overrides; omitted values follow model/profile normalization. AFD resolves GEMM/MoE execution independently against each pool hardware while preserving requested overrides. |
| `engine.kvcache_quant_mode` | KV storage precision: `bfloat16`, `fp8`, or `int8`. Omission follows model/profile normalization and can select FP8 for quantized checkpoints. |
| `engine.attention_backend` | Attention kernel selector, including AFD per-pool construction; subject to canonical model/backend validation. |
| `engine.moe_kernel_source` | Exact collected fused-MoE compute source, including AFD per-pool construction; unsupported with a large-EP MoE graph. |
| `engine.moe_backend` | Runtime graph/backend selector, subject to canonical validation; nondefault overrides remain unsupported for AFD. |
| `engine.enable_eplb`, `wideep_num_slots` | Expert load balancing and redundant expert slots; require compatible MoE execution. |
| `engine.nextn`, `nextn_accepted` | Compute-side MTP identity and separate expected acceptance assumption; both are required for explicit MTP. |
| `engine.enable_chunked_prefill` | Override chunked prefill; omission preserves the backend default. |
| `engine.enable_shared_layer` | Optional shared-data reuse override; explicit `false` disables reuse. Preserved by both AFD pools and their memory preflight. |
| `engine.strict_provenance` | Fail-closed performance-data provenance validation; defaults to `false` in the canonical model. Preserved by both AFD pools and memory preflight. |

For AFD, the original GEMM/MoE precision request is preserved in saved
configuration. Each pool applies the supported hardware execution mapping:
`nvfp4` uses weight-only `nvfp4_wo` with BF16 compute below Blackwell and native
FP4 compute on Blackwell. An explicit `nvfp4_wo` remains weight-only. Unknown or
unsupported hardware execution combinations fail construction; missing measured
data follows the existing data policy without silently changing precision.
Timing and memory/KV sizing share this resolution. Pool provenance reports both
requested overrides and the execution modes present in the compiled graph.

These controls require default timing on ordinary language roles. AFD supports
explicit precision, kernel selectors and MTP; EPLB, redundant slots and nondefault
`moe_backend` remain unsupported in both public YAML and SDK searches. Encoder-pool
configurations reject unsupported controls. Use `--stack engine`
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
Marlin on these GPUs; AISim does not model that path.

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

These are AISim policies, not serving-engine defaults. `torch_dtype: bfloat16`
does not prevent FP8 KV inference. For regular language roles using default timing,
pass `--set engine.kvcache_quant_mode=bfloat16` or set it in YAML to require BF16.
The explicit engine field also accepts `int8`, subject to model/backend validation;
this does not make an `int8` checkpoint KV declaration valid.
Fixed/polynomial timing and EPD currently reject this engine override. AFD passes
explicit KV precision through canonical per-pool construction.

Ordinary `predict` output shows each known role's effective KV dtype and source.
JSON stdout and `prediction.json` summaries expose `kv_cache_precision.<role>`
with `dtype` and `source`: `user_override`, `checkpoint` (a matching explicit KV
declaration), `fpm_profile`, or `aisimulate_policy`. `--detail memory` also includes
`kvcache_quant_mode` beside capacity. Naive fallback and roles without a capacity
estimate omit this disclosure. Fixed/polynomial timing roles currently size capacity
at compile time and do not report the resolved precision through this path.

FP8 uses one byte per KV element; BF16 uses two. At the same byte budget, linear
FP8 KV holds twice as many tokens. This does not imply twice the throughput.

AFD preserves supported controls, including explicit `attention_backend` and
`moe_kernel_source`, through prediction, recommendation, saved replay specifications,
canonical pool construction and memory preflight. Attention uses A; routed MoE
uses F. The A pool retains `moe_kernel_source` in its canonical configuration but
does not apply it to attention, shared, or dense operations. The F pool applies
it to routed MoE operations. Omission preserves pool defaults;
explicit selectors retain the canonical eligibility, data-source and SOL rules
above without changing communication. EPLB, redundant slots and nondefault
`moe_backend` overrides remain unsupported for AFD.

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
