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

The canonical configuration also carries quantization overrides and `attention_backend`, `moe_backend`, `moe_kernel_source` (default absent), `enable_eplb` (default `false`), and `wideep_num_slots` (default absent). These controls reach model construction, KV memory sizing, and replay provenance. EPLB/slots and nondefault MoE backend or kernel-source selection require an MoE model. Collected FPM interpolation cannot represent EPLB, slots, MoE backend, or kernel-source overrides; it rejects an explicit incompatible request and is skipped during automatic selection for those identities.

`moe_kernel_source` selects an exact, nonblank collected `kernel_source` label for fused MoE compute. It is distinct from the existing `moe_backend` graph/backend control; source labels are not backend aliases and are preserved without trimming. `None` keeps the existing default source-selection policy, including eligible low-latency NVFP4 selection. `SILICON` reads only the requested source's table; `EMPIRICAL` derives its estimate from that same source; `HYBRID` may fall back to empirical estimation within that source, but does not substitute a different source. Missing source data remains an error. An explicit `moe_torch_flow_min_latency` requires gated NVFP4 and at most 128 tokens after attention-DP gathering. Pure-roofline `SOL` remains table-independent and does not claim measured support for the requested source.

Selected `prefill_graph_profile` and observed `decode_workload_distribution` profiles require `moe_kernel_source=None`. Their qualified composition and source identity are fixed; an explicit source override is rejected even when its label matches the measured kernel. An absent or null source preserves the approved profile identity and predictions.

`moe_perf.parquet` may include Boolean selection metadata `default_eligible`. An absent column preserves legacy automatic selection; when present, every value must be a non-null Boolean. A `false` row requires a nonblank string `kernel_source`, preserved exactly, and is available only through that named source. It cannot enter automatic standard or low-latency grids, including empirical cross-shape and cross-quant reference selection. Among eligible rows, existing source priority and first-row precedence remain unchanged. Default table views, coverage, and readiness use those same eligible grids; raw Parquet enumeration retains all measured rows. Malformed eligibility metadata is an invalid-data error, never a missing-data fallback. The flag is not a measurement identity dimension. Collector finalization preserves it during merges, including existing annotations when a legacy recollection omits the column; genuinely new legacy keys remain eligible.

An explicit source is rejected for dense graphs, MegaMoE modules, large-EP expert-compute graphs, and any constructed timing phase with no compatible fused MoE operator. It is also incompatible with whole-forward FPM, including the legacy Task `forward_model='fpm'` rewrite. Invalid graph/source combinations fail as invalid configuration rather than triggering estimator fallback. An untrained `fpm_regression` model remains not-ready; retaining a source in its configuration is not evidence of source-specific prediction support.

AFD regular companions currently reject exact-source requests through their legacy estimator and fixed timing paths. The external-FPM companion forwards the source to canonical validation, which rejects the incompatible FPM request. These controls describe standalone AISimulate behavior, not downstream Dynamo planner integration.

`nextn` remains compute-side identity. Expected accepted draft tokens are a
simulator workload assumption, supplied separately by the unified CLI as
`engine.nextn_accepted`; they do not tune the estimator.

### Selection and fallback

`estimation_mode` defaults to `auto`; `fallback_policy` defaults to `deny`.
Auto always searches `op_level`, `fpm_interpolation`, then `fpm_regression`,
including when fallback is denied. For an explicit mode, deny permits only
that estimator; allow tries the requested estimator first, then the remaining
estimators in the same global priority order. Each native mode tries the
ordered `systems_paths` before moving to another estimator. Omitted roots preserve
configured SDK discovery (or the systems-path environment override when the SDK
uses its packaged default); an explicit `default` entry selects the packaged root.
The resolved paths are shared by preflight, construction and capacity estimation.
Selection occurs
at construction; queries do not silently switch estimators on a data-domain error.

Invalid caller configuration does not trigger fallback. A constructed regression
may be unready: nonempty queries return `None` until their selected workload
store has a usable fit. Offline prediction/recommendation reject an untrained
regression instead of fabricating a latency. The current aggregated regression
retains its four workload stores; dedicated roles retain one each.

The returned provenance records the requested and selected modes, failed
selection attempts, effective backend version, data policy, selected root,
and complete estimator configuration. Its resolved config pins the selected
mode with deny so saved replay input repeats that selection.

Prompt-lookup verification uses the same constructor: set `speculation` to
`{"kind": "ngram", "params": {"num_speculative_tokens": 2}}` in Python/JSON, or
`ForwardPassSpeculationConfig::Ngram { num_speculative_tokens: 2 }` in Rust.
It supports vLLM op-level timing with 1–5 draft tokens and `nextn: 0`; auto can
select op-level but cannot fall back to an unsupported speculative estimator.
The cost configuration is retained in provenance and saved recommendations.
Acceptance rates and the scheduler seed stay in the CLI/Replay speculation
configuration; they do not change the model's target-verification graph.

### Estimator controls

`estimator_config` is passed intact through the Python facade, CLI, Sweeper,
and Replay, then parsed and validated in Rust. Unknown fields report their
nested paths. The supported namespaces are:

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
- `correction`: `enabled` (true), independent `sampling`, `min_observations`
  (5), `factor_bounds` (min 0.5, max 2.0), and the existing `max_num_tokens`
  (8192), `max_batch_size` (512), and `max_kv_tokens` (2000000) ranges.
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

- `op_level`: optional `decode_workload_distribution` selects a measured decode-MoE distribution, and `prefill_graph_profile` selects a qualified direct-prefill graph composition. Saved configurations retain the resolved immutable `prefill_graph_profile_id`, which is validated on reload. Unknown fields are rejected.
- `fpm_interpolation`: `text_only` (false) permits text prefill/decode profiles
  for multimodal architectures while retaining encoder weights. It does not
  supply encoder timing. `unrecorded_quant_modes` (empty) may contain `fmha`
  and/or `comm` to match an explicitly unrecorded precision field in a profile.
  The corresponding top-level quant mode must remain unset. This selects null
  profile values exactly; it does not make precision matching a wildcard.
  `fpm_parquet_path` selects an external FPM parquet with its same-stem metadata
  sidecar. Unknown fields are rejected.

Engine replay rank arguments accept `decode_workload_distribution` (alias `aic_decode_workload_distribution`) only with AIC timing. An active selector paired with a non-AIC timing model, including fixed or polynomial timing, is rejected. The AFD companion's fixed timing and legacy estimator paths also reject active selectors because they cannot apply the profile. `None` preserves ordinary timing in these paths.

Sampling defaults to `bins_per_axis: [4, 4]` and `max_observations: 64` per
logical store. Rectangular grids were already supported. Regression's
`sampling.axes` now selects one to six distinct coordinates, defaulting to
`[attention, moe]`; `bins_per_axis` must have the same length, contain positive
integers, and have a representable product. The dimension is the number of
selected axes, independent of how many features the fit uses. Regression uses
dynamic `log1p` retention coordinates and fits standardized feature values. Correction
uses fixed raw workload coordinates; its one-dimensional prefill grid uses
the product of the two axis counts. Retention evicts the oldest sample from
the most populated cell when the store exceeds its budget.

Correction explicitly reports `feature_space: legacy_workload`. Its existing
prefill/decode/mixed stores and median-ratio calculation remain unchanged at
default settings. A shared role-based correction space requires separate
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

## Unified CLI engine fields

These are the public `predict` / `recommend` YAML names. They differ from SDK
constructor names (for example, `parallelism.tensor` becomes `tp`). A Default
Range of `x` is concrete-only; `-` means no automatic domain. Search-domain
rules and complete parallelism presets are in [Sweeper search space](../sweeper/search-space.md).
Scheduling, speculative acceptance, and cache lifecycle are in
[Replay topology and scheduling](../replay/topology-and-scheduling.md) and
[Replay cache](../replay/cache.md).

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `engine.mode` | `aggregated` | `{choices: [aggregated, disaggregated]}` | `-` | `aggregated`, `disaggregated`, or explicit `afd`. AFD cannot be mixed into a recommendation mode domain. |
| `engine.model` | Required | `x` | `-` | Nonempty and fixed during recommendation. |
| `engine.hardware` | Required | `auto` | `-` | Fallback hardware identifier; `recommend` also accepts `auto` resolved from `optimization.hardware`. P/D workers may override it. |
| `engine.backend` | `vllm` | `{choices: [vllm, sglang]}` | `-` | `vllm`, `sglang`, or `trtllm`; explicit choices may include supported alternatives. |
| `engine.backend_version` | `null` | `x` | `-` | Fixed when set. |
| `engine.speculation` | Omitted (disabled) | `x` | `-` | Optional ngram draft count, conditional acceptance rates, and sampling seed; see [prompt lookup](../replay/topology-and-scheduling.md#prompt-lookup-ngram-speculative-decoding). |
| `engine.context_length` | `"max"` | `x` | `-` | `"max"` derives the effective maximum from the resolved Hugging Face model config; a concrete value must be positive. |
| `engine.workers` | Mode-dependent | `x` | `-` | Aggregated role; prefill plus decode roles; or the optional opposite-phase companion for AFD+P/D. Aggregated and disaggregated modes also support an optional analytical `encoder` pool. |
| `engine.workers.prefill.hardware`, `.decode.hardware` | Inherit `engine.hardware` | `x` | `-` | Concrete nonempty SKU; no `auto` or search domain. Disaggregated roles only; aggregated workers and AFD companions reject hardware overrides. Saved recommendations retain the overrides. |
| `engine.workers.encoder.tensor`, `.replicas`, `.batch_size` | `1` | Scalar or finite `choices` | `encoder` | Positive; batch size at most 8. Not a language-worker parallelism preset. |
| `engine.workers.encoder.hardware`, `.backend_version` | Inherit/resolve | `x` | `-` | Encoder hardware and performance data; backend follows language backend. Saved prediction YAML pins resolved values. |
| `engine.workers.encoder.latency_correction`, `.rate_degradation` | `1.0`, `0.9` | `x` | `-` | Finite positive factors; degradation at most 1. See [EPD CLI semantics](../replay/topology-and-scheduling.md#analytical-epd). |
| `engine.workers.<role>.parallelism.preset` | `default` in `recommend` | `auto` | `-` | Generated default space, complete mapping list, `false`, or `{}`. |
| `engine.workers.<role>.parallelism.replicas` | `1` | Feasible positive values within GPU budget | `parallelism` | Positive. |
| `engine.workers.<role>.parallelism.tensor` | `1` | Feasible registry values | `parallelism` | Positive and model/backend compatible. |
| `engine.workers.<role>.parallelism.pipeline` | `1` | Feasible registry values | `parallelism` | Positive and model/backend compatible. |
| `engine.workers.<role>.parallelism.attention_data` | `1` | Feasible registry values | `parallelism` | Positive and model/backend compatible. |
| `engine.workers.<role>.parallelism.moe_tensor` | `1` | Feasible registry values | `parallelism` | Positive and model/backend compatible. |
| `engine.workers.<role>.parallelism.moe_expert` | `1` | Feasible registry values | `parallelism` | Positive and model/backend compatible. |
| `engine.workers.<role>.parallelism.prefill_context` | `1` | `x` | `-` | `predict` only. Prefill context parallelism (SGLang `--attn-cp-size`, vLLM `-pcp`): splits prefill tokens across extra attention ranks; decode stays replicated on them. Requires model/backend CP support. |
| `engine.workers.<role>.parallelism.decode_context` | `1` | `x` | `-` | `predict` only. Decode context parallelism (vLLM `-dcp`, SGLang `--dcp-size`): stripes the decode KV cache across ranks inside the attention group without adding GPUs. Aggregated workers accept at most one of `prefill_context` / `decode_context` above 1; disaggregated roles carry each knob independently. Modeled for DeepSeek-V3-class MLA (vLLM, SGLang), DeepSeek-V3.2 / GLM-5 DSA (vLLM only), Kimi-K3 (SGLang only), and dense / Qwen-MoE GQA (vLLM, `decode_context <= tensor / kv_heads`). See [DCP scope](#decode-context-parallelism) for the op-level versus measured-FPM distinction. |
| `engine.workers.<role>.scheduler.max_batched_tokens` | Aggregated/prefill/decode: `8192` | Prefill/aggregated: `{choices: [8192, 16384, 32768]}`; decode: `-` | `-` | Positive. |
| `engine.workers.<role>.scheduler.max_sequences` | Aggregated `256`; prefill `1`; decode `256` | Prefill: `{choices: [1, 2, 4, 8, 16, 32, 64, 128, 256]}`; aggregated/decode: `{choices: [256, 512, 1024]}` | `-` | Positive. |
| `engine.workers.<role>.scheduler.prefill_schedule_interval` | `1` | `x` | `-` | `predict` only. Positive. Values above one throttle prefill admission only for vLLM attention-DP groups. |
| `engine.workers.<role>.kv_cache.block_size` | vLLM `64`; SGLang `1`; TensorRT-LLM `32` | `-` | `-` | Positive and backend-supported. Defaults are backend-specific, not version-specific. |
| `engine.workers.<role>.kv_cache.prefix_caching` | `true` | `x` | `-` | Backend-supported. |
| `engine.workers.<role>.kv_cache.bytes_per_token` | `auto` | `x` | `-` | Positive when concrete. `auto` resolves once per worker role from the model and that role's TP/PP/MoE shape. |
| `engine.workers.<role>.kv_cache.capacity.type` | `default` | `x` | `-` | `default` or `fixed`. |
| `engine.workers.<role>.kv_cache.capacity.memory_fraction` | vLLM/TensorRT-LLM `0.9`; SGLang `0.88` | `-` | `-` | `(0, 1]`; `default` capacity only. |
| `engine.workers.<role>.kv_cache.capacity.blocks` | `null` | `x` | `-` | Positive; `fixed` capacity only. Required unless `predict` supplies `capacity.bytes`. |
| `engine.workers.<role>.kv_cache.capacity.bytes` | `null` | `-` | `-` | `predict` only. Positive per-rank G1 byte budget; `fixed` capacity only, mutually exclusive with `blocks`. Requires explicit `block_size` and numeric `bytes_per_token`, or automatic K3 state sizing. |
| `engine.workers.<role>.kv_cache.state_cache.bytes_per_request` | Disabled | `-` | `-` | `predict --stack engine` only, aggregated vLLM without host or G3 offload. Positive recurrent-state bytes per request per rank; requires fixed capacity. Omit the byte count to resolve K3 state and token geometry; see [state-cache sizing](../replay/cache.md#manual-state-cache-sizing). |
| `engine.workers.<role>.kv_cache.prefix_match_unit` | Omitted | `-` | `-` | `predict --stack engine` only. Positive divisor of the resolved `block_size`; requires aggregated vLLM G1 `state_cache`. Rejects `engine.speculation`, `engine.nextn > 0`, KV event export, and Belady eviction. See [manual state-cache sizing](../replay/cache.md#manual-state-cache-sizing). |
| `engine.workers.<role>.kv_cache.capacity.cuda_graph_reserved_bytes` | `0` | `-` | `-` | `predict` only. Integer from `0` through `2**53`; `default` capacity only. |
| `engine.workers.<role>.kv_cache.host_offload.scope` | `dp_rank_local` | `x` | `-` | `dp_rank_local` (one cache per DP rank) or `cluster_shared` (one deployment pool). See [G2 host-cache scope](../replay/cache.md#g2-ownership). |
| `engine.workers.<role>.kv_cache.host_offload.num_host_blocks` | Required when `host_offload` is present | `x` | `-` | Positive; fixed descriptor. Per DP rank for `dp_rank_local`, pool total for `cluster_shared`. |
| `engine.workers.<role>.kv_cache.host_offload.d2h_bandwidth_gbps` | `32.0` | `x` | `-` | Per DP rank; finite and nonnegative, `0` is unlimited. |
| `engine.workers.<role>.kv_cache.host_offload.h2d_bandwidth_gbps` | `32.0` | `x` | `-` | Per DP rank; finite and nonnegative, `0` is unlimited. |
| `engine.workers.<role>.kv_cache.host_offload.shared_d2h_bandwidth_gbps` | `80.0` | `x` | `-` | Pool-wide cap; `cluster_shared` only. |
| `engine.workers.<role>.kv_cache.host_offload.shared_h2d_bandwidth_gbps` | `80.0` | `x` | `-` | Pool-wide cap; `cluster_shared` only. |
| `engine.workers.<role>.kv_cache.host_offload.latency_to_first_byte_ms` | `0.0` | `x` | `-` | Delay before a transfer moves bytes; consumes no bandwidth. |
| `engine.workers.<role>.timing.type` | `default` | `x` | `-` | `default`, `fixed`, or `polynomial`. |
| `engine.workers.<role>.timing.prefill_ms` | `null` | `x` | `-` | Nonnegative and required for `fixed` timing. |
| `engine.workers.<role>.timing.decode_ms` | `null` | `x` | `-` | Nonnegative and required for `fixed` timing. |
| `engine.workers.<role>.timing.forward_model` | `op_level` | `x` | `-` | `op_level` or `fpm`; `default` timing only. `fpm` replays whole-forward (FPM) latency measured for the role's exact model, hardware, backend version, parallel shape and quantization, and fails closed when no such cell exists. |
| `engine.workers.<role>.timing.fpm_parquet_path` | `null` | `x` | `-` | External FPM parquet for `forward_model: fpm`; the adjacent same-stem `.metadata.json` sidecar is required. Relative paths are anchored to the working directory when the engine is constructed. Preserved per role in recommendations, candidate YAML, and regular prefill/decode companions in AFD+PD. |
| `engine.workers.<role>.startup_seconds` | `0` | `x` | `-` | Nonnegative. |
| `engine.kv_transfer.bytes_per_token` | `auto` | `x` | `-` | Positive when concrete. Independent from worker KV-cache geometry; `auto` resolves from the prefill/source role's TP/PP/MoE shape. |
| `engine.kv_transfer.bandwidth_gb_per_second` | `null` | `x` | `-` | Positive when set; `null` disables transfer delay. |
| `engine.kv_transfer.timing_mode` | `destination_missing` | `x` | `-` | `full_prompt` or `destination_missing`; disaggregated mode only. |
| `engine.afd.phase` | Required for AFD | `x` | `-` | `both` for pure AFD; `prefill` or `decode` when `combined_with_pd: true`. |
| `engine.afd.combined_with_pd` | Required for AFD | `x` | `-` | Selects pure `afd` or internal `afd+pd`; it is never inferred from workers. |
| `engine.afd.a_batch_size` | Required for AFD | User-supplied finite domain | `-` | Positive, memory-qualified A-worker batch size. Prediction requires one value. |
| `engine.afd.n_a_nodes`, `n_f_nodes`, `tp_a` | Required for AFD prediction | Enumerated within the GPU budget | `-` | Positive concrete topology fields. Recommendation may optionally constrain `tp_a`. |
| `engine.afd.f_moe_ep_size` | `1` for prediction; model-derived domain for recommendation | Optional choices | `-` | Positive; recommendation also accepts `n_f_nodes` or `ffn_tp`. Dense models require `1`. |
| `engine.afd.num_microbatches` | `3` for prediction | `{choices: [2, 3, 4]}` | `-` | Positive. |
| `engine.afd.pipeline_model` | `optimistic` for prediction | `{choices: [optimistic, conservative]}` | `-` | `optimistic`, `conservative`, or `serial`. |
| `engine.afd.comm_overhead_factor` | `1.0` | `x` | `-` | Positive factor applied once by the AFD evaluator. |
| `engine.afd.boundary_on_attn` | `true` | `x` | `-` | Fixed A/F boundary convention. |

Language-worker roles share the top-level model, backend, backend version, and
context length. Per-role overrides of these fields are rejected. Disaggregated
prefill/decode workers may override hardware; an omitted role inherits the
fallback. If backend version is omitted, it must resolve identically on both
SKUs. Pin a common supported version when their defaults differ. The analytical
encoder pool has separate hardware and backend-version settings.

`hardware: auto` is recommendation-only and resolves from the single
`optimization.hardware` identifier. Saved candidate YAML records concrete
hardware and retains role overrides. A recommendation mode domain can declare
all three language-worker roles; each candidate keeps only its active roles.

The canonical `estimation_mode`, `fallback_policy`, `estimator_config`, and
`fpm_profile` controls pass through the CLI into the same native constructor.
A role's `timing.estimator_config` replaces the global dictionary for that role;
include all required controls in that override. Legacy `timing.forward_model: fpm`
and `timing.fpm_parquet_path` normalize to FPM interpolation and its external pair.
The mode does not fall back to op-level on a missing measured cell.

Physical GPUs per role are replicas times pipeline stages times attention TP,
attention DP, and prefill CP. Decode CP partitions existing ranks and adds no
GPUs. MoE TP/EP repartition that shape rather than multiplying the GPU count
again. Recommendation currently searches its supported CP=1 space; prediction
can select supported explicit context parallelism.

## Flat engine precision and execution controls

These additional public YAML fields are concrete controls:

| Field | Meaning |
| --- | --- |
| `engine.gemm_quant_mode`, `moe_quant_mode`, `fmha_quant_mode`, `kvcache_quant_mode`, `comm_quant_mode` | Explicit precision overrides; omitted values follow model/profile normalization. |
| `engine.attention_backend`, `moe_backend`, `moe_kernel_source` | Runtime graph/backend and exact collected source selectors, subject to canonical validation. |
| `engine.enable_eplb`, `wideep_num_slots` | Expert load balancing and redundant expert slots; require compatible MoE execution. |
| `engine.nextn`, `nextn_accepted` | Compute-side MTP identity and separate expected acceptance assumption; both are required for explicit MTP. |
| `engine.enable_chunked_prefill` | Override chunked prefill; omission preserves the backend default. |

These controls require default timing on language roles. AFD and analytical
encoder configurations reject the unsupported controls. Use `--stack engine`
and the supported model/backend; an older external adapter can reject them at
capability validation before Replay. Expected acceptance is workload behavior,
not an estimator-fitting observation. See [Replay features](../replay/features.md)
for execution and cache combinations.

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

The migration adapter rejects any non-null `prefill_graph_profile`, `prefill_graph_profile_id`, or `decode_workload_distribution` field, including an orphan profile ID. These selectors require the canonical `ForwardPassPerfModelConfig.estimator_config.op_level` configuration; pass a saved canonical configuration directly to `RustForwardPassPerfModel.best_available` to preserve its profile identity and supported API restrictions. Profile-free legacy configurations continue to migrate normally.

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
