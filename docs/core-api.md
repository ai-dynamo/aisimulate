# `aisimulate-core` public API contract

The core API is delivered through the repository's two release artifacts at
the same version:

- the `aisimulate` Python wheel, whose estimator API is `aisimulate_core`;
- the `aisimulate-core` Rust crate, imported as `aisimulate_core`.

The single wheel owns the application, estimator SDK, model and system data,
and unified native PyO3 extension. It does not depend on another core
distribution or on Dynamo. The crate owns the compiled engine, forward-pass
model, Replay runtime, KV-cache request/response types, and the embedded
Rust-to-Python construction path. Legacy Python import namespaces are removed
in AISimulate 0.13.0; see the [Python migration guide](MIGRATION.md#python-imports-and-resources).

## Stable Python facade

New Python code should import from the small facade:

```python
from aisimulate_core.sdk import (
    EngineHandle,
    ModelConfig,
    RuntimeConfig,
    RustForwardPassPerfModel,
    compile_engine,
    estimate_kv_cache,
    estimate_num_gpu_blocks,
)
```

The explicit module paths remain supported:

```python
from aisimulate_core.sdk.engine import EngineHandle, compile_engine
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel
from aisimulate_core.sdk.memory import estimate_kv_cache, estimate_num_gpu_blocks
```

`aisimulate_core.sdk.__all__` is the supported high-level surface. The
facade resolves lazily, so importing it does not load the model registry,
performance database, or native engine until a name is used.

The top-level `aisimulate_core` module exposes the lower-level native
extension contract:

- `AicEngine`
- `RustForwardPassPerfModel` (the raw PyO3 class, distinct from the ergonomic
  SDK wrapper)
- `engine_spec_bincode_from_json`
- `_build_smoke`

The wheel includes `py.typed` and a stub for that native extension. The SDK
Python modules carry their own annotations.

### Context-attention kernel queries

`AicEngine.evaluate_context_attention_kernels_json` accepts `ops_json` (a JSON
array of serialized `ContextAttention` operations), `batch_size`, and `s`
(sequence length in tokens), with optional `prefix=0`,
`imbalance_correction_scale=1.0`, and `visual_block_upper_triangle=False`.
It evaluates only the attention kernels, excluding fused QK normalization,
RoPE, and KV-write work. Invalid JSON and other operation families raise
`ValueError`.

With `visual_block_upper_triangle=True`, the query prices only the additional
strict upper-triangle attention pairs inside a bidirectional visual block,
using the underlying causal kernel's database policy and provenance. This
mode requires `prefix=0`; a nonzero prefix raises `ValueError`. A block with
zero or one token has zero additional latency and energy, with `source="sol"`.
The flag is a runtime query option and does not change the serialized op or
engine-spec formats.

Results are `list[tuple[str, float, float, str]]`, containing
`(name, latency_ms, energy_wms, source)` with repeated operation names folded
together. Energy is in watt-milliseconds and is zero when power data is
unavailable. An empty operation list returns an empty list. `aisimulate_core.AicEngine` exposes this
method; `EngineHandle` provides an annotated SDK wrapper with the same query
options.

## Python single-point prediction

Use `aisimulate.predict.run_prediction` to compile and execute one public
`CorePredictionConfig`. It checks runner capabilities, creates and closes the
runner, and returns a `PredictionResult` with the same summary the CLI renders.

In an activated environment with AISimulate and its native extension installed,
run this complete example. Fixed timing and KV capacity avoid model downloads
and performance-data queries; the result demonstrates execution, not measured
model accuracy.

```bash
python - <<'PY'
from aisimulate.config.cli import CorePredictionConfig
from aisimulate.predict import run_prediction
from aisimulate.runner import EngineReplayRunnerFactory

config = CorePredictionConfig.model_validate({
    "engine": {
        "model": "example/model",
        "hardware": "h200_sxm",
        "context_length": 128,
        "workers": {"aggregated": {
            "timing": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
            "kv_cache": {
                "bytes_per_token": 128,
                "capacity": {"type": "fixed", "blocks": 100},
            },
        }},
    },
    "traffic": {
        "source": {"type": "synthetic", "input_tokens": 32, "output_tokens": 4},
        "load": {"type": "concurrency", "concurrency": 1},
        "stop": {"requests": 1},
    },
})
result = run_prediction(
    config, stack="engine", runner_factory=EngineReplayRunnerFactory()
)
assert result.summary["completed_requests"] == 1
print(result.summary)
PY
```

For an existing core prediction YAML, load it with
`CorePredictionConfig.from_yaml("prediction.yaml")` instead. See the
[prediction configuration guide](cli/user-guide.md) for model-backed timing
and workload options.

| Argument | Contract |
| --- | --- |
| `config` | Required validated `CorePredictionConfig`. |
| `stack` | Required keyword naming the execution stack; use `"engine"` with `EngineReplayRunnerFactory`. |
| `runner_factory` | Required keyword supplying a factory compatible with the compiled replay. |
| `adapter_configs` | Optional raw adapter blocks keyed by section name. |
| `providers` | Resolved config adapters keyed by `"<stack>.<section>"`; required for every supplied adapter block. Core config does not accept Router/Planner top-level sections. |
| `execution_mode` | Defaults to `"offline"`; the selected runner must support the requested mode. The built-in engine stack is offline. |
| `output_requirements` | Optional `ReplayOutputRequirements` from `aisimulate.sweeper.replay`. Omission enables raw-report capture except for analytical EPD. An explicit value replaces that default. |

`PredictionResult` contains:

- `summary`: merged prediction metrics, including normalized power fields.
- `native`: the native report with the summary merged in, or a summary fallback
  when the runner supplies no native report. Analytical EPD retains its metadata
  and approximation semantics instead of a token-replay report.
- `replay_spec`: the compiled specification that was executed.
- `report`: the runner's original `ReplayReport`, including metrics and metadata.

After creating a runner, the call closes it even when execution fails. Runner
execution exceptions are wrapped in `PredictionExecutionError`, preserving the
cause and any `fpm_query_coverage` attribute. `KeyboardInterrupt` and
`ResourceLimitError` propagate unchanged; configuration, compilation, capability
and runner-creation failures also propagate directly.

The call does not create output directories, save report files or print results.
The caller owns those actions and resource budgets. Passing the plain engine
factory does not enable the CLI's host-memory admission or subprocess supervision;
use the CLI for the automatic [local resource controls](local-resources.md).

Source: [prediction entry point](../python/aisimulate/src/aisimulate/predict.py).

## Engine context limits

`EngineConfig.max_model_len` is an optional positive prompt-plus-output token
limit for vLLM, TRT-LLM, and SGLang. Recipe adapters can map TRT-LLM
`max_seq_len` and SGLang `context_length` to this field.

- A prompt at or above the limit is rejected before prefill computation.
- A decode destination rejects such a prompt before queuing the handoff or
  reserving KV blocks, even when destination admission is deferred.
- Terminal rejection removes the request's outstanding Belady input demand
  before subsequent cache admission and eviction.
- Generation stops when prompt plus output reaches the limit, including
  speculative bursts and requests with explicit output token IDs. Reports
  retain the requested output length and count only tokens actually generated.
- TRT-LLM completion reservations and SGLang output reservations use the capped
  output budget. Physical KV capacity remains a separate constraint.
- The limit applies to aggregated workers and both roles in disaggregated replay.
  An unset limit preserves the existing backend behavior.

The prediction compiler and recommendation deployment builder pass explicit
`engine.context_length` values to every backend. Prediction with
`context_length: max` retains its existing behavior: vLLM resolves model
metadata, while TRT-LLM and SGLang leave the scheduler limit unset.

This is the simulator's normalized context-limit contract. It does not model
backend-version-specific frontend validation margins or automatic prompt
truncation. Successful replay alone does not establish silicon timing accuracy.
See the [SGLang GPU parity observations](https://github.com/ai-dynamo/aisimulate/pull/261#pullrequestreview-5250881045)
for stricter boundary behavior; they do not establish a fixed token offset
across versions or configurations.

## KV-cache capacity reservation

SGLang's native estimator treats `mem_fraction_static` as a static weights/KV
pool. Peak activation/workspace estimates remain visible in
`memory_breakdown.activations_bytes`, but are not deducted from that pool;
transient execution headroom is already outside the static fraction. Increasing
the prefill token budget alone therefore does not reduce SGLang KV capacity.
Resident runtime/communication estimates reduce the pre-load free-memory pool
before applying the fraction; weights are deducted afterward. The budget is
`(capacity - resident_overhead) * mem_fraction_static - weights`, less any
explicit additional graph reservation. For ordinary SGLang DeepSeek-V3/R1
(non-CP, non-PP, non-speculative, non-large-EP), the estimator also respects
checkpoint dense/MoE layer counts and TP-sharded embeddings independently of
the unchanged timing graph. Other model layouts retain their prior weight
accounting.
vLLM and TRT-LLM continue to deduct activation memory under their own budget
semantics. No measured server capacity is required by this calculation.

`estimate_kv_cache` and `estimate_num_gpu_blocks` accept
`cuda_graph_reserved_bytes=<rank-local bytes>`. The value must be a
non-negative integer no greater than `2**53` and defaults to zero. It is treated
as fixed non-KV memory before the backend-specific KV fraction is applied. For
SGLang, it is an additional reservation beyond graph/runtime headroom already
encoded by `mem_fraction_static`. Native estimates return it as
`memory_breakdown.cuda_graph_reserved_bytes`; naive fallback estimates apply it
but keep `memory_breakdown=None` because the other components are unavailable.

The Rust `KvCacheEstimateRequest` exposes the same field. Engine replay accepts
the value in its engine arguments and carries it through native AIC capacity
rematerialization, so the Python estimator and native scheduler use the same
rank-local KV capacity. The `aisimulate predict` YAML exposes it at
`engine.workers.<role>.kv_cache.capacity.cuda_graph_reserved_bytes`. This API
does not estimate the reservation; callers must supply a value from a source
they trust.

Serialized Rust requests and estimates that omit the field remain compatible
because deserialization defaults it to zero. See [Rust literal migration](MIGRATION.md#rust-resource-and-memory-literals)
for required source changes to request and memory-breakdown structs.

### FPM profile cache groups and byte budgets

FPM resources support `cache_layout: linear` with a positive
`kv_bytes_per_token`, or `cache_layout: grouped` with a nonempty `cache_groups`
list and no scalar token rate. Existing linear profiles keep their serialized
shape. Python exports `FpmCacheGroup` from `aisimulate.fpm_profile` and
`aisimulate_core.fpm_profile`; Rust exports it from `aisimulate_core::perfmodel`.

Memory can be pending, declared, or observed at runtime. The four legacy fields
`weights_bytes`, `activations_bytes`, `runtime_overhead_bytes`, and
`comm_overhead_bytes` are optional. A complete set retains the existing non-KV
budget calculation. A partial set is planning evidence only: missing values are
not zero, and cache sizing or replay fails with an instruction to finalize
runtime memory. Schema validation, collection planning, and direct timing queries
can use a pending profile. Omitted fields remain absent from serialized output.

Alternatively, `resources.runtime_memory` records the selected deployment's
initialized cache allocation:

```json
{
  "kv_cache_bytes": 1073741824,
  "gpu_memory_utilization": 0.9,
  "max_model_len": 32768,
  "provenance": "Verified collection worker initialization; see saved evidence."
}
```

The byte count is a positive integer no greater than `2**53`; utilization is
finite and in `(0, 1]`; context and provenance are required. Runtime memory
cannot coexist with any of the four legacy non-KV fields. Python exposes
`FpmRuntimeMemoryProfile` and resource properties `memory_source`
(`pending`, `declared`, or `runtime`) and `memory_ready`; `require_memory()`
rejects pending resources. Rust exposes `FpmRuntimeMemoryConfig` and
`FpmResourceConfig::require_memory()`. See [Rust literal migration](MIGRATION.md#rust-resource-and-memory-literals)
for resource construction changes.

Each group has a unique `name`, `kind` (`attention` or `convolution`), positive
`num_layers`, `block_size_tokens`, and `page_size_bytes`, and an optional positive
`sliding_window`. An omitted or null window retains full history; convolution
groups require a window. `page_size_bytes` is the **rank-local aggregate for all
layers in the group**, including runtime padding. Do not multiply it by
`num_layers` again. Runtime block sizes and padding are deployment inputs; model
geometry alone does not establish them. See the
[grouped-profile review workflow](fpm-self-service/implementation.md#review-grouped-cache-resources).

Use `RustForwardPassPerfModel.estimate_cache_budget(config, budget)` with the
same canonical `ForwardPassPerfModelConfig` used for timing. Rust exposes
`ForwardPassPerfModelConfig::estimate_cache_budget(&request)` and
`ForwardPassPerfModel::estimate_cache_budget(&request)`, taking
`FpmCacheBudgetRequest` and returning `FpmCacheBudget`. The config method checks
the exact profile deployment and scheduler envelope without constructing a
timing model, loading timings, or building an operation graph. Grouped planning
requires the native extension; it does not require GPU access or measured data.

| Budget field | Meaning |
| --- | --- |
| `total_gpu_capacity_bytes` | Positive rank-local GPU capacity. |
| `memory_fraction_kind`, `memory_fraction_value` | `of_total` and the fraction of GPU memory available to the worker. |
| `max_num_tokens`, `max_batch_size` | Positive rank-local scheduler limits within a declared envelope; exact recorded values for runtime memory. |
| `context_length` | Optional positive request bound, at most the profile context and runtime `max_model_len` when present; defaults to the profile context. |
| `request_occupancy_tokens` | Optional positive logical request length within the selected context, for a separate steady decode footprint with one new token per forward pass. |
| `cuda_graph_reserved_bytes` | Separate nonnegative reservation for declared memory; must be zero for runtime memory. |
| `tolerance_fraction` | Optional fraction in `[0, 1)` deducted from the resulting cache byte budget. |

The result includes `total_kv_size_bytes`, `memory_breakdown`,
`resource_provenance`, `cache_layout`, `cache_groups`, and
`request_peak_cache_bytes`. The latter is a conservative single-request bound
including block alignment and the configured prefill chunk, not a reservation
for every scheduler slot. A smaller context changes this peak bound, not the
declared non-cache resource bounds. `tolerance_adjusted`, when present, provides
the reduced byte budget. For grouped caches, `kv_size_per_token_bytes`,
`total_kv_size_tokens`, and the adjusted token capacity are null (`None` in Python).
There is no equivalent scalar token capacity.

When `request_occupancy_tokens` is supplied, the result also includes
`request_occupancy_cache_bytes`. KV-relative synthetic loads use this native
footprint while retaining the actual scheduler settings and context for resource
admission. Observed linear pools retain their recorded capacity even when timing
coverage is smaller; uncovered queries report timing errors.

Runtime memory uses `kv_cache_bytes` directly and returns
`memory_breakdown: null` (`Option::None` in Rust). It does not invent activation
or other non-KV components. The requested memory fraction must exactly match the
recorded utilization; the device budget must contain the observed cache pool.
A different device capacity argument never rescales the pool or qualifies a new
deployment. Changed scheduler settings require new runtime evidence. Explicit
scalar cache-capacity overrides are rejected for runtime profiles in prediction,
recommendation and replay; use the recorded allocation. A tolerance margin may
reduce it through the canonical budget API.

Python `estimate_kv_cache(..., fpm_profile=..., context_length=...)` delegates
grouped and runtime-memory profiles to this native budget method.
`estimate_num_gpu_blocks` also accepts `context_length` for runtime validation;
it and the
legacy Rust scalar `KvCacheEstimate` transport reject grouped profiles; use the
groups and shared byte budget instead. Existing complete linear declarations
keep their budget calculation and serialized shape.

Grouped native execution supports cold aggregated vLLM, PP1/CP1, HBM-only cache,
non-speculative decoding and `prefix_caching: false`. Prefix reuse, host/G3
offload, disaggregation and fixed scalar cache capacity are rejected. Allocation
is atomic across all groups against one shared rank-local byte budget. A window
of `W` keeps the blocks covering the last `W - 1` computed tokens and all tokens
scheduled for the next forward. Expired completed pages are evicted before
subsequent forwards; temporary prefill pages remain charged through completion.
Full-attention groups retain full history. This physical retention never truncates
logical request progress or the context coordinates sent to FPM timing queries.

The Rust Replay observer's `ReplaySchedulerMetricsSnapshot` reports optional
`kv_cache_used_bytes` and `kv_cache_capacity_bytes` for grouped caches; its
`active_cache_usage` and `physical_cache_usage` use byte occupancy. `active_blocks`
counts resident group pages, while `total_blocks` is zero because heterogeneous
pages have no scalar block capacity. Use the byte fields for capacity comparisons.
These fields are currently exposed through the Rust observer API, not the Python
JSON replay runtime. Linear snapshots retain their existing block metrics.

## Choosing a forward-pass API

Use `RustForwardPassPerfModel.best_available(config)` from Python or
`ForwardPassPerfModel::best_available(config)` from Rust. The canonical
`ForwardPassPerfModelConfig` owns model, hardware, backend, topology, data
policies, a required immutable `worker_type`, and the complete nested
`estimator_config`. Worker roles are `prefill`, `decode`, and `aggregated`.
Topology carries two optional context-parallel knobs next to `tp`, `pp`,
`attention_dp`, `moe_tp_size`, and `moe_ep_size`: `cp_size` (prefill context
parallelism, SGLang `--attn-cp-size` / vLLM `-pcp`; extra attention ranks, so
`tp * attention_dp * cp_size == moe_tp_size * moe_ep_size`) and `dcp_size`
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

### Vera Rubin GLM-5.2 graph-prefill pilot

The canonical system name is `vr_nvl72` (Vera Rubin NVL72). The previous
`vr200_hecate` system and `sglang_glm52_nvfp4_vr200_tp4_graph_v1` selector are
removed without aliases. Update configurations to `vr_nvl72` and
`sglang_glm52_nvfp4_vr_nvl72_tp4_graph_v1`, discard the old resolved
`prefill_graph_profile_id`, and save a fresh configuration from the canonical
constructor. Rebuild compiled engines from that configuration. This rename
preserves the four-GPU pilot's measurements and restrictions; it does not
qualify a complete NVL72 rack. See the [offline artifact migration](../python/aisimulate/collector/sglang_rubin/README.md#migrate-the-published-system-name).

The opt-in `sglang_glm52_nvfp4_vr_nvl72_tp4_graph_v1` profile uses the same canonical constructor and a latency-only direct method. It is qualified for seven homogeneous prefill shapes on the pinned SGLang runtime, TP4/EP1, NVFP4 experts, BF16 projections and FP8 KV. The profile preserves its immutable SHA-256 in `diagnostics()["provenance"]["config"]["estimator_config"]["op_level"]`; save that complete configuration when reproducing a prediction.

```python
from aisimulate_core.sdk import RustForwardPassPerfModel

model = RustForwardPassPerfModel.best_available({
    "model": "nvidia/GLM-5.2-NVFP4",
    "system": "vr_nvl72",
    "backend": "sglang",
    "backend_version": "0.5.18+nvinternal.rubin.0.8full.66997102",
    "worker_type": "prefill",
    "tp": 4, "pp": 1, "attention_dp": 1,
    "moe_tp_size": 4, "moe_ep_size": 1,
    "gemm_quant_mode": "bfloat16", "moe_quant_mode": "nvfp4",
    "fmha_quant_mode": "bfloat16", "kvcache_quant_mode": "fp8",
    "comm_quant_mode": "half",
    "estimation_mode": "op_level", "fallback_policy": "deny",
    "database_mode": "SILICON", "enable_shared_layer": False,
    "estimator_config": {
        "op_level": {"prefill_graph_profile": "sglang_glm52_nvfp4_vr_nvl72_tp4_graph_v1"},
        "correction": {"enabled": False},
    },
})
milliseconds = model.predict_prefill_latency(bs=1, isl=2048, prefix=1024)
```

`isl` is the total input length, including cached tokens. That call processes 1,024 new tokens after a 1,024-token prefix. The admitted `(batch, isl, prefix)` calls are `(1,1024,0)`, `(2,1024,0)`, `(1,2048,1024)`, `(1,8192,0)`, `(2,8192,0)`, `(1,16384,0)` and `(1,32768,16384)`. Arguments must be ordinary Python integers in the unsigned 32-bit range; the Rust API uses `u32`. Other shapes and batch-token products that overflow fail before lookup. The tables have exact keys and do not interpolate or inherit another profile's data.

The independent forward-step comparison passes all seven shapes within 15%, with worst absolute relative error 5.2333%. This is a measured mean forward-time comparison for the exact runtime. Scheduler TTFT, model quality and general Vera Rubin coverage remain unqualified by this prefill comparison. Aggregate telemetry cannot establish each request's exact new/past lengths, so this selected profile rejects `estimate_forward_pass_time_ms`, tuning, static energy/SOL diagnostics and replay-provider construction. Use the direct scalar method; no CLI scheduler selection is supported. Other profiles retain their existing behavior. See the [dedicated collector](../python/aisimulate/collector/sglang_rubin/README.md) and [packaged data provenance](../python/aisimulate/src/aisimulate_core/systems/data/vr_nvl72/README.md).

Decode is validated separately through the default op-level estimator with `worker_type="decode"`, no `prefill_graph_profile`, and correction disabled. Use `static_phase_latency(batch_size=B, input_tokens=K, output_tokens=2, prefill=False)` for one decode step with `K` past KV tokens and attention length `K + 1`; `output_tokens=1` requests zero decode iterations. The [combined accuracy report](vr-nvl72-glm52-accuracy.md) lists all seven prefill and 18 decode cases, the exact configuration and measurement boundary. Decode meets the original ±15% criterion in 17/18 cases; the pilot accepts the remaining observed −17.09% batch-1 residual. This does not extend the opt-in prefill profile to decode or qualify scheduler TTFT.

### External whole-forward FPM data

Set `estimator_config.fpm_interpolation.fpm_parquet_path` on the canonical configuration to use an external parquet and its required same-stem `.metadata.json` sidecar:

```python
config = ForwardPassPerfModelConfig(
    model="Qwen/Qwen3-0.6B",
    system="h200_sxm",
    backend="vllm",
    backend_version="0.25.1",
    worker_type="aggregated",
    estimation_mode="fpm_interpolation",
    estimator_config={
        "fpm_interpolation": {"fpm_parquet_path": "/data/reviewed-fpm.parquet"},
    },
)
model = RustForwardPassPerfModel.best_available(config)
```

The parquet identity must match the requested model, hardware, backend version, topology, and quantization. The systems YAML is still required, but a backend timing-data directory is unnecessary. Relative paths bind to the working directory when the model is constructed; resolved provenance stores the absolute path. The control applies when FPM interpolation is selected; other estimators retain it in provenance without opening the file. Saved legacy `timing.forward_model: fpm` and `timing.fpm_parquet_path` inputs migrate to the same canonical control, which is preserved in replay and per-role recommendation output.

`model.static_phase_latency(batch_size=1, input_tokens=512, output_tokens=4, prefill=False)` exposes the native engine's existing static integration before online correction. Prefill returns one prefill latency; decode returns total decode latency for the output sequence. This method requires a native estimator. The graph-prefill pilot requires `predict_prefill_latency` and rejects this generic static API. AFD+PD uses it for an external-FPM regular companion, dividing total decode latency by `max(1, output_tokens - 1)` for TPOT. AFD attention and FFN workers retain their existing timing provider.

### Engine identity controls

The canonical configuration also carries quantization overrides and `attention_backend`, `moe_backend`, `moe_kernel_source` (default absent), `enable_eplb` (default `false`), and `wideep_num_slots` (default absent). These controls reach model construction, KV memory sizing, and replay provenance. EPLB/slots and nondefault MoE backend or kernel-source selection require an MoE model. Collected FPM interpolation cannot represent EPLB, slots, MoE backend, or kernel-source overrides; it rejects an explicit incompatible request and is skipped during automatic selection for those identities.

`moe_kernel_source` selects an exact, nonblank collected `kernel_source` label for fused MoE compute. It is distinct from the existing `moe_backend` graph/backend control; source labels are not backend aliases and are preserved without trimming. `None` keeps the existing default source-selection policy, including eligible low-latency NVFP4 selection. `SILICON` reads only the requested source's table; `EMPIRICAL` derives its estimate from that same source; `HYBRID` may fall back to empirical estimation within that source, but does not substitute a different source. Missing source data remains an error. An explicit `moe_torch_flow_min_latency` requires gated NVFP4 and at most 128 tokens after attention-DP gathering. Pure-roofline `SOL` remains table-independent and does not claim measured support for the requested source.

Selected `prefill_graph_profile` and observed `decode_workload_distribution` profiles require `moe_kernel_source=None`. Their qualified composition and source identity are fixed; an explicit source override is rejected even when its label matches the measured kernel. An absent or null source preserves the approved profile identity and predictions.

`moe_perf.parquet` may include Boolean selection metadata `default_eligible`. An absent column preserves legacy automatic selection; when present, every value must be a non-null Boolean. A `false` row requires a nonblank string `kernel_source`, preserved exactly, and is available only through that named source. It cannot enter automatic standard or low-latency grids, including empirical cross-shape and cross-quant reference selection. Among eligible rows, existing source priority and first-row precedence remain unchanged. Default table views, coverage, and readiness use those same eligible grids; raw Parquet enumeration retains all measured rows. Malformed eligibility metadata is an invalid-data error, never a missing-data fallback. The flag is not a measurement identity dimension. Collector finalization preserves it during merges, including existing annotations when a legacy recollection omits the column; genuinely new legacy keys remain eligible.

An explicit source is rejected for dense graphs, MegaMoE modules, large-EP expert-compute graphs, and any constructed timing phase with no compatible fused MoE operator. It is also incompatible with whole-forward FPM, including the legacy Task `forward_model='fpm'` rewrite. Invalid graph/source combinations fail as invalid configuration rather than triggering estimator fallback. An untrained `fpm_regression` model remains not-ready; retaining a source in its configuration is not evidence of source-specific prediction support.

AFD regular companions currently reject exact-source requests through their legacy estimator and fixed timing paths. The external-FPM companion forwards the source to canonical validation, which rejects the incompatible FPM request. These controls describe standalone AISimulate behavior, not downstream Dynamo planner integration.

Rust callers using exhaustive `ForwardPassPerfModelConfig` literals must add `moe_backend: None`, `moe_kernel_source: None`, `enable_eplb: false`, and `wideep_num_slots: None`. Direct `EngineConfig` and `MoeOp` literals likewise require the new `moe_kernel_source` field. `ForwardPassPerfModelConfig::new(...)` supplies its default. This extends the canonical configuration introduced by #242.

Rust callers constructing `SyntheticTraceSpec` must also add
`cached_prefix_tokens: 0` to preserve existing prefix-sharing behavior. A positive
value creates shared input tokens; cache hits still depend on runtime state.
The value must align to the trace's `block_size` and must not exceed any sampled
input length. Unified replay uses one-token trace blocks for an exact prefix,
then applies the engine's cache block size when calculating reuse.

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

### Selecting linear or spline regression

Select the fit within `estimator_config.fpm_regression.fit`. The default remains
the existing linear model. The alias `linear` normalizes to `standardized_nnls`
in provenance and saved configuration. Evaluate spline for each deployment before
enabling it: the MiniMax-M2.7 / H200 / vLLM / TP4 prefix in the
[2026-09-25 comparison](fpm-recursive-regression.md#production-spline-validation-2026-09-25)
had 0.87% MAPE with linear, 5.31% with periodic spline, and 1.63% with adaptive
spline, so improved accuracy on other workloads does not imply a universal gain.
To select spline regression explicitly:

```python
config = ForwardPassPerfModelConfig(
    model="Qwen/Qwen3-32B",
    system="h200_sxm",
    backend="vllm",
    worker_type="decode",
    estimation_mode="fpm_regression",
    estimator_config={"fpm_regression": {"fit": {"kind": "spline"}}},
)
model = RustForwardPassPerfModel.best_available(config)
```

The corresponding Rust configuration uses the same canonical constructor and
public typed controls. For example, two knots per axis with periodic search:

```rust
use aisimulate_core::{
    BackendKind, EstimationMode, ForwardPassPerfModel, ForwardPassPerfModelConfig,
    ForwardPassWorkerType, RegressionFitKind, SplineFitConfig, SplineSearchConfig,
};

let mut config = ForwardPassPerfModelConfig::new(
    "Qwen/Qwen3-32B", "h200_sxm", BackendKind::Vllm, ForwardPassWorkerType::Decode,
);
config.estimation_mode = EstimationMode::FpmRegression;
config.estimator_config.fpm_regression.fit.kind = RegressionFitKind::Spline;
config.estimator_config.fpm_regression.fit.spline = Some(SplineFitConfig {
    knots_per_axis: 2,
    search: SplineSearchConfig::periodic(64),
});
let model = ForwardPassPerfModel::best_available(config)?;
```

Rust source compatibility: exhaustive matches on `RegressionFitKind` must now
handle `RegressionFitKind::Spline`. Existing full `RegressionFitConfig` literals
for linear fits must include the optional `linear` and `spline` fields; this type
implements `Default`, so
`..RegressionFitConfig::default()` is also available when its other defaults
are appropriate. Full `ForwardPassRegressionStoreDiagnostics` literals must
likewise provide the new `spline` field (`None` for linear stores). That
diagnostics type does not implement `Default`. Existing serialized linear
configurations and diagnostics continue to omit `spline` when it is `None`.
Likewise, absent `fit.linear` and default regression sampling axes remain omitted.
`FpmRegressionConfig.sampling` uses the public `RegressionSamplingConfig` with
vector-valued axes and bin counts. Correction retains `SamplingConfig` and its
two-element bin array. `LinearFitConfig`, `RegressionFeatureAxis`, and
`RegressionUpdatePolicy` are public Rust types; legacy flat options still migrate
to the unchanged two-axis default. Full `ScheduledRequestMetrics` literals must
initialize `extend_lengths` and `past_kv_lengths` to `None`, or use
`..ScheduledRequestMetrics::default()` when appropriate.

The new `SplineSearchConfig` enum, its policy variants, and
`ForwardPassSplineDiagnostics` are `#[non_exhaustive]` so additional policies or
fields can be introduced without breaking callers that follow the supported
construction and matching patterns. Construct search policies with
`SplineSearchConfig::periodic(step)` or
`SplineSearchConfig::adaptive(window, trigger, tolerance, absolute_tolerance_ms, cooldown)`.
These constructors retain the supplied values; the canonical model constructor
resolves and validates the configuration. External matches need a wildcard arm
for future policies and `..` when destructuring variant fields. Read spline
diagnostics returned by the model instead of constructing diagnostic literals;
destructuring them also requires `..`. JSON/YAML representation and defaults are
unchanged.

Rust expands omitted spline controls to:

```yaml
engine:
  estimation_mode: fpm_regression
  estimator_config:
    fpm_regression:
      fit:
        kind: spline
        spline:
          knots_per_axis: 2
          search:
            kind: adaptive
            window: 16
            trigger: 8
            tolerance: 0.05
            absolute_tolerance_ms: 1.0
            cooldown: 64
```

This YAML is the estimator portion of a prediction/recommendation configuration;
model, hardware, backend, and workers use the usual engine settings. The same
nested paths work with CLI `--set`. Role-specific
`engine.workers.<role>.timing.estimator_config` replaces the global dictionary
for that role, so include every desired control in the role override.

`knots_per_axis` accepts 2 or 3. Constant or sparsely sampled axes may use fewer
distinct knots. To use periodic searches, replace the entire `search` object:

```yaml
search:
  kind: periodic
  step: 64
```

`step`, `window`, `trigger`, and `cooldown` are positive integers; `trigger` must
not exceed `window`. `tolerance` is a positive finite relative error (0.05 means
5%), and `absolute_tolerance_ms` is finite and nonnegative. Policy-specific fields
cannot be mixed. A `spline` block with a linear fit is rejected. Spline fitting
requires `sampling.max_observations >= 32` and the existing capacity/minimum
observation consistency checks still apply.

Each store first searches after `max(32, min_observations)` accepted observations,
provided it retains at least `min_observations` samples. Before that, a usable
linear fit supplies predictions, starting at the usual minimum of five samples.
With default settings, the first search occurs at observation 32. Periodic
searches then occur at accepted-count multiples of `step`: 64, 128, and so on.
Adaptive search requires at least `trigger` excessive errors in the latest
`window` monitored observations and at least `cooldown` accepted observations
since the previous search. The default earliest second search is observation 96.

Adaptive errors use the raw, unclamped spline prediction before consuming the
new target and before the range guard below. An error is excessive when its
absolute value exceeds `max(absolute_tolerance_ms, tolerance * observed_ms)`. Monitoring includes
points outside the retained range; the window resets after each search. A full
window is not required, and the current observation need not itself have an
excessive error once the rolling count and cooldown conditions are satisfied.
Prediction queries and rejected observations do not advance the search clock. Coefficients
continue updating between searches as retained samples are inserted or evicted.

The spline is used only within the current retained samples' raw-feature bounds
and only when the linear fit can predict the same query. Outside that box, or
while the spline is unready, the model uses the linear fit trained on the same
retained observations. Both paths keep the positive prediction floor. If the
linear prediction is unavailable, the query returns `None`, even if a spline
prediction is available inside retained bounds. This guard preserves the shared
linear fit's prediction coverage and limits extrapolation; it does not guarantee
accuracy on unseen workloads. The fit is additive across the two axes, without
interaction terms.

`regression_store_diagnostics()` adds a `spline` object only for spline stores:
`initialized`, `ready`, `accepted_observations`, `knot_searches`,
`last_search_observation`, `numerical_rebuilds`, and `batch_fallbacks`.
The enclosing store is ready only when its shared linear fit is usable.
It can be ready while `spline.ready` is false, using linear predictions during
spline warmup or numerical unavailability. Conversely, `spline.ready` can be
true while the enclosing store reports `ready: false`; no predictions are served
until the shared linear fit is usable again. Spline component readiness describes
its fit, independently of the guard applied to serving.
`initialized` records that an initial search has run, even if its fit
is unready. Search counts are separate from statistics rebuild/fallback counts.
`numerical_rebuilds` includes configured periodic rebuilds and numerical recovery
within fixed-knot epochs; initializing statistics for a new knot search is excluded.

Saved configuration preserves the resolved policy and controls, but not learned
knots, samples, or counters: constructing from it starts a cold model. CLI,
Sweeper, and Replay transport the settings; they still reject cold regression
for offline simulation. Setting `fit.kind: spline` does not change `auto`'s
estimator priority; use `estimation_mode: fpm_regression` to require regression.

### Direct-FPM query coverage

Enable coverage through the same canonical `best_available(config)` construction:

```yaml
estimation_mode: fpm_interpolation
fallback_policy: deny
estimator_config:
  fpm_interpolation:
    method: direct
    collect_coverage: true
```

The rest of the configuration must identify the exact deployment, including its
complete `fpm_profile`, literal backend version and data roots. CLI prediction
places these timing controls under `engine.workers.<role>.timing`. Coverage does
not change point selection, interpolation, correction or latency, and it does
not permit fallback or extrapolation after a missing query.

The returned model exposes `fpm_query_coverage()` as
`Result<Option<FpmQueryCoverage>, AicError>` in Rust and `dict[str, Any] | None`
in the Python SDK. The raw PyO3 method returns a JSON string, using `null` when
disabled. Evidence belongs to that model instance; cloning a Rust model starts
a fresh accumulator so candidate evaluations do not share counts. The public
Rust facade exports `FpmQueryCoverage`, `FpmQueryCoverageCounts`, `FpmQueryGap`
and `FpmQueryPurpose`.

A snapshot has cumulative `queries`, `prefill`, `decode` and
`mixed_decode_baseline` counts, each with `measured`, `interpolated` and
`unsupported`. Classification comes from the native lookup that supplied or
rejected the timing, including mixed-pass decode-floor lookups. Counts describe
`native_lookup_resolutions`; they are not request or replay-iteration counts,
and an external timing-cache hit adds no lookup. The snapshot retains up to 128
distinct gaps with phase, purpose, model/cell identity, query coordinates, reason
and occurrence count. `omitted_gap_queries` counts failed occurrences beyond
that storage limit; unsupported totals are not truncated. Snapshot reads do not
reset evidence, and lookup errors remain errors with partial evidence available.
The snapshot alone does not assess replay completion or prediction accuracy.

For existing static phase consumers, the same returned model provides these
uncorrected native timings, in milliseconds:

| Method | Coordinate contract |
| --- | --- |
| `predict_prefill_latency(batch_size, isl, prefix)` | Full input length and cached-prefix length per request; direct-FPM totals are `(batch_size, batch_size * (isl - prefix), batch_size * prefix)`. |
| `predict_decode_latency_total(batch_size, total_past_kv_tokens)` | Exact total past-KV across the decode batch, preserving the native FPM coordinate without a mean-context conversion. |
| `fpm_decode_kv_ceiling()` | Largest collected decode KV total, or `None`; reading this bound does not record a timing lookup or prove shape-specific coverage. |

These methods preserve the existing engine's phase behavior and record actual
direct lookup evidence when enabled. Empty work does not invent a measured
query. Whole-iteration `estimate_forward_pass_time_ms()` remains the API for
scheduled per-rank FPM telemetry, including mixed work and its existing online
correction behavior.

Ordinary replay uses this returned model for FPM timing and emits
`fpm_query_coverage` in its report when collection is enabled. The CLI also saves
`fpm-coverage.json`, including partial evidence when a native timing error stops
replay; the Python exception retains a `fpm_query_coverage` attribute for that
failure path. A passing replay coverage status requires completed requests,
nonempty resolved queries and no unsupported lookup. It is distinct from
operation/energy evidence, which whole-model FPM does not provide. See
[FPM replay validation](fpm-self-service/implementation.md#validate-fpm-query-coverage-with-agentx-replay)
for the stricter whole-corpus completion checks and saved onboarding artifacts.

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

### Recursive regression and statistics rebuilding

Linear regression maintains centered sufficient statistics for the retained
samples and applies the selected standardized linear fit. The default objective
is unchanged. The retention grid still controls
which samples are kept; it does not create separate fitted planes within a
workload store. Spline regression maintains statistics in its current basis and
rebuilds them when knot positions change. Its knot-search policy and
`rebuild_interval` are separate controls. Knot relocation resets the spline's
fixed-basis mutation clock but does not reset the linear fit's rebuild clock.

See the [recursive regression walkthrough](fpm-recursive-regression.md) for
the update equations, numerical guards, and measured fitting costs.

Set `estimator_config.fpm_regression.fit.rebuild_interval` through the canonical
constructor. Rust owns its default and validation. Omission means `None`
(`null` in JSON), so periodic rebuilding is disabled by default. A positive
integer, such as 4096, opts into a periodic interval. Zero, negative values,
booleans, floating-point values, strings, arrays and objects are rejected with
the nested field path. The explicit setting survives normalization, provenance,
saved configuration and reload. No flat legacy option is added.

When enabled, the interval counts **one insertion and one eviction as separate
mutations**. A rebuild runs after the complete retained-sample update transaction
and resets the mutation counter to zero. With an explicit interval of 4096, a
capacity of 64, an initially empty store, and no earlier recovery or batch fallback, the
first rebuild occurs after 2,080 accepted observations: 64 initial insertions,
then 2,016 insert/evict pairs. Further rebuilds occur every 2,048 accepted
observations while the store stays full. This counts accepted observations per
store, not prediction queries or wall-clock time.

There is at most one periodic rebuild after an update transaction. A full-store
insert/evict pair can cross an odd interval by one mutation; it still produces
one rebuild and a reset to zero. Rejected observations and spatial rebucketing
do not advance the counter. For linear fits, periodic rebuilding, numerical
recovery, and a conservative batch fallback all use one full-rebuild operation:
reaccumulate statistics and recompute batch coefficients from the same retained
rows, then reset the mutation clock. An identifiable all-zero linear candidate
preserves the previous serving snapshot even though the rebuild refreshes the
statistics and resets its clock. Other failed batch fits leave the model unready.
There is no fixed 256-observation gap or separate batch-fallback interval.
The setting is fixed for each store when the model is constructed.

The default and explicit Python `None` both disable only the periodic schedule.
Numerical recovery rebuilds and conservative batch fallbacks remain enabled:

```python
config = ForwardPassPerfModelConfig(
    model="Qwen/Qwen3-32B",
    system="h200_sxm",
    backend="vllm",
    worker_type="decode",
    estimation_mode="fpm_regression",
    estimator_config={"fpm_regression": {"fit": {"rebuild_interval": None}}},
)
model = RustForwardPassPerfModel.best_available(config)
saved = model.diagnostics()["provenance"]["config"]
restored = RustForwardPassPerfModel.best_available(saved)
```

Raw JSON uses `"fit": {"rebuild_interval": null}` at the same nested path.
In Rust the control is the public `RegressionFitConfig::rebuild_interval` field:

```rust
use aisimulate_core::{
    BackendKind, EstimationMode, ForwardPassPerfModel, ForwardPassPerfModelConfig,
    ForwardPassWorkerType,
};

let mut config = ForwardPassPerfModelConfig::new(
    "Qwen/Qwen3-32B", "h200_sxm", BackendKind::Vllm, ForwardPassWorkerType::Decode,
);
config.estimation_mode = EstimationMode::FpmRegression;
// Opt into periodic rebuilding; the default None retains numerical recovery.
config.estimator_config.fpm_regression.fit.rebuild_interval = Some(4096);
let model = ForwardPassPerfModel::best_available(config)?;
```

### DCP self-benchmark profiles

`dcp` is an optional recorded decode-context-parallel dimension within TP.
It must be positive and divide `tp`; it does not multiply GPU or MoE group
counts. Missing DCP and explicit DCP1 remain distinct FPM identities. An
explicit DCP8 request cannot consume ordinary TP8 or unrecorded-DCP data.
DCP greater than one currently supports measured vLLM `fpm_interpolation`
timing; op-level DCP and SOL-dependent transfer paths report unsupported.

FPM v6 accepts optional `dcp` in the Parquet identity and sidecar selector.
The loader also accepts `per_row_single_sample_or_median_of_3` sidecars when
every row declares a consistent `measurement_policy`/`measurement_repeats`
pair: `dynamo_native_single_sample_v1`/1 or `kvwarm_median_of_3`/3. Values are
already aggregated by the producer; the loader preserves their latency.

For a Kimi K3 text profile, an explicit configuration can be:

```python
config = ForwardPassPerfModelConfig(
    model="moonshotai/Kimi-K3", system="gb300", backend="vllm",
    backend_version="0.29.0", worker_type="aggregated",
    tp=8, pp=1, attention_dp=1, moe_tp_size=8, moe_ep_size=1, dcp=8,
    gemm_quant_mode="bfloat16", moe_quant_mode="w4a16_mxfp4",
    kvcache_quant_mode="fp8", attention_backend="FLASHINFER_MLA",
    estimation_mode="fpm_interpolation", fallback_policy="deny",
    systems_paths=("/absolute/profile/systems",),
    estimator_config={
        "fpm_interpolation": {
            "text_only": True,
            "unrecorded_quant_modes": ["fmha", "comm"],
        },
        "correction": {"enabled": False},
    },
)
model = RustForwardPassPerfModel.best_available(config)
```

Only use `unrecorded_quant_modes` for fields that the selected profile actually
leaves unrecorded. The runtime `FLASHINFER_MLA` label is retained for exact FPM
matching while the compiler uses its internal FlashInfer backend description.
The model architecture remains registered once; adding a parallel configuration
does not require another model class.

Place the reviewed primary pair at
`systems/data/gb300/vllm/0.29.0/fpm_forward_perf.{parquet,metadata.json}` and copy
the matching hardware YAML into the systems root. Keep source hashes and the
pinned dataset revision with the profile. Do not combine synthetic-attention
boundary points or nonuniform layouts with a balanced primary profile.

Replay YAML passes recorded DCP through
`engine.workers.<role>.parallelism.decode_context`. Per-worker `timing` accepts
the canonical quant-mode fields and `attention_backend`, alongside estimator
selection and controls. DCP FPM replay requires explicit fixed KV block capacity;
automatic DCP/hybrid capacity sizing is not implemented. Host offload or P/D
transfer also requires explicit KV bytes per token with DCP.
`decode_context` is currently supported by AISimulate's `--stack engine` only.
Dynamo Replay and Planner do not yet support this field; their configuration
propagation, cache identity, and dependency version need a downstream update.
These timing precision/backend overrides are prediction-only; recommendation
rejects them until its feasibility preflight supports the same identity.
Supplying a fixed pool does not add KDA checkpoint, eviction, or chunk-alignment
fidelity to the generic Replay cache/scheduler. This API change enables timing
consumption, not full hybrid-cache simulation or multimodal prediction from
text-only measurements.

The positional engine-spec wire format is version 25, including the new FPM
interpolation field. Recompile older binary EngineSpecs: schema 24 and other
incompatible versions are rejected before payload decoding. Legacy configuration
and profiles without DCP remain accepted as unrecorded DCP.

### Migrating saved configuration

See the [migration guide](MIGRATION.md#saved-performance-model-configuration) for required caller changes.

## Stable Rust facade

New embedded consumers should construct engines with
`aisimulate_core::perfmodel::AicEngineBuilder`. The
builder normalizes configuration into one private build request and enters
Python once to compile an engine specification. Calls on the returned
`AicEngine` are pure Rust and do not re-enter Python. Selected former AIC
crate-root types remain re-exported during the migration window, but the
`perfmodel` namespace is canonical for new code.

Native engine construction in standalone binaries requires the crate's `embed-python`
feature; applications hosted by an initialized Python interpreter do not need
auto-initialization. The matching `aisimulate` wheel must be importable for native
construction. Explicit `fpm_regression` construction works without the Python feature. See the
[crate README](../crates/core/README.md) for setup and usage examples.

See [SDK entry-point migration](MIGRATION.md#sdk-entry-points) for the removed
flat engine adapter and constructor replacements.

The supported `aisimulate_core::perfmodel` Rust surface is grouped as follows:

- compiled engine: `AicEngineBuilder`, `AicEngine`, `AicError`;
- forward-pass estimation: `ForwardPassPerfModel`,
  `ForwardPassWorkerType`, `ForwardPassPerfModelConfig`, `EstimatorConfig`,
  diagnostics/readiness/source types, and the `ForwardPassMetrics` telemetry
  types;
- KV-cache estimation: `estimate_kv_cache`, `KvCacheEstimateRequest`,
  `KvCacheEstimateOptions`, `KvCacheMemoryFraction`, and estimate/result/error
  types;
- FPM profile cache resources: `FpmCacheGroup`, `FpmCacheKind`, `FpmCacheLayout`,
  `FpmResourceConfig`, `FpmRuntimeMemoryConfig`, `FpmCacheBudgetRequest`, `FpmCacheBudget`, and
  `FpmCacheBudgetAdjusted`, through the canonical model/config budget method;
- wire identity: `EngineConfig`, `ParallelMapping`, `QuantizationConfig`,
  `SpeculativeConfig`, `BackendKind`, `DatabaseMode`, and `DataType`;
- schema gates: `ENGINE_CONFIG_SCHEMA_VERSION`,
  `ENGINE_SPEC_SCHEMA_VERSION`, and `FPM_VERSION`.

Advanced consumers may use `perfmodel::engine::{Engine, RuntimeConfig, StaticMode,
StaticResult, PerOpValue}` and `engine::spec::{EngineSpec, OpSpec}` to load and
execute a previously compiled specification directly. `PerOpValue` is the
per-op result tuple `(name, latency_ms, energy_wms, source)` returned by the
`*_per_op` / `evaluate_*` methods (the thin op-list evaluation FFI); per-op
energy is 0.0 wherever the perf tables carry no power columns. That zero is a
missing-data sentinel, not evidence of a zero-power operation. See the
[modeled-power contract](power-model.md) for the latency-weighted coverage gate,
aggregation rules, and public output boundary. Typed per-op energy alone does
not make unified replay power available.

## Direct FPM query evidence

The same model returned by `RustForwardPassPerfModel.best_available(config)` exposes `estimate_forward_pass_detailed(metrics)` (also available on the Rust `ForwardPassPerfModel`). Its `latency_ms` equals `estimate_forward_pass_time_ms(metrics)`; the scalar API and four-element per-operation tuple APIs are unchanged.

```python
result = model.estimate_forward_pass_detailed({
    "scheduled_requests": {
        "num_prefill_requests": 1,
        "sum_prefill_tokens": 512,
        "sum_prefill_kv_tokens": 256,
    },
})
for rank in result["ranks"]:
    for query in rank["queries"]:
        print(query["resolution"], query["query"], query["support"])
```

Each direct lookup records `exact_lookup`, `within_curve_interpolation`, `cross_kv_interpolation`, or `cross_batch_interpolation`, its phase and model, requested coordinates, raw latency, and supporting measured coordinates, latencies and weights. Coordinates are iteration totals: `batch_size`, `total_prefill_tokens` (null for decode), and `total_kv_read_tokens`. Nested interpolation weights are multiplied for each supporting measurement. This evidence preserves `source="silicon"`, whose meaning includes interpolation; it does not establish accuracy or observed CUDA graph dispatch.

Weights describe a raw lookup, before later composition. Each rank records prefill, decode and mixed-pass baseline latencies, plus `marginal_decode_ms = max(decode_ms - decode_baseline_ms, 0)`. A baseline lookup has `decode_baseline=true`: its query is the paired decode request, but its support contains each selected row's actual measured minimum-KV point. `max_rank` is the first zero-based input index attaining the positive native maximum. The result separately records `native_latency_ms`, `correction_factor`, and final `latency_ms`. Empty work has no query evidence. Other estimators also have no direct lookup evidence; an unready regression retains `latency_ms=null`.

Replay and prediction capture this evidence with `ReplayOutputRequirements(capture_performance_diagnostics=True)`; CLI prediction enables it with `--detail source` or `--detail time`. Use `--format json` or inspect the saved `prediction.json` for the query records and measured support; the default source text output shows operation sources and fallbacks. Public `run_recommendation(..., output_requirements=ReplayOutputRequirements(capture_performance_diagnostics=True))` and `Sweeper(..., output_requirements=...)` forward the same request to each candidate's replay. Evidence survives saved results under `ReplayReport.metadata["fpm_query_evidence"]` and each candidate's `provenance.runner_metadata["fpm_query_evidence"]`. The native replay report carries the same top-level key; source details include `fpm_estimates` on whole-model operations. FPM power remains unavailable and SOL comparison has an explicit unavailable reason.

Captured replay records are grouped by timing provider and phase. Each `fpm_estimates` entry holds an `estimate`, an invocation `count`, and `latency_scale` for any synthetic replay speedup; raw support is never rescaled. Identical query coordinates within an immutable replay provider share one counted record. Counts describe timing invocations in the measurement epoch, not independent silicon observations or served requests. Distinct queries are retained without truncation, so capture memory grows with distinct coordinates; ordinary runs do not retain query histories. Aggregated topology has provider index 0; disaggregated topology orders prefill before decode.

## Static phase diagnostics

The canonical `ForwardPassPerfModel` returned by `best_available` exposes
`static_phase_diagnostics(batch_size, context_length, prefix, prefill)` in Rust; the Python
`RustForwardPassPerfModel` wrapper accepts the same named arguments (with `prefix=0`).
It returns name-folded operation latency/energy, source tags, executed MoE communication
measurement substitutions, and optional SOL latency/compute/memory evidence. Decode means one
step at `context_length + 1`; prefill removes the cached prefix. Values precede learned online
correction. Whole-model estimators reject operation decomposition; missing SOL implementations
carry explicit reasons without changing the selected latency estimate.

Replay requests this evidence through `ReplayOutputRequirements(capture_performance_diagnostics=True)`.
`TimingOperationEvidence.details` is optional; providers without it must retain `None` (Rust
struct literals must initialize the new field). The constructor keeps it absent by default.
The CLI's time/source reports sum the observed phase work, including repeated cached timing
queries, and preserve distinct fallback records while folding repeated operation names.
Identical substitutions are deduplicated, so record counts are not execution counts.

## Replay timing evidence

The runtime-neutral `TimingModel` contract exposes optional accumulated
evidence through `evidence_summary()`. Op-level AIC providers return a
`TimingEvidenceSummary` split into prefill and decode phases. Each
`TimingPhaseEvidence` carries accumulated known energy in W-ms, total latency,
latency covered by nonzero energy data, merged provenance, and name-folded
`TimingOperationEvidence` records with the same fields. Missing operation
energy is represented by `None`, never by a synthesized zero. When coverage is
below one, the energy is partial: it includes only the portion with positive
operation-energy evidence. It is not a total-workload energy estimate.
Providers that assemble these public records directly should use
`TimingPhaseEvidence::try_from_operations` and `try_accumulate`; those paths
validate numeric fields and canonicalize covered latency to zero when energy is
missing. Nonempty operation lists must agree with phase totals; a relative
rounding tolerance applies only to this consistency check. The original infallible helpers remain available for already-valid
evidence.

Whole-model FPM timing and the built-in fixed and polynomial timing models are
latency-only and return `None` from `evidence_summary()`. Consumers must keep
that distinction when producing power metrics: absence of evidence is not a
zero-watt prediction. FPM decode timing continues to query the exact total
past-KV coordinate rather than the op-level mean-context coordinate.

## Agentic report source migration (0.13)

See the [migration guide](MIGRATION.md#agentic-report-source-migration) for required caller changes.

## Offload replay API migration

See the [migration guide](MIGRATION.md#offload-replay-api-migration) for required caller changes.

## Compatibility rules

- The `aisimulate` wheel and `aisimulate-core` crate versions must match for
  every release.
- A breaking `EngineConfig`, `EngineSpec`, or `ForwardPassMetrics` wire change
  must bump its corresponding schema constant. Consumers reject unsupported
  schema versions before using the payload.
- A supported facade name is not removed or given a new required parameter
  without a documented deprecation path. The package is pre-1.0, so an
  unavoidable incompatible API change also requires a minor-version bump.
- The canonical `best_available(config)` API replaces the old positional worker
  role/options signatures and separate estimator constructors. This is a source
  migration: use `ForwardPassPerfModelConfig::new(...)` in Rust or the SDK config
  class in Python, and use the explicit migration helper for saved EngineConfig
  values. Downstream Dynamo callers must migrate before this API's stable release;
  keep the crate and wheel versions aligned at the coordinated minor release.
- [The migration checklist](../.github/release-gates.json) and
  `scripts/release/check_release_migrations.py` apply before stable publication. Clear the
  pending entry in a reviewed change after downstream validation and merge.
  There is currently no standalone stable-publication workflow in this repository;
  that release process must invoke the checker for both its policy and target
  declarations (`--target-gates`). Missing or malformed declarations fail closed.
- The raw PyO3 class and ergonomic SDK wrapper intentionally share the name
  `RustForwardPassPerfModel`; callers should import from `aisimulate_core.sdk`
  unless they specifically need the JSON-oriented native binding.

The standalone AIC 0.12 compatibility distributions form a separate package
lineage: `aiconfigurator` pins `aiconfigurator-core==0.12.0`, and that core builds
its own native binding. They do not depend on the AISimulate wheel or crate.
Their retained binding declarations are not downstream callers of this API.
The [replacement installation](../README.md#upgrade-from-standalone-aiconfigurator)
removes both old distributions and installs AISimulate's complete migrated
application and core. Namespace compatibility does not preserve the removed
constructor signatures; direct SDK callers must perform the source migration
above. Release gates track consumers of the new artifacts, such as Dynamo.

## CI contract

Every change is checked from three consumer viewpoints:

1. Python source and isolated installed-wheel imports, including the public
   facade, native stub, bundled data, and upper/core ownership boundary;
2. Rust tests with embedding disabled and with all features enabled;
3. a separate workspace crate that depends on `aisimulate-core` and
   compiles only against its public exports.
