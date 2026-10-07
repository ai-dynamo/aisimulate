<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python performance-model API

The `aisimulate` wheel provides the `aisimulate_core` estimator namespace and
native extension. It does not require a separate core wheel or Dynamo.

Operation energy alone does not guarantee published power. Unified reporting
also applies the [modeled-power coverage gate](../power.md#publication-gate).

## Canonical estimator

```python
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

config = ForwardPassPerfModelConfig(
    model="Qwen/Qwen3-32B", system="h200_sxm", backend="vllm",
    worker_type="aggregated", tp=2,
    estimation_mode="auto", fallback_policy="deny",
)
model = RustForwardPassPerfModel.best_available(config)
```

| Method | Contract |
| --- | --- |
| `estimate_forward_pass_time_ms(metrics)` | One iteration's rank metrics, supplied as a dictionary or list of rank dictionaries; returns milliseconds, or `None` for an unready regression store. Empty work returns zero. |
| `estimate_forward_pass_detailed(metrics)` | Same scalar latency plus native latency, correction, rank composition, and any direct lookup evidence. |
| `tune_with_fpms(iterations)` | Consume observed iterations for correction or regression. Preserve rank/iteration grouping and worker identity. |
| `static_phase_latency(batch_size=..., input_tokens=..., output_tokens=..., prefill=...)` | Native static phase integration before correction. Prefill is one pass; decode is total sequence decode, so `output_tokens=2` represents one decode iteration. |
| `static_phase_diagnostics(...)` | Operation diagnostics for supported native paths; details below. |
| `diagnostics()` | Readiness and requested/resolved estimator provenance. |
| `regression_store_diagnostics()` | Independent per-workload-store fit and retention state. |
| `estimate_cache_budget(config, budget)` | Class-level canonical resource-budget API; see [memory](../memory.md). |

Use [configuration](../configuration.md) for identity and controls,
[whole-forward FPM](../methods/whole-forward.md) for measured query domains,
and [online regression](../methods/online-regression.md) for telemetry semantics.
`best_available` validates configuration before selection; query-domain errors
do not silently change the chosen estimator. The raw PyO3 class and SDK wrapper
are different surfaces: new applications should use the SDK wrapper.

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
