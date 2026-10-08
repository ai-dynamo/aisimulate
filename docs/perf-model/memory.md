<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPU memory and cache capacity

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

For Mistral3's gated Pixtral encoder, the analytical activation estimate includes
the replicated residual alongside the gate and up intermediates. Encoder tensor
parallelism shards those intermediates; encoder data parallelism keeps a full
tower on each rank. This remains an analytical lower bound, not a measured peak.

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
because deserialization defaults it to zero. See [Rust literal migration](../aic-backward-compatibility/migration.md#rust-resource-and-memory-literals)
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
`FpmResourceConfig::require_memory()`. See [Rust literal migration](../aic-backward-compatibility/migration.md#rust-resource-and-memory-literals)
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

Cache reuse, eviction, and offload lifecycle are documented in [Replay KV cache](../replay/engine/kv-cache.md).
