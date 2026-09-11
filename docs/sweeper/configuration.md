---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweeper Configuration
subtitle: Core fields and optional adapter-owned search spaces
---

> [!WARNING]
> **Experimental.** Sweeper's configuration schema may change without a standard deprecation period.

`SmartSearchConfig.search_space` contains backend and deployment fields. Optional feature-specific
search spaces are mappings under `SmartSearchConfig.adapters`.

For fixed-image E+agg/E+P+D search, see [Analytical EPD search](epd.md).
The SDK uses `search_space.encoder` with `workload.images`; it reuses AIC's
encoder model and does not expose per-request EPD replay or deployment outputs.

## Top-Level Shape

```yaml
search_space:
  model_name: example/model
  hardware_sku: h200_sxm
  gpu_budget: 32
  deployment_mode: [disagg, agg]
  backend: [vllm, sglang]

adapters:
  example.policy:
    search_space:
      mode: [balanced, latency]

workload:
  isl: 1024
  osl: 128
  request_rate: 4
  num_request_ratio: 10

goal:
  target: throughput_per_gpu
  # Optional legacy-compatible aggregate gating. Replay goodput remains
  # per-request; strict_sla filters aggregate means before ranking.
  strict_sla: true
  sla:
    ttft_ms: 800
    itl_ms: 30

sweep:
  max_rounds: 10
  candidates_per_round: 8
  parallel_evals: 4
```

The adapter value is a search space, not one concrete runtime configuration. Its provider validates
the whole mapping, contributes optimizer dimensions, and later materializes one concrete adapter
configuration for each candidate.

## Backend Fields

| Field | Default | Purpose |
|---|---|---|
| `model_name` | required | model identifier |
| `hardware_sku` | required | AI Configurator system identifier |
| `deployment_mode` | `[disagg, agg]` | deployment branches to search |
| `backend` | `[vllm]` | engine backends to search |
| `gpu_budget` | `32` | maximum GPUs per candidate |
| `min_gpu_budget` | `None` | optional lower bound during enumeration |
| `context_length` | `None` | optional KV-feasibility sequence length |
| `parallel_configs` | `[]` | optional pinned parallel configurations |
| `startup_time` | `None` | optional simulated worker startup time |
| `aic_nextn` | `None` | optional speculative-decoding depth |

Each engine role also has lists for `max_num_batched_tokens` and `max_num_seqs`, plus pinned block
size, GPU-memory-utilization, prefix-caching, and `<role>_forward_model` fields (`op_level` by default,
or `fpm` for whole-forward timing from a collected FPM cell). A one-item list pins a searched field.

## Explicit Parallel Domains and Preparation Limits

For `agg` and `disagg`, each role (`agg`, `prefill`, `decode`) accepts positive-integer
lists named `<role>_<dimension>_candidates`. Dimensions are `num_gpu` (GPUs per worker),
`tp`, `pp`, `dp` (attention DP), `moe_tp`, `moe_ep`, `cp`, and `num_workers`.
Omitted lists retain the existing menu: PP=CP=1 and GPUs per worker in `[1, 2, 4, 8, 16]`.
An explicit shape domain derives its GPU sizes from the configured dimensions unless
`num_gpu` is supplied; values are then checked against capabilities and the GPU budget.
A worker-count-only override preserves the existing GPU menu. SDK candidate lists cannot
be combined with pinned, custom, or independent controls for the same role.

```yaml
search_space:
  model_name: meta-llama/Llama-3.1-8B-Instruct
  hardware_sku: h200_sxm
  deployment_mode: [agg]
  backend: [sglang]
  gpu_budget: 8
  context_length: 4096
  agg_num_gpu_candidates: [8]
  agg_tp_candidates: [4, 8]
  agg_pp_candidates: [1, 2]
  agg_dp_candidates: [1]
  agg_num_workers_candidates: [1]
  max_parallel_combinations: 1000000
  max_parallel_configs: 100000
```

The example admits TP8/PP1 and TP4/PP2 when both pass model and KV checks. Public
`aisimulate recommend` exposes these axes through worker `parallelism.pipeline` and
`parallelism.context`, alongside the existing parallelism fields. Explicit values are
not intersected with the default menu. PP/CP are opt-in. CP currently requires an AIC model
family that supports CP on SGLang, TP=DP=1, and an aggregated or prefill role. CP with an
encoder, native host offload, or custom KV-transfer geometry is rejected. External runners
must advertise PP/CP support. Native replay carries PP/CP into AIC timing, KV estimation,
and GPU accounting; it retains AIC's analytical pipeline timing model.

`num_gpu_per_replica` restricts total candidate GPU sizes and `max_gpu_per_replica`
sets their ceiling. For disaggregation that total includes both roles and all workers;
`max_prefill_workers` and `max_decode_workers` independently bound their worker counts.
AFD retains its separate topology controls and limits.

Preparation shares a work budget across configured agg/disagg modes and backends.
`max_parallel_combinations` bounds integer-domain expansion, shape products, worker
products, and prefill/decode pairing **before** their loops or allocations.
`max_parallel_configs` bounds each retained topology pool and the backend union.
Exceeding either raises `SearchSpaceLimitError` with the stage and a request to narrow
the domain; no partial domain is returned. These are deterministic preparation guards,
not a process-memory limit or a timeout for model loading and simulation.

The existing Bayesian/random optimizer and `sweep.max_trials` still govern evaluation;
an expanded domain does not imply exhaustive evaluation. Scheduler fields such as
`max_num_seqs` and `max_num_batched_tokens` are capacity limits. They do not pin actual
batch occupancy or context tokens. Actual operating-point parity and broader automatic
PP/CP defaults require separate qualification and resource controls.

## Attention-FFN Disaggregation

AFD is supported by public `aisimulate predict` and `aisimulate recommend` with the built-in
analytical engine runner. See the [AFD Topology Contract](afd-topology.md) for public configuration
and replay limits. The internal `SmartSearchConfig` schema uses topology `afd` or `afd+pd`;
an injected runner must explicitly advertise the selected backend with the chosen topology.

This pinned pure-AFD example creates a finite standard Sweeper branch:

```yaml
search_space:
  deployment_mode: [afd]
  backend: [trtllm]
  model_name: Qwen/Qwen3-32B
  hardware_sku: h200_sxm
  gpu_budget: 32
  afd_phase: both
  afd_pinned_topologies:
    - n_a_nodes: 2
      n_f_nodes: 2
      tp_a: 4
      a_batch_size: 64
      f_moe_ep_size: 1
      num_microbatches: 3
      pipeline_model: optimistic

workload:
  isl: 1024
  osl: 128
  concurrency: 64
  num_request_ratio: 10
```

Use `deployment_mode: [afd+pd]` with `afd_phase: prefill` or `decode` to search a companion for the
opposite phase. The companion uses that phase's ordinary `max_num_batched_tokens`, `max_num_seqs`,
block-size, and memory fields. Optional `afd_companion_parallel_configs` pins its parallel shapes.
Searched (non-pinned) AFD requires an explicit `afd_batch_size_candidates` list. The finite domain
also supports `afd_tp_a_candidates`, `afd_f_moe_ep_size_candidates`,
`afd_microbatch_candidates`, `afd_pipeline_model_candidates`, and `afd_max_candidates`.

AFD rejects `kv_load_ratio` until the execution layer exposes scheduler-visible KV capacity. The
current performance-model adapter also requires concrete positive `isl` and `osl`, so use a
synthetic request rate or absolute concurrency rather than a trace-only workload. The complete
topology and capability contract is documented in [AFD Topology Contract](afd-topology.md).

## Pinned Parallel Configurations

Pinning `parallel_configs` requires exactly one deployment mode. An aggregated entry is one shape:

```yaml
search_space:
  deployment_mode: [agg]
  parallel_configs:
    - tp: 4
      attention_dp: 2
      replicas: 2
```

A disaggregated entry contains `prefill` and `decode` shapes. Every pinned shape must be legal,
KV-feasible, supported by at least one selected backend, and accepted by the configured Replay
runner. If every selected backend/topology pair is runner-incompatible, preflight raises
`aisimulate.sweeper.RunnerIncompatibleError` with the rejected mode and backend names.

## Provider Selection

Adapter names are provider entry-point names. A provider can be installed through the
`aisimulate.sweep_config_providers` entry-point group or injected into the `Sweeper` constructor:

```python
sweeper = Sweeper(
    runner_factory=my_runner_factory,
    providers={"example.policy": my_provider},
)
```

Sweeper loads only names present under `adapters`. See [Sweep Configuration
Providers](sweep-config-provider.md) for the complete ABI.

## Strict Aggregate SLA

The default SLA behavior is replay goodput: requests are individually counted against every
configured bound, and an unset TTFT or ITL field is unbounded. Set `goal.strict_sla: true`
to additionally require the aggregate candidate means to stay within every configured bound
before scalar ranking or Pareto dominance. Bounds are inclusive. Missing metrics and zero
qualifying latency samples fail closed.

## Sampler Algorithm Override

The experimental `AISIMULATE_SWEEPER_VIZIER_ALGO` environment variable overrides the Vizier
algorithm. For example, set it to `RANDOM_SEARCH` to bypass the default GP-bandit designer.
`SPICA_VIZIER_ALGO` remains a deprecated fallback during migration; when both are set, the
AI Simulate variable takes precedence.

## Removed KVBM Fields

Sweeper rejects the old KVBM block-count, transfer-bandwidth, offload-batch-size, and cache-hit
fields. The AI Simulate engine and replay path do not support them, and they have no adapter
migration.
