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

sweep:
  policy: rapid
  seed: 0
  max_rounds: 10
  candidates_per_round: 8
  parallel_evals: 4
```

## Search Policy Fields

| Field | Default | Purpose |
|---|---|---|
| `policy` | `rapid` | `rapid` bounded optimization or `thorough` complete finite enumeration |
| `seed` | `0` | unsigned 32-bit rapid optimizer seed; provenance-only for canonical thorough order |
| `max_rounds` | `20` | rapid optimizer rounds; ignored as a stop rule by thorough |
| `candidates_per_round` | `parallel_evals` | rapid success target or thorough evaluation/callback batch size |
| `parallel_evals` | `16` | replay worker fan-out |
| `max_eval_seconds` | `600` | per-candidate timeout on the worker-pool path |

Thorough search rejects continuous ranges because they do not form a finite candidate set. See
[Search Policies](search-policies.md) for ordering, completion, and reporting semantics.

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

`deployment_mode` also accepts `afd` for pure attention--FFN disaggregation and `afd+pd` for
one AFD phase plus a conventional opposite-phase companion. A runner must advertise the exact
`(backend, deployment_mode)` pair before either branch enters the search domain.

## Parallel and Execution Domains

Every role accepts an explicit finite candidate list for GPUs per worker, TP, PP,
attention DP, MoE TP, MoE EP, CP, actual scheduler batch/context limits, and worker count:

```yaml
search_space:
  deployment_mode: [disagg]
  prefill_num_gpu_candidates: [4, 8]
  prefill_tp_candidates: [1, 2, 4]
  prefill_pp_candidates: [1, 2]
  prefill_dp_candidates: [1]
  prefill_moe_tp_candidates: [1]
  prefill_moe_ep_candidates: [4, 8]
  prefill_cp_candidates: [1, 2, 4]
  prefill_batch_size_candidates: [1, 2, 4]
  prefill_context_tokens_candidates: [8192, 16384]
  prefill_num_workers_candidates: [1, 2]

  decode_num_gpu_candidates: [4, 8]
  decode_tp_candidates: [1, 2, 4]
  decode_pp_candidates: [1]
  decode_dp_candidates: [1, 2, 4]
  decode_moe_tp_candidates: [1]
  decode_moe_ep_candidates: [4, 8]
  decode_cp_candidates: [1]
  decode_batch_size_candidates: [256, 512]
  decode_context_tokens_candidates: [8192]
  decode_num_workers_candidates: [1, 2, 4]

  num_gpu_per_replica: [8, 16, 24, 32]
  max_gpu_per_replica: 32
  max_prefill_workers: 2
  max_decode_workers: 4
```

Use the same fields with the `agg_` prefix for aggregated deployments. An omitted topology
list uses capability-derived defaults: CP is offered only for model/backend combinations that
declare CP support, decode CP remains 1, and PP=2 is added for DeepSeek V3.2/V4 on Blackwell.
Configured lists are authoritative and are pruned deterministically by GPU-count, MoE-width,
backend, KV-feasibility, worker-count, and replica-budget rules before sampling. Rapid and
thorough consume the resulting `BranchSpace.parallel_configs`; thorough enumerates all of it,
while rapid projects optimizer suggestions onto that identical legal pool.

`*_batch_size_candidates` and `*_context_tokens_candidates` are clearer aliases for the
replay scheduler's `max_num_seqs` and `max_num_batched_tokens`; when supplied, they take
precedence over the older lists for that role. `EnumerationDiagnostics` exposes stable
considered/accepted counts plus pruning-reason counts for topology enumeration.

Each engine role also has lists for `max_num_batched_tokens` and `max_num_seqs`, plus pinned block
size, GPU-memory-utilization, and prefix-caching fields. A one-item list pins a searched field.

## AFD Domains

AFD search uses the same legal `BranchSpace.parallel_configs` pool for rapid and thorough
policies. Rapid projects optimizer suggestions onto that finite pool. Thorough enumerates the
entire pool in canonical order. A pure AFD candidate accounts for attention (A) and FFN (F) GPUs;
an `afd+pd` candidate also accounts for every GPU in its conventional prefill or decode
companion.

```yaml
search_space:
  model_name: example/moe-model
  hardware_sku: h200_sxm
  backend: [vllm]
  deployment_mode: [afd+pd]
  gpu_budget: 64

  # AFD owns decode; the companion is drawn from the prefill domain below.
  afd_phase: decode
  afd_tp_a_candidates: [4, 8]
  afd_batch_size_candidates: [64, 128]
  afd_f_moe_ep_size_candidates: [n_f_nodes, ffn_tp]
  afd_microbatch_candidates: [3, 4]
  afd_pipeline_model_candidates: [optimistic, conservative]

  prefill_num_gpu_candidates: [4, 8]
  prefill_tp_candidates: [4, 8]
  prefill_num_workers_candidates: [1, 2]
  prefill_batch_size_candidates: [1, 2, 4]
  prefill_context_tokens_candidates: [8192, 16384]
```

Use `afd_phase: both` only with pure `afd`. For `afd+pd`, select `prefill` or `decode`; Sweeper
draws the companion from the opposite role's execution fields. Decode companions always use
CP=1. The global `gpu_budget` and optional `min_gpu_budget` apply to the complete A+F+P/D
deployment. `afd_max_candidates` bounds the finite topology domain; the default
`afd_candidate_overflow: error` preserves completeness instead of silently truncating it.

Pin exact topologies with `afd_pinned_topologies` and exactly one AFD deployment mode:

```yaml
search_space:
  deployment_mode: [afd]
  afd_phase: both
  afd_pinned_topologies:
    - n_a_nodes: 2
      n_f_nodes: 1
      tp_a: 4
      a_batch_size: 128
      f_moe_ep_size: 8
      num_microbatches: 3
      pipeline_model: optimistic
```

Model, hardware, GPU-per-node, phase, and pure-versus-combined facts remain authoritative and are
added during materialization. AFD candidates currently reject `kv_load_ratio`: the generic replay
path cannot derive scheduler-visible KV capacity for split A/F pools. Use request rate,
concurrency, or a trace instead.

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
KV-feasible, and supported by at least one selected backend.

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

## Sampler Algorithm Override

The experimental `AISIMULATE_SWEEPER_VIZIER_ALGO` environment variable overrides the Vizier
algorithm. For example, set it to `RANDOM_SEARCH` to bypass the default GP-bandit designer.
`SPICA_VIZIER_ALGO` remains a deprecated fallback during migration; when both are set, the
AI Simulate variable takes precedence.

## Removed KVBM Fields

Sweeper rejects the old KVBM block-count, transfer-bandwidth, offload-batch-size, and cache-hit
fields. The AI Simulate engine and replay path do not support them, and they have no adapter
migration.
