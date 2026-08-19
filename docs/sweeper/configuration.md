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
  backend_version:
    vllm: 0.11.0
    sglang: 0.5.6
  database_mode: HYBRID
  transfer_policy: balanced
  forward_model: op_level
  engine_step_backend: rust
  systems_paths: [default]

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
| `backend_version` | `None` | exact version for one backend, or a per-backend version mapping; omitted backends resolve once to latest |
| `database_mode` | `SILICON` | `SILICON`, `HYBRID`, `EMPIRICAL`, or `SOL` data mode |
| `transfer_policy` | `aggressive` | empirical transfer preset or tier list (`xshape`, `xquant`, `xprofile`, `xop`) |
| `forward_model` | `op_level` | granular `op_level` or exact-data `fpm` forward estimation |
| `engine_step_backend` | `rust` | compiled engine-step implementation (the only supported value) |
| `systems_paths` | `[default]` | ordered request-scoped system/data roots; `default` is the packaged Core root |
| `gpu_budget` | `32` | maximum GPUs per candidate |
| `min_gpu_budget` | `None` | optional lower bound during enumeration |
| `context_length` | `None` | compatibility alias for `max_seq_len`; both must match if set |
| `max_seq_len` | model maximum | sequence capacity used by KV feasibility and every engine's `max_model_len` |
| `parallel_configs` | `[]` | optional pinned parallel configurations |
| `startup_time` | `None` | optional simulated worker startup time |
| `aic_nextn` | `None` | optional speculative-decoding depth |
| `nextn_accepted` | `None` | required explicit expected accepted draft tokens when `aic_nextn` is set |
| `enable_chunked_prefill` | `false` | enable chunking on aggregated/prefill roles; selected `max_num_batched_tokens` remains the exact context-token budget |
| `enable_wideep`, `enable_eplb` | `false` | shared MoE WideEP/EPLB controls |
| `wideep_num_slots` | `None` | positive EPLB slot count |
| `moe_backend`, `attention_backend` | `None` | explicit supported MoE/MLA kernel backends |
| `gemm_quant_mode`, `moe_quant_mode`, `kvcache_quant_mode`, `fmha_quant_mode`, `comm_quant_mode` | `None` | shared quantization overrides used by KV feasibility and AIC timing |
| `free_gpu_memory_fraction` | role default | shared memory fraction, mapped to backend-native total/free-memory semantics |

Each engine role also has lists for `max_num_batched_tokens` and `max_num_seqs`, plus pinned block
size, GPU-memory-utilization, and prefix-caching fields. A one-item list pins a searched field.
Shared controls apply to both active roles in a disaggregated candidate. Unsupported model/backend
combinations are rejected before adapter preparation or replay, and the resolved values are retained
in `Candidate.config.engine_request` and `ReplaySpec.backend_deployment.engine_request`.

Estimator controls resolve before branch enumeration. `latest` becomes one concrete
backend/performance-data version per run, custom system paths remain request-scoped, and every
`ReplaySpec` plus returned candidate records the same model architecture, system, backend/version,
data root/mode, normalized transfer policy, forward model, and engine-step backend. Unavailable
versions and incomplete FPM data pairs fail before a sampler study is created.

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
