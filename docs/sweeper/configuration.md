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
Shared controls apply to both active roles in a disaggregated candidate. Unsupported model/backend
combinations are rejected before adapter preparation or replay, and the resolved values are retained
in `Candidate.config.engine_request` and `ReplaySpec.backend_deployment.engine_request`.

Estimator controls resolve before branch enumeration. `latest` becomes one concrete
backend/performance-data version per run, custom system paths remain request-scoped, and every
`ReplaySpec` plus returned candidate records the same model architecture, system, backend/version,
data root/mode, normalized transfer policy, forward model, and engine-step backend. Unavailable
versions and incomplete FPM data pairs fail before a sampler study is created.

## Heterogeneous Prefill/Decode

Disaggregated search accepts `prefill_` and `decode_` overrides for model, system, backend,
backend version, estimator/data controls, sequence capacity, memory fraction, WideEP/EPLB,
MoE/attention backends, and every quantization mode. An omitted role field inherits the matching
unprefixed field. Existing configurations without role overrides therefore use the original
homogeneous search path unchanged.

```yaml
search_space:
  deployment_mode: [disagg]
  model_name: example/shared-model       # inherited when a role omits model_name
  hardware_sku: h200_sxm                 # decode inherits this value
  backend: [vllm]                        # decode inherits this search list
  gpu_budget: 16

  prefill_model_name: example/prefill-model
  prefill_hardware_sku: gb200_nv18
  prefill_backend: [sglang]
  prefill_backend_version: 0.5.6
  prefill_systems_paths: [/data/gb200-systems]
  prefill_enable_wideep: true
  prefill_moe_backend: deepep_moe
  prefill_free_gpu_memory_fraction: 0.81

  decode_model_name: example/decode-model
  decode_backend_version: 0.11.0
  decode_gemm_quant_mode: fp8
  decode_free_gpu_memory_fraction: 0.72
```

Prefill and decode backend lists form searched backend pairs. Each role resolves its model,
system definition, exact performance-data version, engine controls, and KV-feasible worker shapes
independently. Sweeper then pairs role shapes only when their combined GPU count is within
`gpu_budget` (and `min_gpu_budget`, when set). The replay contract retains `prefill_backend`,
`decode_backend`, both exact versions, role estimator specs, and role engine requests; candidate
configuration retains the same role identities and provenance.

The engine runner supports mixed vLLM/SGLang P/D pairs and reports an unsupported backend by role.
TensorRT-LLM remains fail-closed for either disaggregated role. `decode_enable_chunked_prefill=true`
is also rejected because chunked prefill is a prefill-role scheduler control.
Optional adapter providers must explicitly return
`AdapterSearchPlan(supports_heterogeneous_pd=True)`; providers that have not audited their
materialization against role-specific identities are rejected.

Legacy-calibrated P/D matching defaults are explicit inputs: `prefill_rate_degradation=0.9`,
`decode_rate_degradation=0.92`, `prefill_latency_correction=1.1`,
`decode_latency_correction=1.08`, and `ttft_correction_factor=1.8`. The public
`rate_match_disaggregated` result preserves per-role rates, identities, GPU totals, the limiting
role, correction provenance, and role-attributed budget failures.

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
