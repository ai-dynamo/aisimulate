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
  forward_pass_fallback_policy: error
  forward_pass_options:
    min_observations: 5
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
| `backend_version` | `None` | exact version for one backend, or a per-backend version mapping; omitted backends resolve once to latest |
| `database_mode` | `SILICON` | forward-pass estimator data-source policy; see below |
| `transfer_policy` | `None` (all) | Core-owned empirical-transfer preset or tier list; used only by `HYBRID` and `EMPIRICAL` |
| `forward_model` | `op_level` | granular `op_level` or exact-data `fpm` forward estimation |
| `forward_pass_fallback_policy` | `error` | fail closed, or explicitly use observation-gated `regression` when native construction is unsupported |
| `forward_pass_options` | `None` (Core defaults) | runtime tuning controls such as observation limits, regression buckets, correction bounds, and workload-axis capacity |
| `systems_paths` | `[default]` | ordered request-scoped system/data roots; `default` is the packaged Core root |
| `gpu_budget` | `32` | maximum GPUs per candidate |
| `min_gpu_budget` | `None` | optional lower bound during enumeration |
| `context_length` | `None` | optional KV-feasibility sequence length |
| `parallel_configs` | `[]` | optional pinned parallel configurations |
| `startup_time` | `None` | optional simulated worker startup time |
| `aic_nextn` | `None` | optional speculative-decoding depth |

Each engine role also has lists for `max_num_batched_tokens` and `max_num_seqs`, plus pinned block
size, GPU-memory-utilization, and prefix-caching fields. A one-item list pins a searched field.

Estimator controls resolve through Core before branch enumeration. `latest` becomes one concrete
backend/performance-data version per run, custom system paths remain request-scoped, and every
`ReplaySpec` plus returned candidate records the same resolved config, options, and provenance.
Unavailable identities fail before a sampler study is created. Regression is never an implicit
degradation: it must be requested with `forward_pass_fallback_policy: regression`, and it remains
unready until `tune_with_fpms` supplies enough observations for the workload kind.

Database modes choose the source of each operation estimate:

| Mode | Resolution |
|---|---|
| `SILICON` | collected performance data and supported interpolation only |
| `HYBRID` | collected data first; calibrated empirical estimation for uncovered operations |
| `EMPIRICAL` | calibrated estimation for every operation (`latency = SOL / utilization`) |
| `SOL` | uncalibrated analytic speed-of-light estimate |

`transfer_policy` is not a search dimension. It selects which fixed Core transfer kinds the
empirical forward-pass estimator may use when its own calibration slice is missing. It accepts `off`,
`conservative`, `balanced`, or `aggressive`, or an explicit list containing `xshape`, `xquant`,
`xprofile`, and `xop`. Core validates the request and the Sweeper records the normalized explicit
policy in every candidate. The field is ignored by `SILICON` and `SOL`.

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
