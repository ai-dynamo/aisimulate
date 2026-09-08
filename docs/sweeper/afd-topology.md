---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AFD Topology Contract
subtitle: Attention-FFN parallel shapes and complete enumeration
---

> [!WARNING]
> **Experimental.** `SmartSearchConfig` can place AFD topologies in a generic Sweeper study, but
> performance measurement, staged evaluation, the public recommendation schema, production runner,
> and deployment artifacts do not yet support AFD.

Attention-FFN Disaggregation (AFD) places attention operations on an A-worker pool and FFN/MoE
operations on an F-worker pool. `aisimulate.sweeper.afd_parallel` provides the backend-neutral
parallel shape, validation, GPU accounting, and complete finite enumeration. Performance
measurement, staged evaluation, generic search integration, replay, and deployment generation
belong to later layers.

## Legacy Mapping

| Legacy AIC field or result | Sweeper AFD contract |
|---|---|
| `n_a_nodes` / `n_f_nodes` | `AFDTopology.n_a_nodes` / `n_f_nodes` |
| `tp_a` | `AFDTopology.tp_a` |
| derived `tp_f` | `AFDTopology.ffn_tp` (`n_f_nodes * gpus_per_node`) |
| `f_moe_ep_size` | `AFDTopology.f_moe_ep_size` |
| `a_batch_size` | `AFDTopology.a_batch_size` |
| `num_microbatches` | `AFDTopology.num_microbatches` |
| `pipeline_model` | `optimistic`, `conservative`, or `serial` |
| `combined_with_pd` | whether a later layer must add the complementary P/D phase |
| `num_total_gpus` | A GPUs + F GPUs for this AFD topology |
| AFD rejection logs | `AFDInfeasible.category`, detail, and provenance |

Phase-1 F-side semantics are intentionally strict: one F replica spans all F GPUs, so F TP equals
the number of F workers. A-side workers are the A GPU count divided by A TP. Expert parallelism
must divide both F TP and the model expert count when the latter is known.

## Pinned and Searched Shapes

Complete `AFDTopology` objects passed in `AFDSearchConfig.pinned_topologies` are preserved exactly.
When that tuple is empty, `enumerate_afd_topologies` searches:

- A-node and F-node counts within the GPU budget;
- A-worker TP and batch size;
- F-worker MoE EP, including `n_f_nodes` and `ffn_tp` symbolic choices;
- microbatch count and pipeline model; and
- A:F node ratio.

The enumerator preserves the legacy canonical order and evaluates the complete finite domain.
If the domain exceeds `max_candidates`, it fails with `candidate_limit` instead of returning a
partial result.

```python
from aisimulate.sweeper import AFDSearchConfig, enumerate_afd_topologies

domain = enumerate_afd_topologies(
    AFDSearchConfig(
        total_gpus=64,
        gpus_per_node=8,
        is_moe=True,
        num_experts=256,
        tp_a_candidates=(2, 4, 8),
        a_batch_size_candidates=(64, 128),
        f_moe_ep_size_candidates=("n_f_nodes", "ffn_tp"),
        microbatch_candidates=(3, 4),
    )
)
```

## Performance Measurements

`AICAFDPerformanceModel` uses AIC's public estimate API to supply full-precision, non-negative
per-layer A-pool, F-pool, A-to-F transfer, and F-to-A transfer times. The measurement request pins
the model, hardware, backend version, topology, and workload lengths. Transfer inputs are
uncalibrated so the staged engine can apply `comm_overhead_factor` exactly once.

The resulting `ReplaySpec` records the measurement API version, units, source, workload point, and
backend version. Missing phases, duplicate phases, unsupported estimates, and OOM results fail
closed during candidate materialization.

This measurement layer requires a synthetic workload with concrete positive `isl` and `osl`.
A trace-only AFD sweep is rejected because one fixed A/F layer measurement cannot represent
requests with differing sequence lengths. Trace-aware measurement belongs with a later replay
lifecycle.

## Foreground Engine

The foreground-engine layer applies the legacy pipeline regimes:

- optimistic: `max(A, F, A_to_F + F_to_A)`, with the legacy minimum-microbatch check;
- conservative: `max(A + A_to_F, F + F_to_A)`; and
- serial: the sum of all compute and transfer stages.

Global step latency includes pipeline fill and every microbatch-layer cadence. Pure AFD can cover
prefill, decode, or both. When both phases use the same A/F pools, GPU count is not doubled.

`AFDForegroundEngine` expands that same formula into deterministic A, A-to-F, F, and F-to-A
intervals for every layer and microbatch. Starting a pass eagerly fixes the whole non-preemptive
schedule; completion effects remain hidden until its modeled full-pass boundary. A second pass
cannot start while one is in flight, and a late caller wakeup does not inflate the modeled
completion time. A topology covering both phases executes prefill and decode as separate full
passes through the same engine. For `afd+pd`, this engine owns only the configured AFD phase; the
ordinary companion remains a replay-layer responsibility.

## Generic Sweeper Domain

The internal `SmartSearchConfig` schema accepts `deployment_mode: [afd]` for a pure A/F pool and
`deployment_mode: [afd+pd]` for a single-phase A/F pool plus an opposite-phase P/D companion. Both
use one finite `parallel_config_choice` dimension in the standard sampler. AFD+P/D constructs the
complete legal topology-by-companion product within `gpu_budget`; `afd_max_candidates` rejects an
oversized product instead of silently truncating it.

Use `afd_pinned_topologies` to provide concrete A/F shapes, or provide an explicit,
memory-qualified `afd_batch_size_candidates` list with the other `afd_*_candidates` fields to
generate the domain. `afd_companion_parallel_configs` can pin the ordinary parallel shapes used by
the opposite phase. Pure AFD has no ordinary engine argument payload. AFD+P/D materializes engine
arguments only for its companion phase.

KV-relative traffic load is intentionally rejected for AFD in this layer because the A/F pools do
not yet expose scheduler-visible KV capacity. Use a synthetic request rate or absolute concurrency
with concrete `isl` and `osl`.
See [Sweeper Configuration](configuration.md#attention-ffn-disaggregation) for an internal example.

## Infeasibility and Provenance

Every topology failure uses a stable category such as `invalid_topology`, `gpu_budget`,
`expert_divisibility`, or `candidate_limit`. Enumeration reports filter counts, the canonical
candidate dimensions, and whether its finite domain was complete. Each topology records its phase,
parallel shape, and lossless A/F GPU accounting.
