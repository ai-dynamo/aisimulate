---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AFD Topology Contract
subtitle: Attention-FFN disaggregation, pipeline evaluation, and P/D rate matching
---

> [!WARNING]
> **Experimental.** `SmartSearchConfig` can place AFD topologies in a generic Sweeper study, but
> the public recommendation schema and deployment artifacts do not yet support AFD. The built-in
> engine runner supports analytical `backend/afd` and `backend/afd+pd` replay for fixed synthetic
> lengths; it does not claim that a physical serving backend can launch the topology.

Attention-FFN Disaggregation (AFD) places attention operations on an A-worker pool and FFN/MoE
operations on an F-worker pool. `aisimulate.sweeper.afd` provides a backend-neutral contract for
enumerating and evaluating those shapes. Runtime serving and deployment generation remain adapter
responsibilities.

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
| `combined_with_pd` | exact `afd+pd` adapter capability and companion rate matching |
| `num_total_gpus` | A GPUs + F GPUs + any P/D companion GPUs |
| `t_a_layer`, `t_f_layer`, transfers | `AFDLayerTimes` |
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

## Pipeline Evaluation

`AICAFDPerformanceModel` uses AIC's public estimate API to supply full-precision, non-negative
per-layer A-pool, F-pool, A-to-F transfer, and F-to-A transfer times. The measurement request pins
the model, hardware, backend version, topology, and workload lengths. Its transfer inputs are
uncalibrated: the core applies `comm_overhead_factor` exactly once when evaluating the candidate.
The resulting `ReplaySpec` records the measurement API version, units, source, workload point, and
backend version. Missing phases, duplicate phases, unsupported estimates, and OOM results fail
closed during candidate materialization.

This measurement layer currently requires a synthetic workload with concrete positive `isl` and
`osl`. A trace-only AFD sweep is rejected because one fixed A/F layer measurement cannot honestly
represent requests with differing sequence lengths. Trace-aware measurement belongs with the AFD
replay lifecycle in a later layer.

The core applies the legacy pipeline regimes:

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

## Analytical Replay

`EngineReplayRunner` advertises AFD only after validating and consuming the complete measurement
contract. Pure `afd` replay requires `phase: both`; a single AFD phase must use `afd+pd` so the
opposite phase is present. The latter measures a regular companion through AIC static estimation
or an explicit fixed timing model, then schedules the two independent pools as a two-stage flow.
This preserves arrival and queueing delay in TTFT and end-to-end latency while keeping decode step
latency separate as TPOT. Reports include throughput, goodput, GPU-hours, per-request latency on
request, batch/pass counts, and exact A/F plus companion GPU accounting.

Replay currently requires fixed synthetic `isl` and `osl`, `random_range_ratio: 1.0`, and no trace.
Those restrictions keep each request aligned with the performance-model point instead of silently
reusing a measurement at a different sequence length.

## Combined AFD and P/D

`rate_match_afd_with_pd` pairs a single-phase AFD pool with static options for the other phase. It
considers every companion worker count through the rate-matched count, caps end-to-end sequence
rate at the slower phase, applies prefill/decode degradation and latency corrections, and selects
the highest output-tokens/s/GPU feasible combination. The result reports A, F, and companion GPU
counts separately. The companion domain is bounded at 256 candidates by default; like the topology
domain, exceeding that limit fails instead of returning a partial result. The bound counts each
concrete `(option, worker-count)` combination, so the exhaustive search cannot expand into an
unbounded worker loop.

Adapters fail closed. A pure AFD topology requires an explicit `afd` capability; combined AFD+P/D
requires `afd+pd`. An adapter that advertises only `agg` or `disagg` cannot consume or generate an
AFD candidate.

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

Every hard failure uses a stable category such as `gpu_budget`, `expert_divisibility`,
`candidate_limit`, `unsupported_adapter`, or `no_feasible_companion`. Enumeration reports filter
counts and whether its finite domain was complete. Evaluation records the formula, corrected layer
times, topology, degradation factors, latency corrections, companion source, and lossless GPU
accounting.
