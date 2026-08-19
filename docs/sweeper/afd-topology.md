---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AFD Topology Contract
subtitle: Attention-FFN disaggregation, pipeline evaluation, and P/D rate matching
---

> [!WARNING]
> **Experimental.** Sweeper can place AFD topologies in generic rapid and thorough studies, but a
> backend remains unavailable unless its runner explicitly advertises the exact `afd` or `afd+pd`
> capability. Runtime serving and deployment-file generation remain adapter responsibilities.

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

The enumerator preserves the legacy canonical order. `candidate_overflow="error"` is the default;
explicit `"truncate"` returns a deterministic prefix and records `complete=false` in provenance.

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

An estimator adapter supplies non-negative per-layer A compute, F compute, A-to-F transfer, and
F-to-A transfer times. The core applies the legacy pipeline regimes:

- optimistic: `max(A, F, A_to_F + F_to_A)`, with the legacy minimum-microbatch check;
- conservative: `max(A + A_to_F, F + F_to_A)`; and
- serial: the sum of all compute and transfer stages.

Global step latency includes pipeline fill and every microbatch-layer cadence. Pure AFD can cover
prefill, decode, or both. When both phases use the same A/F pools, GPU count is not doubled.

## Combined AFD and P/D

`rate_match_afd_with_pd` pairs a single-phase AFD pool with static options for the other phase. It
considers every companion worker count through the rate-matched count, caps end-to-end sequence
rate at the slower phase, applies prefill/decode degradation and latency corrections, and selects
the highest output-tokens/s/GPU feasible combination. The result reports A, F, and companion GPU
counts separately. The companion domain is bounded at 256 options by default; like the topology
domain, exceeding that limit errors unless deterministic truncation is requested explicitly.

Adapters fail closed. A pure AFD topology requires an explicit `afd` capability; combined AFD+P/D
requires `afd+pd`. An adapter that advertises only `agg` or `disagg` cannot consume or generate an
AFD candidate.

## Generic Sweeper Integration

Set `deployment_mode: [afd]` for pure AFD or `[afd+pd]` for one AFD phase plus a conventional
opposite-phase companion. `afd_phase: both` is valid only for pure AFD. The companion uses the
same role-specific topology and scheduler fields as aggregate/P/D search, so rapid and thorough
consume the same pruned legal pool. AFD enumeration counts, pruning reasons, completeness, and
A/F/P-or-D GPU accounting are retained in the branch and materialized deployment provenance.

See [Sweeper Configuration](configuration.md#afd-domains) for the YAML schema and examples.

## Infeasibility and Provenance

Every hard failure uses a stable category such as `gpu_budget`, `expert_divisibility`,
`candidate_limit`, `unsupported_adapter`, or `no_feasible_companion`. Enumeration reports filter
counts and whether its finite domain was complete. Evaluation records the formula, corrected layer
times, topology, degradation factors, latency corrections, companion source, and lossless GPU
accounting.
