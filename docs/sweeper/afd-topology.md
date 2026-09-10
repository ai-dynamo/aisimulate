---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AFD Topology Contract
subtitle: Attention-FFN parallel shapes and complete enumeration
---

> [!WARNING]
> **Experimental.** The AFD parallel contract is available as a Sweeper-core Python API. The generic search
> domain does not yet place AFD topologies in a `SmartSearchConfig` study. That wiring
> depends on the shared execution-dimension work tracked by AIC-1773.

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

`total_gpus` is an upper-bound budget, not an exact allocation. Because AFD enumeration is
node-granular, a remainder smaller than `gpus_per_node` may remain unused while the enumerator
still covers every full-node topology within the budget.

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

## Infeasibility and Provenance

Every topology failure uses a stable category such as `invalid_topology`, `gpu_budget`,
`expert_divisibility`, or `candidate_limit`. Searched enumeration reports filter counts, the
canonical candidate dimensions, and whether its finite domain was complete. Pinned enumeration
identifies `AFDSearchConfig.pinned_topologies` as its source and does not claim generated candidate
dimensions. Each topology records its phase, parallel shape, and lossless A/F GPU accounting;
generator provenance is recorded by the enumeration that created it, not by manually constructed
topology objects.
