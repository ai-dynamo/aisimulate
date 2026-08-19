---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Minimum-GPU Recommendation
subtitle: Size evaluated candidates for a target request rate or concurrency
---

Minimum-GPU recommendation is a deterministic post-processing step over evaluated
`Candidate` objects. It does not run another simulation or invent topology-specific
capacity. Call `recommend_min_gpus` with exactly one target shape:

```python
from aisimulate.sweeper import LoadTarget, recommend_min_gpus

recommendations = recommend_min_gpus(
    candidates,
    LoadTarget(request_rate=250.0),
    goal=config.goal,
    osl=config.workload.osl,
)
```

`request_rate` and `concurrency` must be positive and finite. Supplying both or neither
is an error. Request-rate sizing reads the evaluated `request_throughput_rps` capacity;
concurrency sizing reads `supported_concurrency`, `kv_load_concurrency_capacity`, or the
evaluated candidate `concurrency`, in that order. Missing, non-positive, or non-finite
capacity fails closed with an actionable reason.

## Sizing and ranking

For one candidate, the uncapped minimum is:

```text
replicas_needed = ceil(target_load / capacity_per_replica)
total_gpus_needed = replicas_needed * candidate.used_gpus
```

Recommendations rank full-service deployments first, then by uncapped GPU requirement,
served percentage, capacity per GPU, latency, and canonical configuration. This keeps
ordering deterministic and prevents a capped partial deployment from outranking a
candidate that can serve the complete target.

When `goal.strict_sla` is enabled, candidates are filtered with the same inclusive
aggregate SLA policy used by scalar and Pareto analysis before they are sized. Per-request
goodput remains part of the evaluated candidate metrics; recommendation does not weaken or
reinterpret it.

## GPU ceilings and partial service

`LoadTarget(max_gpus=N)` is a deployment ceiling, not a change to the true requirement.
If the uncapped minimum exceeds the ceiling, recommendation fails closed by default.
Set `allow_partial=True` to return a capped deployment explicitly:

- `replicas_needed` and `total_gpus_needed` remain the true uncapped minimum;
- `deployed_replicas` and `deployed_gpus` describe the capped deployment;
- `supported_load`, `load_served_pct`, and `partial` make the shortfall visible.

A ceiling smaller than one candidate replica is always infeasible.

## Topology extensions

End-to-end replay capacity is authoritative. Optional role-level keys such as
`encoder_request_throughput_rps`, `prefill_request_throughput_rps`,
`decode_request_throughput_rps`, `attention_request_throughput_rps`, and
`ffn_request_throughput_rps` identify the limiting pool and provide a fallback for EPD and
AFD evaluators that publish only role capacities. GPU accounting always starts from the
candidate's complete `used_gpus`, so every aggregate, P/D, encoder, attention, and FFN
worker represented by the evaluator is included without topology logic in the sizing
layer.

The canonical Sweeper result envelope owns persistent run output. Until that envelope
adopts load recommendations, `LoadRecommendation.model_dump_json()` is the lossless
machine-readable form; deployment artifact generation remains outside this module.
