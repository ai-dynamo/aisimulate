<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance-model limitations

## TensorRT-LLM eager host execution

Op-level GPU costs do not represent all host-side submission gaps and cross-rank
waiting in eager prefill or mixed execution. The current ordinary op-level
configuration does not carry a generic measured eager/graph CPU-dispatch timing
contract. Supplying `cuda_graph_reserved_bytes` changes cache capacity; it does
not select or measure host execution latency.

The [GPT-OSS trace evidence](../../python/aisimulate/collector/trtllm/gym-gpt-eager-20260918.md)
records this gap for its pinned runtime and workload. Its observed client TTFT
residual cannot be treated as a universal constant or charged to the shared MoE
kernel table: graph decode has a different execution boundary, and queue/frontend
time is separate. Preserve the exact runtime, phase, capture coverage, and timing
boundary when interpreting those measurements.

Whole-forward measurements can encode their recorded execution regime, but do
not establish behavior after changing that regime. Likewise, a regression may
fit observed aggregate behavior without an explicit graph/capture input. Neither
route makes unmeasured execution-policy changes qualified.

## Other scoped boundaries

- [Support](support-matrix.md) distinguishes engine queries from Replay and accuracy.
- [Whole-forward](methods/whole-forward.md) documents exact identity and query-domain failures.
- [Memory](memory.md) distinguishes physical payload/resource estimates from observed allocations.
- [DeepSeek-V4.1](models/deepseek-v41.md) separates analytical and measured execution contracts.
- [Replay features](../replay/features.md) records supported scheduler/cache/topology combinations.

These are current modeling boundaries. Completed implementation plans and
historical PR checklists are not part of this guide.
