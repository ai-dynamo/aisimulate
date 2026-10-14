---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AFD Topology Contract
subtitle: Topology enumeration and analytical pass execution
---

This page describes SDK topology enumeration and the native A/F timing model.
For YAML fields, deployment combinations, supported inputs and output artifacts,
see [AFD configuration](../replay/engine/analytical.md#afd).
AFD models analytical pass timing inside continuous SGLang replay; it has no
physical deployment renderer.

## Topology Identity

The enclosing role (`aggregated`, `prefill`, or `decode`) owns phase identity.
Public YAML maps `tp_a` to `a_tp_size` and `a_batch_size` to
`batch_size_per_a_replica`. Internal topology mappings and pool configurations
use canonical names:

| Name | Unit or meaning |
| --- | --- |
| `a_replica_count` | Independent A TP groups, each owning a batch; `a_gpu_count / a_tp_size`. |
| `f_rank_count` | Physical F GPU ranks sharing one routed expert pool. |
| `a_gpu_count`, `f_gpu_count` | Physical GPUs in each pool. |
| `a_tp_size`, `f_moe_tp_size`, `f_moe_ep_size` | Parallel group sizes; F is pure EP with TP1. |
| `batch_size_per_a_replica` | Sequences per A replica. |
| `tokens_per_a_replica` | Microbatch token count in a communication query. |

The compiler's `f_pool_width` is physical pool width, not MoE TP. A TP is
node-local and must divide the node width and active attention head groups.
F EP equals `f_gpu_count` and must divide the expert count. An omitted EP resolves
to that width; an explicitly saved EP1 is not rewritten for a multi-GPU F pool.

<a id="generic-sweeper-domain"></a>

## Pinned and Searched Shapes

`AFDSearchConfig.pinned_topologies` preserves complete shapes after validation.
Otherwise `enumerate_afd_topologies` enumerates node counts, A TP candidates
(default TP1), explicit replica batches, F EP, microbatches and the A:F node ratio.
`total_gpus` is an upper bound; enumeration uses full nodes with independent
A/F widths. `gpus_per_node` is A width; omitted `f_gpus_per_node` inherits it.
Widths must match the corresponding system specs.

Each shape/microbatch count appears once in deterministic node/batch/microbatch
order. Pinned shapes inherit omitted layout/routing settings and must agree when
explicit. Explicit A TP candidates are preserved. Generated F widths requiring
MoE TP greater than 1 are filtered along with expert-divisibility and budget
failures. Fixed EP values are preserved: EP8 on 8-GPU nodes retains one-node F
pools even when the budget also permits larger F pools. If no generated shape
is feasible, enumeration fails with `no_feasible_topology`; concrete and pinned
shapes requiring F TP still fail as not implemented.
F-side routing fails execution preflight rather than being silently filtered.
Exceeding `max_candidates` fails with `candidate_limit`, never truncation.
Role-domain composition and whole-deployment limits belong to
[search space](search-space.md#attention-ffn-disaggregation).

```python
from aisimulate.sweeper import AFDSearchConfig, enumerate_afd_topologies

domain = enumerate_afd_topologies(
    AFDSearchConfig(
        total_gpus=64,
        gpus_per_node=8,
        is_moe=True,
        num_experts=256,
        a_tp_size_candidates=(1,),
        batch_size_per_a_replica_candidates=(64, 128),
        f_moe_ep_size_candidates=("f_gpu_count",),
        microbatch_candidates=(3, 4),
    )
)
```

<a id="native-execution-contract"></a>
<a id="analytical-replay"></a>

## Performance Model

`run_replay_json` materializes the `afd` timing provider on SGLang's continuous
scheduler, including mid-flight admission and retraction. There is no separate
AFD scheduler or compile-time measurement step. Each pool is built through
`ForwardPassPerfModel::best_available`, preserving `estimator_config`, explicit
`enable_shared_layer`/`strict_provenance` choices and its selected data roots.
For different pool systems, inherited F GEMM/MoE precision is cleared and resolved
against F hardware. The [public configuration](../replay/engine/analytical.md#hardware-and-versions)
defines shared version selection and GPU accounting.

Topology resides in `rank.afd`; the rank owns backend, MTP depth and context.
Timing and GPU accounting share this immutable identity. Native `max_model_len`
is an optional declared context window: when absent, the runtime derives a
scheduler limit from `isl + osl` without serializing that derived value back.
Memory preflight retains the declaration. Public memory behavior is documented
[with the configuration](../replay/engine/analytical.md#memory-and-capacity).
Native explicit `num_gpu_blocks` overrides derived capacity; plan buffers are
added once, and only decode activations use MTP verification width.

[`models/afd.py`](../../python/aisimulate/src/aisimulate_core/sdk/models/afd.py)
composes both pool plans; [native validation](../../crates/core/src/perfmodel/afd/validation.rs)
checks matching source identity, ordered layers and operation ownership.
A owns attention, router, dense FFNs, shared experts, boundary norms,
residual/combination work, embedding and logits. Dense FFNs follow A TP;
shared experts retain full weights per A GPU with balanced rank-local token slices.
F owns routed experts only. Attention/dense projections retain A-local TP
reductions; aggregate dispatch is replaced by native AFD communication.
Pure MoE reuses one template; supported mixed plans have a contiguous dense
prefix followed only by routed layers, preserving SWA/global and DSA index reuse.
Timing, physical residency and energy use these same plans.

## Communication

Native `AfdTransfer`, `AfdFAllGather`, `AfdFReduceScatter` and `AfdCombine` model
cross-pool transfers and F/A restoration. A ranks dispatch their token slices;
F-node reductions produce partials that A combines with shared outputs before
all-gathering restored hidden states. Return combine/merge/all-gather latency
belongs to F→A, while its memory and energy remain on A. Uneven slices use actual
token counts, with collective padding where required. Router `topk` determines
expert-selection volume, not a separate top-k selection kernel's latency.

Same-system A↔F transfers select node/rack/scale-out fabric from the combined
A/F GPU span; different systems use scale-out. Each endpoint is priced separately,
and the slower complete transfer wins. Cross-rack latency falls back to
`p2p_latency` and cannot be lower. If the pair exceeds one rack, all cross-pool
traffic uses scale-out rather than a weighted cross-rack fraction. F-local
collectives and A-local combine do not use that deployment-wide span.

## Staged Evaluation

The ordered-stage kernel evaluates `[A, A_to_F, F, F_to_A]` for `four_stage`
or `[A + A_to_F, F + F_to_A]` for `two_stage`. Each stage has one server;
merged stages run continuously. Shared computation joins A; return restoration
joins F→A. The replay loop consumes one complete `step_latency` per pass.

For `m` circulating microbatches, `K` stages of durations `d`, and
`j = layer_index * m + microbatch_index`:

```text
C(j, 0) = max(C(j-1, 0), C(j-m, K-1)) + d[0]
C(j, r) = max(C(j, r-1), C(j-1, r)) + d[r]  (r > 0)
```

Negative indices are zero; completion is `C(m * layers - 1, K - 1)`.
Each pass starts empty and drains completely, retaining no state across calls.
Microbatches can enter the next layer before peers finish the previous one.
With four 1 ms stage times, two layers and `m=2`:

| Layout | Layer 1 F→A completions | Layer 2 F→A completions | Pass time |
| --- | --- | --- | --- |
| `four_stage` | 4, 5 ms | 8, 9 ms | 9 ms |
| `two_stage` | 4, 6 ms | 8, 10 ms | 10 ms |

With `m=1`, both layouts take 8 ms. Work is `O(K * m * layers)` and workspace
`O(K + m)`; depths 1–4 use stack storage. `comm_overhead_factor` scales transfers
once before grouping, excluding local A TP and return restoration; final latency
correction scales the complete pass. Averaged layer diagnostics are not billed
through another pipeline evaluation.

### Batch pricing

The dense prefix, including embedding and A-local TP reductions, runs once at
the exact full A-replica batch before splitting into routed microbatches.
Its queries, energy and activation peak use that batch; its time is amortized
only in layer diagnostics. Without a dense prefix, embedding runs once per
microbatch. Logits always run once per microbatch after the routed layers;
vocabulary-sharded embedding retains its one-time A-TP all-reduce.

Prefill uses actual batch, uncached tokens and prefix. Fixed synthetic decode
(`random_range_ratio: 1.0`) caches a full-batch timing/evidence pair after
computing each KV stride sample's complete pass and averaging those passes.
Trace/session/agentic and variable synthetic decode price each pass at its
current mean KV position. Replica batch is `max(1, ceil(pool_batch / a_replica_count))`;
effective `m` is `min(num_microbatches, replica_batch)`, and each microbatch is
priced at `ceil(replica_batch / m)`. Eight sequences with three microbatches
therefore pay for 3 + 3 + 3 routed sequences, but a dense prefix pays for exactly
8 once. Two sequences use two microbatches of size 1. Configured `a_batch_size`
sets capacity, not effective microbatch count. Tokens include uncached prefill
length or decode verification width; rank-local slices conserve logical tokens.

Instantaneous operation latency and energy come from the same actual-batch/KV
queries as pass timing. For energy normalization see [power](../perf-model/power.md).
[MTP scaling](../replay/engine/speculation.md#mtp--eagle) applies before
pipeline evaluation and evidence, independently of the transfer factor;
compiled A-local TP collectives are not scaled twice. P→D handoff is separate:
the source establishes TTFT, while the destination emits every requested output
token without adding the source's first-token signal to output accounting.

<a id="qualification-artifacts"></a>

## Infeasibility and Provenance

Failures carry categories including `invalid_topology`, `gpu_budget`,
`expert_divisibility`, `out_of_memory` and `candidate_limit`. Generated enumeration
records dimensions, filter counts and completeness; pinned enumeration identifies
`AFDSearchConfig.pinned_topologies` without claiming generated dimensions.
Manually constructed topologies do not invent generator provenance. Topology
schema, ReplaySpec API and qualification schema are version 2; obsolete inputs
follow [migration rules](../aic-backward-compatibility/migration.md#afd-translation).
See [output artifacts](../replay/engine/analytical.md#output-artifacts) for qualification.

Native expectations live in `afd_execution_plan.json` and `afd_e2e.json` under
[`crates/core/parity_tests/perfmodel/goldens/`](../../crates/core/parity_tests/perfmodel/goldens/).
`gen_afd_spec_fixtures.py` rebuilds pool fixtures through the canonical compiler.
Historical comparisons and host-cost measurements belong under
[`benchmarks/evidence/`](../../benchmarks/evidence/), separate from serving accuracy.

<!-- Heterogeneous AFD/fabric behavior adapted and modified from
https://github.com/LPhgh/aisimulate/tree/af56328c4ba7ffa1b8c42c8375244568c2e7bb8c
Original paths and Apache-2.0 attribution: root THIRD_PARTY_NOTICES.md. -->
