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
Each pool resolves the caller's precision request independently against its own
selected system. Requests remain unchanged in saved configurations. The supported
`nvfp4` execution mapping uses `nvfp4_wo` (FP4 weights, BF16 compute) below
Blackwell and native `nvfp4` on Blackwell; explicit `nvfp4_wo` remains weight-only.
No other precision is silently substituted. Unsupported execution combinations
fail at construction; missing performance data follows the existing data policy.
Timing, memory and KV budgeting share this resolution. Pool provenance records
`requested_quantization` and `effective_quantization`; the latter lists compiled
GEMM/MoE modes, including mixed-precision exclusions. These modeling modes are
not a runtime checkpoint deployability guarantee. The [public configuration](../replay/engine/analytical.md#hardware-and-versions)
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
all-gathering restored hidden states. Return combine/merge/all-gather is an
A-resource stage after the F→A transfer, competing with attention from other
microbatches. Its memory, energy and diagnostic latency belong to A. Uneven slices use actual
token counts, with collective padding where required. Router `topk` determines
expert-selection volume, not a separate top-k selection kernel's latency.

Same-system A↔F transfers select node/rack/scale-out fabric from the combined
A/F GPU span; different systems use scale-out. Each endpoint is priced separately,
and the slower complete transfer wins. Cross-rack latency falls back to
`p2p_latency` and cannot be lower. If the pair exceeds one rack, all cross-pool
traffic uses scale-out rather than a weighted cross-rack fraction. F-local
collectives and A-local combine do not use that deployment-wide span.

## Staged Evaluation

The per-layer ingredients are `d0` (A computation, including router/shared
experts), `d1` (A→F), `d2` (F all-gather, routed compute, reduce-scatter), `d3`
(F→A), and `d4` (A combine, shared merge, token all-gather).

`four_stage` uses four resources and the dependency chain
`A[d0] → send[d1] → F[d2] → return[d3] → A[d4]`.
`two_stage` groups compute and its outgoing transfer continuously:
`A[d0+d1] → F[d2+d3] → A[d4]`. The first and final A visits share one server;
restoration is not an independent fifth server. Each microbatch's next layer
waits for its own restoration.

Ready tasks are non-preemptive and ordered by readiness time, with restoration
first on ties, followed by source layer and microbatch ordinal. All completions
at a timestamp settle before resources dispatch more work. The model does not
assume partial GPU-kernel overlap. Every pass starts empty and drains completely;
there is no pipeline state across replay calls. Small depths use stack queues;
larger depths use bounded priority queues, with workspace proportional to depth.

For two microbatches and one layer with `d=[10,1,1,1,10]` ms:

| Layout | A resource work | Pass time |
| --- | --- | --- |
| `four_stage` | two 10 ms computations plus two 10 ms restores | 40 ms |
| `two_stage` | the same plus two 1 ms sends | 42 ms |

`comm_overhead_factor` scales only `d1` and `d3`, once before grouping. F-local
NCCL, A-local TP and restoration are not transfer-scaled. Final latency correction
scales the complete pass. `attention_ms` describes total A service, including
restoration, while `f_to_a_ms` describes only the return transfer. These are
averaged diagnostics; replay consumes the already evaluated complete pass and
never reconstructs it from averaged ingredients.

Communication readiness checks the active F all-gather/reduce-scatter and A
all-gather data buckets against each pool's selected database before replay
creates workers. SOL needs no measured communication table; measured/empirical
modes require the corresponding dtype/operation bucket. Message-size and
interpolation-domain coverage remain query-time checks.

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
current mean KV position. A pool-wide batch is balanced across replicas, with
larger populations assigned first, then each replica is balanced across effective
`m = min(num_microbatches, ceil(pool_batch / a_replica_count))` microbatches.
Idle replicas and short tails contribute no invented tokens. Eight sequences
on one replica and three microbatches split as 3 + 3 + 2; one sequence across
eight replicas still prices F compute for one sequence. A latency follows the
busiest complete replica/rank branch; F compute uses the exact microbatch total.
Communication uses the actual modulo rank mapping, with padding only where
collectives require it. Energy sums participating ranks; its normalization still
uses the deployed GPU count. Configuration-sized memory preflight remains a
capacity reservation. Tokens include uncached prefill length or decode
verification width. At most three distinct microbatch shapes need querying.

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
Manually constructed topologies do not invent generator provenance. AFD topology
and qualification schemas are version 2; ReplaySpec remains API 1. Obsolete inputs
follow [migration rules](../aic-backward-compatibility/migration.md#afd-translation).
See [output artifacts](../replay/engine/analytical.md#output-artifacts) for qualification.

Native expectations live in `afd_execution_plan.json` and `afd_e2e.json` under
[`crates/core/parity_tests/perfmodel/goldens/`](../../crates/core/parity_tests/perfmodel/goldens/).
`gen_afd_spec_fixtures.py` rebuilds pool fixtures through the canonical compiler.
`scripts/afd_parity.py refresh-e2e` and `refresh-layouts` require `--build-manifest`: an immutable source
commit/tree, covered source-file hashes, loaded native-binary hash and data tree.
It refuses unrecorded source/binary/data changes, retains generation identity per
case, checks exact request/token/GPU counts, and replaces the fixture only after
all selected cases succeed. Historical attribution is not current generation
provenance.
Historical comparisons and host-cost measurements belong under
[`benchmarks/evidence/`](../../benchmarks/evidence/), separate from serving accuracy.

<!-- Heterogeneous AFD/fabric behavior adapted and modified from
https://github.com/LPhgh/aisimulate/tree/af56328c4ba7ffa1b8c42c8375244568c2e7bb8c
Original paths and Apache-2.0 attribution: root THIRD_PARTY_NOTICES.md. -->
