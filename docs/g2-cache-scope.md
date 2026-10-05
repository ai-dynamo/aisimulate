<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# G2 host-cache scope

Native vLLM host offload (G2) can give every attention-DP rank its own host
cache, as vLLM does, or model one host pool shared by a whole deployment.
All controls live in `engine.workers.<role>.kv_cache.host_offload`:

```yaml
host_offload:
  scope: cluster_shared          # default: dp_rank_local
  num_host_blocks: 4096
  d2h_bandwidth_gbps: 32         # per DP rank
  h2d_bandwidth_gbps: 32         # per DP rank
  shared_d2h_bandwidth_gbps: 80  # whole pool; cluster_shared only
  shared_h2d_bandwidth_gbps: 80  # whole pool; cluster_shared only
  latency_to_first_byte_ms: 0
```

## Ownership

| Scope | Pools | `num_host_blocks` | Transfer model |
| --- | --- | --- | --- |
| `dp_rank_local` (default) | One per DP rank | Per rank | Independent D2H and H2D FIFO lanes per rank |
| `cluster_shared` | One per deployment | Pool total | Equal sharing under rank and pool caps |

Two replicas with attention DP 2 therefore have four private 4096-block caches,
or one 4096-block shared cache. G1 and request state always stay with the DP
rank, and every completion is delivered to the rank that started the transfer.
`worker_local` is a G3 scope; using it for G2 fails with a hint to use
`dp_rank_local`.

`cluster_shared` is a simulator extension. Native vLLM `CPUOffloadingSpec`
keeps one cache per engine, so there is no native parity reference for it.

## Transfers and bytes

Bandwidths are decimal GB/s; zero removes a limit. D2H and H2D are independent.
First-byte latency delays a transfer without consuming bandwidth.

In a shared pool, a moving transfer runs at
`min(rank_budget / rank_jobs, pool_budget / pool_jobs)` in its direction, and
every admission, completion or cancellation re-divides the bandwidth. Four
ranks at 32 GB/s under an 80 GB/s pool each move at 20 GB/s while all four
transfer. G2 and G3 use the same equal-share service, each with its own budgets.

The bytes of a block are `block_size * bytes_per_token`. With
`bytes_per_token: auto` this is the estimate for **one tensor-parallel shard**;
no TP multiplier is applied to host capacity or transfer time. An explicit
`bytes_per_token` replaces the estimate, so capacity and bandwidth must be
stated in the same unit.

## Topologies and compatibility

G2 supports aggregated and token-only disaggregated vLLM replay with any
attention-DP size. Prefill and decode choose their scopes independently; every
role that selects `cluster_shared` joins the one deployment pool.

Ranks may share a pool only when they agree on the stored KV layout, tensor
parallel size, block size and bytes, pool capacity and pool bandwidths; access
bandwidth and latency may differ per role. Public YAML derives the layout
identity (`kv_layout_id`) from the resolved model, backend and backend version,
TP, PP, DCP, KV quantization, attention backend (including per-worker timing
overrides) and block geometry. Fixed and polynomial timing carry no model
identity, so there only the backend, TP, block size and bytes per token
participate. A value set explicitly to its default, such as
`decode_context: 1`, derives a different identity than leaving it unset, so
such roles do not share a pool. Raw ReplaySpec descriptors must set a
non-blank `native_host_offload.kv_layout_id` themselves. Incompatible
participants fail at construction; the pool is never split silently.

`recommend` keeps `host_offload` fixed. When both prefill and decode use
`cluster_shared`, each role must set `tensor` and `pipeline` to the same integer
with `parallelism.preset: false`; an omitted value is searched per role. Their
KV block geometry, `num_host_blocks`, `shared_d2h_bandwidth_gbps` and
`shared_h2d_bandwidth_gbps` must also match, and either both roles use default
timing or neither does, because only default timing adds the model identity.
Violations are rejected before any candidate runs. Replicas and attention DP may
still be searched.

## Lifecycle

- A store is visible to all ranks when it physically completes, whichever rank
  advances the shared clock; the owner still consumes its own completion.
- A G1 block is never overwritten while its D2H is still in flight. With
  private lanes the block can be reallocated, and the compute that reuses it
  waits for the copy's fixed deadline. A shared pool instead holds the block
  until the copy actually completes, as Mooncake Store and LMCache MP keep a
  storing request's blocks allocated. A held block remains a G1 prefix hit but
  cannot be evicted or reallocated, so a request short of G1 capacity waits or
  preempts as usual. When the copy completes, its blocks return to the free
  LRU tail first, so the shared prefix is evicted last. Those connectors hold
  every block of the storing request; the simulator holds only the blocks
  being copied.
- With G3 write-through, a completed D2H source stays pinned until its owner
  hands it to G3, so no peer can evict it in between. Blocks G3 declines, for
  example because it already holds them, are released at that handoff and then
  follow the same tick's H2D completions in LRU order. This matches native vLLM
  0.27.1, which always runs the secondary store job and releases the CPU source
  on its completion, and applies to both scopes whenever G3 is enabled.
- Cancelling an H2D after it physically completed withdraws only its delivery;
  a departing rank releases its undelivered held sources exactly once.

### Decode ranks

Every decode rank with G2 enabled, in either scope, follows two rules:

| Rule | Behavior |
| --- | --- |
| Handoff destination | Handoff KV is not received into G1 blocks still being copied to G2; the reservation waits until the D2H completes. |
| Admission order | Waiting requests whose handoff KV is already materialized are admitted before other waiting requests. |

The admission order is modeled on native vLLM's FCFS policy, which schedules
requests that finished receiving remote KV (`skipped_waiting`) ahead of the
waiting queue. Decode ranks without G2 keep the existing order. Native vLLM's
priority policy instead compares the heads of the two queues; the simulator
models only FCFS.

### G3 restore bypass

With G3 enabled, a restore promotes blocks into G2 before loading them. When the
recoverable prefix is larger than G2 can hold, each promotion can evict a block
the same request promoted earlier, so the request never reaches compute. The
simulator bounds this thrash:

- Each lookup that submits new G3 promotions for a waiting request is one
  restore round. Waiting for promotions already in flight is not a round.
- The count resets whenever the request's DP rank schedules work or emits
  output, and when the request is admitted or preempted. Only rounds while the
  whole rank is stalled count, so a request waiting behind running work is
  never bypassed.
- After 1,024 consecutive rounds, the request stops restoring from G3 and
  computes what G1 and G2 do not hold, and `g3_offload.bypassed_restores` in the
  report increments. The field is omitted when zero. The bypass lasts until the
  request is admitted or preempted.

The rule has two limits:

- While the rank keeps scheduling work or emitting output, a thrashing request
  is never bypassed. In `cluster_shared` scope its promotions meanwhile keep
  displacing other ranks' G2 blocks, and `bypassed_restores` stays zero.
- A rank can also stall without a real livelock, and a request that would have
  progressed is then bypassed: when H2D or G3 pins hold the capacity it needs
  while the rank is otherwise idle, or when a peer rank evicts this worker's
  promotions from the shared pool.

Native vLLM has no such rule.

## Observability

Reports add `g2_domains` for a shared pool (capacity, resident and used
blocks); it is omitted otherwise. Ranks that publish KV events also receive the
shared pool's residency changes as `Stored`/`Removed` events with
`"tier": "host_pinned"`, starting with a snapshot when a rank joins. A child
block can become resident before its parent, so each `Stored` run carries its
first block's prompt index as `start_position`, and the snapshot is ordered by
position so that parents precede children. The live stream keeps landing order,
and a snapshot taken after a parent was evicted holds its children without it,
so consumers must place blocks by `start_position`, not by arrival. Every rank
of a shared pool retains its blocks' `tokens_hash`, so a block's router identity
does not depend on whether the rank that stored it publishes events.
Device events omit `tier`, and a missing `tier` reads as `device`.
Private G2 keeps the existing G1-only event stream.

## Example

[`examples/cli/shared-g2-predict.yaml`](../examples/cli/shared-g2-predict.yaml)
runs 64 requests sharing an 896-token prefix on two replicas with attention
DP 2 and fixed compute timing. With `cluster_shared` the reuse ratio is 0.861
and one 1,024-block pool holds 568 blocks; overriding `scope=dp_rank_local`
gives 0.820, because each DP rank must first compute the prefix itself. Mean
TTFT rises from 20 ms to 30.6 ms in shared scope: fixed compute time does not
shrink with reuse, while restores add H2D time.

## GPU parity

Private G2 (`dp_rank_local`) was compared with native vLLM 0.25.1 serving
MiniMax-M2.5 with TP2 and attention DP 2 on four B200 GPUs, BF16 KV and
32-token blocks. Each arm ran the same 5,000 Mooncake requests on a fixed
schedule. Each DP engine had 8,192 G1 blocks; the G2 arm added 16,384 G2 blocks
per DP engine. The simulator used the existing vLLM 0.24.0 timing profiles and
the saved GPU inputs: prompt token IDs, ingress times and DP routes.

| Arm | Metric | GPU | Simulator | Absolute relative error |
| --- | --- | ---: | ---: | ---: |
| G1 only | G1 reuse | 25.7605% | 25.6863% | 0.29% |
| | Mean TTFT | 770.268 s | 666.276 s | 13.50% |
| | Mean TPOT | 69.514 ms | 67.556 ms | 2.82% |
| G1 + G2 | G1 reuse | 26.1932% | 25.9547% | 0.91% |
| | G2 reuse | 7.1087% | 6.9254% | 2.58% |
| | Combined reuse | 33.3019% | 32.8801% | 1.27% |
| | Mean TTFT | 755.304 s | 636.553 s | 15.72% |
| | Mean TPOT | 67.305 ms | 63.793 ms | 5.22% |

- GPU tier reuse is the run's delta of `vllm:prompt_tokens_by_source_total`
  (`local_cache_hit` is G1, `external_kv_transfer` is G2) over one common
  denominator of 46,542,292 prompt tokens. Simulator reuse is attributed to the
  tier a request's prefix came from at first admission.
- Errors are single-run aggregate relative errors, not repeated-run MAPE.
- The replay set `bytes_per_token: 253952`, the per-token KV footprint of a
  whole DP engine across its two attention-TP ranks. That is twice the 126,976
  bytes `auto` estimates for one TP shard, so the replay's 32 GB/s D2H and H2D
  limits apply to the whole-engine footprint.
- The G1-only arm recorded 74 native preemptions (42 on DP0, 32 on DP1)
  against 82 simulator readmissions. With G2, native preemptions rose about
  elevenfold to 830 (387 on DP0, 443 on DP1), while the simulator still recorded
  82. The counters are defined differently, and their contribution to the timing
  error was not isolated.
- Event-by-event scheduler parity is not claimed. `cluster_shared` has no native
  reference.

## Unsupported

Non-vLLM backends, native MTP, Belady eviction, recurrent state-cache offload,
and detailed replay artifacts with `cluster_shared` are rejected. G3 still
requires aggregated replay with attention DP 1.

Agentic snapshot/warmup host offload is supported for one aggregated worker or
one prefill plus one decode worker, with vLLM, attention DP=1 on every role,
static worker pools, and speculative decoding disabled. Both `dp_rank_local`
and `cluster_shared` are supported; AgentX excludes G3. See [AgentX with vLLM
host offload](agentx-g2.md) for shared-pool compatibility requirements and
functional qualification limits.
