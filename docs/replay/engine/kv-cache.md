<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV cache

`kv_cache` configures each worker role's KV cache tiers:

| Tier | Where | Configure with |
| --- | --- | --- |
| G1 | GPU memory | `block_size`, `prefix_caching`, `bytes_per_token`, `capacity` |
| G2 | Host (CPU) memory | `host_offload` |
| G3 | Storage shared by workers | `g3_offload` |
| G2 peer fetch (KVCR) | Another rank's G2 | `kvcr` |

Recurrent-state models also use `state_cache`; see
[state cache](#state-cache).

Merge this fragment into an existing `engine` block from the [engine overview](README.md):

```yaml
engine:
  workers:
    aggregated:                  # or prefill / decode
      kv_cache:
        block_size: 64
        prefix_caching: true
        bytes_per_token: auto
        capacity: {type: default, memory_fraction: 0.9}
        host_offload:            # optional G2
          num_host_blocks: 4096
        g3_offload:              # optional G3: aggregated only, requires host_offload
          scope: cluster_shared
          num_g3_blocks: 16384
        # kvcr: {}               # optional KVCR peer fetch: not on decode, requires private host_offload, not with g3_offload
```

## KV cache fields

All paths below are under `engine.workers.<role>.kv_cache`.

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `block_size` | vLLM `64`, SGLang `1`, TensorRT-LLM `32` | Searchable; fixed by default | Tokens per block. vLLM and TensorRT-LLM require at least 2. |
| `prefix_caching` | `true` | fixed | Keep completed prompt blocks for reuse by later requests. |
| `swa_prefix_tails` | `auto` | fixed | SGLang only. Sequence tails per attention-DP rank that keep their DeepSeek-V4 sliding-window KV; `null` turns it off. `auto` uses four per running request for a DeepSeek-V4 checkpoint and is off for other models. See [DeepSeek-V4 SWA prefix reuse](#deepseek-v4-swa-prefix-reuse). |
| `decode_radix_cache` | `false` | fixed | SGLang `decode` only. Enables the decode worker's radix cache (`--disaggregation-decode-enable-radix-cache`); requires `prefix_caching: true` and no speculative decoding. See [SGLang decode radix cache](#sglang-decode-radix-cache). |
| `bytes_per_token` | `auto` | fixed | KV bytes per token on one TP shard. `auto` derives it from the model, KV precision and this role's parallelism. |
| `capacity.type` | `default` | fixed | `default` sizes G1 from GPU memory; `fixed` uses `blocks` or `bytes`. |
| `capacity.memory_fraction` | vLLM/TensorRT-LLM `0.9`, SGLang `0.88` | Searchable; fixed by default | In `(0, 1]`. `default` capacity only. Same meaning as vLLM `gpu_memory_utilization`, SGLang `mem_fraction_static`, TensorRT-LLM `free_gpu_memory_fraction`. |
| `capacity.cuda_graph_reserved_bytes` | `0` | predict only | Bytes held back from G1 for CUDA graphs. `default` capacity only. |
| `capacity.blocks` | Unset | fixed | G1 blocks per attention-DP rank. `fixed` capacity only. |
| `capacity.bytes` | Unset | predict only | G1 bytes per attention-DP rank, instead of `blocks`. Requires explicit `block_size` and numeric `bytes_per_token`, unless `state_cache: {}` infers them. |
| `host_offload` | Unset (off) | fixed | G2 block; see [host offload](#host-offload-g2). |
| `hicache` | Unset (off) | Rejected | SGLang host KV cache for aggregated prediction; see [SGLang HiCache](#sglang-hicache). |
| `g3_offload` | Unset (off) | Rejected | G3 block; see [G3 offload](#g3-offload). |
| `kvcr` | Unset (off) | Rejected | KVCR block; see [KVCR peer fetch](#kvcr-peer-fetch). |
| `state_cache` | Unset (off) | Rejected | Recurrent-state block; see [state cache](#state-cache). |
| `prefix_match_unit` | Unset | Rejected | Prefix-matching granularity with `state_cache`. |

The block-size defaults above are simulation defaults. When a Sweeper candidate's
role block size is absent or `null`, the deployment generator omits `--block-size`
(vLLM), `--page-size` (SGLang), or `kv_cache_config.tokens_per_block`
(TensorRT-LLM), allowing the engine to choose its runtime default. Explicit values
are preserved. For SGLang and TensorRT-LLM, generating a fixed `capacity.blocks`
requires an explicit `block_size` matching the runtime: the generator must
convert the block count to a token capacity and cannot infer runtime geometry
from a simulation default. vLLM accepts the block count directly. Set
`block_size` explicitly when simulation and deployment must use the same value.

## GPU cache (G1)

Each attention-DP rank has its own G1 cache and its own running requests.
Tensor parallelism splits each token's KV across GPUs. `bytes_per_token` and
`capacity` are therefore per TP shard and per DP rank.

With `capacity.type: default`, the number of G1 blocks comes from the
performance model's memory estimate, which applies `memory_fraction` the way
each backend does. The backends differ in what they subtract before and after
the fraction; see [memory accounting](../../perf-model/memory.md#kv-cache-capacity-reservation). Use `fixed` capacity to pin an
exact block count, for example to compare cache sizes or to match a measured
deployment.

With `prefix_caching: true`, a new request reuses complete blocks whose tokens
match the start of its prompt and are still resident on the rank it is placed
on. Partial blocks are never reused. When G1 is full, each backend applies its
own policy: requests wait or are preempted, and unused cached blocks are
evicted least recently used first.

For SGLang, complete prompt pages enter the radix cache after prefill; decoded
pages enter when a request finishes or a running request is cancelled. In overlap
mode, prefill reuse becomes visible when the forward result is observed. A
completed disaggregated source publishes its prompt before the transfer hold;
the hold retains its path lock and partial tail. A transferred destination keeps
its prompt private until running admission, or caches its full pages when a
ready request is cancelled.

SGLang KV events follow those radix insertions, rather than page allocation.
Compatible consecutive stores and removals are batched, with stores retaining
their parent links and session IDs. Events flush after prefill, every 40 decode
forwards and on idle. Under memory pressure, decode retraction stably orders
requests by descending output length and then ascending prompt length, retracts
from the end, and evicts unused cache toward a 20-token reserve per survivor.
These contracts follow [SGLang v0.5.20](https://github.com/sgl-project/sglang/tree/94602c9c2b7cbdb8efd5c52802dac6a1c180089e).

### SGLang decode radix cache

An SGLang decode server runs a chunk cache unless it is launched with
`--disaggregation-decode-enable-radix-cache`, which is off by default, so an
SGLang `decode` worker caches nothing unless `decode_radix_cache: true` is set.
Without it, transferred prompts and decoded output stay private to their
request, the worker publishes no KV events, and every transfer moves the whole
prompt. With it, the worker reuses cached prefixes and publishes a transferred
prompt at running admission, as described above. Like SGLang, the simulator
rejects the decode radix cache with speculative decoding. An SGLang decode
worker always serves its waiting queue first-come first-served. This follows
[SGLang v0.5.22](https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d)
(`arg_groups/pd_disaggregation_hook.py`, `disaggregation/decode.py`).

For raw Sweeper `SearchSpace` inputs, `decode_radix_cache: true` requires
`decode_enable_prefix_caching: true` and no speculation when the domain includes
SGLang disaggregated candidates. Validation rejects invalid combinations before
enumeration; deployment lowering also checks direct sample inputs. Analytical
AFD and AFD+P/D do not model this cache and reject the setting. A sweep may include other backends,
whose samples omit the SGLang-only radix flag. The recommendation path carries
both controls from the worker's `kv_cache` configuration. A search with no
SGLang backend rejects `decode_radix_cache: true`.

To see what was reused, run with `--capture-per-request` and read these fields
in `requests.jsonl`:

| Field | Meaning |
| --- | --- |
| `first_admission_reused_input_tokens` | Prompt tokens reused from all tiers when the request was first admitted; later readmissions do not change it. |
| `reused_input_tokens` | Largest reuse across the request's prefill admissions, or across all admissions without a prefill stage. |
| `first_admission_g1_reused_input_tokens` | Prompt tokens served from G1 when the request was first admitted. |
| `first_admission_host_reused_input_tokens` | Prompt tokens restored from G2, including G3 restores and KVCR pulls. |
| `admission_history` | Every admission of the request, including after preemption, with the pool it reused from. |

The Dynamo KV Router's overlap score is a routing estimate; these fields record
what Replay actually reused. A readmission after preemption can reuse KV the
request itself computed, which raises `reused_input_tokens` but not
`first_admission_reused_input_tokens`; the summary's
`first_admission_prefix_cache_reused_ratio` aggregates the latter.

### SGLang request cache salt

Native `Request`, authored Rust `ReplaySpec.requests`, and configured inline
Python requests accept this per-request field:

| Field | Default | Rules |
| --- | --- | --- |
| `cache_salt` | Absent | Optional string. Empty is equivalent to absent. Nonempty salts require the native SGLang backend. |

Requests reuse a prefix only within the same salt namespace, even when their
token IDs match. The namespace also applies to LPM scores, waiting-queue
deduplication, P/D destination matching, and Belady input-demand forecasts.
Session IDs remain attribution and do not separate cache contents.
Built-in trace formats keep their existing unsalted schema; custom deferred
replay requests carry salt in their `DirectRequest` metadata.

Stored KV events carry `cache_salt`; removals reference the same salted hashes.
AISim retains its XXH3 block/sequence representation, with unchanged identities
for unsalted requests. It models the namespace behavior of
[SGLang v0.5.22](https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d),
without reproducing SGLang's SHA256 event encoding. Placement adapters hashing
a salted prompt should use `ReplayRequestHashes::from_tokens_with_cache_salt`.

This is a simulator contract pinned to SGLang v0.5.22. AISimulate does not check
the installed serving runtime's support for `cache_salt` or the decode radix
flag; verify both capabilities when deploying an older SGLang version.

<a id="deepseek-v4-swa-prefix-reuse"></a>

## DeepSeek-V4 SWA prefix reuse

DeepSeek-V4 layers keep a sliding window of KV next to their compressed KV.
SGLang frees a running request's window slots as the request moves past them,
so a cached sequence holds SWA KV only at its tail. A prefix hit must end where
a whole window is cached (`create_match_validator` in SGLang v0.5.21
`mem_cache/unified_cache/components/swa.py`). A request that forks inside
another cached sequence, such as a subagent's first turn, therefore recomputes
from the last cached window even though full KV covers more. Its recompute
leaves a window at the fork point, so the next request forking there reuses it
(`finalize_match_result_in_tree_core`).

`kv_cache.swa_prefix_tails` turns this on for SGLang `aggregated` and `prefill`
workers. Each finished sequence's tail and each recomputed fork point holds a
window. Replay's generated output tokens differ from the next prompt in a trace,
so a continuation diverges where the previous output began instead of matching
through it to the tail. Replay therefore keeps a finished request's window at
the end of its prompt. At most `swa_prefix_tails` windows stay cached per attention-DP rank,
least recently used first; evicting a sequence's tail also drops its window.
With HiCache, a window stays valid while its node is host-resident.

`auto` sizes the windows as SGLang's `cap` mode does: four per running request
(`max_sequences`). It applies to a checkpoint whose bundled or local config has
architecture `DeepseekV4ForCausalLM` or `DeepseekV41ForCausalLM`. Set a number
to match a server that sized its SWA pool another way, or `null` to match full
KV at any page. These are modeling assumptions: a window is tracked per radix
node rather than per SWA page, and its pool shares no capacity with full KV.
A decode worker's radix cache matches full KV as before.

<a id="host-offload-g2"></a>
<a id="native-vllm-host-offload-prediction"></a>

## SGLang HiCache

Set `engine.workers.aggregated.kv_cache.hicache` to model SGLang v0.5.22
caching on a private host pool for each attention-DP rank.
Prefix caching must be enabled. This is supported for aggregated prediction
and native offline replay; disaggregated roles, recommendation, Belady eviction,
external storage, and peer KVCR fetch are not supported by this option.

| Field under `hicache` | Default | Meaning |
|---|---|---|
| `num_host_blocks` | Required | Positive host capacity per attention-DP rank. |
| `write_policy` | `write_through` | `write_through`, `write_through_selective`, or `write_back`; see below. |
| `d2h_bandwidth_gbps` | `32` | Device-to-host bandwidth in decimal GB/s; zero is unlimited. |
| `h2d_bandwidth_gbps` | `32` | Host-to-device bandwidth in decimal GB/s; zero is unlimited. |
| `latency_to_first_byte_ms` | `0` | Non-negative latency per transfer operation. |

Host pages use the GPU page geometry: `block_size * bytes_per_token` bytes.
The compiler resolves `bytes_per_token: auto` from the model and rank layout.
Native engine configurations place these fields under `sglang.hicache` and
require `kv_cache_bytes_per_token`.

With `write_through`, completed chunk and request inserts back up their canonical
radix pages, including materialized output pages. `write_through_selective` waits
for a node's second insertion observation; each request counts newly covered
prefixes once, including reused pages. These policies back up ancestors first.
Each step submits one merged D2H operation;
its acknowledgement publishes host residency and releases the independent GPU
page locks. GPU eviction preserves backed-up radix nodes. Host eviction removes
only unlocked, host-only leaves in last-access order and may reclaim old entries
even when the new node ultimately cannot fit.

`write_back` instead backs up the selected GPU eviction leaf and waits for its
D2H acknowledgement before releasing the device pages. This wait delays batch
selection and forward execution. Host pressure first reclaims redundant copies
that remain on the GPU, sparing evictable GPU leaves on the first pass, then
reclaims host-only leaves. In-flight transfers and host source locks protect
their copies. If a backup cannot fit, eviction drops the selected device leaf
and its eligible host-only descendants together.

Admission selects the prefill shape before queuing H2D and budgets both the host
prefix and new GPU work. Load-back restores the entire promised host prefix or
recomputes admission without it; host hits below ten tokens are recomputed.
Ordinary forward duration includes the maximum of compute and dependent H2D,
without a per-layer pipeline model. With `host_loop`, this duration belongs to
the launched forward; its results are observed by a later loop iteration.
Admission reports device and host reuse separately. Host transfers continue to
drain and emit cache events after the last request completes or is cancelled.

These semantics follow public SGLang v0.5.22 at
[`baa9befa`](https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d).
The separate `host_offload` option below models vLLM connectors.

## Host offload (G2)

G2 keeps copies of KV blocks in host memory. When a prefix has been evicted
from G1, a later request can restore it over PCIe instead of recomputing it.
It follows vLLM's native CPU offload.

```yaml
# host-offload-prediction.yaml
engine:
  mode: aggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  backend_version: "0.24.0"
  context_length: 8192
  workers:
    aggregated:
      scheduler: {max_batched_tokens: 8192, max_sequences: 16}
      kv_cache:
        block_size: 16
        capacity: {type: fixed, blocks: 2499}
        host_offload:
          num_host_blocks: 4096
          d2h_bandwidth_gbps: 32.0
          h2d_bandwidth_gbps: 32.0

traffic:
  source: {type: synthetic-session, session: {turns: 4}}
  load: {type: concurrency, concurrency: 4}
  stop: {sessions: 16}
```

```bash
aisimulate predict --stack engine --config host-offload-prediction.yaml --capture-per-request
```

| Knob (under `host_offload`) | Default | Rules |
|---|---|---|
| `num_host_blocks` | Required | Positive. Per DP rank with `dp_rank_local`; total pool size with `cluster_shared`. |
| `scope` | `dp_rank_local` | `dp_rank_local` or `cluster_shared`; see [scope](#g2-scope). |
| `d2h_bandwidth_gbps` | `32.0` | GPU→host bandwidth per DP rank, decimal GB/s. `0` means unlimited. |
| `h2d_bandwidth_gbps` | `32.0` | Host→GPU bandwidth per DP rank, decimal GB/s. `0` means unlimited. |
| `latency_to_first_byte_ms` | `0.0` | Fixed delay before each transfer starts moving bytes. |
| `shared_d2h_bandwidth_gbps` | `80.0` | Total GPU→host bandwidth of the pool. `cluster_shared` only. |
| `shared_h2d_bandwidth_gbps` | `80.0` | Total host→GPU bandwidth of the pool. `cluster_shared` only. |
| `connector` | `offloading_v0_24` | vLLM connector whose rules G2 follows: `offloading_v0_24`, `offloading_v0_32` or `simple_cpu_v0_32`. The last two need the aggregated worker, `dp_rank_local` and no `g3_offload` or `kvcr`; see [connector profiles](#g2-connector-profiles). |

`recommend` keeps `host_offload` fixed; its size and bandwidth are not searched.

### How G2 behaves

- A block's size is `block_size × bytes_per_token`, the same per-TP-shard bytes
  as G1. No TP multiplier is applied to host capacity or transfer time, so an
  explicit `bytes_per_token` must use the same unit as the bandwidths.
- Blocks are copied to G2 after they are computed in G1. A request that finds its
  prefix in G2 waits for the host→GPU copy before it is admitted, so restore
  time adds to TTFT.
- G1 blocks that are still being copied to G2 are not overwritten.
- A decode rank with G2 admits requests whose P/D handoff has arrived before
  other waiting requests, as vLLM's FCFS scheduler does.

<a id="g2-connector-profiles"></a>

### Connector profiles

`connector` picks the vLLM CPU-offload connector whose store, lookup and
eviction rules G2 follows:

| | `offloading_v0_24` (default) | `offloading_v0_32` | `simple_cpu_v0_32` |
| --- | --- | --- | --- |
| vLLM connector | `OffloadingConnector`, vLLM 0.24 | `OffloadingConnector` with the CPU backend, vLLM v0.32.0 | `SimpleCPUOffloadConnector` in eager mode, vLLM v0.32.0 |
| Stored blocks | Prompt blocks as they are computed, after any blocks loaded from G2 | The same; a finishing request also offers prompt blocks it could not store earlier | Every block one pass after it is computed, decode blocks included, scanned from block 0 at each admission; blocks already in G2, such as those just loaded, are skipped. A finishing request stores the rest. |
| Eviction order | Least recently used. Each lookup and store makes the prompt's resident blocks the newest, its first block newest of all. | Least recently used by rank. A block is ranked when its store starts; a finishing request re-ranks every block it stored, offered or loaded, its first block newest. | First in, first out by store completion. A completed load or a dropped lookup pin makes its blocks the newest; a block already in G2 is not refreshed. |
| A hit whose store is in flight | Defers the request | Defers the request | Misses |
| A hit another request is loading | Defers the request | Defers the request | Both requests load it |
| The request's own store in flight | Defers its lookup | Defers its lookup | Does not defer |
| A hit before its load | Not pinned | Not pinned | Pinned until the request loads it or looks it up again |
| G1 blocks being copied | Freed with their request; a new owner waits for the copy | The same | Held until the copy completes, then freed first block first |
| A preemption | Waits for the preempted request's stores | The same | Waits for every transfer in flight |

vLLM main has changed some of these rules since v0.32.0. Each profile models the
rules of the version it names; none was compared with a running vLLM. The model
also differs from vLLM v0.32.0 here:

- `offloading_v0_32` offers and re-ranks a finishing request's blocks when the
  request finishes. vLLM does both in its next step, after that step's other
  stores.
- After a preemption, `simple_cpu_v0_32` loads only prompt blocks from G2 and
  recomputes the request's output blocks, which vLLM reloads.

<a id="g2-scope"></a>
<a id="g2-ownership"></a>

### Scope

| `scope` | Pools | `num_host_blocks` means | Bandwidth |
| --- | --- | --- | --- |
| `dp_rank_local` | One per DP rank, as in vLLM | Per rank | Each rank's own D2H and H2D limits |
| `cluster_shared` | One for the whole deployment | Total for the pool | Rank limits, also capped by the pool's shared limits |

For example, two replicas with `attention_data: 2` and `num_host_blocks: 4096`
get four private 4096-block caches with `dp_rank_local`, or one 4096-block
cache with `cluster_shared`. In a shared pool, transfers in the same direction
share bandwidth max-min fairly per transfer, not per rank: four ranks at
32 GB/s, each running one transfer under an 80 GB/s pool, each move at 20 GB/s.
Pool bandwidth a rank cannot use because its own limit binds goes to the other
ranks: with one rank running nine transfers and another running one, both at
32 GB/s, each rank moves 32 GB/s in total. Under a 20 GB/s pool the same ten
transfers each get 2 GB/s, so the busier rank moves 18 GB/s.

`cluster_shared` models a deployment-wide host cache such as Mooncake Store or
LMCache. vLLM's native offload has no equivalent, so it has no native parity
reference. All roles in one shared pool must use the same model, backend,
parallel shape, KV precision and block geometry, plus the same pool capacity
and shared bandwidths; otherwise the run is rejected. In `recommend` with
`cluster_shared` on both P/D roles, pin each role's `tensor` and `pipeline` with
`parallelism.preset: false`.

### Supported

G2 supports vLLM aggregated and token-only disaggregated workers with prefix
caching and any `attention_data`. It is rejected with SGLang, TensorRT-LLM,
native MTP, ngram speculation, `state_cache`, and Belady eviction.

For shared pools, the report's `g2_domains` gives capacity and resident blocks.
Workers that publish KV events report G2 changes with `"tier": "host_pinned"`;
see [KV events](../../adapters/native-composition.md#kv-events).

<a id="g3-offload"></a>
<a id="optional-g3-offload"></a>

## G3 offload

G3 adds a storage tier behind G2. It requires `host_offload`. Add it next to
`host_offload` in the example above:

```yaml
g3_offload:
  scope: cluster_shared
  num_g3_blocks: 8192
```

| Knob (under `g3_offload`) | Default | Rules |
|---|---|---|
| `scope` | Required | `worker_local` (one store per worker) or `cluster_shared` (one store for all workers). |
| `num_g3_blocks` | Required | Positive. Per worker with `worker_local`; total with `cluster_shared`. |
| `latency_to_first_byte_ms` | `0.1` | Fixed delay before each transfer moves bytes. |
| `read_bandwidth_gbps`, `write_bandwidth_gbps` | `10.0` | Per worker, decimal GB/s. `0` means unlimited. |
| `shared_read_bandwidth_gbps`, `shared_write_bandwidth_gbps` | `80.0` | Total for the store. `cluster_shared` only. |

These defaults are modeling assumptions, not measured values.

- Blocks written to G2 are also written to G3 in the background.
- A read restores a prefix from G3 into G2 and then into G1. G3 alone does not
  make blocks ready on the GPU.
- With `cluster_shared`, a block is stored once and any worker can read it,
  including blocks written by a worker that has since been removed. With
  `worker_local`, workers cannot read each other's blocks.
- G3 evicts least recently used blocks that are not being transferred.
- With `worker_local`, total bandwidth grows with the worker count (16 workers
  at 10 GB/s reach 160 GB/s). `cluster_shared` caps it at the shared limits and
  divides them max-min fairly, as in a [shared G2 pool](#g2-scope).

To compare scopes at equal total capacity with `N` workers, use
`num_g3_blocks: C` for `worker_local` and `N × C` for `cluster_shared`. Set the
shared bandwidths to `0` to isolate reuse from bandwidth contention.

The prediction summary includes a `g3_offload` section with lookup hits,
read/write jobs and bytes, evictions, resident blocks and
`cross_worker_read_blocks` (blocks read by a worker other than the writer).

Supported: aggregated vLLM with `attention_data: 1`, prefix caching, and either
G2 scope; fixed or Planner-scaled workers. Rejected: `recommend`, P/D,
native MTP, and a `host_offload.connector` other than `offloading_v0_24`.

When a request's prefix is larger than G2 can hold, restores can keep evicting
each other. If the rank makes no progress for 1,024 consecutive restore rounds,
the request recomputes the missing part instead. Each time this happens
`g3_offload.bypassed_restores` increments. vLLM has no such rule.

<a id="kvcr-peer-fetch"></a>

## KVCR peer fetch

KVCR (KV Cache Runner) lets a rank pull a prompt prefix it is missing from
another rank's G2 over the network, instead of recomputing it. A router attaches
a hint naming the source rank; without hints KVCR does nothing. It requires
private G2 (`host_offload` with the default `scope: dp_rank_local`). Add it next
to `host_offload` in the [host offload example](#host-offload-g2):

```yaml
kvcr:
  latency_to_first_byte_ms: 1.0
  bandwidth_gbps: 50.0
```

| Knob (under `kvcr`) | Default | Rules |
|---|---|---|
| `latency_to_first_byte_ms` | `1.0` | Fixed delay before each pull moves bytes, for the control round trip and the source pin. |
| `bandwidth_gbps` | `50.0` | Per source rank (one attention-DP rank of a worker), decimal GB/s (a 400 Gb/s NIC is `50`). All GPUs of the rank share it, and so do concurrent pulls from it, equally. `0` means unlimited. |

`kvcr: {}` uses the defaults. These defaults are modeling assumptions, not
measured values.

### How KVCR behaves

- When some of a hinted request's prompt blocks are in neither G1 nor the
  rank's own G2, the rank checks the source's G2 block by block, in prompt
  order, and pulls each missing block the source holds. It stops at the first
  block the source lacks.
- A pull starts when the source finishes its current forward pass, including
  a pass that starts at the same instant the pull is requested, waits
  `latency_to_first_byte_ms`, then moves bytes. Each pulled block carries the
  KV of every GPU of the source rank: G2 block bytes times the role's
  `parallelism.tensor` (native `tensor_parallel_size`). The ranks of one worker run
  their passes in lockstep, so a pull from another DP rank of the same worker
  starts when the worker's current pass ends. Pulled blocks land in the rank's
  own G2 and load to G1 like any G2 hit, so pull and load time add to TTFT.
- Requests on one rank that need the same blocks share one pull. A request that
  ends while its pull is in flight leaves the pulled blocks in G2.

The model simplifies KVCR:

- The rank sees the source's G2 contents when it looks up the prefix. The
  source's blocks are then protected from eviction until the rank takes the
  landed pull: as soon as it lands if the rank's worker is idle, otherwise when
  that worker's current pass ends. The pulled blocks become usable in the
  rank's G2 at the same point.
- The rank reserves G2 slots only for the blocks the source holds, all in one
  reservation per pull. A pull never evicts from the rank's G2 a block of the
  hinted prompt that the rank's G1 does not hold; it stops at the first block
  it has no other room for. G2 copies of prompt blocks already in G1 stay
  evictable, since the request does not load them.
- While its pull is in flight, a request is not looked up again, as vLLM skips
  a request that waits for remote KV.
- Pulls do not fail or time out, and the first pull between two ranks has no
  connection setup cost.
- A pull moves one block share per tensor-parallel GPU. KV that each GPU
  replicates (an MLA latent, or a KV head shared by several GPUs when
  `num_kv_heads` is smaller than `tensor`) is counted once per GPU.

When a request's prefix is larger than G2 can hold, pulls can keep evicting
each other. If the rank makes no progress for 1,024 consecutive pull rounds,
the request recomputes the missing part instead. Each time this happens
`kvcr.bypassed` increments. vLLM has no such rule.

The prediction summary includes a `kvcr` section: `pulls` by result (`success`,
or `partial` when the source held only part of the missing blocks or the rank
had no free G2 slot for the rest, including a block the same request already
pulled at that instant), pulled `blocks` and `bytes` (summed over
the GPUs of each source rank),
`remote_deliver_ms` (total time from submit to landing in G2), `source_wait_ms`
(total time spent waiting for the source's pass to end) and `loaded_blocks`
(pulled blocks, counted when their first load into G1 starts). `bypassed`
appears only when it is nonzero. The counters update as the pool's ranks run.
A run stopped by `traffic.stop.max_virtual_time_seconds` (native
`max_sim_time_ms`) can miss pulls that landed during a pass still running at
the cutoff; a run that completes counts every pull.

Ranks without incoming pulls skip shared KVCR deadline lookups. Other KVCR
bookkeeping still queries shared pool state from per-rank calls. Large worker
or attention-DP pools can substantially increase replay wall time. Further
overhead reductions are deferred; they must preserve CanonicalV1 replay output bytes.

### Supported

KVCR supports vLLM aggregated workers with any `attention_data`, and pulls
between the prefill ranks of P/D. Workers may be fixed or Planner-scaled; a
draining worker is removed only after every pull from or into it has landed and
been taken by its target. Rejected: missing or `cluster_shared` `host_offload`,
a `host_offload.connector` other than `offloading_v0_24`, `g3_offload` together with `kvcr`, `kvcr` on the decode role, `parallelism.pipeline` above 1,
`parallelism.prefill_context` above 1 (native `prefill_context_parallel_size`), and `recommend`.
Everything G2 rejects is rejected too, including SGLang and TensorRT-LLM.

Hints come from the placement policy (see
[native composition](../../adapters/native-composition.md#placementpolicy)) or
from `kv_hints` on native ReplaySpec requests (see
[Rust API](../api/rust.md#serializable-spec)). Replay ignores a hint that names
the receiving rank. CLI traffic carries no hints, so
`aisimulate predict --stack engine` runs KVCR idle: apart from the `kvcr`
section, the results equal G2 alone.

For a complete CLI example with router-generated hints, follow
[Simulate KVCR peer reuse](../dynamo.md#simulate-kvcr-peer-reuse). It uses
`--stack dynamo`, two workers with private G2, and a Mooncake trace, and compares
KVCR-enabled replay against a G2-only baseline.

<a id="state-cache"></a>
<a id="manual-state-cache-sizing"></a>

## State cache

Hybrid-KV models keep a fixed-size state per request in addition to per-token
KV: the recurrent state of Kimi K3's KDA layers, or the sliding-window KV and
compressor states of DeepSeek V4. `state_cache` reserves that state in the same
G1 pool. The fragment below goes under
`engine.workers.aggregated`, or under both `prefill` and `decode` for P/D:

```yaml
kv_cache:
  block_size: 64
  bytes_per_token: 16
  capacity: {type: fixed, bytes: 8192}
  state_cache: {bytes_per_request: 1500}
```

This gives eight 1024-byte blocks. Each request's state takes two blocks
(1500 bytes rounded up) in addition to its token KV.

| Knob | Default | Rules |
|---|---|---|
| `state_cache.bytes_per_request` | Inferred | State bytes per request per rank. When set, `block_size` and numeric `bytes_per_token` must also be set. |
| `state_cache.mamba_cache_dtype` | `auto` | Kimi K3 only. `auto` (model dtype), `float16` or `float32`. Used when bytes are inferred. |
| `state_cache.indexer_cache_dtype` | `auto` | DeepSeek V4 only. `auto` (vLLM's FP8), `fp8` or `mxfp4`; vLLM supports `mxfp4` only on Blackwell datacenter GPUs. Used when bytes are inferred. |
| `prefix_match_unit` | Unset (`block_size`) | Positive divisor of `block_size`. Lets prefixes match at a finer granularity than the physical block. |

`state_cache: {}` infers Kimi K3 geometry from the model. `block_size` is then
a requested page size (default 64) that is enlarged to fit one state per layer.
For example, TP8 with FP8 KV and `block_size: 128` resolves to 1536-token blocks
of 13824 bytes per token. The same estimate is available in the SDK:

```python
from aisimulate_core.sdk import estimate_state_cache

state = estimate_state_cache("moonshotai/Kimi-K3", tp_size=8, kvcache_quant_mode="fp8", block_size=128)
```

For DeepSeek V4, `state_cache: {}` follows vLLM's allocation. vLLM stores
`fp8_ds_mla` KV and gives every cache group the same pool row, as wide as the
widest group. A request holds one row per 256 tokens of compressed KV and
indexer keys (`block_size` 256, `bytes_per_token` one row per 256 tokens), plus
a 26-row state for its sliding-window groups (SWA KV and compressor states). For
DeepSeek-V4-Flash, 1,002,240-byte rows give 3915 bytes per token and a
26,058,240-byte state on every TP rank.
Leave `block_size` and `bytes_per_token` unset. Speculative decoding (`nextn`
or `speculation`) widens each sliding window by the draft tokens, so one draft
token gives 30 rows. These are modeling assumptions: 26 rows is vLLM's decode
peak (it dips to 24 between window blocks), vLLM also holds sliding-window slots
for every token of a prefill chunk until the next step (3,360 rows for an
8,192-token chunk), and a prefix hit restores 22 of the 26 rows. None of these
is modeled.

Prefix reuse resumes from a stored state snapshot, so it does not land on every
matching token. With `block_size: 1536` and `prefix_match_unit: 128`, a cold
24,300-token prompt keeps snapshots at 23,040 and 24,192 tokens. A later request
sharing 23,700 tokens resumes at 23,040. This follows vLLM v0.29.0 align mode.

In `mode: disaggregated`, set `state_cache` on both `prefill` and `decode`;
each role sizes its own state. The decode worker reserves the prompt blocks
plus one state when it accepts a request, then continues from the transferred
state. With [`engine.kv_transfer`](kv-transfer.md), the transfer also moves
one state: the prefill role's `bytes_per_request`, charged once per request in
either `timing_mode`. A decode-side prefix hit skips resident token KV; the
state still travels with the rest of the prompt. These are modeling
assumptions: vLLM sends the raw convolution and recurrent state (about 8% less
than the padded `bytes_per_request` for Kimi K3 at TP8), transfers it after all
but the last prompt token and recomputes that token on the decode worker. Replay
transfers the state for the whole prompt and does not model the recompute.

Supported: aggregated and P/D vLLM, fixed `capacity`, `predict --stack engine`;
inferred geometry for Kimi K3 and DeepSeek V4 at PP=1.
Speculative decoding supports `prefix_match_unit` equal to `block_size`.
Sub-block matching with speculation remains unsupported.

Rejected: `recommend`, G2/G3 offload, and speculative decoding with sub-block
`prefix_match_unit`.

## Other cache modes

- **Grouped caches.** Some whole-forward (FPM) profiles describe several KV
  groups, such as sliding-window and full attention, sharing one byte budget.
  Replay then allocates each group separately. Grouped caches require cold
  aggregated vLLM with `prefix_caching: false`, no offload, no P/D and default
  capacity. See [memory accounting](../../perf-model/memory.md).
- **Belady eviction.** The native ReplaySpec field
  `engine.kv_eviction_policy: belady` evicts the block whose next use in the
  trace is farthest away. It is not available in `predict` YAML. It requires an
  open-loop trace on fixed aggregated workers with `attention_data: 1` and no
  offload. Use it to bound how much a better eviction policy could help.

<a id="agentic-g2"></a>

## Agentic traffic with G2

AgentX snapshot, warmup and continuous-profile traffic supports G2 on vLLM with
one aggregated worker, one prefill and one decode worker, or several aggregated
workers that each own a `dp_rank_local` G2. All need `attention_data: 1`, static
workers and no speculative decoding. Either G2 scope works with one worker per
role; G3 is not supported. `kvcr` is accepted but never pulls, because agentic
traffic carries no router hint. Results are qualified `functional_only`.

Two small offline fixtures use fixed timing and need no GPU:

```bash
aisimulate predict --stack engine --config examples/cli/agentx-g2-local.yaml \
  --capture-per-request --output-dir ./agentx-g2-local --format json
aisimulate predict --stack engine --config examples/cli/agentx-g2-shared-pd.yaml \
  --capture-per-request --output-dir ./agentx-g2-shared --format json
```

In both, a child request evicts its parent's prefix from a three-block GPU
cache, and the parent later restores it from host memory. Change
`host_offload.scope`, `h2d_bandwidth_gbps` or `num_host_blocks` to compare.
