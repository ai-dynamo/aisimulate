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
```

## KV cache fields

All paths below are under `engine.workers.<role>.kv_cache`.

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `block_size` | vLLM `64`, SGLang `1`, TensorRT-LLM `32` | Searchable; fixed by default | Tokens per block. vLLM and TensorRT-LLM require at least 2. |
| `prefix_caching` | `true` | fixed | Keep completed prompt blocks for reuse by later requests. |
| `bytes_per_token` | `auto` | fixed | KV bytes per token on one TP shard. `auto` derives it from the model, KV precision and this role's parallelism. |
| `capacity.type` | `default` | fixed | `default` sizes G1 from GPU memory; `fixed` uses `blocks` or `bytes`. |
| `capacity.memory_fraction` | vLLM/TensorRT-LLM `0.9`, SGLang `0.88` | Searchable; fixed by default | In `(0, 1]`. `default` capacity only. Same meaning as vLLM `gpu_memory_utilization`, SGLang `mem_fraction_static`, TensorRT-LLM `free_gpu_memory_fraction`. |
| `capacity.cuda_graph_reserved_bytes` | `0` | predict only | Bytes held back from G1 for CUDA graphs. `default` capacity only. |
| `capacity.blocks` | Unset | fixed | G1 blocks per attention-DP rank. `fixed` capacity only. |
| `capacity.bytes` | Unset | predict only | G1 bytes per attention-DP rank, instead of `blocks`. Requires explicit `block_size` and numeric `bytes_per_token`, unless `state_cache: {}` infers them. |
| `host_offload` | Unset (off) | fixed | G2 block; see [host offload](#host-offload-g2). |
| `g3_offload` | Unset (off) | Rejected | G3 block; see [G3 offload](#g3-offload). |
| `state_cache` | Unset (off) | Rejected | Recurrent-state block; see [state cache](#state-cache). |
| `prefix_match_unit` | Unset | Rejected | Prefix-matching granularity with `state_cache`. |

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

To see what was reused, run with `--capture-per-request` and read these fields
in `requests.jsonl`:

| Field | Meaning |
| --- | --- |
| `first_admission_g1_reused_input_tokens` | Prompt tokens served from G1 when the request was first admitted. |
| `first_admission_host_reused_input_tokens` | Prompt tokens restored from G2 (and G3). |
| `admission_history` | Every admission of the request, including after preemption, with the pool it reused from. |

The Dynamo KV Router's overlap score is a routing estimate; these fields record
what Replay actually reused.

<a id="host-offload-g2"></a>
<a id="native-vllm-host-offload-prediction"></a>

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
share bandwidth equally: four ranks at 32 GB/s under an 80 GB/s pool each move
at 20 GB/s while all four transfer.

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
  at 10 GB/s reach 160 GB/s). `cluster_shared` caps it at the shared limits.

To compare scopes at equal total capacity with `N` workers, use
`num_g3_blocks: C` for `worker_local` and `N × C` for `cluster_shared`. Set the
shared bandwidths to `0` to isolate reuse from bandwidth contention.

The prediction summary includes a `g3_offload` section with lookup hits,
read/write jobs and bytes, evictions, resident blocks and
`cross_worker_read_blocks` (blocks read by a worker other than the writer).

Supported: aggregated vLLM with `attention_data: 1`, prefix caching, and either
G2 scope; fixed or Planner-scaled workers. Rejected: `recommend`, P/D, and
native MTP.

When a request's prefix is larger than G2 can hold, restores can keep evicting
each other. If the rank makes no progress for 1,024 consecutive restore rounds,
the request recomputes the missing part instead. Each time this happens
`g3_offload.bypassed_restores` increments. vLLM has no such rule.

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
Rejected: `recommend`, G2/G3 offload, and speculative decoding with
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
one aggregated worker, or one prefill and one decode worker, all with
`attention_data: 1`, static workers and no speculative decoding. Either G2 scope
works; G3 is not supported. Results are qualified `functional_only`.

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
