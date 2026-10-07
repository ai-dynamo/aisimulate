<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Replay cache

Replay owns cache lifetime and reuse under scheduling pressure. The
[performance model](../perf-model/memory.md) supplies physical capacity and byte
geometry; a capacity estimate alone does not establish reuse or offload support.

## GPU cache and prefix reuse

G1 is the engine's GPU cache. Each attention-DP rank owns its request state and
G1 capacity; TP shards the same requests and does not multiply usable sequence
capacity. Prefix reuse depends on complete cacheable blocks, token identity,
residency, backend admission rules, and placement. Router overlap is a placement
estimate, not evidence of an actual cache hit. Use per-request first-admission
reuse and admission histories to inspect what happened.

Requests can queue or preempt when allocation fails. Completion can leave
reusable prefix pages resident; client completion is distinct from freeing
in-flight handoff or transfer resources. Ordinary eviction uses native backend
policy, with LRU as the default forecast policy.

## Grouped caches

Grouped FPM profiles support cold aggregated vLLM, PP1/CP1, HBM-only cache,
non-speculative decoding and `prefix_caching: false`. Prefix reuse, host/G3
offload, P/D, and fixed scalar cache capacity are rejected. All groups allocate
atomically from one shared rank-local byte budget. A sliding window retains
blocks covering the last `W - 1` computed tokens plus the next scheduled
forward. Temporary prefill pages remain charged until completion; full-attention
groups retain full history. Physical retention never truncates logical progress
or FPM context coordinates.

Grouped capacity has no equivalent scalar token/block capacity. Rust scheduler
telemetry exposes optional used/capacity bytes and byte occupancy; `total_blocks`
is zero for heterogeneous pages. This observer data is not currently exposed by
the Python JSON replay runtime. See [memory accounting](../perf-model/memory.md).

## G2 host-cache configuration

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

## G2 ownership

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

## G2 evidence

The repository's [shared-G2 fixture](../../examples/cli/shared-g2-predict.yaml)
exercises local versus shared scope with fixed compute timing. Private-G2 GPU
comparisons and their model, backend, byte-unit, and metric caveats are retained
in the [replay evidence record](../../benchmarks/evidence/accuracy/replay-evidence.md).
Neither a shared-pool fixture nor AgentX acceptance establishes native shared-pool
performance parity.

## Unsupported

Non-vLLM backends, native MTP, Belady eviction, recurrent state-cache offload,
and detailed replay artifacts with `cluster_shared` are rejected. G3 still
requires aggregated replay with attention DP 1.

Agentic snapshot/warmup host offload is supported for one aggregated worker or
one prefill plus one decode worker, with vLLM, attention DP=1 on every role,
static worker pools, and speculative decoding disabled. Both `dp_rank_local`
and `cluster_shared` are supported; AgentX excludes G3. See [AgentX with vLLM
host offload](#agentic-g2) for shared-pool compatibility requirements and
functional qualification limits.

## Agentic G2

The supported G2 deployment is one aggregated worker or one prefill plus one
decode worker, with vLLM and attention DP=1 on every role. Tensor parallelism may
use multiple GPUs. Worker pools are static, speculative decoding is disabled,
and G3 is excluded. Existing HBM-only vLLM/SGLang support is unchanged.

Each role may omit host offload, use `dp_rank_local`, or use `cluster_shared`.
Local pools are independent. Shared roles join the same deployment-level pool:
`num_host_blocks` is its total capacity, not capacity per participant. Shared
capacity, shared directional bandwidths, and KV layout must agree; incompatible
participants are rejected. Layout includes model/backend, parallel shape, KV
dtype and block geometry. The runtime registry is recreated for each replay.
Per-role D2H/H2D bandwidth and first-byte latency may differ. Shared bandwidth
limits apply to concurrent transfers in addition to each role's link limit.

Dynamo owns its separate transport, Router and AgentX integration. Use the
consumer's dependency requirements and qualification; Engine acceptance does
not qualify a Dynamo installation. See [Dynamo integration](dynamo.md).

### Functional fixtures

From the repository root with a source build containing this feature:

```bash
aisimulate predict --stack engine --config examples/cli/agentx-g2-local.yaml \
  --capture-per-request --output-dir /tmp/agentx-g2-local --format json
aisimulate predict --stack engine --config examples/cli/agentx-g2-shared-pd.yaml \
  --capture-per-request --output-dir /tmp/agentx-g2-shared --format json
```

The small Weka fixture is authored for this repository. Its child request
evicts the parent's prefix from the three-block GPU cache. The parent resumes
after the child and restores eight tokens from host memory. Fixed pass timing
and explicit KV bytes make this an offline functional test; no weights or GPU
are required. These numbers are not measured model performance.

For a local/shared control, change only `host_offload.scope`. For a bandwidth
control, lower `h2d_bandwidth_gbps`; for a capacity control, lower
`num_host_blocks` to 1 (on both roles of a shared pool). To disable G2, remove
each role's `host_offload` mapping. No separate connector configuration is used.

Python callers compile the same configuration with
`prediction_to_replay_spec(CorePredictionConfig.from_yaml(path))`, then execute
it with `EngineReplayRunnerFactory().create(0).run(spec)`. Request
`ReplayOutputRequirements(include_raw_report=True, capture_per_request=True)`
to retain the native report. The native JSON form uses
`spec.engine.rank.native_host_offload` for aggregated workers and
`spec.engine.{prefill,decode}.rank.native_host_offload` for P/D; Python supplies
the shared `kv_layout_id` from the configured geometry. Direct native JSON
callers **must supply a nonempty `kv_layout_id` for `cluster_shared`**. Use the
same value only for genuinely compatible layouts; the registry also validates
physical block geometry, TP, capacity and shared bandwidths.

### Result interpretation

Inspect `first_admission_g1_reused_input_tokens`,
`first_admission_host_reused_input_tokens`, and `admission_history` in the
per-request report. History identifies the admission `pool`; P/D observations
are not added twice to request-first reuse. `g2_domains` reports shared capacity
once. P→D handoff and G1↔G2 transfers remain separate operations. A pending
store or restore cannot supply available cache early; H2D wait delays admission
and first token.

Warmup waits for necessary engine work and retains its G1/G2 state across the
measurement barrier. Tool gaps, dependency joins, lane recycling and play cache
identities retain their AgentX semantics. Duration/cancellation do not force a
background-transfer drain beyond the existing measurement contract.

Qualification remains `functional_only`. This implementation reuses the native
vLLM G2 model and does not establish performance equivalence with SemiAnalysis's
Mooncake recipe, SGLang offload or TRT-LLM offload.

## Best-effort Belady eviction

Set `engine.kv_eviction_policy` to `"belady"` in the native replay descriptor to
evict pages whose next input demand is farthest away. LRU remains the default;
there is no Cargo gate or new CLI wiring. Supported runs are open-loop, complete
input traces on fixed aggregated SGLang, vLLM, or TRT-LLM workers, with prefix
caching and one attention-DP rank per worker. Closed-loop/generated/agentic or
delta inputs, scaling, disaggregation, and host/G3 offload are rejected.

The forecast counts complete input prefix blocks in global trace order. It does
not predict which worker will use them or forecast output blocks; native output
caching still works when a later input matches. Demand retires at the request's
first committed prefill or terminal removal, never merely because time passes.
Later chunks and preemption retries do not recreate demand. These are intentional
assumptions: adding execution-dependent refinement would change the model.

The oracle supplies eviction rankings only. The engine preserves causal arrivals,
execution, and cache ownership. SGLang evicts unlocked leaf tails; vLLM/TRT evicts
inactive copies, preferring duplicates before the last useful copy. Multiworker
forecasts may retain data needed elsewhere, so optimal reuse is not guaranteed.
Reports label the assumption `global_input_trace_order_v1`; compare
`first_admission_prefix_cache_reused_ratio` and `committed_prefill_tokens` on
completed runs, alongside serving throughput and latency.


The scoped simulator comparisons are in the [replay evidence record](../../benchmarks/evidence/accuracy/replay-evidence.md); better reuse does not guarantee improved throughput or latency tails.

<a id="native-vllm-host-offload-prediction"></a>

## Native vLLM host-offload prediction

The public host-offload surface supports vLLM aggregated and token-only disaggregated workers with
prefix caching enabled, any attention-DP size, and no native speculative decoding. The descriptor is
fixed in both `predict` and `recommend`; host capacity and bandwidths are not search dimensions.
`bytes_per_token` belongs to `kv_cache`, not `host_offload`, and is resolved for the worker role
before lowering to the native rank; with `auto` it is one tensor-parallel shard's footprint. Each DP
rank has its own G2 cache by default; `scope: cluster_shared` models one pool for the deployment.
See [G2 host-cache scope](#g2-ownership) for ownership, bandwidth sharing and compatibility.

```yaml
# host-offload-prediction.yaml
engine:
  mode: aggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  backend_version: "0.24.0"
  context_length: 4096
  workers:
    aggregated:
      parallelism: {replicas: 1, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 16}
      kv_cache:
        block_size: 16
        prefix_caching: true
        bytes_per_token: auto
        capacity: {type: fixed, blocks: 2499}
        host_offload:
          num_host_blocks: 4096
          d2h_bandwidth_gbps: 32.0
          h2d_bandwidth_gbps: 32.0
      timing: {type: default}

traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 4}
  stop: {requests: 16}
```

Run it with:

```bash
aisimulate predict --stack engine --config host-offload-prediction.yaml
```

<a id="optional-g3-offload"></a>

### Optional G3 offload

G3 is an optional extension to native vLLM host offload, not a standalone cache
mode. It requires `host_offload` to be enabled. Add this mapping inside the
existing `engine.workers.aggregated.kv_cache`, as a sibling of `host_offload`,
and use the same `predict --stack engine` command above:

```yaml
g3_offload:
  scope: cluster_shared
  num_g3_blocks: 8192
```

Only `scope` and `num_g3_blocks` are required. `num_g3_blocks` is a positive integer,
not a nested `capacity` object. Optional controls and defaults are:

- `latency_to_first_byte_ms`: 0.1 ms per transfer.
- `read_bandwidth_gbps` and `write_bandwidth_gbps`: 10 GB/s each per worker.
- `shared_read_bandwidth_gbps` and `shared_write_bandwidth_gbps`: 80 GB/s each
  across the deployment, applied only in `cluster_shared` scope.

These are modeling defaults, not measured or GPU-calibrated values.
Latency is in milliseconds; bandwidth is in decimal
GB/s. Latency and bandwidth must be finite and non-negative. Zero bandwidth
means unlimited, not disabled. Block bytes are `block_size * bytes_per_token`;
`bytes_per_token: auto` uses the existing model/parallelism estimate. G3 stores
complete prefix-block identities, not real tensors or files. In native
ReplaySpec JSON, `g3_offload` and `native_host_offload` are sibling rank fields.

The ownership and bandwidth unit is a replica worker, not a physical host or
an individual TP rank. Set the initial worker count
with `engine.workers.aggregated.parallelism.replicas`:

- `worker_local`: each worker gets `num_g3_blocks` of independent capacity,
  like G2's `num_host_blocks`. Workers cannot reuse each other's stored blocks.
- `cluster_shared`: workers share one pool of `num_g3_blocks`. Duplicate prefix
  blocks occupy capacity once, regardless of how many workers use them.

Independent runs never share cached blocks. With N fixed workers, equal total
capacity means local `num_g3_blocks: C` versus shared `num_g3_blocks: N*C`.
Keep workload, G1/G2 settings, latency, and bandwidth identical for that comparison.
To isolate cache sharing from backend contention, make both shared bandwidth
caps non-binding (for example, set them to zero for unlimited bandwidth).
During scaling, local total capacity changes with worker count; shared capacity
does not. New workers start with cold G1/G2 and local G3, but can read existing
shared G3 blocks. Scale-in drains that worker's accepted I/O and releases its
pins before removing its local pool. Shared blocks survive their writer's exit.
Worker IDs are not reused. Replay requires at least one initial worker; its
existing scaling lifecycle permits scaling to zero and later adding new workers.

Completed G2 stores asynchronously write through to G3. Reads restore a
contiguous prefix through G3 → G2 → G1: G3 completion alone does not make GPU
blocks ready. The adapter reserves G2 destinations one block at a time; a later
miss or capacity failure keeps earlier accepted promotions. New promotions in
one lookup form one read job, and pending promotions defer H2D until a retry.
G3 writes retain the leading new blocks that fit while preserving blocks from
the same write cohort already in G3. A cohort larger than the available capacity is
partially stored; if no block fits, that optional insertion is skipped.
Resident, unpinned G3 blocks use deterministic LRU eviction.

Pending G2 destinations cannot be evicted. Completed promotions become ordinary
evictable G2 entries; a lookup hit alone does not pin them. H2D takes its own
source pins. Request termination detaches from accepted promotions, which may
still finish into G2 without activating G1 for the terminated request.

Transfers use the replay virtual clock and begin first-byte latency when
accepted. Reads start at the current lookup time. First-byte waiters consume
no bandwidth. Read and write budgets are independent; each moving job gets
an equal share of its worker's bandwidth. In `cluster_shared` scope, this is
also capped by its equal share of shared backend bandwidth. `worker_local`
ignores shared bandwidth limits entirely: for example, 16 workers at 10 GB/s
can reach 160 GB/s combined, while the default shared backend caps that at
80 GB/s. Unused shares are not redistributed. This is a fluid bandwidth
model, with no thread-pool or job-concurrency limit; finite backend execution
concurrency can therefore make real transfers slower.

For zero-duration I/O, a repeated request/key at the same timestamp falls back
to cache-miss handling while the recoverable prefix cannot fit in G2. Capacity
relief or time advancement permits retry. This simulator guard adds no pins or
invented latency and does not model native CPU retry overhead.

A restore can also thrash across time: when the recoverable prefix is larger
than G2 can hold, each promotion evicts a block promoted earlier for the same
request. After 1,024 restore rounds (lookups that submit G3 reads) in which its
DP rank schedules no work and emits no output, a request stops restoring from
G3 and computes what G1 and G2 do not hold, until it is admitted or
preempted. Rank progress, admission or preemption starts the count over, so a
request is never bypassed while its rank keeps working. Native vLLM has no such
rule.

The prediction summary includes `g3_offload` only when enabled, alongside the
existing TTFT, TPOT, and throughput metrics:

For Python `EngineReplayRunner`, each `run` creates a new native replay, so G3
cache contents and counters do not carry across reports, even when the Python
runner object is reused.

- `lookup_probes`, `lookup_hits`, `lookup_pending` count block probes, including
  retries. Hit ratio is hits / probes; pending probes are not hits.
- `read` and `write` report submitted, completed, and canceled jobs, completed
  bytes, and summed transfer milliseconds including first-byte latency.
  Canceled jobs add no completed bytes.
- `evictions`, `resident_blocks`, `pending_blocks`, and
  `cross_worker_read_blocks` describe tier state and reuse. Cross-worker reuse
  counts completed reads of blocks first written by another worker, not lookup
  hits. `--capture-per-request` retains the existing `requests.jsonl` output.
- `bypassed_restores` counts the times a request stopped restoring from G3
  under the rule above. It is omitted when zero.

G3 supports aggregated vLLM with fixed or dynamically scaled workers, prefix caching enabled,
attention DP equal to one, and no native speculative decoding. It does not
support `recommend`, disaggregated mode, or hardware integration. Either G2
scope can be combined with G3; a completed G2 store stays pinned until it is
handed to G3 write-through.
Replay owns the deployment-wide tier; direct scheduler construction cannot
provide it. Omit `g3_offload` to keep existing G1/G2 behavior.

These controls do not establish filesystem or real-GPU performance parity.
The existing G2 full-external-hit boundary remains: Replay may recompute one
full block where the reference vLLM external-receive path recomputes one token.
G3 byte counters do not resolve that difference.

<a id="manual-state-cache-sizing"></a>

## State-cache sizing

For recurrent-state models, set `state_cache.bytes_per_request` under
`engine.workers.aggregated.kv_cache`. It is disabled by default. Supply the total state size
per request per simulated rank, including any padding, separately from token KV bytes:

```yaml
kv_cache:
  block_size: 64
  bytes_per_token: 16
  capacity: {type: fixed, bytes: 8192}
  state_cache: {bytes_per_request: 1500}
```

This gives eight 1024-byte blocks. Each request's state uses two blocks, rounded up, in
addition to its token KV. `capacity: {type: fixed, blocks: 8}` is equivalent. Explicit
state bytes preserve the authored block geometry without loading model configuration.

Use `state_cache: {}` to resolve Kimi K3 cache geometry automatically. The existing
K3 model supplies target MLA KV bytes/token using `engine.kvcache_quant_mode`;
`kv_cache.bytes_per_token: auto` is the default. `block_size` is a requested
page granularity (default 64), enlarged to fit one KDA state per layer. Both the
resolved token geometry and state size are passed to the worker before capacity
checks. With TP8, BF16 KV resolves to block 768 / 27648 bytes per token; FP8 with
requested block 128 resolves to block 1536 / 13824 bytes per token.

The allocation rule follows vLLM KDA none/align mode at commit
`a474da28131f61684849b31e29af0eebaaedc383`, verified with Triton MLA. Requested
blocks must be multiples of 16 and satisfy the selected kernel's minimum alignment;
use 128 for a kernel requiring 128. This is independent of the performance database
version. TP must divide KDA heads; PP must be 1.

Each layer has three convolution buffers and one FP32 recurrent matrix. Optional
`state_cache.mamba_cache_dtype` accepts `auto`, `float16` or `float32`; auto uses the
model dtype. The estimate includes page padding for one state copy. Additional
speculative slots are outside this estimate; checkpoint copies are managed by the runtime.
Fixed G1 capacity remains required; physical blocks are charged after rounding
state bytes up to whole pool blocks. Results record resolved block size, token
bytes, state bytes and calculation provenance in `prediction.json`.

The same resolver is available through the SDK:

```python
from aisimulate_core.sdk import estimate_state_cache

state = estimate_state_cache(
    "moonshotai/Kimi-K3", tp_size=8, kvcache_quant_mode="fp8", block_size=128,
)
# block_size=1536, kv_bytes_per_token=13824, bytes_per_request=61046784
```

An explicit `state_cache.bytes_per_request` bypasses inference and requires
explicit block size and token bytes; omit `state_cache` or set it to `null` to
disable state caching. Automatic sizing also accepts an explicit token byte rate.
It reuses the SDK model loader rather than maintaining a separate config parser.
State caching supports aggregated vLLM with fixed G1 capacity on `predict --stack
engine`; other runner stacks must advertise support. Host/G3 offload,
disaggregated mode and `recommend` remain outside this interface.

With state caching enabled, `prefix_match_unit` controls prefix-matching
granularity while `block_size` still controls physical KV allocation. The match
unit must be positive and divide `block_size`. Omitting it preserves existing
state-cache behavior. It works with both inferred (`state_cache: {}`) and explicit
state sizes. Changing the match unit does not change the bytes in one state copy
or the physical block size; the state manager accounts for retained checkpoints
and temporary restore copies separately.

For example, this aggregated-worker configuration allocates 1536-token KV blocks
and allows prefix matches at 128-token boundaries. The byte sizes are
illustrative, not measured K3 memory sizes or automatic TP/DCP sizing:

```yaml
scheduler:
  max_batched_tokens: 8192
kv_cache:
  block_size: 1536
  prefix_match_unit: 128
  bytes_per_token: 16
  capacity: {type: fixed, blocks: 64}
  state_cache: {bytes_per_request: 24576}
```

Each physical block is 24576 bytes. `state_cache.bytes_per_request` specifies the
size of one state copy, which occupies one block in this example.
`capacity: {type: fixed, bytes: 1572864}` is equivalent to the 64-block capacity.
Capacity covers token KV, working and cached states, and temporary copies; cached
checkpoints may be evicted under pressure.

With this configuration, a cold 24,300-token prompt finishes prefill steps at
7680 / 15360 / 23040 / 24192 / 24300; the last step computes the remaining 108
tokens. Only the snapshots at 23040 and 24192 are retained for reuse.
A later request sharing 24192 tokens can resume there;
one sharing 23700 tokens resumes at 23040. A finer match unit does not create
state snapshots at every matching boundary.

This follows [vLLM v0.29.0](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/v1/core/sched/scheduler.py)
align mode without internal prefill checkpoints or periodic retention.
`kda_prefill_backend` and `prefix-cache-retention-interval` are not exposed;
shared-prefix junction retention, native DCP cache-group layout and overlapping
asynchronous forwards are not modeled. With explicit `prefix_match_unit`, use
the default LRU eviction policy; speculative decoding, KV event export and Belady
eviction are rejected. Prefill alignment is inactive when prefix caching is disabled.
