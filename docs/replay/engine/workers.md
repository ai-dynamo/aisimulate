<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Workers

A worker role (`aggregated`, `prefill` or `decode`) describes one pool of
identical workers. This page covers the role's shape and scheduler. KV cache,
timing and P/D transfer have their own pages; see the
[engine overview](README.md#read-by-block).

Merge this fragment into an existing `engine` block from the [engine overview](README.md):

```yaml
engine:
  workers:
    aggregated:                 # or prefill / decode
      startup_seconds: 0
      parallelism:
        replicas: 2
        tensor: 4
        attention_data: 1
      scheduler:
        max_batched_tokens: 8192
        max_sequences: 256
```

## Worker fields

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `workers.<role>.hardware` | `engine.hardware` | fixed | `prefill` and `decode` in disaggregated mode only. One concrete system identifier; `auto` is rejected. |
| `workers.<role>.context_length` | `engine.context_length` | fixed | Positive token limit. Recommendation accepts it for `prefill` and `decode`; aggregated recommendations use `engine.context_length`. See [Context limits](#context-limits). |
| `workers.<role>.startup_seconds` | `0` | fixed | Nonnegative. Delay before a worker added during the run becomes ready. Initial workers are ready at time zero. |
| `workers.<role>.parallelism.preset` | Not applicable | `default` | Recommend only. `default` searches the generated parallelism space; a list pins candidate shapes; `false` or `{}` disables presets so the individual fields below define the space. See [search space](../../sweeper/search-space.md). |
| `workers.<role>.parallelism.replicas` | `1` | Searched | Number of workers in this role. |
| `workers.<role>.parallelism.tensor` | `1` | Searched | Attention tensor parallelism. |
| `workers.<role>.parallelism.pipeline` | `1` | Searched | Pipeline stages. Engine replay requires `1`, except for the [AFD+PD companion](analytical.md#afd). |
| `workers.<role>.parallelism.attention_data` | `1` | Searched | Attention data-parallel ranks per worker. |
| `workers.<role>.parallelism.moe_tensor` | `1` | Searched | MoE tensor parallelism. Repartitions the worker's GPUs and does not add GPUs. |
| `workers.<role>.parallelism.moe_expert` | `1` | Searched | MoE expert parallelism. Repartitions the worker's GPUs and does not add GPUs. |
| `workers.<role>.parallelism.prefill_context` | Unset (1) | predict only | Prefill context parallelism (vLLM `-pcp`, SGLang `--attn-cp-size`). Adds GPUs. |
| `workers.<role>.parallelism.decode_context` | Unset (1) | predict only | Decode context parallelism (vLLM `-dcp`, SGLang `--dcp-size`). Must divide `tensor`; adds no GPUs. |
| `workers.<role>.scheduler.max_batched_tokens` | `8192` | Prefill and aggregated: `{choices: [8192, 16384, 32768]}`; decode: fixed | Positive. Token budget for one scheduler pass on one attention-DP rank. See [SGLang note](#batch-limits). |
| `workers.<role>.scheduler.max_sequences` | `256`; `1` for `prefill` | Prefill: `{choices: [1, 2, 4, 8, 16, 32, 64, 128, 256]}`; others: `{choices: [256, 512, 1024]}` | Positive. Maximum running requests on one attention-DP rank. |
| `workers.<role>.scheduler.prefill_schedule_interval` | `1` | predict only | vLLM only. See [vLLM prefill schedule interval](#vllm-prefill-schedule-interval). |
| `workers.<role>.scheduler.prefill_decode_interval` | `0` | predict only | SGLang only. See [SGLang prefill/decode interval](#sglang-prefilldecode-interval). |
| `workers.<role>.scheduler.async_scheduling` | vLLM's default: on, off with ngram drafts | predict only | vLLM only. Boolean. `true` is rejected with ngram drafts. See [vLLM async scheduling](#vllm-async-scheduling). |
| `workers.aggregated.scheduler.launch` | Unset | predict only | vLLM only, with async scheduling, on one-GPU workers (`tensor`, `pipeline`, `attention_data` and `prefill_context` of 1) and `--stack engine`. `{decode_ms, piecewise_ms, eager_ms, max_cudagraph_capture_size, eager_queue_fraction}`: CPU launch costs measured on the deployment. Outputs leave once the next pass is launched, and an arrival joins the first pass decided after it. See [vLLM pass launch](#vllm-pass-launch). |
| `workers.{aggregated,prefill}.host_loop` | `false` | fixed | SGLang only; `aggregated` in aggregated mode, `prefill` in disaggregated mode. Each pass is one overlap-scheduler iteration. Implied by `frontend` and `host_profile`. See [SGLang host loop, frontend and vision](#sglang-host-loop-frontend-and-vision). |
| `workers.{aggregated,prefill}.frontend` | Unset | fixed | Ordered `stages`, each `{workers, service_ms, concurrency_scale}`, plus optional `measured_for` (model, frontend, feature transport and image shape the stages were measured on). Exclusive with `host_profile`. Unset together with `host_profile`, a worker with images encodes them on the GPU and prices no frontend stage, so its TTFT starts at the scheduler. |
| `workers.{aggregated,prefill}.host_profile` | Unset | fixed | `{path, frontend}` with `frontend` `python` (SGLang's default HTTP server with its `TokenizerManager`) or `rust` (the embedded Rust server, `SGLANG_RUST_SERVER=1`, whose Rust extensions are built from an SGLang source tree). Frontend stages from a host cost table written by `python -m aisimulate.vl.collect`; a relative `path` resolves against the working directory of `collect` and `predict` alike. The table describes SGLang's CPU image-processor path (`--image-processor-backend pil`); SGLang 0.5.19's default `auto` on a GPU host runs the fast torchvision processor on the serving GPU, which is not priced, so a deployment matches its table only when launched with that flag. No table is read or measured unless this field is set. |
| `workers.{aggregated,prefill}.vision.cache_mib` | `100` | fixed | Non-negative MiB; `0` disables the embedding cache. SGLang multimodal embedding cache (`SGLANG_VLM_CACHE_SIZE_MB`) for images encoded on the worker. |
| `workers.{aggregated,prefill}.vision.encoder_parallel` | `tp` | fixed | `tp` shards the vision tower over the tensor-parallel group; `dp` (`--mm-enable-dp-encoder`) replicates it and splits the images. Sets encoder timing, collectives and per-rank tower weights. Adds no ranks; see the four parallelism knobs under [Compatibility and scope](sglang-vl-host-loop.md#compatibility-and-scope). |
| `engine.enable_chunked_prefill` | Backend default (on) | fixed | Boolean. Applies to aggregated and prefill roles; `false` is rejected for SGLang and for AFD. |

A backend-specific scheduler field set to a non-default value on another
backend fails validation.

## Context limits

Each worker's `context_length` overrides `engine.context_length` for that role.
Prediction supports the override on aggregated, prefill and decode workers;
aggregated recommendations must use the shared engine field.
Saved candidate YAML keeps inherited limits unset on the worker, so editing
`engine.context_length` still updates every worker without an explicit override.

For synthetic recommendation traffic, an explicit prefill limit must cover the
input tokens plus one generated token. An explicit decode limit must cover input
plus output tokens. The default synthetic workload applies when traffic is omitted.
Trace traffic is checked by runtime admission, not this fixed-length validator.

Sweeper uses the role limit, then the shared limit, then the model's maximum
context for KV-feasibility filtering. Generated worker payloads use the role
or shared limit; if both are unset, `max_model_len` remains unset. The model
fallback used for filtering is not copied into that payload. AFD companion
workers follow the same rule for their role. These are serving limits, not
measured performance data.

Native SGLang admission follows v0.5.22. For context limit `C`, KV token
capacity `K`, page size `B`, and prompt length `P`, its request limit is
`R = min(C - 1, K - 1)` (with unset `C` bounded by `K`). Prompts at or above
`R - 5`, or whose prompt plus one page cannot fit strictly below `K`, are
rejected. Requested outputs are clamped to the nonnegative minimum of the
requested count, `R - P - 1`, and `K - ceil(P / B) * B - B - 1`.
This can shorten generation even when the configured context is larger than
the prompt plus the requested outputs.

## Parallelism

### GPU count

A role uses

```text
replicas × pipeline × tensor × attention_data × prefill_context
```

GPUs. `moe_tensor`, `moe_expert` and `decode_context` partition those GPUs
differently and do not add more. A deployment's GPU count is the sum over its
roles. In `recommend`, `optimization.constraints` bounds this sum.

Engine replay reports each role's GPUs per worker as
`tensor × attention_data × prefill_context`. Engine replay rejects pipeline
parallelism greater than one before capacity estimation. GPU-hours sum each
role's worker-seconds times this count, divided by 3600. Aggregated workers use
the `decode_gpus_per_worker` and `decode_worker_seconds` report fields. Neither
DCP nor MoE EP adds another multiplier.

[AFD+PD](analytical.md#afd) is the exception. Its companion worker is priced
analytically, not replayed, so it accepts `pipeline` above one. Its GPU count is
`replicas × tensor × pipeline × attention_data`.

`prefill_context` and `decode_context` are modeled only for some model and
backend combinations, and an aggregated worker may set at most one of them above
1. See [decode context parallelism](../../perf-model/configuration.md#decode-context-parallelism)
for the supported models. Recommend searches only CP=1.

### Replicas and attention DP

`replicas` is the number of logical workers. Each worker has
`attention_data` rank schedulers. Each rank owns its own running requests,
scheduler limits and GPU KV cache. Tensor parallelism shards the same requests
across GPUs and does not add capacity for more sequences.

The ranks of one worker run their passes in lockstep. Every rank picks its own
batch, and the worker's pass ends when the slowest rank finishes. Tokens, KV
cache changes and timing become visible at that point. A rank with no work
still waits for the others, so a request that arrives mid-pass cannot start
early.

The default timing prices the whole pass at once: every rank's batch, idle
ranks included, goes to the performance model as one multi-rank
`ForwardPassMetrics` query, and each rank's forward takes that shared time from
its compute start. Fixed and polynomial timing, and the exceptions under
[engine replay timing](../../perf-model/configuration.md#engine-replay-timing),
keep pricing each rank's own prefill and decode, so the pass ends with the
slowest rank's own time.

On the `engine` stack, requests are placed round-robin across workers and then
across the ranks of each worker. A trace request that names a `dp_rank` keeps
that rank. Worker counts are fixed for the whole run. The
[Dynamo stack](../dynamo.md) replaces placement with the KV Router and can scale
worker counts with the Planner; `startup_seconds` then delays each added worker.

## Scheduler

<a id="batch-limits"></a>

### Batch limits

`max_batched_tokens` and `max_sequences` are per attention-DP rank, matching
how vLLM and SGLang apply them inside each DP engine. A prefill role defaults
to `max_sequences: 1`, so each prefill rank runs one request at a time unless
you raise it.

> **SGLang note.** The SGLang scheduler currently takes its per-pass prefill
> budget from SGLang's default `chunked_prefill_size` (8192 tokens, divided by
> `attention_data`), not from `max_batched_tokens`. Changing
> `max_batched_tokens` does not change SGLang replay results, except when
> `host_loop` is enabled or images are encoded on the worker (without a
> separate encoder pool). In those cases it is the worker's `chunked_prefill_size`.

With chunked prefill (the default), a long prompt is split across passes that
each respect the token budget. With `enable_chunked_prefill: false` on vLLM, a
prompt larger than `max_batched_tokens` can never be scheduled and the run
fails. Keep `max_batched_tokens` at least as large as the longest prompt.

SGLang checks the whole remaining prompt before admitting its first chunk.
The check protects the matched prefix from eviction and reserves one page
plus the remaining output budget (clipped at 4096 tokens by default).
Each admitted chunk charges page-rounded input and one page of overhead;
only the final chunk also reserves outputs. Thus, a small chunk size cannot
make an otherwise unfit request admissible. An in-flight chunk runs before
new requests, including with LPM, and may use a partial-page compute budget
to make progress.

Running requests reserve their remaining output budget at `new_token_ratio`.
Its initial value is `min(0.7 * schedule_conservativeness, 1)`, its floor is
14% of that initial value, and each non-retracting running-batch update
decays it by one six-hundredth of the initial-to-floor distance. Retraction
re-estimates it from the survivors. Fully idle workers reset it; queued or
transfer-held work does not. A skipped decode waiting for an in-flight
completion neither retracts requests nor decays the ratio.

### SGLang computed KV at completion

Sampling an output token does not compute that token's KV. Prefill commits
the prompt, and a later decode forward consumes the sampled token. For a
prompt of `P` tokens and `N` outputs, serial decoding and aggregated overlap
length stops commit `P + N - 1` tokens. PD overlap can run the final token
through another forward before observing completion, committing `P + N`.
Cached prefixes are rounded down to whole pages.

Speculative decoding commits accepted drafts but leaves the sampled bonus
token uncomputed. A length stop at the bonus commits `P + N - 1`; a stop
inside an accepted draft run commits `P + N`. An already-finished speculative
row does not commit more KV while awaiting result observation. These are
computed-token boundaries using existing token-page keys; EAGLE bigram
identity and EOS-based stops are separate from this model.

These boundaries apply to the simulator's emitted token stream. The existing
P/D replay regenerates the first output on the decode worker instead of carrying
the prefill worker's sampled token, so speculative burst alignment can differ
from SGLang's transferred-token path. Speculative decode radix caching and decode
overlap are not supported by the public configuration.

### SGLang disaggregated capacity

SGLang decode preallocation reserves **512 tokens per active request** by
default, including requests in transfer and transferred requests waiting to
run. It also protects enough space for one running request to finish. A
candidate's matched prefix is pinned before checking the budget; only its
missing, page-rounded prompt needs new KV pages. Admission stops at the first
request that cannot fit. A prompt whose footprint plus the configured reserve
exceeds the whole pool is rejected at the destination command boundary.

The native Rust/JSON `engine.sglang` controls are:

| Field | Default | Meaning |
|---|---|---|
| `num_reserved_decode_tokens` | `512` | Decode headroom per preallocated request. A smaller value supports intentionally small synthetic KV pools; zero still retains the completion guard. |
| `disaggregation_decode_extra_slots` | unset | Extra decode request rows for transfers. Unset uses twice the explicit per-rank sequence limit when it is at most 32, otherwise zero. |

These controls apply to PD decode and are not YAML scheduler fields.
The effective running-request limit is capped at half the KV token capacity.
With native `max_num_seqs: null`, it is derived from KV capacity and context
length, bounded to 2,048–4,096 before that cap. An omitted `max_num_seqs` keeps
AISim's explicit default of 256. Decode has that many running
slots, the configured extra transfer rows, and twice the running limit in
transfer metadata slots. Transfer completion releases metadata; request rows
remain occupied until completion, cancellation, or retraction.

Retracted decodes restore private KV before new transfers are admitted or
transferred requests enter the running batch. They do not re-match the radix
cache or run a new prefill. The simulator models allocation and restoration
ordering, without a separate host-copy latency for these backups. On prefill
workers, completed sources retain their request row and KV until transfer
release, so waiting sources may be blocked even when no prefill is running.

<a id="vllm-prefill-schedule-interval"></a>

### vLLM prefill schedule interval

`prefill_schedule_interval: N` admits prefill work only on every `N`-th pass of
an attention-DP group. It mirrors vLLM's `prefill_schedule_interval`, which
keeps decode steps balanced across DP ranks. It has an effect only when
`attention_data` is above 1.

```yaml
engine:
  mode: aggregated
  model: nvidia/Kimi-K2.5-NVFP4
  hardware: b200_sxm
  backend: vllm
  workers:
    aggregated:
      parallelism: {attention_data: 8, moe_expert: 8}
      scheduler: {prefill_schedule_interval: 4}
```

On the passes in between:

- running decodes continue;
- a prompt with more than one prefill token remaining waits;
- connector loads, received P/D handoffs, and prompts with at most one token
  left can still advance.

If an admitting pass had to leave requests waiting for scheduler capacity, the
next pass may admit again. The counter resets when the whole DP group drains.
vLLM checks for a drained group every 32 steps and may run extra dummy steps;
Replay resets immediately, so the phase can differ for requests that arrive in
that window. Behavior follows vLLM at commit
[`e2fa285`](https://github.com/vllm-project/vllm/tree/e2fa28594f7baad142a426b0b6a2cfe2c79201c7).

<a id="vllm-async-scheduling"></a>

### vLLM async scheduling

`async_scheduling` mirrors vLLM's `--async-scheduling` and
`--no-async-scheduling`. Unset, replay turns it on, except with ngram drafts,
which turn it off. vLLM v0.28.0 resolves the same default for the
configurations replay models; it also turns async scheduling off for pooling
models, some speculative and executor settings and ROCm DeepEP DBO, so set it
explicitly to match such a deployment.

With async scheduling, a busy vLLM engine keeps two batches committed: one
running and the next queued behind it. When the running batch finishes, vLLM
processes its outputs, reads the requests that have arrived, and commits
another batch. Replay assumes that this happens before a client can react to
those outputs; without [`launch`](#vllm-pass-launch) it does not model when
outputs reach the client. The earliest batch a request may join is therefore
the first one committed after it arrives, and sequence limits, the token
budget or KV capacity can delay it further:

- A request that arrives while a pass runs skips the pass already queued and
  joins the one after it.
- A request that arrives as a pass ends counts as a reaction to that pass's
  outputs, such as a closed-loop client's next request, and also skips the
  pass committed at that boundary. An open-loop arrival that lands exactly on
  a pass end is therefore admitted one pass late.
- An idle engine admits an arrival at the next pass. So does an engine whose
  queued pass has no other work, because vLLM runs that empty batch at once.

Requests submitted to a disaggregated prefill role follow the same rule. A
decode role admits received KV handoffs as before.

```yaml
engine:
  backend: vllm
  workers:
    aggregated:
      scheduler: {async_scheduling: false}
```

With `async_scheduling: false`, an arrival joins the next pass. Behavior
follows `EngineCore.step_with_batch_queue` and `VllmConfig.max_concurrent_batches`
in vLLM
[v0.28.0](https://github.com/vllm-project/vllm/tree/2cf0a6915ce544dc493a0990f2ea38d81601128a).
Pipeline parallelism deepens vLLM's batch queue, but engine replay does not
model it. AFD companion workers do not model async scheduling and reject
`true`.

Under KV pressure, async scheduling also changes when blocks are freed and
which requests can be preempted:

- vLLM frees a finished request's blocks when it processes the outputs of the
  pass the request finished in, after it has committed the next pass. Blocks
  that a request frees in pass k therefore first serve pass k+2. A pass with no
  work lets vLLM process the previous pass at once, so replay then commits
  another pass at the same instant.
- With a CPU tier (`kv_cache.host_offload`), vLLM does not free the blocks of a
  request that the pass in flight runs until that pass ends, so it does not
  preempt that request. A running request that needs its blocks waits for the
  next pass instead.

vLLM defers the same frees for a P/D connector that consumes KV (`kv_role`
`kv_consumer` or `kv_both`). Replay does not model a P/D connector's role, so
it applies the second rule only with a CPU tier. These rules follow `Scheduler`
in vLLM
[v0.30.0](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/sched/scheduler.py),
the first release that stops the allocation instead of preempting such a
request.

<a id="vllm-pass-launch"></a>

### vLLM pass launch

Each iteration of vLLM's async engine loop decides a pass, launches it on the
GPU, then waits for the previous pass's sampled tokens and returns that pass's
outputs. `launch` models when this happens from the CPU time each launch
takes, so that:

- a pass's outputs reach the client once the next pass is launched, never
  before the pass ends. Without `launch` they leave at the pass end, which is
  too early when a short decode pass is followed by a pass with a long launch;
- an arrival joins the first pass decided after it. vLLM decides a pass once
  it has launched the previous one and the pass before that has ended, so a
  request that arrives early in a long pass can join the next pass, while
  counting passes would make it skip the pass already queued.

```yaml
engine:
  backend: vllm
  workers:
    aggregated:
      scheduler:
        launch:
          decode_ms: 5
          piecewise_ms: 50
          eager_ms: 150
          max_cudagraph_capture_size: 512
          eager_queue_fraction: 0.4
```

A pass with more than `max_cudagraph_capture_size` scheduled tokens (vLLM's
option of that name) runs eager, with or without prefill, and its launch takes
`eager_ms`. A smaller pass takes `piecewise_ms` with prefill and `decode_ms`
without. An eager pass launches many kernels, so once the CUDA launch queue is
full its launch ends no earlier than `eager_queue_fraction` of the way through
the pass. A launch never ends after its pass. An engine that was idle decides
a pass when a request arrives.

Measure the launch costs on your own deployment, for example from an Nsight
Systems trace of the engine core: the time from `schedule()` to the end of
`sample_tokens()` for each kind of pass, and for eager passes how far the GPU
has got when that launch ends. They depend mostly on the model, the engine
version and the host CPU; the values above are only an illustration.

`launch` needs aggregated vLLM workers with async scheduling and one GPU each
(`tensor`, `pipeline`, `attention_data` and `prefill_context` of 1): only then
does vLLM run its uniprocess executor, which launches a pass inside the engine
loop. With more GPUs its multiprocess executor returns a pass's outputs
independently of the next launch. `launch` moves no pass. With
[response detokenization](README.md#detokenization), a pass's outputs start
detokenizing when vLLM returns them. Only offline replay (`--stack engine`)
releases the outputs a rank holds, so other runners reject `launch`. Behavior
follows `EngineCore.step_with_batch_queue` in vLLM
[v0.30.0](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/engine/core.py),
its
[uniprocess executor](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/executor/uniproc_executor.py),
which
[`ParallelConfig`](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/config/parallel.py)
selects for one GPU, and the CUDA-graph
[`dispatch`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/cudagraph_utils.py)
of v0.32.0.

<a id="sglang-prefilldecode-interval"></a>

### SGLang prefill/decode interval

`prefill_decode_interval: N` blocks new prefill and prefill-chunk continuation
for the next `N` scheduler rounds after a round that ran prefill (an EXTEND
forward). `0` disables it. For `N = 2`, with prefill still queued:

| Round | Work on a rank |
| --- | --- |
| 0 | Prefill; blocks the next two rounds |
| 1 | Decode, or idle |
| 2 | Decode, or idle |
| 3 | Prefill; blocks the next two rounds |

```yaml
engine:
  mode: aggregated
  model: nvidia/Kimi-K2.5-NVFP4
  hardware: b200_sxm
  backend: sglang
  workers:
    aggregated:
      parallelism: {attention_data: 8, moe_expert: 8}
      scheduler: {prefill_decode_interval: 20}
```

Each blocked round counts once, whether it decodes, emits several speculative
tokens, or is idle. Idle blocked rounds take zero modeled time. If any rank of
the DP group ran prefill, every rank of the group is blocked, as in SGLang's
synchronized DP attention. Received P/D handoffs can still start decoding.

Not modeled: mixed-chunk forwards, SGLang's separate prefill delayer, CPU and
collective overhead, and idle sleeping. In speculative DP attention, SGLang can
force a peer rank idle while another rank prefills; Replay still lets that peer
decode in the same round. Behavior follows SGLang at commit
[`20621aa`](https://github.com/sgl-project/sglang/blob/20621aa14bda7726a8a968f326198eac61717fef/python/sglang/srt/managers/scheduler.py#L1211).

<a id="sglang-host-loop-frontend-and-vision"></a>

### SGLang host loop, frontend and vision

`host_loop: true` models each pass of an SGLang aggregated or prefill worker as
one iteration of the overlap scheduler loop (`event_loop_overlap`): requests are
received at iteration boundaries and a forward's outputs become visible one
iteration later. `frontend` or `host_profile` adds the request path ahead of
the scheduler as pools of black-box stages and implies `host_loop`; `vision`
sizes the embedding cache and lays out the vision tower for images encoded on
the worker. In disaggregated mode the fields go on `workers.prefill` and
`workers.decode` stays a plain worker.

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 128
    output_tokens: 32
    images: {height: 1024, width: 1024, count: 1, encoding: png, identity: unique}
  load: {type: concurrency, concurrency: 4}
  stop: {requests: 32}
engine:
  model: Qwen/Qwen3-VL-8B-Instruct
  hardware: h200_sxm
  backend: sglang
  workers:
    aggregated:
      scheduler: {max_batched_tokens: 8192, max_sequences: 64}
      host_profile: {path: ./host-costs.json, frontend: python}
      vision: {cache_mib: 100, encoder_parallel: tp}
```

```bash
python -m aisimulate.vl.collect --config prediction.yaml --sglang-python <serving venv python>
aisimulate predict --stack engine --config prediction.yaml
```

`collect` times the real SGLang frontend objects on the serving host (no GPU
needed) and adds one row per feature transport the frontend serves, keyed by
model, frontend, feature transport and image shape, to the table at
`host_profile.path`; a prediction whose workload has no row fails and prints
the `collect` command that adds it. Requires
`backend: sglang`, `pipeline: 1`, `attention_data: 1` and context parallelism
(`prefill_context`, `decode_context`) at `1`; images encoded on the
worker also require default timing. Image traffic takes either this path or an
encoder pool ([EPD](analytical.md#epd)), never both. The reported stage means
and per-request milestones are listed in the
[CLI reference](../../reference/cli.md#standard-output); mechanics and
approximations are on [SGLang VL host loop](sglang-vl-host-loop.md).

`collect` runs from an interpreter with aisimulate installed (Python 3.11 to
3.13) and launches the measurement in the serving venv named by
`--sglang-python`; the two are separate environments whenever the serving
venv's Python is older than aisimulate supports. `engine.backend_version`
selects the packaged GPU performance data, while the table is measured with the
serving host's sglang 0.5.19, on which the frontend stages are defined; the two
pins are independent.

### Backend scheduling differences

Each backend uses its own scheduling and preemption rules. With attention DP
above 1, SGLang ranks divide `chunked_prefill_size` by `attention_data` and
scale the output-reservation conservativeness by 0.3, as SGLang does at launch.
P/D handoff order also differs by backend; see
[P/D KV transfer](kv-transfer.md#handoff).

## Limitations

- Placement on the `engine` stack is round-robin and ignores cache overlap.
- Worker counts are fixed unless the Dynamo Planner scales them.
- Attention-DP ranks are synchronized per pass. Rank-to-rank communication time
  is part of the forward-pass estimate, not a separately modeled event.
