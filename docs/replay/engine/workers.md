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
| `workers.<role>.startup_seconds` | `0` | fixed | Nonnegative. Delay before a worker added during the run becomes ready. Initial workers are ready at time zero. |
| `workers.<role>.parallelism.preset` | Not applicable | `default` | Recommend only. `default` searches the generated parallelism space; a list pins candidate shapes; `false` or `{}` disables presets so the individual fields below define the space. See [search space](../../sweeper/search-space.md). |
| `workers.<role>.parallelism.replicas` | `1` | Searched | Number of workers in this role. |
| `workers.<role>.parallelism.tensor` | `1` | Searched | Attention tensor parallelism. |
| `workers.<role>.parallelism.pipeline` | `1` | Searched | Pipeline stages. |
| `workers.<role>.parallelism.attention_data` | `1` | Searched | Attention data-parallel ranks per worker. |
| `workers.<role>.parallelism.moe_tensor` | `1` | Searched | MoE tensor parallelism. Repartitions the worker's GPUs and does not add GPUs. |
| `workers.<role>.parallelism.moe_expert` | `1` | Searched | MoE expert parallelism. Repartitions the worker's GPUs and does not add GPUs. |
| `workers.<role>.parallelism.prefill_context` | Unset (1) | predict only | Prefill context parallelism (vLLM `-pcp`, SGLang `--attn-cp-size`). Adds GPUs. |
| `workers.<role>.parallelism.decode_context` | Unset (1) | predict only | Decode context parallelism (vLLM `-dcp`, SGLang `--dcp-size`). Must divide `tensor`; adds no GPUs. |
| `workers.<role>.scheduler.max_batched_tokens` | `8192` | Prefill and aggregated: `{choices: [8192, 16384, 32768]}`; decode: fixed | Positive. Token budget for one scheduler pass on one attention-DP rank. See [SGLang note](#batch-limits). |
| `workers.<role>.scheduler.max_sequences` | `256`; `1` for `prefill` | Prefill: `{choices: [1, 2, 4, 8, 16, 32, 64, 128, 256]}`; others: `{choices: [256, 512, 1024]}` | Positive. Maximum running requests on one attention-DP rank. |
| `workers.<role>.scheduler.prefill_schedule_interval` | `1` | predict only | vLLM only. See [vLLM prefill schedule interval](#vllm-prefill-schedule-interval). |
| `workers.<role>.scheduler.prefill_decode_interval` | `0` | predict only | SGLang only. See [SGLang prefill/decode interval](#sglang-prefilldecode-interval). |
| `engine.enable_chunked_prefill` | Backend default (on) | fixed | Boolean. Applies to aggregated and prefill roles; `false` is rejected for SGLang and for AFD. |

A backend-specific scheduler field set to a non-default value on another
backend fails validation.

## Parallelism

### GPU count

A role uses

```text
replicas × pipeline × tensor × attention_data × prefill_context
```

GPUs. `moe_tensor`, `moe_expert` and `decode_context` partition those GPUs
differently and do not add more. A deployment's GPU count is the sum over its
roles. In `recommend`, `optimization.constraints` bounds this sum.

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
> `max_batched_tokens` does not change SGLang replay results.

With chunked prefill (the default), a long prompt is split across passes that
each respect the token budget. With `enable_chunked_prefill: false` on vLLM, a
prompt larger than `max_batched_tokens` can never be scheduled and the run
fails. Keep `max_batched_tokens` at least as large as the longest prompt.

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
