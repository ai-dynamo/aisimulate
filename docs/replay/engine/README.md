<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Replay engine

`engine` describes the deployment that replays the [workload](../workloads.md):
the model, hardware and backend, the worker pools and their parallelism,
scheduler limits, KV cache, P/D transfer and speculative decoding. Every
`predict` and `recommend` configuration has one `engine` block next to
`traffic`.

```yaml
engine:
  mode: aggregated            # deployment shape
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  workers:                    # one entry per worker role
    aggregated:
      parallelism: {}         # replicas, TP, PP, attention DP, MoE TP/EP, CP
      scheduler: {}           # batch limits and backend cadence
      kv_cache: {}            # G1 capacity, prefix caching, G2/G3 offload
      timing: {}              # forward-pass timing provider
  # kv_transfer: {...}        # P/D only
  # speculation: {...}        # optional ngram speculative decoding
```

The smallest valid block sets `model`, `hardware` and an empty worker role:

```yaml
engine:
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  workers:
    aggregated: {}
```

With every other field omitted, it is equivalent to:

```yaml
engine:
  mode: aggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  context_length: max
  workers:
    aggregated:
      parallelism: {replicas: 1, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}
      kv_cache:
        block_size: 64
        prefix_caching: true
        bytes_per_token: auto
        capacity: {type: default, memory_fraction: 0.9}
      timing: {type: default}
      startup_seconds: 0
```

In `recommend`, omitted `mode`, `backend` and parallelism fields become search
domains instead of the single values above; see
[Sweeper search space](../../sweeper/search-space.md).

## Read by block

| Block | Page |
| --- | --- |
| `engine.mode`, `model`, `hardware`, `backend`, `backend_version`, `context_length`, `workers` roles, `tokenization`, `detokenization` | [This page](#deployment-fields) |
| `engine.workers.<role>.{hardware, startup_seconds, parallelism, scheduler}`, `engine.enable_chunked_prefill` | [Workers](workers.md) |
| `engine.workers.<role>.kv_cache` | [KV cache](kv-cache.md) |
| `engine.kv_transfer` | [P/D KV transfer](kv-transfer.md) |
| `engine.speculation`, `engine.nextn`, `engine.nextn_accepted` | [Speculative decoding](speculation.md) |
| `engine.workers.<role>.afd`, `engine.workers.encoder` | [Analytical AFD and EPD](analytical.md) |
| `engine.workers.{aggregated,prefill}.{host_loop, frontend, host_profile, vision}` | [Workers](workers.md#sglang-host-loop-frontend-and-vision); design on [SGLang VL host loop](sglang-vl-host-loop.md) |
| `engine.workers.<role>.timing`, `estimation_mode`, `estimator_config`, `database_mode`, quantization and kernel selectors | [Performance-model configuration](../../perf-model/configuration.md) |

Engine prediction and recommendation accept `auto`, `op_level`,
`fpm_interpolation`, and `fpm_regression` in their global and worker timing
selection schemas. They reject `fpm_learned` during configuration validation.
Native replay also rejects it before loading artifacts: phase timing cannot
provide the complete iteration features required by the
[learned estimate/tune API](../../perf-model/fpm_ml/configuration.md).

Field tables on these pages use the following columns:

- **Default** is the value used when the field is omitted in `predict`.
- **Recommend** is `fixed` when `recommend` accepts only a concrete value,
  `predict only` when `recommend` rejects the field, or the default search
  domain when `recommend` searches it.

<a id="deployment-fields"></a>

## Deployment fields

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `engine.mode` | `aggregated` | `{choices: [aggregated, disaggregated]}` | `aggregated` or `disaggregated`. AFD is a per-role worker type described in [Analytical AFD and EPD](analytical.md). |
| `engine.model` | Required | fixed | Hugging Face ID or local model directory. |
| `engine.hardware` | Required | `auto` allowed | System identifier, such as `h200_sxm`. `auto` is recommendation-only and resolves from `optimization.hardware`. |
| `engine.backend` | `vllm` | `{choices: [vllm, sglang]}` | `vllm`, `sglang`, or `trtllm`. Selects both the scheduler semantics and the performance data. |
| `engine.backend_version` | `current` | fixed | Selects performance data only. It does not change scheduler behavior. Must be a queryable version for the hardware and backend: `current`, `previous` when configured, `next` when newer data is available, or a version one of those available aliases resolves to. See [backend versions](#backend-versions). |
| `engine.context_length` | `max` | fixed | Positive prompt-plus-output token limit, or `max`. With a number, prompts at or above the limit are rejected and generation stops at the limit. `max` applies the model config's maximum on vLLM; SGLang and TensorRT-LLM then run without a limit. |
| `engine.workers` | Required | Per mode | Role mappings; see [Worker roles](#worker-roles). A role may be `{}` to use all defaults. |
| `engine.tokenization` | Unset (off) | predict only | Optional `workers` (positive integer, default `1`), `base_ms` and `per_input_token_us` (finite and non-negative, default `0`). Tokenizes each request on its worker before the engine; see [Request tokenization](#tokenization). Aggregated and disaggregated token replay on `--stack engine` only; online replay, worker scaling, AFD, encoder pools and SGLang frontend stages (which already price the tokenizer) are rejected. |
| `engine.detokenization` | Unset (off) | predict only | Optional `workers` (positive integer, default `1`), `base_ms` and `per_output_us` (finite and non-negative, default `0`). Detokenizes each pass's outputs on the worker that emits tokens before the client receives them; see [Response detokenization](#detokenization). Aggregated and disaggregated token replay on `--stack engine` only; online replay, worker scaling, AFD, encoder pools and `traffic.load.agentic_profile` are rejected. |

<a id="tokenization"></a>

## Request tokenization

A serving engine tokenizes each request in its server process before the
scheduler sees it, and longer prompts take longer. `engine.tokenization` models
this on every worker that receives new requests: the aggregated workers, or the
prefill workers in a disaggregated deployment.

```yaml
engine:
  tokenization:
    workers: 1
    base_ms: 0.7
    per_input_token_us: 1.6
```

Each worker has its own tokenizer with `workers` servers. After placement, a
request waits for a free server on its worker, takes
`base_ms + per_input_token_us × input tokens` on it, and then reaches that
worker's engine. Servers are taken first come, first served.

- With `workers: 1`, requests on a worker are tokenized one at a time in
  arrival order. A long prompt delays every request behind it, and requests
  reach the engine in arrival order.
- With more workers, a short prompt can finish before a long one that arrived
  earlier and reach the engine first.

Without this block, requests reach each engine at their arrival time, and
requests that arrive together keep their authored order; see
[same-timestamp admission](../workloads.md#same-timestamp-admission).

Set `workers` to the number of requests your engine tokenizes concurrently per
worker. The corresponding engine options are:

| Engine | Option | Default |
|---|---|---|
| vLLM | `--renderer-num-workers` per API server, times `--api-server-count` | 1, times 1 without data parallelism |
| SGLang | `--tokenizer-worker-num` | 1 |
| TensorRT-LLM (`trtllm-serve`) | `--num_input_processor_workers`, where available | 8 |

Measure `base_ms` and `per_input_token_us` on your own deployment. One way is
to send prompts of different lengths one at a time and fit the time from
request arrival to engine admission against input length. The per-token cost
depends mostly on the model's tokenizer and the host CPU. The values above
are only an illustration.

- TTFT and end-to-end latency are measured from the arrival time, so they
  include tokenization, as a client would observe.
- With `--capture-per-request`, each record has `tokenized_ms`, the time the
  request reached its worker's engine. Subtract it from `first_token_ms` for
  the engine-side TTFT, or from `first_admit_ms` for the engine queue time.
- Under `load.type: concurrency`, a request being tokenized holds its
  concurrency slot.
- With `traffic.stop.max_virtual_time_seconds`, a request still being
  tokenized at the cutoff is reported as incomplete.
- When an agentic profile cancels a request that is still being tokenized,
  the request never reaches the engine. Its tokenizer time stays reserved,
  so requests behind it on that worker are not released earlier.
- Setting both costs to `0` is the same as omitting the block.

<a id="detokenization"></a>

## Response detokenization

A serving engine returns generated tokens through its server processes, which
detokenize each request's new tokens, format streamed chunks and write them to
the clients. The client therefore receives every token, the first one
included, some time after the engine produced it, and more so when many
requests are running. `engine.detokenization` models this on every worker
that emits tokens: the aggregated workers, or the decode workers in a
disaggregated deployment.

```yaml
engine:
  detokenization:
    workers: 1
    base_ms: 0.5
    per_output_us: 40
```

Each worker has `workers` detokenizer servers, shared by its attention-DP
ranks, and each request belongs to one of them. When a pass ends, each server
takes the outputs of that pass that belong to it as one job, after the jobs it
has taken before. A job takes `base_ms + per_output_us × outputs`, where an
output is one request's new tokens in the pass, and the client receives every
output of the job when it ends.

- A pass with many running requests therefore delays all of its tokens, and
  when passes end faster than a server finishes its jobs, later passes wait
  for it.
- With more workers, the outputs of a pass are split over the servers and
  each server handles fewer of them.
- A request is assigned to a server by a stable hash of its request ID. This
  stands in for how the serving engine spreads connections over its server
  processes.
- A job's outputs reach the client together. vLLM hands a pass's outputs to
  the connections in chunks of `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` (128 by
  default), so with more than 128 outputs per server in one pass, replay
  delivers the earlier chunks somewhat late.

Set `workers` to the number of server processes that detokenize outputs. The
corresponding engine option is:

| Engine | Option | Default |
|---|---|---|
| vLLM | `--api-server-count` | 1 without data parallelism; with internal data-parallel load balancing, the data-parallel size |

Measure `base_ms` and `per_output_us` on your own deployment: record when the
engine produces each pass's outputs and when the client receives each token,
at a few concurrencies, and fit the delays. They depend mostly on the serving
frontend and the host CPU. The values above are only an illustration.

- Token times, TTFT, TPOT, end-to-end latency and throughput use the time the
  client receives each token. With `--capture-per-request`, `first_token_ms`,
  `last_token_ms` and `terminal_time_ms` are client receive times.
- A request finishes when the client receives the end of its response. A
  request that waits on it, such as the next request of a concurrency slot or
  a session's next turn, is released at that time.
- A rejected request carries no output, so it skips detokenization: the client
  learns of the rejection when the engine rejects the request, in both
  aggregated and disaggregated deployments, and the rejection does not delay
  the other outputs of its pass.
- The TTFT stage breakdown of an SGLang `host_loop` worker adds
  `mean_detokenization_ms`, the time from the engine emitting the first token
  to the client receiving it, so the stages still sum to TTFT. Without
  detokenization this stage is zero and TTFT ends when the engine emits the
  first token.
- With `traffic.stop.max_virtual_time_seconds`, a request whose response the
  client has not received by the cutoff is reported as incomplete.
- Replay telemetry samples count a completion when the engine finishes the
  request, before the client receives it.
- Setting both costs to `0` is the same as omitting the block.

## Backend versions

The accepted versions are the populated slots defined by
[`query_versions.yaml`](../../../python/aisimulate/src/aisimulate_core/systems/query_versions.yaml):

- `current` is the newest completed full collection for the backend, a
  systematic campaign over the whole case plan on the fleet;
- `previous` is the full collection that `current` replaced;
- `next` is the in-development version: the highest version newer than
  `current` for which any measurements have landed (single-op or
  single-model drops; whatever it lacks is served from `current` and older
  data). When nothing newer than `current` has data, `next` is the same
  version as `current`.

Versions outside these slots, including some support-matrix `PASS` rows, are
rejected with an error listing the accepted versions. Set
`engine.backend_version: current` or use a version in that list.
See [the support-matrix version boundary](../../perf-model/support-matrix.md#versions-the-cli-accepts).

<a id="worker-roles"></a>

## Worker roles

`engine.workers` holds one mapping per worker role. The roles that must be
present follow `engine.mode`:

| `mode` | Required roles | Rejected roles |
| --- | --- | --- |
| `aggregated` | `aggregated` | `prefill`, `decode`, and `engine.kv_transfer` |
| `disaggregated` | `prefill` and `decode` | `aggregated` |
| AFD worker type | The same aggregated or P/D roles | Each role independently chooses ordinary or AFD execution; see [AFD](analytical.md#afd). |

Each ordinary (`type: monolithic`) role has the same sub-blocks (`parallelism`, `scheduler`, `kv_cache`,
`timing`, plus `hardware` and `startup_seconds`), configured independently. In
`recommend`, a mode domain may declare all three roles; each candidate keeps
only the roles its mode uses. An optional `encoder` role adds an analytical or
native EPD encoder pool; see [EPD](analytical.md#epd).

All roles share `model`, `backend` and `backend_version`;
per-role overrides of these fields are rejected. Only disaggregated `prefill`
and `decode` workers may override `hardware`.
Workers may override the shared `context_length`; see
[context limits](workers.md#context-limits) for prediction and recommendation rules.

## A complete P/D example

```yaml
engine:
  mode: disaggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: sglang
  workers:
    prefill:
      parallelism: {replicas: 2, tensor: 2}
    decode:
      parallelism: {replicas: 1, tensor: 4}
      scheduler: {max_sequences: 512}
  kv_transfer:
    bandwidth_gb_per_second: 50
```

This deployment uses `2 × 2 + 1 × 4 = 8` GPUs. Run it with:

```bash
aisimulate predict --stack engine --config prediction.yaml --output-dir ./pd-prediction
```

## Where engine options are validated

Replay validates the `engine` block before running. Unsupported combinations,
such as a backend-specific field on another backend or G3 offload in P/D,
fail with an error naming the field. They are never silently ignored. For which
features combine with which backends and topologies, see
[feature support](../features.md).
