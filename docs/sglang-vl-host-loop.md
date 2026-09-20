<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang VL host loop and frontend modeling

Native VL replay predicts a vision-language deployment end to end on one
aggregated SGLang worker: requests cross the serving frontend, the scheduler
thread receives and batches them, the vision encoder runs inside the prefill
forward, and outputs become visible one scheduler iteration later. Configure it
in an `aisimulate predict` configuration by giving the aggregated worker `host`
costs (and optionally `frontend` pools) next to an image workload:

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 128
    output_tokens: 64
    images: {height: 1024, width: 1024, count: 1, encoding: png, identity: unique}
  load: {type: concurrency, concurrency: 4}
  stop: {requests: 32}
engine:
  mode: aggregated
  model: Qwen/Qwen3-VL-8B-Instruct
  hardware: h200_sxm
  backend: sglang
  workers:
    aggregated:
      parallelism: {replicas: 1, tensor: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 64}
      host:
        receive: {const_ms: 0.4, per_mib_ms: 0.02}
        select: {const_ms: 0.2, per_request_ms: 0.01}
        prepare_extend: {const_ms: 1.0, per_ktoken_ms: 0.1}
        launch_extend: {const_ms: 6.0, per_ktoken_ms: 0.5}
        prepare_vision: {const_ms: 1.0, per_mib_ms: 0.05}
        launch_vision: {const_ms: 3.0, per_image_ms: 1.5}
        launch_decode: {const_ms: 2.5}
        result: {const_ms: 0.3, per_request_ms: 0.02}
      frontend:
        io_workers: 16
        processor_workers: 2
        stages:
          - {resource: tm_loop, unit: request, cost: {const_ms: 0.3}}
          - {resource: io_decode, unit: image, cost: {const_ms: 4.0, per_ktoken_ms: 0.2}}
          - {resource: processor, unit: request, cost: {const_ms: 12.0, per_image_ms: 9.0}}
          - {resource: tm_loop, unit: request, cost: {const_ms: 0.8, per_mib_ms: 0.03}}
      vision: {cache_mib: 100, encoder_parallel: tp}
```

```bash
aisimulate predict --config prediction.yaml --capture-per-request
```

Every cost is an affine function of the work it is applied to: a constant plus
per-request, per-image, per-thousand-token, and per-MiB terms. The tables are
measured data, not tuning knobs: `host_profile: {path, frontend}` takes them
from a profile written by `python -m aisimulate.vl.calibrate` on a serving host
(see `tools/frontend/README.md` for the Rust frontend). Stage costs are constants
of the workload they were sampled with, so a profile fails closed when it was
measured for another model, frontend, image shape, count or encoding, text
length, or SGLang revision, or lacks a cost the deployment needs; its content
digest travels with every prediction and recommendation that used it.
`host_profile.on_missing: calibrate` samples the workload into `path` when the
file does not exist. `recommend` resolves a profile once and writes the scored
tables into each saved candidate, so a saved candidate never depends on a
profile file that can change afterwards.

Lower-level Runner specifications set the same tables on the SGLang rank:
`rank.sglang.host`, `rank.sglang.vlm_cache_bytes`, `rank.frontend`, and
`rank.vision: true`; Rust callers set `SglangConfig::host`,
`EngineConfig::frontend`, and `EngineConfig::vision`.

## Scheduling contract

Without `host`, an SGLang pass is one GPU forward whose outputs are visible
when it ends. With `host`, one pass is one iteration of SGLang's overlap
scheduler loop, and the per-request report carries the stage timestamps
`frontend_ready_ms`, `scheduler_received_ms`, `selected_ms`, and
`prefill_complete_ms` next to `first_token_ms`:

| Step | Modeled time |
| --- | --- |
| Receive | Requests that reached the scheduler since the last iteration are drained at the iteration start; each is charged `receive`. |
| Select | `tp_sync_ms` plus `select` over the batch; admissions are dated here. |
| Prepare | `prepare_extend` (plus `prepare_vision` when the batch encodes cache-miss images): input preparation the first kernel depends on. The GPU cannot start the forward before it ends. |
| Launch | `launch_extend` (plus `launch_vision`) or `launch_decode`: kernel enqueueing that overlaps the forward. A DECODE launch first waits for the previous forward, as SGLang's position update synchronizes the stream. |
| Forward | Starts when the inputs are ready and the GPU is free; encoder time for cache-miss images precedes the language-model forward; a forward cannot end before its launch does. |
| Result | The previous forward's outputs, terminals, prefix-cache commits (`maybe_cache_unfinished_req`), and KV release are processed after this launch, at `max(launch end, previous forward end) + result`. |

Consequences the tests pin: a request's first token is observed one iteration
after the forward that produced it; the radix cache learns a completed
prefill's prefix only when that forward is observed, so a request selected one
iteration later reuses nothing from it; a request that finished in forward `k`
still occupies a slot in batch `k+1` and, when that batch is a decode, is
allocated its row's KV slot, which is committed and cached with the request
when the finish is observed (only an unaligned tail page is freed); a
request that needs no output token completes when its prefill is observed,
not when it launches; a request cancelled while its forward is in
flight produces no output from it, while the device work stays charged;
requests arriving during an iteration are received at the next one; an
all-zero table reproduces the legacy token times exactly.

The vision encoder runs once per prefill batch over the cache-miss images whose
placeholders overlap the chunk being computed, deduplicated by image identity,
and its outputs are kept in an LRU embedding cache of `vision.cache_mib`
(SGLang's `SGLANG_VLM_CACHE_SIZE_MB`, default 100 MiB). The tower is priced by
the same canonical estimator as the language model
(`ForwardPassPerfModelConfig.encoder_parallel`): `vision.encoder_parallel: tp`
(SGLang's default) shards it over the tensor-parallel group so every rank
encodes every image, `dp` (`--mm-enable-dp-encoder`) replicates it and splits
the images. Timing, collectives, and the per-rank weights deducted from the KV
budget all follow that one setting, as does the embedding cache. Image geometry
follows the checkpoint's processor: each image becomes one or more encoder
sequences (Qwen3-VL: one, after `smart_resize` within the processor's
`min_pixels`/`max_pixels` budget; Llama 4: one per tile plus the global tile)
with their own patch, transformer, and merged token counts, and a placeholder
span that may add structural tokens. `traffic.source.images.min_pixels` and
`max_pixels` follow a served processor whose budget differs from the
checkpoint's. Repeated images (`identity: {pool: N}`) hit the cache; a cached
prompt prefix skips the encoder entirely, as it does upstream.

Frontend pools model the request path before the scheduler. Image stages fan
one job per image out to a pool and join; request stages run one job; jobs
sharing a resource are repriced by the stage's `concurrency_scale`; a stage on
the tokenizer-manager loop stalls the dispatch of arrivals and continuations
while it runs, but work already handed to the executor pools continues. The
Python frontend is modeled on its PIL/PNG CPU path; the Rust frontend uses the
same mechanics over its `mm_worker` pool.

Reports add the mean time to first token split by milestone: `mean_frontend_ms`
(arrival to leaving the frontend pools), `mean_scheduler_inbox_wait_ms`
(frontend exit to the scheduler draining the request), `mean_receive_to_admit_ms`
(received to selected, including receive preparation, TP sync, and selection),
`mean_prefill_elapsed_ms` (selected to the prompt's last forward finishing on
the device, across all its chunks), and `mean_result_observation_delay_ms`
(device completion to the scheduler observing the first token). These are
server-internal latencies to scheduler observation, not client-visible ones.

## Compatibility and scope

`host`, `frontend`, and `vision` require `backend: sglang`, `mode: aggregated`,
`pipeline: 1`, `attention_data: 1`, and default (AIC) timing. Image workloads
take either this path or the analytical encoder pool (`engine.workers.encoder`),
never both. On this path `scheduler.max_batched_tokens` becomes SGLang's
`chunked_prefill_size` and `scheduler.max_prefill_tokens` its EXTEND budget;
other SGLang predictions keep their existing behavior. `recommend` accepts the
same fields as fixed data; candidates search load, scheduling, and replicas,
and their saved prediction YAML reproduces the scored metrics through
`aisimulate predict`.

Not modeled: the GPU image processor path, video, PD or EPD disaggregation
with a host loop, speculative decoding, mixed image resolutions in one
workload, and the output side (detokenizer and tokenizer-manager work after the
scheduler observes a token): the reported latencies end at scheduler
observation. Every replica is assumed to have its own, sufficient CPU for its
frontend pools and scheduler thread; a search over TP and replicas on one
shared node has no common CPU budget in this model. Batch-level prepare,
launch, and result costs need measurements from a serving run with a GPU; a
profile lists unmeasured costs and predictions using it fail rather than treat
them as free.

Support matrix: the mechanics are exercised end to end for Qwen3-VL on the
packaged `h200_sxm` SGLang data with the Python frontend. Llama 4 tile geometry
and TP `2`/`4` encoder layouts are covered by contract tests on the geometry
and the compiled tower, not by end-to-end serving comparisons. The Rust frontend
and scheduler calibrators follow the pinned source but were not run against a
serving host in this tree; measured accuracy against SGLang remains to be
established for every combination.

## Behavior source and validation

The host loop, embedding cache, and frontend pools are independent
implementations of the observed behavior of sgl-project/sglang v0.5.19 at
immutable revision `0bcd822377da7b5718e674eaf9c870d349424dd1`:

- [Overlap scheduler loop](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/managers/scheduler.py)
  (`event_loop_overlap`, `run_batch`, `get_next_batch_to_run`)
- [Request receiver and result processing](https://github.com/sgl-project/sglang/tree/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/managers/scheduler_components)
  (`maybe_cache_unfinished_req`, `release_kv_cache`) and the
  [decode batch preparation](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/managers/schedule_batch.py)
  (`prepare_for_decode`, `filter_batch`)
- [Encoder parallelism default](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/server_args.py)
  (`mm_enable_dp_encoder`) and the
  [Qwen3-VL tower's TP resolution](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/models/qwen3_vl.py)
- [Per-image encoder batching](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/managers/mm_schedule.py)
  and the [multimodal embedding cache](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/mem_cache/multimodal_cache.py)
- [Tokenizer manager](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/managers/tokenizer_manager.py)
  and [multimodal processors](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/multimodal/processors/base_processor.py)

No upstream source or tests are copied; `THIRD_PARTY_NOTICES.md` records the
attribution and `tools/frontend/` the measurement patch. Deterministic Rust
tests cover the iteration arithmetic with fixed costs, mid-iteration arrivals,
ghost decode members and their KV slot, deferred prefix-cache commits,
cancellation of forwards in flight, input-ready preparation ahead of the
forward, the zero-cost equivalence with the legacy pass model, cache-miss
selection per chunk, frontend pool sharing, loop blocking and arrivals settling
past completions, and end-to-end stage timestamps through the ReplaySpec JSON
boundary. Python tests
exercise the public configuration gates, the lowering to engine arguments, host
profile matching and calibration lowering, the native predict path on the
packaged `h200_sxm` SGLang data, and the recommend-to-predict round trip.
