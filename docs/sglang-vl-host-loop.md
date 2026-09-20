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
        launch_extend: {const_ms: 6.0, per_ktoken_ms: 0.5}
        launch_vision: {const_ms: 3.0, per_image_ms: 1.5, per_mib_ms: 0.05}
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
      vision: {cache_mib: 100}
```

```bash
aisimulate predict --config prediction.yaml --capture-per-request
```

Every cost is an affine function of the work it is applied to: a constant plus
per-request, per-image, per-thousand-token, and per-MiB terms. The tables are
measured data, not tuning knobs: `host_profile: {path, frontend}` takes them
from a profile written by `python -m aisimulate.vl.calibrate` on a serving host
(see `tools/frontend/README.md` for the Rust frontend) and fails closed when the
profile was measured for another model, frontend, image encoding, or SGLang
revision, or lacks a cost the deployment needs. `host_profile.on_missing:
calibrate` samples the workload into `path` when the file does not exist.

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
| Launch | `launch_extend` (plus `launch_vision` when the batch encodes cache-miss images) or `launch_decode`. A DECODE launch first waits for the previous forward, as SGLang's position update synchronizes the stream. |
| Forward | Starts when the launch begins and the GPU is free; encoder time for cache-miss images precedes the language-model forward; a forward cannot end before its launch does. |
| Result | The previous forward's outputs, terminals, and KV release are processed after this launch, at `max(launch end, previous forward end) + result`. |

Consequences the tests pin: a request's first token is observed one iteration
after the forward that produced it; a request that finished in forward `k`
still occupies a slot in batch `k+1`; requests arriving during an iteration are
received at the next one; an all-zero table reproduces the legacy token times
exactly.

The vision encoder runs once per prefill batch over the cache-miss images whose
placeholders overlap the chunk being computed, deduplicated by image identity,
and its outputs are kept in an LRU embedding cache of `vision.cache_mib`
(SGLang's `SGLANG_VLM_CACHE_SIZE_MB`, default 100 MiB). The cache and the
encoder weights are deducted from the KV budget of a worker that hosts the
tower. Repeated images (`identity: {pool: N}`) hit the cache; a cached prompt
prefix skips the encoder entirely, as it does upstream.

Frontend pools model the request path before the scheduler. Image stages fan
one job per image out to a pool and join; request stages run one job; jobs
sharing a resource are repriced by the stage's `concurrency_scale`; a stage on
the tokenizer-manager loop stalls the dispatch of arrivals and continuations
while it runs, but work already handed to the executor pools continues. The
Python frontend is modeled on its PIL/PNG CPU path; the Rust frontend uses the
same mechanics over its `mm_worker` pool.

Reports add the mean time to first token split by stage:
`mean_frontend_ms`, `mean_scheduler_inbox_wait_ms`, `mean_receive_to_admit_ms`,
`mean_prefill_elapsed_ms`, and `mean_result_observation_delay_ms`.

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
workload, per-token detokenizer costs (fold them into `result`), and processor
`min_pixels`/`max_pixels` rescaling (image geometry follows the checkpoint's
patch and merge sizes, shared with the encoder phase). Batch-level launch and
result costs need measurements from a serving run with a GPU; a profile lists
unmeasured costs and predictions using it fail rather than treat them as free.

## Behavior source and validation

The host loop, embedding cache, and frontend pools are independent
implementations of the observed behavior of sgl-project/sglang v0.5.19 at
immutable revision `0bcd822377da7b5718e674eaf9c870d349424dd1`:

- [Overlap scheduler loop](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/managers/scheduler.py)
  (`event_loop_overlap`, `run_batch`, `get_next_batch_to_run`)
- [Request receiver and result processing](https://github.com/sgl-project/sglang/tree/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/managers/scheduler_components)
- [Per-image encoder batching](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/managers/mm_schedule.py)
  and the [multimodal embedding cache](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/mem_cache/multimodal_cache.py)
- [Tokenizer manager](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/managers/tokenizer_manager.py)
  and [multimodal processors](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/multimodal/processors/base_processor.py)

No upstream source or tests are copied; `THIRD_PARTY_NOTICES.md` records the
attribution and `tools/frontend/` the measurement patch. Deterministic Rust
tests cover the iteration arithmetic with fixed costs, mid-iteration arrivals,
ghost decode members, the zero-cost equivalence with the legacy pass model,
cache-miss selection per chunk, frontend pool sharing and loop blocking, and
end-to-end stage timestamps through the ReplaySpec JSON boundary. Python tests
exercise the public configuration gates, the lowering to engine arguments, host
profile matching and calibration lowering, the native predict path on the
packaged `h200_sxm` SGLang data, and the recommend-to-predict round trip.
