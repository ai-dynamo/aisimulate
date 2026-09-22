<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang VL host loop and frontend modeling

Native VL replay predicts a vision-language deployment end to end on one
aggregated SGLang worker, or on the prefill worker of a disaggregated
deployment (see [Disaggregated prefill](#disaggregated-prefill)): requests
cross the serving frontend, the scheduler receives and batches them at
iteration boundaries, the vision encoder runs inside the prefill forward, and
outputs become visible one scheduler iteration later. Configure it in an
`aisimulate predict` configuration by pointing the host-aware worker's
`host_profile` at a host cost table measured on the serving host, next to an
image workload:

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
      host_profile: {path: ./host-costs.json, frontend: python}
      vision: {cache_mib: 100, encoder_parallel: tp}
```

```bash
# Once per image workload, on the serving host; no GPU needed.
python -m aisimulate.vl.collect --config prediction.yaml --sglang-python /path/to/serving/venv/bin/python
aisimulate predict --config prediction.yaml
```

The only CPU costs the model carries are the frontend stages a request crosses
before the scheduler admits it, measured as black boxes at request level on the
real SGLang objects. Each stage is a pool of `workers` with a `service_ms` for one
request running alone and a `concurrency_scale` that reprices it while several
requests share the pool. The Python frontend is three stages: `process`, the
multimodal processor path (image decode, Hugging Face processor, layout) as a
pool the width of its IO executor; `send`, the tokenizer-manager loop's
synchronous shared-memory wrap and msgpack encode, one worker; and `receive`, the
scheduler's per-request preparation (shared-memory materialization, feature
hashing, placeholder padding), one worker. The Rust frontend is `process`, the
multimodal worker pool timed from the HTTP send to the scheduler-side drain,
followed by `receive`. The single-worker stages serialize requests but do not
stall the dispatch of arrivals as the real loop thread does; on the measured H20
grid that moved TTFT by at most a few percent at eight concurrent requests.

A host cost table belongs to one serving environment (CPU model, sglang
release, Python) and holds one row per measurement: model, frontend, feature
transport, and image shape (height, width, count, encoding, processor pixel
budget). The host name, thread count and sampling time of each collection stay in
the row's provenance. The feature transport follows sglang on one node: the
Python tokenizer manager always parks image features in POSIX shared memory, the
Rust workers keep them inline on a single rank and use shared memory once the
request is broadcast across tensor-parallel ranks, so a Rust table needs a row
per transport the tensor-parallel candidates use (`collect --tp`). A prediction
whose workload has no row fails closed and prints the `collect` command that adds
it; a failed lowering keeps the raw recording next to the table, and
`collect --recording` lowers it again. The stages a prediction or candidate runs
with carry the measurement they came from (`frontend.measured_for`), so a saved
recommendation refuses to price another model, image shape or transport with
stale constants, and the row's content digest travels in the prediction
metadata. Explicit `frontend.stages` remain available for hand-written tables
and tests. Text length is recorded with the row but not part of its key: on the
measured grid it moved the frontend stages by under one percent.

Lower-level Runner specifications set `rank.sglang.host_loop: true`,
`rank.sglang.vlm_cache_bytes`, `rank.frontend`, and `rank.vision: true`; Rust
callers set `SglangConfig::host_loop`, `EngineConfig::frontend`, and
`EngineConfig::vision`.

## Scheduling contract

Without `host_loop`, an SGLang pass is one GPU forward whose outputs are
visible when it ends. With `host_loop` (implied by `frontend` or
`host_profile`), one pass is one iteration of SGLang's overlap scheduler loop,
and the per-request report carries the stage timestamps `frontend_ready_ms`,
`scheduler_received_ms`, `selected_ms`, and `prefill_complete_ms` next to
`first_token_ms`:

| Step | Modeled time |
| --- | --- |
| Receive | Requests that left the frontend stages since the last iteration are drained at the iteration start. |
| Select | Batch selection; admissions are dated here. The scheduler thread itself is free: its batch-level CPU work is not charged (see the omitted costs below). |
| Forward | Starts when the batch is selected and the GPU is free; encoder time for cache-miss images precedes the language-model forward. With a free thread, EXTEND and DECODE launches share this timeline. |
| Result | The previous forward's outputs, terminals, prefix-cache commits (`maybe_cache_unfinished_req`), and KV release are observed at `max(selected, previous forward end)`. |

Consequences the tests pin: a request's first token is observed one iteration
after the forward that produced it; the radix cache learns a completed
prefill's prefix only when that forward is observed, so a request selected one
iteration later reuses nothing from it; a request that finished in forward `k`
still occupies a slot in batch `k+1` and, when that batch is a decode, is
allocated its row's KV slot, which is committed and cached with the request
when the finish is observed (only an unaligned tail page is freed); a request
that needs no output token completes when its prefill is observed, not when it
launches; a request retracted or cancelled while its forward is in flight
neither receives that forward's token nor keeps it, while the device work stays
charged; requests arriving during an iteration are received at the next one;
with a free scheduler thread the loop reproduces the legacy token times
exactly. Forward-pass telemetry records the forward's own device time, not the
iteration it was launched in.

The vision encoder runs once per prefill batch over the cache-miss images whose
placeholders overlap the chunk being computed, deduplicated by image identity,
and its outputs are kept in an LRU embedding cache of `vision.cache_mib`
(SGLang's `SGLANG_VLM_CACHE_SIZE_MB`, default 100 MiB). Image workloads without
an encoder pool take this path whether or not the host loop is on. The tower is
priced by the same canonical estimator as the language model
(`ForwardPassPerfModelConfig.encoder_parallel`): `vision.encoder_parallel: tp`
(SGLang's default) shards it over the tensor-parallel group so every rank
encodes every image, `dp` (`--mm-enable-dp-encoder`) replicates it and splits
the images; sglang implements `dp` for its Qwen-VL, GLM, InternVL, Kimi and
MiMo towers, and a Llama 4 configuration asking for it is rejected. Timing,
collectives, and the per-rank weights deducted from the KV budget all follow
that one setting, as does the embedding cache. Image geometry follows the
architecture's processor rules: each image becomes one or more encoder
sequences (Qwen3-VL: one, after `smart_resize` within the architecture's default
pixel budget; Llama 4: one per tile plus the global tile) with their own patch,
transformer, and merged token counts, and a placeholder span that may add
structural tokens. `traffic.source.images.min_pixels` and `max_pixels` override
that budget for a served processor whose limits differ. Repeated images
(`identity: {pool: N}`) hit the cache; a cached prompt prefix skips the encoder
entirely, as it does upstream.

Reports add the mean time to first token split by milestone: `mean_frontend_ms`
(arrival to leaving the frontend stages, receive preparation included),
`mean_scheduler_inbox_wait_ms` (frontend exit to the scheduler draining the
request), `mean_receive_to_admit_ms` (received to selected, the waiting-queue
time when the batch budget is full), `mean_prefill_elapsed_ms` (selected to the
prompt's last forward finishing on the device, across all its chunks), and
`mean_result_observation_delay_ms` (device completion to the scheduler
observing the first token). These are server-internal latencies to scheduler
observation, not client-visible ones, and they are reported with or without
per-request capture.

## Disaggregated prefill

In a disaggregated deployment the prefill worker hosts the same loop: put
`host_loop`, `frontend` or `host_profile`, and `vision` on `workers.prefill`
(`mode: disaggregated`, see `examples/cli/vl-predict-disaggregated.yaml`) and
leave `workers.decode` a plain worker. The modeled path follows SGLang's PD
prefill server: the request crosses the prefill rank's frontend stages, its
scheduler receives it at an iteration boundary, the extend runs with the vision
encoder, and the KV handoff to the decode rank starts when the next iteration
observes that forward, where SGLang sends the last KV chunk and the request
metadata while processing the batch result. A finished extend leaves the batch
at once (`disagg_prefill_inflight_queue` keeps it out of the next batch), so the
prefill rank never prices a ghost decode step. The decode rank keeps the
existing PD contract: it activates the request when the transfer completes and
its first decode step produces the first visible token.

Approximations specific to this path:

| Approximation | Reason |
| --- | --- |
| The decode rank's frontend is not modeled. SGLang's decode-side tokenizer manager repeats the image preprocessing, on another host and concurrently with the prefill rank's; the critical path is the prefill frontend, and the decode rank's registration is immediate, as for text PD. | Keeps the decode rank identical to text PD; the error is the decode rank's pickup latency, one decode iteration. |
| The decode rank does not deduct the resident vision tower from its KV pool (SGLang loads it in pure PD; `--language-model-only` is incompatible with PD). | The tower is a small fraction of HBM for the supported models, and the decode rank needs no vision oracle. |
| The decode rank runs without `host_loop`. | Its per-token interval is the same either way, and the one-iteration observation delay would compound the existing first-token contract, under which the first visible token comes from the decode rank's first forward rather than from SGLang's prebuilt batch. |
| KV transfer starts after the whole prompt is extended, as for text PD. | SGLang sends KV per chunk, overlapped with later chunks; the transfer model is unchanged. |

## Native encoder disaggregation

`engine.workers.encoder` with `mode: native` replays SGLang's encoder
disaggregation event by event instead of adding one batch latency to the mean
TTFT (`mode: analytical`, the default, is unchanged). Each of the `replicas`
encoder servers (`--encoder-only`) runs one serial loop, as `EncoderScheduler`
does: it takes the queued requests up to `batch_size` (the loop's cap,
`SGLANG_ENCODER_MAX_BATCH_SIZE`, default 8), runs the image processor and one
encoder forward over the batch, then pushes each request's embeddings to the
language rank. The push is asynchronous, so the instance takes its next batch
when the forward ends; requests arriving at one instant share a batch because
they are already queued when the loop collects it. A request reaches the
language worker only when its embeddings arrived, and its time to first token
counts the wait; the report adds `encoder_latency_ms` (mean arrival-to-delivery)
and, per request, `encoder_ready_ms`.

Service terms per request (`examples/cli/vl-predict-epd-native.yaml`):

| Term | Source |
| --- | --- |
| CPU preprocessing | The `process` stage of the encoder host's cost table (`host_profile`, Python frontend): the encoder servers run the same HF image processor as the tokenizer manager. A batch of `k` requests costs `k` times that stage (the processor works per image; the four parallel image loaders are not modeled). |
| Encoder forward | The AIC encoder model at the pool's `tensor` width for every batch size up to the cap, the same estimator the analytical mode uses for its single batch point. |
| Transfer | `visual_tokens x embedding bytes per token x language tensor-parallel width` at `transfer.bandwidth_gb_per_second`: `zmq_to_scheduler` fans the embeddings out to every rank. Requests are timed independently at that bandwidth, the same precision boundary as the prefill-to-decode KV handoff, with which it shares the bytes-over-bandwidth arithmetic. |

The language worker runs `--language-only`: it neither hosts the vision tower
nor prices image frontend stages (`vision`, `frontend`, and `host_profile` are
rejected on it), while `host_loop` still models its scheduler loop. Visual
placeholders occupy its prompt as they do on the analytical path. Not modeled:
`--enable-adaptive-dispatch-to-encoder` (single-image requests encoded locally,
off by default), the mooncake GPU-direct transfer and the encoder's prefix
embedding cache, and the resident tower weights that Qwen-VL still loads on a
language-only rank.

## Compatibility and scope

`host_loop`, `frontend`, and `vision` require `backend: sglang`, `pipeline: 1`,
`attention_data: 1`, and default (AIC) timing on the worker that hosts them:
`workers.aggregated` in `mode: aggregated` or `workers.prefill` in
`mode: disaggregated`. Image workloads
take either this path or an encoder pool (`engine.workers.encoder`, analytical
or native), never both, and `min_pixels`/`max_pixels` are honored only on this
path. On this
path `scheduler.max_batched_tokens` becomes SGLang's `chunked_prefill_size`;
other SGLang predictions keep their existing behavior. `recommend` accepts the
same fields as fixed data; candidates search load, scheduling, tensor
parallelism and replicas, and their saved prediction YAML reproduces the scored
metrics through `aisimulate predict`.

Not modeled: the GPU image processor path, video, speculative decoding, mixed image resolutions in one
workload, and the output side (detokenizer and tokenizer-manager work after the
scheduler observes a token): the reported latencies end at scheduler
observation. Every replica is assumed to have its own, sufficient CPU for its
frontend stages; a search over TP and replicas on one shared node has no common
CPU budget in this model.

Omitted CPU costs, by decision: the scheduler thread's batch-level work is not
priced. On a measured Qwen3-VL-8B serving host (SGLang 0.5.19, image shapes
480²-1024² × 1-16 images, text 128) the omitted terms were small next to
`receive`, which is kept as a frontend stage: pixel-value host-to-device copies
before the encoder about 2% of TTFT for the largest image batches, the eager
language-model kernel launch (about 37 ms per prefill batch) exposed only when
a prompt's forward is shorter than that (below roughly 300 tokens), the decode
step launch about 2 ms per step, and selection plus result processing about
1 ms per iteration. Keeping `receive` as a frontend stage rather than on the
scheduler thread means received requests are admitted as they finish preparing
instead of as one drained group; on the same grid the two placements agreed
within the measurement noise. Predictions are therefore optimistic for very
short prompts, for decode-step latency at small batch sizes, and for
tensor-parallel scheduler synchronization, which is not measured. HTTP parsing
and chat templating in the Python server are outside the measured stages
(3-10 ms per MB of request body); the Rust stage includes its HTTP receive and
tokenization, and the Rust `process` stage starts at the client's JSON encoding
of the request body. The collector measures the CPU image path only and must not
run next to a serving process on the same host; it times `receive` with torch's
intra-op pool at one thread, as the scheduler process runs it (`ModelRunner.load_model`),
since the default pool on the two cores the Rust server leaves the scheduler
turns a 25 MB shared-memory clone from about 12 ms into hundreds. `receive` runs
the placeholder padding pattern Qwen3-VL and Llama 4 use
(`MultiModalityDataPaddingPatternMultimodalTokens`); other VL families may pad
differently and have not been measured.

Support matrix: the mechanics are exercised end to end for Qwen3-VL on the
packaged `h200_sxm` SGLang data with hand-written stages; the collector's Python
and Rust paths were run on a serving host with SGLang 0.5.19 for Qwen3-VL-8B.
Llama 4 tile geometry and TP `2`/`4` encoder layouts are covered by contract
tests on the geometry and the compiled tower, not by end-to-end serving
comparisons.

## Behavior source and validation

The host loop, embedding cache, and frontend pools are independent
implementations of the observed behavior of sgl-project/sglang v0.5.19 at
immutable revision `0bcd822377da7b5718e674eaf9c870d349424dd1`:

- [Overlap scheduler loop](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/managers/scheduler.py)
  (`event_loop_overlap`, `run_batch`, `get_next_batch_to_run`)
- [Request receiver and result processing](https://github.com/sgl-project/sglang/tree/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/managers/scheduler_components)
  (`maybe_cache_unfinished_req`, `release_kv_cache`, the `is_retracted` skip) and the
  [decode batch preparation](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/managers/schedule_batch.py)
  (`prepare_for_decode`, `filter_batch`, `retract_decode`)
- [Encoder parallelism default](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/server_args.py)
  (`mm_enable_dp_encoder`) and the
  [Qwen3-VL tower's TP resolution](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/models/qwen3_vl.py)
- [Per-image encoder batching](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/managers/mm_schedule.py)
  and the [multimodal embedding cache](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/mem_cache/multimodal_cache.py)
- [Tokenizer manager](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/managers/tokenizer_manager.py),
  [multimodal processors](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/multimodal/processors/base_processor.py),
  and the [Rust multimodal workers' feature transport](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/rust_server/multimodal.py)
  (`_use_feature_shm`)

No upstream source or tests are copied; `THIRD_PARTY_NOTICES.md` records the
attribution. The collector (`python -m aisimulate.vl.collect`) instantiates the
pinned upstream frontend objects unmodified on the serving host and times them
from the outside. Deterministic Rust tests cover the iteration structure,
mid-iteration arrivals, ghost decode members and their KV slot, deferred
prefix-cache commits, retraction and cancellation of forwards in flight, the
equivalence with the legacy pass model, cache-miss selection per chunk, pool
ownership and arrivals settling past completions, the forward-duration
telemetry, the vision tower's DP share, and end-to-end stage timestamps
through the ReplaySpec JSON boundary. Python tests exercise the public
configuration gates, the lowering to engine arguments, host cost table lookup
and the miss message, recording lowering, the measured-workload check on saved
candidates, the native predict path on the packaged `h200_sxm` SGLang data
with and without per-request capture, and the recommend-to-predict round trip.
