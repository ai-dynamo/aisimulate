<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang VL host loop and frontend modeling

Native VL replay predicts a vision-language deployment end to end on one
aggregated SGLang worker: requests cross the serving frontend, the scheduler
receives and batches them at iteration boundaries, the vision encoder runs
inside the prefill forward, and outputs become visible one scheduler iteration
later. Configure it in an `aisimulate predict` configuration by enabling the
aggregated worker's `host_loop` and pointing `host_profile` at a host cost
table measured on the serving host, next to an image workload:

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
python -m aisimulate.vl.collect --table host-costs.json --model Qwen/Qwen3-VL-8B-Instruct \
  --frontend python --images 1024x1024x1 --encoding png --text-tokens 128 \
  --sglang-python /path/to/serving/venv/bin/python   # once per workload shape, no GPU needed
aisimulate predict --config prediction.yaml --capture-per-request
```

The only CPU costs the model carries are the frontend stages a request crosses
before the scheduler admits it, measured as black boxes at request level on the
real SGLang objects and stored in the table by serving host, SGLang revision,
model, frontend, and workload shape (image height, width, count, encoding,
processor pixel budget, text length). The Python frontend is three stages: the
multimodal processor path (image decode, Hugging Face processor, layout) as a
pool the width of the highest measured concurrency, the tokenizer-manager loop's
synchronous send continuation, and the scheduler's per-request receive
preparation (shared-memory materialization, feature hashing, placeholder
padding) as a single-worker pool. The Rust frontend is one pool stage the width
of its multimodal worker pool, timed from the HTTP send to the scheduler-side
drain. Every stage cost is an affine function of the request (`const_ms`,
`per_request_ms`, `per_image_ms`, `per_ktoken_ms`, `per_mib_ms`); the collector
fills the constant and a per-concurrency `concurrency_scale`. A table row is a
constant of the shape it was sampled with: a prediction whose shape has no row
fails closed and prints the exact `collect` command that adds it, and the row's
content digest travels with every prediction and recommendation that used it.
`recommend` resolves the row once and writes the stages into each saved
candidate, so a saved candidate never depends on a table file that can change
afterwards. Explicit `frontend.stages` (`resource: pool | tm_loop`, `workers`,
`cost`, `concurrency_scale`) remain available for hand-written tables and tests.

Lower-level Runner specifications set `rank.sglang.host_loop: true`,
`rank.sglang.vlm_cache_bytes`, `rank.frontend`, and `rank.vision: true`; Rust
callers set `SglangConfig::host_loop`, `EngineConfig::frontend`, and
`EngineConfig::vision`.

## Scheduling contract

Without `host_loop`, an SGLang pass is one GPU forward whose outputs are
visible when it ends. With `host_loop`, one pass is one iteration of SGLang's
overlap scheduler loop, and the per-request report carries the stage timestamps
`frontend_ready_ms`, `scheduler_received_ms`, `selected_ms`, and
`prefill_complete_ms` next to `first_token_ms`:

| Step | Modeled time |
| --- | --- |
| Receive | Requests that left the frontend stages since the last iteration are drained at the iteration start. |
| Select | Batch selection; admissions are dated here. The scheduler thread itself is free: its batch-level CPU work is not charged (see the omitted costs below). |
| Launch | An EXTEND launch starts at selection; a DECODE launch first waits for the previous forward, as SGLang's position update synchronizes the stream. |
| Forward | Starts when the launch begins and the GPU is free; encoder time for cache-miss images precedes the language-model forward. |
| Result | The previous forward's outputs, terminals, prefix-cache commits (`maybe_cache_unfinished_req`), and KV release are observed at `max(launch, previous forward end)`. |

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
requests arriving during an iteration are received at the next one; with a free
scheduler thread the loop reproduces the legacy token times exactly.

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
(arrival to leaving the frontend stages, receive preparation included),
`mean_scheduler_inbox_wait_ms` (frontend exit to the scheduler draining the
request), `mean_receive_to_admit_ms` (received to selected; zero with a free
scheduler thread), `mean_prefill_elapsed_ms` (selected to the prompt's last
forward finishing on the device, across all its chunks), and
`mean_result_observation_delay_ms` (device completion to the scheduler
observing the first token). These are server-internal latencies to scheduler
observation, not client-visible ones.

## Compatibility and scope

`host_loop`, `frontend`, and `vision` require `backend: sglang`, `mode: aggregated`,
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
frontend stages; a search over TP and replicas on one shared node has no common
CPU budget in this model.

Omitted CPU costs, by decision: the scheduler thread's batch-level work is not
priced. On a Qwen3-VL-8B H20 serving host (SGLang 0.5.19, image shapes
480²-1024² × 1-16 images, text 128) the omitted terms were small next to
`receive`, which is kept as a frontend stage: pixel-value host-to-device copies
before the encoder about 2% of TTFT for the largest image batches, the eager
language-model kernel launch (about 37 ms per prefill batch) exposed only when
a prompt's forward is shorter than that (below roughly 300 tokens), the decode
step launch about 2 ms per step, and selection plus result processing about
1 ms per iteration. Predictions are therefore optimistic for very short prompts,
for decode-step latency at small batch sizes, and for tensor-parallel scheduler
synchronization, which is not measured. HTTP parsing and chat templating in the
Python server are outside the measured stages (3-10 ms per MB of request body);
the Rust stage includes its HTTP receive and tokenization.

Support matrix: the mechanics are exercised end to end for Qwen3-VL on the
packaged `h200_sxm` SGLang data with hand-written stages; the collector's Python
and Rust paths were run on an H20 serving host with SGLang 0.5.19 for Qwen3-VL-8B.
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
attribution. The collector (`python -m aisimulate.vl.collect`) instantiates the
pinned upstream frontend objects unmodified on the serving host and times them
from the outside. Deterministic Rust tests cover the iteration structure,
mid-iteration arrivals, ghost decode members and their KV slot, deferred
prefix-cache commits, cancellation of forwards in flight, the equivalence with
the legacy pass model, cache-miss selection per chunk, frontend pool ownership,
loop blocking and arrivals settling past completions, and end-to-end stage
timestamps through the ReplaySpec JSON boundary. Python tests exercise the
public configuration gates, the lowering to engine arguments, host cost table
lookup and the miss message, recording lowering, the native predict path on the
packaged `h200_sxm` SGLang data, and the recommend-to-predict round trip.
