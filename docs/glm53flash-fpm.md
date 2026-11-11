<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.3-Flash FPM

Whole-forward (FPM) profiles for GLM-5.3-Flash on GB300, GB200, B300, H200
and H100, collected with engine-native timing on the default serving
configuration (prefix/radix cache on). Text only, pure TP, DP = PP = EP = 1.

| System | vLLM stock `0.31.0` (`db9527a4`) | SGLang `0.5.20` (`94602c9c`) | KV |
|---|---|---|---|
| GB300 | FP8 TP2/TP4, NVFP4 TP1/TP2/TP4 | FP8 TP2/TP4, NVFP4 TP2/TP4 | FP8 |
| GB200 | FP8 TP4, NVFP4 TP2 | FP8 TP4, NVFP4 TP2/TP4 | FP8 |
| B300 SXM | FP8 TP2, FP8 TP4 (development), NVFP4 TP1/TP2/TP4 | FP8 TP2/TP4, NVFP4 TP2/TP4 | FP8 |
| H200 | FP8 TP4/TP8 | FP8 TP4/TP8 | BF16 |
| H100 | FP8 TP8 | FP8 TP8 | BF16 |

Checkpoints: `zai-org/GLM-5.3-Flash@eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`
and `nvidia/GLM-5.3-Flash-NVFP4@09b04e5e74bca08ca8549fc736d4cdd8624bfde3`.
FP8 TP1 does not fit on one GPU. Hopper profiles use BF16 KV for both backends.
The 15 published vLLM profiles are data from stock vLLM 0.31.0. New vLLM
collections run the [producer](#vllm-producer) on vLLM 0.33.0 with the Dynamo
scheduler file of the `release/1.7.0` branch that `runtime-source-sha256.json`
pins.

The tables (`fpm_forward_perf.{parquet,metadata.json}`) are not stored in this
repository. They are published in `nvidia/aisimulate-fpm-dataset`
([GB300 discussion 15](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/discussions/15),
[experimental B300 discussion 19](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/discussions/19),
[GB200/B300/H200/H100 discussion 20](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/discussions/20)).
[`hf_dataset.json`](../python/aisimulate/src/aisimulate_core/systems/profiles/glm53flash_fpm/hf_dataset.json)
pins the 29 profiles to immutable dataset revision
`dbfcdd9eb62ea733efbc3f0cbd78277f2960061a` and records each file's SHA-256.
Profile keys are `<system>-<backend>-<fp8|nvfp4>-tp<N>` with system `gb300`,
`gb200`, `b300`, `h200` or `h100`, for example `gb300-vllm-nvfp4-tp1` or
`h100-sglang-fp8-tp8`. Pass the profile identity's `kv_cache_dtype` and
`fmha_quant_mode` (bfloat16 on H200/H100). Materialize one with
`python -m aisimulate_core.sdk.fpm_dataset <manifest> <key>` (or
`materialize_fpm_profile`) and pass the printed root as `systems_paths`, with
`estimation_mode="fpm_interpolation"` and the profile identity's
`backend_version`.

The experimental B300 key is `b300-vllm-fp8-tp4`. Default materialization is blocked; use `--allow-unqualified` only for diagnostic reproduction. It uses system `b300_sxm`, FP8 KV, TP4 and vLLM `0.31.0`. Its expanded table has 427 prefill and 175 decode measurements. Its collection covers batch sizes up to 32 and per-request contexts up to 131,072 tokens. Validation covers align-4 forward-pass inputs only; end-to-end serving, cache reuse, 1M context and a supported DynoSim dependency pair remain unqualified.

## vLLM producer

`python/aisimulate/collector/fpm_forward/runtime/glm53flash/` contains the
producer. It is staged next to `fpm_exec.sh` and `preflight.py` and runs with
the pinned Dynamo `InstrumentedScheduler` overlay on `PYTHONPATH`. The first
part of the overlay below is the `ai-dynamo 1.7.0.dev20261109` build, whose
scheduler file equals `ai-dynamo/dynamo@9abd2ef0d9` on `release/1.7.0`, and
`runtime-source-sha256.json` pins that file. It also pins
`dynamo/vllm/worker_factory.py`, whose merge checks every measurement record
before the engine writes its merged artifact.

The producer runs on exactly stock vLLM 0.33.0 (`4ba787ec`) with the Dynamo
scheduler file of the `release/1.7.0` branch that `runtime-source-sha256.json`
pins. It refuses every other vLLM version, a patch release such as 0.33.1
included, before the scheduler loads.

- `sitecustomize.py` checks the installed vLLM/Dynamo sources against
  `runtime-source-sha256.json` before the scheduler loads. It then replaces
  `InstrumentedScheduler` with `Glm53FlashPrefixSeedScheduler` from
  `glm53flash_prefix_scheduler.py`.
- Timing is Dynamo's unchanged FPM `wall_time`. Prefill runs from the end of
  `schedule()` to the arrival of `update_from_output`. Steady decode is the
  interval between consecutive output arrivals.
- The adapter only adds requests to the native waiting queue. vLLM's own
  scheduler schedules the seed, warmup and measured requests, so prefix-cache
  lookup, Mamba `align` chunking, KDA/conv, MLA and IndexPool state all run
  their serving code paths.
- Stock v0.31.0 does not complete IndexPool entries at a cached or chunked
  prefill start that is not a multiple of 4, and the sources do not show that
  v0.33.0 removed the defect. Inputs are therefore built on the 4-token grid,
  and the producer fails the run on any unaligned prefill chunk start. A decode
  point keeps its true context `c`: the prompt has `c - 1` tokens and the
  cached seed `(c - 2) // hash_block * hash_block`, where `hash_block` is the
  hash block size of the engine.
- Each point runs 5 warmups and 10 measurements. The table stores the median
  of measurements 2 to 10, which is Dynamo's `adjacent_upper_median` estimate.
  Measurement 1 is a valid measured repetition that this estimate leaves out.
  The merge of Dynamo 1.7 accepts no other median of several samples. Each raw
  sample of the record is numbered by its position. The first repetition of a
  prefill point with a prefix is rejected. It runs before the warmups and
  counts as neither a warmup nor a measurement, and the later repetitions are
  accepted: 5 warmups, then 10 measurements. With the default
  `prefix_cache_retention_interval` of 0, the seed does not keep the Mamba
  state at the end of the prefix. vLLM 0.33.0 stops the first repetition there,
  at the shared-prefix junction, and caches the state. The later repetitions
  hit the whole prefix. One rejected repetition for each such prefill point is
  the expected result. A decode point has none.
- Request tokens come from `collector/glm53flash_attention_tokens.py`
  (seed 53, ordinary BPE vocabulary of the pinned tokenizer).
- A pinned revision needs the configuration of its own checkpoint. The
  producer accepts any other revision whose configuration equals the
  configuration of a pinned checkpoint. `DYN_FPM_TOKENIZER_REVISION` is the
  40-hex revision that was loaded. A folder launch reads its folder, and a
  launch by Hub ID reads that revision's snapshot in the local Hugging Face
  cache. The artifact records the loaded revision and the pinned checkpoint
  that matched.
- The producer refuses a run whose vLLM model revision is set and differs from
  `DYN_FPM_TOKENIZER_REVISION`. A launch by Hub ID therefore passes
  `--revision` with the same value. Without it, vLLM loads the default branch.
- `--max-model-len` must be at least the measured context plus 7 internal
  positions, and a larger value is accepted. `DYN_FPM_GLM53FLASH_MEASURED_CONTEXT`
  sets the measured context (131,072 at most). The producer refuses a point
  whose per-request length exceeds the measured context.
- The producer refuses a point of more than 32 requests, the batch limit that
  its artifact states.
- Before a decode point decodes, each request prefills its prompt beyond its
  cached seed. The producer refuses a decode point of two or more requests whose
  suffixes together exceed `--max-num-batched-tokens`. If they exceed it, vLLM
  prefills them over several steps. The first decode step then does not hold
  every request, and the producer rejects every repetition. One request can
  prefill its suffix over several steps. No other request decodes before it,
  and the producer records no step with a prefill request.
- With prefix caching, vLLM runs GLM-5.3-Flash in Mamba `align` mode. vLLM
  0.33.0 stops a prefill chunk at the last block boundary before the prompt
  end, `(end - 1) // block * block`. After a prefix that ends inside a block, it
  also stops the chunk at the next block boundary. A prefill point whose chunk
  holds such a stop runs in more than one step, and the producer accepts one
  step per repetition. So the producer refuses the point before it seeds
  anything. Without a prefill checkpoint block, new tokens of at most one block
  after a prefix of whole blocks run in one step.
- With the FlashKDA prefill backend, each KDA state group holds one prefill
  checkpoint block with an alignment of 16 tokens. vLLM 0.33.0 then saves the
  checkpoint inside the last chunk and takes neither stop when the chunk starts
  on a hash block boundary, ends more than one block after its start, and its
  last block boundary before the prompt end lies a multiple of 16 tokens after
  its start. If the block is a multiple of 16 tokens, every prefill after a
  prefix of whole blocks then runs in one step. The Triton backend gives no
  checkpoint block. `_glm_align_stop` in the producer holds the exact rule,
  and the artifact records the checkpoint blocks and their alignment in
  `serving_config.mamba_prefill_checkpoint`.

The collector reads the artifact of a GLM-5.3-Flash cell only as an artifact of
this producer. The cell selects these rules, not the artifact:

- The `producer` record must equal the record that this producer writes,
  field by field: the instrumentation revision `54960177`, the scheduler
  revision `9abd2ef0d9` and the `ai-dynamo 1.7.0.dev20261109` distribution,
  the sources of the real-seed and queued-request ports, vLLM `0.33.0`, the
  reviewed vLLM revision `4ba787ec`, the digests of the
  `runtime-source-sha256.json` and `glm53flash_prefix_scheduler.py` files of
  the checkout that reads the artifact, the context policy version and the
  repetition counts above. A missing, changed or added field is refused.
- The vLLM version of the record must equal the runtime that the Collector
  provenance of the launch names.
- The measurement protocol must be present. It must hold the producer's record
  of 5 warmups, 10 measurements and their reduction, the `seeded_tokens`
  content and Dynamo's execution evidence contract.
- Every point must hold the 10 raw samples and the `adjacent_upper_median`
  estimate above. The warm-up history must be recorded, and each raw sample
  must have its timing start and an observed CUDA graph dispatch.
- A rank artifact of another model that carries the producer's protocol record
  is refused.

The first two rules apply wherever the collector reads the artifact,
publication included. Readiness and repeatability assessment apply the other
three. Publication (`aggregate_cell`) does not apply them. The warm-up and raw
sample checks need measurement evidence version 2, the default.

The two digests are those of the checkout that reads the artifact. A campaign
is therefore readable only by a collector whose `glm53flash_prefix_scheduler.py`
and `runtime-source-sha256.json` have the bytes of the files that the launch
staged. After any change of either file, a comment included, the collector
refuses earlier GLM-5.3-Flash campaigns at `producer.overlay_sha256` or
`producer.runtime_source_manifest_sha256`. This holds for every step that reads
the raw artifacts again: readiness, repeatability assessment, publication and
finalization, serving validation and the runtime memory import. Read such a
campaign with the collector revision that collected it, which
`collection-plan.json` records as `aic_revision`, or collect a new smoke and a
new campaign. A published table keeps loading, because loading reads the table
and not the raw artifacts. Artifacts of other models are not affected.

Engine arguments, beyond the usual Dynamo benchmark flags:
`--enable-prefix-caching --kv-cache-dtype fp8
--no-async-scheduling --language-model-only --cudagraph-metrics
--max-num-batched-tokens 8192 --max-num-seqs 32 --benchmark-warmup-iterations 0`.
Without `--prefix-match-unit`, the hash block is the native block of the
engine. The published vLLM 0.31.0 profiles used `--prefix-match-unit 4`, and
the producer refuses that launch on vLLM 0.33.0, because vLLM 0.33.0 turns on
partial hash hits (`mamba_partial_cache_hit`) for a hash block below the Mamba
block, and the `align` rule of the producer does not model them.

Environment:

| Variable | Value |
|---|---|
| `DYN_FPM_GLM53FLASH_REAL_KV` | `1` |
| `DYN_FPM_GLM53FLASH_PREFIX_SEED` | `1` |
| `DYN_FPM_GLM53FLASH_MEASURED_CONTEXT` | `131072` |
| `DYN_FPM_TOKENIZER_REVISION` | checkpoint revision |
| `DYN_FPM_GLM53FLASH_DECODE_CONTEXTS` | optional `[[batch, total, [contexts...]], ...]` with the per-request decode split |

## Guided onboarding collection

`aisimulate onboard collect-fpm` runs the producer above as the runtime adapter
for GLM-5.3-Flash collection. The
[self-service guide](perf-model/fpm-self-service/implementation.md) describes
the workflow. The collector renders the producer launch for each prefill and
decode phase of a `Glm5NextForConditionalGeneration` collection, formal or
smoke. A runtime probe of the same deployment keeps Dynamo's generic grid and
does not run the producer.

Both phases get `--language-model-only`, `--cudagraph-metrics`,
`--no-async-scheduling`, `--benchmark-warmup-iterations 0`,
`--benchmark-points-file` for the staged point list and `--revision` with the
value of `DYN_FPM_TOKENIZER_REVISION`. The request sets
`--max-model-len`, `--max-num-batched-tokens` and `--max-num-seqs`. The
collector passes no `--prefix-match-unit`. It also sets these variables:

| Variable | Value |
|---|---|
| `DYN_FPM_GLM53FLASH_REAL_KV` | `1` |
| `DYN_FPM_GLM53FLASH_PREFIX_SEED` | `1` |
| `DYN_FPM_BENCH_COLLECT_IMBALANCED` | `1` |
| `DYN_FPM_TOKENIZER_REVISION` | launch revision of the accepted probe, else the model revision of the accepted profile |
| `DYN_FPM_GLM53FLASH_MEASURED_CONTEXT` | the smaller of `131072` and the runtime context minus 7 |
| `PYTHONPATH` | `/tmp/fpm-bench:/opt/glm53flash-dynamo` |

`DYN_FPM_GLM53FLASH_PREFIX_MATCH_UNIT` stays unset, so the native hash block of
the engine is the prefix unit. The planner refuses a hash block that is not a
multiple of 4. The producer still checks every prefill chunk start. A decode
point seeds `(c - 2) // hash_block * hash_block` tokens, as in the launch
above.

The container needs one overlay directory at `/opt/glm53flash-dynamo`. A Slurm
run can mount it with `--container-mount`. The collector uses that path unless
the deployment environment declares its own `PYTHONPATH`. The overlay is a
composite of two parts. The first part is the `dynamo` files of the
`ai-dynamo 1.7.0.dev20261109` build, the Dynamo 1.7 scheduler and merge whose
files `runtime-source-sha256.json` pins. The second part is the `collector`,
`aisimulate_core` and `aisimulate` packages of the AISimulate checkout that
plans the collection. The producer imports them. Run the runtime probe with the
deployment of the collection, the overlay mount included, because the collector
refuses a collection deployment that differs from the accepted probe.

The request must follow these rules:

- `collection.warmup_iterations` is 0. The producer runs its own warm-ups and
  refuses a global warm-up. Set it before the runtime probe, because the
  collector refuses a collection whose count differs from the accepted probe.
- The runtime context is at least the measured context plus 7. The runtime
  context is `search.context_length`, and a larger value is accepted. The
  collector sets the measured context to the smaller of 131,072 and the runtime
  context minus 7.
- The token budget `max_num_tokens` is a multiple of 4.
- `identity.model_revision` is the 40-hex revision of the loaded checkpoint. The
  revision rule above applies, with its checkpoint lookup for a folder launch
  and for a launch by Hub ID.

Plan the collection from the accepted runtime probe. The planner generates the
point list from `max_num_tokens`, `max_batch_size` and `context_length` of the
resolved request. When the request leaves `max_num_tokens` or `max_batch_size`
unset, that limit takes the accepted profile's bound. The planner takes
`hash_block_size_tokens` and the cache capacity from the accepted profile, and
it refuses a profile without the hash block. The planner freezes the list into
the collection plan, so the plan hash covers it. The producer divides each
total evenly. Every request of a point therefore has the same new tokens,
prefix and decode context. These are the axes:

- Prefill new tokens per request: the powers of two from 4 to the token budget,
  every whole hash block up to the budget, and the budget itself. The list caps
  the budget at 8,192. The list keeps a length only if vLLM's `align` mode runs
  it in one step after each planned prefix, as the producer rules above say.
  Without a prefill checkpoint block, that leaves no length above one hash
  block. With the FlashKDA checkpoint block and a hash block that is a multiple
  of 16 tokens, it keeps every length. At default serving the hash block is the
  Mamba block.
- Prefill requests per batch: the powers of two up to the smallest of 32,
  `max_batch_size` and the number of requests whose new tokens fit the budget,
  and that smallest value itself.
- Prefill prefix per request, in whole hash blocks: 0, the powers of two, and
  the largest number that keeps prefix and new tokens within the measured
  context.
- Decode context per request: the powers of two from 4 to the measured context,
  and the measured context itself.
- Decode requests per batch: the powers of two up to the smallest of 32,
  `max_batch_size` and the number of requests whose suffixes fit the budget,
  and that smallest value itself. Here the budget is the whole token budget.
  The 8,192 cap applies to prefill only. The suffix of a request is the part
  of its prompt beyond its seed, `(c - 1) - (c - 2) // hash_block * hash_block`
  tokens. If the suffixes of a batch fit the budget together, vLLM prefills
  them in one step, and the first decode step holds every request. The axis
  always holds a batch of one request. If its suffix exceeds the budget, vLLM
  prefills it over several steps, and no other request decodes before it.

The planner also drops a point that does not fit its estimate of the accepted
cache capacity. It counts the points that each rule drops: the align rule for
prefill, the token budget for decode, and the cache capacity for both. Each
rule counts against the candidates that it sees. The decode candidates at a
context are the batches without the token budget and the largest batch whose
suffixes fit the budget. The planner logs the counts in one line. Guided
onboarding sets no log handler, so the collection plan also records them in
`point_generation.dropped_points`. The planner refuses a plan with no prefill
point or no decode point. The estimate counts the attention blocks of each
request, one block in each state group and one more block in each state group
with a prefill checkpoint block. A request with a prefix hit can hold more than
one block in a recurrent-state group, for example the cached block of its
prefix and the block of its running state. So a point near the capacity limit
can still fail when it runs. At TP4 with 2,176-token hash
blocks, 8,192 tokens, 256 sequences and a 1,048,576-token context, a FlashKDA
launch (five state groups, four of them KDA groups with a checkpoint block) has
496 prefill points and 81 decode points. The align rule drops none of 496
prefill candidates, and the token budget drops 16 of 97 decode candidates.
Without a prefill checkpoint block, the align rule drops 40 prefill candidates,
and the list has 456 prefill points.
The largest decode batch is 32 at contexts up to 256 and at 32,768 and 65,536.
It is 16 at 512 and 131,072, 8 at 1,024, 4 at 2,048 to 8,192, and 7 at 16,384.
By the estimate, the FlashKDA list needs an accepted cache of at least 2,241
pool blocks (the profile's `kv_cache_bytes` divided by its largest
`page_size_bytes`), and the list without a checkpoint block needs 2,113.

The smoke measures two points of this list for each phase that the plan
collects. If the list holds the third prefill point below, the smoke also
measures it. An aggregated request collects both phases, so its smoke measures
four or five points. A prefill or decode role collects one phase, so its smoke
measures the points of that phase. The smoke takes no point of the other phase.
The smoke runs under the plan hash of formal collection and keeps its own
campaign directory. The prefill points are one request with 64 new tokens, once
without a prefix and once after one hash block of prefix. The third prefill
point is one request with two hash blocks of new tokens after one hash block of
prefix. The list holds it only for a launch with a prefill checkpoint block.
Its chunk holds a block boundary before its end, and vLLM's `align` mode saves
the checkpoint inside the chunk instead of stopping there. So the smoke shows
that such a point runs in one step. An aggregated smoke therefore measures
four points when the launch has no prefill checkpoint block, and five points
with the FlashKDA backend, as for the TP4 list above. The decode points are one
request and two requests at the smallest context above one hash block where
the list holds both. The token
budget can leave a smaller context with one request only. If no context above
one hash block holds two requests, the smoke takes one request at each of the
two smallest contexts above the block. A `max_batch_size` of 1 or a cache with
room for one request causes this. So does a budget too small for two suffixes
at each context above the block.
A direct `collector --fpm` call can pass its own manifest with
`--fpm-benchmark-points-file`. The collector refuses that file together with
`--smoke`.

Cached prefill rows and decode rows publish the regime `real_kv`, the only
regime that the consumer accepts for them. A prefill row without a prefix
publishes `n/a`. The native artifacts keep the `real_prefix` stamp for cached
prefill.

## SGLang client

`python/aisimulate/collector/fpm_forward/sglang_native_fpm.py` launches stock
`sglang.launch_server --enable-forward-pass-metrics` and reads the native
rank-0 `ForwardPassMetrics` ZMQ stream. `wall_time` is the native DeviceTimer
forward interval.

- Each geometry is built from real requests: a warm request per prefix, then B
  requests that hit the cached prefix. The radix cache is flushed per point.
- The client does no work while a round is in flight.
- The measured decode step is attributed by geometry, not by message order.
- Request tokens come from the same seeded generator as the vLLM producer.
  The client checks the model's `tokenizer.json` against the pinned hash
  first. The run metadata (`input_provenance`) records the generator spec and
  the token-stream hash.

Usage, with `python/aisimulate` on `PYTHONPATH`:
`python3 -m collector.fpm_forward.sglang_native_fpm --model <dir>
--model-revision <rev> --tp {1,2,4,8} --quant {fp8,nvfp4} --mem-fraction <f>
--phase {prefill,decode} --points <manifest.json> --out <dir>
[--kv-cache-dtype {fp8_e4m3,bfloat16}] [--port <p>] [--ipc <name>]`.

- `--kv-cache-dtype` defaults to `fp8_e4m3`. The client checks that the
  server resolved the requested dtype and records it in
  `serving_config.kv_cache_dtype` (`fp8` or `bfloat16`). BF16 KV is a distinct
  identity and is never mixed with FP8-KV rows.
- The resolved attention, MoE and other `*_backend` server arguments are
  recorded in `serving_config.resolved_server_args` and are not asserted. The
  resolved page size must equal the 64-token hit unit of the request plans;
  otherwise the run fails at startup (`failed.json` lists the failed checks).
- Servers packed on one node need distinct `--port` values and GPUs
  (`CUDA_VISIBLE_DEVICES`). The default IPC name includes the run id and port.

The published SGLang 0.5.20 tables were collected with an earlier client
revision that took its request tokens from a text corpus. That input
provenance is recorded in the HF dataset metadata. Future collections use the
seeded generator.

## Consumer

`crates/core/src/perfmodel/perf_database/fpm_forward.rs` admits two schema-7
median policies. Each policy is bound to its backend, state protocol and timing
boundary (see [whole-forward](perf-model/methods/whole-forward.md)):

| Backend | `measurement_policy` | `state_protocol` | `timing_boundary` |
|---|---|---|---|
| vLLM | `vllm_native_fpm_prefix_seed_median_v1` | `glm53flash_prefix_cache_real_seed_v1` | `dynamo_vllm_instrumented_scheduler_fpm_wall_time` |
| SGLang | `sglang_native_fpm_radix_seed_median_v1` | `glm53flash_sglang_radix_real_seed_v1` | `sglang_native_fpm_rank0_device_timer_wall_time` |

Align-4 constrains collection only. The consumer answers queries for any
prefix length.

The pinned checkpoints are listed in `DefaultHFModels`, so model configs
resolve offline from the bundled copies.

## Systems and parallelism

The producer and the SGLang client have no system gate and accept pure TP1, TP2,
TP4 and TP8 (DP = PP = EP = 1). Whether a deployment fits a system (for example
H100, H200, B300, GB200 or GB300) is decided when the runtime loads weights and
allocates KV; a deployment that does not fit fails observably. FP8 TP1 does not
fit on one GPU of any of these systems. Profile keys in `hf_dataset.json`
follow `<system>-<backend>-<fp8|nvfp4>-tp<N>`.
