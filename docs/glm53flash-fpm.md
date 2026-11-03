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
part of the overlay below is the `ai-dynamo 1.6.0.dev20261010` build, whose
scheduler file equals `ai-dynamo/dynamo@395f0240`, and
`runtime-source-sha256.json` pins that file.

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
  prefill start that is not a multiple of 4. Inputs are therefore built on the
  4-token grid, and the producer fails the run on any unaligned prefill chunk
  start. A decode point keeps its true context `c`: the prompt has `c - 1`
  tokens and the cached seed `4 * floor((c - 2) / 4)`.
- Each point runs 5 warmups and 10 measurements, and the table stores the
  median.
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

Engine arguments, beyond the usual Dynamo benchmark flags:
`--enable-prefix-caching --prefix-match-unit 4 --kv-cache-dtype fp8_e4m3
--no-async-scheduling --language-model-only --cudagraph-metrics
--max-num-batched-tokens 8192 --max-num-seqs 32 --benchmark-warmup-iterations 0`.

Environment:

| Variable | Value |
|---|---|
| `DYN_FPM_GLM53FLASH_REAL_KV` | `1` |
| `DYN_FPM_GLM53FLASH_PREFIX_SEED` | `1` |
| `DYN_FPM_GLM53FLASH_PREFIX_MATCH_UNIT` | `4` |
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
multiple of 4. The producer still checks every prefill chunk start. In guided
collection, a decode point seeds `(c - 2) // hash_block * hash_block` tokens,
not the `4 * floor((c - 2) / 4)` of the launch above with
`--prefix-match-unit 4`.

The container needs one overlay directory at `/opt/glm53flash-dynamo`. A Slurm
run can mount it with `--container-mount`. The collector uses that path unless
the deployment environment declares its own `PYTHONPATH`. The overlay is a
composite of two parts. The first part is the `dynamo` files of the
`ai-dynamo 1.6.0.dev20261010` build, whose scheduler file
`runtime-source-sha256.json` pins. The second part is the `collector`,
`aisimulate_core` and `aisimulate` packages of the AISimulate checkout that
plans the collection. The producer imports them.

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
  the budget at 8,192.
- Prefill requests per batch: the powers of two up to the smallest of 32,
  `max_batch_size` and the number of requests whose new tokens fit the budget,
  and that smallest value itself.
- Prefill prefix per request, in whole hash blocks: 0, the powers of two, and
  the largest number that keeps prefix and new tokens within the measured
  context.
- Decode context per request: the powers of two from 4 to the measured context,
  and the measured context itself.
- Decode requests per batch: the powers of two up to the smaller of 32 and
  `max_batch_size`, and that value itself.

The planner drops a point that does not fit the accepted cache capacity. It
refuses a plan with no prefill point or no decode point. At TP4 with 2,176-token
hash blocks, 8,192 tokens, 256 sequences and a 1,048,576-token context, the
list has 496 prefill points and 96 decode points. That needs an accepted cache
of at least 2,113 pool blocks (the profile's `kv_cache_bytes` divided by its
largest `page_size_bytes`).

The smoke measures four points of this list. It runs under the plan hash of
formal collection and keeps its own campaign directory. The prefill points are
one request with 64 new tokens, once without a prefix and once after one hash
block of prefix. The decode points are one request and two requests, each at the
smallest context of the list above one hash block. A direct `collector --fpm`
call can pass its own manifest with `--fpm-benchmark-points-file`. The collector
refuses that file together with `--smoke`.

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
