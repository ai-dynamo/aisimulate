<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.3-Flash FPM

Whole-forward (FPM) profiles for GLM-5.3-Flash on GB300, collected with
engine-native timing on the default serving configuration (prefix/radix cache
on). Text only, pure TP, DP = PP = EP = 1.

| Backend | Runtime | Deployments |
|---|---|---|
| vLLM | stock `0.31.0` (`db9527a4`) | FP8 TP2/TP4, NVFP4 TP1/TP2/TP4 |
| SGLang | `0.5.20` (`94602c9c`) | FP8 TP2/TP4, NVFP4 TP2/TP4 |

Checkpoints: `zai-org/GLM-5.3-Flash@eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`
and `nvidia/GLM-5.3-Flash-NVFP4@09b04e5e74bca08ca8549fc736d4cdd8624bfde3`.
FP8 TP1 does not fit on one GB300.

The tables (`fpm_forward_perf.{parquet,metadata.json}`) are not stored in this
repository. They are published in `nvidia/aisimulate-fpm-dataset`
([discussion 15](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/discussions/15)).
[`hf_dataset.json`](../python/aisimulate/src/aisimulate_core/systems/profiles/glm53flash_fpm/hf_dataset.json)
pins the nine profiles to its merge commit
`08771888388a1ce9cd8d421b2e42f353fd5e2abb` and records each file's SHA-256.
Profile keys are `gb300-<backend>-<fp8|nvfp4>-tp<N>`, for example
`gb300-vllm-nvfp4-tp1` or `gb300-sglang-fp8-tp2`. Materialize one with
`python -m aisimulate_core.sdk.fpm_dataset <manifest> <key>` (or
`materialize_fpm_profile`) and pass the printed root as `systems_paths`, with
`estimation_mode="fpm_interpolation"` and the profile identity's
`backend_version`.

## vLLM producer

`python/aisimulate/collector/fpm_forward/runtime/glm53flash/` contains the
producer. It is staged next to `fpm_exec.sh` and `preflight.py` and runs with
the pinned Dynamo `InstrumentedScheduler` overlay on `PYTHONPATH`.

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
--model-revision <rev> --tp {2,4} --quant {fp8,nvfp4} --mem-fraction <f>
--phase {prefill,decode} --points <manifest.json> --out <dir>`.

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
