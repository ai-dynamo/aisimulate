<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM 0.31.0 B300 qualification

- Collector commit: `0bba6709ed4ef3a0fcae878d2ad2fb61bbd11a04`.
- Slurm job `984997`: COMPLETED, exit 0:0, 1 GPU, elapsed 2m19s. Eight isolated case subprocesses: all passed; no skipped or unattempted cases.
- Hardware: NVIDIA B300 SXM6 AC (SM103), node pool0-0003. CUDA-reported usable memory 287,428,640,768 bytes (267.689 GiB); driver 580.126.09.
- Runtime: official vLLM 0.31.0 image; torch 2.13.0+cu130; FlashInfer 0.7.0.post1.
- Native backend: FLASHINFER, page size 16. Prefill source `vllm_flashinfer_trtllmprefill`; decode source `vllm_flashinfer_flashinfertrtllmapidecode_trtllm_gen`.
- Validation-only raw rows remain here; none are eligible for publication into the 95-row Qwen data shard.

| Case | Kernel latency (ms) | Collector call (s) | Isolated process (s) |
|---|---:|---:|---:|
| qwen_bf16_prefill_4096 | 0.056085 | 14.880 | 29.497 |
| qwen_bf16_prefill_32768 | 1.530997 | 2.724 | 9.724 |
| qwen_bf16_prefill_262144 | 132.734263 | 4.461 | 11.357 |
| qwen_bf16_decode_4096 | 0.013232 | 2.757 | 9.333 |
| qwen_fp8_prefill_4096 | 0.050048 | 2.845 | 9.945 |
| qwen_fp8_decode_4096 | 0.012837 | 2.886 | 10.023 |
| head512_fp8_prefill_128 | 0.013637 | 2.677 | 10.214 |
| head512_fp8_decode_4096 | 0.019589 | 2.891 | 10.024 |

The 262,144-token case is BF16, batch 1, 4 query heads, 1 KV head, head dimension 256. Its metadata contains 262,144 prefill tokens and zero decode tokens. Native timing uses 3 warmups and 6 measured runs with CUDA graphs. Recorded warmup/capture/timing/cleanup phases sum to 1.628 seconds.

These timings measure one representative long-context geometry, not the full default grid. Each process pays import/startup costs, and the first 4K case includes cold compilation/cache setup. Do not infer marginal grid cost by subtracting these process durations. The prior fleet-rate estimate remains a budgeting proxy, not a measured fleet cost.

The head-dimension 512 FP8 prefill/decode cases executed the native quantized-query-capable implementation successfully. This validates the collector execution and phase/source metadata, not numerical output accuracy or the whole-model hybrid-cache allocator.

Slurm requested GPU frequency high. Saved nvidia-smi records show a 1100 W power limit and application clocks 2032/3996 MHz, but do not establish fixed in-kernel SM clocks. No power sampling was enabled. These measurements are qualification/cost probes, not a production performance baseline.

Initial job 984980 failed before running cases because the launcher used python instead of python3. It consumed 41 seconds of one-GPU allocation; retry changed only the launcher. Total allocation including that failed start was 180 seconds (0.05 GPU-hours).

Image OCI child: `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`.
Image squashfs SHA256: `35b6881618f1ab48ef054126d659e18aebf2a009dd918b726f4220b32a274eae`.
Collector file SHA256: `04b8d97f835b0fc18bfa17a09d798f8ca0ca5daecf4ecf60ea0213adae127097`.

Raw evidence: `job-984997/cases/*/result.json`, native perf rows, per-case logs, image/source provenance, Slurm accounting, and before/after nvidia-smi records. Observation wrappers only record native metadata and benchmark results; they do not change the values returned to the frozen collector.

## SGLang coverage and memory admission

The same frozen collector restores the default 250-pair GLM decode grid,
including batch 52 and histories through 720,896. The prior grid contained
154 pairs. CPU regressions preserve this FP8 grid on 80 GiB, B200-size and
B300-size devices; the 80.29 GiB BF16 cache case is rejected on 80 GiB and
retained on the larger devices. Native allocation checks remain required.

The current collector initializes every historical KV and index-cache slot.
For this grid, the 96 added pairs account for 678,494,052 of 687,237,812
history tokens per operation. Collection cost therefore cannot be estimated
reliably from the historical 0.5.20 collector that did not seed all history.
The [SGLang data PR](https://github.com/ai-dynamo/aisimulate/pull/665) tracks
that separate 0.5.21 campaign, its exact source, complete grid and measured
per-shape setup/measurement costs. Its measurements are not included in this
collector-only PR.
