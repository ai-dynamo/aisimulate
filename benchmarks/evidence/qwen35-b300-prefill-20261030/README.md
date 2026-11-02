<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.5 B300 vLLM 0.31 prefill measurements

The published `context_attention_perf.parquet` contains the 95 successful
measurements from the full collection, copied without changes alongside its raw
`collection_meta.yaml` into
`python/aisimulate/src/aisimulate_core/systems/data/b300_sxm/attention/vllm/0.31.0/`.
The separate one-token decode control and smoke results are excluded. No other
performance table is added or replaced.

## Collection and integrity

- Collector source: [13e15164522c495af025cfb4d683c7dd06666328](https://github.com/ai-dynamo/aisimulate/tree/13e15164522c495af025cfb4d683c7dd06666328).
  The [campaign guide](https://github.com/ai-dynamo/aisimulate/blob/13e15164522c495af025cfb4d683c7dd06666328/python/aisimulate/collector/README.qwen35_prefill.md)
  records the source audit, exact plan and runtime pins. CI checkout and artifact
  verification establish this revision; the raw sidecar's `collector_ref: unknown`
  is preserved, rather than rewritten as an original collection claim.
- Auto-collector [parent pipeline 77828291](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-auto-collector/-/pipelines/77828291),
  [child pipeline 77828358](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-auto-collector/-/pipelines/77828358),
  [job 518571254](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-auto-collector/-/jobs/518571254),
  Slurm job `138021`: 95/95 cases, zero errors, 95 distinct row identities and
  finite positive latencies. The rows match the full plan exactly.
- Runtime: `vllm/vllm-openai:v0.31.0`, vLLM source
  `db9527a46873454610df6dbedf79a36d6bf1a7f6`, amd64 image digest
  `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`.
  Separate image-resolution evidence records the multi-platform index
  `sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`.
  The raw sidecar records the executed amd64 child digest shown above.
- Hardware: one NVIDIA B300 SXM6 AC, measuring the per-rank dense-attention
  geometry of Qwen3.5-397B-A17B TP8. This is an operator measurement, not a
  complete eight-GPU serving benchmark.

| Artifact | SHA-256 |
| --- | --- |
| Published Parquet | `d1188fe9cdb287bb5c87a9094026a17fe50d2401266a71a657facadbfec45700` |
| Raw sidecar | `6c16ec43beed1f94ff28b9c6ff46020a552da03f8789d886dd0ae4a3d0f175a4` |
| Full job result archive | `4b3396c07c201fa7f12c7b2e8f54c5b83c92f9423a2b78b48e6c5823b5f25d97` |

## Measurement scope

- Fixed geometry: 4 query heads, 1 KV head, head dimension 256, BF16 attention
  and KV cache, beam width 1, no sliding window, `step=0`.
- Full causal prefill: query length equals sequence length. Batch sizes
  `[1, 2, 4, 8, 16]` cross sequence lengths
  `[16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
  10240, 12288, 16384, 24576, 32768, 40960]`.
- All 95 dispatch records report framework backend `FLASHINFER`, actual source
  `vllm_flashinfer_trtllmprefill`, kernel page size 16, `LBHNC` cache layout,
  and CUDA graph timing. Latency is milliseconds for KV update plus attention.
  Kernel page size 16 is distinct from the target serving hybrid allocator/APC
  block size 528. No power measurement is provided.

## Collection cost and collector dependency

Default coverage expansion and current collector qualification belong to
[#664](https://github.com/ai-dynamo/aisimulate/pull/664). This data PR preserves
the original 95 measured rows and their provenance; qualification rows from
later collector revisions are not added to this table.

The original 95-case collection took 52.751 seconds through parquet output,
including 26.312 seconds of worker execution. CI took 143.526 seconds,
excluding queue time. Its 18 extra long-sequence anchors were not timed
separately. PR #664 records the broader default sweep cost and the separate
262,144-token qualification.

## Loading and limits

- With vLLM 0.31 requested and `attention_backend=flashinfer`, existing consumer
  ordering prefers this primary prefill lane over the older decode-labelled
  context donor. The research consumer check verified native selection with
  unchanged SDK and native binary. This publication changes no selection code,
  calibration, reuse policy or default framework version.
- The existing reuse manifest already admits this source as shared data. It can
  fill compatible shapes for other models; publication is not limited by model
  name. Default `current` (0.30) does not look forward to these 0.31 rows.
  Table-wide lane ordering can also change which older donor other geometries
  use at `next`; the current-main impact check below records that effect.
- **Page size is not a lookup key.** These page-16 rows share the existing
  attention family with page-64 collector/decode and donor measurements. The
  loader cannot enforce page compatibility. Shape or version reuse must not be
  interpreted as evidence that pages 16 and 64 have equivalent timing.
- The context table does not encode query phase independently of sequence
  length or measure a cached-tail prefix axis. One-token APC tails remain
  unqualified; adding decode-labelled Q1 rows to this primary context table can
  change lane precedence again. Cached-prefix estimates retain the existing
  full-context work-ratio approximation.
- Graph-timed operator measurements do not establish eager serving-prefill
  overhead or whole-model accuracy. Other GDN, MoE and communication donors are
  unchanged. Collection completeness applies to this 95-case plan only.

## Historical publication checks

- Artifact hashes, schema equality with the existing context table, the exact
  95-case grid, unique identities and positive finite latencies pass.
- At original publication, all seven `check_collector_data.py` rules, the 4,050-slice
  `backend_facts.py --check`, and 125 existing metadata, evidence-policy,
  data-layout and backend-fact tests pass.
- All 95 exact anchors match the native attention consumer, including the
  intended primary lane and strict provenance. This check used Python source
  `fad97a78a87081f99db2dbf931f0fc59540beff2` and the existing research native
  binary against the newly published data root. Its sidecar parser is identical
  to publication base `62519bfd26ee9d9c4cc9b53aba81fb30426f8a3d`; the native
  binary SHA-256 is
  `0ffd574ec41029b541b479e75e132c019db4769660e3a35c770c202f47381df2`.
  This is consumer compatibility evidence, not an E2E rerun on current main.

## Fresh-main replay and wider lookup impact

[Fresh-main evidence](fresh-main-validation.json) uses Python and a newly built
Rust extension from `e6b1fe2bb8e5847c3f6eee0ed6c5c4042b290dc8`. Native SHA-256:
`67ae1963467a327fc6f8b884b3a119c7948f5c0927bdc76b3fa8231a76f309c9`.
Both variants use that same binary; the candidate adds only the unchanged
95-row table and sidecar to main's systems tree. All seven collector-data
rules pass on this fresh runtime, and the current backend census matches
4,038 slices. Independent review verifies both original artifact hashes, the
exact 95-row grid, positive finite latencies, and the saved replay evidence.

- Reused B300/vLLM 0.31 silicon: 36 runs, 2,016 measured requests; 4K/32K
  initial prompts, 1/4/16 lanes, cold/warm caches, three repeats. Preserve each
  run's recorded token lengths, BF16 TP8, hybrid block 528, FlashInfer,
  correction disabled and SILICON fallback denied.
- All 36 cases, 2,016 request pairs and 2,142 cache checks pass in each variant.
  An independent timestamp-based audit reproduces all request and aggregate
  scores. No numeric accuracy threshold or calibration was applied.

| Request MAPE | Fresh main data | Fresh main + 95 rows | Archived candidate, separate runtime |
| --- | ---: | ---: | ---: |
| TTFT | 4,367.37% | 64.66% | 62.89% |
| TPOT | 1,034.58% | 18.80% | 18.71% |
| E2E | 1,556.90% | 25.12% | 24.78% |

Some operating points worsen: 4K/4-lane cold TPOT MAPE increases from 26.14%
to 29.55%; warm-cache throughput MAPE increases by 0.68 and 7.89 percentage
points at 1 and 4 lanes. The pooled improvement is not universal accuracy.

Adding the Qwen primary rows changes table-wide lane ordering at `next`.
The non-Qwen control uses Llama-3.1-70B TP8 geometry (8 query heads, one KV
head, head dimension 128, BF16), a single attention layer, and no QK norm.
It selects the existing **0.30 true-prefill donor** instead of the existing
0.30 decode-labelled donor. No D128 rows are added by this PR.

| Non-Qwen attention lookup at `next` | Main data (ms) | Main + Qwen rows (ms) |
| --- | ---: | ---: |
| Batch 1, full sequence 4,096 | 234.760717 | 0.054101 |
| Batch 4, full sequence 16,384 | 3,732.635900 | 1.395179 |

All four Llama and four Qwen controls at `current` (0.30) are exactly unchanged.
Forced-lane native evaluations match the selected-lane results. This establishes
lookup impact, not non-Qwen whole-model or silicon accuracy. The remaining
page-size and cached-tail limitations above still apply.
