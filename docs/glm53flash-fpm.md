<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.3-Flash native FPM collection and GB300 profiles

## Implementation contract

Extend native whole-forward collection to GLM-5.3-Flash on vLLM and SGLang. Preserve the actual timing boundary, frozen request geometry and runtime identity. Require real KDA/conv, MLA KV, pooled index and tail state before calibration. Publish admitted profiles and provenance to nvidia/aisimulate-fpm-dataset with immutable consumer pins. Independent prefill and decode holdouts must each achieve MAPE <=10% in every required deployment cell.

The required matrix is GB300 × vLLM/SGLang × native FP8/NVIDIA NVFP4 × TP2/TP4, through 131072 context tokens. NVFP4 TP1 is optional after measured memory admission. Text-only scope excludes vision execution, MTP, expert parallelism, offload and cross-node serving.

## Source identities

- FP8: `zai-org/GLM-5.3-Flash@eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`.
- NVFP4: `nvidia/GLM-5.3-Flash-NVFP4@09b04e5e74bca08ca8549fc736d4cdd8624bfde3`.
- vLLM: stock `v0.31.0`, source `db9527a46873454610df6dbedf79a36d6bf1a7f6` (earlier campaigns:
  `v0.30.0`, source `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, and its local repairs below).
- SGLang: `v0.5.20`, source `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`.

### vLLM v0.31.0 recollection (stock runtime)

The vLLM FPM is recollected on stock vLLM `0.31.0` without a local patch. The
default serving configuration is unchanged: prefix caching on (Mamba `align`
mode, `--prefix-match-unit 4`), `FULL_AND_PIECEWISE` graphs with the 62 capture
sizes, `--no-async-scheduling`, GPU memory 0.92, budget 8192. Timing is the
engine-native Dynamo `InstrumentedScheduler` FPM `wall_time` of the
prefix-seed producer (`glm53flash_prefix_scheduler.py`).

- Stock v0.31.0 does not complete IndexPool entries at a cached or chunked
  prefill start that is not a multiple of 4. Every prefill prompt, prefix and
  new-token length is a multiple of 4. Decode points keep their true context
  `c`: the prompt is `c - 1` and its cached seed prefix is `4 * floor((c - 2) / 4)`,
  so the measured shot starts on the grid. The measured step is the second
  pure decode step. The producer records every prefill chunk start from the
  native `SchedulerOutput` and fails on an unaligned start. Geometry moved to
  satisfy this is executed geometry with reason `kpool_align4`. The consumer
  still rejects unaligned cached-prefill queries on stock runtimes. Optional
  per-request decode contexts come from a frozen sidecar
  (`DYN_FPM_GLM53FLASH_DECODE_CONTEXTS`), because Dynamo's explicit decode
  points carry totals only.
- Inputs are the seeded token stream of `collector/glm53flash_attention_tokens.py`
  (generator `sha256_counter_rejection` v1, seed 53, ids `[0, 154820)` of the
  pinned tokenizer), recorded in `input_provenance`; there is no text corpus.
- Deployments: FP8 and NVFP4 at TP2/TP4, plus NVFP4 TP1. FP8 TP1 does not
  fit one GB300: a stock v0.31.0 startup ran out of memory while creating the
  weights (275.26 of 276.62 GiB allocated). NVFP4 TP1 reads back a Mamba block
  of 8576 tokens and 7,198,858 KV tokens at memory 0.92. The block is larger
  than the 8192 budget, so a long prefill advances by whole 8192-token chunks
  and then stops at the next block boundary. Both are multiples of 4.
- Dynamo: the instrumentation stays at `54960177`. Its `_compute_queued`
  reads `Scheduler.skipped_waiting`, which v0.31.0 removed. The producer
  overrides that one method with the v0.31.0 version from Dynamo `395f0240`.
- Source pins: `runtime-source-sha256.json` holds the v0.31.0 stock closure.
  It uses the moved `vllm/models/glm5next/common/` and `nvidia/sparse_indexer.py`
  modules. The 0.30.0 closure that the historical repairs extend is kept in
  `runtime-source-sha256-vllm-0.30.0.json`.

These are qualification candidates, not evidence of measured GB300 coverage. Preserve the checkpoint's actual per-module precision; do not relabel all weights as FP8 or NVFP4. TensorRT-LLM serving support is outside this initial matrix.

## Acceptance and data status

The required matrix has eight deployments: vLLM/SGLang × FP8/NVFP4 × TP2/TP4.
All eight have complete per-deployment holdout evaluations in two separate
serving configurations (see [Measured results](#measured-results)). CPU tests,
runtime qualification and individual native completions are separate evidence;
none grants dataset publication.

Implemented interfaces include model-specific planning, the native same-request
hybrid-state protocol, SGLang Engine collection, Generator/Slurm/Kubernetes
routing, strict request/dispatch/state validation, bounded original-point shards,
installed-consumer validation and qualified dataset staging. Each retained point
uses five warmups and ten measurements. Calibration and holdout corpus, request
and geometry identities stay disjoint.

Stock vLLM cached-prefill restrictions remain attached to the stock identity.
The former `0.30.0+glm53kpool.bf5f6b0e689d` candidate is quarantined after
circular-tail out-of-bounds evidence; earlier successes do not restore eligibility.
The separately identified `0.30.0+glm53tail.eb4704514fdf` repair has exact
source/binary closure and an immutable four-deployment functional qualification
summary, SHA256
`8fc691d6054f48741c248eb7937b7b4db6220ff1ea337b968ff656c56ba8cf45`.
Its three-profile comparisons, cache oracle and one NVFP4 TP2 128K ordinary-request
regression are functional evidence. Each deployment still needs its own capacity,
producer, full-campaign and independent accuracy checks. The reference runtime
is not a measured producer. Licensed patches and qualification commands remain in
the [runtime qualification sources](../python/aisimulate/collector/fpm_forward/runtime/glm53flash_vllm_tail_repair/README.md).

The Rust FPM cached-prefill guard admits that exact reviewed tail version in
parity with the Python native reader. Stock unaligned starts, unknown local
suffixes and the quarantined KPool runtime remain rejected. This admits queries
to existing calibration tables; it does not grant holdout coverage or accuracy
acceptance, which still requires predictions for every original requested point.

The current tail runtime has passed the original nine-point producer qualification
and strict prefill/decode readers in all four vLLM deployments, followed by the
formal 397-calibration/221-holdout campaign. SGLang uses native 0.5.20 with an
explicit 16384 MiB allocator split limit and a per-deployment static-memory
fraction (see [Serving configurations](#serving-configurations)). Allocation and
graph policy changes require new qualified campaign identity; historical unknown
allocator settings stay unknown.

Formal hardware admission binds each native worker's GB300/sm103 identity,
source, attempt and distinct UUID where available. SGLang state tensor devices
must agree. Historical canaries without this contract cannot be backfilled into
formal acceptance. Original startup, state, capacity and collection failures
remain available, including host-loader OOM separately from GPU-memory failures.

All eight cells must provide complete independent holdout predictions with
**prefill MAPE <=10% and decode MAPE <=10%**, plus WAPE, P95/max APE,
context-band results and every original failed/missing request. Exact table
self-queries establish integrity only. The dataset staging gate requires all
sixteen phase cells to pass; immutable Hugging Face ingestion, consumer pin,
installed-wheel replay and offline-cache loading remain separate required steps.
HTTP TTFT/TPOT is reported separately. Existing model data remains unchanged.

Track [AIC-1999](https://linear.app/nvidia/issue/AIC-1999). SOL provides the common
model contract; Ops has its own timing, data and accuracy acceptance.

## Native collection interfaces

`python -m collector.fpm_forward` accepts `--backend vllm` or `--backend sglang`
for this model, a frozen `--fpm-benchmark-points-file`, and pure TP2/TP4.
Use `--fpm-input-text` to freeze a corpus by content hash and
`--fpm-dataset-role holdout` for independent validation runs; holdout runs never
publish calibration rows. Native SGLang 0.5.20 uses phase-specific graph flags;
FPM translates the shared Generator's legacy decode graph-size options to
`--cuda-graph-bs-decode` and `--cuda-graph-max-bs-decode`. The normal plan/run/resume/checkpoint workflow and
`--fpm-executor slurm` transport are retained. A Slurm run requires an existing
owned allocation, an explicit container image, and checkpoint/runtime mounts.

For SGLang, `--sglang-mem-fraction-static 0.82` explicitly passes the native
static-memory fraction. The value must be finite and strictly between zero and
one; the option is rejected for other backends. Omitting it preserves SGLang's
runtime default. An explicit value is frozen in the plan, cell and shard
identities, and both archived declared and resolved ServerArgs must match it.
Calibration and holdout must also have identical actual serving settings
(except the native per-process random seed). A changed fraction requires a new
qualified campaign; it does not turn a historical OOM or earlier default-setting
measurement into a successful observation.

`--sglang-allocator-max-split-size-mb` passes the native allocator split setting
before framework imports. It accepts a positive integer from 20 through
8796093022207 MiB, and is rejected for other backends. Omission preserves the
native default; it is not an observed historical default. Requested and actual
per-worker allocator identities are recorded and validated across calibration,
holdout and shards. FP8 TP2 uses 16384 only in its explicitly frozen campaign;
it does not change the other three SGLang cells or justify dropping failed points.

Keep writable runtime caches separate from a read-only checkpoint mount.
The generated launcher defaults an unset `FLASHINFER_CUBIN_DIR` to
`${HF_HOME}/flashinfer-cubins`; this requires a writable `HF_HOME`. A campaign
that mounts its model/HF directory read-only must explicitly set the cubin
directory through `extra_env` or its frozen pre-import environment hook.
Set `FLASHINFER_WORKSPACE_BASE` separately: relocating that workspace does not
relocate downloaded cubins. Use allocation-private, node-local directories and
separate each worker's Triton cache before framework imports.

Qualify the complete generated shell environment and fresh worker startup,
including any inherited cache settings. In the actual container, check
FlashInfer's resolved cubin path and an owned write/read/delete there; printing
an environment variable alone does not prove the directory is usable. Retain
the hook source/hash and process-specific cache receipts with the campaign.
Preserve an original startup failure and freeze changed environment inputs for
its replacement attempt without changing the requested point set or corpus.

The SGLang driver uses the native Scheduler, ScheduleBatch and request/cache
objects. Its retained-request benchmark protocol seeds each request through
actual forwards, parks completed prefixes in the native ChunkCache, then
forms the frozen homogeneous target batch from those same requests. Decode
also executes its real admission forward. Actual KV/Mamba slots, prefix maps,
input/output tokens and every TP rank's completed target/release receipts are
validated. Arbitrary prefix counters and fake cache state are forbidden. The
protocol requires its own GPU and ordinary-Engine output qualification; CPU
lifecycle tests alone do not grant admission.

SGLang timing uses the native DeviceTimer forward interval, including its
native logits boundary. The producer publishes rank zero's median; FPM rows and
holdout scoring use the fastest-TP-rank median (see
[SGLang TP latency](#sglang-tp-latency)). Token readback happens outside that interval and is
recorded as an explicit telemetry policy; it can serialize serving overlap.
vLLM preserves the native scheduler/output interval, including host work.
Neither boundary is interchangeable with HTTP TTFT/TPOT. Ops-instrumented
latency cannot be admitted to the FPM database.

The strict reader checks the complete frozen point set and the retained raw
repetitions before the common Parquet publisher records each median. Raw
producer failure artifacts remain available even when no table is published.

SGLang distinguishes its measured context limit from its internal admission
reserve. For inclusive 131072 measured tokens, the generated engine context is
131079: the native worker reserves six slots and input admission rejects
equality. Checkpoints support this context without a model override. Both
limits are recorded, and real allocator admission remains necessary. The
reader binds every seed/target/rank trace to the run, checkpoint/execution
identity, telemetry policy and context policy. It also verifies hashed native
source-preflight and declared/resolved configuration receipts.

Use the [candidate generator](../python/aisimulate/collector/fpm_forward/README.glm53flash-sampling.md)
and [installed-consumer holdout validator](../python/aisimulate/collector/fpm_forward/README.glm53flash-validation.md)
for reproducible campaign construction and independent acceptance. The default 547 geometries per cell are unqualified candidates. The additive
Ops-bracketing v2 bundle contains 618 per cell (397 calibration, 221 holdout),
preserving every original holdout byte and ID. Necessary geometry coverage is
not measured kernel coverage or accuracy acceptance. Long campaigns use
[bounded shards](../python/aisimulate/collector/fpm_forward/README.glm53flash-shards.md)
with immutable original-point mapping and complete-union publication; retries
preserve previous attempts and their raw evidence.

## Pinned model configurations

`zai-org/GLM-5.3-Flash` and `nvidia/GLM-5.3-Flash-NVFP4` are registered in
`DefaultHFModels`. Model-config resolution therefore uses the bundled
`config.json`/`hf_quant_config.json`, which are byte-identical to the pinned
checkpoint revisions above, and never downloads Hugging Face `main`. This applies
to every producer, host planner, strict reader and evaluator process that
resolves these IDs. The fix was needed because the upstream NVFP4 `main` edit
`da920bb0` (2026-09-29) added MTP layer-45 ignore entries; with live resolution,
every native producer and validator that compares the loaded checkpoint with the
pinned identity rejected byte-identical pinned checkpoints. The resulting
execution identities are unchanged: FP8 `3629e61b...`, NVFP4 `e54aa2aa...`.

## Prefill CUDA graphs

Both runtimes run GLM-5.3-Flash prefill eagerly in their default configuration
above a small size. This adds a fixed ~105–120 ms floor to every prefill forward:

- **vLLM** captures CUDA graphs only up to 64 tokens by default (11 sizes:
  1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64; `FULL_AND_PIECEWISE`). Prefills above
  64 new tokens run eager. Raising only `--max-cudagraph-capture-size 8192`
  captured 531 sizes and used 89 GiB per GPU of graph pool. That left 126 KV
  blocks at FP8 TP2, and the frozen B32/T8192/KV40960 point became infeasible.
- **SGLang 0.5.20** disables its breakable prefill CUDA graph for KDA hybrid
  linear-attention models by default. Its log reports: "Breakable CUDA graph is
  incompatible with KDA hybrid linear attention; disabling prefill CUDA graph."
  Only decode graphs (`full`, 1..32) are captured.

Explicit graph settings are opt-in, frozen collection identity:

- `--vllm-cudagraph-capture-sizes N [N ...]` (vLLM, GLM only) renders
  `--cudagraph-capture-sizes` as a backend policy
  `explicit-cudagraph_capture_sizes=n<count>-max<max>-<sha>`. The resolved
  `config.engine_args.cudagraph_capture_sizes` marker and every native rank's
  `cudagraph` block (`capture_sizes`, `prefill_capture_sizes`,
  `max_capture_size`) must match. Omitting it preserves existing plans and cell
  IDs.
- `--sglang-cuda-graph-backend-prefill breakable --sglang-cuda-graph-max-bs-prefill N`
  (SGLang, GLM only; the two are required together) renders the native opt-in.
  The plan, cell, shard and cell ID record it. Declared and resolved ServerArgs
  must carry the same backend and maximum. Neither may disable CUDA graphs, and
  the resolved prefill `cuda_graph_config` must use that backend with buckets up
  to N.
- Calibration, holdout and every shard of one deployment must have identical
  backend policies and SGLang prefill-graph settings. The evaluator also
  re-checks the vLLM resolved-config markers for each holdout run. Default-config
  and graph-mode data therefore never mix.

With graphs on, the fixed floor disappears in every deployment. Median holdout
prefill latency for 64 < T ≤ 512 drops to 0.25–0.28× (vLLM) and 0.33–0.40×
(SGLang) of the default configuration. The ratio rises with real work and
reaches 0.75–0.99 near T = 8192. Decode is unchanged within a few percent; its
graphs were already captured. In the SGLang A/B smoke, first sampled tokens
matched the eager configuration except at short contexts, where the eager
baseline is itself nondeterministic.

Cost: vLLM startup grows by 5–7 minutes and KV capacity shrinks (FP8 TP2 3844
blocks instead of 4059). SGLang prefill capture adds 390–415 s and
~21.6 GB per GPU at TP2 (~13.6 GB at TP4). At TP2 that requires a lower static
fraction to avoid OOM in the bs32 × prefix-130816 point.

## Serving configurations

Shared by both configurations:

- GB300; pure TP2/TP4; DP=PP=CP=EP=1.
- Text only. MTP/speculation, EPLB and offload are disabled.
- FP8 KV cache.
- 131072-token measured context, 8192-token prefill chunk, batch ≤ 32.
- The same frozen 397 calibration + 221 holdout points and disjoint corpora,
  in 18 whole children per deployment.
- 5 warmups + 10 measurements per point.

| Setting | Default configuration | Graph mode |
|---|---|---|
| vLLM runtime | `0.30.0+glm53tail.eb4704514fdf` | same |
| vLLM CUDA graphs | runtime default: `FULL_AND_PIECEWISE`, capture sizes up to 64 | `--cudagraph-capture-sizes`: 62 sizes = vLLM default ∪ SGLang's 58 prefill buckets, max 8192; prefill runs `PIECEWISE` |
| SGLang runtime | `0.5.20` | same |
| SGLang prefill graph | disabled (KDA auto-disable) | `--cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 8192` (58 buckets 4..8192) |
| SGLang decode graph | `full`, 1..32 | same |
| SGLang allocator split | 16384 MiB | 16384 MiB |
| SGLang static-memory fraction | FP8 TP2 0.82, FP8 TP4 0.75, NVFP4 TP2 0.82, NVFP4 TP4 0.82 | FP8 TP2 0.78, FP8 TP4 0.75, NVFP4 TP2 0.78, NVFP4 TP4 0.78 |

In graph mode, both TP2 deployments needed 0.78 because 0.82 ran out of memory
in the bs32 × prefix-130816 point. NVFP4 TP4 at 0.82 completed, but every
forward of that point freed and re-allocated a 32 GiB transient (1320 allocator
retries; 1410 ms instead of ~220 ms). Its campaign was therefore re-collected at
0.78. The 0.82 run is kept as history.

## SGLang TP latency

The native SGLang records carry one device-timer duration per TP rank and
forward, but no per-rank start or end timestamps. On TP>1, rank 0's duration
includes waiting for later-arriving ranks inside the collectives. For each
measured repetition, FPM rows and holdout scoring take the fastest TP rank's
duration, then the median over the 10 repetitions
(`sglang_tp_fastest_rank_duration_median_v1`). With a common collective end,
this equals common end minus latest start. The producer's published rank-0
median is still checked against the raw traces. Both the calibration table and
holdout values come from the same validated observations, and the evaluator
records `latency_reduction` in its native evidence. The row
`measurement_policy` label stays `sglang_native_real_hybrid_median_v1` because
the installed Rust consumer admits only that label for SGLang schema-7 rows.

## Measured results

Per-deployment FULL holdout evaluations: 397 calibration rows, 144 prefill +
77 decode holdout points, every point predicted, no point dropped. Values are
MAPE %, prefill / decode. The public phase gate is MAPE ≤ 10%.

| Deployment | vLLM default | vLLM graph mode | SGLang default | SGLang graph mode |
|---|---|---|---|---|
| FP8 TP2 | 2.13 / 3.05 | 1.23 / 2.67 | 1.73 / 1.13 | 2.32 / 0.80 |
| FP8 TP4 | 1.60 / 2.68 | 1.53 / 2.49 | 2.34 / 1.83 | 1.33 / 1.31 |
| NVFP4 TP2 | 2.44 / 3.08 | 1.38 / 2.58 | 2.01 / 0.71 | 3.95 / 0.75 |
| NVFP4 TP4 | 2.09 / 2.53 | 1.83 / 2.28 | 2.02 / 0.60 | 2.32 / 0.62 |

The two configurations are separate datasets with separate campaign
identities. The proposed Hugging Face dataset update is
[nvidia/aisimulate-fpm-dataset discussion 15](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/discussions/15).

## Weight sizing during rendering

The Generator reuses the shared GLM SOL resident-weight model for the native
checkpoint's mixed precision and layer schedule. Its TP1 weight estimate still
feeds the existing 1.5× naive sizing heuristic; it is not a measured memory
qualification and excludes serving cache, CUDA graphs and workspace. Frozen
FPM plans require no additional model lookup while rendering. Requested FPM TP2
and TP4 configurations still use their explicitly frozen pure-TP topology.
