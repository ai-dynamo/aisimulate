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
- vLLM: `v0.30.0`, source `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
- SGLang: `v0.5.20`, source `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`.

These are qualification candidates, not evidence of measured GB300 coverage. Preserve the checkpoint's actual per-module precision; do not relabel all weights as FP8 or NVFP4. TensorRT-LLM serving support is outside this initial matrix.

## Acceptance and data status

Implemented: GLM-specific planning and configuration identity; source-pinned vLLM real-hybrid scheduling; native SGLang Engine collection; Generator, Slurm and Kubernetes backend routing; exact request/dispatch/state evidence validation; and five warmup plus ten measurement medians. The native runtime adapters are shared with the companion Ops implementation. CPU contract tests and all eight deployment renders pass.

GPU qualification and producer qualification are separate. All eight native
model/request smoke configurations passed on GB300: vLLM job 601652 and SGLang
job 603053. The long request used 131008 input tokens plus 32 decode tokens.
The subsequent FP8 TP2 producer job 604763 completed exact inclusive 128K
prefill at B1/B32 P131040/Q32 and decode at B1 past131071. All three points
passed their historical strict reader and aggregation with five warmups and
ten retained samples. Long ordinary-Engine output comparison and the new
formal hardware contract remain separately unqualified.

The vLLM FP8 TP2 producer canary in job 603053 completed two prefill and two
decode points, each with five warmups and ten retained observations. The
production native reader and common aggregator passed all four points. Prefill
Q1024/P0 used NONE dispatch; Q3/P4096 used PIECEWISE with four total tokens after padding (one padding token).
Decode B1/P1024 and B4/P4099 used FULL for admission and measurement. These
are producer receipts; independent ordinary-Engine output replay, full-matrix
collection, MAPE acceptance and immutable Hugging Face publication are pending.
SGLang job 604896 completed four prefill and three decode points, including
inclusive 128K, and passed historical strict readers/aggregation. Ordinary
Engine decode replay matched all twelve requests. Prefill matched fifteen of
sixteen requests; a B2 P7/Q3 request differed from its ordinary B1 replay.
Both TP ranks agreed on the retained state/token evidence. Matched-batch and
chunking controls are required to distinguish numerical differences from a
state error; the original failure remains unqualified and is not discarded.

Formal hardware admission requires native per-worker GB300/sm103 identity,
runtime/source/attempt binding, and distinct native UUIDs when available.
SGLang state tensor devices are bound to those observations. Historical
canaries missing these fields are deliberately denied formal publication;
receipts are never backfilled after measurement.

Native GB300 split/one-shot IndexPool probes in job 604200 found wrong pooled
cache entries for stock vLLM P4097/Q3 (B1/B2) and P4097/Q4 (B1). Aligned
controls and all one-shot oracle cases passed. Producer, reader and Rust FPM
queries reject the conservative unaligned-start contract P%4 != 0 with Q >= 2;
requested coordinates remain in coverage. A source-pinned runtime repair must
pass independent qualification before those points can be collected. The
[versioned repair candidate](../python/aisimulate/collector/fpm_forward/runtime/glm53flash_vllm_kpool_candidate/README.md)
preserves the exact patch, native binary lineage, actual build receipt and
qualification commands. It remains explicitly `NOT_QUALIFIED`; building and
installing a wheel does not remove the stock-runtime gate. No new profile is
claimed as accepted.

Formal collection retains at least five warmups and ten observations per point, and separate calibration/holdout token streams and geometry. Exact table self-queries verify integrity, not independent accuracy. Record complete coverage, phase MAPE, WAPE and tail errors. Existing data remains unchanged.

Track [AIC-1999](https://linear.app/nvidia/issue/AIC-1999). SOL is the common prerequisite; FPM and Ops can progress in parallel after their shared execution contract is established.

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

The SGLang driver uses the native Scheduler, ScheduleBatch and request/cache
objects. Its retained-request benchmark protocol seeds each request through
actual forwards, parks completed prefixes in the native ChunkCache, then
forms the frozen homogeneous target batch from those same requests. Decode
also executes its real admission forward. Actual KV/Mamba slots, prefix maps,
input/output tokens and every TP rank's completed target/release receipts are
validated. Arbitrary prefix counters and fake cache state are forbidden. The
protocol requires its own GPU and ordinary-Engine output qualification; CPU
lifecycle tests alone do not grant admission.

SGLang timing is rank zero's native DeviceTimer forward interval, including its
native logits boundary. Token readback happens outside that interval and is
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


## Weight sizing during rendering

The Generator reuses the shared GLM SOL resident-weight model for the native
checkpoint's mixed precision and layer schedule. Its TP1 weight estimate still
feeds the existing 1.5× naive sizing heuristic; it is not a measured memory
qualification and excludes serving cache, CUDA graphs and workspace. Frozen
FPM plans require no additional model lookup while rendering. Requested FPM TP2
and TP4 configurations still use their explicitly frozen pure-TP topology.
