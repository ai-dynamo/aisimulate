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

GPU qualification and producer qualification are separate. Native vLLM FP8 and NVFP4, each at TP2/TP4, have completed real requests on GB300, including 131008 input tokens plus 32 decode tokens with max model length 131072. This does not qualify the exact inclusive 128K timing boundary: decode past KV is at most 131071. The first formal producer canary failed before any measured point because inherited synthetic-prefix admission required block-aligned cached lengths. The GLM producer now admits exact same-request token prefixes against native allocator capacity, disables cross-request prefix caching, and preserves the failed tail point for a fresh GPU attempt. SGLang native qualification, full data matrix, independent MAPE acceptance and immutable Hugging Face publication are pending. No new profile is claimed as accepted.

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

The SGLang driver uses its unchanged native scheduler and observes actual
coordinates. A requested cached extension that the native scheduler does not
produce is missing coverage, never a substituted geometry. With a fixed native chunk budget, ordinary Engine scheduling cannot realize
arbitrary cached-prefix lengths or homogeneous cached-prefill batches. Such
points remain missing; the retained-state native benchmark protocol needs its
own qualification before those points can be published. Long contexts are
initialized by real forwards on the same requests. Every TP rank retains the
actual input IDs and completed prefix chain, native DeviceTimer intervals,
graph mode/padding, and allocated KDA/MLA/index-tail tensor layout.

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
for reproducible campaign construction and independent acceptance. The 547
default geometries per cell are unqualified candidates, not an accepted dataset.
