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

The required matrix has eight deployments: vLLM/SGLang × FP8/NVFP4 × TP2/TP4.
No complete deployment has yet passed independent prefill and decode accuracy
acceptance. CPU tests, runtime qualification and individual native completions
are separate evidence; none grants dataset publication or a passing MAPE.

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
and strict prefill/decode readers in all four vLLM deployments. Its formal
397-calibration/221-holdout campaign is collecting on GB300. Current SGLang
qualification uses native 0.5.20, explicit memory fraction 0.82, the native default
allocator for FP8 TP4/NVFP4 TP2/TP4, and the separately recorded 16384 MiB split
limit for FP8 TP2. Allocation and graph policy changes require new qualified
campaign identity; historical unknown allocator settings stay unknown.

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
