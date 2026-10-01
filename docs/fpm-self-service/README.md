<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FPM self-service overview

## Motivation and when to use self-service

AISimulate needs engine forward-pass timings to predict serving performance.
Existing performance data may not represent your checkpoint, hardware or engine
configuration. FPM self-service lets you collect whole-forward measurements on
the target configuration and use them in AISimulate prediction and recommendation.

Use this workflow when:

- A custom engine build, newer kernel, quantization mode or CUDA Graph policy
  changes iteration latency.
- Your GPU, interconnect, parallel configuration or workload shapes lack measured
  coverage.
- You need the combined cost of operations, launch overhead and communication
  within the measured iteration, rather than composing separate operation tables.
- You want to calibrate a supported deployment while retaining AISimulate's
  existing scheduler, routing and KV-cache simulation.

**Self-collection** runs Dynamo self-benchmark on the target engine. **Onboarding**
validates the measured timing profile and its deployment/resource metadata, then
loads them through the canonical performance-model API. Direct FPM interpolation
does not require a new analytical model class, per-operation GPU profiling, a
registry service or regression training.

If matching measurements already exist, reuse their verified data and provenance
instead of collecting again. A timing table and a resource profile are separate
artifacts: the table alone does not establish serving memory capacity. The
[examples](examples.md) cover both importing an existing Kimi profile and
collecting a new MiniMax profile.

### Parallelism within an iteration

Self-service is intended to cover parallelism whose computation and communication
fit inside the measured forward iteration while preserving Replay's iteration-level
execution contract. Whole-forward measurements include those costs without needing
an analytical model for every internal operation. DCP is already supported by the
measured-FPM consumer and engine Replay, as demonstrated by the
[Kimi TP8+DCP8 example](examples.md#example-a-onboard-the-collected-kimi-k3-tp8dcp8-profile).

Timing is only part of that contract. Keep the actual per-rank work, topology,
cache geometry and usable capacity consistent with the measured deployment;
DCP changes KV sharding and therefore cannot reuse TP-only resource assumptions.
The supported timing paths and the configurations exposed by the guided CLI
are tracked separately in the [parallelism implementation details](implementation.md#parallelism-support-and-guided-cli-coverage).

### When this workflow does not apply

Self-service supplies customized **engine step times** for a measured identity
and supported query shapes. It does not implement new serving behavior. Layer-wise
KV transfer, a custom scheduler or overlap policy, and unsupported recurrent-cache
semantics need corresponding Replay support; more timing samples cannot add it.
If a parallel execution scheme changes scheduling or dependencies across
iterations, such as an overlapping pipeline, one measured forward time alone
cannot represent those changes.

It also does not qualify end-to-end latency or throughput by itself. TTFT,
ITL/TPOT and throughput depend on scheduling, traffic, cache behavior and transfers
as well as forward time. Validate them separately against matched serving runs.
For online learning from per-iteration telemetry, use the separate
[regression workflow](../fpm-recursive-regression.md).

## Support

| Area | Current support and boundary |
| --- | --- |
| Self-benchmark backend | vLLM configurations that pass model/runtime qualification. SGLang and TensorRT-LLM are not supported by this guided collection workflow yet; this does not describe their broader AISimulate modeling support. |
| Model metadata | A local pinned `config.json` or an existing FPM profile. Direct interpolation can use a verified architecture without a registered analytical class. New architectures may need benchmark state preparation, measurement hooks or an audited resource-observation adapter. |
| Parallelism | Whole-forward timing can capture intra-iteration computation and communication. DCP is supported through measured FPM and engine Replay. Exact topology/resource identity and query coverage remain required; see [current guided CLI coverage](implementation.md#parallelism-support-and-guided-cli-coverage) for collection automation limits. |
| Serving roles | Aggregated workers collect both prefill and decode. Prefill and decode can also be collected independently with separate accepted configurations. Their role exports must be composed separately for serving; `onboard validate-fpm` currently validates aggregated replay. |
| GPU execution | Kubernetes by default, or Slurm/Pyxis inside a caller-owned allocation. Both have multi-node launch paths; placement and GPU requirements come from the generated plan. The workflow does not provision the cluster or acquire a Slurm allocation. |
| Cache and multimodal scope | The profile route covers supported linear/grouped cache layouts and text decoders. Grouped Replay currently requires cold aggregated execution, HBM-only cache, no speculation and no prefix caching. A multimodal checkpoint's text profile excludes encoder, projector and preprocessing latency. |
| Readiness | Successful planning or ordinary engine serving does not prove self-benchmark support. Qualify state seeding, actual scheduled work and timing on the exact runtime with bounded smoke measurements before full collection. |

Use a source revision containing the required CLI and consumer features; a
published wheel may predate this guide. See [environment setup](implementation.md#prepare-the-environment),
[executor prerequisites](implementation.md#choose-the-collection-executor) and
[execution-route constraints](implementation.md#choose-the-model-execution-route).

## Workflow

The six stages below match the [agent onboarding procedure](implementation.md#onboard-with-an-agent).
Keep one session checkpoint and track every selected configuration independently.
An existing profile still needs identity, resource and coverage review; it can
skip new GPU collection when matching evidence is available.

### 1. Inspect the model and target

Identify the pinned checkpoint, accessible local model config, GPU system and
runtime. Inspect metadata before asking for facts that can be derived from it.
Record unsupported or unresolved architecture/cache details. Create the session
checkpoint early so partial investigation survives a restart.

Details: [model inspection](implementation.md#1-inspect-the-model-and-target),
[environment setup](implementation.md#prepare-the-environment), and
[checkpoint/resume](implementation.md#checkpoint-and-resume-an-onboarding-session).

### 2. Choose the worker and collection limits

Choose aggregated or independent prefill/decode roles, exact parallelism,
supported precision, runtime version and image. Review context, scheduled-token
and sequence limits, memory fraction and CUDA Graph policy. Treat these as
collection inputs; fixed request lengths, latency targets and validation traces
do not determine the sampling grid. Selected topology establishes minimum GPU
needs, while actual allocation and placement are checked before execution.

Details: [runtime investigation](implementation.md#investigate-runtime-constraints-and-precision-options),
[collection limits](implementation.md#runtime-and-collection-limits), and
[serving roles](implementation.md#serving-roles-and-cuda-graph-settings).

### 3. Derive, review and save the profile

Use `onboard init --model-config` or a supplied `--fpm-profile` to prepare a
request for each configuration. Review identity, precision, topology, cache
geometry, assumptions and their sources before accepting it. Config-derived
memory can remain pending until runtime observation; do not replace unknown
memory with guessed byte bounds. Save each draft, decision and accepted request.

Details: [profile review](implementation.md#3-derive-review-and-save-the-profile),
[local config intake](implementation.md#start-from-a-local-model-config),
[multiple configurations](implementation.md#onboard-multiple-parallel-configurations),
and [runtime probing](implementation.md#resolve-cache-geometry-with-a-runtime-probe).

### 4. Plan collection

`onboard plan` saves the request, profile, collector commands and simulation
inputs. Preview `onboard collect-fpm` and inspect the generated collector plan,
executor requirements and node placement. AISimulate supplies the reviewed
limits; Dynamo self-benchmark derives the exact feasible grid from the initialized
engine. A plan does not establish target readiness or measured coverage.

For existing data, compare its identity, provenance and coverage with the intended
deployment before following the import path. The [Kimi example](examples.md#example-a-onboard-the-collected-kimi-k3-tp8dcp8-profile)
demonstrates the supported SDK and engine Replay path for a measured DCP profile.

Details: [plan and preview](implementation.md#plan-preview-and-explicitly-execute),
[executors](implementation.md#choose-the-collection-executor), and
[sampling-grid ownership](implementation.md#how-the-collection-grid-is-determined).

### 5. Collect, verify and finalize

For new measurements, run `onboard collect-fpm --execute --smoke`, inspect
readiness for every required phase, then run formal collection with `--execute`.
Smoke retains diagnostics but publishes no formal timing pair. The collector
launches benchmark workers; it does not need a prestarted HTTP serving endpoint.

Inspect the Parquet/metadata pair and run `onboard validate-collection` for native
campaign quality. If memory is pending, `onboard finalize` uses verified runtime
observations to write a fresh resolved plan, whose resource profile is reviewed
separately. Imported historical profiles retain their original provenance and
consumer limitations; successful loading does not manufacture missing quality
or memory evidence.

Details: [smoke/readiness](implementation.md#check-timing-readiness-before-full-collection),
[pair installation](implementation.md#validate-and-install-the-fpm-profile),
[collection quality](implementation.md#stage-5-collection-quality),
[memory finalization](implementation.md#finalize-runtime-memory), and
[recovery](implementation.md#recovery-and-cleanup).

### 6. Validate and use the model

Load the exact profile through `RustForwardPassPerfModel.best_available`, select
`fpm_interpolation` with denied fallback and `method: direct` for the guided
profile route, and check measured-point queries. `onboard validate-fpm` then
checks aggregated trace coverage. Use the accepted plan's generated `predict`
and `recommend` configurations for ordinary simulation.

Keep collection quality, resolved memory, query coverage and measured serving
accuracy as separate results. Missing query coverage is not fixed by changing
identity labels or silently choosing another estimator. Use `onboard validate-serving`
for matched end-to-end assessment; exploratory simulation with resolved resources
does not itself qualify accuracy.

Details: [SDK construction](implementation.md#construct-and-check-the-performance-model),
[Replay validation](implementation.md#validate-fpm-query-coverage-with-agentx-replay),
[ordinary configurations](implementation.md#run-the-generated-ordinary-configurations),
and [serving validation](implementation.md#stage-6-matched-serving).

Continue with the [worked examples](examples.md) or the
[implementation and CLI reference](implementation.md). For analytical model
registration and SOL transfer, see the optional [model-integration guide](model-integration.md).
