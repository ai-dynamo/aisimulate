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

### Performance within an iteration

Self-service captures performance changes within a forward iteration, including
changes to kernels, quantization, CUDA Graph execution and parallel computation
or communication, while preserving Replay's iteration-level execution contract.
Whole-forward measurements include their combined cost without needing an
analytical model for every internal operation.

Timing is only part of that contract. Keep the actual per-rank work, topology,
KV cache geometry and usable capacity consistent with the measured deployment.
The configurations currently exposed by the guided CLI are described in the
[parallelism implementation details](implementation.md#parallelism-support-and-guided-cli-coverage).

### When this workflow does not apply

Self-service supplies customized **engine step times** for a measured identity
and supported query shapes. It does not implement new serving behavior. Layer-wise
KV transfer, a custom scheduler or overlap policy, and unsupported recurrent-cache
semantics need corresponding Replay support; more timing samples cannot add it.
If an execution change alters scheduling or dependencies across
iterations, such as an overlapping pipeline, one measured forward time alone
cannot represent those changes.

It also does not qualify end-to-end latency or throughput by itself. TTFT,
ITL/TPOT and throughput depend on scheduling, traffic, cache behavior and transfers
as well as forward time. Validate them separately against matched serving runs.
For online learning from per-iteration telemetry, use the separate
[regression workflow](../methods/online-regression.md).

## Support

| Area | Current support and boundary |
| --- | --- |
| Self-benchmark backend | vLLM configurations that pass model/runtime qualification. SGLang and TensorRT-LLM are not supported by this guided collection workflow yet; this does not describe their broader AISimulate modeling support. |
| Model metadata | A local pinned `config.json` or an existing FPM profile. Direct interpolation can use a verified architecture without a registered analytical class. New architectures may need benchmark state preparation, measurement hooks or an audited resource-observation adapter. |
| Parallelism | Whole-forward timing can capture intra-iteration computation and communication. Exact topology/resource identity and query coverage remain required; see [current guided CLI coverage](implementation.md#parallelism-support-and-guided-cli-coverage) for collection automation limits. |
| Serving roles | Aggregated workers collect both prefill and decode. Prefill and decode can also be collected independently with separate accepted configurations. Their role exports must be composed separately for serving; `onboard validate-fpm` currently validates aggregated replay. |
| GPU execution | Kubernetes by default, or Slurm/Pyxis inside a caller-owned allocation. Both have multi-node launch paths; placement and GPU requirements come from the generated plan. The workflow does not provision the cluster or acquire a Slurm allocation. |
| KV cache | Supports standard per-token KV caches, as used by Llama, Qwen3 and MiniMax-M2. Models with linear-attention state, such as Kimi K3's KDA, use the [measured-profile path](examples.md#example-a-onboard-the-collected-kimi-k3-tp8dcp8-profile), with the KV cache and state-management limits described in that example. For linear-attention and recurrent-state configs, `onboard init --model-config` stops with guidance to run the [runtime probe](implementation.md#cache-geometry-pending-the-runtime-probe), which measures the cache geometry; importing the recurrent-state cache groups a probe reports is planned. Other KV cache types may also work, but may need changes to the self-benchmarker to initialize their state and measure the forward pass correctly. |
| Multimodality | For multimodal models, this workflow covers text generation only; image/audio processing time is excluded. |

Use a source revision containing the required CLI and consumer features; a
published wheel may predate this guide. See [environment setup](implementation.md#prepare-the-environment),
[executor prerequisites](implementation.md#choose-the-collection-executor) and
[execution-route constraints](implementation.md#choose-the-model-execution-route).

## Reuse a published profile

Published FPM tables use checked-in pin manifests under
[`aisimulate_core/systems/profiles/`](../../../python/aisimulate/src/aisimulate_core/systems/profiles/).
Each model family has an `hf_dataset.json`; additional manifests can preserve
separate historical campaigns, such as `dsv41_fpm/four_gpu_hf_dataset.json`.
Discover the bundled manifests and their profile keys without downloading data:

```bash
python -m aisimulate_core.sdk.fpm_dataset --list
python -m aisimulate_core.sdk.fpm_dataset --list kimi_k3_fpm
```

Listing includes each profile's admission and, when recorded, `admission_reason`.
Review that reason before opting in: some entries provide original artifacts that
the native consumer cannot load. The opt-in warning also reports this reason.

Materialize one key to obtain a verified systems root. This Kimi profile is
development data, so reproducing it requires an explicit opt-in:

```bash
python -m aisimulate_core.sdk.fpm_dataset \
  kimi_k3_fpm gb300-vllm-mxfp4-tp8-dcp8-primary-v0.29.0 \
  --allow-unqualified
```

The CLI prints the resulting root. The equivalent SDK call is
`materialize_fpm_profile("kimi_k3_fpm", key, allow_unqualified=True)` from
`aisimulate_core.sdk.fpm_dataset`. Both interfaces also accept an explicit
manifest file path. Supply the returned root in `systems_paths` when constructing
the canonical performance model or in `engine.systems_paths` for prediction.
See [Example A](examples.md#a1-import-the-measured-pair) for the complete Kimi
identity, resource settings and Replay input; a key alone does not configure them.

Each profile gets its own cache root, even when another profile uses the same
system, backend and version. The native loader reads one
`data/<system>/<backend>/<version>/fpm_forward_perf.parquet` and its
`fpm_forward_perf.metadata.json` pair per root. Do not combine distinct
deployments into that path or put multiple alternative roots in a request as a
substitute for selecting its exact profile.

Downloads occur only on explicit materialization. File hashes are checked on
download and every cache reuse. `--local-files-only` (SDK:
`local_files_only=True`) permits verified cached data only and never downloads;
`--cache-dir` (SDK: `cache_dir`) selects the cache location. Keep materialized
roots unchanged. A new pin produces a new cache root, while corrupt cached
files fail verification. For profiles with `packaged_system`, the cache identity
also includes the installed hardware YAML's actual SHA-256. If those bytes change
after a package update, materialize a new root. Offline use requires a cache
prepared for the currently installed hardware bytes as well as the dataset pin.

### Pin format and admission

| Field | Meaning |
| --- | --- |
| `format_version` | `1` for the pin-manifest format; independent of the timing sidecar schema version. |
| `repo_id` | Hugging Face dataset repository, such as `nvidia/aisimulate-fpm-dataset`. |
| `revision` | Full lowercase 40-character dataset commit SHA; branches and tags are rejected. |
| `profiles.<key>.admission` | `serving`, `development`, or `quarantined`. Only `serving` materializes by default. |
| `profiles.<key>.files` | Pinned dataset files, including the timing pair and an archived system YAML when available. Retain published resource/provenance artifacts where available. |
| `files[].path` | Relative POSIX source path within the pinned dataset. |
| `files[].sha256` | Full lowercase SHA-256 of the exact source bytes, also verified after materialization. |
| `files[].target` | Relative POSIX destination within this profile's systems root. |
| `profiles.<key>.packaged_system` | Optional system name, such as `gb300`, that explicitly requests the installed `aisimulate_core/systems/<system>.yaml` as a hardware reference. Its actual hash participates in cache identity and verification. |
| `profiles.<key>.system_config_source` | Provenance annotation. `packaged_reference_not_collection` identifies a reference YAML when the dataset lacks the collection-time YAML. |
| `profiles.<key>.admission_reason` | Reason for the declared admission, including missing qualification evidence or native-consumer incompatibility. Exposed in profile listings and opt-in warnings when recorded. |
| `profiles.<key>.identity` | Legacy descriptive metadata only. The downloader does not compare it with table rows or a requested deployment. |

Name new keys `<system>-<backend>-<quant>-tp<N>`, adding deployment qualifiers
such as DCP/EP, execution scope, KV precision, or runtime version when needed to
distinguish entries. Quantization in a key is a readable shorthand; mixed-precision
deployments still require every actual precision field from their measurements.
Existing keys remain valid. New manifests omit `identity` to avoid duplicating
the authoritative table/sidecar identity. Existing `identity` blocks remain for
compatibility and cache stability: the complete profile entry contributes to
the cache fingerprint, but this is not an identity validation step. Native
loading and canonical model queries still enforce their schema, provenance and
exact deployment checks.

Prefer the archived collection-time system YAML. Some historical publications
lack that artifact; their new pins explicitly request the installed hardware YAML
through `packaged_system`,
with `system_config_source: packaged_reference_not_collection` and are not
serving-admitted. This current hardware reference supplies loader configuration;
it does not establish the collected system's settings. The materializer copies
only that hardware YAML from the package. Attention-lane defaults retain the
loader's ordinary package fallback, and the profile root has no release-slot
aliases from `query_versions.yaml`: use the exact recorded `backend_version`.
Do not present installed references as collection provenance or silently replace
an archived YAML. DeepSeek V4.1 and GLM-5.3-Flash retain their existing archived
system configurations and admission decisions.

MiniMax-M3-NVFP4 currently has published measurement artifacts but no loadable FPM
Parquet/metadata pair at the pinned dataset revision, so it has no materialization
key. A follow-up publication must supply the pair and its provenance before adding
a pin. Across the six newly indexed model families, 30 entries are `development`
and three are `quarantined`: the Kimi synthetic-attention comparator and two
MiniMax-M2.7 H200 TP4/EP4 KV-on/off artifacts. Those MiniMax sidecars lack
`coordinate_system`, `measurement_policy`, and top-level `system` and
`backend_version`, so the native FPM consumer cannot load them.
`--allow-unqualified` retrieves these original artifacts; it does not repair
their metadata, make predictions possible, or promote their admission.

### Maintain the pins

The collection or dataset-PR author owns the AISimulate follow-up PR after a
dataset change merges; the performance-model CODEOWNERS review it. Pin the merged
dataset commit, verify every file hash and canonical target, retain provenance
and applicable notices, and review each profile's admission against its evidence.
Hardware-reference changes belong in the canonical packaged system YAML; the
materializer accounts for changed bytes when selecting a cache root. Do not
advance a pin automatically or change admission merely because a download or
smoke query succeeds. Changes to these manifests go through ordinary repository
review; no registry service is required.

Materialization verifies storage integrity and applies the manifest's admission
gate. It does not establish resource capacity, workload coverage, or prediction
accuracy. Preserve these separate results and perform
[matched serving validation](implementation.md#stage-6-matched-serving) for the
intended deployment. This reuse path also does not expand guided onboarding's
backend, topology, or execution-identity support, or the built-in model registry.
For example, the current model-name construction path rejects Inkling's
`InklingForConditionalGeneration` architecture even though its artifacts can be
materialized; a compatible consumer is still required.

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
supported precision, runtime build and image. An explicit backend version string
is a dataset identity label and may differ from the observed package version;
retain the actual runtime evidence separately. Omit `--framework-version` to
resolve it from the collection image when execution starts. The optional Dynamo
version is provenance only. Review context, scheduled-token
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

With no backend version supplied, init/plan/preview leave the label pending.
For both explicit and omitted labels, first execution requires an explicit image
and observes its package version without GPUs before creating a
`runtime-resolved/` child plan. An explicit label is preserved; an omitted label
takes the observed version. Continue with the resolved paths printed by the
command; the original registration stays intact.

For existing data, compare its identity, provenance and coverage with the intended
deployment before following the import path. The [Kimi example](examples.md#example-a-onboard-the-collected-kimi-k3-tp8dcp8-profile)
demonstrates the supported SDK and engine Replay path for a measured DCP profile.

Details: [how self-benchmarking works](implementation.md#how-self-benchmarking-works),
[plan and preview](implementation.md#plan-preview-and-explicitly-execute),
[executors](implementation.md#choose-the-collection-executor), and
[sampling-grid ownership](implementation.md#how-the-collection-grid-is-determined).

### 5. Collect, verify and finalize

For new measurements, run `onboard collect-fpm --execute --smoke`, inspect
readiness for every required phase, then run formal collection with `--execute`.
Smoke retains diagnostics but publishes no formal timing pair. The collector
launches benchmark workers; it does not need a prestarted HTTP serving endpoint.

Benchmarking typically takes about one hour, depending on the model, hardware
and sampling range.

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
[implementation and CLI reference](implementation.md).
