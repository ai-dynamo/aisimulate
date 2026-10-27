<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Op-level timing

Python resolves the model's phase operations, dimensions, precision, and
parallel placement into an `EngineSpec`. Rust loads the resolved performance
database and evaluates that specification without re-entering Python on the hot
path. Compilation produces a versioned operation plan, not a native executable.
Use the [canonical estimator configuration](../configuration.md); low-level
compiled-engine entry points are documented in the [Python](../api/python.md)
and [Rust](../api/rust.md) references.

## Operation composition

An operation can use a collected table, a table interpolation, or its supported
analytical formula. The model graph determines repetition, communication,
overlap, and phase boundaries. Summing arbitrary kernel timings outside that
graph can double-count fused modules or omit dispatch work. The returned
per-operation tuple is `(name, latency_ms, energy_wms, source)`; zero energy
means missing energy evidence, not a zero-power operation.

| Database mode | Meaning |
| --- | --- |
| `SILICON` | Collected-data lookup/interpolation for table-backed operations, including approved shared sources; no HYBRID transfer rescue. Inherently analytical operations retain their own source. |
| `HYBRID` | Try collected data, then an implemented empirical/transfer path for a data gap. |
| `EMPIRICAL` | Use the implemented data-calibrated empirical path. Calibration is still required for table-backed operations. |
| `SOL` / `SOL_FULL` | Analytical speed-of-light paths where implemented; detailed decomposition can have narrower support. |

Data-source inheritance and empirical transfer are different. A row inherited
from an approved measured donor is still collected data. An estimated
cross-shape or cross-precision transfer must retain its transfer provenance.

## External execution profiles

An externally stored profile can supply a qualified operation graph and exact
measured composite scopes without adding its measurements or runtime identity
to the package. It uses the same `RustForwardPassPerfModel.best_available`
constructor and Rust operation executor as ordinary op-level prediction.

| Field | Default | Meaning |
| --- | --- | --- |
| `estimator_config.op_level.execution_profile` | Unset | Opt in to qualified external phase graphs, including an aggregated mixed-step composition. |
| `execution_profile.path` | Required when selected | Absolute path to the external JSON profile. |
| `execution_profile.sha256` | Required when selected | Lowercase SHA-256 of the exact profile bytes; preserved in saved configuration and provenance. |

Set `estimation_mode: op_level`, `fallback_policy: deny`, `database_mode: SILICON`,
`enable_shared_layer: false`, and `estimator_config.correction.enabled: false`.
Choose `worker_type: prefill`, `decode`, or `aggregated`, an explicit backend version and a
single PP/attention-DP/CP/DCP rank. Keep `encoder_parallel` unset: these profiles
qualify the language-model phase only. Speculation, decoder replay, automatic
estimator selection and shared-source inheritance are not supported.
Supply the external hardware and performance-data root through `systems_paths`.

The version-1 profile JSON has these fields:

| Field | Contract |
| --- | --- |
| `schema_version` | `1`. |
| `basis` | The complete ordinary compiled `EngineSpec` used to qualify the profile. Both operation lists and every engine identity field must match the current compiler output; only `engine.systems_path` is excluded from comparison. |
| `worker_type` | `prefill`, `decode`, or `aggregated`, matching the caller. |
| `ops` | Replacement operations for the selected phase, or the prefill graph for `aggregated`, in execution order, using the existing externally tagged operation format. Resident weight bytes must match the original phase. |
| `generation_ops` | Required for `aggregated`; absent for other roles. Replacement decode graph, with resident weight bytes matching the original decode graph. |
| `mixed_composition` | Required for `aggregated`; absent for other roles. Set to `shared_non_attention` to declare the three-pass composition described below. |
| `retained_files` | Bundle-relative `path`, lowercase `sha256`, and `size_bytes` for every required hardware YAML, source/version metadata file and retained operation table. |

Profile authors can inspect an ordinary `compile_engine` result using the
internal `_native.engine_spec_json_from_bincode(bytes)` helper. This is an
authoring aid, not a separate prediction constructor. Compile that basis
against the retained inventory: a systems root that holds exactly the files
listed in `retained_files`. The compiler reads the tables present in its root
(for example, attention `lane_order` grows with the kernel sources it
discovers), so a basis compiled from a fuller collection root does not match
the consumer's compile from the prepared root, and admission fails naming the
first differing field. Compiler/schema changes that alter the basis require
requalification and a new profile pin.

`MeasuredComposite` represents a measured scope, which may contain multiple
kernels or operations, as one operation. Its internal operations are not modeled
individually; the name identifies the scope and does not imply confidentiality.
It is an exact, latency-only operation with `name`, `system`,
`backend`, `version`, `is_context`, `weight_bytes`, `scale_factor`, and `cells`.
Each cell contains `shape` and `latency_ms`. Sort cells by their unique shape:

- Prefill: `[batch, total_input_length - prefix, prefix]`, where input length
  and cached prefix are per request; the second entry is the new token count.
- Decode: `[batch, total_past_kv_tokens / batch + 1, 0]`, where the second entry
  is the per-request attention length (past KV plus the current token).

Latencies and repetition factors must be finite and positive. The operator
multiplies the measured scope latency by its repetition factor. It performs
no interpolation, extrapolation, correction, or SOL substitution. Missing
cells fail explicitly. Exact measured results report `silicon` source and zero
energy to indicate missing energy evidence. Existing operations keep their
own lookup rules. Profiles may use `Overlap` for parallel operation groups;
fallback graphs, speculative token scaling, whole-forward, vision and other
nested stage wrappers are rejected.

Construction verifies the profile pin, compares the actual compiled basis,
and copies verified retained-file bytes into engine-owned storage. Configuration
normalization checks the selection and supported controls without opening the
profile; its contents and worker phase are checked at construction. Lazy table
reads use that snapshot, including on their first query. Changed files fail a
new construction; they cannot change an existing model. Bundle paths cannot
escape the root or traverse symlinks; hardware `data_dir`, backend versions and
`misc.nccl_version`/`misc.oneccl_version` must remain bundle-relative. Missing retained inputs fail normally
during database loading or readiness checks.

The model returned by `RustForwardPassPerfModel.best_available(config)` accepts
`estimate_forward_pass_time_ms(metrics_by_rank)` and
`estimate_forward_pass_detailed(metrics_by_rank)` for execution profiles. Callers
use the same interface as other estimators; AISimulate dispatches the scheduled
work to the selected profile. Queued metrics do not contribute to the estimate.

Supported metrics contain one rank: equal uncached prefill prompts,
one prefill request with a known KV prefix, or one decode step with equal context
lengths. An aggregated profile also accepts both supported phase shapes in the
same step. Totals must identify integral per-request lengths. Prefill prompt variance
does not establish equal chunk/prefix pairs, so multi-request prefill with nonzero
past KV remains unsupported. Existing aggregate fields are used; no telemetry
schema change is required. Missing cells, unsupported shapes, and a mismatched
phase raise the same prediction errors through the common interface, without
switching estimators or relaxing the selected fallback policy.

For mixed steps, an aggregated profile declares `mixed_composition:
shared_non_attention` and supplies both phase graphs. This is an author-supplied
modeling contract: the graphs must separate attention from shared non-attention
work. AISimulate reuses its three-pass operation composition:

1. Query prefill non-attention operations once at the combined new-token count
   (prefill tokens plus decode requests), with batch 1 and the total prefill prefix.
2. Query prefill attention at the prefill batch's own new-token and prefix lengths.
3. Query decode attention at the decode batch's own length, including its current token.

Use the existing top-level `context_attention` and `generation_attention` names
for the attention scopes (typed GLM-5.3 attention operations already carry their
phase). Other scopes belong to shared non-attention work; decode non-attention
operations are used for decode-only steps. Do not place attention inside an
`Overlap` group. Measured scopes can use this contract when attention and shared
work have been measured separately; an opaque whole-phase measurement cannot
establish the split. The declaration does not validate that decomposition against
silicon. For measured scopes, every shape queried by these passes needs an exact
cell, including combined-token shapes. Separate complete prefill/decode timings
are not added together.

For example, 10 prefill tokens in two equal prompts and two decode requests with
four past KV tokens each query shared work at `[1, 12, 0]`, prefill attention at
`[2, 5, 0]`, and decode attention at `[2, 5, 0]`. Existing single-phase profile files
remain valid; combining two such files does not implicitly create an aggregated
profile. Dataset preparation accepts an aggregated profile through the same
`prepare_execution_profile` function and manifest schema.

Direct `predict_prefill_latency(batch, total_input_length, prefix)` and
`predict_decode_latency_total(batch, total_past_kv_tokens)` calls remain supported.
Multi-step decode trajectories and the static `mixed_step_latency` surface remain
unsupported for profiles; use the common forward-pass API for mixed work. Successful
queries do not establish scheduler-inclusive Replay, Planner sizing, or end-to-end
serving accuracy. Caller workload approximations remain the caller's responsibility.

Profile files, saved configurations and diagnostics can reveal runtime
identities or private paths. Keep them with their source dataset; the package
does not upload or download profiles automatically.

### Shared dataset layout and local preparation

Public and private repositories use the same additive `catalog/dataset.json`
contract (`aisim-dataset-index/v1`). Category entries reference hashed indexes;
`aisim-category-index/v1` lists configuration identities and hashed manifests.
Repository identity and authentication do not change the schema. Existing FPM
catalogs remain accessible through an explicit `legacy-fpm` reference; their
measurement protocols and historical paths are preserved.

Operation tables live under
`data/op-latency/<hardware>/<operation-family>/<framework>/<version>/`.
They retain their original Parquet bytes, columns, units and collection metadata.
Model-specific execution profiles and hardware descriptions live under
`metadata/`; original input evidence belongs under `provenance/`. Predictions,
comparisons and gap reports remain outside the dataset.

An `aisim-artifact-manifest/v1` operation manifest records:

| Field | Contract |
| --- | --- |
| `category`, `purpose` | `op-latency`, with `operation-tables` or `execution-profile`. |
| `configuration_id`, `identity` | SHA-256 of the versioned canonical category, purpose and effective identity; preserve unknown fields as null with an unresolved source discriminator. Operation-table identities have no model partition. |
| `requested_settings` | Original requested settings, or null; never substitute these for observed effective settings. |
| `artifacts` | Stable ID, repository-relative path, role, format, schema identifier, SHA-256, bytes and Parquet row count. Optional `runtime_path` maps an input to the local systems directory. |
| `provenance`, `evidence_gaps` | Original identities/revisions, collection evidence and explicit missing or unsupported facts. |

Use the shared Python API with an already downloaded local selection:

```python
from pathlib import Path
from aisimulate_core.datasets import check_manifest, prepare_execution_profile
from aisimulate_core.sdk import RustForwardPassPerfModel

root = Path("/local/pinned-dataset")
manifest = "metadata/execution-profiles/example/prefill.manifest.json"
check_manifest(root, manifest)
config = prepare_execution_profile(root, manifest, Path("/local/prepared-prefill"))
model = RustForwardPassPerfModel.best_available(config)
```

The output directory must not exist. Preparation verifies the declared files,
row counts, configuration identity and the profile's exact retained-input
inventory. The manifest's `hardware`, `framework` and `framework_version` must
match the predictor configuration's `system`, `backend` and `backend_version`.
The model identity may use a model ID while the predictor references a local
model directory. Preparation copies unchanged inputs into the runtime layout,
then calls the canonical constructor to compare the compiled baseline and validate the retained
runtime inputs before returning the configuration with absolute local data paths.
Malformed profiles and incomplete required inputs fail preparation and remove
the partial output. Tables used only by replaced baseline operations are not
required, provided the basis was compiled without them (see above). Shape
coverage remains a query-time check.

Preparation requires a bundled model ID or an existing local model directory;
unbundled remote IDs fail with an instruction to provide local model metadata.
It never changes profile or measurement bytes, accesses the network, runs stored
scripts or makes a prediction. The caller keeps the prepared directory and any
referenced local model directory for saved-config reloads; preparation does not
copy model metadata. `check_dataset(root)`
also verifies shared catalog references; existing FPM protocol validation is
still separate. This adapter does not change the default operation database or
implement the broader E2E/Operation DB migration in issue #435.

## Empirical utilization and transfers

The native [utilization estimator](../../../crates/core/src/perfmodel/operators/util_empirical.rs)
models a calibrated operation as

```text
util = SOL(measured_shape) / measured_latency
latency(query) = SOL(query) / util(query)
```

`util` is an effective calibration factor, not a physical efficiency bounded
by one. Categorical/kernel slices are selected before numeric interpolation.
Within a slice, numeric coordinates are normalized in log space; the grid
preserves exact hits, clamps each axis to the measured range, and blends the two
nearest samples with inverse-distance weights (`k=2`, `p=1`). It does not need a
Cartesian measurement grid. No calibration and no permitted donor means an
explicit empirical-coverage error, not `SOL / constant`.

Transfer kinds are admitted by `transfer_policy`:

| Kind | Reference | Correction |
| --- | --- | --- |
| `xshape` | Another collected shape slice with the same precision | Query SOL with the selected reference utilization. |
| `xquant` | Another precision with the same memory/compute profile | Same SOL coefficients; no profile-level correction. |
| `xprofile` | A different memory/compute precision profile | Multiply reference utilization by the query/reference efficiency-level ratio. |
| `xop` | A related operation | Explicit operation-specific level alignment. |

Policies are `off`, `conservative`, `balanced`, and `aggressive`, or the
supported explicit set. The resolved policy is part of configuration and
provenance. No transfer policy turns a missing runtime implementation into
measured support. Windowed-attention fallbacks use a window-aware SOL ratio
only in their implemented empirical path.

## GEMM precision transfer

The [native GEMM operator](../../../crates/core/src/perfmodel/operators/gemm.rs)
uses the query's own SOL cost and a reference utilization:

```text
latency = SOL_query / (util_reference * level(query_profile) / level(reference_profile))
```

GEMM already pools all `(m, n, k)` shapes for one precision, so its `xshape`
reference list is empty. `xquant` uses the first same-profile sibling in table
order. Cross-profile candidates are ordered by compute distance, then memory
distance: a weight-only mode therefore prefers the matching arithmetic family.
MoE keeps its own historical profile-distance ordering. These tie-breaks are
observable prediction semantics, not opportunities to sort data arbitrarily.

`fp8_static` is composite: it needs the base FP8 and its overhead tables.
Transfer reachability alone cannot admit it. Validation requires known
precision profiles and the resolved transfer policy; a pure SILICON request
still needs data. New precision support requires native profile
levels, their exported Python reachability view, and tests described in [Extending](../extending.md).

## Diagnostics and limits

`static_phase_diagnostics` exposes operation timing, sources, measurement
substitutions, and available SOL detail before online correction. Missing SOL
breakdowns carry explicit reasons. The [FPE matrix](../support-matrix.md)
tests strict native coverage; it does not certify the HYBRID ladder or silicon
accuracy. See [DSA CP](context-parallel-dsa.md), [DeepEP-LL](deepep-ll.md),
and [execution limitations](../limitations.md) for specialized boundaries.
