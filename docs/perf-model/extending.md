<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Extend performance models

AISimulate owns the application, model definitions, performance data, and native
estimator. Start from the [development environment](../../DEVELOPMENT.md).
All paths below are relative to the repository root unless linked otherwise.
The `aisimulate_core` source namespace is retained inside the `aisimulate`
wheel; installing or rebuilding a separate AIConfigurator package is unnecessary.

For whole-forward FPM onboarding, start with the [self-service config/profile workflow](fpm-self-service/implementation.md#onboard-with-an-agent),
which supports direct interpolation without an analytical model class. The
[SOL-assisted FPM checks](#sol-assisted-fpm-checks) describe the
additional operation, memory and native SOL tests needed when adding an analytical
model for `method: sol`. Neither FPM route requires per-operation GPU timing collection.

## Choose the smallest extension

| What changed? | Start here |
|---|---|
| A new architecture name with an existing operation pipeline | Map the architecture to an existing family in `python/aisimulate/src/aisimulate_core/sdk/common.py`. |
| A new layer composition or model family | Add a registered class under `python/aisimulate/src/aisimulate_core/sdk/models/` and map its architecture. |
| Existing operations need additional measured shapes or a backend version | Add collector cases and collect the missing performance-data cells. |
| A genuinely new operation or execution contract | Extend the native operator and wire contract, then the Python model description and collection path. |

An architecture mapping alone does not establish data coverage, quantization,
KV-memory correctness, Replay support, or prediction accuracy. Validate the
specific model/backend/hardware cell and the public workflow it will serve.

## 1. Resolve and register the model

The [models package](../../python/aisimulate/src/aisimulate_core/sdk/models/README.md) owns
registry-based model construction. Resolution reads model configuration,
resolves the architecture to a family, then selects the `@register_model`
class. `models/blocks/` contains reusable composition helpers and must not
register model classes.

For an existing family, inspect
[`common.py`](../../python/aisimulate/src/aisimulate_core/sdk/common.py) and the selected model
class before adding `ARCHITECTURE_TO_MODEL_FAMILY` entries. Verify layer counts,
attention/KV heads, head dimensions, quantization defaults, and any custom
Hugging Face configuration fields. A model name resembling another model is
not sufficient evidence that their operation pipelines match.

For a new family, implement `BaseModel.create(...)`, register the family, and
build its context/generation operations using the existing family-specific
examples. See the [registry extension examples](../../python/aisimulate/src/aisimulate_core/sdk/models/README.md#adding-a-new-model).
Reuse the shared MoE block builder where its contract applies, including the
separate large-EP registration described there.

Mamba2 kernels and NemotronH hybrid model descriptions already exist in
[`operations/mamba.py`](../../python/aisimulate/src/aisimulate_core/sdk/operations/mamba.py) and
[`models/nemotron_h.py`](../../python/aisimulate/src/aisimulate_core/sdk/models/nemotron_h.py).
They are useful references, not evidence that every Mamba variant or topology
is supported. In particular, AFD partitioning has separate restrictions below.

## 2. Extend the operation contract when necessary

Python describes model work; Rust computes per-operation latency, energy,
and SOL values. Keep that single-oracle boundary when adding an operation:

1. Implement the native operator in
   [`crates/core/src/perfmodel/operators/`](../../crates/core/src/perfmodel/operators)
   and any table loader in
   [`perf_database/`](../../crates/core/src/perfmodel/perf_database).
   Add an independently justified numerical test, including unsupported and
   missing-data behavior.
2. Add the typed Python operation in
   [`sdk/operations/`](../../python/aisimulate/src/aisimulate_core/sdk/operations). Follow its
   existing construction, weight-sizing, and engine-backed table-view
   conventions. Do not add a Python interpolation or performance-query oracle.
3. Extend `_to_opspec` in
   [`sdk/engine.py`](../../python/aisimulate/src/aisimulate_core/sdk/engine.py), the native
   [`Op` representation](../../crates/core/src/perfmodel/operators/op.rs),
   and the [`engine specification`](../../crates/core/src/perfmodel/engine/spec.rs).
   Preserve positional enum compatibility; a schema-breaking change requires
   coordinated versioning and consumer updates.
4. Wire the operation into the model pipeline and validate its parameter
   conversion. Run the existing
   [OpSpec coverage test](../../python/aisimulate/tests/unit/sdk/test_opspec_coverage.py) and
   [single-oracle contract](../../python/aisimulate/tests/cross_package/test_single_oracle_contract.py).
5. Add a representative estimator parity case when a model reaches the new
   path. Follow the [parity README](../../crates/core/parity_tests/perfmodel/README.md)
   and review the numerical evidence before updating goldens.

Read the applicable rules linked from [AGENTS.md](../../AGENTS.md) before
changing collector or generator code. The current operator instructions in
this page use the unified repository layout.

## 3. Collect only the missing data

Use the [Collector README](../../python/aisimulate/collector/README.md) and the collector rules
required by `AGENTS.md`. Cases live under `python/aisimulate/collector/cases/`;
model-specific cases belong under `models/`, and reusable operation shapes
under `base_ops/`. For a new collector operation, follow the
[collector operation runbook](../../python/aisimulate/.claude/skills/aic-collector-op-development/SKILL.md).

Run the intended backend/runtime on the intended hardware, retain the effective
model and runtime identity, and finalize accepted staging output as Parquet
with the required collection/reuse metadata. The canonical data root is:

```text
python/aisimulate/src/aisimulate_core/systems/data/<system>/<family>/<backend>/<version>/
```

Do not relabel another backend version's data as newly measured. Update query
version/support metadata only when the new cell meets its collection and
consumer contracts. For whole-forward profiles, use the separate
[FPM self-service overview](fpm-self-service/README.md).

## 4. Validate the intended public path

Rebuild the editable package after native or packaging changes:

```bash
uv sync --project python/aisimulate --extra dev
source python/aisimulate/.venv/bin/activate
python -m pytest -c python/aisimulate/pytest.ini   python/aisimulate/tests/unit/sdk/test_opspec_coverage.py   python/aisimulate/tests/cross_package/test_single_oracle_contract.py
```

Also run the focused tests for the changed model, collector, and native
operator. On macOS, use the local pytest guidance in `AGENTS.md`.

Check the exact model/system/backend/version with `aiconfigurator cli support`
from this environment, then run the intended `aisimulate predict` or
`recommend` configuration. Keep model revision, precision, topology, workload,
performance-data identity, and reports with the test results. If deployment
artifacts are in scope, exercise their generator path too.

Keep these outcomes separate in the change description: model construction,
required-data coverage, native execution, replay/generator compatibility, and
accuracy against measured hardware. A support check or numerical parity test
alone does not prove end-to-end predictive accuracy.

### AFD execution-plan compatibility

AFD compiles pool-specific operations and a source-ordered layer plan through
the canonical FPM constructor. [`models/afd.py`](../../python/aisimulate/src/aisimulate_core/sdk/models/afd.py)
composes the graph; [native validation](../../crates/core/src/perfmodel/afd/validation.rs)
and the execution-plan validator reject unsupported architectures and invalid
pool/layer/op coverage before prediction.

When extending a model:

- Preserve source layer order and identity in both plans. Support only a contiguous
  dense prefix followed by routed layers until native execution supports more.
- Assign operations using the [existing ownership contract](../sweeper/afd-topology.md#performance-model).
  F must not validate unused attention heads. Preserve A-local TP reductions,
  replace aggregate dispatch with native AFD communication, and expand overlap
  groups before assigning ownership.
- Reject unknown operations and unsupported attention/state stages or routing.
  Add source composition, native timing, physical residency and independent
  regression coverage before declaring support.
- Extend the existing plan rather than adding a classifier or executor. Timing,
  energy and memory must agree on layer/endpoint counts, dense-prefix batch size
  and one-time work; see [pass execution](../sweeper/afd-topology.md#staged-evaluation).

## SOL-assisted FPM checks

Direct self-service can use an unregistered architecture with a verified
profile; SOL-assisted FPM requires an analytical model and support in the native
[`fpm_sol` evaluator](../../crates/core/src/perfmodel/operators/fpm_sol.rs).
An ordinary operator or a `SOL_FULL` diagnostic does not automatically implement
that evaluator's contract.

Before timing collection, record the immutable checkpoint configuration,
actual layer types/counts, attention/KV heads and dimensions, expert geometry,
precision, sharding, runtime kernels, and context/cache layout. Validate:

1. Both context and generation operation lists are nonempty and lower to native
   `OpSpec` values with correct per-rank dimensions and repetition.
2. Weight inventory stays resident even when execution skips work. Cache sizing
   and its inverse agree across window, compression, and page boundaries; a
   one-token slope is insufficient for a nonlinear cache.
3. `database_mode: SOL`, `estimation_mode: op_level`, and `fallback_policy: deny`
   produce finite analytical diagnostics for the intended shapes and phases.
4. Every retained operation also works through FPM's own SOL evaluator.
5. After a reviewed measurement pair is available, explicit
   `estimation_mode: fpm_interpolation`, `method: sol`, and `fallback_policy: deny`
   construct and query the exact model/deployment cell. Keep this distinct from
   the earlier data-free analytical check.

The model keeps its phase operations as `sol_ops` inside the whole-forward
operation, and its weight inventory supplies `weight_bytes`. Do not duplicate
phase replacement, interpolation, or SOL formulas in Python.

## Add a GEMM precision profile

Add the correct memory/compute coefficients to the precision enum and inspect
its consumer graph before claiming support. The native
[`GEMM_QUANT_UTIL_LEVEL`](../../crates/core/src/perfmodel/operators/gemm.rs)
and MoE equivalent own cross-profile efficiency levels. Python's level views
are exported from the native runtime; do not hand-maintain a second level table.
Add a measured or explicitly inferred level for a genuinely new profile,
with its derivation beside the native entry. Validation must admit only the
relations enabled by the chosen transfer policy and available reference data.

Exercise SILICON failure without data, permitted HYBRID/EMPIRICAL transfer,
restrictive-policy failure, reference ordering, and eventual own-data precedence.
`fp8_static` additionally requires its overhead tables. Checkpoint-to-precision
resolution and hardware compatibility are separate contracts; reachability of an
estimate does not prove that the backend can execute the checkpoint.
