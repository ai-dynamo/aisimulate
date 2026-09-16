<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# How to add a new model

AISimulate owns the application, model definitions, performance data, and native
estimator. Start from the [development environment](../../../DEVELOPMENT.md).
All paths below are relative to the repository root unless linked otherwise.
The `aiconfigurator_core` source namespace is retained inside the `aisimulate`
wheel; installing or rebuilding a separate AIConfigurator package is unnecessary.

## Choose the smallest extension

| What changed? | Start here |
|---|---|
| A new architecture name with an existing operation pipeline | Map the architecture to an existing family in `python/aisimulate/src/aiconfigurator_core/sdk/common.py`. |
| A new layer composition or model family | Add a registered class under `python/aisimulate/src/aiconfigurator_core/sdk/models/` and map its architecture. |
| Existing operations need additional measured shapes or a backend version | Add collector cases and collect the missing performance-data cells. |
| A genuinely new operation or execution contract | Extend the native operator and wire contract, then the Python model description and collection path. |

An architecture mapping alone does not establish data coverage, quantization,
KV-memory correctness, Replay support, or prediction accuracy. Validate the
specific model/backend/hardware cell and the public workflow it will serve.

## 1. Resolve and register the model

The [models package](../src/aiconfigurator_core/sdk/models/README.md) owns
registry-based model construction. Resolution reads model configuration,
resolves the architecture to a family, then selects the `@register_model`
class. `models/blocks/` contains reusable composition helpers and must not
register model classes.

For an existing family, inspect
[`common.py`](../src/aiconfigurator_core/sdk/common.py) and the selected model
class before adding `ARCHITECTURE_TO_MODEL_FAMILY` entries. Verify layer counts,
attention/KV heads, head dimensions, quantization defaults, and any custom
Hugging Face configuration fields. A model name resembling another model is
not sufficient evidence that their operation pipelines match.

For a new family, implement `BaseModel.create(...)`, register the family, and
build its context/generation operations using the existing family-specific
examples. See the [registry extension examples](../src/aiconfigurator_core/sdk/models/README.md#adding-a-new-model).
Reuse the shared MoE block builder where its contract applies, including the
separate large-EP registration described there.

Mamba2 kernels and NemotronH hybrid model descriptions already exist in
[`operations/mamba.py`](../src/aiconfigurator_core/sdk/operations/mamba.py) and
[`models/nemotron_h.py`](../src/aiconfigurator_core/sdk/models/nemotron_h.py).
They are useful references, not evidence that every Mamba variant or topology
is supported. In particular, AFD partitioning has separate restrictions below.

## 2. Extend the operation contract when necessary

Python describes model work; Rust computes per-operation latency, energy,
and SOL values. Keep that single-oracle boundary when adding an operation:

1. Implement the native operator in
   [`crates/core/src/perfmodel/operators/`](../../../crates/core/src/perfmodel/operators/)
   and any table loader in
   [`perf_database/`](../../../crates/core/src/perfmodel/perf_database/).
   Add an independently justified numerical test, including unsupported and
   missing-data behavior.
2. Add the typed Python operation in
   [`sdk/operations/`](../src/aiconfigurator_core/sdk/operations/). Follow its
   existing construction, weight-sizing, and engine-backed table-view
   conventions. Do not add a Python interpolation or performance-query oracle.
3. Extend `_to_opspec` in
   [`sdk/engine.py`](../src/aiconfigurator_core/sdk/engine.py), the native
   [`Op` representation](../../../crates/core/src/perfmodel/operators/op.rs),
   and the [`engine specification`](../../../crates/core/src/perfmodel/engine/spec.rs).
   Preserve positional enum compatibility; a schema-breaking change requires
   coordinated versioning and consumer updates.
4. Wire the operation into the model pipeline and validate its parameter
   conversion. Run the existing
   [OpSpec coverage test](../tests/unit/sdk/test_opspec_coverage.py) and
   [single-oracle contract](../tests/cross_package/test_single_oracle_contract.py).
5. Add a representative estimator parity case when a model reaches the new
   path. Follow the [parity README](../../../crates/core/parity_tests/perfmodel/README.md)
   and review the numerical evidence before updating goldens.

Read the applicable rules linked from [AGENTS.md](../../../AGENTS.md) before
changing collector or generator code. The current operator instructions in
this page use the unified repository layout.

## 3. Collect only the missing data

Use the [Collector README](../collector/README.md) and the collector rules
required by `AGENTS.md`. Cases live under `python/aisimulate/collector/cases/`;
model-specific cases belong under `models/`, and reusable operation shapes
under `base_ops/`. For a new collector operation, follow the
[collector operation runbook](../.claude/skills/aic-collector-op-development/SKILL.md).

Run the intended backend/runtime on the intended hardware, retain the effective
model and runtime identity, and finalize accepted staging output as Parquet
with the required collection/reuse metadata. The canonical data root is:

```text
python/aisimulate/src/aiconfigurator_core/systems/data/<system>/<family>/<backend>/<version>/
```

Do not relabel another backend version's data as newly measured. Update query
version/support metadata only when the new cell meets its collection and
consumer contracts. For whole-forward profiles, use the separate
[FPM collection-to-prediction workflow](fpm/end-to-end-workflow.md).

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

### AFD Operation Partitioning Compatibility

Attention-FFN Disaggregated (AFD) estimate mode has one additional maintenance contract beyond the normal aggregated and P/D-disaggregated paths. [`sdk/afd_partition.py`](../src/aiconfigurator/sdk/afd_partition.py) splits a model's `context_ops` / `generation_ops` into A-worker and F-worker pools by operation name. When adding a new model family or new operation, make sure the generated operation names can be classified by the AFD partitioner.

The current AFD partitioning contract is:

1. **A-worker ops**: operations that belong to the embedding / attention side, such as `embedding`, `add_norm_1`, attention norms, `qkv`, MLA, BMM, RoPE, attention kernels, and projection GEMMs.
2. **F-worker ops**: operations that belong to the FFN / MoE side, such as router GEMMs, dense `ffn` / `mlp` ops, routed expert ops, shared expert ops, and activation / gate / up / down GEMMs.
3. **Boundary ops**: operations that naturally sit at the A/F boundary, such as `add_norm_2`, FFN/MoE/MLP norms, `logits_gemm`, `reduce_add`, and `*_combine`. These default to the A-worker and can be moved to the F-worker with the AFD boundary placement option.
4. **Skipped model-internal communication ops**: communication or dispatch operations already represented by the AFD communication model, such as `CustomAllReduce`, `P2P`, `NCCL`, TP all-gather / reduce-scatter, and MoE dispatch ops. These should not be counted again in either compute pool because AFD adds its own cross-pool and intra-pool communication through `AFDTransfer`, `AFDFAllGather`, `AFDFReduceScatter`, and `AFDCombine`.
5. **Overlap ops**: an `OverlapOp` can stay atomic only when every non-skipped inner op belongs to the same side. If an overlap group spans the A/F boundary, the partitioner must fail or be extended to split / rebuild that overlap explicitly.
6. **Layer families that need explicit rules**: Mamba and GDN layers are not covered by the current attention/FFN partition rules. Until a dedicated partitioning rule, operation model, and communication / memory accounting are added, the partitioner raises an explicit `AFDPartitionError` for these ops instead of falling back to an unknown-op side assignment.

If a new operation cannot be classified, do not rely on an unknown-op fallback for production use. Update `sdk/afd_partition.py` with an explicit classification rule and add a focused unit test in `tests/unit/sdk/test_afd_partition.py`.
