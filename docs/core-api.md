# `aisimulate-core` public API contract

The core API is delivered through the repository's two release artifacts at
the same version:

- the `aisimulate` Python wheel, whose estimator API is `aisimulate_core`;
- the `aisimulate-core` Rust crate, imported as `aisimulate_core`.

The single wheel owns the application, estimator SDK, model and system data,
and unified native PyO3 extension. It does not depend on another core
distribution or on Dynamo. The crate owns the compiled engine, forward-pass
model, Replay runtime, KV-cache request/response types, and the embedded
Rust-to-Python construction path. Legacy Python import namespaces are removed
in AISimulate 0.13.0; see the [Python migration guide](python-source-migration.md).

## Stable Python facade

New Python code should import from the small facade:

```python
from aisimulate_core.sdk import (
    EngineHandle,
    ModelConfig,
    RuntimeConfig,
    RustForwardPassPerfModel,
    compile_engine,
    estimate_kv_cache,
    estimate_num_gpu_blocks,
)
```

The explicit module paths remain supported:

```python
from aisimulate_core.sdk.engine import EngineHandle, compile_engine
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel
from aisimulate_core.sdk.memory import estimate_kv_cache, estimate_num_gpu_blocks
```

`aisimulate_core.sdk.__all__` is the supported high-level surface. The
facade resolves lazily, so importing it does not load the model registry,
performance database, or native engine until a name is used.

The top-level `aisimulate_core` module exposes the lower-level native
extension contract:

- `AicEngine`
- `RustForwardPassPerfModel` (the raw PyO3 class, distinct from the ergonomic
  SDK wrapper)
- `engine_spec_bincode_from_json`
- `_build_smoke`

The wheel includes `py.typed` and a stub for that native extension. The SDK
Python modules carry their own annotations.

### Context-attention kernel queries

`AicEngine.evaluate_context_attention_kernels_json` accepts `ops_json` (a JSON
array of serialized `ContextAttention` operations), `batch_size`, and `s`
(sequence length in tokens), with optional `prefix=0`,
`imbalance_correction_scale=1.0`, and `visual_block_upper_triangle=False`.
It evaluates only the attention kernels, excluding fused QK normalization,
RoPE, and KV-write work. Invalid JSON and other operation families raise
`ValueError`.

With `visual_block_upper_triangle=True`, the query prices only the additional
strict upper-triangle attention pairs inside a bidirectional visual block,
using the underlying causal kernel's database policy and provenance. This
mode requires `prefix=0`; a nonzero prefix raises `ValueError`. A block with
zero or one token has zero additional latency and energy, with `source="sol"`.
The flag is a runtime query option and does not change the serialized op or
engine-spec formats.

Results are `list[tuple[str, float, float, str]]`, containing
`(name, latency_ms, energy_wms, source)` with repeated operation names folded
together. Energy is in watt-milliseconds and is zero when power data is
unavailable. An empty operation list returns an empty list. `aisimulate_core.AicEngine` exposes this
method; `EngineHandle` provides an annotated SDK wrapper with the same query
options.

## KV-cache capacity reservation

SGLang's native estimator treats `mem_fraction_static` as a static weights/KV
pool. Peak activation/workspace estimates remain visible in
`memory_breakdown.activations_bytes`, but are not deducted from that pool;
transient execution headroom is already outside the static fraction. Increasing
the prefill token budget alone therefore does not reduce SGLang KV capacity.
Resident runtime/communication estimates reduce the pre-load free-memory pool
before applying the fraction; weights are deducted afterward. The budget is
`(capacity - resident_overhead) * mem_fraction_static - weights`, less any
explicit additional graph reservation. For ordinary SGLang DeepSeek-V3/R1
(non-CP, non-PP, non-speculative, non-large-EP), the estimator also respects
checkpoint dense/MoE layer counts and TP-sharded embeddings independently of
the unchanged timing graph. Other model layouts retain their prior weight
accounting.
vLLM and TRT-LLM continue to deduct activation memory under their own budget
semantics. No measured server capacity is required by this calculation.

`estimate_kv_cache` and `estimate_num_gpu_blocks` accept
`cuda_graph_reserved_bytes=<rank-local bytes>`. The value must be a
non-negative integer no greater than `2**53` and defaults to zero. It is treated
as fixed non-KV memory before the backend-specific KV fraction is applied. For
SGLang, it is an additional reservation beyond graph/runtime headroom already
encoded by `mem_fraction_static`. Native estimates return it as
`memory_breakdown.cuda_graph_reserved_bytes`; naive fallback estimates apply it
but keep `memory_breakdown=None` because the other components are unavailable.

The Rust `KvCacheEstimateRequest` exposes the same field. Engine replay accepts
the value in its engine arguments and carries it through native AIC capacity
rematerialization, so the Python estimator and native scheduler use the same
rank-local KV capacity. The `aisimulate predict` YAML exposes it at
`engine.workers.<role>.kv_cache.capacity.cuda_graph_reserved_bytes`. This API
does not estimate the reservation; callers must supply a value from a source
they trust.

Serialized Rust requests and estimates that omit the field remain compatible
because deserialization defaults it to zero. Rust source that constructs
`KvCacheEstimateRequest` with a struct literal must add
`cuda_graph_reserved_bytes: 0`; exhaustive `MemoryBreakdown` literals and
patterns must include the new field. This source migration is part of the next
minor API update.

## Choosing a forward-pass API

Use `RustForwardPassPerfModel.best_available(config)` from Python or
`ForwardPassPerfModel::best_available(config)` from Rust. The canonical
`ForwardPassPerfModelConfig` owns model, hardware, backend, topology, data
policies, a required immutable `worker_type`, and the complete nested
`estimator_config`. Worker roles are `prefill`, `decode`, and `aggregated`.

```python
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

config = ForwardPassPerfModelConfig(
    model="Qwen/Qwen3-32B",
    system="h200_sxm",
    backend="vllm",
    worker_type="aggregated",
    tp=2,
    estimation_mode="auto",
    fallback_policy="deny",
    estimator_config={
        "features": {"attention_kv_weight": 1.0},
        "fpm_regression": {
            "sampling": {"bins_per_axis": [4, 16], "max_observations": 128},
            "min_observations": 5,
        },
        "correction": {"enabled": True},
    },
)
model = RustForwardPassPerfModel.best_available(config)
print(model.diagnostics()["provenance"])
```

### Engine identity controls

The canonical configuration also carries quantization overrides and
`attention_backend`, `moe_backend`, `enable_eplb` (default `false`), and
`wideep_num_slots` (default absent). These controls reach model construction,
KV memory sizing, and replay provenance. EPLB/slots and nondefault MoE backend
selection require an MoE model. Collected FPM interpolation cannot represent
EPLB, slots, or MoE backend overrides; it rejects an explicit incompatible
request and is skipped during automatic selection for those identities.

Rust callers using exhaustive `ForwardPassPerfModelConfig` literals must add
`moe_backend: None`, `enable_eplb: false`, and `wideep_num_slots: None`.
`ForwardPassPerfModelConfig::new(...)` supplies these defaults. This extends
the canonical configuration introduced by #242.

Rust callers constructing `SyntheticTraceSpec` must also add
`cached_prefix_tokens: 0` to preserve existing prefix-sharing behavior. A positive
value creates shared input tokens; cache hits still depend on runtime state.
The value must align to the trace's `block_size` and must not exceed any sampled
input length. Unified replay uses one-token trace blocks for an exact prefix,
then applies the engine's cache block size when calculating reuse.

`nextn` remains compute-side identity. Expected accepted draft tokens are a
simulator workload assumption, supplied separately by the unified CLI as
`engine.nextn_accepted`; they do not tune the estimator.

### Selection and fallback

`estimation_mode` defaults to `auto`; `fallback_policy` defaults to `deny`.
Auto always searches `op_level`, `fpm_interpolation`, then `fpm_regression`,
including when fallback is denied. For an explicit mode, deny permits only
that estimator; allow tries the requested estimator first, then the remaining
estimators in the same global priority order. Each native mode tries the
ordered `systems_paths` before moving to another estimator. Omitted roots preserve
configured SDK discovery (or the systems-path environment override when the SDK
uses its packaged default); an explicit `default` entry selects the packaged root.
The resolved paths are shared by preflight, construction and capacity estimation.
Selection occurs
at construction; queries do not silently switch estimators on a data-domain error.

Invalid caller configuration does not trigger fallback. A constructed regression
may be unready: nonempty queries return `None` until their selected workload
store has a usable fit. Offline prediction/recommendation reject an untrained
regression instead of fabricating a latency. The current aggregated regression
retains its four workload stores; dedicated roles retain one each.

The returned provenance records the requested and selected modes, failed
selection attempts, effective backend version, data policy, selected root,
and complete estimator configuration. Its resolved config pins the selected
mode with deny so saved replay input repeats that selection.

Prompt-lookup verification uses the same constructor: set `speculation` to
`{"kind": "ngram", "params": {"num_speculative_tokens": 2}}` in Python/JSON, or
`ForwardPassSpeculationConfig::Ngram { num_speculative_tokens: 2 }` in Rust.
It supports vLLM op-level timing with 1–5 draft tokens and `nextn: 0`; auto can
select op-level but cannot fall back to an unsupported speculative estimator.
The cost configuration is retained in provenance and saved recommendations.
Acceptance rates and the scheduler seed stay in the CLI/Replay speculation
configuration; they do not change the model's target-verification graph.

### Estimator controls

`estimator_config` is passed intact through the Python facade, CLI, Sweeper,
and Replay, then parsed and validated in Rust. Unknown fields report their
nested paths. The supported namespaces are:

- `features`: `attention_kv_weight`, `prefill_attention_pair_weight`, and
  `ffn_token_weight`, each defaulting to 1.0. These currently affect regression
  only; positive finite values are required when regression is constructed.
- `fpm_regression`: independent `sampling`, `min_observations` (5), and `fit`.
  The fit kind is `standardized_nnls`, with a free intercept and nonnegative
  slopes. `singular_ridge_scale` defaults to `1e-9` and applies only when retrying
  a singular equation.
- `correction`: `enabled` (true), independent `sampling`, `min_observations`
  (5), `factor_bounds` (min 0.5, max 2.0), and the existing `max_num_tokens`
  (8192), `max_batch_size` (512), and `max_kv_tokens` (2000000) ranges.
- `op_level` and `fpm_interpolation`: reserved typed namespaces with no
  additional knobs yet; unknown fields are rejected.

Sampling defaults to `bins_per_axis: [4, 4]` and `max_observations: 64` per
logical store. Rectangular grids are supported. Regression uses dynamic
`log1p` retention coordinates and fits standardized raw features. Correction
uses fixed raw workload coordinates; its one-dimensional prefill grid uses
the product of the two axis counts. Retention evicts the oldest sample from
the most populated cell when the store exceeds its budget.

Correction explicitly reports `feature_space: legacy_workload`. Its existing
prefill/decode/mixed stores and median-ratio calculation remain unchanged at
default settings. A shared role-based correction space requires separate
accuracy validation and is not accepted as a configuration value in this release.

Use `regression_store_diagnostics()` for per-store counts/readiness. Summary
readiness means at least one store is ready; another cold store can still
return `None`. `tune_with_fpms()` preserves the established FPM observation
contract. Native construction still uses Python model compilation; estimator
selection, regression, correction, and latency computation are owned by Rust.

### Migrating saved configuration

Use `ForwardPassPerfModelConfig.from_legacy_engine_config(old_config,
worker_type, old_options, allow_regression=False)` to convert a saved flat
EngineConfig and tuning options. It pins the old explicit native mode instead
of changing it to auto. Set `allow_regression=True` only for an old caller that
allowed direct regression fallback; the migration preserves that two-mode
order rather than adding interpolation. Legacy `forward_model: fpm` maps to
`fpm_interpolation`, and `fallback_policy: error` maps to deny. The deprecated
`regression` policy remains readable for these saved direct-fallback requests.

Previously saved CLI timing with `forward_model` retains explicit selection
and deny. Newly authored requests without a selection use auto. `ForwardPassPerfOptions`
is retained as a legacy migration value type; new construction has one complete
config and no separate options argument. The raw PyO3 class also exposes
`normalize_config` and migration helpers for JSON-oriented consumers.

The [FPM regression design](../python/aisimulate/docs/fpm/aic-fpm-regression-design.md)
explains the retained workload routing and feature mathematics.

## Stable Rust facade

New embedded consumers should construct engines with
`aisimulate_core::perfmodel::AicEngineBuilder`. The
builder normalizes configuration into one private build request and enters
Python once to compile an engine specification. Calls on the returned
`AicEngine` are pure Rust and do not re-enter Python. Selected former AIC
crate-root types remain re-exported during the migration window, but the
`perfmodel` namespace is canonical for new code.

Native engine construction in standalone binaries requires the crate's `embed-python`
feature; applications hosted by an initialized Python interpreter do not need
auto-initialization. The matching `aisimulate` wheel must be importable for native
construction. Explicit `fpm_regression` construction works without the Python feature. See the
[crate README](../crates/core/README.md) for setup and usage examples.

The flat `build_aic_engine` adapter was removed from `main`; consumers must use
`AicEngineBuilder`.

The supported `aisimulate_core::perfmodel` Rust surface is grouped as follows:

- compiled engine: `AicEngineBuilder`, `AicEngine`, `AicError`;
- forward-pass estimation: `ForwardPassPerfModel`,
  `ForwardPassWorkerType`, `ForwardPassPerfModelConfig`, `EstimatorConfig`,
  diagnostics/readiness/source types, and the `ForwardPassMetrics` telemetry
  types;
- KV-cache estimation: `estimate_kv_cache`, `KvCacheEstimateRequest`,
  `KvCacheEstimateOptions`, `KvCacheMemoryFraction`, and estimate/result/error
  types;
- wire identity: `EngineConfig`, `ParallelMapping`, `QuantizationConfig`,
  `SpeculativeConfig`, `BackendKind`, `DatabaseMode`, and `DataType`;
- schema gates: `ENGINE_CONFIG_SCHEMA_VERSION`,
  `ENGINE_SPEC_SCHEMA_VERSION`, and `FPM_VERSION`.

Advanced consumers may use `perfmodel::engine::{Engine, RuntimeConfig, StaticMode,
StaticResult, PerOpValue}` and `engine::spec::{EngineSpec, OpSpec}` to load and
execute a previously compiled specification directly. `PerOpValue` is the
per-op result tuple `(name, latency_ms, energy_wms, source)` returned by the
`*_per_op` / `evaluate_*` methods (the thin op-list evaluation FFI); per-op
energy is 0.0 wherever the perf tables carry no power columns. That zero is a
missing-data sentinel, not evidence of a zero-power operation. See the
[modeled-power contract](power-model.md) for the latency-weighted coverage gate,
aggregation rules, and public output boundary. Typed per-op energy alone does
not make unified replay power available.

## Replay timing evidence

The runtime-neutral `TimingModel` contract exposes optional accumulated
evidence through `evidence_summary()`. Op-level AIC providers return a
`TimingEvidenceSummary` split into prefill and decode phases. Each
`TimingPhaseEvidence` carries accumulated known energy in W-ms, total latency,
latency covered by nonzero energy data, merged provenance, and name-folded
`TimingOperationEvidence` records with the same fields. Missing operation
energy is represented by `None`, never by a synthesized zero. When coverage is
below one, the energy is partial: it includes only the portion with positive
operation-energy evidence. It is not a total-workload energy estimate.
Providers that assemble these public records directly should use
`TimingPhaseEvidence::try_from_operations` and `try_accumulate`; those paths
validate numeric fields and canonicalize covered latency to zero when energy is
missing. Nonempty operation lists must agree with phase totals; a relative
rounding tolerance applies only to this consistency check. The original infallible helpers remain available for already-valid
evidence.

Whole-model FPM timing and the built-in fixed and polynomial timing models are
latency-only and return `None` from `evidence_summary()`. Consumers must keep
that distinction when producing power metrics: absence of evidence is not a
zero-watt prediction. FPM decode timing continues to query the exact total
past-KV coordinate rather than the op-level mean-context coordinate.

## Compatibility rules

- The `aisimulate` wheel and `aisimulate-core` crate versions must match for
  every release.
- A breaking `EngineConfig`, `EngineSpec`, or `ForwardPassMetrics` wire change
  must bump its corresponding schema constant. Consumers reject unsupported
  schema versions before using the payload.
- A supported facade name is not removed or given a new required parameter
  without a documented deprecation path. The package is pre-1.0, so an
  unavoidable incompatible API change also requires a minor-version bump.
- The canonical `best_available(config)` API replaces the old positional worker
  role/options signatures and separate estimator constructors. This is a source
  migration: use `ForwardPassPerfModelConfig::new(...)` in Rust or the SDK config
  class in Python, and use the explicit migration helper for saved EngineConfig
  values. Downstream Dynamo callers must migrate before this API's stable release;
  keep the crate and wheel versions aligned at the coordinated minor release.
- [The migration checklist](../.github/release-gates.json) and
  `scripts/check_release_migrations.py` apply before stable publication. Clear the
  pending entry in a reviewed change after downstream validation and merge.
  There is currently no standalone stable-publication workflow in this repository;
  that release process must invoke the checker for both its policy and target
  declarations (`--target-gates`). Missing or malformed declarations fail closed.
- The raw PyO3 class and ergonomic SDK wrapper intentionally share the name
  `RustForwardPassPerfModel`; callers should import from `aisimulate_core.sdk`
  unless they specifically need the JSON-oriented native binding.

The standalone AIC 0.12 compatibility distributions form a separate package
lineage: `aiconfigurator` pins `aiconfigurator-core==0.12.0`, and that core builds
its own native binding. They do not depend on the AISimulate wheel or crate.
Their retained binding declarations are not downstream callers of this API.
The [replacement installation](../README.md#upgrade-from-standalone-aiconfigurator)
removes both old distributions and installs AISimulate's complete migrated
application and core. Namespace compatibility does not preserve the removed
constructor signatures; direct SDK callers must perform the source migration
above. Release gates track consumers of the new artifacts, such as Dynamo.

## CI contract

Every change is checked from three consumer viewpoints:

1. Python source and isolated installed-wheel imports, including the public
   facade, native stub, bundled data, and upper/core ownership boundary;
2. Rust tests with embedding disabled and with all features enabled;
3. a separate workspace crate that depends on `aisimulate-core` and
   compiles only against its public exports.
