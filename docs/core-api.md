# `aisimulate-core` public API contract

The core API is delivered through the repository's two release artifacts at
the same version:

- the `aisimulate` Python wheel, imported as `aisimulate_core` or through the
  compatibility namespace `aiconfigurator_core`;
- the `aisimulate-core` Rust crate, imported as `aisimulate_core`.

The single wheel owns the application, estimator SDK, model and system data,
and unified native PyO3 extension. It does not depend on another core
distribution or on Dynamo. The crate owns the compiled engine, forward-pass
model, Replay runtime, KV-cache request/response types, and the embedded
Rust-to-Python construction path. The legacy `aiconfigurator_core` Python
namespace remains available during the AIC 0.12.0 compatibility window.

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

## KV-cache capacity reservation

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
rank-local KV capacity. This API does not estimate the reservation; callers
must supply a value from a source they trust.

Serialized Rust requests and estimates that omit the field remain compatible
because deserialization defaults it to zero. Rust source that constructs
`KvCacheEstimateRequest` with a struct literal must add
`cuda_graph_reserved_bytes: 0`; exhaustive `MemoryBreakdown` literals and
patterns must include the new field. This source migration is part of the next
minor API update.

## Choosing a forward-pass API

For adaptive forward-pass modeling, use
`RustForwardPassPerfModel.best_available(config, worker_type, options=None)`
from Python or
`ForwardPassPerfModel::best_available(config, worker_type, options)` from Rust.
`worker_type` is an immutable property of the engine model, not an inferred
property of one FPM iteration. Python accepts exactly `"prefill"`, `"decode"`,
or `"aggregated"`; Rust uses
`ForwardPassWorkerType::{Prefill, Decode, Aggregated}`.

This path uses the native AIC estimate when the native estimator can be built,
learns online correction factors from FPM observations, and falls back to the
regression associated with `worker_type` for eligible native build or
data-availability failures. These include unsupported models and missing or
unreadable model, system, or performance data, plus malformed system YAML. A
successful native build keeps the existing workload-kind inference and
correction behavior; `worker_type` and regression-only weights do not alter
native estimates. Check
`diagnostics()` to determine whether the active source is `aic`,
`aic_with_correction`, or `fallback_regression`, and to inspect any fallback
warning.

Native online corrections default to an absolute factor range of `[0.5, 2.0]`.
Pass `None` explicitly as `min_faster_correction_factor` or
`max_slower_correction_factor` in the options dictionary to remove the bound
in that direction. Regression fallback ignores both options.

Use `from_native(...)` instead when native AIC support is required and an
unsupported configuration or native data failure should surface rather than
fall back. This strict-native constructor does not take `worker_type`.

Use `RustForwardPassPerfModel.from_regression(worker_type, options=None)` or
`ForwardPassPerfModel::from_regression(worker_type, options)` for a
regression-only model. It owns one two-dimensional retained sample set and one
fit for the engine's fixed role. Its axes are consistently ordered as
`[critical attention, global FFN/MoE]`; bucket retention uses `log1p` of those
raw features, while fitting uses standardized raw features. The optional
regression weights below default to `1.0` and must be finite and strictly
positive:

- `regression_attention_kv_weight` (`alpha`);
- `regression_prefill_attention_pair_weight` (`beta`);
- `regression_ffn_token_weight` (`gamma`).

The ergonomic Python facade accepts ordinary Python floats. For these three
fields only, it marshals nonfinite values on a shallow copy of the options
dictionary as the exact valid-JSON string sentinels `"NaN"`, `"Infinity"`, and
`"-Infinity"`; finite values remain JSON numbers. Callers of the JSON-oriented
raw PyO3 API may use the same sentinels directly. The sentinels preserve values
for backend-dependent validation rather than making them valid regression
weights: `from_native` and a successful native `best_available` ignore all
three fields, while `from_regression` and a fallback `best_available` reject a
decoded nonfinite value with the corresponding field-specific error. Other
strings and value types remain invalid.

The formulas, role-compatibility rules, and fitting pipeline are specified in
the [FPM regression design](../python/aisimulate/docs/fpm/aic-fpm-regression-design.md).

`AicEngineBuilder` serves a different purpose: it constructs the strict native
Rust engine for direct public prefill and decode latency calls. It does not
provide regression fallback or online correction, so it is not a replacement
for `best_available(...)`.

```python
from aisimulate_core.sdk import RustForwardPassPerfModel

# Engine-config and per-rank FPM dictionary setup is omitted here.
model = RustForwardPassPerfModel.best_available(config, "decode")
diagnostics = model.diagnostics()
print(diagnostics["source"])
if diagnostics["last_warning"] is not None:
    print(diagnostics["last_warning"])

estimate_ms = model.estimate_forward_pass_time_ms(metrics_by_rank)
if estimate_ms is None:
    # Regression fallback starts without observations for this worker type.
    # Supply observed FPM iterations with positive wall_time until the configured
    # min_observations threshold is reached, then retry the estimate.
    model.tune_with_fpms(observed_iterations)  # Observed-iteration setup omitted.
    estimate_ms = model.estimate_forward_pass_time_ms(metrics_by_rank)
```

## Stable Rust facade

New embedded consumers should construct engines with
`aisimulate_core::perfmodel::AicEngineBuilder`. The
builder normalizes configuration into one private build request and enters
Python once to compile an engine specification. Calls on the returned
`AicEngine` are pure Rust and do not re-enter Python. Selected former AIC
crate-root types remain re-exported during the migration window, but the
`perfmodel` namespace is canonical for new code.

Standalone binaries must enable the crate's `embed-python` feature; applications
hosted by an initialized Python interpreter do not. In either case, the matching
`aisimulate` wheel must be importable. See the
[crate README](../crates/core/README.md) for setup and usage examples.

The flat `build_aic_engine` adapter was removed from `main`; consumers must use
`AicEngineBuilder`.

The supported `aisimulate_core::perfmodel` Rust surface is grouped as follows:

- compiled engine: `AicEngineBuilder`, `AicEngine`, `AicError`;
- forward-pass estimation: `ForwardPassPerfModel`,
  `ForwardPassWorkerType`, `ForwardPassPerfOptions`,
  diagnostics/readiness/source types, and the `ForwardPassMetrics` telemetry
  types;
- KV-cache estimation: `estimate_kv_cache`, `KvCacheEstimateRequest`,
  `KvCacheEstimateOptions`, `KvCacheMemoryFraction`, and estimate/result/error
  types;
- wire identity: `EngineConfig`, `ParallelMapping`, `QuantizationConfig`,
  `SpeculativeConfig`, `BackendKind`, and `DataType`;
- schema gates: `ENGINE_CONFIG_SCHEMA_VERSION`,
  `ENGINE_SPEC_SCHEMA_VERSION`, and `FPM_VERSION`.

Advanced consumers may use `perfmodel::engine::{Engine, RuntimeConfig, StaticMode,
StaticResult, PerOpValue}` and `engine::spec::{EngineSpec, OpSpec}` to load and
execute a previously compiled specification directly. `PerOpValue` is the
per-op result tuple `(name, latency_ms, energy_wms, source)` returned by the
`*_per_op` / `evaluate_*` methods (the thin op-list evaluation FFI); per-op
energy is 0.0 wherever the perf tables carry no power columns.

## Compatibility rules

- The `aisimulate` wheel and `aisimulate-core` crate versions must match for
  every release.
- A breaking `EngineConfig`, `EngineSpec`, or `ForwardPassMetrics` wire change
  must bump its corresponding schema constant. Consumers reject unsupported
  schema versions before using the payload.
- A supported facade name is not removed or given a new required parameter
  without a documented deprecation path. The package is pre-1.0, so an
  unavoidable incompatible API change also requires a minor-version bump.
- Requiring `worker_type` in `from_regression` and `best_available` is a
  documented incompatible change for the next minor release. Adding the three
  public regression-weight fields to `ForwardPassPerfOptions` is also a Rust
  source break for downstream exhaustive struct literals. Rust callers should
  prefer update syntax such as
  `ForwardPassPerfOptions { bucket_count: 16, ..Default::default() }` so future
  option fields do not require source changes. This feature does not itself
  change package versions; release coordination must keep the crate and wheel
  versions aligned.
- The raw PyO3 class and ergonomic SDK wrapper intentionally share the name
  `RustForwardPassPerfModel`; callers should import from `aisimulate_core.sdk`
  unless they specifically need the JSON-oriented native binding.

## CI contract

Every change is checked from three consumer viewpoints:

1. Python source and isolated installed-wheel imports, including the public
   facade, native stub, bundled data, and upper/core ownership boundary;
2. Rust tests with embedding disabled and with all features enabled;
3. a separate workspace crate that depends on `aisimulate-core` and
   compiles only against its public exports.
