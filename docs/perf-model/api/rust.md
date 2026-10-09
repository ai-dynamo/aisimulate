<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust performance-model API

Use `aisimulate_core::perfmodel::ForwardPassPerfModel::best_available` with
`ForwardPassPerfModelConfig` for estimator selection, correction, and regression.
`AicEngineBuilder` is the lower-level compiled operation-engine entry point;
it is not a replacement constructor for the canonical estimator.

```rust
use aisimulate_core::perfmodel::{
    BackendKind, EstimationMode, ForwardPassPerfModel,
    ForwardPassPerfModelConfig, ForwardPassWorkerType,
};

let mut config = ForwardPassPerfModelConfig::new(
    "Qwen/Qwen3-32B", "h200_sxm", BackendKind::Vllm,
    ForwardPassWorkerType::Decode,
);
config.estimation_mode = EstimationMode::FpmRegression;
let model = ForwardPassPerfModel::best_available(config)?;
// Supply observed iterations before requesting nonempty regression predictions.
```

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
[crate README](../../../crates/core/README.md) for setup and usage examples.

See [SDK entry-point migration](../../aic-backward-compatibility/migration.md#sdk-entry-points) for the removed
flat engine adapter and constructor replacements.

The supported `aisimulate_core::perfmodel` Rust surface is grouped as follows:

- compiled engine: `AicEngineBuilder`, `AicEngine`, `AicError`;
- forward-pass estimation: `ForwardPassPerfModel`,
  `ForwardPassWorkerType`, `ForwardPassPerfModelConfig`, `EstimatorConfig`,
  diagnostics/readiness/source types, and the `ForwardPassMetrics` telemetry
  types;
- KV-cache estimation: `estimate_kv_cache`, `KvCacheEstimateRequest`,
  `KvCacheEstimateOptions`, `KvCacheMemoryFraction`, and estimate/result/error
  types;
- FPM profile cache resources: `FpmCacheGroup`, `FpmCacheKind`, `FpmCacheLayout`,
  `FpmResourceConfig`, `FpmRuntimeMemoryConfig`, `FpmCacheBudgetRequest`, `FpmCacheBudget`, and
  `FpmCacheBudgetAdjusted`, through the canonical model/config budget method;
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
[modeled-power contract](../power.md) for the latency-weighted coverage gate,
aggregation rules, and public output boundary. Whole-forward, fixed, and polynomial providers remain latency-only; the
power contract describes which Replay paths can publish energy evidence.

## Wire and version boundary

The wheel and crate are released together. `EngineSpec` is a serialized operation
plan, not a stable cross-version binary format. Use the exported schema constants
rather than hard-coding a number; the loader rejects mismatched bincode schemas
before interpreting operation payloads. JSON defaults do not imply binary
compatibility. Recompile saved engines when updating the paired runtime.

Schema 27 adds an optional `vision` section to `EngineSpec`: the vision tower
compiled under `ForwardPassPerfModelConfig::encoder_parallel`, priced by
`ForwardPassPerfModel::vision_operations` and `predict_vision_ms` over
`EncoderImageShape` groups (`sequences`, per-sequence `patch_tokens`,
`transformer_tokens`, `output_tokens`, `images`). The section is absent when
`encoder_parallel` is unset.

The public exports are defined in
[`perfmodel/mod.rs`](../../../crates/core/src/perfmodel/mod.rs).
Python-backed construction and `estimate_kv_cache` are feature-gated; explicit
regression construction does not need Python embedding. Prefer constructors and
`Default` where offered instead of exhaustive literals for evolving controls.
