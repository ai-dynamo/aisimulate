// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-neutral inference simulation, deterministic replay, and performance
//! modeling.
//!
//! [`engine`] owns scheduling, native GPU KV accounting, preemption, timing,
//! and attention-DP composition. [`replay`] owns virtual time, logical-worker
//! lifecycle, placement/scaling composition, and report collection.
//! [`perfmodel`] is the imported AIConfigurator latency and memory model. It is
//! deliberately namespaced so its `EngineConfig` and compiled engine do not
//! collide with the replay engine's public API.
//!
//! The crate root exposes the stable, commonly used configuration and replay
//! surface. Advanced engine and adapter contracts remain available through
//! their explicit module paths.

pub mod engine;
// The mirror contains compatibility and diagnostics helpers that are reached
// only from the optional Python surface or parity harnesses.
#[allow(dead_code)]
pub mod perfmodel;
#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
pub use python::execute_replay_json_with_composition;
pub mod replay;

pub use engine::{
    EngineConfig as ReplayEngineConfig, TimingEvidenceSource, TimingEvidenceSummary, TimingModel,
    TimingModelConfig, TimingOperationEvidence, TimingPhaseEvidence,
};
pub use replay::{ReplayReport, ReplaySpec, Replayer};

/// Identity of the serialized replay contract compiled into this library.
///
/// Optional extensions compare this with the base Python runtime before
/// execution. A source digest also catches incompatible development wheels
/// sharing a version number, without relying on checkout paths or Git state.
pub fn native_replay_contract() -> serde_json::Value {
    serde_json::json!({
        "api_version": 1,
        "core_version": env!("CARGO_PKG_VERSION"),
        "core_source_sha256": env!("AISIMULATE_CORE_SOURCE_SHA256"),
    })
}

// Preserve the former published AIC crate-root surface. Replay's conflicting
// engine configuration remains available as `engine::EngineConfig` and under
// the explicit `ReplayEngineConfig` alias above.
pub use perfmodel::EngineConfig;
pub use perfmodel::{
    AicError, BackendKind, CorrectionConfig, DataType, DatabaseMode, ENGINE_CONFIG_SCHEMA_VERSION,
    ENGINE_SPEC_SCHEMA_VERSION, EstimateSource, EstimationMode, EstimatorConfig, FPM_VERSION,
    ForwardPassFallbackPolicy, ForwardPassMetrics, ForwardPassPerfDiagnostics,
    ForwardPassPerfModel, ForwardPassPerfModelConfig, ForwardPassPerfOptions,
    ForwardPassPerfProvenance, ForwardPassPerfReadiness, ForwardPassPerfSource,
    ForwardPassRegressionStoreDiagnostics, ForwardPassRegressionWorkloadKind,
    ForwardPassSpeculationConfig, ForwardPassWorkerType, FpmRegressionConfig, KvCacheEstimate,
    KvCacheEstimateAdjusted, KvCacheEstimateError, KvCacheEstimateOptions, KvCacheEstimateRequest,
    KvCacheMemoryFraction, MemoryBreakdown, ParallelMapping, QuantizationConfig,
    QueuedRequestMetrics, RegressionFeatureWeights, SamplingConfig, ScheduledRequestMetrics,
    SpeculativeConfig,
};
pub use perfmodel::{
    CorrectionFactorBounds, CorrectionFeatureSpace, FpmInterpolationConfig, OpLevelConfig,
    RegressionFitConfig, RegressionFitKind,
};

#[cfg(feature = "python")]
pub use perfmodel::{
    AicEngine,
    // Low-level Rust embedder API for compiled step-latency handles. This is
    // deliberately not registered on the Python module and is not an
    // alternative ForwardPassPerfModel construction boundary.
    AicEngineBuilder,
    estimate_kv_cache,
};

// The imported perf-model sources historically used additional crate-root
// module paths. Keep these module aliases crate-private so the mirror subtree
// can stay mechanically syncable while external users use `perfmodel::*` for
// advanced APIs.
#[allow(unused_imports)]
pub(crate) use perfmodel::{
    common, config, fpm, memory, operators, perf_database, repo_relative, session,
    validate_forward_pass_metrics,
};

#[cfg(feature = "python")]
#[allow(unused_imports)]
pub(crate) use perfmodel::{py, py_ops};
