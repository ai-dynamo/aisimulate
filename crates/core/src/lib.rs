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
pub mod replay;

pub use engine::{EngineConfig as ReplayEngineConfig, TimingModel, TimingModelConfig};
pub use replay::{ReplayReport, ReplaySpec, Replayer};

// Preserve the former published AIC crate-root surface. Replay's conflicting
// engine configuration remains available as `engine::EngineConfig` and under
// the explicit `ReplayEngineConfig` alias above.
pub use perfmodel::EngineConfig;
pub use perfmodel::{
    AicError, BackendKind, DataType, ENGINE_CONFIG_SCHEMA_VERSION, ENGINE_SPEC_SCHEMA_VERSION,
    EstimateSource, FPM_VERSION, ForwardPassFallbackPolicy, ForwardPassMetrics,
    ForwardPassModelKind, ForwardPassPerfDiagnostics, ForwardPassPerfModel,
    ForwardPassPerfModelConfig, ForwardPassPerfOptions, ForwardPassPerfProvenance,
    ForwardPassPerfReadiness, ForwardPassPerfSource, KvCacheEstimate, KvCacheEstimateAdjusted,
    KvCacheEstimateError, KvCacheEstimateOptions, KvCacheEstimateRequest, KvCacheMemoryFraction,
    MemoryBreakdown, ParallelMapping, QuantizationConfig, QueuedRequestMetrics,
    ScheduledRequestMetrics, SpeculativeConfig,
};

#[cfg(feature = "python")]
pub use perfmodel::{AicEngine, AicEngineBuilder, estimate_kv_cache};

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
