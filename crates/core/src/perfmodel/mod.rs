// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native AIConfigurator performance-model implementation.
//!
//! The compiled-engine path is the only supported entry point: Python's
//! `compile_engine` walks the model once and emits an [`engine::spec::EngineSpec`]
//! (op lists + [`EngineConfig`] identity); the Rust [`engine::Engine`] executes
//! it without re-entering Python. With the `python` feature enabled,
//! [`AicEngineBuilder`] is the preferred Rust → Python → Rust embedded build
//! entry point and [`AicEngine`] is the PyO3 hot-path pyclass.
//!
//! This directory remains a stable mirror of the former AIConfigurator Rust
//! crate. Keeping the imported implementation behind one namespace makes
//! upstream syncs mechanical while the crate root remains owned by replay.

use std::path::PathBuf;

#[cfg(feature = "python")]
pub(crate) mod py;
#[cfg(feature = "python")]
pub(crate) mod py_ops;

// Modular core. `common/` holds shared foundation types (enums, error,
// system_spec) with no AIC-domain knowledge. Top-level files (`config`,
// `session`) and directories (`operators`, `perf_database`) carry the domain
// logic the compiled `engine` executes.
pub(crate) mod common;
pub(crate) mod config;
pub mod engine;
pub(crate) mod fpm;
pub mod memory;
pub(crate) mod operators;
pub(crate) mod perf_database;
pub(crate) mod session;

pub use common::{AicError, enums::DatabaseMode};
// Forward-pass perf model (PR #1152): a forward-pass latency model with online
// correction, regression fallback, diagnostics, and readiness, built on the
// compiled [`engine::Engine`]. Re-exported so Rust embedders (the Dynamo
// planner / Mocker) can use it natively; also exposed to Python via the
// `RustForwardPassPerfModel` pyclass in `py.rs`.
pub use fpm::{
    ForwardPassPerfDiagnostics, ForwardPassPerfModel, ForwardPassPerfOptions,
    ForwardPassPerfReadiness, ForwardPassPerfSource, ForwardPassRegressionStoreDiagnostics,
    ForwardPassRegressionWorkloadKind, ForwardPassWorkerType, LEARNED_AGGREGATE_FEATURE_NAMES,
    LEARNED_FEATURE_COUNT, LEARNED_REQUEST_FEATURE_NAMES, LEARNED_SCHEMA_NAME,
    LEARNED_SCHEMA_VERSION, LEARNED_SLOT_COUNT, learned_feature_names,
};
// Forward-pass metrics telemetry types and schema version, plus the
// crate-internal validation helper. Re-exported at the crate root so existing
// `crate::ForwardPassMetrics` / `crate::FPM_VERSION` references (in `py.rs`,
// `engine/runtime.rs`) keep resolving after the types moved into `fpm`.
pub(crate) use fpm::validate_forward_pass_metrics;
pub use fpm::{FPM_VERSION, ForwardPassMetrics, QueuedRequestMetrics, ScheduledRequestMetrics};
// KV-cache memory API. Top-level surface, not a method on
// `AicEngine`: estimation runs once at startup, separate from the latency path.
#[cfg(feature = "python")]
pub use memory::estimate_kv_cache;
pub use memory::{
    EstimateSource, KvCacheEstimate, KvCacheEstimateAdjusted, KvCacheEstimateError,
    KvCacheEstimateOptions, KvCacheEstimateRequest, KvCacheMemoryFraction, MemoryBreakdown,
};
// PyO3 bindings. `AicEngine` is the Python -> Rust hot-path pyclass;
// `AicEngineBuilder` is the Rust -> Python -> Rust entry point. They must be
// `pub`-re-exported here because the `py` module itself is private.
#[cfg(feature = "python")]
pub use py::{AicEngine, AicEngineBuilder};
// Public wire/identity config types live in `config`. Re-exported at the crate
// root so existing `crate::EngineConfig` / `crate::BackendKind` / ... paths
// resolve unchanged across the crate and for external consumers.
pub use config::{
    BackendKind, DataType, ENGINE_CONFIG_SCHEMA_VERSION, ENGINE_SPEC_SCHEMA_VERSION, EngineConfig,
    ParallelMapping, QuantizationConfig, SpeculativeConfig,
};

/// Resolve a repo-relative path by walking up from the crate manifest dir.
/// Used by [`py`] to locate the bundled data roots when developing in-tree.
pub(crate) fn repo_relative(rel: &str) -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest_dir.ancestors() {
        let candidate = ancestor.join(rel);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Register the compatibility AIConfigurator API on the unified native
/// `aisimulate._runtime` extension module.
#[cfg(feature = "python")]
pub fn register_python(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    py::register(module)
}
