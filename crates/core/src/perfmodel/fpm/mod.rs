// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Forward-pass-level performance model with optional online tuning (PR #1152).
//!
//! Built on the compiled [`crate::perfmodel::engine::Engine`]: the `Native` variant holds
//! an `Arc<Engine>` and the native estimate routes through
//! [`crate::perfmodel::engine::Engine::forward_pass_time_ms`]. The online correction /
//! regression / diagnostics / readiness logic is engine-agnostic.
//!
//! Constructing a native model crosses into Python exactly once to compile the
//! model into an [`crate::perfmodel::engine::spec::EngineSpec`] (mirroring
//! [`crate::perfmodel::AicEngineBuilder`]); after that the hot path
//! (`estimate_forward_pass_time_ms` / `tune_with_fpms`) is pure Rust over the
//! `Engine` with no Python re-entry.
//!
//! Submodules:
//! - [`metrics`]: the `ForwardPassMetrics` telemetry types and validation.
//! - [`model`]: the public [`ForwardPassPerfModel`] and its diagnostics.
//! - [`correction`]: the native online-correction grid.
//! - [`regression`]: the regression fallback.
//! - [`learned`]: offline-trained tree-ensemble model over FPM features.
//! - [`samples`]: shared bucketed-sample infrastructure.
//! - [`options`]: tuning controls.

mod correction;
mod learned;
mod metrics;
mod model;
mod options;
mod regression;
mod samples;

#[cfg(test)]
mod tests;

pub use learned::{
    AGGREGATE_FEATURE_NAMES as LEARNED_AGGREGATE_FEATURE_NAMES,
    FEATURE_COUNT as LEARNED_FEATURE_COUNT, LEARNED_SCHEMA_NAME, LEARNED_SCHEMA_VERSION,
    REQUEST_FEATURE_NAMES as LEARNED_REQUEST_FEATURE_NAMES, SLOT_COUNT as LEARNED_SLOT_COUNT,
    feature_names as learned_feature_names,
};
pub(crate) use metrics::validate_forward_pass_metrics;
pub use metrics::{FPM_VERSION, ForwardPassMetrics, QueuedRequestMetrics, ScheduledRequestMetrics};
pub use model::{
    ForwardPassPerfDiagnostics, ForwardPassPerfModel, ForwardPassPerfReadiness,
    ForwardPassPerfSource, ForwardPassRegressionStoreDiagnostics,
    ForwardPassRegressionWorkloadKind, ForwardPassWorkerType,
};
pub use options::ForwardPassPerfOptions;
