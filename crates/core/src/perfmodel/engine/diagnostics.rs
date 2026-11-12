// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Evidence for one native static phase, before online correction.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolDiagnostics {
    pub latency_ms: f64,
    pub math_ms: f64,
    pub memory_ms: f64,
}

impl SolDiagnostics {
    /// Component-wise `self + other`.
    pub(crate) fn plus(&self, other: &Self) -> Self {
        Self {
            latency_ms: self.latency_ms + other.latency_ms,
            math_ms: self.math_ms + other.math_ms,
            memory_ms: self.memory_ms + other.memory_ms,
        }
    }

    /// Every component is finite and nonnegative.
    pub(crate) fn is_valid(&self) -> bool {
        [self.latency_ms, self.math_ms, self.memory_ms]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutedFallback {
    pub inference_phase: String,
    pub comm_backend: String,
    pub requested_ep_size: u32,
    pub requested_node_num: u32,
    pub measurement_ep_size: u32,
    pub measurement_node_num: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperationDetails {
    pub sol: Option<SolDiagnostics>,
    pub sol_unavailable_reason: Option<String>,
    /// Executed measurement substitutions; empty means no recorded substitution.
    pub fallbacks: Vec<ExecutedFallback>,
    /// Whole-forward estimates, counted by identical result and replay scale.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fpm_estimates: Vec<crate::FpmEstimateEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StaticOperationDiagnostics {
    pub name: String,
    pub latency_ms: f64,
    pub energy_wms: f64,
    pub source: String,
    pub details: OperationDetails,
}
