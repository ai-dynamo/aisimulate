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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StaticOperationDiagnostics {
    pub name: String,
    pub latency_ms: f64,
    pub energy_wms: f64,
    pub source: String,
    pub details: OperationDetails,
}
