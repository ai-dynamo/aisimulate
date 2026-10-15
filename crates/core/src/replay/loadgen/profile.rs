// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Duration-limited agentic replay configuration and client-side evidence.
//! The scheduler implements these rules locally. The behavioral reference is
//! SemiAnalysisAI/agentx-harness at 754356e9a39acc6cc6afb242d123bb57c3fb6f75,
//! timing/phase/runner.py and timing/strategies/agentic_replay.py (Apache-2.0).

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgenticProfileOptions {
    pub duration_seconds: f64,
    pub response_grace_seconds: f64,
    /// Upper bound for client cancellation acknowledgements, not server cleanup.
    /// Supported offline runtimes acknowledge cancellation synchronously.
    pub cancel_drain_seconds: f64,
    pub tree_idle_cap_seconds: f64,
    pub global_idle_cap_seconds: f64,
}

impl Default for AgenticProfileOptions {
    fn default() -> Self {
        Self {
            duration_seconds: 3600.0,
            response_grace_seconds: 30.0,
            cancel_drain_seconds: 10.0,
            tree_idle_cap_seconds: 300.0,
            global_idle_cap_seconds: 10.0,
        }
    }
}

impl AgenticProfileOptions {
    pub fn validate(&self) -> Result<()> {
        for (name, value, positive) in [
            ("duration_seconds", self.duration_seconds, true),
            ("response_grace_seconds", self.response_grace_seconds, false),
            ("cancel_drain_seconds", self.cancel_drain_seconds, false),
            ("tree_idle_cap_seconds", self.tree_idle_cap_seconds, true),
            (
                "global_idle_cap_seconds",
                self.global_idle_cap_seconds,
                true,
            ),
        ] {
            if !value.is_finite()
                || value < 0.0
                || (positive && value == 0.0)
                || !(value * 1000.0).is_finite()
            {
                bail!(
                    "agentic_profile.{name} must be finite and {}",
                    if positive { "positive" } else { "non-negative" }
                );
            }
        }
        if !((self.duration_seconds + self.response_grace_seconds + self.cancel_drain_seconds)
            * 1000.0)
            .is_finite()
        {
            bail!("agentic_profile deadlines overflow");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AgenticIdleShift {
    pub at_ms: f64,
    /// None denotes the global guard; Some identifies a single tree.
    pub play_id: Option<String>,
    pub shifted_by_ms: f64,
    pub timer_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AgenticProfileReport {
    pub schema: &'static str,
    pub options: AgenticProfileOptions,
    pub profile_start_ms: Option<f64>,
    pub admission_cutoff_ms: Option<f64>,
    pub response_grace_deadline_ms: Option<f64>,
    pub cancel_drain_deadline_ms: Option<f64>,
    pub admission_closed: bool,
    pub finished_at_ms: Option<f64>,
    /// Whether client cancellation acknowledgement exceeded its budget.
    /// False for synchronous offline cancellation, even with unsettled server work.
    pub cancel_drain_timed_out: bool,
    pub unsettled_server_requests: usize,
    pub plays_started: usize,
    pub client_completed_plays: usize,
    pub retired_plays: usize,
    pub server_quiescent_plays: usize,
    pub issued_requests: usize,
    pub successful_responses: usize,
    pub canceled_requests: usize,
    pub never_issued_requests: usize,
    pub client_in_flight_requests: usize,
    pub server_unsettled_requests: usize,
    pub first_request_ms: Option<f64>,
    pub last_successful_response_ms: Option<f64>,
    pub observation_duration_ms: Option<f64>,
    pub successful_request_throughput: Option<f64>,
    pub corpus_cursor: u64,
    pub idle_shifts: Vec<AgenticIdleShift>,
}

#[derive(Debug)]
pub(super) struct AgenticProfileState {
    pub options: AgenticProfileOptions,
    pub start_ms: Option<f64>,
    pub last_advanced_ms: f64,
    pub cutoff: bool,
    pub cursor: u64,
    pub active_plays: Vec<usize>,
    pub next_ordinals: Vec<u64>,
    pub replacements_at_ms: (Option<f64>, usize),
    pub tree_idle_since_ms: Vec<Option<f64>>,
    pub global_idle_since_ms: Option<f64>,
    pub issued_requests: usize,
    pub successful_responses: usize,
    pub canceled_requests: usize,
    pub never_issued_requests: usize,
    pub first_request_ms: Option<f64>,
    pub last_successful_response_ms: Option<f64>,
    pub idle_shifts: Vec<AgenticIdleShift>,
}

impl AgenticProfileState {
    pub fn deadlines(&self) -> Option<(f64, f64, f64)> {
        let cutoff = self.start_ms? + self.options.duration_seconds * 1000.0;
        let grace = cutoff + self.options.response_grace_seconds * 1000.0;
        Some((
            cutoff,
            grace,
            grace + self.options.cancel_drain_seconds * 1000.0,
        ))
    }
}
