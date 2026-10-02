// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded evidence from actual direct-FPM lookups. This is not a replay
//! completion or prediction-accuracy assessment.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

use serde::{Deserialize, Serialize};

use crate::AicError;
use crate::operators::fpm_forward::{FpmForwardOp, FpmPhase};
use crate::perf_database::fpm_forward::{
    FPM_CELL_MATCH_COLUMNS, FPM_DECODE_AXES, FPM_PREFILL_AXES,
};

const GAP_LIMIT: usize = 128;

/// Counts of native table lookup resolutions, not requests or iterations.
/// Reusing a cached timing does not perform another native lookup.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FpmQueryCoverageCounts {
    pub measured: u64,
    pub interpolated: u64,
    pub unsupported: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FpmQueryPurpose {
    ForwardPass,
    /// A mixed pass subtracts the measured decode curve floor, or an
    /// interpolation of floors, using the same covered rows as decode.
    MixedDecodeBaseline,
}

/// One unsupported query, retained after the estimation call returns its error.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FpmQueryGap {
    pub phase: String,
    pub purpose: FpmQueryPurpose,
    pub model_path: String,
    pub cell_identity: BTreeMap<String, String>,
    pub cell_ids: Vec<String>,
    pub coordinates: BTreeMap<String, f64>,
    pub reason: String,
    pub occurrences: u64,
}

/// Cumulative evidence for one model instance. A snapshot cannot certify that
/// replay completed: the replay consumer must report completion separately.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FpmQueryCoverage {
    /// Explicit unit prevents callers from interpreting these as replay counts.
    pub counting_unit: String,
    pub queries: FpmQueryCoverageCounts,
    pub prefill: FpmQueryCoverageCounts,
    pub decode: FpmQueryCoverageCounts,
    pub mixed_decode_baseline: FpmQueryCoverageCounts,
    pub gaps: Vec<FpmQueryGap>,
    pub gap_limit: usize,
    /// Failed lookup occurrences omitted after the distinct-gap limit.
    pub omitted_gap_queries: u64,
}

impl Default for FpmQueryCoverage {
    fn default() -> Self {
        Self {
            counting_unit: "native_lookup_resolutions".into(),
            queries: Default::default(),
            prefill: Default::default(),
            decode: Default::default(),
            mixed_decode_baseline: Default::default(),
            gaps: Vec::new(),
            gap_limit: GAP_LIMIT,
            omitted_gap_queries: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DirectFpmResolution {
    Measured,
    Interpolated,
}

impl FpmQueryCoverageCounts {
    fn record(&mut self, result: Result<DirectFpmResolution, &AicError>) {
        match result {
            Ok(DirectFpmResolution::Measured) => self.measured += 1,
            Ok(DirectFpmResolution::Interpolated) => self.interpolated += 1,
            Err(_) => self.unsupported += 1,
        }
    }
}

impl FpmQueryCoverage {
    pub(crate) fn record(
        &mut self,
        op: &FpmForwardOp,
        cell_ids: &[String],
        coords: &[f64],
        purpose: FpmQueryPurpose,
        result: Result<DirectFpmResolution, &AicError>,
    ) {
        self.queries.record(result);
        match purpose {
            FpmQueryPurpose::MixedDecodeBaseline => &mut self.mixed_decode_baseline,
            FpmQueryPurpose::ForwardPass => match op.phase {
                FpmPhase::Prefill => &mut self.prefill,
                FpmPhase::Decode => &mut self.decode,
            },
        }
        .record(result);
        let Err(error) = result else { return };
        let axes: &[&str] = match op.phase {
            FpmPhase::Prefill => &FPM_PREFILL_AXES,
            FpmPhase::Decode => &FPM_DECODE_AXES,
        };
        let mut gap = FpmQueryGap {
            phase: op.phase.as_str().into(),
            purpose,
            model_path: op.model_path.clone(),
            cell_identity: FPM_CELL_MATCH_COLUMNS
                .iter()
                .zip(&op.match_identity)
                .map(|(name, value)| ((*name).into(), value.clone()))
                .collect(),
            cell_ids: cell_ids.to_vec(),
            coordinates: axes
                .iter()
                .zip(coords)
                .map(|(k, v)| ((*k).into(), *v))
                .collect(),
            reason: error.to_string(),
            occurrences: 0,
        };
        if let Some(existing) = self.gaps.iter_mut().find(|entry| {
            entry.phase == gap.phase
                && entry.purpose == gap.purpose
                && entry.model_path == gap.model_path
                && entry.cell_identity == gap.cell_identity
                && entry.cell_ids == gap.cell_ids
                && entry.coordinates == gap.coordinates
                && entry.reason == gap.reason
        }) {
            existing.occurrences += 1;
        } else if self.gaps.len() < GAP_LIMIT {
            gap.occurrences = 1;
            self.gaps.push(gap);
        } else {
            self.omitted_gap_queries += 1;
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct FpmCoverageState(Mutex<FpmQueryCoverage>);

impl Clone for FpmCoverageState {
    fn clone(&self) -> Self {
        // Model clones may be used for another replay/candidate. The loaded
        // engine is shared, but diagnostics must start empty for each clone.
        Self::default()
    }
}

impl FpmCoverageState {
    pub(crate) fn lock(&self) -> Result<MutexGuard<'_, FpmQueryCoverage>, AicError> {
        self.0.lock().map_err(|_| {
            AicError::InvalidEngineConfig("FPM query coverage accumulator was poisoned".into())
        })
    }
}
