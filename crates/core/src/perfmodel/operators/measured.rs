// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Externally measured, fixed-latency stage operators.
//!
//! This is deliberately not a perf-database table. A caller must establish
//! that an external measurement matches the requested workload and topology;
//! the fixed value then composes normally, including inside `OverlapOp`.

use serde::{Deserialize, Serialize};

use crate::common::error::AicError;
use crate::operators::{PerformanceResult, Source};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasuredStageOp {
    pub name: String,
    pub latency_ms: f64,
}

impl MeasuredStageOp {
    pub fn query(&self) -> Result<PerformanceResult, AicError> {
        if !self.latency_ms.is_finite() || self.latency_ms <= 0.0 {
            return Err(AicError::InvalidEngineConfig(format!(
                "MeasuredStageOp '{}' latency_ms must be finite and positive, got {}",
                self.name, self.latency_ms
            )));
        }
        Ok(PerformanceResult::new(self.latency_ms, Source::External))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_external_measurement_without_energy() {
        let result = MeasuredStageOp {
            name: "generation_measured_moe_stage".into(),
            latency_ms: 3.5,
        }
        .query()
        .expect("valid measured stage");

        assert_eq!(result.latency_ms, 3.5);
        assert_eq!(result.energy_wms, 0.0);
        assert_eq!(result.source, Source::External);
    }

    #[test]
    fn rejects_non_positive_or_non_finite_latency() {
        for latency_ms in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(MeasuredStageOp {
                name: "bad".into(),
                latency_ms,
            }
            .query()
            .is_err());
        }
    }
}
