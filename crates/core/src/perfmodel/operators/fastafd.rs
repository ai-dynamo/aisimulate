// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::common::error::AicError;
use crate::operators::{PerformanceResult, Source};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FastAfdMoeStagePoint {
    pub num_tokens: u32,
    pub latency_ms: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FastAfdMoeStageOp {
    pub name: String,
    pub points: Vec<FastAfdMoeStagePoint>,
    pub profile_sha256: String,
    pub weight_bytes: f64,
}

impl FastAfdMoeStageOp {
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("FastAFD MoE stage name cannot be empty".into());
        }
        if self.points.is_empty() {
            return Err("FastAFD MoE stage requires at least one point".into());
        }
        if self.profile_sha256.len() != 64
            || !self
                .profile_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err("FastAFD profile_sha256 must be a lowercase SHA-256".into());
        }
        if !self.weight_bytes.is_finite() || self.weight_bytes < 0.0 {
            return Err("FastAFD MoE stage weight_bytes must be finite and non-negative".into());
        }
        let mut previous = 0;
        for point in &self.points {
            if point.num_tokens == 0 || point.num_tokens <= previous {
                return Err(
                    "FastAFD MoE stage token counts must be positive and strictly increasing".into(),
                );
            }
            if !point.latency_ms.is_finite() || point.latency_ms <= 0.0 {
                return Err("FastAFD MoE stage latencies must be finite and positive".into());
            }
            previous = point.num_tokens;
        }
        Ok(())
    }

    pub fn query(&self, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        self.validate().map_err(AicError::InvalidEngineConfig)?;
        let point = self
            .points
            .binary_search_by_key(&num_tokens, |point| point.num_tokens)
            .ok()
            .map(|index| &self.points[index])
            .ok_or_else(|| {
                AicError::InvalidEngineConfig(format!(
                    "no exact FastAFD MoE stage measurement for {num_tokens} tokens"
                ))
            })?;
        Ok(PerformanceResult::new(point.latency_ms, Source::Silicon))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage() -> FastAfdMoeStageOp {
        FastAfdMoeStageOp {
            name: "generation_fastafd_moe_stage".into(),
            points: vec![
                FastAfdMoeStagePoint {
                    num_tokens: 8,
                    latency_ms: 1.25,
                },
                FastAfdMoeStagePoint {
                    num_tokens: 16,
                    latency_ms: 1.75,
                },
            ],
            profile_sha256: "a".repeat(64),
            weight_bytes: 1024.0,
        }
    }

    #[test]
    fn lookup_is_exact() {
        assert_eq!(stage().query(16).unwrap().latency_ms, 1.75);
        let error = stage().query(12).unwrap_err();
        assert!(error.to_string().contains("no exact FastAFD"));
    }

    #[test]
    fn rejects_unsorted_points() {
        let mut op = stage();
        op.points.reverse();
        assert!(op.validate().unwrap_err().contains("strictly increasing"));
    }
}
