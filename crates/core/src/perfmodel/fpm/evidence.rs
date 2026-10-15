// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Executed direct-FPM lookups and their forward-pass composition.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FpmCoordinates {
    pub batch_size: f64,
    pub total_prefill_tokens: Option<f64>,
    pub total_kv_read_tokens: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FpmMeasurementSupport {
    pub coordinates: FpmCoordinates,
    pub latency_ms: f64,
    /// Product of the linear interpolation weights at this measured point.
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FpmQueryResolution {
    ExactLookup,
    WithinCurveInterpolation,
    CrossKvInterpolation,
    CrossBatchInterpolation,
    /// Prefill fallback that interpolates KV at each supported token count first.
    KvFirstInterpolation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectFpmQueryEvidence {
    pub phase: String,
    pub model_path: String,
    /// True for the mixed-pass baseline: support uses each selected row's
    /// measured KV floor, while `query` retains the paired decode request.
    pub decode_baseline: bool,
    pub query: FpmCoordinates,
    pub resolution: FpmQueryResolution,
    /// Raw lookup latency, before composition, correction or replay speedup.
    pub latency_ms: f64,
    pub support: Vec<FpmMeasurementSupport>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FpmRankEstimate {
    pub rank: usize,
    pub latency_ms: f64,
    pub prefill_ms: Option<f64>,
    pub decode_ms: Option<f64>,
    pub decode_baseline_ms: Option<f64>,
    /// For a mixed pass, max(decode_ms - decode_baseline_ms, 0).
    pub marginal_decode_ms: Option<f64>,
    pub queries: Vec<DirectFpmQueryEvidence>,
}

/// Additive detailed result from the canonical returned model. Native timing
/// reduces ranks by maximum, then applies `correction_factor`. Rank lookup
/// evidence is supplied for whole-model FPM; other estimators leave it empty.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForwardPassEstimate {
    pub latency_ms: Option<f64>,
    pub native_latency_ms: Option<f64>,
    pub correction_factor: Option<f64>,
    /// First rank attaining the positive native maximum; None for empty work.
    pub max_rank: Option<usize>,
    pub ranks: Vec<FpmRankEstimate>,
}

/// Identical complete estimates are retained once, with the number of timing
/// invocations that used them. No records are truncated. `latency_scale` is
/// replay's synthetic speedup multiplier and does not alter measured support.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FpmEstimateEvidence {
    pub estimate: ForwardPassEstimate,
    pub count: u64,
    pub latency_scale: f64,
}

pub(crate) fn accumulate_fpm_estimates(
    target: &mut Vec<FpmEstimateEvidence>,
    incoming: Vec<FpmEstimateEvidence>,
) -> Result<(), &'static str> {
    for record in incoming {
        if let Some(existing) = target.iter_mut().find(|existing| {
            existing.estimate == record.estimate && existing.latency_scale == record.latency_scale
        }) {
            existing.count = existing
                .count
                .checked_add(record.count)
                .ok_or("FPM evidence invocation count overflow")?;
        } else {
            target.push(record);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::FpmQueryResolution;

    #[test]
    fn query_resolution_wire_values_preserve_existing_indices() {
        for (resolution, name, index) in [
            (FpmQueryResolution::ExactLookup, "exact_lookup", 0u32),
            (
                FpmQueryResolution::WithinCurveInterpolation,
                "within_curve_interpolation",
                1,
            ),
            (
                FpmQueryResolution::CrossKvInterpolation,
                "cross_kv_interpolation",
                2,
            ),
            (
                FpmQueryResolution::CrossBatchInterpolation,
                "cross_batch_interpolation",
                3,
            ),
            (
                FpmQueryResolution::KvFirstInterpolation,
                "kv_first_interpolation",
                4,
            ),
        ] {
            let json = serde_json::to_string(&resolution).unwrap();
            assert_eq!(json, format!("\"{name}\""));
            assert_eq!(
                serde_json::from_str::<FpmQueryResolution>(&json).unwrap(),
                resolution
            );
            assert_eq!(
                bincode::serialize(&resolution).unwrap(),
                index.to_le_bytes()
            );
        }
    }
}
