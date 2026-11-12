// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Online correction policies for the forward-pass perf model.
//!
//! Buckets observed `(workload feature, observed_ms / native_ms)` correction
//! samples into the legacy fixed grid or an expanding local calibrator. Hybrid
//! keeps independent full-budget stores and selects fixed correction inside the
//! configured grid, including empty/cold cells, and dynamic correction outside.

use serde::{Deserialize, Serialize};

use super::dynamic_correction::{DynamicCorrection, DynamicCorrectionSnapshot};
use super::options::{CorrectionMode, ForwardPassPerfOptions, validate_options};
use super::samples::{
    AxisRange, BucketedSamples, FixedSamplesSnapshot, StoreStats, WithOptions, median_ratio,
};
use crate::AicError;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct FixedCorrectionBuckets {
    samples: BucketedSamples<CorrectionObservation>,
    min_observations: usize,
    min_faster_correction_factor: Option<f64>,
    max_slower_correction_factor: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorrectionObservation {
    correction_factor: f64,
}

impl WithOptions for FixedCorrectionBuckets {
    fn with_options(options: &ForwardPassPerfOptions, axis_ranges: &[AxisRange]) -> Self {
        Self {
            samples: BucketedSamples::new_fixed(options, axis_ranges),
            min_observations: options.min_observations,
            min_faster_correction_factor: options.min_faster_correction_factor,
            max_slower_correction_factor: options.max_slower_correction_factor,
        }
    }
}

impl StoreStats for FixedCorrectionBuckets {
    fn observation_count(&self) -> usize {
        self.samples.total_observations
    }

    fn is_ready(&self) -> bool {
        // Match planner's regression readiness semantics: min_observations is
        // checked across the whole inferred workload kind, not per region.
        // Regions only decide which correction factor to apply once the
        // workload kind is ready.
        self.samples.total_observations >= self.min_observations
    }
}

impl FixedCorrectionBuckets {
    pub(crate) fn add_observation(&mut self, x: Vec<f64>, observed_ms: f64, native_ms: f64) {
        if native_ms.is_finite() && native_ms > 0.0 && observed_ms.is_finite() && observed_ms > 0.0
        {
            // Corrections are absolute observed/native samples, not
            // incremental multipliers. Bound each sample before the median is
            // computed so repeated outliers cannot exceed either directional
            // limit.
            let correction_factor = observed_ms / native_ms;
            let lower_bounded_correction_factor = self
                .min_faster_correction_factor
                .map_or(correction_factor, |min_factor| {
                    correction_factor.max(min_factor)
                });
            let bounded_correction_factor = self
                .max_slower_correction_factor
                .map_or(lower_bounded_correction_factor, |max_factor| {
                    lower_bounded_correction_factor.min(max_factor)
                });
            self.samples.add(
                x,
                CorrectionObservation {
                    correction_factor: bounded_correction_factor,
                },
            );
        }
    }

    pub(crate) fn correction_factor_for(&self, x: &[f64]) -> f64 {
        if !self.is_ready() {
            return 1.0;
        }
        // Every region has an implicit correction factor of 1.0. A populated
        // in-range region overrides that default with its local median
        // observed/native ratio after the workload-kind-wide readiness gate passes.
        let Some(key) = self.samples.bucket_key_if_in_bounds(x) else {
            return 1.0;
        };
        let Some(bucket) = self.samples.buckets.get(&key) else {
            return 1.0;
        };
        median_ratio(
            bucket
                .iter()
                .map(|(_, observation)| observation.correction_factor),
        )
        .unwrap_or(1.0)
    }

    pub(crate) fn ready_bucket_count(&self) -> usize {
        if self.is_ready() {
            self.samples.buckets.len()
        } else {
            0
        }
    }

    pub(crate) fn correction_factors(&self) -> Vec<f64> {
        if !self.is_ready() {
            return Vec::new();
        }
        self.samples
            .buckets
            .values()
            .filter_map(|bucket| {
                median_ratio(
                    bucket
                        .iter()
                        .map(|(_, observation)| observation.correction_factor),
                )
            })
            .collect()
    }
}

/// Dispatch keeps native fixed calibration unchanged while allowing learned
/// forests to retain and calibrate observations beyond the old grid limits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum CorrectionBuckets {
    Fixed(FixedCorrectionBuckets),
    Dynamic(DynamicCorrection),
    Hybrid(Box<HybridCorrection>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct HybridCorrection {
    fixed: FixedCorrectionBuckets,
    dynamic: DynamicCorrection,
}

pub(crate) fn same_optional_bits(left: Option<f64>, right: Option<f64>) -> bool {
    left.map(f64::to_bits) == right.map(f64::to_bits)
}

/// A ratio admitted by dynamic correction: finite, positive, and inside the
/// configured directional bounds.
pub(crate) fn valid_bounded_ratio(ratio: f64, lower: Option<f64>, upper: Option<f64>) -> bool {
    ratio.is_finite()
        && ratio > 0.0
        && lower.is_none_or(|bound| ratio >= bound)
        && upper.is_none_or(|bound| ratio <= bound)
}

impl FixedCorrectionBuckets {
    fn validate_restored(&self, fresh: &Self) -> Result<(), String> {
        if self.min_observations != fresh.min_observations
            || !same_optional_bits(
                self.min_faster_correction_factor,
                fresh.min_faster_correction_factor,
            )
            || !same_optional_bits(
                self.max_slower_correction_factor,
                fresh.max_slower_correction_factor,
            )
        {
            return Err("fixed correction controls differ from configuration".into());
        }
        let (lower, upper) = (
            self.min_faster_correction_factor,
            self.max_slower_correction_factor,
        );
        self.samples
            .validate_restored(&fresh.samples, |_, observation| {
                // Ratios of finite positive latencies may overflow to infinity
                // or underflow to zero when a bound is disabled; never NaN.
                let ratio = observation.correction_factor;
                if !ratio.is_nan()
                    && ratio >= 0.0
                    && lower.is_none_or(|bound| ratio >= bound)
                    && upper.is_none_or(|bound| ratio <= bound)
                {
                    Ok(())
                } else {
                    Err("fixed correction ratio is outside its admitted range".into())
                }
            })
    }
}

impl CorrectionBuckets {
    /// Validate a decoded store against one built from the same configuration.
    pub(crate) fn validate_restored(&self, fresh: &Self) -> Result<(), String> {
        match (self, fresh) {
            (Self::Fixed(saved), Self::Fixed(fresh)) => saved.validate_restored(fresh),
            (Self::Dynamic(saved), Self::Dynamic(fresh)) => saved.validate_restored(fresh),
            (Self::Hybrid(saved), Self::Hybrid(fresh)) => {
                saved.fixed.validate_restored(&fresh.fixed)?;
                saved.dynamic.validate_restored(&fresh.dynamic)
            }
            _ => Err("correction policy differs from configuration".into()),
        }
    }
}

/// Exact retained state for the new deterministic mixed calibrators. Restore
/// additionally checks these settings against the model's constructed policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HybridCorrectionSnapshot {
    schema_version: u32,
    min_observations: usize,
    min_faster_correction_factor: Option<f64>,
    max_slower_correction_factor: Option<f64>,
    fixed: FixedSamplesSnapshot<CorrectionObservation>,
    dynamic: DynamicCorrectionSnapshot,
}

impl WithOptions for CorrectionBuckets {
    fn with_options(options: &ForwardPassPerfOptions, axis_ranges: &[AxisRange]) -> Self {
        match options.correction_mode {
            CorrectionMode::Hybrid => Self::Hybrid(Box::new(HybridCorrection {
                fixed: FixedCorrectionBuckets::with_options(options, axis_ranges),
                dynamic: DynamicCorrection::new(options, axis_ranges.len()),
            })),
            CorrectionMode::DynamicLocal => {
                Self::Dynamic(DynamicCorrection::new(options, axis_ranges.len()))
            }
            CorrectionMode::Auto | CorrectionMode::Fixed => {
                Self::Fixed(FixedCorrectionBuckets::with_options(options, axis_ranges))
            }
        }
    }
}

impl StoreStats for CorrectionBuckets {
    fn observation_count(&self) -> usize {
        match self {
            Self::Fixed(value) => value.observation_count(),
            Self::Dynamic(value) => value.observation_count(),
            Self::Hybrid(value) => {
                value.fixed.observation_count() + value.dynamic.observation_count()
            }
        }
    }

    fn is_ready(&self) -> bool {
        match self {
            Self::Fixed(value) => value.is_ready(),
            Self::Dynamic(value) => value.is_ready(),
            Self::Hybrid(value) => value.fixed.is_ready() || value.dynamic.is_ready(),
        }
    }
}

impl CorrectionBuckets {
    pub(crate) fn new_mixed_hybrid(
        options: &ForwardPassPerfOptions,
        axis_ranges: &[AxisRange],
    ) -> Result<Self, AicError> {
        validate_options(options)?;
        if options.correction_mode != CorrectionMode::Hybrid || axis_ranges.len() != 2 {
            return Err(invalid_mixed_snapshot(
                "mixed correction requires hybrid mode and two score axes",
            ));
        }
        let mut result = Self::with_options(options, axis_ranges);
        let Self::Hybrid(value) = &mut result else {
            unreachable!("validated hybrid correction");
        };
        value.fixed.samples.use_deterministic_fixed_ties()?;
        Ok(result)
    }

    pub(crate) fn mixed_admission_count(&self) -> Result<u64, AicError> {
        let Self::Hybrid(value) = self else {
            return Err(invalid_mixed_snapshot(
                "admission count requires a hybrid calibrator",
            ));
        };
        Ok(value.dynamic.admission_count())
    }

    pub(crate) fn snapshot_mixed_hybrid(&self) -> Result<HybridCorrectionSnapshot, AicError> {
        let Self::Hybrid(value) = self else {
            return Err(invalid_mixed_snapshot(
                "snapshot requires a hybrid calibrator",
            ));
        };
        Ok(HybridCorrectionSnapshot {
            schema_version: 1,
            min_observations: value.fixed.min_observations,
            min_faster_correction_factor: value.fixed.min_faster_correction_factor,
            max_slower_correction_factor: value.fixed.max_slower_correction_factor,
            fixed: value.fixed.samples.fixed_snapshot()?,
            dynamic: value.dynamic.snapshot(),
        })
    }

    pub(crate) fn restore_mixed_hybrid(
        &mut self,
        snapshot: HybridCorrectionSnapshot,
    ) -> Result<(), AicError> {
        let Self::Hybrid(value) = self else {
            return Err(invalid_mixed_snapshot(
                "restore requires a hybrid calibrator",
            ));
        };
        if snapshot.schema_version != 1
            || snapshot.min_observations != value.fixed.min_observations
            || snapshot.min_faster_correction_factor != value.fixed.min_faster_correction_factor
            || snapshot.max_slower_correction_factor != value.fixed.max_slower_correction_factor
        {
            return Err(invalid_mixed_snapshot(
                "schema or calibration settings differ",
            ));
        }
        for (_, row) in &snapshot.fixed.rows {
            let ratio = row.correction_factor;
            if !ratio.is_finite()
                || ratio <= 0.0
                || snapshot
                    .min_faster_correction_factor
                    .is_some_and(|bound| ratio < bound)
                || snapshot
                    .max_slower_correction_factor
                    .is_some_and(|bound| ratio > bound)
            {
                return Err(invalid_mixed_snapshot("invalid fixed correction ratio"));
            }
        }
        // Validate both policies before replacing either history.
        let mut restored = value.clone();
        restored
            .fixed
            .samples
            .restore_fixed_snapshot(snapshot.fixed)?;
        restored.dynamic.restore_snapshot(snapshot.dynamic)?;
        *value = restored;
        Ok(())
    }

    pub(crate) fn add_observation(&mut self, x: Vec<f64>, observed_ms: f64, base_ms: f64) {
        match self {
            Self::Fixed(value) => value.add_observation(x, observed_ms, base_ms),
            Self::Dynamic(value) => value.add_observation(x, observed_ms, base_ms),
            Self::Hybrid(value) => {
                // Preserve the fixed sampler's admission and eviction behavior.
                // Dynamic must also see in-grid feedback for its global fallback.
                value.fixed.add_observation(x.clone(), observed_ms, base_ms);
                value.dynamic.add_observation(x, observed_ms, base_ms);
            }
        }
    }

    pub(crate) fn correction_factor_for(&self, x: &[f64]) -> f64 {
        match self {
            Self::Fixed(value) => value.correction_factor_for(x),
            Self::Dynamic(value) => value.correction_factor_for(x),
            Self::Hybrid(value) => {
                // Geometry alone selects the policy: readiness and empty cells
                // must not divert an in-grid query to the dynamic fallback.
                if value.fixed.samples.is_in_bounds(x) {
                    value.fixed.correction_factor_for(x)
                } else {
                    value.dynamic.correction_factor_for(x)
                }
            }
        }
    }

    pub(crate) fn ready_bucket_count(&self) -> usize {
        match self {
            Self::Fixed(value) => value.ready_bucket_count(),
            Self::Dynamic(value) => value.ready_bucket_count(),
            Self::Hybrid(value) => {
                value.fixed.ready_bucket_count() + value.dynamic.ready_bucket_count()
            }
        }
    }

    pub(crate) fn correction_factors(&self) -> Vec<f64> {
        match self {
            Self::Fixed(value) => value.correction_factors(),
            Self::Dynamic(value) => value.correction_factors(),
            Self::Hybrid(value) => {
                let mut factors = value.fixed.correction_factors();
                factors.extend(value.dynamic.correction_factors());
                factors
            }
        }
    }
}

fn invalid_mixed_snapshot(reason: &str) -> AicError {
    AicError::InvalidEngineConfig(format!(
        "invalid mixed hybrid calibration snapshot: {reason}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_hybrid_snapshot_preserves_predictions_and_future_admissions() {
        let options = ForwardPassPerfOptions {
            correction_mode: CorrectionMode::Hybrid,
            max_observations: 8,
            min_observations: 2,
            bucket_shape: Some([2, 2]),
            ..Default::default()
        };
        let ranges = [AxisRange::from_zero_to_f64(100.5).unwrap(); 2];
        let mut original = CorrectionBuckets::new_mixed_hybrid(&options, &ranges).unwrap();
        let feed = |model: &mut CorrectionBuckets, index: usize| {
            let x = [
                [0.0, 0.0],
                [20.0, 80.0],
                [80.0, 20.0],
                [100.5, 100.5],
                [200.0, 50.0],
                [50.0, 1000.0],
            ][index % 6];
            model.add_observation(x.to_vec(), [5.0, 10.0, 15.0, 20.0][index % 4], 10.0);
        };
        for index in 0..13 {
            feed(&mut original, index);
        }
        let snapshot = original.snapshot_mixed_hybrid().unwrap();
        let json = serde_json::to_string(&snapshot).unwrap();
        let mut restored = CorrectionBuckets::new_mixed_hybrid(&options, &ranges).unwrap();
        restored
            .restore_mixed_hybrid(serde_json::from_str(&json).unwrap())
            .unwrap();
        assert_eq!(restored.snapshot_mixed_hybrid().unwrap(), snapshot);
        // Continue beyond the age horizon, with full grids and tied buckets.
        for index in 13..100 {
            for x in [
                [0.0, 0.0],
                [50.0, 50.0],
                [100.5, 100.5],
                [500.0, 50.0],
                [1e6, 1e6],
            ] {
                assert_eq!(
                    original.correction_factor_for(&x),
                    restored.correction_factor_for(&x)
                );
            }
            feed(&mut original, index);
            feed(&mut restored, index);
            assert_eq!(
                original.snapshot_mixed_hybrid().unwrap(),
                restored.snapshot_mixed_hybrid().unwrap()
            );
        }
    }

    #[test]
    fn mixed_hybrid_snapshot_rejects_incompatible_or_corrupt_state_atomically() {
        let options = ForwardPassPerfOptions {
            correction_mode: CorrectionMode::Hybrid,
            min_observations: 1,
            ..Default::default()
        };
        let ranges = [AxisRange::from_zero_to(100); 2];
        let mut model = CorrectionBuckets::new_mixed_hybrid(&options, &ranges).unwrap();
        model.add_observation(vec![10.0, 20.0], 15.0, 10.0);
        let before = model.snapshot_mixed_hybrid().unwrap();
        let value = serde_json::to_value(&before).unwrap();
        for (pointer, replacement) in [
            ("/schema_version", serde_json::json!(2)),
            ("/min_observations", serde_json::json!(2)),
            ("/fixed/axis_max/0", serde_json::json!(200.0)),
            ("/fixed/rows/0/0/0", serde_json::json!(-1.0)),
            ("/fixed/rows/0/1/correction_factor", serde_json::json!(0.0)),
            ("/dynamic/next_admission", serde_json::json!(0)),
            ("/dynamic/recent/0/ratio", serde_json::json!(1.6)),
            ("/dynamic/primary/0/admission", serde_json::json!(4)),
            ("/dynamic/log_bounds/0/0", serde_json::json!(-1.0)),
        ] {
            let mut corrupt = value.clone();
            *corrupt.pointer_mut(pointer).unwrap() = replacement;
            let snapshot = serde_json::from_value(corrupt).unwrap();
            assert!(model.restore_mixed_hybrid(snapshot).is_err(), "{pointer}");
            assert_eq!(model.snapshot_mixed_hybrid().unwrap(), before, "{pointer}");
        }
        let legacy = CorrectionBuckets::with_options(&options, &ranges);
        assert!(legacy.snapshot_mixed_hybrid().is_err());
    }

    #[test]
    fn mixed_hybrid_keeps_existing_factors_before_capacity_eviction() {
        let options = ForwardPassPerfOptions {
            correction_mode: CorrectionMode::Hybrid,
            ..Default::default()
        };
        let ranges = [AxisRange::from_zero_to(100); 2];
        let mut mixed = CorrectionBuckets::new_mixed_hybrid(&options, &ranges).unwrap();
        let mut legacy = CorrectionBuckets::with_options(&options, &ranges);
        // Five historical ratios of 1.1 populate one fixed cell; the current
        // measured 18/16 ratio is admitted only after its 17.6 prediction.
        for _ in 0..5 {
            mixed.add_observation(vec![10.0, 10.0], 17.6, 16.0);
            legacy.add_observation(vec![10.0, 10.0], 17.6, 16.0);
        }
        assert_eq!(16.0 * mixed.correction_factor_for(&[10.0, 10.0]), 17.6);
        assert_eq!(mixed.correction_factor_for(&[80.0, 80.0]), 1.0);
        for x in [[10.0, 10.0], [80.0, 80.0], [200.0, 200.0]] {
            assert_eq!(
                mixed.correction_factor_for(&x),
                legacy.correction_factor_for(&x)
            );
        }
        mixed.add_observation(vec![10.0, 10.0], 18.0, 16.0);
        assert_eq!(16.0 * mixed.correction_factor_for(&[10.0, 10.0]), 17.6);
    }

    #[test]
    fn hybrid_uses_fixed_geometry_even_for_cold_and_empty_cells() {
        for ndim in [1, 2] {
            let options = ForwardPassPerfOptions {
                correction_mode: CorrectionMode::Hybrid,
                ..Default::default()
            };
            let ranges = vec![AxisRange::from_zero_to(100); ndim];
            let mut hybrid = CorrectionBuckets::with_options(&options, &ranges);
            for _ in 0..5 {
                hybrid.add_observation(vec![200.0; ndim], 20.0, 10.0);
            }
            // Dynamic is ready but fixed has no evidence. The inclusive upper
            // boundary and all cold in-grid cells must still use fixed's 1.0.
            assert_eq!(hybrid.correction_factor_for(&vec![100.0; ndim]), 1.0);
            assert_eq!(hybrid.correction_factor_for(&vec![25.0; ndim]), 1.0);
            for axis in 0..ndim {
                let mut outside = vec![100.0; ndim];
                outside[axis] = 101.0;
                assert_eq!(hybrid.correction_factor_for(&outside), 2.0);
            }

            for _ in 0..5 {
                hybrid.add_observation(vec![100.0; ndim], 5.0, 10.0);
            }
            assert_eq!(hybrid.correction_factor_for(&vec![100.0; ndim]), 0.5);
            assert_eq!(hybrid.correction_factor_for(&vec![0.0; ndim]), 1.0);
            // Dynamic receives in-grid feedback too: the recent median of five
            // ratios 2 and five ratios 0.5 is 1.25, used beyond observed bounds.
            assert_eq!(hybrid.correction_factor_for(&vec![300.0; ndim]), 1.25);
        }
    }

    #[test]
    fn hybrid_preserves_both_full_histories_through_eviction() {
        for ndim in [1, 2] {
            let options = ForwardPassPerfOptions {
                correction_mode: CorrectionMode::Hybrid,
                max_observations: 10,
                min_observations: 5,
                ..Default::default()
            };
            let ranges = vec![AxisRange::from_zero_to(100); ndim];
            let mut hybrid = CorrectionBuckets::with_options(&options, &ranges);
            let CorrectionBuckets::Hybrid(_) = &hybrid else {
                panic!("hybrid must construct both stores");
            };
            // An independently constructed fixed store must break eviction
            // ties exactly like the hybrid's own fixed store.
            let mut fixed = FixedCorrectionBuckets::with_options(&options, &ranges);
            let mut dynamic = DynamicCorrection::new(&options, ndim);
            for i in 0..80 {
                let x = vec![[0.0, 25.0, 100.0, 200.0][i % 4]; ndim];
                let observed = [5.0, 10.0, 15.0][i % 3];
                hybrid.add_observation(x.clone(), observed, 10.0);
                fixed.add_observation(x.clone(), observed, 10.0);
                dynamic.add_observation(x, observed, 10.0);
                for coordinate in [0.0, 25.0, 50.0, 100.0, 101.0, 200.0, 300.0] {
                    let query = vec![coordinate; ndim];
                    let expected = if coordinate <= 100.0 {
                        fixed.correction_factor_for(&query)
                    } else {
                        dynamic.correction_factor_for(&query)
                    };
                    assert_eq!(
                        hybrid.correction_factor_for(&query).to_bits(),
                        expected.to_bits()
                    );
                }
            }
            let CorrectionBuckets::Hybrid(stores) = hybrid else {
                unreachable!()
            };
            assert_eq!(stores.fixed.observation_count(), 10);
            assert_eq!(stores.dynamic, dynamic);
        }
    }
}
