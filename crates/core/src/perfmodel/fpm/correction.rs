// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Online correction policies for the forward-pass perf model.
//!
//! Buckets observed `(workload feature, observed_ms / native_ms)` correction
//! samples into the legacy fixed grid or an expanding local calibrator. Hybrid
//! keeps independent full-budget stores and selects fixed correction inside the
//! configured grid, including empty/cold cells, and dynamic correction outside.

use super::dynamic_correction::DynamicCorrection;
use super::options::{CorrectionMode, ForwardPassPerfOptions};
use super::samples::{AxisRange, BucketedSamples, StoreStats, WithOptions, median_ratio};

#[derive(Clone, Debug)]
pub(crate) struct FixedCorrectionBuckets {
    samples: BucketedSamples<CorrectionObservation>,
    min_observations: usize,
    min_faster_correction_factor: Option<f64>,
    max_slower_correction_factor: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
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
#[derive(Clone, Debug)]
pub(crate) enum CorrectionBuckets {
    Fixed(FixedCorrectionBuckets),
    Dynamic(DynamicCorrection),
    Hybrid(Box<HybridCorrection>),
}

#[derive(Clone, Debug)]
pub(crate) struct HybridCorrection {
    fixed: FixedCorrectionBuckets,
    dynamic: DynamicCorrection,
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
