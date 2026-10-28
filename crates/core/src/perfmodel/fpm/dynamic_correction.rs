// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Online correction with observation-driven, expanding log-space bounds.
//!
//! A balanced local sample set supplies bucket medians; a separate recent FIFO
//! supplies the global fallback, including outside the observed bounds. Samples
//! can occur in both sets. Their combined slot budget is bounded, and samples
//! expire after twice that budget in admitted observations. All rebucketing and
//! statistics happen on admission, so prediction neither scans samples nor
//! changes geometry.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use super::correction::{same_optional_bits, valid_bounded_ratio};
use super::options::ForwardPassPerfOptions;
use super::samples::integer_sqrt;

const SHRINKAGE_OBSERVATIONS: f64 = 20.0;
type BucketKey = [usize; 2];

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct Observation {
    admission: usize,
    log_x: [f64; 2],
    ratio: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct DynamicCorrection {
    ndim: usize,
    bins: [usize; 2],
    log_bounds: Option<[[f64; 2]; 2]>,
    primary: Vec<Observation>,
    recent: VecDeque<Observation>,
    primary_capacity: usize,
    recent_capacity: usize,
    max_age: usize,
    next_admission: usize,
    min_observations: usize,
    lower_factor: Option<f64>,
    upper_factor: Option<f64>,
    fallback: f64,
    /// Derived from `primary`/`recent` on admission. Snapshots keep it so a
    /// restore can recompute and compare it, detecting semantic drift.
    #[serde(with = "crate::fpm::snapshot::codec::sorted_map")]
    factors: BTreeMap<BucketKey, f64>,
}

impl DynamicCorrection {
    pub(crate) fn new(options: &ForwardPassPerfOptions, ndim: usize) -> Self {
        assert!((1..=2).contains(&ndim));
        let shape = options.bucket_shape.unwrap_or_else(|| {
            let width = integer_sqrt(options.bucket_count);
            [width, width]
        });
        let bins = if ndim == 1 {
            [
                options
                    .bucket_shape
                    .map_or(options.bucket_count, |shape| shape[0] * shape[1]),
                1,
            ]
        } else {
            shape
        };
        // The global fallback must be able to become ready for every valid
        // budget. Small budgets may leave too few primary samples for local
        // factors, in which case all queries use the recent-global fallback.
        let recent_capacity = (options.max_observations / 4).max(options.min_observations);
        Self {
            ndim,
            bins,
            log_bounds: None,
            primary: Vec::new(),
            recent: VecDeque::new(),
            primary_capacity: options.max_observations - recent_capacity,
            recent_capacity,
            max_age: options.max_observations.saturating_mul(2),
            next_admission: 0,
            min_observations: options.min_observations,
            lower_factor: options.min_faster_correction_factor,
            upper_factor: options.max_slower_correction_factor,
            fallback: 1.0,
            factors: BTreeMap::new(),
        }
    }

    pub(crate) fn add_observation(&mut self, x: Vec<f64>, observed_ms: f64, base_ms: f64) {
        let Some(log_x) = self.log_coordinates(&x) else {
            return;
        };
        if !observed_ms.is_finite() || observed_ms <= 0.0 || !base_ms.is_finite() || base_ms <= 0.0
        {
            return;
        }
        let ratio = observed_ms / base_ms;
        // Reject overflow and underflow before clipping. Invalid evidence must
        // not grow bounds, advance sample age, or contaminate the fallback.
        if !ratio.is_finite() || ratio <= 0.0 {
            return;
        }
        let observation = Observation {
            admission: self.next_admission,
            log_x,
            ratio: self.project(ratio),
        };
        self.next_admission = self.next_admission.saturating_add(1);
        let oldest = self.next_admission.saturating_sub(self.max_age);
        self.primary.retain(|row| row.admission >= oldest);
        while self
            .recent
            .front()
            .is_some_and(|row| row.admission < oldest)
        {
            self.recent.pop_front();
        }
        self.primary.push(observation);
        if self.recent_capacity > 0 {
            self.recent.push_back(observation);
            if self.recent.len() > self.recent_capacity {
                self.recent.pop_front();
            }
        }

        let bounds = self
            .log_bounds
            .get_or_insert([[log_x[0]; 2], [log_x[1]; 2]]);
        for axis in 0..self.ndim {
            bounds[axis][0] = bounds[axis][0].min(log_x[axis]);
            bounds[axis][1] = bounds[axis][1].max(log_x[axis]);
        }
        self.rebuild_factors();
    }

    pub(crate) fn correction_factor_for(&self, x: &[f64]) -> f64 {
        let Some(log_x) = self.log_coordinates(x) else {
            return 1.0;
        };
        self.bucket_key(log_x)
            .and_then(|key| self.factors.get(&key).copied())
            .unwrap_or(self.fallback)
    }

    /// Number of distinct retained observations, excluding duplicate FIFO slots.
    pub(crate) fn observation_count(&self) -> usize {
        // The recent FIFO contains every admitted ID from its oldest entry to
        // the newest. Primary rows before that interval are the only extras.
        self.recent.front().map_or(self.primary.len(), |oldest| {
            self.recent.len()
                + self
                    .primary
                    .iter()
                    .take_while(|row| row.admission < oldest.admission)
                    .count()
        })
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.primary.len() >= self.min_observations || self.recent_is_ready()
    }

    pub(crate) fn ready_bucket_count(&self) -> usize {
        self.factors.len()
    }

    pub(crate) fn correction_factors(&self) -> Vec<f64> {
        let mut factors = self.factors.values().copied().collect::<Vec<_>>();
        if self.recent_is_ready() {
            factors.push(self.fallback);
        }
        factors
    }

    fn recent_is_ready(&self) -> bool {
        self.recent.len() >= self.min_observations
    }

    fn log_coordinates(&self, x: &[f64]) -> Option<[f64; 2]> {
        if x.len() != self.ndim || x.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return None;
        }
        let mut log_x = [0.0; 2];
        for (target, value) in log_x.iter_mut().zip(x) {
            *target = value.ln_1p();
        }
        Some(log_x)
    }

    fn bucket_key(&self, log_x: [f64; 2]) -> Option<BucketKey> {
        let bounds = self.log_bounds?;
        let mut key = [0; 2];
        for axis in 0..self.ndim {
            let [lo, hi] = bounds[axis];
            if log_x[axis] < lo || log_x[axis] > hi {
                return None;
            }
            if hi > lo {
                key[axis] = (((log_x[axis] - lo) / (hi - lo) * self.bins[axis] as f64) as usize)
                    .min(self.bins[axis] - 1);
            }
        }
        Some(key)
    }

    fn project(&self, factor: f64) -> f64 {
        let factor = self.lower_factor.map_or(factor, |lower| factor.max(lower));
        self.upper_factor.map_or(factor, |upper| factor.min(upper))
    }

    fn rebuild_factors(&mut self) {
        let mut buckets: BTreeMap<BucketKey, Vec<f64>> = BTreeMap::new();
        for row in &self.primary {
            let key = self
                .bucket_key(row.log_x)
                .expect("admitted row is in bounds");
            buckets.entry(key).or_default().push(row.ratio);
        }
        if self.primary.len() > self.primary_capacity {
            // BTreeMap order and strict size comparison choose the
            // lexicographically first fullest bucket. Primary admission order
            // survives rebucketing, so its first matching row is the oldest.
            let mut fattest = ([0; 2], 0);
            for (&key, values) in &buckets {
                if values.len() > fattest.1 {
                    fattest = (key, values.len());
                }
            }
            let index = self
                .primary
                .iter()
                .position(|row| self.bucket_key(row.log_x) == Some(fattest.0))
                .expect("fullest bucket contains an observation");
            self.primary.remove(index);
            let values = buckets.get_mut(&fattest.0).expect("bucket exists");
            values.remove(0);
            if values.is_empty() {
                buckets.remove(&fattest.0);
            }
        }

        self.fallback = if self.recent_is_ready() {
            median(&mut self.recent.iter().map(|row| row.ratio).collect::<Vec<_>>())
        } else {
            1.0
        };
        self.factors.clear();
        if self.primary.len() < self.min_observations {
            return;
        }
        for (key, mut values) in buckets {
            let support = values.len() as f64;
            let estimate = median(&mut values);
            let weight = support / (support + SHRINKAGE_OBSERVATIONS);
            let factor = (weight * estimate.ln() + (1.0 - weight) * self.fallback.ln()).exp();
            // A convex combination in log-space lies between its endpoints.
            // Enforce this property at representable extremes, where exp can
            // otherwise round to zero or infinity when bounds are disabled.
            let factor = factor.clamp(estimate.min(self.fallback), estimate.max(self.fallback));
            self.factors.insert(key, self.project(factor));
        }
    }
}

impl DynamicCorrection {
    /// Validate a decoded store against one built from the same configuration.
    /// Rows outside the stored bounds would otherwise panic during the next
    /// rebuild; cached factors must equal a recomputation from the rows.
    pub(crate) fn validate_restored(&self, fresh: &Self) -> Result<(), String> {
        if self.ndim != fresh.ndim
            || self.bins != fresh.bins
            || self.primary_capacity != fresh.primary_capacity
            || self.recent_capacity != fresh.recent_capacity
            || self.max_age != fresh.max_age
            || self.min_observations != fresh.min_observations
            || !same_optional_bits(self.lower_factor, fresh.lower_factor)
            || !same_optional_bits(self.upper_factor, fresh.upper_factor)
        {
            return Err("dynamic correction controls differ from configuration".into());
        }
        if self.log_bounds.is_some() != (self.next_admission > 0) {
            return Err("dynamic correction bounds disagree with its admission clock".into());
        }
        if let Some(bounds) = self.log_bounds {
            for (axis, [lo, hi]) in bounds.iter().copied().enumerate() {
                let valid = if axis < self.ndim {
                    lo.is_finite() && hi.is_finite() && lo >= 0.0 && lo <= hi
                } else {
                    lo == 0.0 && hi == 0.0
                };
                if !valid {
                    return Err("dynamic correction bounds are invalid".into());
                }
            }
        }
        if self.primary.len() > self.primary_capacity || self.recent.len() > self.recent_capacity {
            return Err("dynamic correction exceeds its retention budget".into());
        }
        let oldest = self.next_admission.saturating_sub(self.max_age);
        let row_valid = |row: &Observation| {
            let Some(bounds) = self.log_bounds else {
                return false;
            };
            row.admission >= oldest
                && row.admission < self.next_admission
                && valid_bounded_ratio(row.ratio, self.lower_factor, self.upper_factor)
                && (0..2).all(|axis| {
                    let value = row.log_x[axis];
                    if axis < self.ndim {
                        value.is_finite()
                            && value >= 0.0
                            && value >= bounds[axis][0]
                            && value <= bounds[axis][1]
                    } else {
                        value == 0.0
                    }
                })
        };
        if !self.primary.iter().chain(&self.recent).all(row_valid) {
            return Err("dynamic correction row is outside its bounds or admission window".into());
        }
        if self
            .primary
            .windows(2)
            .any(|pair| pair[0].admission >= pair[1].admission)
        {
            return Err("dynamic correction primary rows are not in admission order".into());
        }
        // Every admission enters the recent FIFO, so it is a contiguous run
        // ending at the newest admission.
        let first_recent = self.next_admission.checked_sub(self.recent.len());
        if self.recent_capacity > 0 && self.recent.is_empty() != (self.next_admission == 0) {
            return Err("dynamic correction recent rows disagree with its admission clock".into());
        }
        if self
            .recent
            .iter()
            .enumerate()
            .any(|(offset, row)| first_recent.map(|first| first + offset) != Some(row.admission))
        {
            return Err("dynamic correction recent rows are not contiguous".into());
        }
        if let Some(first) = first_recent {
            for row in &self.primary {
                if row.admission >= first && self.recent[row.admission - first] != *row {
                    return Err("dynamic correction duplicate rows differ".into());
                }
            }
        }
        let mut recomputed = self.clone();
        recomputed.rebuild_factors();
        if recomputed != *self {
            return Err("dynamic correction factors differ from their retained rows".into());
        }
        Ok(())
    }
}

fn median(values: &mut [f64]) -> f64 {
    debug_assert!(!values.is_empty());
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        let sum = values[middle - 1] + values[middle];
        if sum.is_finite() {
            sum / 2.0
        } else {
            values[middle - 1] / 2.0 + values[middle] / 2.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perfmodel::fpm::options::validate_options;

    #[test]
    fn bounds_expand_and_rebucket_only_on_feedback() {
        for ndim in [1, 2] {
            let options = ForwardPassPerfOptions {
                min_observations: 1,
                ..Default::default()
            };
            let mut correction = DynamicCorrection::new(&options, ndim);
            let original = [100.0, 1000.0][..ndim].to_vec();
            correction.add_observation(original.clone(), 20.0, 10.0);
            let original_log = correction.log_coordinates(&original).unwrap();
            assert_eq!(correction.bucket_key(original_log), Some([0, 0]));

            // New evidence covers zero and coordinates beyond the fixed limits.
            correction.add_observation(vec![0.0; ndim], 20.0, 10.0);
            let upper = [100_000.0_f64, 3_000_000.0][..ndim].to_vec();
            correction.add_observation(upper.clone(), 20.0, 10.0);
            for (axis, hi) in upper.iter().enumerate() {
                assert_eq!(correction.log_bounds.unwrap()[axis], [0.0, hi.ln_1p()]);
            }
            assert_ne!(correction.bucket_key(original_log), Some([0, 0]));
            assert!(
                correction
                    .factors
                    .contains_key(&correction.bucket_key(original_log).unwrap())
            );
            assert_eq!(correction.observation_count(), 3);

            let before = correction.clone();
            for query in [original, upper, vec![10_000_000.0; ndim]] {
                // Every observation has ratio 2, so local and global answers are 2.
                assert_eq!(correction.correction_factor_for(&query), 2.0);
            }
            assert_eq!(
                correction, before,
                "queries must not grow bounds or age rows"
            );
        }
    }

    #[test]
    fn local_shrinkage_and_recent_fallback_for_empty_and_outside_queries() {
        let mut correction = DynamicCorrection::new(&ForwardPassPerfOptions::default(), 1);
        for _ in 0..20 {
            correction.add_observation(vec![0.0], 20.0, 10.0);
        }
        for _ in 0..16 {
            correction.add_observation(vec![100.0], 5.0, 10.0);
        }
        // Twenty local ratios of 2 have equal weight to the recent median 0.5:
        // their geometric mean is exactly 1. No capacity eviction has occurred.
        assert_eq!(correction.observation_count(), 36);
        assert!((correction.correction_factor_for(&[0.0]) - 1.0).abs() < 1e-12);
        assert_eq!(correction.correction_factor_for(&[100.0]), 0.5);
        let empty = correction.bucket_key([10.0_f64.ln_1p(), 0.0]).unwrap();
        assert!(!correction.factors.contains_key(&empty));
        assert_eq!(correction.correction_factor_for(&[10.0]), 0.5);
        assert_eq!(correction.correction_factor_for(&[1000.0]), 0.5);
    }

    #[test]
    fn recent_fifo_can_become_ready_for_small_valid_budgets() {
        for (capacity, minimum) in [(1, 1), (10, 5), (10, 6), (10, 10), (64, 64)] {
            let options = ForwardPassPerfOptions {
                max_observations: capacity,
                min_observations: minimum,
                ..Default::default()
            };
            validate_options(&options).unwrap();
            let mut correction = DynamicCorrection::new(&options, 1);
            for _ in 0..minimum - 1 {
                correction.add_observation(vec![10.0], 15.0, 10.0);
            }
            assert!(!correction.is_ready());
            assert_eq!(correction.correction_factor_for(&[100.0]), 1.0);
            correction.add_observation(vec![10.0], 15.0, 10.0);
            assert!(correction.is_ready());
            assert_eq!(correction.correction_factor_for(&[100.0]), 1.5);
            assert!(correction.primary.len() + correction.recent.len() <= capacity);
            if capacity - minimum < minimum {
                assert_eq!(correction.ready_bucket_count(), 0);
                assert_eq!(correction.correction_factors(), vec![1.5]);
            }
        }
    }

    #[test]
    fn age_expiry_removes_sparse_old_sample_without_contracting_bounds() {
        let options = ForwardPassPerfOptions {
            max_observations: 8,
            min_observations: 1,
            ..Default::default()
        };
        let mut correction = DynamicCorrection::new(&options, 1);
        correction.add_observation(vec![0.0], 20.0, 10.0);
        // The singleton at zero survives capacity eviction from the full bucket.
        for _ in 0..15 {
            correction.add_observation(vec![100.0], 5.0, 10.0);
        }
        assert_eq!(correction.primary[0].admission, 0);
        assert!(correction.correction_factor_for(&[0.0]) > 0.5);
        let bounds = correction.log_bounds;

        // With age limit 16, admission 17 expires row 0, independently of capacity.
        correction.add_observation(vec![100.0], 5.0, 10.0);
        assert!(correction.primary.iter().all(|row| row.admission > 0));
        assert_eq!(correction.log_bounds, bounds);
        assert_eq!(correction.correction_factor_for(&[0.0]), 0.5);
    }

    #[test]
    fn eviction_removes_oldest_from_fullest_bucket_with_stable_ties() {
        let options = ForwardPassPerfOptions {
            max_observations: 4,
            min_observations: 1,
            bucket_shape: Some([2, 1]),
            ..Default::default()
        };
        let mut correction = DynamicCorrection::new(&options, 1);
        for x in [0.0, 100.0, 100.0, 100.0] {
            correction.add_observation(vec![x], 10.0, 10.0);
        }
        let ids = |c: &DynamicCorrection| {
            c.primary
                .iter()
                .map(|row| row.admission)
                .collect::<Vec<_>>()
        };
        // Three primary slots: retain the sparse oldest row; evict row 1.
        assert_eq!(ids(&correction), vec![0, 2, 3]);
        correction.add_observation(vec![0.0], 10.0, 10.0);
        // Two rows per bucket now tie; the lower-coordinate bucket loses row 0.
        assert_eq!(ids(&correction), vec![2, 3, 4]);
        assert_eq!(correction.recent.front().unwrap().admission, 4);
    }

    #[test]
    fn invalid_feedback_does_not_mutate_cold_or_ready_state() {
        for ready in [false, true] {
            let mut correction = DynamicCorrection::new(&ForwardPassPerfOptions::default(), 2);
            if ready {
                for _ in 0..5 {
                    correction.add_observation(vec![10.0, 20.0], 15.0, 10.0);
                }
            }
            let before = correction.clone();
            for x in [
                vec![],
                vec![1.0],
                vec![1.0, 2.0, 3.0],
                vec![-1.0, 20.0],
                vec![10.0, f64::NAN],
                vec![f64::INFINITY, 20.0],
            ] {
                correction.add_observation(x.clone(), 15.0, 10.0);
                assert_eq!(correction.correction_factor_for(&x), 1.0);
                assert_eq!(correction, before);
            }
            for (observed, base) in [
                (0.0, 1.0),
                (-1.0, 1.0),
                (f64::NAN, 1.0),
                (f64::INFINITY, 1.0),
                (1.0, 0.0),
                (1.0, -1.0),
                (1.0, f64::NAN),
                (1.0, f64::INFINITY),
                (f64::MAX, f64::MIN_POSITIVE),
                (f64::MIN_POSITIVE, f64::MAX),
            ] {
                correction.add_observation(vec![1e9, 1e9], observed, base);
                assert_eq!(correction, before, "invalid ratio {observed}/{base}");
            }
        }
    }
}
