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

use super::options::ForwardPassPerfOptions;
use super::samples::integer_sqrt;

const SHRINKAGE_OBSERVATIONS: f64 = 20.0;
type BucketKey = [usize; 2];

#[derive(Clone, Copy, Debug, PartialEq)]
struct Observation {
    admission: usize,
    log_x: [f64; 2],
    ratio: f64,
}

#[derive(Clone, Debug, PartialEq)]
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
