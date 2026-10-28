// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared bucketed-sample infrastructure for the forward-pass perf model.
//!
//! [`BucketedSamples`] retains a bounded set of `(feature vector, value)`
//! observations partitioned into per-axis buckets, used by both the native
//! correction grid ([`super::correction`]) and the regression fallback
//! ([`super::regression`]). [`AxisRange`] describes a fixed correction axis
//! bound. [`WithOptions`] constructs native workload-kind stores, while
//! [`StoreStats`] provides the count/readiness view shared by native correction
//! and the single role-bound regression store.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use super::options::ForwardPassPerfOptions;
use crate::AicError;

pub(crate) trait WithOptions {
    fn with_options(options: &ForwardPassPerfOptions, axis_ranges: &[AxisRange]) -> Self;
}

pub(crate) trait StoreStats {
    fn observation_count(&self) -> usize;
    fn is_ready(&self) -> bool;
}

/// The accepted sample may immediately evict an observation from a full store.
/// Returning the actual value keeps incremental consumers in sync with the
/// sampler's retention policy, including ties between equally full buckets.
#[derive(Debug, PartialEq)]
pub(crate) enum SampleInsertion<T> {
    Accepted { evicted: Option<T> },
    Rejected,
}

/// State snapshots encode every field in declaration order; a layout change
/// requires a `STATE_LAYOUT_VERSION` bump.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: Deserialize<'de>"))]
pub(crate) struct BucketedSamples<T> {
    /// Ordered by bucket key: [`Self::observations`] (the order every refit
    /// consumes) and the choice among equally full buckets on eviction must
    /// not depend on a per-process hasher seed, or identical inputs fit
    /// different models in different processes. Each bucket keeps its
    /// within-bucket order, which eviction and refits consume.
    #[serde(with = "crate::fpm::snapshot::codec::sorted_map")]
    pub(crate) buckets: BTreeMap<Vec<usize>, VecDeque<(Vec<f64>, T)>>,
    pub(crate) total_observations: usize,
    axis_min: Vec<f64>,
    axis_max: Vec<f64>,
    fixed_bounds: bool,
    buckets_per_axis: Vec<usize>,
    max_observations: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AxisRange {
    min: f64,
    max: f64,
}

impl AxisRange {
    pub(crate) fn from_zero_to(max: u32) -> Self {
        Self {
            min: 0.0,
            max: f64::from(max),
        }
    }
}

impl<T: Clone> BucketedSamples<T> {
    /// Construct the explicitly dimensioned regression grid. Canonical config
    /// validation uses the same checks before model selection; retaining them
    /// here prevents a malformed internal shape from reaching bucket indexing.
    pub(crate) fn new_regression(
        axes_shape: &[usize],
        max_observations: usize,
    ) -> Result<Self, AicError> {
        validate_regression_shape(axes_shape, max_observations)?;
        Ok(Self {
            buckets: BTreeMap::new(),
            total_observations: 0,
            axis_min: vec![f64::INFINITY; axes_shape.len()],
            axis_max: vec![f64::NEG_INFINITY; axes_shape.len()],
            fixed_bounds: false,
            buckets_per_axis: axes_shape.to_vec(),
            max_observations,
        })
    }

    pub(crate) fn new_dynamic(options: &ForwardPassPerfOptions, ndim: usize) -> Self {
        let buckets_per_axis = if let Some(shape) = options.bucket_shape {
            if ndim == 1 {
                vec![shape[0] * shape[1]]
            } else {
                shape.to_vec()
            }
        } else if ndim == 1 {
            vec![options.bucket_count]
        } else {
            vec![integer_sqrt(options.bucket_count); ndim]
        };
        Self {
            buckets: BTreeMap::new(),
            total_observations: 0,
            axis_min: vec![f64::INFINITY; ndim],
            axis_max: vec![f64::NEG_INFINITY; ndim],
            fixed_bounds: false,
            buckets_per_axis,
            max_observations: options.max_observations,
        }
    }

    pub(crate) fn new_fixed(options: &ForwardPassPerfOptions, axis_ranges: &[AxisRange]) -> Self {
        let mut samples = Self::new_dynamic(options, axis_ranges.len());
        samples.axis_min = axis_ranges.iter().map(|range| range.min).collect();
        samples.axis_max = axis_ranges.iter().map(|range| range.max).collect();
        samples.fixed_bounds = true;
        samples
    }

    pub(crate) fn add(&mut self, x: Vec<f64>, y: T) -> bool {
        matches!(
            self.add_with_eviction(x, y),
            SampleInsertion::Accepted { .. }
        )
    }

    pub(crate) fn add_with_eviction(&mut self, x: Vec<f64>, y: T) -> SampleInsertion<T> {
        if x.len() != self.axis_min.len() || !x.iter().all(|value| value.is_finite()) {
            return SampleInsertion::Rejected;
        }

        if self.fixed_bounds {
            if !self.is_in_bounds(&x) {
                return SampleInsertion::Rejected;
            }
        } else {
            let bounds_changed = self.update_axis_bounds(&x);
            if bounds_changed && self.total_observations > 0 {
                self.rebuild_buckets();
            }
        }

        let key = self.bucket_key(&x);
        self.buckets.entry(key).or_default().push_back((x, y));
        self.total_observations += 1;

        let evicted = if self.total_observations > self.max_observations {
            self.retire_from_fattest_bucket()
        } else {
            None
        };
        SampleInsertion::Accepted { evicted }
    }

    pub(crate) fn observations(&self) -> Vec<(Vec<f64>, T)> {
        self.buckets
            .values()
            .flat_map(|bucket| bucket.iter().cloned())
            .collect()
    }

    fn bucket_key(&self, x: &[f64]) -> Vec<usize> {
        x.iter()
            .enumerate()
            .map(|(i, value)| {
                let lo = self.axis_min[i];
                let hi = self.axis_max[i];
                if hi <= lo {
                    0
                } else {
                    let idx =
                        ((*value - lo) / (hi - lo) * self.buckets_per_axis[i] as f64) as isize;
                    idx.clamp(0, self.buckets_per_axis[i] as isize - 1) as usize
                }
            })
            .collect()
    }

    pub(crate) fn bucket_key_if_in_bounds(&self, x: &[f64]) -> Option<Vec<usize>> {
        if self.total_observations == 0 || x.len() != self.axis_min.len() {
            return None;
        }

        // Estimation must not clamp outside configured correction-grid
        // workload ranges into edge regions.
        let mut key = Vec::with_capacity(x.len());
        for (i, value) in x.iter().enumerate() {
            let lo = self.axis_min[i];
            let hi = self.axis_max[i];
            if !value.is_finite() || !lo.is_finite() || !hi.is_finite() {
                return None;
            }
            if hi <= lo {
                if *value != lo {
                    return None;
                }
                key.push(0);
                continue;
            }
            if *value < lo || *value > hi {
                return None;
            }

            let idx = ((*value - lo) / (hi - lo) * self.buckets_per_axis[i] as f64) as isize;
            key.push(idx.clamp(0, self.buckets_per_axis[i] as isize - 1) as usize);
        }
        Some(key)
    }

    pub(crate) fn is_in_bounds(&self, x: &[f64]) -> bool {
        if x.len() != self.axis_min.len() {
            return false;
        }
        x.iter().enumerate().all(|(i, value)| {
            value.is_finite()
                && self.axis_min[i].is_finite()
                && self.axis_max[i].is_finite()
                && *value >= self.axis_min[i]
                && *value <= self.axis_max[i]
        })
    }

    fn update_axis_bounds(&mut self, x: &[f64]) -> bool {
        let mut changed = false;
        for (i, value) in x.iter().enumerate() {
            if *value < self.axis_min[i] {
                self.axis_min[i] = *value;
                changed = true;
            }
            if *value > self.axis_max[i] {
                self.axis_max[i] = *value;
                changed = true;
            }
        }
        changed
    }

    fn rebuild_buckets(&mut self) {
        let observations = self.observations();
        self.buckets.clear();
        for (x, y) in observations {
            let key = self.bucket_key(&x);
            self.buckets.entry(key).or_default().push_back((x, y));
        }
    }

    fn retire_from_fattest_bucket(&mut self) -> Option<T> {
        // Eviction removes samples only. Dynamic regression bounds remain
        // monotonic; fixed correction-grid workload ranges are configured at
        // model creation. Among equally full buckets `max_by_key` keeps the
        // last, i.e. the greatest bucket key.
        let key = self
            .buckets
            .iter()
            .max_by_key(|(_, bucket)| bucket.len())
            .map(|(key, _)| key.clone())?;

        let mut evicted = None;
        if let Some(bucket) = self.buckets.get_mut(&key) {
            if let Some((_, value)) = bucket.pop_front() {
                evicted = Some(value);
                self.total_observations -= 1;
            }
            if bucket.is_empty() {
                self.buckets.remove(&key);
            }
        }
        evicted
    }
}

impl<T: Clone> BucketedSamples<T> {
    /// Check a decoded snapshot against a store freshly configured from the
    /// same canonical configuration. Every condition that later indexing,
    /// eviction or refits assume is verified here instead of panicking.
    pub(crate) fn validate_restored(
        &self,
        fresh: &Self,
        mut sample: impl FnMut(&[f64], &T) -> Result<(), String>,
    ) -> Result<(), String> {
        if self.buckets_per_axis != fresh.buckets_per_axis
            || self.max_observations != fresh.max_observations
            || self.fixed_bounds != fresh.fixed_bounds
        {
            return Err("sample grid differs from the configured grid".into());
        }
        let dims = self.buckets_per_axis.len();
        if self.axis_min.len() != dims || self.axis_max.len() != dims {
            return Err("sample bounds do not match the grid dimension".into());
        }
        let same_bits = |a: &[f64], b: &[f64]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        if self.fixed_bounds || self.total_observations == 0 {
            // Fixed grids keep configured limits; an empty dynamic grid has
            // never admitted a sample and retains its infinite sentinels.
            if !same_bits(&self.axis_min, &fresh.axis_min)
                || !same_bits(&self.axis_max, &fresh.axis_max)
            {
                return Err("sample bounds differ from the configured or empty bounds".into());
            }
        } else if self
            .axis_min
            .iter()
            .zip(&self.axis_max)
            .any(|(lo, hi)| !lo.is_finite() || !hi.is_finite() || lo > hi)
        {
            return Err("dynamic sample bounds must be finite and ordered".into());
        }
        let mut total = 0usize;
        for (key, bucket) in &self.buckets {
            if bucket.is_empty() {
                return Err("empty sample bucket".into());
            }
            if key.len() != dims
                || key
                    .iter()
                    .zip(&self.buckets_per_axis)
                    .any(|(index, bins)| index >= bins)
            {
                return Err("sample bucket key is outside the grid".into());
            }
            for (x, value) in bucket {
                if x.len() != dims || x.iter().any(|v| !v.is_finite()) {
                    return Err("retained sample coordinates are invalid".into());
                }
                if x.iter()
                    .enumerate()
                    .any(|(axis, v)| *v < self.axis_min[axis] || *v > self.axis_max[axis])
                {
                    return Err("retained sample lies outside the sample bounds".into());
                }
                if self.bucket_key(x) != *key {
                    return Err("retained sample is stored under the wrong bucket key".into());
                }
                sample(x, value)?;
            }
            total = total
                .checked_add(bucket.len())
                .ok_or("retained sample count overflows")?;
        }
        if total != self.total_observations || total > self.max_observations {
            return Err("retained sample count does not match the store total or budget".into());
        }
        Ok(())
    }
}

pub(super) fn validate_regression_shape(
    axes_shape: &[usize],
    max_observations: usize,
) -> Result<(), AicError> {
    let invalid = |field: &str, reason: &str| {
        AicError::InvalidEngineConfig(format!(
            "estimator_config.fpm_regression.sampling.{field} {reason}"
        ))
    };
    if !(1..=6).contains(&axes_shape.len()) {
        return Err(invalid("bins_per_axis", "must contain 1 to 6 bin counts"));
    }
    let product = axes_shape.iter().try_fold(1_usize, |count, &bins| {
        if bins == 0 {
            None
        } else {
            count.checked_mul(bins)
        }
    });
    if product.is_none_or(|count| count > isize::MAX as usize) {
        return Err(invalid(
            "bins_per_axis",
            "must be positive and have a representable product",
        ));
    }
    if max_observations == 0 {
        return Err(invalid("max_observations", "must be positive"));
    }
    Ok(())
}

pub(crate) fn median_ratio(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut values = values
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[mid - 1] + values[mid]) / 2.0)
    } else {
        Some(values[mid])
    }
}

pub(crate) fn integer_sqrt(value: usize) -> usize {
    (value as f64).sqrt() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_regression_dimensions_rebucket_and_return_actual_eviction() {
        let mut samples = BucketedSamples::new_regression(&[2, 3, 4], 4).unwrap();
        for (index, x) in [
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [7.0, 7.0, 8.0],
            [8.0, 8.0, 9.0],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                samples.add_with_eviction(x.to_vec(), index),
                SampleInsertion::Accepted { evicted: None }
            );
        }
        // The [1,2,3] bucket has three samples; retire its oldest (index 1),
        // retaining the global oldest in the less populated [0,0,0] bucket.
        assert_eq!(samples.buckets[&vec![1, 2, 3]].len(), 3);
        assert_eq!(
            samples.add_with_eviction(vec![2.0, 0.0, 0.0], 4),
            SampleInsertion::Accepted { evicted: Some(1) }
        );
        assert_eq!(samples.total_observations, 4);
        assert_eq!(samples.axis_max, [10.0, 10.0, 10.0]);
        let before = samples.clone();
        for invalid in [vec![0.0, 0.0], vec![0.0, 0.0, f64::NAN]] {
            assert_eq!(
                samples.add_with_eviction(invalid, 5),
                SampleInsertion::Rejected
            );
            assert_eq!(samples.buckets, before.buckets);
            assert_eq!(samples.axis_max, before.axis_max);
        }
        // Bounds expansion rebuckets retained observations without inventing
        // or losing rows before enforcing capacity.
        assert!(samples.add(vec![20.0, 20.0, 20.0], 5));
        assert_eq!(samples.total_observations, 4);
        assert_eq!(samples.buckets[&vec![1, 2, 3]], vec![(vec![20.0; 3], 5)]);

        let mut one_axis = BucketedSamples::new_regression(&[3], 2).unwrap();
        assert!(one_axis.add(vec![0.0], 0));
        assert!(one_axis.add(vec![9.0], 1));
        assert_eq!(one_axis.buckets[&vec![2]], vec![(vec![9.0], 1)]);
        for shape in [
            vec![],
            vec![0],
            vec![usize::MAX],
            vec![usize::MAX, 2],
            vec![1; 7],
        ] {
            assert!(BucketedSamples::<usize>::new_regression(&shape, 4).is_err());
        }
        assert!(BucketedSamples::<usize>::new_regression(&[1], 0).is_err());
    }

    #[test]
    fn insertion_returns_the_actual_fattest_bucket_eviction() {
        let options = ForwardPassPerfOptions {
            max_observations: 4,
            bucket_shape: Some([2, 3]),
            ..ForwardPassPerfOptions::default()
        };
        let ranges = [AxisRange::from_zero_to(10); 2];
        let mut samples = BucketedSamples::new_fixed(&options, &ranges);
        for (index, x) in [[0.0, 0.0], [10.0, 10.0], [7.0, 7.0], [8.0, 8.0]]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                samples.add_with_eviction(x.to_vec(), index),
                SampleInsertion::Accepted { evicted: None }
            );
        }
        // The upper bucket is uniquely fattest (three rows). Its oldest row
        // is [10,10], whereas the global oldest row [0,0] must be retained.
        assert_eq!(
            samples.add_with_eviction(vec![2.0, 0.0], 4),
            SampleInsertion::Accepted { evicted: Some(1) }
        );
        let mut retained = samples
            .observations()
            .into_iter()
            .map(|(_, value)| value)
            .collect::<Vec<_>>();
        retained.sort_unstable();
        assert_eq!(retained, [0, 2, 3, 4]);
        assert_eq!(samples.total_observations, 4);
        assert_eq!(samples.axis_max, [10.0, 10.0]);
    }

    #[test]
    fn retention_order_and_eviction_ties_do_not_depend_on_the_store_instance() {
        // A 4 x 4 grid filled round-robin keeps every bucket equally full, so
        // each eviction breaks a tie. Every store built here would draw its
        // own hasher seed if the buckets were hashed.
        let run = || {
            let mut samples = BucketedSamples::new_regression(&[4, 4], 8).unwrap();
            let mut evicted = Vec::new();
            for index in 0..64_usize {
                let x = vec![(index % 4) as f64, ((index / 4) % 4) as f64];
                match samples.add_with_eviction(x, index) {
                    SampleInsertion::Accepted {
                        evicted: Some(value),
                    } => evicted.push(value),
                    SampleInsertion::Accepted { evicted: None } => {}
                    SampleInsertion::Rejected => panic!("finite sample rejected"),
                }
            }
            let retained = samples
                .observations()
                .into_iter()
                .map(|(_, value)| value)
                .collect::<Vec<_>>();
            (retained, evicted)
        };
        let first = run();
        assert_eq!(first.1.len(), 56);
        // The ninth sample stretches the second axis to [0, 2], leaving nine
        // singleton cells; the greatest key among them is [3, 2], which holds
        // sample 7 = (3, 1).
        assert_eq!(first.1[0], 7);
        for _ in 0..16 {
            assert_eq!(run(), first);
        }
    }

    #[test]
    fn rejected_insertions_do_not_change_samples_or_bounds() {
        let options = ForwardPassPerfOptions {
            max_observations: 2,
            ..ForwardPassPerfOptions::default()
        };
        let mut samples = BucketedSamples::new_fixed(&options, &[AxisRange::from_zero_to(10); 2]);
        assert!(samples.add(vec![2.0, 3.0], 1));
        let before = samples.clone();
        for x in [
            vec![4.0],
            vec![f64::NAN, 3.0],
            vec![2.0, f64::INFINITY],
            vec![11.0, 3.0],
        ] {
            assert_eq!(samples.add_with_eviction(x, 2), SampleInsertion::Rejected);
            assert_eq!(samples.buckets, before.buckets);
            assert_eq!(samples.axis_min, before.axis_min);
            assert_eq!(samples.axis_max, before.axis_max);
            assert_eq!(samples.total_observations, before.total_observations);
        }
        assert_eq!(
            samples.add_with_eviction(vec![3.0, 4.0], 2),
            SampleInsertion::Accepted { evicted: None }
        );
    }

    #[test]
    fn repeated_fifo_eviction_preserves_order_through_rebucketing() {
        let mut samples = BucketedSamples::new_regression(&[1], 3).unwrap();
        for value in 0..6 {
            assert_eq!(
                samples.add_with_eviction(vec![0.0], value),
                SampleInsertion::Accepted {
                    evicted: (value >= 3).then(|| value - 3),
                }
            );
        }
        assert_eq!(
            samples.observations(),
            vec![(vec![0.0], 3), (vec![0.0], 4), (vec![0.0], 5)]
        );
        // Expand the bounds after repeated head removals. Rebuilding must
        // retain FIFO order, including entries in a wrapped queue.
        assert_eq!(
            samples.add_with_eviction(vec![10.0], 6),
            SampleInsertion::Accepted { evicted: Some(3) }
        );
        assert_eq!(
            samples.observations(),
            vec![(vec![0.0], 4), (vec![0.0], 5), (vec![10.0], 6)]
        );
        assert_eq!(samples.total_observations, 3);
    }

    #[test]
    fn dynamic_rebucketing_preserves_rows_and_rectangular_geometry() {
        let options = ForwardPassPerfOptions {
            max_observations: 8,
            bucket_shape: Some([2, 3]),
            ..ForwardPassPerfOptions::default()
        };
        let mut samples = BucketedSamples::new_dynamic(&options, 2);
        for (index, x) in [[0.0, 0.0], [1.0, 1.0], [15.0, 15.0]]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                samples.add_with_eviction(x.to_vec(), index),
                SampleInsertion::Accepted { evicted: None }
            );
        }
        assert_eq!(samples.total_observations, 3);
        assert_eq!(samples.buckets[&vec![0, 0]].len(), 2);
        assert_eq!(samples.buckets[&vec![1, 2]], vec![(vec![15.0, 15.0], 2)]);
        assert!(samples.buckets[&vec![0, 0]].contains(&(vec![1.0, 1.0], 1)));
    }

    #[test]
    fn boolean_add_preserves_retention_including_bucket_ties() {
        let options = ForwardPassPerfOptions {
            max_observations: 16,
            bucket_shape: Some([2, 3]),
            ..ForwardPassPerfOptions::default()
        };
        let mut with_events = BucketedSamples::new_dynamic(&options, 2);
        let mut boolean_only = BucketedSamples::new_dynamic(&options, 2);
        for i in 0..100 {
            let x = vec![(i % 17) as f64, ((i * 7) % 23) as f64];
            assert!(boolean_only.add(x.clone(), i));
            let SampleInsertion::Accepted { evicted } = with_events.add_with_eviction(x, i) else {
                panic!("valid sample rejected");
            };
            assert_eq!(evicted.is_some(), i >= 16);
            assert_eq!(with_events.buckets, boolean_only.buckets);
            assert_eq!(
                with_events.total_observations,
                boolean_only.total_observations
            );
        }
    }
}
