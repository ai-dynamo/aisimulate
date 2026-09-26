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

use std::collections::HashMap;

use super::options::ForwardPassPerfOptions;

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

#[derive(Clone, Debug)]
pub(crate) struct BucketedSamples<T> {
    pub(crate) buckets: HashMap<Vec<usize>, Vec<(Vec<f64>, T)>>,
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
            buckets: HashMap::new(),
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
        self.buckets.entry(key).or_default().push((x, y));
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

    fn is_in_bounds(&self, x: &[f64]) -> bool {
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
            self.buckets.entry(key).or_default().push((x, y));
        }
    }

    fn retire_from_fattest_bucket(&mut self) -> Option<T> {
        // Eviction removes samples only. Dynamic regression bounds remain
        // monotonic; fixed correction-grid workload ranges are configured at
        // model creation.
        let key = self
            .buckets
            .iter()
            .max_by_key(|(_, bucket)| bucket.len())
            .map(|(key, _)| key.clone())?;

        let mut evicted = None;
        if let Some(bucket) = self.buckets.get_mut(&key) {
            if !bucket.is_empty() {
                evicted = Some(bucket.remove(0).1);
                self.total_observations -= 1;
            }
            if bucket.is_empty() {
                self.buckets.remove(&key);
            }
        }
        evicted
    }
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
        // Clone the initial hasher so tied bucket choices are comparable.
        let mut boolean_only = with_events.clone();
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
