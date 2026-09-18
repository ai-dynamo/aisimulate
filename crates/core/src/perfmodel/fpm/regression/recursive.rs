// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Centered sufficient statistics for the existing constrained regression fit.
//!
//! Insertions and evictions update the mean and scatter of `(x0, x1, y)`.
//! Standardization and the small nonnegative slope search are then recomputed
//! from those statistics. Numerically ambiguous cases use the original batch
//! fitter on the exact retained observations.

use super::{
    FEATURE_DIMENSION, INACTIVE_SCALE_RELATIVE_TOLERANCE, LinearFit, RegressionObservation,
    Standardization, fit_regression_with_ridge, solve_linear_system,
    solve_regularized_linear_system,
};
const SCORE_RELATIVE_TOLERANCE: f64 = 1e-10;
const MIN_COVARIANCE_DETERMINANT_RATIO: f64 = 1e-8;

#[derive(Clone, Debug)]
pub(super) struct RecursiveFit {
    count: usize,
    means: [f64; 3],
    scatter: [[f64; 3]; 3],
    mutations_since_rebuild: usize,
    damaged_downdate: bool,
    ridge_scale: f64,
    rebuild_interval: Option<usize>,
}

struct Candidate {
    fit: LinearFit,
    squared_error: f64,
    error_tolerance: f64,
}

impl RecursiveFit {
    pub(super) fn new(ridge_scale: f64, rebuild_interval: Option<usize>) -> Self {
        Self {
            count: 0,
            means: [0.0; 3],
            scatter: [[0.0; 3]; 3],
            mutations_since_rebuild: 0,
            damaged_downdate: false,
            ridge_scale,
            rebuild_interval,
        }
    }

    pub(super) fn observation_count(&self) -> usize {
        self.count
    }

    #[cfg(test)]
    pub(super) fn mutations_since_rebuild(&self) -> usize {
        self.mutations_since_rebuild
    }

    // The caller supplies validated samples and accounts for each eviction.
    pub(super) fn add(&mut self, observation: RegressionObservation) {
        self.insert_statistics(observation);
        self.mutations_since_rebuild = self.mutations_since_rebuild.saturating_add(1);
    }

    fn insert_statistics(&mut self, observation: RegressionObservation) {
        let z = [
            observation.raw_x[0],
            observation.raw_x[1],
            observation.observed_ms,
        ];
        if self.count == 0 {
            self.count = 1;
            self.means = z;
            self.scatter = [[0.0; 3]; 3];
            return;
        }
        let delta = std::array::from_fn::<_, 3, _>(|i| z[i] - self.means[i]);
        self.count += 1;
        for (mean, difference) in self.means.iter_mut().zip(delta) {
            *mean += difference / self.count as f64;
        }
        let remaining = std::array::from_fn::<_, 3, _>(|i| z[i] - self.means[i]);
        for i in 0..3 {
            for j in i..3 {
                // The diagonal matches production's Welford operation. The
                // symmetric cross term avoids privileging either coordinate.
                let increment = if i == j {
                    delta[i] * remaining[i]
                } else {
                    0.5 * delta[i] * remaining[j] + 0.5 * delta[j] * remaining[i]
                };
                self.scatter[i][j] += increment;
                self.scatter[j][i] = self.scatter[i][j];
            }
        }
    }

    pub(super) fn remove(&mut self, observation: RegressionObservation) {
        self.mutations_since_rebuild = self.mutations_since_rebuild.saturating_add(1);
        if self.count == 0 {
            self.damaged_downdate = true;
            return;
        }
        if self.count == 1 {
            self.count = 0;
            self.means = [0.0; 3];
            self.scatter = [[0.0; 3]; 3];
            return;
        }
        let z = [
            observation.raw_x[0],
            observation.raw_x[1],
            observation.observed_ms,
        ];
        let delta = std::array::from_fn::<_, 3, _>(|i| z[i] - self.means[i]);
        self.count -= 1;
        for (mean, difference) in self.means.iter_mut().zip(delta) {
            *mean -= difference / self.count as f64;
        }
        let remaining = std::array::from_fn::<_, 3, _>(|i| z[i] - self.means[i]);
        for i in 0..3 {
            for j in i..3 {
                let previous = self.scatter[i][j];
                let decrement = if i == j {
                    delta[i] * remaining[i]
                } else {
                    0.5 * delta[i] * remaining[j] + 0.5 * delta[j] * remaining[i]
                };
                self.scatter[i][j] -= decrement;
                self.scatter[j][i] = self.scatter[i][j];
                // Losing almost all variance can expose previously hidden
                // cancellation, including a constant axis becoming active.
                if i == j
                    && (self.scatter[i][i] < 0.0
                        || (previous > 0.0 && self.scatter[i][i] < previous * 1e-8))
                {
                    self.damaged_downdate = true;
                }
            }
        }
    }

    fn rebuild(&mut self, retained: &[RegressionObservation]) {
        self.count = 0;
        self.means = [0.0; 3];
        self.scatter = [[0.0; 3]; 3];
        for &observation in retained {
            self.insert_statistics(observation);
        }
        self.mutations_since_rebuild = 0;
        self.damaged_downdate = false;
    }

    fn statistics_are_finite(&self) -> bool {
        self.means.iter().all(|value| value.is_finite())
            && self.scatter.iter().flatten().all(|value| value.is_finite())
            && (0..3).all(|i| self.scatter[i][i] >= 0.0)
    }

    fn fallback(
        &self,
        retained: &mut impl FnMut() -> Vec<RegressionObservation>,
        min_observations: usize,
    ) -> Option<LinearFit> {
        // A batch fallback does not reset the statistics or rebuild clock.
        fit_regression_with_ridge(&retained(), min_observations, self.ridge_scale)
    }

    #[cfg(test)]
    pub(super) fn fit(
        &mut self,
        retained: &[RegressionObservation],
        min_observations: usize,
    ) -> Option<LinearFit> {
        if self.count != retained.len() {
            self.rebuild(retained);
        }
        self.fit_lazy(min_observations, || retained.to_vec())
    }

    /// Fit after the complete insertion/eviction transaction. One insertion
    /// and one eviction count as two mutations; rebucketing counts as none.
    /// The interval is checked once here, so an odd threshold can be exceeded
    /// by one mutation before a rebuild resets the clock. Disabling periodic
    /// rebuilds leaves numerical recovery and batch fallbacks enabled.
    ///
    /// The healthy path does not materialize or inspect retained samples.
    /// The caller must account for every insertion and actual eviction.
    pub(super) fn fit_lazy(
        &mut self,
        min_observations: usize,
        mut retained: impl FnMut() -> Vec<RegressionObservation>,
    ) -> Option<LinearFit> {
        let periodic = self
            .rebuild_interval
            .is_some_and(|interval| self.mutations_since_rebuild >= interval);
        if self.damaged_downdate || periodic || !self.statistics_are_finite() {
            self.rebuild(&retained());
        }
        if self.count < min_observations || self.count == 0 {
            return None;
        }
        if !self.statistics_are_finite() {
            return self.fallback(&mut retained, min_observations);
        }

        let mut standardization = Standardization {
            means: [self.means[0], self.means[1]],
            scales: [0.0; FEATURE_DIMENSION],
            active: [false; FEATURE_DIMENSION],
        };
        let mut varying_axes = Vec::with_capacity(FEATURE_DIMENSION);
        for axis in 0..FEATURE_DIMENSION {
            let scale = (self.scatter[axis][axis] / self.count as f64)
                .max(0.0)
                .sqrt();
            let threshold = INACTIVE_SCALE_RELATIVE_TOLERANCE * self.means[axis].abs().max(1.0);
            standardization.scales[axis] = scale;
            standardization.active[axis] = scale.is_finite() && scale > threshold;
            // Very small nonzero spread is sensitive to mean rounding and
            // updates can disagree with batch fitting about active axes.
            if scale > 0.0 && scale <= threshold * 1e4 {
                return self.fallback(&mut retained, min_observations);
            }
            if standardization.active[axis] {
                varying_axes.push(axis);
            }
        }
        if varying_axes.is_empty() {
            return None;
        }

        if varying_axes.len() == 2 {
            let correlation = (self.scatter[0][1] / standardization.scales[0])
                / standardization.scales[1]
                / self.count as f64;
            if !correlation.is_finite()
                || 1.0 - correlation * correlation <= MIN_COVARIANCE_DETERMINANT_RATIO
            {
                // Production can pick a different face under near-perfect
                // collinearity; matching only in-sample predictions is unsafe.
                return self.fallback(&mut retained, min_observations);
            }
        }

        let target_scale = (self.scatter[2][2] / self.count as f64).sqrt();
        let slope_tolerance = target_scale * 1e-10 + self.means[2].abs() * f64::EPSILON * 32.0;
        let mut candidates = Vec::with_capacity(1 << varying_axes.len());
        for mask in 0..(1usize << varying_axes.len()) {
            let fitted_axes = varying_axes
                .iter()
                .enumerate()
                .filter_map(|(i, &axis)| ((mask & (1 << i)) != 0).then_some(axis))
                .collect::<Vec<_>>();
            let size = fitted_axes.len() + 1;
            let mut lhs = vec![vec![0.0; size]; size];
            let mut rhs = vec![0.0; size];
            lhs[0][0] = self.count as f64;
            rhs[0] = self.count as f64 * self.means[2];
            for (i, &axis_i) in fitted_axes.iter().enumerate() {
                rhs[i + 1] = self.scatter[axis_i][2] / standardization.scales[axis_i];
                for (j, &axis_j) in fitted_axes.iter().enumerate() {
                    lhs[i + 1][j + 1] = (self.scatter[axis_i][axis_j]
                        / standardization.scales[axis_i])
                        / standardization.scales[axis_j];
                }
            }
            let solution = solve_linear_system(lhs.clone(), rhs.clone()).or_else(|| {
                solve_regularized_linear_system(lhs.clone(), rhs.clone(), self.ridge_scale)
            });
            let Some(solution) = solution else {
                return self.fallback(&mut retained, min_observations);
            };
            if !solution.iter().all(|value| value.is_finite()) {
                return self.fallback(&mut retained, min_observations);
            }
            if solution[1..]
                .iter()
                .any(|value| value.abs() <= slope_tolerance)
            {
                return self.fallback(&mut retained, min_observations);
            }
            if solution[1..].iter().any(|value| *value < 0.0) {
                continue;
            }

            let mut coefficients = [0.0; FEATURE_DIMENSION];
            let mut linear_term = 0.0;
            let mut quadratic_term = 0.0;
            let mut magnitude = self.scatter[2][2].abs();
            for (i, &axis) in fitted_axes.iter().enumerate() {
                coefficients[axis] = solution[i + 1];
                let term = 2.0 * solution[i + 1] * rhs[i + 1];
                linear_term += term;
                magnitude += term.abs();
                for j in 0..fitted_axes.len() {
                    let term = solution[i + 1] * lhs[i + 1][j + 1] * solution[j + 1];
                    quadratic_term += term;
                    magnitude += term.abs();
                }
            }
            let intercept_error = solution[0] - self.means[2];
            // Score every face by unpenalized SSE, including ridge retries:
            // Cyy - 2 b' D^-1 Cxy + b' D^-1 Cxx D^-1 b + n(a - mean_y)^2.
            // Near ties require batch residuals because these terms can cancel.
            let squared_error = self.scatter[2][2] - linear_term
                + quadratic_term
                + self.count as f64 * intercept_error * intercept_error;
            let error_tolerance = SCORE_RELATIVE_TOLERANCE * magnitude.max(1e-24);
            if !squared_error.is_finite() || squared_error < -error_tolerance {
                return self.fallback(&mut retained, min_observations);
            }
            candidates.push(Candidate {
                fit: LinearFit {
                    intercept: solution[0],
                    coefficients,
                    standardization,
                },
                squared_error,
                error_tolerance,
            });
        }

        let mut best_index = 0;
        for i in 1..candidates.len() {
            if candidates[i].squared_error < candidates[best_index].squared_error {
                best_index = i;
            }
        }
        let best = candidates.get(best_index)?;
        if candidates.iter().enumerate().any(|(index, other)| {
            index != best_index
                && (other.squared_error - best.squared_error).abs()
                    <= other.error_tolerance + best.error_tolerance
        }) {
            return self.fallback(&mut retained, min_observations);
        }
        let best = candidates.swap_remove(best_index).fit;
        let is_underdetermined = self.count <= varying_axes.len();
        let has_load_signal = best.coefficients.iter().any(|value| *value > 0.0);
        (is_underdetermined || has_load_signal).then_some(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RegressionFitConfig;
    use crate::fpm::options::ForwardPassPerfOptions;
    use crate::fpm::regression::BucketedRegression;

    fn observation(i: usize) -> RegressionObservation {
        let x = [(i % 43) as f64 + 1.0, ((i * 29) % 67) as f64 + 1.0];
        RegressionObservation {
            raw_x: x,
            observed_ms: 4.0 + 1.7 * x[0] + 0.4 * x[1] + (i % 7) as f64 * 0.01,
        }
    }

    fn default_recursive() -> RecursiveFit {
        let config = RegressionFitConfig::default();
        RecursiveFit::new(config.singular_ridge_scale, config.rebuild_interval)
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-8 + 1e-8 * expected.abs(),
            "expected {expected:.17e}, got {actual:.17e}"
        );
    }

    fn compare_fits(
        incremental: Option<&LinearFit>,
        batch: Option<&LinearFit>,
        retained: &[RegressionObservation],
    ) {
        assert_eq!(incremental.is_some(), batch.is_some());
        if let (Some(incremental), Some(batch)) = (incremental, batch) {
            assert_eq!(
                incremental.standardization.active,
                batch.standardization.active
            );
            close(incremental.intercept, batch.intercept);
            for axis in 0..FEATURE_DIMENSION {
                assert_eq!(
                    incremental.coefficients[axis] > 0.0,
                    batch.coefficients[axis] > 0.0
                );
                close(incremental.coefficients[axis], batch.coefficients[axis]);
                close(
                    incremental.standardization.means[axis],
                    batch.standardization.means[axis],
                );
                close(
                    incremental.standardization.scales[axis],
                    batch.standardization.scales[axis],
                );
            }
            for x in retained.iter().map(|sample| sample.raw_x).chain([
                [0.0, 0.0],
                [17.0, 91.0],
                [137.0, 4.0],
                [1000.0, 1000.0],
            ]) {
                close(incremental.predict(&x).unwrap(), batch.predict(&x).unwrap());
            }
        }
    }

    fn compare(recursive: &mut RecursiveFit, retained: &[RegressionObservation], min: usize) {
        let incremental = recursive.fit(retained, min);
        let batch = fit_regression_with_ridge(retained, min, recursive.ridge_scale);
        compare_fits(incremental.as_ref(), batch.as_ref(), retained);
    }

    #[test]
    fn centered_updates_track_batch_across_evictions_and_periodic_rebuilds() {
        for interval in [RegressionFitConfig::default().rebuild_interval, Some(4096)] {
            let mut recursive = RecursiveFit::new(1e-9, interval);
            let mut retained = Vec::new();
            for i in 0..4300 {
                let incoming = observation(i);
                retained.push(incoming);
                recursive.add(incoming);
                if retained.len() > 64 {
                    recursive.remove(retained.remove(0));
                }
                compare(&mut recursive, &retained, 5);
            }
        }
    }

    #[test]
    fn sliding_window_tracks_scale_changes_and_axes_becoming_inactive() {
        for ridge_scale in [0.0, 0.25] {
            let mut recursive = RecursiveFit::new(ridge_scale, None);
            let mut retained = Vec::new();
            for i in 0..128 {
                let raw_x = match i / 32 {
                    0 => [((i % 11) + 1) as f64 * 1e12, 7.0],
                    1 => [
                        ((i % 11) + 1) as f64 * 1e12,
                        (((i * 7) % 19) + 1) as f64 * 0.01,
                    ],
                    2 => [3e12, (((i * 7) % 19) + 1) as f64 * 0.01],
                    _ => [((i % 11) + 1) as f64 * 1000.0, 7.0],
                };
                let sample = RegressionObservation {
                    raw_x,
                    observed_ms: 5.0 + 2.5e-9 * raw_x[0] + 4.0 * raw_x[1],
                };
                retained.push(sample);
                recursive.add(sample);
                if retained.len() > 16 {
                    recursive.remove(retained.remove(0));
                }
                compare(&mut recursive, &retained, 5);
                if i == 31 || i == 95 || i == 127 {
                    let fit = recursive.fit(&retained, 5).unwrap();
                    assert_eq!(
                        fit.standardization.active,
                        if i == 95 {
                            [false, true]
                        } else {
                            [true, false]
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn periodic_schedule_counts_mutations_and_checks_once_per_transaction() {
        let cases = [
            (Some(1), 4, 12, (1..=12).collect::<Vec<_>>()),
            // Four warmup insertions, then two mutations per replacement:
            // threshold five is first crossed at six and reset to zero.
            (Some(5), 4, 12, vec![5, 8, 11]),
            (Some(4096), 64, 4300, vec![2080, 4128]),
            (
                RegressionFitConfig::default().rebuild_interval,
                64,
                4300,
                vec![],
            ),
        ];
        for (interval, capacity, updates, expected) in cases {
            let mut recursive = RecursiveFit::new(0.25, interval);
            let mut retained = Vec::new();
            let mut rebuilt_at = Vec::new();
            for input in 1..=updates {
                let sample = observation(input);
                retained.push(sample);
                recursive.add(sample);
                if retained.len() > capacity {
                    recursive.remove(retained.remove(0));
                }
                // An unreachable readiness threshold isolates rebuild work
                // from numerical candidate fallbacks without test counters.
                let mut reads = 0;
                assert!(
                    recursive
                        .fit_lazy(usize::MAX, || {
                            reads += 1;
                            retained.clone()
                        })
                        .is_none()
                );
                if reads > 0 {
                    assert_eq!(reads, 1);
                    assert_eq!(recursive.mutations_since_rebuild, 0);
                    rebuilt_at.push(input);
                }
            }
            assert_eq!(rebuilt_at, expected, "interval {interval:?}");
            assert_eq!(recursive.rebuild_interval, interval);
            assert_eq!(recursive.ridge_scale, 0.25);
        }
    }

    #[test]
    fn healthy_fit_does_not_materialize_retained_samples() {
        let mut recursive = default_recursive();
        for sample in (0..64).map(observation) {
            recursive.add(sample);
        }
        assert!(
            recursive
                .fit_lazy(5, || panic!("healthy fit must be incremental"))
                .is_some()
        );
    }

    #[test]
    fn collinear_fallback_preserves_configured_ridge_and_rebuild_clock() {
        let mut recursive = RecursiveFit::new(0.25, None);
        let retained = (0..16)
            .map(|i| RegressionObservation {
                raw_x: [i as f64, 2.0 * i as f64],
                observed_ms: 3.0 + 7.0 * i as f64,
            })
            .collect::<Vec<_>>();
        for &sample in &retained {
            recursive.add(sample);
        }
        let mut reads = 0;
        let incremental = recursive.fit_lazy(5, || {
            reads += 1;
            retained.clone()
        });
        let batch = fit_regression_with_ridge(&retained, 5, 0.25);
        compare_fits(incremental.as_ref(), batch.as_ref(), &retained);
        assert_eq!(reads, 1);
        assert_eq!(recursive.mutations_since_rebuild, retained.len());
    }

    #[test]
    fn zero_slope_boundary_uses_batch_readiness() {
        let mut recursive = default_recursive();
        let retained = (0..16)
            .map(|i| RegressionObservation {
                raw_x: [i as f64, 0.0],
                observed_ms: 3.0,
            })
            .collect::<Vec<_>>();
        for &sample in &retained {
            recursive.add(sample);
        }
        let mut reads = 0;
        assert!(
            recursive
                .fit_lazy(5, || {
                    reads += 1;
                    retained.clone()
                })
                .is_none()
        );
        assert_eq!(reads, 1);
        assert!(fit_regression_with_ridge(&retained, 5, recursive.ridge_scale).is_none());
    }

    #[test]
    fn disabled_periodic_rebuilds_still_recover_after_outlier_removal() {
        let mut recursive = RecursiveFit::new(0.25, None);
        let retained = (0..16)
            .map(|i| RegressionObservation {
                raw_x: [10.0, i as f64],
                observed_ms: 3.0 + 7.0 * i as f64,
            })
            .collect::<Vec<_>>();
        let outlier = RegressionObservation {
            raw_x: [1e12, 1e6],
            observed_ms: 1e12,
        };
        for &sample in &retained {
            recursive.add(sample);
        }
        recursive.add(outlier);
        recursive.remove(outlier);
        assert!(recursive.damaged_downdate);
        compare(&mut recursive, &retained, 5);
        assert!(!recursive.damaged_downdate);
        assert_eq!(recursive.mutations_since_rebuild, 0);
        assert_eq!(recursive.scatter[0][0], 0.0);
        assert_eq!(recursive.rebuild_interval, None);
        assert_eq!(recursive.ridge_scale, 0.25);
    }

    #[test]
    fn disabled_periodic_rebuilds_still_recover_nonfinite_statistics() {
        let mut recursive = RecursiveFit::new(0.25, None);
        let retained = (0..64).map(observation).collect::<Vec<_>>();
        for &sample in &retained {
            recursive.add(sample);
        }
        recursive.scatter[0][1] = f64::INFINITY;
        compare(&mut recursive, &retained, 5);
        assert!(recursive.statistics_are_finite());
        assert_eq!(recursive.mutations_since_rebuild, 0);
    }

    #[test]
    fn mutation_clock_saturates_when_periodic_rebuilds_are_disabled() {
        let mut recursive = RecursiveFit::new(0.25, None);
        for sample in (0..64).map(observation) {
            recursive.add(sample);
        }
        recursive.mutations_since_rebuild = usize::MAX - 1;
        let sample = observation(100);
        recursive.add(sample);
        recursive.remove(sample);
        assert_eq!(recursive.mutations_since_rebuild, usize::MAX);
        assert!(
            recursive
                .fit_lazy(usize::MAX, || panic!("disabled periodic rebuild"))
                .is_none()
        );
        // A maximum finite interval must still trigger at the saturated value.
        recursive.rebuild_interval = Some(usize::MAX);
        let retained = (0..64).map(observation).collect::<Vec<_>>();
        recursive.fit_lazy(usize::MAX, || retained.clone());
        assert_eq!(recursive.mutations_since_rebuild, 0);
    }

    #[test]
    fn clone_preserves_configuration_and_has_independent_statistics() {
        let mut original = RecursiveFit::new(0.25, Some(37));
        for sample in (0..20).map(observation) {
            original.add(sample);
        }
        let means = original.means;
        let scatter = original.scatter;
        let mut cloned = original.clone();
        assert_eq!(cloned.rebuild_interval, Some(37));
        assert_eq!(cloned.ridge_scale, 0.25);
        assert_eq!(cloned.mutations_since_rebuild, 20);
        cloned.add(observation(100));
        assert_eq!(original.count, 20);
        assert_eq!(original.mutations_since_rebuild, 20);
        assert_eq!(original.means, means);
        assert_eq!(original.scatter, scatter);
        assert_eq!(cloned.count, 21);
        assert_ne!(cloned.means, means);
    }

    #[test]
    fn removing_last_observation_preserves_configuration_and_allows_reuse() {
        let mut recursive = RecursiveFit::new(0.25, None);
        let first = observation(0);
        recursive.add(first);
        recursive.remove(first);
        assert!(recursive.fit(&[], 1).is_none());
        assert_eq!(recursive.ridge_scale, 0.25);
        assert_eq!(recursive.rebuild_interval, None);
        let retained = (1..20).map(observation).collect::<Vec<_>>();
        for &sample in &retained {
            recursive.add(sample);
        }
        compare(&mut recursive, &retained, 5);
    }

    #[test]
    fn bucketed_updates_match_exact_retained_batch_with_rectangular_grid() {
        let options = ForwardPassPerfOptions {
            min_observations: 5,
            max_observations: 32,
            bucket_shape: Some([2, 3]),
            regression_ridge_scale: 0.25,
            ..ForwardPassPerfOptions::default()
        };
        for interval in [None, Some(17)] {
            let mut regression = BucketedRegression::new(&options, interval);
            for i in 0..300 {
                let sample = observation(i);
                assert!(regression.add_observation(sample.raw_x, sample.observed_ms));
                let retained = regression
                    .samples
                    .observations()
                    .into_iter()
                    .map(|(_, sample)| sample)
                    .collect::<Vec<_>>();
                assert_eq!(regression.recursive.count, retained.len());
                let batch = fit_regression_with_ridge(&retained, 5, 0.25);
                compare_fits(regression.fit.as_ref(), batch.as_ref(), &retained);
            }
        }
    }

    #[test]
    fn rejected_inputs_and_rebucketing_do_not_add_mutations() {
        let options = ForwardPassPerfOptions {
            min_observations: 5,
            max_observations: 4,
            bucket_shape: Some([2, 3]),
            ..ForwardPassPerfOptions::default()
        };
        let mut regression = BucketedRegression::new(&options, None);
        for (index, x) in [[0.0, 0.0], [1.0, 1.0], [15.0, 15.0]]
            .into_iter()
            .enumerate()
        {
            assert!(regression.add_observation(x, 1.0 + index as f64));
            assert_eq!(regression.recursive.mutations_since_rebuild, index + 1);
            assert_eq!(regression.recursive.count, index + 1);
        }
        assert!(!regression.add_observation([f64::NAN, 1.0], 2.0));
        assert!(!regression.add_observation([1.0, 1.0], -1.0));
        assert_eq!(regression.recursive.mutations_since_rebuild, 3);
        assert_eq!(regression.recursive.count, 3);
        assert!(regression.add_observation([3.0, 0.0], 4.0));
        assert!(regression.add_observation([5.0, 5.0], 5.0));
        assert_eq!(regression.recursive.count, 4);
        assert_eq!(regression.recursive.mutations_since_rebuild, 6);
    }
}
