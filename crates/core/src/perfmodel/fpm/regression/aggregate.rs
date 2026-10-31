// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in linear fit on the learned model's cross-rank `fpm5` aggregates.
//!
//! Seventeen features rule out the face enumeration and incremental statistics
//! of the small fits. Each publication instead refits the retained rows (at
//! most `sampling.max_observations`) with Lawson–Hanson nonnegative least
//! squares, or a signed solve, on standardized features. The intercept is free
//! and the configured ridge always regularizes the slopes, so collinear
//! aggregates still have one solution. Constant and exactly duplicated features
//! are inactive. Retention keeps using the `sampling.axes` buckets.
use super::super::estimator::RegressionFeatureAxis;
use super::super::learned::{FPM5_FEATURE_COUNT, fpm5_features};
use super::*;

const D: usize = FPM5_FEATURE_COUNT;
/// Outer Lawson–Hanson iterations allowed per active feature.
const NNLS_ITERATIONS_PER_FEATURE: usize = 30;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AggregateRegression {
    sampling_axes: Vec<RegressionFeatureAxis>,
    samples: BucketedSamples<RegressionObservation<D>>,
    min_observations: usize,
    ridge_scale: f64,
    non_negative: bool,
    fit: Option<LinearFit<D>>,
    update: update::LinearUpdateState,
}

impl AggregateRegression {
    pub(super) fn new(
        options: &ForwardPassPerfOptions,
        fit: &RegressionFitConfig,
        sampling: &RegressionSamplingConfig,
    ) -> Self {
        let linear = fit.linear.clone().unwrap_or_default();
        Self {
            sampling_axes: sampling.axes.clone(),
            samples: BucketedSamples::new_regression(
                &sampling.bins_per_axis,
                sampling.max_observations,
            )
            .expect("validated regression sampling"),
            min_observations: options.min_observations,
            ridge_scale: options.regression_ridge_scale,
            non_negative: linear.non_negative,
            fit: None,
            update: update::LinearUpdateState::new(linear.update_policy),
        }
    }

    /// Retain an observation and, when the update policy asks for it, refit
    /// the retained rows. Retention coordinates are projected first, so an
    /// iteration they reject changes nothing.
    pub(super) fn add(
        &mut self,
        scores: [f64; 2],
        target: f64,
        metrics: &[ForwardPassMetrics],
    ) -> Result<bool, AicError> {
        let projected = feature_axes::project(scores, metrics, &self.sampling_axes)?;
        let bucket = projected[..self.sampling_axes.len()]
            .iter()
            .map(|value| value.ln_1p())
            .collect();
        let raw_x = fpm5_features(metrics);
        if !valid_features(&raw_x) || !target.is_finite() || target <= 0.0 {
            return Ok(false);
        }
        let prior = if self.update.is_always() {
            None
        } else {
            self.fit.as_ref().and_then(|fit| fit.predict(&raw_x))
        };
        let observation = RegressionObservation {
            raw_x,
            observed_ms: target,
        };
        let SampleInsertion::Accepted { .. } = self.samples.add_with_eviction(bucket, observation)
        else {
            return Ok(false);
        };
        // Every publication is a batch fit, so there is no rebuild clock.
        if !self.update.is_always()
            && !self
                .update
                .accepted(prior, target, self.fit.is_some(), false)
        {
            return Ok(true);
        }
        let retained = self
            .samples
            .observations()
            .into_iter()
            .map(|(_, observation)| observation)
            .collect::<Vec<_>>();
        let candidate = fit_aggregate(
            &retained,
            self.min_observations,
            self.ridge_scale,
            self.non_negative,
        );
        let usable = candidate
            .as_ref()
            .is_some_and(|fit| fit.is_usable(retained.len()));
        // Only a usable candidate replaces the serving fit. An all-zero slope
        // candidate, or a refit that has fewer retained rows than newly active
        // aggregates (for example when multi-rank traffic first appears), keeps
        // serving the previous fit instead of dropping readiness.
        if usable {
            self.fit = candidate;
        }
        self.update.fitted(usable);
        Ok(true)
    }

    pub(super) fn predict(&self, metrics: &[ForwardPassMetrics]) -> Option<f64> {
        let prediction = self.fit.as_ref()?.predict(&fpm5_features(metrics))?;
        Some(prediction.max(MIN_POSITIVE_PREDICTION_MS))
    }

    pub(super) fn restore_snapshot(&mut self, fresh: &Self) -> Result<(), String> {
        if self.sampling_axes != fresh.sampling_axes
            || self.min_observations != fresh.min_observations
            || self.ridge_scale.to_bits() != fresh.ridge_scale.to_bits()
            || self.non_negative != fresh.non_negative
        {
            return Err("aggregate regression controls differ from configuration".into());
        }
        self.samples
            .validate_restored(&fresh.samples, |x, observation| {
                if !observation.is_valid() {
                    return Err("retained regression observation is invalid".into());
                }
                // The aggregate row does not determine retention coordinates;
                // they are `ln_1p` of nonnegative projections.
                if x.iter().any(|bucket| *bucket < 0.0) {
                    return Err("retained regression bucket coordinate is negative".into());
                }
                Ok(())
            })?;
        self.update.restore_snapshot(&fresh.update)?;
        if let Some(fit) = &self.fit {
            fit.validate(self.non_negative)?;
        }
        Ok(())
    }
}

impl StoreStats for AggregateRegression {
    fn observation_count(&self) -> usize {
        self.samples.total_observations
    }

    fn is_ready(&self) -> bool {
        self.fit.is_some()
    }
}

/// Batch fit of the retained rows. A candidate needs at least
/// `max(min_observations, active features + 1)` rows.
fn fit_aggregate(
    observations: &[RegressionObservation<D>],
    min_observations: usize,
    ridge_scale: f64,
    non_negative: bool,
) -> Option<LinearFit<D>> {
    if observations.len() < min_observations {
        return None;
    }
    let mut standardization = Standardization::from_observations(observations)?;
    // With one attention-DP rank every `max_rank_*` equals its sum, and a
    // single prefill request makes the means equal their sums. The first of
    // identical columns stays active.
    for axis in 0..D {
        if standardization.active[axis]
            && (0..axis).any(|earlier| {
                standardization.active[earlier]
                    && observations
                        .iter()
                        .all(|row| row.raw_x[earlier] == row.raw_x[axis])
            })
        {
            standardization.active[axis] = false;
        }
    }
    let active = (0..D)
        .filter(|axis| standardization.active[*axis])
        .collect::<Vec<_>>();
    if active.is_empty() || observations.len() <= active.len() {
        return None;
    }

    let rows = observations
        .iter()
        .map(|observation| standardization.transform(&observation.raw_x))
        .collect::<Option<Vec<_>>>()?;
    let count = observations.len() as f64;
    // Centering the standardized columns again removes mean roundoff, so the
    // free intercept separates exactly from the slopes.
    let column_means = active
        .iter()
        .map(|axis| rows.iter().map(|row| row[*axis]).sum::<f64>() / count)
        .collect::<Vec<_>>();
    let target_mean = observations
        .iter()
        .map(|observation| observation.observed_ms)
        .sum::<f64>()
        / count;
    let size = active.len();
    let mut gram = vec![vec![0.0_f64; size]; size];
    let mut rhs = vec![0.0_f64; size];
    for (row, observation) in rows.iter().zip(observations) {
        let centered = active
            .iter()
            .zip(&column_means)
            .map(|(axis, mean)| row[*axis] - mean)
            .collect::<Vec<_>>();
        let target = observation.observed_ms - target_mean;
        for i in 0..size {
            rhs[i] += centered[i] * target;
            for j in 0..size {
                gram[i][j] += centered[i] * centered[j];
            }
        }
    }
    // Same ridge magnitude as the small fits' singular retry, but always
    // applied to the slopes.
    let ridge = ridge_scale * (count + (0..size).map(|i| gram[i][i]).sum::<f64>()).max(1.0);
    for (i, row) in gram.iter_mut().enumerate() {
        row[i] += ridge;
    }
    let slopes = if non_negative {
        nonnegative_least_squares(&gram, &rhs)?
    } else {
        solve_linear_system(gram, rhs)?
    };
    if !slopes.iter().all(|slope| slope.is_finite()) {
        return None;
    }
    let mut coefficients = [0.0; D];
    let mut intercept = target_mean;
    for ((axis, slope), mean) in active.iter().zip(&slopes).zip(&column_means) {
        coefficients[*axis] = *slope;
        intercept -= slope * mean;
    }
    intercept.is_finite().then_some(LinearFit {
        intercept,
        coefficients,
        standardization,
    })
}

/// Lawson–Hanson active-set NNLS on the normal equations `gram * x = rhs`
/// (the covariance form of Bro and De Jong). Returns `None` when a passive
/// system is singular or the iteration bound is reached.
fn nonnegative_least_squares(gram: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let size = rhs.len();
    let norm = (0..size)
        .map(|j| gram.iter().map(|row| row[j].abs()).sum::<f64>())
        .fold(0.0, f64::max);
    let tolerance = 10.0 * f64::EPSILON * norm * size as f64;
    let mut passive = vec![false; size];
    // Roundoff can give the entering coefficient a nonpositive passive
    // solution; skip such candidates until the solution moves.
    let mut blocked = vec![false; size];
    let mut solution = vec![0.0; size];
    for _ in 0..NNLS_ITERATIONS_PER_FEATURE * size {
        let gradient = (0..size)
            .map(|i| rhs[i] - (0..size).map(|j| gram[i][j] * solution[j]).sum::<f64>())
            .collect::<Vec<_>>();
        let Some(entering) = (0..size)
            .filter(|&i| !passive[i] && !blocked[i] && gradient[i] > tolerance)
            .max_by(|&a, &b| gradient[a].total_cmp(&gradient[b]))
        else {
            return Some(solution);
        };
        passive[entering] = true;
        let mut first = true;
        loop {
            let trial = solve_passive(gram, rhs, &passive)?;
            if (0..size).all(|i| !passive[i] || trial[i] > 0.0) {
                solution = trial;
                blocked.fill(false);
                break;
            }
            if first && trial[entering] <= 0.0 {
                passive[entering] = false;
                blocked[entering] = true;
                break;
            }
            first = false;
            // Move toward the trial point until the first passive coefficient
            // reaches zero, then release every nonpositive coefficient.
            let (leaving, step) = (0..size)
                .filter(|&i| passive[i] && trial[i] <= 0.0)
                .map(|i| (i, solution[i] / (solution[i] - trial[i])))
                .min_by(|a, b| a.1.total_cmp(&b.1))?;
            for i in 0..size {
                if passive[i] {
                    solution[i] += step * (trial[i] - solution[i]);
                    if i == leaving || solution[i] <= 0.0 {
                        passive[i] = false;
                        solution[i] = 0.0;
                    }
                }
            }
        }
    }
    None
}

/// Solve the normal equations restricted to the passive coefficients; the
/// others are zero.
fn solve_passive(gram: &[Vec<f64>], rhs: &[f64], passive: &[bool]) -> Option<Vec<f64>> {
    let indices = (0..rhs.len()).filter(|i| passive[*i]).collect::<Vec<_>>();
    let lhs = indices
        .iter()
        .map(|i| indices.iter().map(|j| gram[*i][*j]).collect())
        .collect();
    let reduced = indices.iter().map(|i| rhs[*i]).collect();
    let solved = solve_linear_system(lhs, reduced)?;
    let mut trial = vec![0.0; rhs.len()];
    for (index, value) in indices.into_iter().zip(solved) {
        trial[index] = value;
    }
    Some(trial)
}

#[cfg(test)]
mod tests;
