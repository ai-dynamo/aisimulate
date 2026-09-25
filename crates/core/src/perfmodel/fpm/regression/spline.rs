// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Learned-knot spline and linear fallback sharing one production sampler.
//!
//! Search and monitor semantics originate in this repository's locally authored
//! work/spline_regression_period_20260921/rust/{replay,triggered}.rs. The sampler
//! and linear fallback here are the production implementations, not the research
//! replay approximations. No external source was incorporated.

use std::collections::VecDeque;

use super::super::estimator::{SplineFitConfig, SplineSearchConfig};
use super::{
    BucketedSamples, ForwardPassPerfOptions, ForwardPassSplineDiagnostics, LinearFit,
    MIN_POSITIVE_PREDICTION_MS, RecursiveFit, RegressionObservation, SampleInsertion, StoreStats,
    valid_features,
};

mod fit;
mod recursive;
#[cfg(test)]
mod tests;

use fit::{Model, Row};
use recursive::RecursiveSpline;

#[derive(Clone, Copy, Debug)]
struct RetainedObservation {
    sequence: u64,
    observation: RegressionObservation,
}

#[derive(Clone, Debug, Default)]
struct ErrorMonitor {
    errors: VecDeque<bool>,
    bad: usize,
}

impl ErrorMonitor {
    fn observe(&mut self, bad: bool, window: usize) {
        self.errors.push_back(bad);
        self.bad += usize::from(bad);
        if self.errors.len() > window {
            self.bad -= usize::from(self.errors.pop_front().unwrap_or(false));
        }
    }

    fn clear(&mut self) {
        self.errors.clear();
        self.bad = 0;
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BucketedSpline {
    samples: BucketedSamples<RetainedObservation>,
    config: SplineFitConfig,
    min_observations: usize,
    rebuild_interval: Option<usize>,
    linear: RecursiveFit,
    linear_fit: Option<LinearFit>,
    recursive: Option<RecursiveSpline>,
    spline_fit: Option<Model>,
    scale: [f64; 2],
    minimum: [f64; 2],
    maximum: [f64; 2],
    accepted: u64,
    searches: u64,
    last_search: Option<u64>,
    completed_rebuilds: u64,
    completed_fallbacks: u64,
    monitor: ErrorMonitor,
}

fn scaled(observation: RegressionObservation, scale: [f64; 2]) -> Row {
    [
        observation.raw_x[0] / scale[0],
        observation.raw_x[1] / scale[1],
        observation.observed_ms,
    ]
}

fn chronological_rows(samples: &BucketedSamples<RetainedObservation>, scale: [f64; 2]) -> Vec<Row> {
    let mut retained = samples.observations();
    retained.sort_unstable_by_key(|(_, entry)| entry.sequence);
    retained
        .into_iter()
        .map(|(_, entry)| scaled(entry.observation, scale))
        .collect()
}

impl BucketedSpline {
    pub(super) fn new(
        options: &ForwardPassPerfOptions,
        config: SplineFitConfig,
        rebuild_interval: Option<usize>,
    ) -> Self {
        Self {
            samples: BucketedSamples::new_dynamic(options, 2),
            config,
            min_observations: options.min_observations,
            rebuild_interval,
            linear: RecursiveFit::new(options.regression_ridge_scale, rebuild_interval),
            linear_fit: None,
            recursive: None,
            spline_fit: None,
            scale: [1.0; 2],
            minimum: [f64::INFINITY; 2],
            maximum: [f64::NEG_INFINITY; 2],
            accepted: 0,
            searches: 0,
            last_search: None,
            completed_rebuilds: 0,
            completed_fallbacks: 0,
            monitor: ErrorMonitor::default(),
        }
    }

    fn raw_spline_prediction(&self, raw_x: &[f64; 2]) -> Option<f64> {
        let model = self.spline_fit.as_ref().filter(|model| model.is_finite())?;
        let value = model.predict([raw_x[0] / self.scale[0], raw_x[1] / self.scale[1]]);
        value.is_finite().then_some(value)
    }

    fn inside(&self, raw_x: &[f64; 2]) -> bool {
        (0..2).all(|axis| raw_x[axis] >= self.minimum[axis] && raw_x[axis] <= self.maximum[axis])
    }

    pub(super) fn predict(&self, raw_x: &[f64; 2]) -> Option<f64> {
        if !valid_features(raw_x) {
            return None;
        }
        if self.inside(raw_x) && self.spline_fit.as_ref().is_some_and(Model::is_ready) {
            if let Some(value) = self.raw_spline_prediction(raw_x) {
                return Some(value.max(MIN_POSITIVE_PREDICTION_MS));
            }
        }
        self.linear_fit
            .as_ref()?
            .predict(raw_x)
            .map(|v| v.max(MIN_POSITIVE_PREDICTION_MS))
    }

    fn search_due(&self) -> bool {
        if self.samples.total_observations < self.min_observations {
            return false;
        }
        let Some(previous) = self.last_search else {
            return self.accepted >= self.min_observations.max(32) as u64;
        };
        match self.config.search {
            SplineSearchConfig::Periodic { step } => self.accepted.is_multiple_of(step as u64),
            SplineSearchConfig::Adaptive {
                trigger, cooldown, ..
            } => {
                self.accepted.saturating_sub(previous) >= cooldown as u64
                    && self.monitor.bad >= trigger
            }
        }
    }

    fn search(&mut self) {
        let raw = chronological_rows(&self.samples, [1.0; 2]);
        let next_scale = std::array::from_fn(|a| raw.iter().map(|r| r[a]).fold(1.0, f64::max));
        let previous = self.spline_fit.as_ref().map(|model| {
            std::array::from_fn(|a| {
                model.knots[a]
                    .iter()
                    .map(|&v| v * (self.scale[a] / next_scale[a]))
                    .collect()
            })
        });
        self.scale = next_scale;
        let rows: Vec<Row> = raw
            .into_iter()
            .map(|r| [r[0] / self.scale[0], r[1] / self.scale[1], r[2]])
            .collect();
        let next_model = fit::learn(&rows, self.config.knots_per_axis, previous);
        let knots = next_model
            .as_ref()
            .map(|m| m.knots.clone())
            .unwrap_or_else(|| fit::init_knots(&rows, self.config.knots_per_axis));
        if let Some(old) = self.recursive.take() {
            self.completed_rebuilds = self.completed_rebuilds.saturating_add(old.rebuilds);
            self.completed_fallbacks = self.completed_fallbacks.saturating_add(old.batch_fallbacks);
        }
        self.recursive = Some(RecursiveSpline::new(&rows, &knots, self.rebuild_interval));
        self.spline_fit = next_model;
        self.searches = self.searches.saturating_add(1);
        self.last_search = Some(self.accepted);
        self.monitor.clear();
    }

    fn update_bounds(
        &mut self,
        observation: RegressionObservation,
        evicted: Option<RegressionObservation>,
    ) {
        // Bucket boundaries are historical; the guard must use current retention.
        if evicted.is_some_and(|old| {
            old == observation
                || (0..2)
                    .any(|a| old.raw_x[a] == self.minimum[a] || old.raw_x[a] == self.maximum[a])
        }) {
            self.minimum = [f64::INFINITY; 2];
            self.maximum = [f64::NEG_INFINITY; 2];
            for (_, retained) in self.samples.observations() {
                for a in 0..2 {
                    self.minimum[a] = self.minimum[a].min(retained.observation.raw_x[a]);
                    self.maximum[a] = self.maximum[a].max(retained.observation.raw_x[a]);
                }
            }
        } else {
            for a in 0..2 {
                self.minimum[a] = self.minimum[a].min(observation.raw_x[a]);
                self.maximum[a] = self.maximum[a].max(observation.raw_x[a]);
            }
        }
    }

    pub(super) fn add_observation(&mut self, raw_x: [f64; 2], observed_ms: f64) -> bool {
        if !valid_features(&raw_x) || !observed_ms.is_finite() || observed_ms <= 0.0 {
            return false;
        }
        // The error belongs to the model BEFORE it sees this target, irrespective
        // of whether the retained-range guard would have served linear output.
        let prior = if matches!(self.config.search, SplineSearchConfig::Adaptive { .. }) {
            self.raw_spline_prediction(&raw_x)
        } else {
            None
        };
        let observation = RegressionObservation { raw_x, observed_ms };
        let entry = RetainedObservation {
            sequence: self.accepted,
            observation,
        };
        let SampleInsertion::Accepted { evicted } = self
            .samples
            .add_with_eviction(raw_x.map(f64::ln_1p).to_vec(), entry)
        else {
            return false;
        };
        let evicted = evicted.map(|entry| entry.observation);
        self.accepted = self.accepted.saturating_add(1);
        if let (
            Some(prediction),
            SplineSearchConfig::Adaptive {
                window,
                tolerance,
                absolute_tolerance_ms,
                ..
            },
        ) = (prior, &self.config.search)
        {
            self.monitor.observe(
                (prediction - observed_ms).abs()
                    > absolute_tolerance_ms.max(tolerance * observed_ms),
                *window,
            );
        }
        self.update_bounds(observation, evicted);
        self.linear.add(observation);
        if let Some(old) = evicted {
            self.linear.remove(old);
        }
        self.linear_fit = self.linear.fit_lazy(self.min_observations, || {
            self.samples
                .observations()
                .into_iter()
                .map(|(_, entry)| entry.observation)
                .collect()
        });
        if self.search_due() {
            self.search();
        } else if let Some(state) = &mut self.recursive {
            state.add(scaled(observation, self.scale));
            if let Some(old) = evicted {
                state.remove(scaled(old, self.scale));
            }
            debug_assert_eq!(state.count, self.samples.total_observations);
            self.spline_fit = state.fit_lazy(|| chronological_rows(&self.samples, self.scale));
        }
        true
    }

    pub(super) fn diagnostics(&self) -> ForwardPassSplineDiagnostics {
        ForwardPassSplineDiagnostics {
            initialized: self.recursive.is_some(),
            ready: self.spline_fit.as_ref().is_some_and(Model::is_ready),
            accepted_observations: self.accepted,
            knot_searches: self.searches,
            last_search_observation: self.last_search,
            numerical_rebuilds: self
                .completed_rebuilds
                .saturating_add(self.recursive.as_ref().map_or(0, |s| s.rebuilds)),
            batch_fallbacks: self
                .completed_fallbacks
                .saturating_add(self.recursive.as_ref().map_or(0, |s| s.batch_fallbacks)),
        }
    }

    #[cfg(test)]
    pub(super) fn mutations_since_rebuild(&self) -> usize {
        self.linear.mutations_since_rebuild()
    }
}

impl StoreStats for BucketedSpline {
    fn observation_count(&self) -> usize {
        self.samples.total_observations
    }
    fn is_ready(&self) -> bool {
        self.linear_fit.is_some() || self.spline_fit.as_ref().is_some_and(Model::is_ready)
    }
}
