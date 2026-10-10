// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Recursive centered sufficient statistics for the exact retained sample set.
//!
//! Adapted from the locally authored research implementation in
//! work/spline_regression_period_20260921/rust/recursive.rs. No external code.
//! That private research artifact is not published in this repository; the path
//! records provenance and is not a reproducible validation input.
//! This solves nonnegative least squares after each rank-one moment update;
//! it does not clip unconstrained inverse-RLS coefficients.
use super::fit::{Model, P, Row, features, fit as batch_fit, kkt_valid, nnls};

const D: usize = P + 1;

#[derive(Clone, Debug)]
pub(super) struct RecursiveSpline {
    knots: [Vec<f64>; 2],
    p: usize,
    means: [f64; D],
    mean_compensation: [f64; D],
    scatter: [[f64; D]; D],
    scatter_compensation: [[f64; D]; D],
    rebuild_interval: Option<usize>,
    damaged: bool,
    pub(super) count: usize,
    pub(super) mutations: u64,
    pub(super) rebuilds: u64,
    pub(super) batch_fallbacks: u64,
}

fn compensated_add(total: &mut f64, compensation: &mut f64, increment: f64) {
    let adjusted = increment - *compensation;
    let next = *total + adjusted;
    *compensation = (next - *total) - adjusted;
    *total = next;
}

fn rank_deficient(gram: &[[f64; P]; P], p: usize) -> bool {
    let ids: Vec<usize> = (0..p).filter(|&i| gram[i][i] > 0.0).collect();
    let n = ids.len();
    let mut a = [[0.0; P]; P];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = gram[ids[i]][ids[j]];
        }
    }
    // Symmetric diagonal pivoting of the normalized covariance matrix. Any
    // unsupported or nearly dependent varying direction uses chronological batch
    // coefficients to keep predictions deterministic outside the retained span.
    for i in 0..n {
        let pivot = (i..n).max_by(|&j, &k| a[j][j].total_cmp(&a[k][k])).unwrap();
        if a[pivot][pivot] < 1e-10 {
            return true;
        }
        a.swap(i, pivot);
        for row in &mut a {
            row.swap(i, pivot);
        }
        for j in i + 1..n {
            for k in j..n {
                a[j][k] -= a[j][i] * a[k][i] / a[i][i];
                a[k][j] = a[j][k];
            }
        }
    }
    false
}

impl RecursiveSpline {
    pub(super) fn new(
        data: &[Row],
        knots: &[Vec<f64>; 2],
        rebuild_interval: Option<usize>,
    ) -> Self {
        let p = 2 + knots[0].len() + knots[1].len();
        assert!(p <= P);
        let mut state = Self {
            knots: knots.clone(),
            p,
            means: [0.0; D],
            mean_compensation: [0.0; D],
            scatter: [[0.0; D]; D],
            scatter_compensation: [[0.0; D]; D],
            rebuild_interval,
            damaged: false,
            count: 0,
            mutations: 0,
            rebuilds: 0,
            batch_fallbacks: 0,
        };
        // A fresh knot epoch gets a two-pass centered rebuild in the retained
        // chronological order, reducing accumulated rank-update roundoff.
        state.rebuild(data);
        state.rebuilds = 0;
        state.mutations = 0;
        state
    }

    fn vector(&self, row: Row) -> [f64; D] {
        let f = features([row[0], row[1]], &self.knots);
        let mut z = [0.0; D];
        z[..self.p].copy_from_slice(&f[..self.p]);
        z[self.p] = row[2];
        z
    }

    pub(super) fn add(&mut self, row: Row) {
        self.mutations = self.mutations.saturating_add(1);
        let z = self.vector(row);
        if self.count == 0 {
            self.means = z;
            self.mean_compensation = [0.0; D];
            self.scatter = [[0.0; D]; D];
            self.scatter_compensation = [[0.0; D]; D];
            self.count = 1;
            self.damaged = false;
            return;
        }
        let delta: [f64; D] = std::array::from_fn(|i| z[i] - self.means[i]);
        self.count += 1;
        for i in 0..=self.p {
            compensated_add(
                &mut self.means[i],
                &mut self.mean_compensation[i],
                delta[i] / self.count as f64,
            );
        }
        let after: [f64; D] = std::array::from_fn(|i| z[i] - self.means[i]);
        for i in 0..=self.p {
            for j in i..=self.p {
                let increment = if i == j {
                    delta[i] * after[i]
                } else {
                    0.5 * delta[i] * after[j] + 0.5 * delta[j] * after[i]
                };
                compensated_add(
                    &mut self.scatter[i][j],
                    &mut self.scatter_compensation[i][j],
                    increment,
                );
                self.scatter[j][i] = self.scatter[i][j];
            }
        }
    }

    pub(super) fn remove(&mut self, row: Row) {
        self.mutations = self.mutations.saturating_add(1);
        if self.count <= 1 {
            self.count = 0;
            self.means = [0.0; D];
            self.mean_compensation = [0.0; D];
            self.scatter = [[0.0; D]; D];
            self.scatter_compensation = [[0.0; D]; D];
            self.damaged = false;
            return;
        }
        let z = self.vector(row);
        let delta: [f64; D] = std::array::from_fn(|i| z[i] - self.means[i]);
        self.count -= 1;
        for i in 0..=self.p {
            compensated_add(
                &mut self.means[i],
                &mut self.mean_compensation[i],
                -delta[i] / self.count as f64,
            );
        }
        let after: [f64; D] = std::array::from_fn(|i| z[i] - self.means[i]);
        for i in 0..=self.p {
            for j in i..=self.p {
                let previous = self.scatter[i][j];
                let decrement = if i == j {
                    delta[i] * after[i]
                } else {
                    0.5 * delta[i] * after[j] + 0.5 * delta[j] * after[i]
                };
                compensated_add(
                    &mut self.scatter[i][j],
                    &mut self.scatter_compensation[i][j],
                    -decrement,
                );
                self.scatter[j][i] = self.scatter[i][j];
                if i == j
                    && (self.scatter[i][i] < 0.0
                        || (previous > 0.0 && self.scatter[i][i] < previous * 1e-8))
                {
                    self.damaged = true;
                }
            }
        }
    }

    fn rebuild(&mut self, rows: &[Row]) {
        self.count = rows.len();
        self.mutations = 0;
        self.means = [0.0; D];
        self.mean_compensation = [0.0; D];
        self.scatter = [[0.0; D]; D];
        self.scatter_compensation = [[0.0; D]; D];
        self.damaged = false;
        self.rebuilds = self.rebuilds.saturating_add(1);
        if self.count == 0 {
            return;
        }
        let vectors: Vec<[f64; D]> = rows.iter().map(|&r| self.vector(r)).collect();
        for z in &vectors {
            for i in 0..=self.p {
                compensated_add(
                    &mut self.means[i],
                    &mut self.mean_compensation[i],
                    z[i] / self.count as f64,
                );
            }
        }
        for i in 0..=self.p {
            if vectors.iter().all(|z| z[i] == vectors[0][i]) {
                self.means[i] = vectors[0][i];
                self.mean_compensation[i] = 0.0;
            }
        }
        for z in &vectors {
            for i in 0..=self.p {
                for j in i..=self.p {
                    compensated_add(
                        &mut self.scatter[i][j],
                        &mut self.scatter_compensation[i][j],
                        (z[i] - self.means[i]) * (z[j] - self.means[j]),
                    );
                    self.scatter[j][i] = self.scatter[i][j];
                }
            }
        }
    }

    fn valid_statistics(&self) -> bool {
        (0..=self.p).all(|i| {
            self.means[i].is_finite() && self.scatter[i][i].is_finite() && self.scatter[i][i] >= 0.0
        }) && self.scatter.iter().flatten().all(|v| v.is_finite())
    }

    /// Check after the complete insert/evict transaction, before a lazy skip.
    pub(super) fn rebuild_required(&self) -> bool {
        self.damaged
            || !self.valid_statistics()
            || self
                .rebuild_interval
                .is_some_and(|interval| self.mutations >= interval as u64)
    }

    fn rebuild_and_fit(&mut self, rows: &[Row]) -> Option<Model> {
        self.rebuild(rows);
        // The caller replaces the whole model, including an unusable None result.
        batch_fit(rows, &self.knots)
    }

    #[cfg(test)]
    pub(super) fn damage_for_test(&mut self) {
        self.scatter[0][0] = f64::NAN;
    }

    /// Healthy fixed-basis updates do not collect or traverse retained samples.
    /// Full rebuilds use one retained snapshot to refresh statistics and the
    /// reference batch coefficients, including when periodic rebuilding is due.
    pub(super) fn fit_lazy(&mut self, mut retained: impl FnMut() -> Vec<Row>) -> Option<Model> {
        if self.rebuild_required() {
            return self.rebuild_and_fit(&retained());
        }
        if self.count == 0 {
            return None;
        }
        let threshold = 1e-12 * (self.count as f64).sqrt();
        let norms: [f64; P] = std::array::from_fn(|j| {
            if j < self.p {
                self.scatter[j][j].max(0.0).sqrt()
            } else {
                0.0
            }
        });
        let mut gram = [[0.0; P]; P];
        let mut rhs = [0.0; P];
        for i in 0..self.p {
            if norms[i] <= threshold {
                continue;
            }
            rhs[i] = self.scatter[i][self.p] / norms[i];
            for j in 0..self.p {
                if norms[j] > threshold {
                    gram[i][j] = self.scatter[i][j] / norms[i] / norms[j];
                }
            }
        }
        // Tiny spread around a large offset is sensitive to mean rounding.
        // Match the chronological batch solution instead of amplifying moment
        // error into very large slopes, including predictions outside support.
        let fragile_spread = (0..self.p).any(|j| {
            let scale = norms[j] / (self.count as f64).sqrt();
            scale > 0.0 && scale <= 1e-8 * self.means[j].abs().max(1.0)
        });
        if !self.valid_statistics() || fragile_spread || rank_deficient(&gram, self.p) {
            self.batch_fallbacks = self.batch_fallbacks.saturating_add(1);
            return self.rebuild_and_fit(&retained());
        }
        let standardized = nnls(&gram, &rhs, self.p);
        let slopes: [f64; P] = std::array::from_fn(|j| {
            if norms[j] > threshold {
                standardized[j] / norms[j]
            } else {
                0.0
            }
        });
        let intercept =
            self.means[self.p] - (0..self.p).map(|j| slopes[j] * self.means[j]).sum::<f64>();
        let cross = (0..self.p)
            .map(|i| slopes[i] * self.scatter[i][self.p])
            .sum::<f64>();
        let quadratic = (0..self.p)
            .map(|i| {
                slopes[i]
                    * (0..self.p)
                        .map(|j| self.scatter[i][j] * slopes[j])
                        .sum::<f64>()
            })
            .sum::<f64>();
        let sse = self.scatter[self.p][self.p] - 2.0 * cross + quadratic;
        let roundoff = 1e-10
            * (self.scatter[self.p][self.p].abs() + 2.0 * cross.abs() + quadratic.abs()).max(1.0);
        if !intercept.is_finite()
            || slopes.iter().any(|v| !v.is_finite() || *v < 0.0)
            || !sse.is_finite()
            || sse < -roundoff
            || !kkt_valid(&gram, &rhs, &standardized, self.p)
        {
            self.batch_fallbacks = self.batch_fallbacks.saturating_add(1);
            return self.rebuild_and_fit(&retained());
        }
        Some(Model {
            knots: self.knots.clone(),
            slopes,
            intercept,
            sse: sse.max(0.0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_rebuild_coalesces_triggers_and_preserves_failed_fit_readiness() {
        let knots = [vec![], vec![]];
        // Independent hand oracle: 5 + 2*x + 3*z gives 34 at (7, 5).
        let rows = (0..16)
            .map(|i| {
                let (x, z) = ((i % 4) as f64, (i / 4) as f64);
                [x, z, 5.0 + 2.0 * x + 3.0 * z]
            })
            .collect::<Vec<_>>();
        let mut state = RecursiveSpline::new(&rows, &knots, Some(1));
        let means = state.means;
        let scatter = state.scatter;
        state.mutations = 1;
        state.damage_for_test();
        assert!(state.rebuild_required());
        let mut reads = 0;
        let model = state
            .fit_lazy(|| {
                reads += 1;
                rows.clone()
            })
            .unwrap();
        assert_eq!(reads, 1);
        assert_eq!(state.rebuilds, 1);
        assert_eq!(state.mutations, 0);
        assert_eq!(state.means, means);
        assert_eq!(state.scatter, scatter);
        assert!(!state.rebuild_required());
        assert!((model.predict([7.0, 5.0]) - 34.0).abs() < 1e-9);

        // Finite targets whose squared residuals overflow are not a usable fit.
        let huge = [[0.0, 0.0, f64::MAX], [1.0, 1.0, 1.0]];
        let mut failed = RecursiveSpline::new(&huge, &knots, None);
        let mut reads = 0;
        assert!(
            failed
                .fit_lazy(|| {
                    reads += 1;
                    huge.to_vec()
                })
                .is_none()
        );
        assert_eq!(reads, 1);
        assert_eq!(failed.rebuilds, 1);
        assert!(failed.rebuild_required());
    }
}
