// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Continuous additive linear splines with nonnegative segment slopes.
//!
//! Adapted from this repository's original research implementation in
//! work/spline_regression_period_20260921/rust/{main,replay}.rs.
//! No external code was incorporated. There is no penalty on the free intercept
//! or slopes; the bounded face enumeration is a constrained-solver recovery.

pub(super) const P: usize = 8;
pub(super) type Row = [f64; 3];

#[derive(Clone, Debug)]
pub(super) struct Model {
    pub(super) knots: [Vec<f64>; 2],
    pub(super) slopes: [f64; P],
    pub(super) intercept: f64,
    pub(super) sse: f64,
}

pub(super) fn features(x: [f64; 2], knots: &[Vec<f64>; 2]) -> [f64; P] {
    let mut out = [0.0; P];
    let mut j = 0;
    for axis in 0..2 {
        let mut left = 0.0;
        for &right in &knots[axis] {
            out[j] = (x[axis] - left).max(0.0).min(right - left);
            left = right;
            j += 1;
        }
        out[j] = (x[axis] - left).max(0.0);
        j += 1;
    }
    out
}

impl Model {
    #[inline]
    pub(super) fn predict(&self, x: [f64; 2]) -> f64 {
        let mut prediction = self.intercept;
        let mut j = 0;
        for axis in 0..2 {
            let mut left = 0.0;
            for &right in &self.knots[axis] {
                prediction += self.slopes[j] * (x[axis] - left).max(0.0).min(right - left);
                left = right;
                j += 1;
            }
            prediction += self.slopes[j] * (x[axis] - left).max(0.0);
            j += 1;
        }
        prediction
    }
}

// Solve a small passive-set system. Rank-deficient columns are assigned zero;
// this is sufficient for an NNLS optimum when their gradient also vanishes.
fn passive_solve(gram: &[[f64; P]; P], rhs: &[f64; P], passive: &[bool; P], p: usize) -> [f64; P] {
    let mut ids = [0usize; P];
    let mut n = 0;
    for i in 0..p {
        if passive[i] {
            ids[n] = i;
            n += 1;
        }
    }
    let mut a = [[0.0; P + 1]; P];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = gram[ids[i]][ids[j]];
        }
        a[i][n] = rhs[ids[i]];
    }
    let mut pivots = [0usize; P];
    let mut rank = 0;
    for col in 0..n {
        let row = (rank..n)
            .max_by(|&a0, &a1| a[a0][col].abs().total_cmp(&a[a1][col].abs()))
            .unwrap();
        if a[row][col].abs() < 1e-12 {
            continue;
        }
        a.swap(rank, row);
        for next in rank + 1..n {
            let mult = a[next][col] / a[rank][col];
            for j in col..=n {
                a[next][j] -= mult * a[rank][j];
            }
        }
        pivots[rank] = col;
        rank += 1;
        if rank == n {
            break;
        }
    }
    let mut local = [0.0; P];
    for i in (0..rank).rev() {
        let col = pivots[i];
        let remaining: f64 = ((col + 1)..n).map(|j| a[i][j] * local[j]).sum();
        local[col] = (a[i][n] - remaining) / a[i][col];
    }
    let mut out = [0.0; P];
    for i in 0..n {
        out[ids[i]] = local[i];
    }
    out
}

// Primal active-set nonnegative least squares, using centered, normalized columns.
// Each outer step activates a positive residual gradient, then a feasible line
// search removes any passive coefficient that reaches the constraint boundary.
pub(super) fn nnls(gram: &[[f64; P]; P], rhs: &[f64; P], p: usize) -> [f64; P] {
    let mut x = [0.0; P];
    let mut passive = [false; P];
    let tol = rhs[..p].iter().map(|v| v.abs()).fold(1.0, f64::max) * 1e-12;
    for _ in 0..(4 * p + 8) {
        let mut best = None;
        let mut best_gradient = tol;
        for i in 0..p {
            let gradient = rhs[i] - (0..p).map(|j| gram[i][j] * x[j]).sum::<f64>();
            if !passive[i] && gradient > best_gradient {
                best = Some(i);
                best_gradient = gradient;
            }
        }
        let Some(enter) = best else {
            return x;
        };
        passive[enter] = true;
        for _ in 0..(4 * p + 8) {
            let z = passive_solve(gram, rhs, &passive, p);
            if (0..p).all(|i| !passive[i] || z[i] > 0.0) {
                x = z;
                break;
            }
            let mut alpha = 1.0_f64;
            for i in 0..p {
                if passive[i] && z[i] <= 0.0 {
                    let denominator = x[i] - z[i];
                    alpha = alpha.min(if denominator > 0.0 {
                        x[i] / denominator
                    } else {
                        0.0
                    });
                }
            }
            for i in 0..p {
                x[i] += alpha * (z[i] - x[i]);
                if passive[i] && x[i] <= tol * 1e-3 {
                    x[i] = 0.0;
                    passive[i] = false;
                }
            }
        }
    }
    // Degenerate passive sets can cycle. Since there are at most eight slopes,
    // enumerate feasible stationary points of every face as a bounded fallback.
    // This remains a constrained solve, not clipping an unconstrained fit.
    let mut best = [0.0; P];
    let mut best_objective = 0.0;
    for mask in 1usize..(1usize << p) {
        let passive = std::array::from_fn(|j| j < p && (mask & (1 << j)) != 0);
        let z = passive_solve(gram, rhs, &passive, p);
        if z[..p].iter().any(|&v| !v.is_finite() || v < 0.0) {
            continue;
        }
        let objective = (0..p)
            .map(|i| z[i] * ((0..p).map(|j| gram[i][j] * z[j]).sum::<f64>() - 2.0 * rhs[i]))
            .sum::<f64>();
        if objective < best_objective {
            best = z;
            best_objective = objective;
        }
    }
    best
}

pub(super) fn fit(data: &[Row], knots: &[Vec<f64>; 2]) -> Option<Model> {
    let p = 2 + knots[0].len() + knots[1].len();
    if p > P || data.is_empty() || data.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    let n = data.len() as f64;
    let ymean = if data.iter().all(|r| r[2] == data[0][2]) {
        data[0][2]
    } else {
        data.iter().map(|r| r[2] / n).sum::<f64>()
    };
    let mut matrix: Vec<[f64; P]> = data.iter().map(|r| features([r[0], r[1]], knots)).collect();
    let mut means = [0.0; P];
    let mut norms = [0.0; P];
    for row in &matrix {
        for j in 0..p {
            means[j] += row[j] / n;
        }
    }
    // A constant column has exactly zero variance even when repeated summation
    // of a large offset would leave a rounding residual after centering.
    for j in 0..p {
        if matrix.iter().all(|row| row[j] == matrix[0][j]) {
            means[j] = matrix[0][j];
        }
    }
    for row in &mut matrix {
        for j in 0..p {
            row[j] -= means[j];
            norms[j] += row[j] * row[j];
        }
    }
    for j in 0..p {
        norms[j] = norms[j].sqrt();
    }
    let active_norm_threshold = 1e-12 * n.sqrt();
    for row in &mut matrix {
        for j in 0..p {
            row[j] = if norms[j] > active_norm_threshold {
                row[j] / norms[j]
            } else {
                0.0
            };
        }
    }
    let mut gram = [[0.0; P]; P];
    let mut rhs = [0.0; P];
    for (row, raw) in matrix.iter().zip(data) {
        for i in 0..p {
            rhs[i] += row[i] * (raw[2] - ymean);
            for j in i..p {
                gram[i][j] += row[i] * row[j];
            }
        }
    }
    for i in 0..p {
        for j in 0..i {
            gram[i][j] = gram[j][i];
        }
    }
    if !ymean.is_finite()
        || gram
            .iter()
            .flatten()
            .chain(rhs.iter())
            .any(|v| !v.is_finite())
    {
        return None;
    }
    let standardized = nnls(&gram, &rhs, p);
    if !kkt_valid(&gram, &rhs, &standardized, p) {
        return None;
    }
    let slopes = std::array::from_fn(|j| {
        if norms[j] > active_norm_threshold {
            standardized[j] / norms[j]
        } else {
            0.0
        }
    });
    let intercept = ymean - (0..p).map(|j| slopes[j] * means[j]).sum::<f64>();
    let mut model = Model {
        knots: knots.clone(),
        slopes,
        intercept,
        sse: 0.0,
    };
    model.sse = data
        .iter()
        .map(|r| (r[2] - model.predict([r[0], r[1]])).powi(2))
        .sum();
    model.is_finite().then_some(model)
}

fn unique_axis(data: &[Row], axis: usize) -> Vec<f64> {
    let mut values: Vec<f64> = data.iter().map(|r| r[axis]).collect();
    values.sort_by(f64::total_cmp);
    values.dedup();
    values
}

fn quantile(values: &[f64], q: f64) -> f64 {
    let position = (values.len() - 1) as f64 * q;
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    values[lo] + (values[hi] - values[lo]) * (position - lo as f64)
}

pub(super) fn init_knots(data: &[Row], k: usize) -> [Vec<f64>; 2] {
    std::array::from_fn(|axis| {
        let values = unique_axis(data, axis);
        let effective = k.min(values.len().saturating_sub(1));
        let mut out: Vec<f64> = (1..=effective)
            .map(|j| quantile(&values, j as f64 / (effective + 1) as f64))
            .collect();
        out.dedup();
        out
    })
}

fn candidates(data: &[Row], axis: usize) -> Vec<f64> {
    let values = unique_axis(data, axis);
    let mut candidates: Vec<f64> = (0..19)
        .map(|j| quantile(&values, 0.05 + j as f64 * 0.05))
        .collect();
    if let Some(min_positive) = values.iter().copied().find(|&x| x > 0.0) {
        let low = min_positive.max(1e-8).ln();
        let high = values.last().unwrap().ln();
        for j in 1..18 {
            candidates.push((low + j as f64 * (high - low) / 18.0).exp());
        }
    }
    candidates.sort_by(f64::total_cmp);
    candidates.dedup();
    candidates
}

// A finite feasible fit is distinct from a useful load-dependent predictor.
impl Model {
    pub(super) fn is_finite(&self) -> bool {
        self.intercept.is_finite()
            && self.sse.is_finite()
            && self.slopes.iter().all(|v| v.is_finite() && *v >= 0.0)
    }
    pub(super) fn is_ready(&self) -> bool {
        self.is_finite() && self.slopes.iter().any(|v| *v > 0.0)
    }
}

pub(super) fn kkt_valid(
    gram: &[[f64; P]; P],
    rhs: &[f64; P],
    coefficients: &[f64; P],
    p: usize,
) -> bool {
    let tolerance = rhs[..p].iter().map(|v| v.abs()).fold(1.0, f64::max) * 1e-8;
    (0..p).all(|i| {
        let gradient = rhs[i] - (0..p).map(|j| gram[i][j] * coefficients[j]).sum::<f64>();
        coefficients[i].is_finite()
            && coefficients[i] >= 0.0
            && gradient.is_finite()
            && if coefficients[i] > 0.0 {
                gradient.abs() <= tolerance
            } else {
                gradient <= tolerance
            }
    })
}

/// Two coordinate sweeps over training-only quantile and geometric candidates.
/// The previous epoch's knots have already been mapped to the new normalization.
pub(super) fn learn(data: &[Row], k: usize, previous: Option<[Vec<f64>; 2]>) -> Option<Model> {
    if data.is_empty() {
        return None;
    }
    let mut knots = init_knots(data, k);
    let mut best = fit(data, &knots)?;
    if let Some(old) = previous {
        if (0..2).all(|a| old[a].len() == knots[a].len()) {
            if let Some(trial) = fit(data, &old) {
                if trial.sse < best.sse {
                    knots = old;
                    best = trial;
                }
            }
        }
    }
    let grids = [candidates(data, 0), candidates(data, 1)];
    for _ in 0..2 {
        for axis in 0..2 {
            for j in 0..knots[axis].len() {
                let low = if j == 0 { 0.0 } else { knots[axis][j - 1] };
                let high = if j + 1 == knots[axis].len() {
                    data.iter().map(|r| r[axis]).fold(0.0, f64::max)
                } else {
                    knots[axis][j + 1]
                };
                let mut chosen = knots[axis][j];
                for &candidate in &grids[axis] {
                    if candidate <= low + 1e-10 || candidate >= high - 1e-10 {
                        continue;
                    }
                    knots[axis][j] = candidate;
                    if let Some(trial) = fit(data, &knots) {
                        if trial.sse < best.sse - 1e-12 {
                            best = trial;
                            chosen = candidate;
                        }
                    }
                }
                knots[axis][j] = chosen;
            }
        }
    }
    Some(best)
}
