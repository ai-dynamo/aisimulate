// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, bail};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::types::ArrivalSpec;

impl ArrivalSpec {
    pub fn timestamps(&self, request_count: usize, seed: u64) -> Result<Vec<f64>> {
        let mean_gap_ms = self.mean_gap_ms()?;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut timestamps = Vec::with_capacity(request_count);
        let mut next_arrival_ms = 0.0;

        for request_idx in 0..request_count {
            if request_idx > 0 {
                let gap_ms = self.sample_gap_ms(mean_gap_ms, &mut rng)?;
                // A finite `mean_gap_ms` is not sufficient: the gamma arm divides
                // it by `smoothness`, so a tiny smoothness overflows the scale and
                // `inf * 0.0` in the shape<1 branch produces a NaN gap.
                if !gap_ms.is_finite() {
                    bail!("arrival gap must be a finite number, got {gap_ms}");
                }
                next_arrival_ms += gap_ms;
            }
            if !next_arrival_ms.is_finite() {
                bail!("arrival timestamp must be a finite number, got {next_arrival_ms}");
            }
            timestamps.push(next_arrival_ms);
        }

        Ok(timestamps)
    }

    fn mean_gap_ms(&self) -> Result<f64> {
        match self {
            Self::Burst => Ok(0.0),
            Self::ConstantQps { qps } | Self::PoissonQps { qps } | Self::GammaQps { qps, .. } => {
                if !qps.is_finite() || *qps <= 0.0 {
                    bail!("qps must be a finite positive number, got {qps}");
                }
                // A denormal qps is finite and positive but its reciprocal is not.
                let mean_gap_ms = 1000.0 / qps;
                if !mean_gap_ms.is_finite() {
                    bail!("qps {qps} is too small to produce a finite mean arrival gap");
                }
                Ok(mean_gap_ms)
            }
        }
    }

    fn sample_gap_ms(&self, mean_gap_ms: f64, rng: &mut StdRng) -> Result<f64> {
        match self {
            Self::Burst => Ok(0.0),
            Self::ConstantQps { .. } => Ok(mean_gap_ms),
            Self::PoissonQps { .. } => Ok(sample_exponential_ms(mean_gap_ms, rng)),
            Self::GammaQps { smoothness, .. } => {
                if !smoothness.is_finite() || *smoothness <= 0.0 {
                    bail!("gamma smoothness must be a finite positive number, got {smoothness}");
                }
                Ok(sample_gamma_ms(*smoothness, mean_gap_ms / smoothness, rng))
            }
        }
    }
}

fn sample_exponential_ms(mean_ms: f64, rng: &mut StdRng) -> f64 {
    if mean_ms == 0.0 {
        return 0.0;
    }
    let u = (1.0 - rng.random::<f64>()).clamp(f64::MIN_POSITIVE, 1.0);
    -mean_ms * u.ln()
}

fn sample_gamma_ms(shape: f64, scale: f64, rng: &mut StdRng) -> f64 {
    if scale == 0.0 {
        return 0.0;
    }
    if shape < 1.0 {
        let u = (1.0 - rng.random::<f64>()).clamp(f64::MIN_POSITIVE, 1.0);
        return sample_gamma_ms(shape + 1.0, scale, rng) * u.powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = (1.0 / (9.0 * d)).sqrt();
    loop {
        let u1 = (1.0 - rng.random::<f64>()).clamp(f64::MIN_POSITIVE, 1.0);
        let u2 = rng.random::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        let v = (1.0 + c * z).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = rng.random::<f64>();
        if u < 1.0 - 0.0331 * z.powi(4) {
            return d * v * scale;
        }
        if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
            return d * v * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gap_moments(timestamps: &[f64]) -> (f64, f64) {
        let gaps = timestamps
            .windows(2)
            .map(|values| values[1] - values[0])
            .collect::<Vec<_>>();
        assert!(gaps.iter().all(|gap| *gap > 0.0));

        let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let variance = gaps
            .iter()
            .map(|gap| {
                let delta = gap - mean;
                delta * delta
            })
            .sum::<f64>()
            / gaps.len() as f64;
        (mean, variance.sqrt())
    }

    #[test]
    fn fixed_schedule_has_exact_cadence() {
        assert_eq!(ArrivalSpec::Burst.timestamps(4, 17).unwrap(), vec![0.0; 4]);
        let fixed = ArrivalSpec::ConstantQps { qps: 10.0 }
            .timestamps(4, 17)
            .unwrap();
        assert_eq!(fixed, vec![0.0, 100.0, 200.0, 300.0]);
    }

    #[test]
    fn poisson_schedule_is_seeded_and_matches_exponential_moments() {
        let poisson = ArrivalSpec::PoissonQps { qps: 10.0 }
            .timestamps(100_001, 17)
            .unwrap();
        let repeated = ArrivalSpec::PoissonQps { qps: 10.0 }
            .timestamps(100_001, 17)
            .unwrap();
        assert_eq!(poisson, repeated);

        let (mean_gap_ms, stddev_gap_ms) = gap_moments(&poisson);
        assert!(
            (mean_gap_ms - 100.0).abs() < 1.0,
            "expected a 100ms mean gap, got {mean_gap_ms}"
        );
        assert!(
            (stddev_gap_ms - 100.0).abs() < 2.0,
            "expected a 100ms gap standard deviation, got {stddev_gap_ms}"
        );
    }

    #[test]
    fn gamma_schedule_is_seeded_and_matches_requested_moments() {
        let spec = ArrivalSpec::GammaQps {
            qps: 20.0,
            smoothness: 4.0,
        };
        let timestamps = spec.timestamps(100_001, 23).unwrap();
        assert_eq!(timestamps, spec.timestamps(100_001, 23).unwrap());

        let (mean_gap_ms, stddev_gap_ms) = gap_moments(&timestamps);
        assert!(
            (mean_gap_ms - 50.0).abs() < 0.5,
            "expected a 50ms mean gap, got {mean_gap_ms}"
        );
        assert!(
            (stddev_gap_ms - 25.0).abs() < 0.5,
            "expected a 25ms gap standard deviation, got {stddev_gap_ms}"
        );
    }

    #[test]
    fn arrival_parameters_are_validated_before_sampling() {
        for qps in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                ArrivalSpec::PoissonQps { qps }
                    .timestamps(2, 42)
                    .unwrap_err()
                    .to_string()
                    .contains("qps must be")
            );
        }
        for smoothness in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                ArrivalSpec::GammaQps {
                    qps: 1.0,
                    smoothness,
                }
                .timestamps(2, 42)
                .unwrap_err()
                .to_string()
                .contains("gamma smoothness")
            );
        }
    }

    /// Denormal parameters pass the finite-and-positive gates but their
    /// reciprocals do not, so the emitted schedule was `[0.0, inf, inf, ..]`
    /// (constant/poisson) or all-NaN gaps (gamma, via `inf * 0.0`).
    #[test]
    fn denormal_arrival_parameters_cannot_emit_non_finite_timestamps() {
        let error = ArrivalSpec::ConstantQps { qps: 1e-320 }
            .timestamps(3, 42)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("too small"),
            "a denormal qps must be refused, got: {error}"
        );

        let error = ArrivalSpec::GammaQps {
            qps: 10.0,
            smoothness: 1e-320,
        }
        .timestamps(3, 42)
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("finite"),
            "a denormal smoothness must be refused, got: {error}"
        );

        for timestamp in ArrivalSpec::GammaQps {
            qps: 10.0,
            smoothness: 0.5,
        }
        .timestamps(64, 42)
        .expect("ordinary gamma parameters must remain accepted")
        {
            assert!(timestamp.is_finite(), "emitted {timestamp}");
        }
    }
}
