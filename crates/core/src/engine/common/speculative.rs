// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project.
// Modified Rust simulation of vLLM b3563fc65ae0f5359802593d78e7ea097e1fed31
// scheduling contracts; immutable sources and license: THIRD_PARTY_NOTICES.md.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DEFAULT_CONDITIONAL_ACCEPT_RATES: [f64; 5] = [0.85, 0.3, 0.0, 0.0, 0.0];

/// Parse and normalize MTP/EAGLE conditional draft-token acceptance rates.
///
/// The result always has exactly `nextn` entries. Missing trailing rates are
/// zero-filled and extra entries are ignored, matching scheduler semantics.
pub fn normalize_conditional_accept_rates(
    nextn: usize,
    rates: Option<&str>,
) -> anyhow::Result<Vec<f64>> {
    anyhow::ensure!(
        (1..=5).contains(&nextn),
        "aic_nextn must be in 1..=5, got {nextn}"
    );
    let mut parsed = match rates.map(str::trim).filter(|rates| !rates.is_empty()) {
        Some(rates) => rates
            .split(',')
            .map(|rate| {
                rate.trim().parse::<f64>().map_err(|error| {
                    anyhow::anyhow!("invalid aic_nextn_accept_rates value {rate:?}: {error}")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?,
        None => DEFAULT_CONDITIONAL_ACCEPT_RATES.to_vec(),
    };

    for (index, rate) in parsed.iter().copied().enumerate() {
        anyhow::ensure!(
            rate.is_finite() && (0.0..=1.0).contains(&rate),
            "aic_nextn_accept_rates[{index}] must be finite and in [0, 1], got {rate}"
        );
    }

    parsed.resize(nextn, 0.0);
    Ok(parsed)
}

pub(crate) struct SpeculativeDecodeSampler {
    conditional_accept_rates: Vec<f64>,
    rng: StdRng,
}

impl SpeculativeDecodeSampler {
    pub(crate) fn new(conditional_accept_rates: Vec<f64>, seed: u64) -> Self {
        Self {
            conditional_accept_rates,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample verification acceptance including the base token. Callers clamp
    /// emitted output separately so telemetry retains pre-truncation acceptance.
    pub(crate) fn sample_accepted_tokens(&mut self) -> usize {
        self.sample_accepted_tokens_up_to(usize::MAX)
    }

    /// Verification cannot accept drafts that were excluded by the scheduler's
    /// token budget. Do not consume RNG draws for those unscheduled positions.
    pub(crate) fn sample_accepted_tokens_up_to(&mut self, max_tokens: usize) -> usize {
        debug_assert!(max_tokens > 0);
        let mut output_tokens = 1;
        for rate in self
            .conditional_accept_rates
            .iter()
            .take(max_tokens.saturating_sub(1))
        {
            if !self.rng.random_bool(*rate) {
                break;
            }
            output_tokens += 1;
        }
        output_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_one_rates_are_exact() {
        let mut zero = SpeculativeDecodeSampler::new(vec![0.0, 1.0], 42);
        assert_eq!(zero.sample_accepted_tokens(), 1);

        let mut one = SpeculativeDecodeSampler::new(vec![1.0, 1.0], 42);
        assert_eq!(one.sample_accepted_tokens(), 3);
    }

    #[test]
    fn sampling_stops_at_first_rejection() {
        let mut sampler = SpeculativeDecodeSampler::new(vec![1.0, 0.0, 1.0], 42);
        assert_eq!(sampler.sample_accepted_tokens(), 2);
    }

    #[test]
    fn unscheduled_drafts_do_not_advance_the_rng() {
        let rates = vec![0.7, 0.5, 0.2];
        let mut clipped = SpeculativeDecodeSampler::new(rates.clone(), 42);
        let mut untouched = SpeculativeDecodeSampler::new(rates, 42);
        assert_eq!(clipped.sample_accepted_tokens_up_to(1), 1);
        assert_eq!(
            clipped.sample_accepted_tokens(),
            untouched.sample_accepted_tokens()
        );
        let mut full_accept = SpeculativeDecodeSampler::new(vec![1.0; 3], 42);
        assert_eq!(full_accept.sample_accepted_tokens_up_to(2), 2);
    }

    #[test]
    fn equal_seeds_produce_equal_streams() {
        let rates = vec![0.8, 0.6, 0.4];
        let mut left = SpeculativeDecodeSampler::new(rates.clone(), 123);
        let mut right = SpeculativeDecodeSampler::new(rates, 123);
        let left_samples: Vec<_> = (0..100).map(|_| left.sample_accepted_tokens()).collect();
        let right_samples: Vec<_> = (0..100).map(|_| right.sample_accepted_tokens()).collect();
        assert_eq!(left_samples, right_samples);
    }

    #[test]
    fn worker_seed_offsets_produce_distinct_streams() {
        let rates = vec![0.8, 0.6, 0.4];
        let mut left = SpeculativeDecodeSampler::new(rates.clone(), 42);
        let mut right = SpeculativeDecodeSampler::new(rates, 43);
        let left_samples: Vec<_> = (0..100).map(|_| left.sample_accepted_tokens()).collect();
        let right_samples: Vec<_> = (0..100).map(|_| right.sample_accepted_tokens()).collect();
        assert_ne!(left_samples, right_samples);
    }

    #[test]
    fn empirical_mean_matches_conditional_expectation() {
        let rates = vec![0.7, 0.5, 0.2];
        let expected = 1.0 + 0.7 + 0.7 * 0.5 + 0.7 * 0.5 * 0.2;
        let mut sampler = SpeculativeDecodeSampler::new(rates, 42);
        let samples = 200_000;
        let mean = (0..samples)
            .map(|_| sampler.sample_accepted_tokens() as f64)
            .sum::<f64>()
            / samples as f64;
        assert!(
            (mean - expected).abs() < 0.01,
            "mean={mean}, expected={expected}"
        );
    }

    #[test]
    fn rates_are_validated_then_padded_or_truncated() {
        assert_eq!(
            normalize_conditional_accept_rates(3, Some("1, 0.5")).unwrap(),
            vec![1.0, 0.5, 0.0]
        );
        assert_eq!(
            normalize_conditional_accept_rates(2, Some("1,0.5,0.25")).unwrap(),
            vec![1.0, 0.5]
        );
        for rates in ["nan", "inf", "-0.1", "1.1", "not-a-rate"] {
            assert!(normalize_conditional_accept_rates(1, Some(rates)).is_err());
        }
    }
}
