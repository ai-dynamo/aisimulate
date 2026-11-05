// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Removable integer-length statistics for scheduler waiting queues.
//!
//! Keep exact moments while they fit, avoiding both repeated queue walks and
//! cancellation from subtracting two rounded floating-point second moments.
//! The histogram supplies a centered, stable fallback for extreme u64 lengths.

use std::collections::BTreeMap;

use super::{ForwardPassSnapshot, WelfordAcc};

#[derive(Default)]
pub(super) struct LengthStats {
    count: u32,
    sum: u128,
    squares: Option<u128>,
    lengths: BTreeMap<u64, u32>,
}

impl LengthStats {
    pub(super) fn add(&mut self, value: u64) {
        if self.count == 0 {
            self.squares = Some(0);
        }
        self.count = self
            .count
            .checked_add(1)
            .expect("queued request count overflow");
        self.sum += u128::from(value);
        self.squares = self
            .squares
            .and_then(|sum| sum.checked_add(u128::from(value).pow(2)));
        *self.lengths.entry(value).or_default() += 1;
    }

    pub(super) fn remove(&mut self, value: u64) {
        let count = self
            .lengths
            .get_mut(&value)
            .expect("queued length must exist");
        *count -= 1;
        if *count == 0 {
            self.lengths.remove(&value);
        }
        self.count -= 1;
        self.sum -= u128::from(value);
        self.squares = self.squares.map(|sum| sum - u128::from(value).pow(2));
        if self.count == 0 {
            self.squares = Some(0);
        }
    }

    fn accumulator(&self) -> WelfordAcc {
        if self.count == 0 {
            return WelfordAcc::default();
        }
        let n = u128::from(self.count);
        let numerator = self.squares.and_then(|squares| {
            squares
                .checked_mul(n)?
                .checked_sub(self.sum.checked_mul(self.sum)?)
        });
        let variance = if let Some(numerator) = numerator {
            numerator as f64 / self.count as f64 / self.count as f64
        } else {
            // Subtract the minimum before converting integers to f64. In
            // particular, adjacent lengths above 2^53 must remain distinct.
            let origin = *self.lengths.first_key_value().expect("nonempty lengths").0;
            let mut acc = WelfordAcc::default();
            for (&length, &count) in &self.lengths {
                let value = (length - origin) as f64;
                let total = acc.count + count;
                let delta = value - acc.mean;
                acc.m2 += delta * delta * acc.count as f64 * count as f64 / total as f64;
                acc.mean += delta * count as f64 / total as f64;
                acc.count = total;
            }
            acc.variance()
        };
        WelfordAcc {
            count: self.count,
            sum: self.sum as f64,
            mean: self.sum as f64 / self.count as f64,
            m2: variance * self.count as f64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum QueuedLength {
    Prefill(u64),
    Decode(u64),
}

#[derive(Default)]
pub(super) struct QueueStats {
    prefill: LengthStats,
    decode: LengthStats,
}

impl QueueStats {
    pub(super) fn add(&mut self, value: QueuedLength) {
        match value {
            QueuedLength::Prefill(value) => self.prefill.add(value),
            QueuedLength::Decode(value) => self.decode.add(value),
        }
    }

    pub(super) fn remove(&mut self, value: QueuedLength) {
        match value {
            QueuedLength::Prefill(value) => self.prefill.remove(value),
            QueuedLength::Decode(value) => self.decode.remove(value),
        }
    }

    pub(super) fn maximum_length(&self) -> Option<u64> {
        self.prefill
            .lengths
            .last_key_value()
            .map(|(&value, _)| value)
            .into_iter()
            .chain(
                self.decode
                    .lengths
                    .last_key_value()
                    .map(|(&value, _)| value),
            )
            .max()
    }

    pub(super) fn snapshot(&self) -> QueueSnapshot {
        let decode_origin = self
            .decode
            .lengths
            .first_key_value()
            .map_or(0, |(&value, _)| value);
        let mut decode = self.decode.accumulator();
        if decode.count > 0 {
            decode.mean = (self.decode.sum - u128::from(decode_origin) * u128::from(decode.count))
                as f64
                / decode.count as f64;
        }
        QueueSnapshot {
            prefill: self.prefill.accumulator(),
            decode,
            prefill_sum: self.prefill.sum,
            decode_sum: self.decode.sum,
            decode_origin,
        }
    }
}

pub(super) struct QueueSnapshot {
    prefill: WelfordAcc,
    decode: WelfordAcc,
    prefill_sum: u128,
    decode_sum: u128,
    decode_origin: u64,
}

impl QueueSnapshot {
    pub(super) fn add_decode(&mut self, value: u64) {
        if self.decode.count == 0 {
            self.decode_origin = value;
        }
        self.decode_sum += u128::from(value);
        self.decode
            .add((i128::from(value) - i128::from(self.decode_origin)) as f64);
    }

    pub(super) fn apply(self, snapshot: &mut ForwardPassSnapshot) {
        snapshot.num_queued_prefill = self.prefill.count;
        // Retain the report's existing saturation at u64::MAX, while avoiding
        // rounding representable integer sums through f64.
        snapshot.sum_queued_prefill_tokens = self.prefill_sum.min(u128::from(u64::MAX)) as u64;
        snapshot.var_queued_prefill_length = self.prefill.variance();
        snapshot.num_queued_decode = self.decode.count;
        snapshot.sum_queued_decode_kv_tokens = self.decode_sum.min(u128::from(u64::MAX)) as u64;
        snapshot.var_queued_decode_kv_tokens = self.decode.variance();
    }
}

#[cfg(test)]
pub(super) fn assert_queue_metrics(actual: &ForwardPassSnapshot, expected: &ForwardPassSnapshot) {
    assert_eq!(actual.num_queued_prefill, expected.num_queued_prefill);
    assert_eq!(
        actual.sum_queued_prefill_tokens,
        expected.sum_queued_prefill_tokens
    );
    assert_eq!(actual.num_queued_decode, expected.num_queued_decode);
    assert_eq!(
        actual.sum_queued_decode_kv_tokens,
        expected.sum_queued_decode_kv_tokens
    );
    for (a, b) in [
        (
            actual.var_queued_prefill_length,
            expected.var_queued_prefill_length,
        ),
        (
            actual.var_queued_decode_kv_tokens,
            expected.var_queued_decode_kv_tokens,
        ),
    ] {
        assert!(
            (a - b).abs() <= 1e-7 + b.abs() * 1e-10,
            "queue variance: {a} != {b}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removal_and_reinsertion_match_a_fresh_reference() {
        let mut stats = LengthStats::default();
        let mut values = Vec::new();
        let mut rng = 19u64;
        for step in 0..10_000 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if !values.is_empty() && step % 3 == 0 {
                let index = rng as usize % values.len();
                stats.remove(values.remove(index));
            } else {
                let value = 1_000_000 + rng % 17;
                values.push(value);
                stats.add(value);
            }
            let actual = stats.accumulator();
            let mut expected = WelfordAcc::default();
            for value in &values {
                expected.add(*value as f64);
            }
            assert_eq!(actual.count, expected.count);
            assert_eq!(actual.sum, expected.sum);
            assert!((actual.variance() - expected.variance()).abs() < 1e-7);
        }
        for value in values {
            stats.remove(value);
        }
        let actual = stats.accumulator();
        assert_eq!((actual.count, actual.sum, actual.variance()), (0, 0.0, 0.0));
    }

    #[test]
    fn snapshot_combines_large_pending_lengths_without_rounding_away_variance() {
        let mut stats = QueueStats::default();
        stats.add(QueuedLength::Decode((1u64 << 54) + 1));
        let mut snapshot = stats.snapshot();
        snapshot.add_decode((1u64 << 54) + 2);
        let mut fpm = ForwardPassSnapshot::default();
        snapshot.apply(&mut fpm);
        assert_eq!(fpm.sum_queued_decode_kv_tokens, (1u64 << 55) + 3);
        assert_eq!(fpm.var_queued_decode_kv_tokens, 0.25);
        let mut empty = QueueStats::default().snapshot();
        empty.add_decode(u64::MAX - 1);
        empty.add_decode(u64::MAX);
        empty.apply(&mut fpm);
        assert_eq!(fpm.sum_queued_decode_kv_tokens, u64::MAX);
        assert_eq!(fpm.var_queued_decode_kv_tokens, 0.25);
    }

    #[test]
    fn large_adjacent_lengths_do_not_cancel_or_overflow() {
        let mut stats = LengthStats::default();
        stats.add(u64::MAX - 1);
        stats.add(u64::MAX);
        assert_eq!(stats.accumulator().variance(), 0.25);
        stats.remove(u64::MAX);
        assert_eq!(stats.accumulator().variance(), 0.0);
        stats.add(u64::MAX);
        assert_eq!(stats.accumulator().variance(), 0.25);
        stats.remove(u64::MAX - 1);
        stats.remove(u64::MAX);
        stats.add(7);
        assert_eq!(stats.accumulator().sum, 7.0);
    }
}
