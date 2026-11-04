// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Empirical complete-SD-round estimate from an ordinary STP FPM reference.

use serde::{Deserialize, Serialize};

use super::{ForwardPassEstimate, ForwardPassMetrics, validate_forward_pass_metrics};
use crate::AicError;

/// An estimated complete speculative decode iteration, with the unmodified
/// STP lookup evidence kept separate from the empirical multiplier.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeForwardPassEstimate {
    pub latency_ms: Option<f64>,
    pub draft_length: u32,
    pub scaling_factor: f64,
    pub stp_reference: ForwardPassEstimate,
}

pub(super) fn reference_metrics(
    metrics: &[ForwardPassMetrics],
    draft_length: u32,
) -> Result<(Vec<ForwardPassMetrics>, f64), AicError> {
    let invalid = |message: &str| AicError::InvalidForwardPassMetrics(message.into());
    if draft_length == 0 {
        return Err(invalid(
            "draft_length must be positive; omit it for ordinary STP",
        ));
    }
    let width = draft_length
        .checked_add(1)
        .ok_or_else(|| invalid("draft_length + 1 exceeds the supported request count"))?;
    let mut reference = Vec::with_capacity(metrics.len());
    for rank in metrics {
        validate_forward_pass_metrics(rank)?;
        let original = &rank.scheduled_requests;
        // Token-based, like every rank dispatch: a fully prefix-cached prefill
        // computes no fresh token, so its rank is decode-only. Its prefill
        // request and KV metadata pass through unchanged.
        if original.sum_prefill_tokens > 0 {
            return Err(invalid(
                "draft_length scaling supports pure decode iterations only, not prefill or mixed work",
            ));
        }
        let mut baseline = rank.clone();
        let scheduled = &mut baseline.scheduled_requests;
        scheduled.num_decode_requests = original
            .num_decode_requests
            .checked_mul(width)
            .ok_or_else(|| invalid("(draft_length + 1) * num_decode_requests overflows"))?;
        // Widen the STP request count while retaining the exact total past KV.
        // This is an empirical reference mapping, not a physical verify graph.
        scheduled.var_decode_kv_tokens = original.var_decode_kv_tokens / f64::from(width).powi(2);
        if let (Some(extend), Some(past)) = (&original.extend_lengths, &original.past_kv_lengths) {
            // The lists are unordered. Zero-token prefill requests are the
            // ones that compute nothing, and they are carried over unchanged.
            let cached = |extend: u64| original.num_prefill_requests > 0 && extend == 0;
            if extend.iter().filter(|&&e| cached(e)).count()
                != original.num_prefill_requests as usize
            {
                return Err(invalid(
                    "request lists must give each zero-token prefill request an extend length of 0",
                ));
            }
            let count =
                u64::from(original.num_prefill_requests) + u64::from(scheduled.num_decode_requests);
            let mut widened_extend = Vec::new();
            let mut widened = Vec::new();
            widened_extend
                .try_reserve_exact(count as usize)
                .and_then(|()| widened.try_reserve_exact(count as usize))
                .map_err(|_| {
                    invalid("cannot allocate widened STP request features for draft_length")
                })?;
            let (mut decode_sum, mut decode_count) = (0.0_f64, 0_usize);
            for (&e, &length) in extend.iter().zip(past) {
                if cached(e) {
                    widened_extend.push(e);
                    widened.push(length);
                    continue;
                }
                let quotient = length / u64::from(width);
                let remainder = length % u64::from(width);
                // Distribute the integer remainder instead of dropping KV tokens.
                for i in 0..width {
                    let share = quotient + u64::from(u64::from(i) < remainder);
                    widened_extend.push(1);
                    widened.push(share);
                    decode_sum += share as f64;
                    decode_count += 1;
                }
            }
            if decode_count > 0 {
                let mean = decode_sum / decode_count as f64;
                scheduled.var_decode_kv_tokens = widened_extend
                    .iter()
                    .zip(&widened)
                    .filter(|&(&e, _)| !cached(e))
                    .map(|(_, &v)| (v as f64 - mean).powi(2))
                    .sum::<f64>()
                    / decode_count as f64;
            }
            scheduled.extend_lengths = Some(widened_extend);
            scheduled.past_kv_lengths = Some(widened);
        }
        reference.push(baseline);
    }
    Ok((reference, 1.25 + 0.08 * f64::from(draft_length)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScheduledRequestMetrics;

    #[test]
    fn preserves_exact_kv_and_input_while_widening_request_features() {
        let input = ForwardPassMetrics {
            scheduled_requests: ScheduledRequestMetrics {
                num_decode_requests: 2,
                sum_decode_kv_tokens: 23,
                extend_lengths: Some(vec![1, 1]),
                past_kv_lengths: Some(vec![10, 13]),
                var_decode_kv_tokens: 2.25,
                ..Default::default()
            },
            ..Default::default()
        };
        let (reference, factor) = reference_metrics(std::slice::from_ref(&input), 2).unwrap();
        let mapped = &reference[0].scheduled_requests;
        assert_eq!(mapped.num_decode_requests, 6);
        assert_eq!(mapped.sum_decode_kv_tokens, 23);
        assert_eq!(
            mapped.past_kv_lengths.as_deref(),
            Some([4, 3, 3, 5, 4, 4].as_slice())
        );
        assert_eq!(mapped.extend_lengths.as_deref(), Some([1; 6].as_slice()));
        assert!((factor - 1.41).abs() < 1e-12);
        assert_eq!(input.scheduled_requests.num_decode_requests, 2);
        validate_forward_pass_metrics(&reference[0]).unwrap();
    }

    #[test]
    fn a_fully_cached_prefill_is_decode_only_and_keeps_its_metadata() {
        let input = ForwardPassMetrics {
            scheduled_requests: ScheduledRequestMetrics {
                num_prefill_requests: 1,
                sum_prefill_kv_tokens: 4096,
                num_decode_requests: 2,
                sum_decode_kv_tokens: 23,
                extend_lengths: Some(vec![1, 0, 1]),
                past_kv_lengths: Some(vec![10, 4096, 13]),
                var_decode_kv_tokens: 2.25,
                ..Default::default()
            },
            ..Default::default()
        };
        let (reference, _) = reference_metrics(std::slice::from_ref(&input), 2).unwrap();
        let mapped = &reference[0].scheduled_requests;
        assert_eq!(
            (
                mapped.num_prefill_requests,
                mapped.sum_prefill_tokens,
                mapped.sum_prefill_kv_tokens
            ),
            (1, 0, 4096)
        );
        assert_eq!(mapped.num_decode_requests, 6);
        assert_eq!(mapped.sum_decode_kv_tokens, 23);
        // The cached request keeps its position; only decode entries widen.
        assert_eq!(
            mapped.extend_lengths.as_deref(),
            Some([1, 1, 1, 0, 1, 1, 1].as_slice())
        );
        assert_eq!(
            mapped.past_kv_lengths.as_deref(),
            Some([4, 3, 3, 4096, 5, 4, 4].as_slice())
        );
        // Population variance of the widened decode KV {4, 3, 3, 5, 4, 4}:
        // mean 23/6, squared deviations 3/36 + 50/36 + 49/36 over 6 = 17/36.
        assert!((mapped.var_decode_kv_tokens - 17.0 / 36.0).abs() < 1e-12);
        validate_forward_pass_metrics(&reference[0]).unwrap();

        // Aggregate-only telemetry needs no request lists.
        let mut aggregate = input.clone();
        aggregate.scheduled_requests.extend_lengths = None;
        aggregate.scheduled_requests.past_kv_lengths = None;
        let (reference, _) = reference_metrics(&[aggregate], 2).unwrap();
        assert_eq!(reference[0].scheduled_requests.num_prefill_requests, 1);
        assert_eq!(reference[0].scheduled_requests.num_decode_requests, 6);

        // Lists that cannot identify the cached request are rejected.
        let mut unmarked = input;
        unmarked.scheduled_requests.extend_lengths = Some(vec![1, 1, 1]);
        assert!(
            reference_metrics(&[unmarked], 2)
                .unwrap_err()
                .to_string()
                .contains("extend length of 0")
        );
    }

    #[test]
    fn rejects_invalid_depth_overflow_and_prefill_without_saturating() {
        let empty = ForwardPassMetrics::default();
        assert!(reference_metrics(std::slice::from_ref(&empty), 0).is_err());
        assert!(reference_metrics(&[empty], u32::MAX).is_err());
        let overflow = ForwardPassMetrics {
            scheduled_requests: ScheduledRequestMetrics {
                num_decode_requests: u32::MAX,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(reference_metrics(&[overflow], 1).is_err());
        let prefill = ForwardPassMetrics {
            scheduled_requests: ScheduledRequestMetrics {
                num_prefill_requests: 1,
                sum_prefill_tokens: 10,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            reference_metrics(&[prefill], 1)
                .unwrap_err()
                .to_string()
                .contains("pure decode")
        );
    }
}
