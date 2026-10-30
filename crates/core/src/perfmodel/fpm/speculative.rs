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
        if original.num_prefill_requests != 0 {
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
        if let Some(past) = &original.past_kv_lengths {
            let mut widened = Vec::new();
            widened
                .try_reserve_exact(scheduled.num_decode_requests as usize)
                .map_err(|_| {
                    invalid("cannot allocate widened STP request features for draft_length")
                })?;
            for &length in past {
                let quotient = length / u64::from(width);
                let remainder = length % u64::from(width);
                // Distribute the integer remainder instead of dropping KV tokens.
                widened.extend((0..width).map(|i| quotient + u64::from(u64::from(i) < remainder)));
            }
            if !widened.is_empty() {
                let mean = widened.iter().map(|&v| v as f64).sum::<f64>() / widened.len() as f64;
                scheduled.var_decode_kv_tokens = widened
                    .iter()
                    .map(|&v| (v as f64 - mean).powi(2))
                    .sum::<f64>()
                    / widened.len() as f64;
            }
            scheduled.extend_lengths = Some(vec![1; widened.len()]);
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
