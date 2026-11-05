// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Original implementation of the feature formulas documented in the AISimulate
//! design at https://github.com/ai-dynamo/aisimulate/blob/6aa40ae50a75dcfd03f23d45b25a73b3a2b69b2b/python/aisimulate/docs/fpm_ml/design.md
//! No upstream implementation was copied. Canonical aggregate scores remain separate.

use super::super::estimator::RegressionFeatureAxis;
use crate::{AicError, ForwardPassMetrics};

pub(super) fn project(
    scores: [f64; 2],
    metrics: &[ForwardPassMetrics],
    axes: &[RegressionFeatureAxis],
) -> Result<[f64; 12], AicError> {
    use RegressionFeatureAxis::*;
    let invalid = |message: &str| AicError::InvalidForwardPassMetrics(message.into());
    let mut out = [0.0; 12];
    // Scalar-only projections never require or scan request lists.
    let needs_requests = axes
        .iter()
        .any(|axis| !matches!(axis, Attention | Moe | Count | LogCount | CountSquared));
    let needs_count = needs_requests
        || axes
            .iter()
            .any(|axis| matches!(axis, Count | LogCount | CountSquared));
    if !needs_count {
        for (index, axis) in axes.iter().enumerate() {
            out[index] = if *axis == Attention {
                scores[0]
            } else {
                scores[1]
            };
        }
        return Ok(out);
    }
    let (mut n, mut e, mut p, mut e2, mut p2, mut f) = (0.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut max_e, mut max_p, mut min_p) = (0.0_f64, 0.0_f64, f64::INFINITY);
    for metric in metrics {
        let scheduled = &metric.scheduled_requests;
        let count =
            u64::from(scheduled.num_prefill_requests) + u64::from(scheduled.num_decode_requests);
        if count == 0 {
            continue;
        }
        if !needs_requests {
            n += count as f64;
            continue;
        }
        let (Some(extend), Some(past)) = (&scheduled.extend_lengths, &scheduled.past_kv_lengths)
        else {
            return Err(invalid(
                "selected request features require extend_lengths and past_kv_lengths on every active rank",
            ));
        };
        if extend.len() != past.len() || extend.len() as u64 != count {
            return Err(invalid(
                "request feature list length differs from scheduled request count",
            ));
        }
        for (&extend, &past) in extend.iter().zip(past) {
            let (x, y) = (extend as f64, past as f64);
            n += 1.0;
            e += x;
            p += y;
            e2 += x * x;
            p2 += y * y;
            f += x * y + x * x / 2.0;
            max_e = max_e.max(x);
            max_p = max_p.max(y);
            min_p = min_p.min(y);
        }
    }
    if n == 0.0 {
        return Err(invalid(
            "request feature projection needs at least one scheduled request",
        ));
    }
    let cv2 = |sum: f64, squares: f64| -> Result<f64, AicError> {
        if sum == 0.0 {
            return Ok(0.0);
        }
        let ratio = n * squares / (sum * sum);
        let value = ratio - 1.0;
        if !value.is_finite() || value < -1e-10 * ratio.abs().max(1.0) {
            return Err(invalid(
                "request moments imply invalid coefficient of variation",
            ));
        }
        Ok(value.max(0.0))
    };
    for (index, axis) in axes.iter().enumerate() {
        out[index] = match axis {
            Attention => scores[0],
            Moe => scores[1],
            Count => n,
            Extend => e,
            Past => p,
            MaxExtend => max_e,
            MaxPast => max_p,
            MinPast => min_p,
            PastSquared => p2,
            AttentionPairs => f,
            CountExtend => n * e,
            LogAttentionPairs => f.ln_1p(),
            MeanExtend => e / n,
            MeanPast => p / n,
            ExtendCvSquared => cv2(e, e2)?,
            PastCvSquared => cv2(p, p2)?,
            LogCount => n.ln_1p(),
            CountSquared => n * n,
            LogPast => p.ln_1p(),
        };
    }
    if out.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(invalid(
            "selected regression features must be finite and nonnegative",
        ));
    }
    Ok(out)
}
