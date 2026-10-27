// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Performance model for timing simulations in the mocker.
//!
//! Scheduler algorithms keep their historical polynomial fallback while
//! provider-backed timing enters through the runtime-neutral engine contract.

use anyhow::{Context, Result, ensure};
use std::sync::Arc;

use crate::engine::timing::{MixedPassMs, PrefillTotals, mean_prefill_shape};

/// Performance model for predicting prefill and decode timing
#[derive(Default)]
pub enum PerfModel {
    /// Default polynomial-based model using hardcoded formulas
    #[default]
    Polynomial,
    /// Runtime-resolved provider supplied through the neutral engine boundary.
    External {
        timing: Arc<dyn crate::engine::TimingModel>,
    },
}

impl Clone for PerfModel {
    fn clone(&self) -> Self {
        match self {
            PerfModel::Polynomial => PerfModel::Polynomial,
            PerfModel::External { timing } => PerfModel::External {
                timing: Arc::clone(timing),
            },
        }
    }
}

impl std::fmt::Debug for PerfModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PerfModel::Polynomial => write!(f, "PerfModel::Polynomial"),
            PerfModel::External { .. } => write!(f, "PerfModel::External"),
        }
    }
}

impl PerfModel {
    pub fn validate_prefill_batch(&self, requests: &[(usize, usize)]) -> Result<()> {
        if let PerfModel::External { timing } = self {
            timing
                .validate_prefill_batch(requests)
                .context("external prefill batch cannot be represented")?;
        }
        Ok(())
    }

    /// Predict the vision-encoder time in milliseconds for one batch of
    /// distinct cache-miss images, grouped by shape for the provider.
    pub fn predict_vision_time(&self, images: &[crate::engine::ImageSpec]) -> Result<f64> {
        if images.is_empty() {
            return Ok(0.0);
        }
        let PerfModel::External { timing } = self else {
            anyhow::bail!("polynomial timing does not support vision batches");
        };
        let mut shapes: Vec<crate::engine::VisionShape> = Vec::new();
        for image in images {
            match shapes
                .iter_mut()
                .find(|shape| shape.encoder == image.encoder)
            {
                Some(shape) => shape.count += 1,
                None => shapes.push(crate::engine::VisionShape {
                    encoder: image.encoder,
                    count: 1,
                }),
            }
        }
        let time = timing
            .predict_vision_ms(&shapes)
            .context("external vision prediction failed")?
            .context("timing provider does not support vision batches")?;
        ensure!(
            time.is_finite() && time >= 0.0,
            "vision timing provider returned invalid duration {time}ms"
        );
        Ok(time)
    }

    /// Predict prefill time in milliseconds from batch totals: `new_tokens`
    /// computed by the batch and `prefix_tokens` of cached or earlier-chunk
    /// KV that its requests read.
    ///
    /// The polynomial fallback keeps its per-request mean. An injected
    /// provider receives the exact totals.
    pub fn predict_prefill_time(
        &self,
        batch_size: usize,
        new_tokens: usize,
        prefix_tokens: usize,
    ) -> Result<f64> {
        if batch_size == 0 || new_tokens == 0 {
            return Ok(0.0);
        }
        let time = match self {
            PerfModel::Polynomial => {
                let (isl, prefix) = mean_prefill_shape(batch_size, new_tokens, prefix_tokens);
                polynomial_prefill_time(batch_size, isl - prefix)
            }
            PerfModel::External { timing } => timing
                .predict_prefill_totals_ms(batch_size, new_tokens, prefix_tokens)
                .context("external prefill prediction failed")?,
        };
        // Validate before the floor: f64::max would hide NaN or -infinity.
        ensure!(
            time.is_finite(),
            "prefill timing provider returned non-finite duration {time}ms"
        );
        Ok(time.max(0.0))
    }

    /// Predict decode time in milliseconds.
    ///
    /// `active_kv_tokens` is the sum of logical context lengths in the scheduled
    /// batch, not the number of distinct physically resident tokens.
    ///
    /// Callers always pass all parameters; each variant uses what it needs:
    /// - Polynomial uses logical active KV tokens relative to total capacity,
    ///   clamped to full utilization.
    /// - An injected provider receives the full scheduler-local context.
    pub fn predict_decode_time(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        context_length: usize,
        total_kv_tokens: usize,
    ) -> Result<f64> {
        if batch_size == 0 {
            return Ok(0.0);
        }
        let time = match self {
            PerfModel::Polynomial => polynomial_decode_time(active_kv_tokens, total_kv_tokens),
            PerfModel::External { timing } => timing
                .predict_decode_ms(
                    batch_size,
                    active_kv_tokens,
                    context_length,
                    total_kv_tokens,
                )
                .context("external decode prediction failed")?,
        };
        // Reject non-finite provider output before it reaches the scheduler.
        ensure!(
            time.is_finite(),
            "decode timing provider returned non-finite duration {time}ms"
        );
        tracing::trace!(
            "Decode time prediction: batch_size={batch_size}, active_kv_tokens={active_kv_tokens}, context_length={context_length}, time={time:.2}ms"
        );
        // The polynomial applies its own historical floor. External measurements
        // can legitimately be below one millisecond and must retain that value.
        Ok(time)
    }

    /// Predict one forward pass, in milliseconds, that computes `prefill`
    /// together with a decode batch. A provider that prices such a pass as one
    /// forward returns its prefill and decode parts. Other providers and the
    /// polynomial fallback return `None`; callers then price the prefill with
    /// [`Self::predict_prefill_time`] and a separate decode pass with
    /// [`Self::predict_decode_time`].
    pub fn predict_mixed_pass_time(
        &self,
        prefill: PrefillTotals,
        batch_size: usize,
        active_kv_tokens: usize,
        context_length: usize,
        total_kv_tokens: usize,
    ) -> Result<Option<MixedPassMs>> {
        let PerfModel::External { timing } = self else {
            return Ok(None);
        };
        if batch_size == 0 || prefill.batch_size == 0 || prefill.new_tokens == 0 {
            return Ok(None);
        }
        let Some(pass) = timing
            .predict_mixed_pass_ms(
                prefill,
                batch_size,
                active_kv_tokens,
                context_length,
                total_kv_tokens,
            )
            .context("external mixed pass prediction failed")?
        else {
            return Ok(None);
        };
        ensure!(
            [pass.prefill_ms, pass.decode_ms]
                .iter()
                .all(|ms| ms.is_finite() && *ms >= 0.0),
            "mixed pass timing provider returned invalid durations {pass:?}"
        );
        Ok(Some(pass))
    }
}

pub(crate) fn polynomial_prefill_time(batch_size: usize, new_tokens_per_request: usize) -> f64 {
    if batch_size == 0 || new_tokens_per_request == 0 {
        return 0.0;
    }
    // Total tokens across the batch — GPU processes them in parallel.
    let tokens = (batch_size * new_tokens_per_request) as f64;
    4.209989e-07 * tokens.powi(2) + 1.518344e-02 * tokens + 1.650142e+01
}

pub(crate) fn polynomial_decode_time(active_kv_tokens: usize, total_kv_tokens: usize) -> f64 {
    let active_perc = if total_kv_tokens > 0 {
        (active_kv_tokens as f64 / total_kv_tokens as f64).min(1.0)
    } else {
        tracing::warn!("Total KV tokens is 0, using 1.0 as capacity");
        1.0
    };
    (-25.74 * active_perc.powi(2) + 54.01 * active_perc + 5.74).max(1.0)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{PerfModel, polynomial_decode_time};
    use crate::engine::{MixedPassMs, PrefillTotals, TimingModel};

    struct EchoBatchTiming;

    impl TimingModel for EchoBatchTiming {
        fn predict_prefill_ms(
            &self,
            batch_size: usize,
            _mean_isl: usize,
            _mean_prefix: usize,
        ) -> anyhow::Result<f64> {
            Ok(batch_size as f64)
        }

        fn predict_decode_ms(
            &self,
            batch_size: usize,
            _active_kv_tokens: usize,
            _mean_context_length: usize,
            _total_kv_tokens: usize,
        ) -> anyhow::Result<f64> {
            Ok(batch_size as f64)
        }
    }

    struct FailingTiming;

    impl TimingModel for FailingTiming {
        fn predict_prefill_ms(
            &self,
            _batch_size: usize,
            _mean_isl: usize,
            _mean_prefix: usize,
        ) -> anyhow::Result<f64> {
            anyhow::bail!("missing prefill point")
        }

        fn predict_decode_ms(
            &self,
            _batch_size: usize,
            _active_kv_tokens: usize,
            _mean_context_length: usize,
            _total_kv_tokens: usize,
        ) -> anyhow::Result<f64> {
            anyhow::bail!("missing decode point")
        }
    }

    #[test]
    fn fully_cached_prompt_skips_prefill() {
        assert_eq!(
            PerfModel::default()
                .predict_prefill_time(1, 0, 128)
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn external_mixed_pass_keeps_valid_parts_and_rejects_invalid_ones() {
        struct FixedPass(f64, f64);
        impl TimingModel for FixedPass {
            fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
                Ok(1.0)
            }

            fn predict_decode_ms(
                &self,
                _: usize,
                _: usize,
                _: usize,
                _: usize,
            ) -> anyhow::Result<f64> {
                Ok(1.0)
            }

            fn predict_mixed_pass_ms(
                &self,
                _: PrefillTotals,
                _: usize,
                _: usize,
                _: usize,
                _: usize,
            ) -> anyhow::Result<Option<MixedPassMs>> {
                Ok(Some(MixedPassMs {
                    prefill_ms: self.0,
                    decode_ms: self.1,
                }))
            }
        }
        let prefill = PrefillTotals {
            batch_size: 1,
            new_tokens: 8,
            prefix_tokens: 0,
        };
        let pass = |prefill_ms, decode_ms| {
            PerfModel::External {
                timing: Arc::new(FixedPass(prefill_ms, decode_ms)),
            }
            .predict_mixed_pass_time(prefill, 1, 9, 9, 1024)
        };
        // A decode share can legitimately be zero.
        assert_eq!(
            pass(2.0, 0.0).unwrap(),
            Some(MixedPassMs {
                prefill_ms: 2.0,
                decode_ms: 0.0
            })
        );
        for (prefill_ms, decode_ms) in [(f64::NAN, 1.0), (1.0, -0.5), (f64::INFINITY, 1.0)] {
            assert!(pass(prefill_ms, decode_ms).is_err());
        }
        // The polynomial fallback has no one-forward composition.
        assert_eq!(
            PerfModel::default()
                .predict_mixed_pass_time(prefill, 1, 9, 9, 1024)
                .unwrap(),
            None
        );
    }

    #[test]
    fn external_provider_receives_scheduler_local_batch() {
        let model = PerfModel::External {
            timing: Arc::new(EchoBatchTiming),
        };
        assert_eq!(model.predict_prefill_time(7, 128, 0).unwrap(), 7.0);
        assert_eq!(model.predict_decode_time(9, 0, 128, 0).unwrap(), 9.0);
    }

    #[test]
    fn external_decode_preserves_finite_values_and_rejects_non_finite() {
        struct FixedDecode(f64);
        impl TimingModel for FixedDecode {
            fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
                Ok(0.0)
            }
            fn predict_decode_ms(
                &self,
                _: usize,
                _: usize,
                _: usize,
                _: usize,
            ) -> anyhow::Result<f64> {
                Ok(self.0)
            }
        }
        // Finite values retain the provider's submillisecond resolution;
        // modeled_duration_ms still rejects negative durations downstream.
        for value in [0.6, 0.0, -1.0] {
            let model = PerfModel::External {
                timing: Arc::new(FixedDecode(value)),
            };
            assert_eq!(model.predict_decode_time(1, 128, 128, 1024).unwrap(), value);
        }
        // Main's finite-time boundary must reject NaN and both infinities before
        // any duration normalization or scheduling can hide invalid output.
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let model = PerfModel::External {
                timing: Arc::new(FixedDecode(value)),
            };
            let error = model.predict_decode_time(1, 128, 128, 1024).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("decode timing provider returned non-finite duration")
            );
        }
    }

    #[test]
    fn external_prefill_errors_propagate_with_context() {
        let error = PerfModel::External {
            timing: Arc::new(FailingTiming),
        }
        .predict_prefill_time(2, 128, 32)
        .unwrap_err();
        assert_eq!(error.to_string(), "external prefill prediction failed");
        assert_eq!(error.root_cause().to_string(), "missing prefill point");
    }

    #[test]
    fn external_decode_errors_propagate_with_context() {
        let error = PerfModel::External {
            timing: Arc::new(FailingTiming),
        }
        .predict_decode_time(2, 64, 128, 1024)
        .unwrap_err();
        assert_eq!(error.to_string(), "external decode prediction failed");
        assert_eq!(error.root_cause().to_string(), "missing decode point");
    }

    #[test]
    fn polynomial_decode_utilization_is_clamped_to_capacity() {
        assert_eq!(
            polynomial_decode_time(2_048, 1_024),
            polynomial_decode_time(1_024, 1_024)
        );
    }
}
