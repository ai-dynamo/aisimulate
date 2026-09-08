// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-neutral forward-pass timing models.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::engine::common::perf_model::{polynomial_decode_time, polynomial_prefill_time};

/// Provenance reported by a timing provider for modeled latency and energy.
///
/// The known variants mirror the native performance-model sources. `Other`
/// deliberately preserves an unfamiliar provider tag instead of discarding
/// provenance at the runtime boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimingEvidenceSource {
    Silicon,
    Empirical,
    Sol,
    Estimated,
    Mixed,
    Other(String),
}

impl TimingEvidenceSource {
    pub fn from_provider(source: impl Into<String>) -> Self {
        let source = source.into();
        match source.as_str() {
            "silicon" => Self::Silicon,
            "empirical" => Self::Empirical,
            "sol" => Self::Sol,
            "estimated" => Self::Estimated,
            "mixed" => Self::Mixed,
            _ => Self::Other(source),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Silicon => "silicon",
            Self::Empirical => "empirical",
            Self::Sol => "sol",
            Self::Estimated => "estimated",
            Self::Mixed => "mixed",
            Self::Other(source) => source,
        }
    }

    fn merge(self, other: Self) -> Self {
        if self == other { self } else { Self::Mixed }
    }
}

/// Evidence for one name-folded operation evaluated by a timing provider.
///
/// A missing `energy_wms` is intentional: latency-only and uncovered results
/// must not be interpreted as zero-power measurements. `covered_latency_ms`
/// is kept explicitly so later consumers can compute latency-weighted power
/// coverage after composing or scaling phases.
#[derive(Debug, Clone, PartialEq)]
pub struct TimingOperationEvidence {
    pub name: String,
    pub energy_wms: Option<f64>,
    pub latency_ms: f64,
    pub covered_latency_ms: f64,
    pub source: TimingEvidenceSource,
}

impl TimingOperationEvidence {
    pub fn new(
        name: impl Into<String>,
        latency_ms: f64,
        energy_wms: Option<f64>,
        source: TimingEvidenceSource,
    ) -> Result<Self> {
        let covered_latency_ms = energy_wms
            .is_some_and(|energy| energy > 0.0)
            .then_some(latency_ms)
            .unwrap_or(0.0);
        Self {
            name: name.into(),
            energy_wms,
            latency_ms,
            covered_latency_ms,
            source,
        }
        .canonicalized()
    }

    fn canonicalized(mut self) -> Result<Self> {
        ensure!(
            !self.name.is_empty(),
            "timing evidence operation name cannot be empty"
        );
        ensure!(
            self.latency_ms.is_finite() && self.latency_ms >= 0.0,
            "timing evidence operation {:?} returned invalid latency {}ms",
            self.name,
            self.latency_ms
        );
        ensure!(
            self.energy_wms
                .is_none_or(|energy| energy.is_finite() && energy >= 0.0),
            "timing evidence operation {:?} returned invalid energy {:?}W-ms",
            self.name,
            self.energy_wms
        );
        ensure!(
            self.covered_latency_ms.is_finite()
                && self.covered_latency_ms >= 0.0
                && self.covered_latency_ms <= self.latency_ms,
            "timing evidence operation {:?} returned invalid covered latency {}ms for {}ms total",
            self.name,
            self.covered_latency_ms,
            self.latency_ms
        );
        self.energy_wms = self.energy_wms.filter(|energy| *energy > 0.0);
        if self.energy_wms.is_none() {
            self.covered_latency_ms = 0.0;
        }
        Ok(self)
    }

    fn accumulate(&mut self, other: Self) -> Result<()> {
        let mut combined = self.clone().canonicalized()?;
        let other = other.canonicalized()?;
        combined.latency_ms += other.latency_ms;
        combined.covered_latency_ms += other.covered_latency_ms;
        combined.energy_wms = match (combined.energy_wms, other.energy_wms) {
            (Some(left), Some(right)) => Some(left + right),
            (Some(energy), None) | (None, Some(energy)) => Some(energy),
            (None, None) => None,
        };
        combined.source = combined.source.merge(other.source);
        *self = combined.canonicalized()?;
        Ok(())
    }
}

/// Typed operation evidence and totals for one forward-pass phase.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TimingPhaseEvidence {
    pub energy_wms: Option<f64>,
    pub latency_ms: f64,
    pub covered_latency_ms: f64,
    pub source: Option<TimingEvidenceSource>,
    pub operations: Vec<TimingOperationEvidence>,
}

impl TimingPhaseEvidence {
    /// Build phase totals from operation evidence, rejecting invalid public
    /// field values and canonicalizing uncovered latency before aggregation.
    pub fn try_from_operations(operations: Vec<TimingOperationEvidence>) -> Result<Self> {
        let mut phase = Self::default();
        for operation in operations {
            phase.accumulate_operation(operation.canonicalized()?)?;
        }
        phase.canonicalized()
    }

    /// Build phase totals from valid operation evidence.
    ///
    /// Call [`Self::try_from_operations`] when accepting evidence assembled
    /// through the public fields so invalid values can be handled as errors.
    #[track_caller]
    pub fn from_operations(operations: Vec<TimingOperationEvidence>) -> Self {
        Self::try_from_operations(operations).expect("invalid timing operation evidence")
    }

    pub fn coverage(&self) -> f64 {
        let has_valid_energy = self
            .energy_wms
            .is_some_and(|energy| energy.is_finite() && energy > 0.0);
        if has_valid_energy
            && self.latency_ms.is_finite()
            && self.latency_ms > 0.0
            && self.covered_latency_ms.is_finite()
            && self.covered_latency_ms >= 0.0
            && self.covered_latency_ms <= self.latency_ms
        {
            (self.covered_latency_ms / self.latency_ms).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Fallibly add another phase after validating and canonicalizing both
    /// operands. The update is atomic when validation fails.
    pub fn try_accumulate(&mut self, other: Self) -> Result<()> {
        let mut combined = self.clone().canonicalized()?;
        let other = other.canonicalized()?;
        combined.latency_ms += other.latency_ms;
        combined.covered_latency_ms += other.covered_latency_ms;
        combined.energy_wms = match (combined.energy_wms, other.energy_wms) {
            (Some(left), Some(right)) => Some(left + right),
            (Some(energy), None) | (None, Some(energy)) => Some(energy),
            (None, None) => None,
        };
        combined.source = match (combined.source.take(), other.source) {
            (Some(left), Some(right)) => Some(left.merge(right)),
            (Some(source), None) | (None, Some(source)) => Some(source),
            (None, None) => None,
        };
        for operation in other.operations {
            combined.merge_operation(operation)?;
        }
        *self = combined.canonicalized()?;
        Ok(())
    }

    /// Add another phase that is already known to contain valid evidence.
    ///
    /// Call [`Self::try_accumulate`] at public or provider boundaries.
    #[track_caller]
    pub fn accumulate(&mut self, other: Self) {
        self.try_accumulate(other)
            .expect("invalid timing phase evidence");
    }

    fn canonicalized(mut self) -> Result<Self> {
        ensure!(
            self.latency_ms.is_finite() && self.latency_ms >= 0.0,
            "timing phase evidence returned invalid latency {}ms",
            self.latency_ms
        );
        ensure!(
            self.energy_wms
                .is_none_or(|energy| energy.is_finite() && energy >= 0.0),
            "timing phase evidence returned invalid energy {:?}W-ms",
            self.energy_wms
        );
        ensure!(
            self.covered_latency_ms.is_finite()
                && self.covered_latency_ms >= 0.0
                && self.covered_latency_ms <= self.latency_ms,
            "timing phase evidence returned invalid covered latency {}ms for {}ms total",
            self.covered_latency_ms,
            self.latency_ms
        );
        self.energy_wms = self.energy_wms.filter(|energy| *energy > 0.0);
        if self.energy_wms.is_none() {
            self.covered_latency_ms = 0.0;
        }
        self.operations = self
            .operations
            .into_iter()
            .map(TimingOperationEvidence::canonicalized)
            .collect::<Result<Vec<_>>>()?;
        Ok(self)
    }

    fn accumulate_operation(&mut self, operation: TimingOperationEvidence) -> Result<()> {
        self.latency_ms += operation.latency_ms;
        self.covered_latency_ms += operation.covered_latency_ms;
        self.energy_wms = match (self.energy_wms, operation.energy_wms) {
            (Some(left), Some(right)) => Some(left + right),
            (Some(energy), None) | (None, Some(energy)) => Some(energy),
            (None, None) => None,
        };
        self.source = Some(match self.source.take() {
            Some(source) => source.merge(operation.source.clone()),
            None => operation.source.clone(),
        });
        self.merge_operation(operation)
    }

    fn merge_operation(&mut self, operation: TimingOperationEvidence) -> Result<()> {
        if let Some(existing) = self
            .operations
            .iter_mut()
            .find(|existing| existing.name == operation.name)
        {
            existing.accumulate(operation)?;
        } else {
            self.operations.push(operation);
        }
        Ok(())
    }
}

/// Provider-owned evidence accumulated across the predictions in one replay.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TimingEvidenceSummary {
    pub prefill: TimingPhaseEvidence,
    pub decode: TimingPhaseEvidence,
}

/// Serializable timing-provider selection.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum TimingModelConfig {
    /// Current-main polynomial fallback.
    #[default]
    Polynomial,
    /// Deterministic constant latency, primarily useful for parity fixtures.
    Fixed { prefill_ms: f64, decode_ms: f64 },
    /// Process-local provider loaded by a Runner or binding.
    External {
        provider: String,
        #[serde(default)]
        config: Value,
    },
}

/// Runtime latency model injected at the engine boundary.
///
/// Implementations may call AIC, interpolate profiler data, or use another
/// provider without adding that dependency to `aisimulate-core`.
pub trait TimingModel: Send + Sync {
    /// Predict one prefill batch's latency in milliseconds.
    fn predict_prefill_ms(
        &self,
        batch_size: usize,
        mean_isl: usize,
        mean_prefix: usize,
    ) -> Result<f64>;

    /// Predict one decode batch's latency in milliseconds.
    fn predict_decode_ms(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        mean_context_length: usize,
        total_kv_tokens: usize,
    ) -> Result<f64>;

    /// Return typed evidence accumulated by this provider so far.
    ///
    /// Latency-only providers retain the default `None`. In particular, the
    /// built-in polynomial/fixed models and whole-model FPM timing do not
    /// fabricate energy or coverage values.
    fn evidence_summary(&self) -> Option<TimingEvidenceSummary> {
        None
    }
}

struct PolynomialTimingModel;

impl TimingModel for PolynomialTimingModel {
    fn predict_prefill_ms(
        &self,
        batch_size: usize,
        mean_isl: usize,
        mean_prefix: usize,
    ) -> Result<f64> {
        Ok(polynomial_prefill_time(
            batch_size,
            mean_isl.saturating_sub(mean_prefix),
        ))
    }

    fn predict_decode_ms(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        _mean_context_length: usize,
        total_kv_tokens: usize,
    ) -> Result<f64> {
        if batch_size == 0 {
            return Ok(0.0);
        }
        Ok(polynomial_decode_time(active_kv_tokens, total_kv_tokens))
    }
}

struct FixedTimingModel {
    prefill_ms: f64,
    decode_ms: f64,
}

impl TimingModel for FixedTimingModel {
    fn predict_prefill_ms(
        &self,
        batch_size: usize,
        _mean_isl: usize,
        _mean_prefix: usize,
    ) -> Result<f64> {
        Ok(if batch_size == 0 {
            0.0
        } else {
            self.prefill_ms
        })
    }

    fn predict_decode_ms(
        &self,
        batch_size: usize,
        _active_kv_tokens: usize,
        _mean_context_length: usize,
        _total_kv_tokens: usize,
    ) -> Result<f64> {
        Ok(if batch_size == 0 { 0.0 } else { self.decode_ms })
    }
}

pub(crate) fn built_in_timing_model(config: &TimingModelConfig) -> Result<Arc<dyn TimingModel>> {
    match config {
        TimingModelConfig::Polynomial => Ok(Arc::new(PolynomialTimingModel)),
        TimingModelConfig::Fixed {
            prefill_ms,
            decode_ms,
        } => Ok(Arc::new(FixedTimingModel {
            prefill_ms: *prefill_ms,
            decode_ms: *decode_ms,
        })),
        TimingModelConfig::External { provider, .. } => {
            bail!("timing provider '{provider}' requires EngineFactory::with_timing_model")
        }
    }
}

pub(crate) fn modeled_duration_ms(raw_ms: f64, speedup_ratio: f64) -> Result<f64> {
    ensure!(
        raw_ms.is_finite() && raw_ms >= 0.0,
        "timing provider returned invalid duration {raw_ms}ms"
    );
    ensure!(
        speedup_ratio.is_finite() && speedup_ratio >= 0.0,
        "modeled speedup ratio must be finite and non-negative, got {speedup_ratio}"
    );
    let unscaled = Duration::try_from_secs_f64(raw_ms / 1_000.0)
        .map_err(|error| anyhow::anyhow!("timing duration {raw_ms}ms is out of range: {error}"))?;
    let modeled = if speedup_ratio > 0.0 && unscaled > Duration::ZERO {
        Duration::try_from_secs_f64(unscaled.as_secs_f64() / speedup_ratio).map_err(|error| {
            anyhow::anyhow!(
                "scaled timing duration is out of range for speedup {speedup_ratio}: {error}"
            )
        })?
    } else {
        unscaled
    };
    Ok(modeled.as_secs_f64() * 1_000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modeled_duration_applies_speedup_and_zero_means_unscaled() {
        assert_eq!(modeled_duration_ms(12.0, 3.0).unwrap(), 4.0);
        assert_eq!(modeled_duration_ms(12.0, 0.0).unwrap(), 12.0);
        assert_eq!(modeled_duration_ms(0.0, 3.0).unwrap(), 0.0);
    }

    #[test]
    fn modeled_duration_rejects_invalid_provider_values() {
        for raw_ms in [f64::NAN, f64::INFINITY, -1.0] {
            assert!(modeled_duration_ms(raw_ms, 1.0).is_err());
        }
        for speedup in [f64::NAN, f64::INFINITY, -1.0] {
            assert!(modeled_duration_ms(1.0, speedup).is_err());
        }
    }

    #[test]
    fn fixed_model_returns_zero_for_empty_batches() {
        let model = built_in_timing_model(&TimingModelConfig::Fixed {
            prefill_ms: 7.0,
            decode_ms: 3.0,
        })
        .unwrap();
        assert_eq!(model.predict_prefill_ms(0, 128, 0).unwrap(), 0.0);
        assert_eq!(model.predict_decode_ms(0, 128, 64, 1024).unwrap(), 0.0);
        assert_eq!(model.predict_prefill_ms(2, 128, 0).unwrap(), 7.0);
        assert_eq!(model.predict_decode_ms(2, 128, 64, 1024).unwrap(), 3.0);
    }

    #[test]
    fn external_provider_requires_runner_resolution() {
        let error = built_in_timing_model(&TimingModelConfig::External {
            provider: "example".to_string(),
            config: Value::Null,
        })
        .err()
        .expect("external descriptor cannot be materialized in the neutral crate");
        assert!(
            error
                .to_string()
                .contains("EngineFactory::with_timing_model")
        );
    }

    #[test]
    fn phase_evidence_accumulates_by_operation_without_fabricating_energy() {
        let mut phase = TimingPhaseEvidence::from_operations(vec![
            TimingOperationEvidence::new(
                "gemm",
                10.0,
                Some(4_000.0),
                TimingEvidenceSource::Silicon,
            )
            .unwrap(),
            TimingOperationEvidence::new("attention", 5.0, None, TimingEvidenceSource::Empirical)
                .unwrap(),
        ]);
        phase.accumulate(TimingPhaseEvidence::from_operations(vec![
            TimingOperationEvidence::new("gemm", 2.0, Some(800.0), TimingEvidenceSource::Silicon)
                .unwrap(),
        ]));

        assert_eq!(phase.energy_wms, Some(4_800.0));
        assert_eq!(phase.latency_ms, 17.0);
        assert_eq!(phase.covered_latency_ms, 12.0);
        assert_eq!(phase.coverage(), 12.0 / 17.0);
        assert_eq!(phase.source, Some(TimingEvidenceSource::Mixed));
        assert_eq!(phase.operations.len(), 2);
        assert_eq!(phase.operations[0].energy_wms, Some(4_800.0));
        assert_eq!(phase.operations[1].energy_wms, None);
    }

    #[test]
    fn zero_energy_is_canonicalized_to_missing() {
        let operation = TimingOperationEvidence::new(
            "attention",
            5.0,
            Some(0.0),
            TimingEvidenceSource::Empirical,
        )
        .unwrap();

        assert_eq!(operation.energy_wms, None);
        assert_eq!(operation.covered_latency_ms, 0.0);
    }

    #[test]
    fn public_operation_fields_cannot_fabricate_coverage() {
        let operation = TimingOperationEvidence {
            name: "attention".into(),
            energy_wms: None,
            latency_ms: 10.0,
            covered_latency_ms: 10.0,
            source: TimingEvidenceSource::Empirical,
        };

        let raw_phase = TimingPhaseEvidence {
            energy_wms: None,
            latency_ms: 10.0,
            covered_latency_ms: 10.0,
            source: Some(TimingEvidenceSource::Empirical),
            operations: vec![operation.clone()],
        };
        assert_eq!(raw_phase.coverage(), 0.0);

        let phase = TimingPhaseEvidence::try_from_operations(vec![operation]).unwrap();

        assert_eq!(phase.energy_wms, None);
        assert_eq!(phase.covered_latency_ms, 0.0);
        assert_eq!(phase.coverage(), 0.0);
        assert_eq!(phase.operations[0].covered_latency_ms, 0.0);

        let mut accumulated = TimingPhaseEvidence::default();
        accumulated
            .try_accumulate(TimingPhaseEvidence {
                energy_wms: None,
                latency_ms: 10.0,
                covered_latency_ms: 10.0,
                source: Some(TimingEvidenceSource::Empirical),
                operations: Vec::new(),
            })
            .unwrap();
        assert_eq!(accumulated.covered_latency_ms, 0.0);
        assert_eq!(accumulated.coverage(), 0.0);
    }

    #[test]
    fn fallible_phase_accumulation_is_atomic_on_invalid_public_fields() {
        let mut phase = TimingPhaseEvidence::from_operations(vec![
            TimingOperationEvidence::new("gemm", 2.0, Some(900.0), TimingEvidenceSource::Silicon)
                .unwrap(),
        ]);
        let before = phase.clone();
        let invalid = TimingPhaseEvidence {
            latency_ms: f64::NAN,
            ..TimingPhaseEvidence::default()
        };

        assert!(phase.try_accumulate(invalid).is_err());
        assert_eq!(phase, before);
    }

    #[test]
    fn built_in_timing_models_expose_no_energy_evidence() {
        let fixed = built_in_timing_model(&TimingModelConfig::Fixed {
            prefill_ms: 7.0,
            decode_ms: 3.0,
        })
        .unwrap();
        let polynomial = built_in_timing_model(&TimingModelConfig::Polynomial).unwrap();

        assert_eq!(fixed.evidence_summary(), None);
        assert_eq!(polynomial.evidence_summary(), None);
    }
}
