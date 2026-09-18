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
    pub details: Option<crate::perfmodel::engine::diagnostics::OperationDetails>,
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
            details: None,
        }
        .canonicalized()
    }

    fn canonicalized(mut self) -> Result<Self> {
        ensure!(
            !self.name.trim().is_empty(),
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
        if let Some(details) = &self.details {
            ensure!(
                details.sol.is_some() != details.sol_unavailable_reason.is_some(),
                "SOL evidence requires exactly one of a value or an unavailable reason"
            );
            ensure!(
                details
                    .sol_unavailable_reason
                    .as_ref()
                    .is_none_or(|reason| !reason.trim().is_empty()),
                "SOL unavailable reason cannot be empty"
            );
            if let Some(sol) = &details.sol {
                ensure!(
                    [sol.latency_ms, sol.math_ms, sol.memory_ms]
                        .iter()
                        .all(|v| v.is_finite() && *v >= 0.0),
                    "invalid SOL evidence"
                );
            }
            for fallback in &details.fallbacks {
                ensure!(
                    matches!(fallback.inference_phase.as_str(), "context" | "generation"),
                    "invalid fallback inference phase"
                );
                ensure!(
                    !fallback.comm_backend.trim().is_empty(),
                    "fallback comm backend cannot be empty"
                );
                ensure!(
                    fallback.requested_ep_size > 0
                        && fallback.requested_node_num > 0
                        && fallback.measurement_ep_size > 0
                        && fallback.measurement_node_num > 0,
                    "fallback topology sizes must be positive"
                );
            }
        }
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
        combined.details = match (combined.details, other.details) {
            (Some(mut left), Some(right)) => {
                left.sol = match (left.sol, right.sol) {
                    (Some(mut a), Some(b)) => {
                        a.latency_ms += b.latency_ms;
                        a.math_ms += b.math_ms;
                        a.memory_ms += b.memory_ms;
                        Some(a)
                    }
                    _ => None,
                };
                if left.sol.is_none() {
                    left.sol_unavailable_reason = left
                        .sol_unavailable_reason
                        .or(right.sol_unavailable_reason)
                        .or_else(|| Some("some accumulated operations lack SOL evidence".into()));
                }
                for fallback in right.fallbacks {
                    if !left.fallbacks.contains(&fallback) {
                        left.fallbacks.push(fallback);
                    }
                }
                Some(left)
            }
            (Some(mut details), None) | (None, Some(mut details)) => {
                details.sol = None;
                details.sol_unavailable_reason.get_or_insert_with(|| {
                    "some accumulated operations lack diagnostic evidence".into()
                });
                Some(details)
            }
            (None, None) => None,
        };
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

/// Immutable provider evidence. Construction validates the public fields once;
/// callers can borrow the result but cannot change its validated values.
#[derive(Debug, Default)]
pub(crate) struct ValidatedTimingPhase(TimingPhaseEvidence);

impl ValidatedTimingPhase {
    pub(crate) fn from_operations(operations: Vec<TimingOperationEvidence>) -> Result<Self> {
        TimingPhaseEvidence::try_from_operations(operations).map(Self)
    }

    pub(crate) fn as_phase(&self) -> &TimingPhaseEvidence {
        &self.0
    }
}

/// Provider-owned state; all inputs pass through `ValidatedTimingPhase`.
/// The public, checked accumulator remains the fallback for changing layouts.
#[derive(Default)]
pub(crate) struct TimingEvidenceAccumulator {
    summary: TimingEvidenceSummary,
    pending: Vec<OperationUpdate>,
}

struct OperationUpdate {
    latency_ms: f64,
    energy_wms: Option<f64>,
    covered_latency_ms: f64,
}

impl TimingEvidenceAccumulator {
    pub(crate) fn snapshot(&self) -> TimingEvidenceSummary {
        self.summary.clone()
    }

    pub(crate) fn record(&mut self, incoming: &ValidatedTimingPhase, prefill: bool) -> Result<()> {
        let phase = if prefill {
            &mut self.summary.prefill
        } else {
            &mut self.summary.decode
        };
        let incoming = incoming.as_phase();
        if phase.operations.is_empty()
            || incoming.operations.is_empty()
            || phase.operations.len() != incoming.operations.len()
            || !phase
                .operations
                .iter()
                .zip(&incoming.operations)
                .all(|(left, right)| left.name == right.name)
        {
            return phase.try_accumulate(incoming.clone());
        }

        // Stage values before changing any public result. Match the checked
        // accumulator's arithmetic and validation order, including its errors.
        self.pending.clear();
        for (left, right) in phase.operations.iter().zip(&incoming.operations) {
            let latency_ms = left.latency_ms + right.latency_ms;
            let covered_latency_ms = left.covered_latency_ms + right.covered_latency_ms;
            let energy_wms = match (left.energy_wms, right.energy_wms) {
                (Some(left), Some(right)) => Some(left + right),
                (Some(energy), None) | (None, Some(energy)) => Some(energy),
                (None, None) => None,
            };
            ensure!(
                latency_ms.is_finite() && latency_ms >= 0.0,
                "timing evidence operation {:?} returned invalid latency {}ms",
                left.name,
                latency_ms
            );
            ensure!(
                energy_wms.is_none_or(|energy| energy.is_finite() && energy >= 0.0),
                "timing evidence operation {:?} returned invalid energy {:?}W-ms",
                left.name,
                energy_wms
            );
            ensure!(
                covered_latency_ms.is_finite()
                    && covered_latency_ms >= 0.0
                    && covered_latency_ms <= latency_ms,
                "timing evidence operation {:?} returned invalid covered latency {}ms for {}ms total",
                left.name,
                covered_latency_ms,
                latency_ms
            );
            let energy_wms = energy_wms.filter(|energy| *energy > 0.0);
            self.pending.push(OperationUpdate {
                latency_ms,
                energy_wms,
                covered_latency_ms: if energy_wms.is_some() {
                    covered_latency_ms
                } else {
                    0.0
                },
            });
        }

        // Keep the same three reductions as reconcile_operation_totals. In
        // particular, energy uses reduce rather than a zero-seeded sum.
        let latency_ms: f64 = self.pending.iter().map(|op| op.latency_ms).sum();
        let energy_wms = self
            .pending
            .iter()
            .filter_map(|op| op.energy_wms)
            .reduce(|left, right| left + right);
        let covered_latency_ms: f64 = self.pending.iter().map(|op| op.covered_latency_ms).sum();
        ensure!(
            latency_ms.is_finite() && latency_ms >= 0.0,
            "timing phase evidence returned invalid latency {}ms",
            latency_ms
        );
        ensure!(
            energy_wms.is_none_or(|energy| energy.is_finite() && energy >= 0.0),
            "timing phase evidence returned invalid energy {:?}W-ms",
            energy_wms
        );
        ensure!(
            covered_latency_ms.is_finite()
                && covered_latency_ms >= 0.0
                && covered_latency_ms <= latency_ms,
            "timing phase evidence returned invalid covered latency {}ms for {}ms total",
            covered_latency_ms,
            latency_ms
        );

        // All fallible work is complete. Retain names and unchanged Other
        // source strings instead of copying them on each prediction.
        for ((left, right), update) in phase
            .operations
            .iter_mut()
            .zip(&incoming.operations)
            .zip(&self.pending)
        {
            left.latency_ms = update.latency_ms;
            left.energy_wms = update.energy_wms;
            left.covered_latency_ms = update.covered_latency_ms;
            if left.source != right.source {
                left.source = TimingEvidenceSource::Mixed;
            }
        }
        if let Some(source) = &incoming.source {
            match &phase.source {
                Some(existing) if existing != source => {
                    phase.source = Some(TimingEvidenceSource::Mixed);
                }
                None => phase.source = Some(source.clone()),
                _ => {}
            }
        }
        phase.latency_ms = latency_ms;
        phase.energy_wms = energy_wms;
        phase.covered_latency_ms = if energy_wms.is_some() {
            covered_latency_ms
        } else {
            0.0
        };
        Ok(())
    }
}

impl TimingPhaseEvidence {
    /// Build phase totals from operation evidence, rejecting invalid public
    /// field values and canonicalizing uncovered latency before aggregation.
    pub fn try_from_operations(operations: Vec<TimingOperationEvidence>) -> Result<Self> {
        let mut phase = Self::default();
        for operation in operations {
            phase.accumulate_operation(operation.canonicalized()?)?;
        }
        phase.reconcile_operation_totals();
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
            && self.operation_totals_match()
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
        let has_phase_only_totals = |phase: &Self| {
            phase.operations.is_empty() && (phase.latency_ms != 0.0 || phase.energy_wms.is_some())
        };
        let complete_operations =
            !has_phase_only_totals(&combined) && !has_phase_only_totals(&other);
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
        if complete_operations {
            combined.reconcile_operation_totals();
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
        ensure!(
            self.operation_totals_match(),
            "timing phase totals disagree with operation evidence"
        );
        Ok(self)
    }

    fn operation_totals_match(&self) -> bool {
        if self.operations.is_empty() {
            return true;
        }
        // Phase and name-folded operation sums can differ by rounding. This
        // tolerance validates redundant totals; it never changes the power gate.
        let matches = |left: f64, right: f64| {
            left.is_finite()
                && right.is_finite()
                && (left - right).abs() <= 1e-9 * left.abs().max(right.abs())
        };
        let latency: f64 = self.operations.iter().map(|op| op.latency_ms).sum();
        let energy: f64 = self.operations.iter().filter_map(|op| op.energy_wms).sum();
        let covered: f64 = self
            .operations
            .iter()
            .map(|op| {
                if op
                    .energy_wms
                    .is_some_and(|energy| energy.is_finite() && energy > 0.0)
                {
                    op.covered_latency_ms
                } else {
                    0.0
                }
            })
            .sum();
        matches(self.latency_ms, latency)
            && matches(self.energy_wms.unwrap_or(0.0), energy)
            && matches(self.covered_latency_ms, covered)
    }

    fn reconcile_operation_totals(&mut self) {
        if self.operations.is_empty() {
            return;
        }
        // Inputs are validated before merging. Derive redundant totals in the
        // same order as the merged operations so replay length cannot amplify
        // a rounding difference between per-step and per-operation sums.
        self.latency_ms = self.operations.iter().map(|op| op.latency_ms).sum();
        self.energy_wms = self
            .operations
            .iter()
            .filter_map(|op| op.energy_wms)
            .reduce(|left, right| left + right);
        self.covered_latency_ms = self.operations.iter().map(|op| op.covered_latency_ms).sum();
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
    /// Whether `validate_prefill_batch` may reject geometry. A false return
    /// guarantees only that validation accepts every batch, not that prediction
    /// or duration conversion is infallible. Admission must still protect those
    /// later operations for external providers.
    fn prefill_batch_validation_can_fail(&self) -> bool {
        true
    }

    /// Validate actual (new tokens, cached prefix) pairs before a scheduler
    /// reduces them to means. Providers with nonlinear per-request execution
    /// policies may reject batches that their aggregate API cannot represent.
    fn validate_prefill_batch(&self, _requests: &[(usize, usize)]) -> Result<()> {
        Ok(())
    }

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

    /// Start a new measurement epoch without changing timing predictions or
    /// provider caches. Called only after preparation work has settled.
    ///
    /// Latency-only providers need no reset. Evidence-producing providers must
    /// override this method so preparation cannot leak into profile evidence.
    fn reset_evidence(&self) -> Result<()> {
        ensure!(
            self.evidence_summary().is_none(),
            "timing provider exposes evidence but does not support resetting its measurement epoch"
        );
        Ok(())
    }
}

struct PolynomialTimingModel;

impl TimingModel for PolynomialTimingModel {
    fn prefill_batch_validation_can_fail(&self) -> bool {
        false
    }

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
    fn prefill_batch_validation_can_fail(&self) -> bool {
        false
    }

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
        assert!(!model.prefill_batch_validation_can_fail());
        assert_eq!(model.predict_prefill_ms(0, 128, 0).unwrap(), 0.0);
        assert_eq!(model.predict_decode_ms(0, 128, 64, 1024).unwrap(), 0.0);
        assert_eq!(model.predict_prefill_ms(2, 128, 0).unwrap(), 7.0);
        assert_eq!(model.predict_decode_ms(2, 128, 64, 1024).unwrap(), 3.0);
    }

    #[test]
    fn polynomial_model_does_not_need_admission_checkpoint() {
        let model = built_in_timing_model(&TimingModelConfig::Polynomial).unwrap();
        assert!(!model.prefill_batch_validation_can_fail());
        model.validate_prefill_batch(&[(4, 0), (12, 8)]).unwrap();
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
    fn sol_evidence_requires_one_explicit_availability_state() {
        use crate::perfmodel::engine::diagnostics::{OperationDetails, SolDiagnostics};
        for (has_sol, reason, valid) in [
            (false, None, false),
            (false, Some(""), false),
            (false, Some("  "), false),
            (false, Some("unsupported operation"), true),
            (true, Some("unsupported operation"), false),
            (true, None, true),
        ] {
            let mut op =
                TimingOperationEvidence::new("test", 2.0, None, TimingEvidenceSource::Estimated)
                    .unwrap();
            op.details = Some(OperationDetails {
                sol: has_sol.then_some(SolDiagnostics {
                    latency_ms: 1.0,
                    math_ms: 0.0,
                    memory_ms: 1.0,
                }),
                sol_unavailable_reason: reason.map(str::to_string),
                fallbacks: Vec::new(),
            });
            assert_eq!(
                TimingPhaseEvidence::try_from_operations(vec![op]).is_ok(),
                valid
            );
        }
    }

    #[test]
    fn diagnostic_fallback_records_are_validated_at_the_public_boundary() {
        use crate::perfmodel::engine::diagnostics::{ExecutedFallback, OperationDetails};
        for name in ["", " \t\n"] {
            assert!(
                TimingOperationEvidence::new(name, 2.0, None, TimingEvidenceSource::Estimated)
                    .is_err()
            );
        }
        let valid = ExecutedFallback {
            inference_phase: "context".into(),
            comm_backend: "deepep_ll".into(),
            requested_ep_size: 16,
            requested_node_num: 2,
            measurement_ep_size: 8,
            measurement_node_num: 1,
        };
        let mut records = vec![(valid.clone(), true)];
        for (key, value) in [
            ("inference_phase", serde_json::json!("generation")),
            ("inference_phase", serde_json::json!("invalid")),
            ("comm_backend", serde_json::json!(" \t\n")),
            ("requested_ep_size", serde_json::json!(0)),
            ("requested_node_num", serde_json::json!(0)),
            ("measurement_ep_size", serde_json::json!(0)),
            ("measurement_node_num", serde_json::json!(0)),
        ] {
            let mut record = serde_json::to_value(&valid).unwrap();
            record[key] = value.clone();
            records.push((
                serde_json::from_value(record).unwrap(),
                value == serde_json::json!("generation"),
            ));
        }
        for (fallback, expected_valid) in records {
            let mut operation = TimingOperationEvidence::new(
                "dispatch",
                2.0,
                None,
                TimingEvidenceSource::Estimated,
            )
            .unwrap();
            operation.details = Some(OperationDetails {
                sol: None,
                sol_unavailable_reason: Some("unsupported operation".into()),
                fallbacks: vec![valid.clone(), fallback],
            });
            assert_eq!(
                TimingPhaseEvidence::try_from_operations(vec![operation]).is_ok(),
                expected_valid
            );
        }
    }

    #[test]
    fn mixed_diagnostics_preserve_known_fallbacks_in_both_orders() {
        use crate::perfmodel::engine::diagnostics::{
            ExecutedFallback, OperationDetails, SolDiagnostics,
        };
        let plain =
            TimingOperationEvidence::new("dispatch", 2.0, None, TimingEvidenceSource::Estimated)
                .unwrap();
        let mut detailed = plain.clone();
        detailed.details = Some(OperationDetails {
            sol: Some(SolDiagnostics {
                latency_ms: 1.0,
                math_ms: 0.0,
                memory_ms: 1.0,
            }),
            sol_unavailable_reason: None,
            fallbacks: vec![ExecutedFallback {
                inference_phase: "generation".into(),
                comm_backend: "deepep_ll".into(),
                requested_ep_size: 16,
                requested_node_num: 2,
                measurement_ep_size: 8,
                measurement_node_num: 1,
            }],
        });
        for prior_reason in [None, Some("unsupported shape")] {
            let mut detailed = detailed.clone();
            if let Some(reason) = prior_reason {
                let details = detailed.details.as_mut().unwrap();
                details.sol = None;
                details.sol_unavailable_reason = Some(reason.into());
            }
            for ops in [
                vec![plain.clone(), detailed.clone()],
                vec![detailed.clone(), plain.clone()],
            ] {
                let phase = TimingPhaseEvidence::try_from_operations(ops).unwrap();
                assert_eq!(phase.latency_ms, 4.0);
                let details = phase.operations[0].details.as_ref().unwrap();
                assert_eq!(
                    details.fallbacks,
                    detailed.details.as_ref().unwrap().fallbacks
                );
                assert!(details.sol.is_none());
                assert_eq!(
                    details.sol_unavailable_reason.as_deref(),
                    Some(
                        prior_reason
                            .unwrap_or("some accumulated operations lack diagnostic evidence")
                    )
                );
            }
        }
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
    fn operation_evidence_rejects_invalid_public_values() {
        assert!(
            TimingOperationEvidence::new("", 1.0, None, TimingEvidenceSource::Silicon).is_err()
        );

        for latency_ms in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            assert!(
                TimingOperationEvidence::new(
                    "gemm",
                    latency_ms,
                    None,
                    TimingEvidenceSource::Silicon,
                )
                .is_err()
            );
        }

        for energy_wms in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            assert!(
                TimingOperationEvidence::new(
                    "gemm",
                    1.0,
                    Some(energy_wms),
                    TimingEvidenceSource::Silicon,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn public_operation_fields_cannot_fabricate_coverage() {
        let operation = TimingOperationEvidence {
            name: "attention".into(),
            energy_wms: None,
            latency_ms: 10.0,
            covered_latency_ms: 10.0,
            source: TimingEvidenceSource::Empirical,
            details: None,
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
    fn phase_totals_must_match_nonempty_operation_evidence() {
        let phase = TimingPhaseEvidence::from_operations(vec![
            TimingOperationEvidence::new(
                "gemm",
                10.0,
                Some(4_000.0),
                TimingEvidenceSource::Silicon,
            )
            .unwrap(),
        ]);
        for field in 0..3 {
            let mut invalid = phase.clone();
            match field {
                0 => invalid.energy_wms = Some(5_000.0),
                1 => invalid.latency_ms = 11.0,
                _ => invalid.covered_latency_ms = 9.0,
            }
            let mut result = TimingPhaseEvidence::default();
            assert!(result.try_accumulate(invalid.clone()).is_err());
            assert_eq!(result, TimingPhaseEvidence::default());
            assert_eq!(invalid.coverage(), 0.0);
        }
        let mut rounded = phase.clone();
        rounded.energy_wms = Some(4_000.0 + 1e-10);
        assert!(
            TimingPhaseEvidence::default()
                .try_accumulate(rounded)
                .is_ok()
        );
    }

    #[test]
    fn phase_accumulation_reconciles_rounding_in_operation_totals() {
        // This is a valid accumulator at the rounding-tolerance boundary:
        // tiny per-step additions can be lost in the phase total while they
        // remain representable in the independently accumulated small op.
        let mut phase = TimingPhaseEvidence {
            latency_ms: 1e16,
            energy_wms: Some(1e16),
            covered_latency_ms: 1e16,
            source: Some(TimingEvidenceSource::Silicon),
            operations: vec![
                TimingOperationEvidence::new(
                    "large",
                    1e16,
                    Some(1e16),
                    TimingEvidenceSource::Silicon,
                )
                .unwrap(),
                TimingOperationEvidence::new(
                    "small",
                    1e7,
                    Some(1e7),
                    TimingEvidenceSource::Silicon,
                )
                .unwrap(),
            ],
        };
        assert!(phase.clone().canonicalized().is_ok());
        let step = TimingPhaseEvidence::from_operations(vec![
            TimingOperationEvidence::new("small", 1.0, Some(1.0), TimingEvidenceSource::Silicon)
                .unwrap(),
        ]);
        for _ in 0..4 {
            phase.try_accumulate(step.clone()).unwrap();
        }
        assert_eq!(phase.latency_ms, 1e16 + 1e7 + 4.0);
        assert_eq!(phase.energy_wms, Some(phase.latency_ms));
        assert_eq!(phase.covered_latency_ms, phase.latency_ms);
        assert_eq!(phase.coverage(), 1.0);
    }

    #[test]
    fn phase_only_accumulation_preserves_totals_and_rejects_incomplete_operations() {
        let phase_only = TimingPhaseEvidence {
            latency_ms: 2.0,
            energy_wms: Some(800.0),
            covered_latency_ms: 2.0,
            source: Some(TimingEvidenceSource::Silicon),
            operations: Vec::new(),
        };
        let mut accumulated = TimingPhaseEvidence::default();
        accumulated.try_accumulate(phase_only.clone()).unwrap();
        accumulated.try_accumulate(phase_only.clone()).unwrap();
        assert_eq!(accumulated.latency_ms, 4.0);
        assert_eq!(accumulated.energy_wms, Some(1600.0));
        assert_eq!(accumulated.covered_latency_ms, 4.0);

        let operation_phase = TimingPhaseEvidence::from_operations(vec![
            TimingOperationEvidence::new("gemm", 2.0, Some(800.0), TimingEvidenceSource::Silicon)
                .unwrap(),
        ]);
        let before = accumulated.clone();
        assert!(accumulated.try_accumulate(operation_phase.clone()).is_err());
        assert_eq!(accumulated, before);
        let mut accumulated = operation_phase.clone();
        assert!(accumulated.try_accumulate(phase_only).is_err());
        assert_eq!(accumulated, operation_phase);
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
        fixed.reset_evidence().unwrap();
        polynomial.reset_evidence().unwrap();
    }

    fn assert_phase_bits(actual: &TimingPhaseEvidence, expected: &TimingPhaseEvidence) {
        assert_eq!(actual, expected);
        let numbers = |phase: &TimingPhaseEvidence| {
            let mut values = vec![
                Some(phase.latency_ms.to_bits()),
                phase.energy_wms.map(f64::to_bits),
                Some(phase.covered_latency_ms.to_bits()),
            ];
            for op in &phase.operations {
                values.extend([
                    Some(op.latency_ms.to_bits()),
                    op.energy_wms.map(f64::to_bits),
                    Some(op.covered_latency_ms.to_bits()),
                ]);
            }
            values
        };
        assert_eq!(numbers(actual), numbers(expected));
    }

    #[test]
    fn provider_accumulator_matches_checked_updates_bit_for_bit() {
        let mut fast = TimingEvidenceAccumulator::default();
        let mut checked = TimingEvidenceSummary::default();
        for step in 0..4096 {
            let source = if step % 3 == 0 {
                TimingEvidenceSource::Other("provider-specific".into())
            } else {
                TimingEvidenceSource::Silicon
            };
            let op = |name: &str, latency, energy| {
                TimingOperationEvidence::new(name, latency, energy, source.clone()).unwrap()
            };
            let mut operations = vec![
                op("large", 1e8, Some(1e9)),
                op("small", 0.1, (step % 4 != 0).then_some(0.3)),
                op("zero", -0.0, Some(-0.0)),
            ];
            match step % 29 {
                0 => operations.clear(),
                1 => operations.reverse(),
                2 => {
                    operations.remove(1);
                }
                3 => operations.push(op("small", 0.2, None)),
                _ => {}
            }
            let incoming = ValidatedTimingPhase::from_operations(operations).unwrap();
            let prefill = step % 7 == 0;
            fast.record(&incoming, prefill).unwrap();
            let phase = if prefill {
                &mut checked.prefill
            } else {
                &mut checked.decode
            };
            phase.try_accumulate(incoming.as_phase().clone()).unwrap();
            let snapshot = fast.snapshot();
            assert_phase_bits(&snapshot.prefill, &checked.prefill);
            assert_phase_bits(&snapshot.decode, &checked.decode);
        }
        assert!(
            !fast.pending.is_empty(),
            "the matching-layout path must run"
        );
    }

    #[test]
    fn provider_accumulator_preserves_checked_errors_and_atomicity() {
        for (latency, energy, count) in [
            (f64::MAX, None, 1),
            (1.0, Some(f64::MAX), 1),
            (f64::MAX * 0.3, None, 3),
            (1.0, Some(f64::MAX * 0.3), 3),
        ] {
            let incoming = ValidatedTimingPhase::from_operations(
                (0..count)
                    .map(|i| {
                        TimingOperationEvidence::new(
                            i.to_string(),
                            latency,
                            energy,
                            TimingEvidenceSource::Silicon,
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap();
            let mut fast = TimingEvidenceAccumulator::default();
            fast.record(&incoming, false).unwrap();
            let before = fast.snapshot();
            let mut checked = before.decode.clone();
            let expected = checked
                .try_accumulate(incoming.as_phase().clone())
                .unwrap_err();
            let actual = fast.record(&incoming, false).unwrap_err();
            assert_eq!(actual.to_string(), expected.to_string());
            assert_phase_bits(&fast.snapshot().decode, &before.decode);
            // A valid update after failure must not observe partial scratch values.
            fast.record(&ValidatedTimingPhase::default(), false)
                .unwrap();
            assert_phase_bits(&fast.snapshot().decode, &before.decode);
        }
        // Phase-only evidence always takes the public checked fallback. Such
        // values cannot currently be constructed by the provider adapter.
        let phase_only = ValidatedTimingPhase(TimingPhaseEvidence {
            latency_ms: 2.0,
            energy_wms: Some(800.0),
            covered_latency_ms: 2.0,
            source: Some(TimingEvidenceSource::Silicon),
            operations: Vec::new(),
        });
        let mut fast = TimingEvidenceAccumulator::default();
        let mut checked = TimingPhaseEvidence::default();
        for _ in 0..2 {
            fast.record(&phase_only, false).unwrap();
            checked
                .try_accumulate(phase_only.as_phase().clone())
                .unwrap();
            assert_phase_bits(&fast.snapshot().decode, &checked);
        }
        let incoming = ValidatedTimingPhase::from_operations(vec![
            TimingOperationEvidence::new("gemm", 1.0, None, TimingEvidenceSource::Silicon).unwrap(),
        ])
        .unwrap();
        assert_eq!(
            fast.record(&incoming, false).unwrap_err().to_string(),
            checked
                .try_accumulate(incoming.as_phase().clone())
                .unwrap_err()
                .to_string()
        );
        assert_phase_bits(&fast.snapshot().decode, &checked);
        for value in [f64::NAN, f64::INFINITY, -1.0] {
            let invalid = TimingOperationEvidence {
                name: "bad".into(),
                latency_ms: value,
                energy_wms: None,
                covered_latency_ms: 0.0,
                source: TimingEvidenceSource::Silicon,
            };
            assert!(ValidatedTimingPhase::from_operations(vec![invalid]).is_err());
        }
    }
}
