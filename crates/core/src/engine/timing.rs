// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-neutral forward-pass timing models.

#[cfg(test)]
use std::borrow::Cow;
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
    #[cfg(feature = "python")]
    pub(crate) fn from_native(source: &str) -> Self {
        Self::known_tag(source).unwrap_or_else(|| Self::Other(source.to_owned()))
    }

    pub fn from_provider(source: impl Into<String>) -> Self {
        let source = source.into();
        Self::known_tag(&source).unwrap_or(Self::Other(source))
    }

    fn known_tag(source: &str) -> Option<Self> {
        Some(match source {
            "silicon" => Self::Silicon,
            "empirical" => Self::Empirical,
            "sol" => Self::Sol,
            "estimated" => Self::Estimated,
            "mixed" => Self::Mixed,
            _ => return None,
        })
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

/// Unavailable reason once accumulated operations disagree on SOL coverage.
const PARTIAL_SOL_REASON: &str = "some accumulated operations lack SOL evidence";

/// Accumulated SOL of two operations, available only when both export it.
fn summed_sol(
    left: &crate::perfmodel::engine::diagnostics::OperationDetails,
    right: &crate::perfmodel::engine::diagnostics::OperationDetails,
) -> Option<crate::perfmodel::engine::diagnostics::SolDiagnostics> {
    Some(left.sol.as_ref()?.plus(right.sol.as_ref()?))
}

/// Merge `right`'s unavailable reason and fallbacks after `left.sol` holds
/// the accumulated SOL. FPM estimates are counted separately.
fn merge_reason_and_fallbacks(
    left: &mut crate::perfmodel::engine::diagnostics::OperationDetails,
    right: &crate::perfmodel::engine::diagnostics::OperationDetails,
) {
    if left.sol.is_none() && left.sol_unavailable_reason.is_none() {
        left.sol_unavailable_reason = Some(
            right
                .sol_unavailable_reason
                .clone()
                .unwrap_or_else(|| PARTIAL_SOL_REASON.into()),
        );
    }
    for fallback in &right.fallbacks {
        if !left.fallbacks.contains(fallback) {
            left.fallbacks.push(fallback.clone());
        }
    }
}

/// Validate one operation's values, reporting errors in this order.
fn check_operation_values(
    name: &str,
    latency_ms: f64,
    energy_wms: Option<f64>,
    covered_latency_ms: f64,
) -> Result<()> {
    ensure!(
        latency_ms.is_finite() && latency_ms >= 0.0,
        "timing evidence operation {:?} returned invalid latency {}ms",
        name,
        latency_ms
    );
    ensure!(
        energy_wms.is_none_or(|energy| energy.is_finite() && energy >= 0.0),
        "timing evidence operation {:?} returned invalid energy {:?}W-ms",
        name,
        energy_wms
    );
    ensure!(
        covered_latency_ms.is_finite()
            && covered_latency_ms >= 0.0
            && covered_latency_ms <= latency_ms,
        "timing evidence operation {:?} returned invalid covered latency {}ms for {}ms total",
        name,
        covered_latency_ms,
        latency_ms
    );
    Ok(())
}

/// Validate one phase's totals, reporting errors in this order.
fn check_phase_totals(
    latency_ms: f64,
    energy_wms: Option<f64>,
    covered_latency_ms: f64,
) -> Result<()> {
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
    Ok(())
}

/// `(latency_ms, energy_wms, covered_latency_ms)` totals of operations'
/// `(latency_ms, energy_wms, covered_latency_ms)`, reduced in order. Energy
/// uses reduce rather than a zero-seeded sum.
fn operation_totals(
    operations: impl Iterator<Item = (f64, Option<f64>, f64)> + Clone,
) -> (f64, Option<f64>, f64) {
    (
        operations.clone().map(|op| op.0).sum(),
        operations
            .clone()
            .filter_map(|op| op.1)
            .reduce(|left, right| left + right),
        operations.map(|op| op.2).sum(),
    )
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
        check_operation_values(
            &self.name,
            self.latency_ms,
            self.energy_wms,
            self.covered_latency_ms,
        )?;
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
                ensure!(sol.is_valid(), "invalid SOL evidence");
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
                left.sol = summed_sol(&left, &right);
                merge_reason_and_fallbacks(&mut left, &right);
                crate::perfmodel::fpm::accumulate_fpm_estimates(
                    &mut left.fpm_estimates,
                    right.fpm_estimates,
                )
                .map_err(anyhow::Error::msg)?;
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
/// callers can read and record the result but cannot change its validated
/// values.
#[derive(Debug, Default)]
#[cfg(any(feature = "python", test))]
pub(crate) struct ValidatedTimingPhase(ValidatedEvidence);

#[derive(Debug)]
#[cfg(any(feature = "python", test))]
enum ValidatedEvidence {
    /// Operations that carry their own names and optional diagnostics.
    Named(TimingPhaseEvidence),
    /// Native operations without diagnostics, named by shared slot names.
    Slotted(SlottedPhase),
}

#[cfg(any(feature = "python", test))]
impl Default for ValidatedEvidence {
    fn default() -> Self {
        Self::Named(TimingPhaseEvidence::default())
    }
}

/// One operation's values, named by its slot in a [`SlottedPhase`].
#[derive(Debug, Clone)]
#[cfg(any(feature = "python", test))]
pub(crate) struct OperationValues {
    latency_ms: f64,
    energy_wms: Option<f64>,
    covered_latency_ms: f64,
    source: TimingEvidenceSource,
}

#[cfg(any(feature = "python", test))]
impl OperationValues {
    /// The values `TimingOperationEvidence::new` keeps for `name`, with the
    /// same validation and errors, without allocating `name`.
    pub(crate) fn new(
        name: &str,
        latency_ms: f64,
        energy_wms: Option<f64>,
        source: TimingEvidenceSource,
    ) -> Result<Self> {
        let covered_latency_ms = if energy_wms.is_some_and(|energy| energy > 0.0) {
            latency_ms
        } else {
            0.0
        };
        ensure!(
            !name.trim().is_empty(),
            "timing evidence operation name cannot be empty"
        );
        check_operation_values(name, latency_ms, energy_wms, covered_latency_ms)?;
        let energy_wms = energy_wms.filter(|energy| *energy > 0.0);
        Ok(Self {
            latency_ms,
            energy_wms,
            covered_latency_ms: if energy_wms.is_some() {
                covered_latency_ms
            } else {
                0.0
            },
            source,
        })
    }
}

/// Name-folded operations whose names are slot names shared by every pass of
/// one compiled phase, so a pass owns only its values. Totals and source are
/// those `from_name_folded_operations` derives from the named operations.
#[derive(Debug, Clone)]
#[cfg(any(feature = "python", test))]
struct SlottedPhase {
    names: Arc<[String]>,
    operations: Vec<OperationValues>,
    latency_ms: f64,
    energy_wms: Option<f64>,
    covered_latency_ms: f64,
    source: Option<TimingEvidenceSource>,
}

#[cfg(any(feature = "python", test))]
impl SlottedPhase {
    /// `TimingPhaseEvidence::reconcile_operation_totals` over these values.
    fn reconcile_operation_totals(&mut self) {
        if self.operations.is_empty() {
            return;
        }
        (self.latency_ms, self.energy_wms, self.covered_latency_ms) = operation_totals(
            self.operations
                .iter()
                .map(|op| (op.latency_ms, op.energy_wms, op.covered_latency_ms)),
        );
    }

    fn to_named(&self) -> TimingPhaseEvidence {
        TimingPhaseEvidence {
            energy_wms: self.energy_wms,
            latency_ms: self.latency_ms,
            covered_latency_ms: self.covered_latency_ms,
            source: self.source.clone(),
            operations: self
                .names
                .iter()
                .zip(&self.operations)
                .map(|(name, op)| TimingOperationEvidence {
                    name: name.clone(),
                    energy_wms: op.energy_wms,
                    latency_ms: op.latency_ms,
                    covered_latency_ms: op.covered_latency_ms,
                    source: op.source.clone(),
                    details: None,
                })
                .collect(),
        }
    }
}

/// Stretch latency, covered latency and energy by an online correction.
#[cfg(feature = "python")]
fn scale_values(
    latency_ms: &mut f64,
    covered_latency_ms: &mut f64,
    energy_wms: &mut Option<f64>,
    factor: f64,
) {
    *latency_ms *= factor;
    *covered_latency_ms *= factor;
    *energy_wms = energy_wms.map(|energy| energy * factor);
}

#[cfg(any(feature = "python", test))]
impl ValidatedTimingPhase {
    /// Native rows have unique names in first-encounter order. Each operation
    /// is unchanged since `TimingOperationEvidence::new`, except for attached
    /// diagnostics, which are validated here. Keep this private contract
    /// separate from the checked constructor for public field values.
    #[cfg(feature = "python")]
    pub(crate) fn from_name_folded_operations(
        operations: Vec<TimingOperationEvidence>,
    ) -> Result<Self> {
        debug_assert!(operations.iter().enumerate().all(|(index, operation)| {
            !operations[..index]
                .iter()
                .any(|previous| previous.name == operation.name)
        }));
        // Same validation and error order as `try_from_operations`.
        let operations = operations
            .into_iter()
            .map(|operation| {
                if operation.details.is_some() {
                    operation.canonicalized()
                } else {
                    Ok(operation)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        debug_assert!(operations.iter().all(|operation| {
            matches!(operation.clone().canonicalized(), Ok(canonical) if canonical == *operation)
        }));
        let mut phase = TimingPhaseEvidence {
            operations,
            ..TimingPhaseEvidence::default()
        };
        for operation in &phase.operations {
            phase.source = Some(match phase.source.take() {
                Some(source) => source.merge(operation.source.clone()),
                None => operation.source.clone(),
            });
        }
        // Use the checked path's exact sum/reduce order, including signed zero.
        // All row checks precede phase-total checks, even if a total overflows.
        phase.reconcile_operation_totals();
        check_phase_totals(phase.latency_ms, phase.energy_wms, phase.covered_latency_ms)?;
        if phase.energy_wms.is_none() {
            phase.covered_latency_ms = 0.0;
        }
        Ok(Self(ValidatedEvidence::Named(phase)))
    }

    /// [`Self::from_name_folded_operations`] over operation `k` named
    /// `names[k]`. A compiled engine phase shares `names` across passes, so
    /// passes allocate no names; they are copied only when evidence is read.
    pub(crate) fn from_slot_operations(
        names: &Arc<[String]>,
        operations: Vec<OperationValues>,
    ) -> Result<Self> {
        ensure!(
            operations.len() == names.len(),
            "timing evidence has {} operations for {} slot names",
            operations.len(),
            names.len()
        );
        let mut source: Option<TimingEvidenceSource> = None;
        for operation in &operations {
            source = Some(match source.take() {
                Some(source) => source.merge(operation.source.clone()),
                None => operation.source.clone(),
            });
        }
        let mut phase = SlottedPhase {
            names: Arc::clone(names),
            operations,
            latency_ms: 0.0,
            energy_wms: None,
            covered_latency_ms: 0.0,
            source,
        };
        phase.reconcile_operation_totals();
        check_phase_totals(phase.latency_ms, phase.energy_wms, phase.covered_latency_ms)?;
        if phase.energy_wms.is_none() {
            phase.covered_latency_ms = 0.0;
        }
        Ok(Self(ValidatedEvidence::Slotted(phase)))
    }

    /// Checked reference constructor for arbitrary operation layouts.
    #[cfg(test)]
    pub(crate) fn from_operations(operations: Vec<TimingOperationEvidence>) -> Result<Self> {
        TimingPhaseEvidence::try_from_operations(operations)
            .map(|phase| Self(ValidatedEvidence::Named(phase)))
    }

    /// This phase stretched by an online correction `factor`: operation
    /// latencies and energies scale with it and the totals are reduced again,
    /// so power is unchanged. SOL compares the same work and stays the
    /// unscaled physical baseline.
    #[cfg(feature = "python")]
    pub(crate) fn scaled(&self, factor: f64) -> Result<Self> {
        ensure!(
            factor.is_finite() && factor >= 0.0,
            "online correction factor must be finite and non-negative, got {factor}"
        );
        let (scaled, (latency_ms, covered_latency_ms, energy_wms)) = match &self.0 {
            ValidatedEvidence::Named(phase) => {
                let mut phase = phase.clone();
                scale_values(
                    &mut phase.latency_ms,
                    &mut phase.covered_latency_ms,
                    &mut phase.energy_wms,
                    factor,
                );
                for operation in &mut phase.operations {
                    scale_values(
                        &mut operation.latency_ms,
                        &mut operation.covered_latency_ms,
                        &mut operation.energy_wms,
                        factor,
                    );
                }
                phase.reconcile_operation_totals();
                let totals = (phase.latency_ms, phase.covered_latency_ms, phase.energy_wms);
                (ValidatedEvidence::Named(phase), totals)
            }
            ValidatedEvidence::Slotted(phase) => {
                let mut phase = phase.clone();
                scale_values(
                    &mut phase.latency_ms,
                    &mut phase.covered_latency_ms,
                    &mut phase.energy_wms,
                    factor,
                );
                for operation in &mut phase.operations {
                    scale_values(
                        &mut operation.latency_ms,
                        &mut operation.covered_latency_ms,
                        &mut operation.energy_wms,
                        factor,
                    );
                }
                phase.reconcile_operation_totals();
                let totals = (phase.latency_ms, phase.covered_latency_ms, phase.energy_wms);
                (ValidatedEvidence::Slotted(phase), totals)
            }
        };
        ensure!(
            latency_ms.is_finite()
                && covered_latency_ms.is_finite()
                && energy_wms.is_none_or(f64::is_finite),
            "online correction factor {factor} overflowed timing evidence"
        );
        Ok(Self(scaled))
    }

    pub(crate) fn latency_ms(&self) -> f64 {
        match &self.0 {
            ValidatedEvidence::Named(phase) => phase.latency_ms,
            ValidatedEvidence::Slotted(phase) => phase.latency_ms,
        }
    }

    /// The evidence as named operations, copying slot names when needed.
    #[cfg(test)]
    pub(crate) fn to_phase(&self) -> Cow<'_, TimingPhaseEvidence> {
        match &self.0 {
            ValidatedEvidence::Named(phase) => Cow::Borrowed(phase),
            ValidatedEvidence::Slotted(phase) => Cow::Owned(phase.to_named()),
        }
    }
}

/// Provider-owned state; all inputs pass through `ValidatedTimingPhase`.
/// The public, checked accumulator remains the fallback for changing layouts
/// and optional operation diagnostics, whose merge rules it owns.
#[derive(Default)]
#[cfg(any(feature = "python", test))]
pub(crate) struct TimingEvidenceAccumulator {
    summary: TimingEvidenceSummary,
    pending: Vec<OperationUpdate>,
    /// Slot names that `summary.prefill` operations equal, in order and
    /// without diagnostics; slotted passes on them update by index.
    prefill_names: Option<Arc<[String]>>,
    /// The same for `summary.decode`.
    decode_names: Option<Arc<[String]>>,
}

#[cfg(any(feature = "python", test))]
struct OperationUpdate {
    latency_ms: f64,
    energy_wms: Option<f64>,
    covered_latency_ms: f64,
    /// Summed SOL when both operations carry diagnostics with SOL.
    sol: Option<crate::perfmodel::engine::diagnostics::SolDiagnostics>,
}

/// One incoming operation of an update aligned with the accumulated ones.
#[cfg(any(feature = "python", test))]
struct IncomingOperation<'a> {
    latency_ms: f64,
    energy_wms: Option<f64>,
    covered_latency_ms: f64,
    source: &'a TimingEvidenceSource,
    details: Option<&'a crate::perfmodel::engine::diagnostics::OperationDetails>,
}

#[cfg(any(feature = "python", test))]
impl<'a> IncomingOperation<'a> {
    fn named(operation: &'a TimingOperationEvidence) -> Self {
        Self {
            latency_ms: operation.latency_ms,
            energy_wms: operation.energy_wms,
            covered_latency_ms: operation.covered_latency_ms,
            source: &operation.source,
            details: operation.details.as_ref(),
        }
    }

    fn slotted(operation: &'a OperationValues) -> Self {
        Self {
            latency_ms: operation.latency_ms,
            energy_wms: operation.energy_wms,
            covered_latency_ms: operation.covered_latency_ms,
            source: &operation.source,
            details: None,
        }
    }
}

/// Whether `phase` holds exactly the operations `names` lists, in order and
/// without diagnostics.
#[cfg(any(feature = "python", test))]
fn matches_slot_names(phase: &TimingPhaseEvidence, names: &[String]) -> bool {
    !names.is_empty()
        && phase.operations.len() == names.len()
        && phase
            .operations
            .iter()
            .zip(names)
            .all(|(operation, name)| operation.name == *name && operation.details.is_none())
}

#[cfg(any(feature = "python", test))]
impl TimingEvidenceAccumulator {
    pub(crate) fn snapshot(&self) -> TimingEvidenceSummary {
        self.summary.clone()
    }

    pub(crate) fn record(&mut self, incoming: &ValidatedTimingPhase, prefill: bool) -> Result<()> {
        let (phase, names) = if prefill {
            (&mut self.summary.prefill, &mut self.prefill_names)
        } else {
            (&mut self.summary.decode, &mut self.decode_names)
        };
        match &incoming.0 {
            // `phase` equals these slot names without diagnostics, so the
            // named update would take the aligned path; skip its name checks.
            ValidatedEvidence::Slotted(slotted)
                if names
                    .as_ref()
                    .is_some_and(|names| Arc::ptr_eq(names, &slotted.names)) =>
            {
                Self::accumulate_aligned(
                    phase,
                    &mut self.pending,
                    slotted.operations.iter().map(IncomingOperation::slotted),
                    slotted.source.as_ref(),
                )
            }
            ValidatedEvidence::Slotted(slotted) => {
                Self::record_named(phase, &mut self.pending, &slotted.to_named())?;
                *names =
                    matches_slot_names(phase, &slotted.names).then(|| Arc::clone(&slotted.names));
                Ok(())
            }
            ValidatedEvidence::Named(incoming) => {
                Self::record_named(phase, &mut self.pending, incoming)?;
                if names
                    .as_ref()
                    .is_some_and(|names| !matches_slot_names(phase, names))
                {
                    *names = None;
                }
                Ok(())
            }
        }
    }

    fn record_named(
        phase: &mut TimingPhaseEvidence,
        pending: &mut Vec<OperationUpdate>,
        incoming: &TimingPhaseEvidence,
    ) -> Result<()> {
        // Private provider inputs are empty or built from operations, so stored
        // operation-backed totals are already reconciled. This shortcut relies
        // on that input contract. Keep empty and phase-only accumulated states
        // on the checked path, including -0.0.
        if !phase.operations.is_empty()
            && incoming.operations.is_empty()
            && incoming.latency_ms == 0.0
            && incoming.energy_wms.is_none()
            && incoming.covered_latency_ms == 0.0
            && incoming.source.is_none()
        {
            return Ok(());
        }
        // Diagnostics merge in place when both sides carry them and no FPM
        // estimates need counting; other layouts use the checked path.
        if phase.operations.is_empty()
            || incoming.operations.is_empty()
            || phase.operations.len() != incoming.operations.len()
            || !phase
                .operations
                .iter()
                .zip(&incoming.operations)
                .all(|(left, right)| {
                    left.name == right.name
                        && match (&left.details, &right.details) {
                            (None, None) => true,
                            (Some(_), Some(right)) => right.fpm_estimates.is_empty(),
                            _ => false,
                        }
                })
        {
            return phase.try_accumulate(incoming.clone());
        }
        Self::accumulate_aligned(
            phase,
            pending,
            incoming.operations.iter().map(IncomingOperation::named),
            incoming.source.as_ref(),
        )
    }

    /// Add `incoming` operation by operation to `phase`'s nonempty, equally
    /// named operations, each pair carrying diagnostics on both sides or on
    /// neither. `incoming_source` is the incoming phase's source.
    fn accumulate_aligned<'a>(
        phase: &mut TimingPhaseEvidence,
        pending: &mut Vec<OperationUpdate>,
        incoming: impl Iterator<Item = IncomingOperation<'a>> + Clone,
        incoming_source: Option<&TimingEvidenceSource>,
    ) -> Result<()> {
        // Stage values before changing any public result. Match the checked
        // accumulator's arithmetic and validation order, including its errors.
        pending.clear();
        for (left, right) in phase.operations.iter().zip(incoming.clone()) {
            let latency_ms = left.latency_ms + right.latency_ms;
            let covered_latency_ms = left.covered_latency_ms + right.covered_latency_ms;
            let energy_wms = match (left.energy_wms, right.energy_wms) {
                (Some(left), Some(right)) => Some(left + right),
                (Some(energy), None) | (None, Some(energy)) => Some(energy),
                (None, None) => None,
            };
            check_operation_values(&left.name, latency_ms, energy_wms, covered_latency_ms)?;
            let sol = match (&left.details, right.details) {
                (Some(left), Some(right)) => summed_sol(left, right),
                _ => None,
            };
            ensure!(
                sol.as_ref().is_none_or(|sol| sol.is_valid()),
                "invalid SOL evidence"
            );
            let energy_wms = energy_wms.filter(|energy| *energy > 0.0);
            pending.push(OperationUpdate {
                latency_ms,
                energy_wms,
                covered_latency_ms: if energy_wms.is_some() {
                    covered_latency_ms
                } else {
                    0.0
                },
                sol,
            });
        }

        // Keep the same three reductions as reconcile_operation_totals.
        let (latency_ms, energy_wms, covered_latency_ms) = operation_totals(
            pending
                .iter()
                .map(|op| (op.latency_ms, op.energy_wms, op.covered_latency_ms)),
        );
        check_phase_totals(latency_ms, energy_wms, covered_latency_ms)?;

        // All fallible work is complete. Retain names and unchanged Other
        // source strings instead of copying them on each prediction.
        for ((left, right), update) in phase.operations.iter_mut().zip(incoming).zip(&*pending) {
            left.latency_ms = update.latency_ms;
            left.energy_wms = update.energy_wms;
            left.covered_latency_ms = update.covered_latency_ms;
            if left.source != *right.source {
                left.source = TimingEvidenceSource::Mixed;
            }
            if let (Some(left), Some(right)) = (&mut left.details, right.details) {
                left.sol = update.sol.clone();
                merge_reason_and_fallbacks(left, right);
            }
        }
        if let Some(source) = incoming_source {
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
        check_phase_totals(self.latency_ms, self.energy_wms, self.covered_latency_ms)?;
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
        (self.latency_ms, self.energy_wms, self.covered_latency_ms) = operation_totals(
            self.operations
                .iter()
                .map(|op| (op.latency_ms, op.energy_wms, op.covered_latency_ms)),
        );
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

/// One homogeneous group of images in a vision-encoder batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VisionShape {
    pub encoder: crate::engine::EncoderShape,
    pub count: u32,
}

/// Exact totals of the prefill batch one forward pass computes: its requests,
/// the tokens they compute, and the cached or earlier-chunk KV they read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillTotals {
    pub batch_size: usize,
    pub new_tokens: usize,
    pub prefix_tokens: usize,
}

/// One forward pass that computes prefill and decode work together, in
/// milliseconds: the prefill latency and the latency the decode batch adds.
/// Their sum is the provider's estimate of the whole pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixedPassMs {
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

/// Runtime latency model injected at the engine boundary.
///
/// Implementations may call AIC, interpolate profiler data, or use another
/// provider without adding that dependency to `aisimulate-core`.
pub trait TimingModel: Send + Sync {
    /// Validate actual (new tokens, cached prefix) pairs before a scheduler
    /// reduces them to means. Providers with nonlinear per-request execution
    /// policies may reject batches that their aggregate API cannot represent.
    fn validate_prefill_batch(&self, _requests: &[(usize, usize)]) -> Result<()> {
        Ok(())
    }

    /// Predict one vision-encoder batch's latency in milliseconds. Latency-only
    /// providers keep the default `None`; the scheduler rejects image work then.
    fn predict_vision_ms(&self, _shapes: &[VisionShape]) -> Result<Option<f64>> {
        Ok(None)
    }

    /// Predict one prefill batch's latency in milliseconds.
    fn predict_prefill_ms(
        &self,
        batch_size: usize,
        mean_isl: usize,
        mean_prefix: usize,
    ) -> Result<f64>;

    /// Predict one prefill batch from its exact totals: `new_tokens` computed
    /// by the batch and `prefix_tokens` of cached or earlier-chunk KV that its
    /// requests read. Integer means cannot represent every heterogeneous
    /// batch, so providers indexed by these totals override this method. The
    /// default keeps the per-request means of [`Self::predict_prefill_ms`].
    fn predict_prefill_totals_ms(
        &self,
        batch_size: usize,
        new_tokens: usize,
        prefix_tokens: usize,
    ) -> Result<f64> {
        let (mean_isl, mean_prefix) = mean_prefill_shape(batch_size, new_tokens, prefix_tokens);
        self.predict_prefill_ms(batch_size, mean_isl, mean_prefix)
    }

    /// Predict one decode batch's latency in milliseconds.
    fn predict_decode_ms(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        mean_context_length: usize,
        total_kv_tokens: usize,
    ) -> Result<f64>;

    /// Predict one forward pass that computes `prefill` together with a decode
    /// batch. Arguments after `prefill` describe the decode batch as in
    /// [`Self::predict_decode_ms`].
    ///
    /// A provider that prices such a pass as one forward returns its prefill
    /// latency and the latency the decode batch adds, which sum to its
    /// estimate of that pass. The default `None` means the provider has no
    /// such composition; schedulers then price the prefill with
    /// [`Self::predict_prefill_totals_ms`] and a separate decode pass.
    fn predict_mixed_pass_ms(
        &self,
        _prefill: PrefillTotals,
        _batch_size: usize,
        _active_kv_tokens: usize,
        _mean_context_length: usize,
        _total_kv_tokens: usize,
    ) -> Result<Option<MixedPassMs>> {
        Ok(None)
    }

    /// Return typed evidence accumulated by this provider so far.
    ///
    /// Latency-only providers retain the default `None`. In particular, the
    /// built-in polynomial/fixed models and whole-model FPM timing do not
    /// fabricate energy or coverage values.
    fn evidence_summary(&self) -> Option<TimingEvidenceSummary> {
        None
    }

    /// Latency-only whole-model FPM evidence is separate from energy evidence.
    /// It must not change the provider's existing power publication contract.
    fn fpm_evidence_summary(&self) -> Option<TimingEvidenceSummary> {
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

/// Per-request integer means `(isl, prefix)` of a prefill batch's totals.
pub(crate) fn mean_prefill_shape(
    batch_size: usize,
    new_tokens: usize,
    prefix_tokens: usize,
) -> (usize, usize) {
    let mean = |tokens: usize| tokens.checked_div(batch_size).unwrap_or(0);
    (mean(new_tokens + prefix_tokens), mean(prefix_tokens))
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
    fn provider_source_tags_preserve_known_and_unknown_values() {
        for (tag, expected) in [
            ("silicon", TimingEvidenceSource::Silicon),
            ("empirical", TimingEvidenceSource::Empirical),
            ("sol", TimingEvidenceSource::Sol),
            ("estimated", TimingEvidenceSource::Estimated),
            ("mixed", TimingEvidenceSource::Mixed),
            (
                "transferred",
                TimingEvidenceSource::Other("transferred".into()),
            ),
            ("Silicon", TimingEvidenceSource::Other("Silicon".into())),
            ("", TimingEvidenceSource::Other(String::new())),
        ] {
            let owned = tag.to_owned();
            let original = owned.as_ptr();
            let actual = TimingEvidenceSource::from_provider(owned);
            assert_eq!(actual, expected);
            if let TimingEvidenceSource::Other(value) = &actual {
                assert_eq!(value.as_ptr(), original);
            }
            #[cfg(feature = "python")]
            assert_eq!(TimingEvidenceSource::from_native(tag), expected);
        }
    }

    #[cfg(all(feature = "python", debug_assertions))]
    #[test]
    fn native_builder_asserts_its_private_input_contract() {
        let valid =
            TimingOperationEvidence::new("op", 1.0, Some(2.0), TimingEvidenceSource::Sol).unwrap();
        assert!(
            std::panic::catch_unwind(|| {
                ValidatedTimingPhase::from_name_folded_operations(vec![
                    valid.clone(),
                    valid.clone(),
                ])
            })
            .is_err()
        );
        for change in 0..4 {
            let mut operation = valid.clone();
            match change {
                0 => operation.name.clear(),
                1 => operation.latency_ms = f64::NAN,
                2 => operation.energy_wms = Some(0.0),
                _ => operation.covered_latency_ms = -1.0,
            }
            assert!(
                std::panic::catch_unwind(|| {
                    ValidatedTimingPhase::from_name_folded_operations(vec![operation])
                })
                .is_err(),
                "mutation {change} bypassed the debug contract"
            );
        }
    }

    #[cfg(feature = "python")]
    #[test]
    fn native_builder_validates_diagnostics_like_checked_builder() {
        use crate::perfmodel::engine::diagnostics::{
            ExecutedFallback, OperationDetails, SolDiagnostics,
        };
        let operations = |reason: Option<&str>| {
            let mut operation =
                TimingOperationEvidence::new("op", 1.0, Some(2.0), TimingEvidenceSource::Silicon)
                    .unwrap();
            operation.details = Some(OperationDetails {
                sol: Some(SolDiagnostics {
                    latency_ms: 0.5,
                    math_ms: 0.25,
                    memory_ms: 0.5,
                }),
                sol_unavailable_reason: reason.map(Into::into),
                fallbacks: vec![ExecutedFallback {
                    inference_phase: "generation".into(),
                    comm_backend: "nccl".into(),
                    requested_ep_size: 8,
                    requested_node_num: 2,
                    measurement_ep_size: 8,
                    measurement_node_num: 1,
                }],
                fpm_estimates: Vec::new(),
            });
            vec![operation]
        };
        assert_phase_bits(
            &ValidatedTimingPhase::from_name_folded_operations(operations(None))
                .unwrap()
                .to_phase(),
            &ValidatedTimingPhase::from_operations(operations(None))
                .unwrap()
                .to_phase(),
        );
        // SOL with an unavailable reason is invalid; both builders reject it alike.
        assert_eq!(
            ValidatedTimingPhase::from_name_folded_operations(operations(Some("both")))
                .unwrap_err()
                .to_string(),
            ValidatedTimingPhase::from_operations(operations(Some("both")))
                .unwrap_err()
                .to_string()
        );
    }

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
    fn polynomial_model_accepts_heterogeneous_prefill_batches() {
        let model = built_in_timing_model(&TimingModelConfig::Polynomial).unwrap();
        model.validate_prefill_batch(&[(4, 0), (12, 8)]).unwrap();
    }

    #[test]
    fn mean_only_providers_keep_receiving_per_request_means() {
        struct MeanProbe(std::sync::Mutex<Vec<(usize, usize, usize)>>);
        impl TimingModel for MeanProbe {
            fn predict_prefill_ms(&self, batch: usize, isl: usize, prefix: usize) -> Result<f64> {
                self.0.lock().unwrap().push((batch, isl, prefix));
                Ok(1.0)
            }
            fn predict_decode_ms(&self, _: usize, _: usize, _: usize, _: usize) -> Result<f64> {
                Ok(1.0)
            }
        }
        let probe = MeanProbe(Default::default());
        // Three requests computing 16 tokens after reading 8: means (24 / 3, 8 / 3).
        probe.predict_prefill_totals_ms(3, 16, 8).unwrap();
        probe.predict_prefill_totals_ms(0, 0, 0).unwrap();
        assert_eq!(*probe.0.lock().unwrap(), vec![(3, 8, 2), (0, 0, 0)]);
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
                fpm_estimates: Vec::new(),
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
                fpm_estimates: Vec::new(),
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
            fpm_estimates: Vec::new(),
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
                if let Some(sol) = op.details.as_ref().and_then(|details| details.sol.as_ref()) {
                    values.extend(
                        [sol.latency_ms, sol.math_ms, sol.memory_ms].map(|v| Some(v.to_bits())),
                    );
                }
            }
            values
        };
        assert_eq!(numbers(actual), numbers(expected));
    }

    fn slot_names(names: &[&str]) -> Arc<[String]> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    /// A slotted pass over `names` with `(latency_ms, energy_wms, source)`.
    fn slotted_pass(
        names: &Arc<[String]>,
        values: impl IntoIterator<Item = (f64, Option<f64>, TimingEvidenceSource)>,
    ) -> Result<ValidatedTimingPhase> {
        let operations = names
            .iter()
            .zip(values)
            .map(|(name, (latency_ms, energy_wms, source))| {
                OperationValues::new(name, latency_ms, energy_wms, source)
            })
            .collect::<Result<Vec<_>>>()?;
        ValidatedTimingPhase::from_slot_operations(names, operations)
    }

    #[test]
    fn slotted_passes_accumulate_by_slot_with_hand_derived_totals() {
        use TimingEvidenceSource::{Empirical, Mixed, Silicon, Sol};
        let names = slot_names(&["qkv_gemm", "attention", "allreduce"]);
        // Zero energy is uncovered latency, not a zero-power measurement.
        let first = slotted_pass(
            &names,
            [
                (2.0, Some(300.0), Silicon),
                (1.0, Some(0.0), Sol),
                (0.5, Some(50.0), Silicon),
            ],
        )
        .unwrap();
        assert_eq!(first.latency_ms(), 3.5);
        let operation = |name: &str, latency_ms, energy_wms, covered_latency_ms, source| {
            TimingOperationEvidence {
                name: name.into(),
                energy_wms,
                latency_ms,
                covered_latency_ms,
                source,
                details: None,
            }
        };
        assert_eq!(
            *first.to_phase(),
            TimingPhaseEvidence {
                energy_wms: Some(350.0),
                latency_ms: 3.5,
                covered_latency_ms: 2.5,
                source: Some(Mixed),
                operations: vec![
                    operation("qkv_gemm", 2.0, Some(300.0), 2.0, Silicon),
                    operation("attention", 1.0, None, 0.0, Sol),
                    operation("allreduce", 0.5, Some(50.0), 0.5, Silicon),
                ],
            }
        );
        let second = slotted_pass(
            &names,
            [
                (2.0, Some(300.0), Empirical),
                (1.0, Some(100.0), Sol),
                (0.5, Some(50.0), Silicon),
            ],
        )
        .unwrap();
        let mut accumulator = TimingEvidenceAccumulator::default();
        accumulator.record(&first, false).unwrap();
        assert!(Arc::ptr_eq(
            accumulator.decode_names.as_ref().unwrap(),
            &names
        ));
        accumulator.record(&second, false).unwrap();
        assert_eq!(
            accumulator.pending.len(),
            3,
            "the second pass updates by slot"
        );
        let summary = accumulator.snapshot();
        assert_eq!(summary.prefill, TimingPhaseEvidence::default());
        // qkv_gemm sources differ across passes; attention is covered once.
        assert_eq!(
            summary.decode,
            TimingPhaseEvidence {
                energy_wms: Some(800.0),
                latency_ms: 7.0,
                covered_latency_ms: 6.0,
                source: Some(Mixed),
                operations: vec![
                    operation("qkv_gemm", 4.0, Some(600.0), 4.0, Mixed),
                    operation("attention", 2.0, Some(100.0), 1.0, Sol),
                    operation("allreduce", 1.0, Some(100.0), 1.0, Silicon),
                ],
            }
        );
        // An online correction of 1.5 stretches every value; power is unchanged.
        #[cfg(feature = "python")]
        {
            let scaled = first.scaled(1.5).unwrap();
            assert_eq!(scaled.latency_ms(), 5.25);
            assert_eq!(
                *scaled.to_phase(),
                TimingPhaseEvidence {
                    energy_wms: Some(525.0),
                    latency_ms: 5.25,
                    covered_latency_ms: 3.75,
                    source: Some(Mixed),
                    operations: vec![
                        operation("qkv_gemm", 3.0, Some(450.0), 3.0, Silicon),
                        operation("attention", 1.5, None, 0.0, Sol),
                        operation("allreduce", 0.75, Some(75.0), 0.75, Silicon),
                    ],
                }
            );
        }
    }

    #[test]
    fn slotted_passes_match_named_and_checked_accumulation_bit_for_bit() {
        let layouts = [
            slot_names(&["large", "small", "zero"]),
            slot_names(&["small", "large"]),
        ];
        let mut slotted = TimingEvidenceAccumulator::default();
        let mut named = TimingEvidenceAccumulator::default();
        let mut checked = TimingEvidenceSummary::default();
        let mut slot_updates = 0;
        for step in 0..2048 {
            let source = if step % 3 == 0 {
                TimingEvidenceSource::Other("provider-specific".into())
            } else {
                TimingEvidenceSource::Silicon
            };
            let names = &layouts[usize::from(step % 31 == 5)];
            let incoming = slotted_pass(
                names,
                names.iter().map(|name| match name.as_str() {
                    "large" => (1e8, Some(1e9), source.clone()),
                    "small" => (0.1, (step % 4 != 0).then_some(0.3), source.clone()),
                    _ => (-0.0, Some(-0.0), source.clone()),
                }),
            )
            .unwrap();
            let reference =
                ValidatedTimingPhase::from_operations(incoming.to_phase().operations.clone())
                    .unwrap();
            assert_phase_bits(&incoming.to_phase(), &reference.to_phase());
            assert_eq!(
                incoming.latency_ms().to_bits(),
                reference.latency_ms().to_bits()
            );
            // Decode sees only slotted passes; prefill also sees a foreign
            // operation, as a vision rank's encoder adds, and empty phases.
            let prefill = step % 7 == 0;
            let mut updates = vec![(incoming, reference)];
            if prefill && step % 2 == 0 {
                let foreign = ValidatedTimingPhase::from_operations(vec![
                    TimingOperationEvidence::new("encoder", 0.25, Some(1.0), source.clone())
                        .unwrap(),
                ])
                .unwrap();
                updates.push((
                    ValidatedTimingPhase::default(),
                    ValidatedTimingPhase::default(),
                ));
                updates.push((
                    foreign,
                    ValidatedTimingPhase::from_operations(vec![
                        TimingOperationEvidence::new("encoder", 0.25, Some(1.0), source.clone())
                            .unwrap(),
                    ])
                    .unwrap(),
                ));
            }
            for (incoming, reference) in updates {
                let by_slot = !prefill
                    && slotted
                        .decode_names
                        .as_ref()
                        .is_some_and(|names| matches!(&incoming.0, ValidatedEvidence::Slotted(phase) if Arc::ptr_eq(names, &phase.names)));
                slot_updates += usize::from(by_slot);
                slotted.record(&incoming, prefill).unwrap();
                named.record(&reference, prefill).unwrap();
                let phase = if prefill {
                    &mut checked.prefill
                } else {
                    &mut checked.decode
                };
                phase
                    .try_accumulate(reference.to_phase().into_owned())
                    .unwrap();
            }
            for accumulator in [&slotted, &named] {
                let snapshot = accumulator.snapshot();
                assert_phase_bits(&snapshot.prefill, &checked.prefill);
                assert_phase_bits(&snapshot.decode, &checked.decode);
            }
        }
        assert!(
            slot_updates > 1000,
            "slotted decode passes must update by slot"
        );
        assert!(
            slotted.prefill_names.is_none(),
            "foreign operations keep names checked"
        );
    }

    #[test]
    fn slotted_passes_preserve_checked_errors_and_atomicity() {
        use TimingEvidenceSource::Silicon;
        for (name, latency, energy) in [
            (" ", 1.0, None),
            ("op", f64::NAN, None),
            ("op", -1.0, Some(1.0)),
            ("op", 1.0, Some(-1.0)),
            ("op", 1.0, Some(f64::INFINITY)),
            ("op", f64::INFINITY, Some(1.0)),
        ] {
            assert_eq!(
                OperationValues::new(name, latency, energy, Silicon)
                    .unwrap_err()
                    .to_string(),
                TimingOperationEvidence::new(name, latency, energy, Silicon)
                    .unwrap_err()
                    .to_string()
            );
        }
        // Valid operations whose phase total overflows fail like the checked builder.
        let names = slot_names(&["first", "second"]);
        let overflow = [(f64::MAX, None, Silicon), (f64::MAX, None, Silicon)];
        assert_eq!(
            slotted_pass(&names, overflow.clone())
                .unwrap_err()
                .to_string(),
            ValidatedTimingPhase::from_operations(
                names
                    .iter()
                    .zip(overflow)
                    .map(|(name, (latency, energy, source))| {
                        TimingOperationEvidence::new(name.as_str(), latency, energy, source)
                            .unwrap()
                    })
                    .collect()
            )
            .unwrap_err()
            .to_string()
        );
        assert!(
            ValidatedTimingPhase::from_slot_operations(&names, Vec::new()).is_err(),
            "every slot name needs one operation"
        );
        // Accumulated overflow through the slot path reports the checked
        // error and leaves the state intact.
        for (latency, energy, count) in [
            (f64::MAX, None, 1),
            (1.0, Some(f64::MAX), 1),
            (f64::MAX * 0.3, None, 3),
            (1.0, Some(f64::MAX * 0.3), 3),
        ] {
            let names: Arc<[String]> = (0..count).map(|i: u32| i.to_string()).collect();
            let incoming =
                slotted_pass(&names, (0..count).map(|_| (latency, energy, Silicon))).unwrap();
            let mut accumulator = TimingEvidenceAccumulator::default();
            accumulator.record(&incoming, false).unwrap();
            assert!(accumulator.decode_names.is_some());
            let before = accumulator.snapshot();
            let expected = before
                .decode
                .clone()
                .try_accumulate(incoming.to_phase().into_owned())
                .unwrap_err();
            let actual = accumulator.record(&incoming, false).unwrap_err();
            assert_eq!(actual.to_string(), expected.to_string());
            assert_phase_bits(&accumulator.snapshot().decode, &before.decode);
            accumulator
                .record(&ValidatedTimingPhase::default(), false)
                .unwrap();
            assert_phase_bits(&accumulator.snapshot().decode, &before.decode);
            assert!(
                accumulator.decode_names.is_some(),
                "a failed pass keeps the slots"
            );
        }
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
            phase
                .try_accumulate(incoming.to_phase().into_owned())
                .unwrap();
            fast.record(&ValidatedTimingPhase::default(), prefill)
                .unwrap();
            phase
                .try_accumulate(TimingPhaseEvidence::default())
                .unwrap();
            let snapshot = fast.snapshot();
            assert_phase_bits(&snapshot.prefill, &checked.prefill);
            assert_phase_bits(&snapshot.decode, &checked.decode);
        }
        assert!(
            !fast.pending.is_empty(),
            "the matching-layout path must run"
        );
        // Optional diagnostics retain the checked accumulator's merge rules.
        for detailed in [true, true, false] {
            let mut op = TimingOperationEvidence::new(
                "diagnostic",
                1.0,
                None,
                TimingEvidenceSource::Silicon,
            )
            .unwrap();
            if detailed {
                op.details = Some(crate::perfmodel::engine::diagnostics::OperationDetails {
                    sol: Some(crate::perfmodel::engine::diagnostics::SolDiagnostics {
                        latency_ms: 0.5,
                        math_ms: 0.25,
                        memory_ms: 0.5,
                    }),
                    sol_unavailable_reason: None,
                    fallbacks: Vec::new(),
                    fpm_estimates: Vec::new(),
                });
            }
            let incoming = ValidatedTimingPhase::from_operations(vec![op]).unwrap();
            fast.record(&incoming, false).unwrap();
            checked
                .decode
                .try_accumulate(incoming.to_phase().into_owned())
                .unwrap();
            fast.record(&ValidatedTimingPhase::default(), false)
                .unwrap();
            checked
                .decode
                .try_accumulate(TimingPhaseEvidence::default())
                .unwrap();
            assert_phase_bits(&fast.snapshot().decode, &checked.decode);
        }
    }

    #[test]
    fn provider_accumulator_merges_matching_diagnostics_in_place() {
        use crate::perfmodel::engine::diagnostics::{
            ExecutedFallback, OperationDetails, SolDiagnostics,
        };
        let mut fast = TimingEvidenceAccumulator::default();
        let mut checked = TimingPhaseEvidence::default();
        for step in 0..32u32 {
            let operations = (0..3u32)
                .map(|op| {
                    let mut operation = TimingOperationEvidence::new(
                        format!("op{op}"),
                        0.1 * f64::from(step + op + 1),
                        (step % 5 != op).then_some(0.3),
                        TimingEvidenceSource::Silicon,
                    )
                    .unwrap();
                    let has_sol = (step + op) % 7 != 0;
                    operation.details = Some(OperationDetails {
                        sol: has_sol.then(|| SolDiagnostics {
                            latency_ms: 0.01 * f64::from(step),
                            math_ms: 0.003 * f64::from(op),
                            memory_ms: 1e-17,
                        }),
                        sol_unavailable_reason: (!has_sol).then(|| format!("reason {}", step % 3)),
                        fallbacks: (1..=(step + op) % 3)
                            .map(|ep| ExecutedFallback {
                                inference_phase: "generation".into(),
                                comm_backend: "nccl".into(),
                                requested_ep_size: ep,
                                requested_node_num: 1,
                                measurement_ep_size: 1,
                                measurement_node_num: 1,
                            })
                            .collect(),
                        fpm_estimates: Vec::new(),
                    });
                    operation
                })
                .collect();
            let incoming = ValidatedTimingPhase::from_operations(operations).unwrap();
            fast.record(&incoming, false).unwrap();
            checked
                .try_accumulate(incoming.to_phase().into_owned())
                .unwrap();
            assert_phase_bits(&fast.snapshot().decode, &checked);
        }
        assert_eq!(fast.pending.len(), 3, "diagnostics must take the fast path");
        // A SOL overflow reports the checked error and leaves the state intact.
        let mut operation =
            TimingOperationEvidence::new("op", 1.0, Some(1.0), TimingEvidenceSource::Silicon)
                .unwrap();
        operation.details = Some(OperationDetails {
            sol: Some(SolDiagnostics {
                latency_ms: f64::MAX,
                math_ms: 0.0,
                memory_ms: 0.0,
            }),
            sol_unavailable_reason: None,
            fallbacks: Vec::new(),
            fpm_estimates: Vec::new(),
        });
        let overflow = ValidatedTimingPhase::from_operations(vec![operation]).unwrap();
        let mut fast = TimingEvidenceAccumulator::default();
        fast.record(&overflow, false).unwrap();
        let before = fast.snapshot().decode;
        let expected = before
            .clone()
            .try_accumulate(overflow.to_phase().into_owned())
            .unwrap_err();
        let actual = fast.record(&overflow, false).unwrap_err();
        assert_eq!(actual.to_string(), expected.to_string());
        assert_phase_bits(&fast.snapshot().decode, &before);
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
                .try_accumulate(incoming.to_phase().into_owned())
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
        let phase_only = ValidatedTimingPhase(ValidatedEvidence::Named(TimingPhaseEvidence {
            latency_ms: 2.0,
            energy_wms: Some(800.0),
            covered_latency_ms: 2.0,
            source: Some(TimingEvidenceSource::Silicon),
            operations: Vec::new(),
        }));
        let mut fast = TimingEvidenceAccumulator::default();
        let mut checked = TimingPhaseEvidence::default();
        for _ in 0..2 {
            fast.record(&phase_only, false).unwrap();
            checked
                .try_accumulate(phase_only.to_phase().into_owned())
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
                .try_accumulate(incoming.to_phase().into_owned())
                .unwrap_err()
                .to_string()
        );
        assert_phase_bits(&fast.snapshot().decode, &checked);
        // Empty updates must retain the checked semantics for phase-only
        // states, including signed zero, and for source-only incoming phases.
        for initial in [
            TimingPhaseEvidence::default(),
            TimingPhaseEvidence {
                latency_ms: -0.0,
                covered_latency_ms: -0.0,
                ..Default::default()
            },
            phase_only.to_phase().into_owned(),
            incoming.to_phase().into_owned(),
        ] {
            for update in [
                TimingPhaseEvidence::default(),
                TimingPhaseEvidence {
                    source: Some(TimingEvidenceSource::Mixed),
                    ..Default::default()
                },
                phase_only.to_phase().into_owned(),
            ] {
                let mut fast = TimingEvidenceAccumulator::default();
                fast.summary.decode = initial.clone();
                let mut checked = initial.clone();
                let expected = checked.try_accumulate(update.clone());
                let actual = fast.record(
                    &ValidatedTimingPhase(ValidatedEvidence::Named(update)),
                    false,
                );
                assert_eq!(
                    actual.err().map(|e| e.to_string()),
                    expected.err().map(|e| e.to_string())
                );
                assert_phase_bits(&fast.snapshot().decode, &checked);
            }
        }
        for value in [f64::NAN, f64::INFINITY, -1.0] {
            let invalid = TimingOperationEvidence {
                name: "bad".into(),
                latency_ms: value,
                energy_wms: None,
                covered_latency_ms: 0.0,
                source: TimingEvidenceSource::Silicon,
                details: None,
            };
            assert!(ValidatedTimingPhase::from_operations(vec![invalid]).is_err());
        }
    }
}
