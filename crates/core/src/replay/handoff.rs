// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay-owned prefill/decode handoff ordering and virtual-transfer state.

use crate::engine::Backend;
pub use crate::engine::{HandoffId, HandoffTransferTiming};
use anyhow::{Result, bail};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum HandoffOrder {
    SourceFirst,
    DestinationFirst,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HandoffFact {
    SourceHeld {
        handoff_id: HandoffId,
        transfer_timing: HandoffTransferTiming,
    },
    DestinationReserved {
        handoff_id: HandoffId,
        transferable_prompt_tokens: usize,
    },
    TransferCompleted {
        handoff_id: HandoffId,
    },
    Failed {
        handoff_id: HandoffId,
    },
    TimedOut {
        handoff_id: HandoffId,
    },
    Canceled {
        handoff_id: HandoffId,
    },
}

impl HandoffFact {
    fn handoff_id(&self) -> HandoffId {
        match *self {
            Self::SourceHeld { handoff_id, .. }
            | Self::DestinationReserved { handoff_id, .. }
            | Self::TransferCompleted { handoff_id }
            | Self::Failed { handoff_id }
            | Self::TimedOut { handoff_id }
            | Self::Canceled { handoff_id } => handoff_id,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum HandoffAction {
    SubmitPrefill {
        handoff_id: HandoffId,
    },
    ReserveDestination {
        handoff_id: HandoffId,
    },
    StartTransfer {
        handoff_id: HandoffId,
        delay_ms: f64,
    },
    ActivateDestination {
        handoff_id: HandoffId,
    },
    ReleaseSource {
        handoff_id: HandoffId,
    },
    CancelSource {
        handoff_id: HandoffId,
    },
    CancelDestination {
        handoff_id: HandoffId,
    },
    Complete {
        handoff_id: HandoffId,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct HandoffActionId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct IssuedHandoffAction {
    pub id: HandoffActionId,
    pub action: HandoffAction,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum HandoffActionOutcome {
    Submitted,
    Accepted,
    Scheduled,
    Applied,
    Noop,
    Failed(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CoordinatorMode {
    Active,
    CleaningUp,
    Complete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandoffCompletion {
    Success,
    Canceled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NormalizedHandoffEvent {
    SourceHeld,
    DestinationAccepted,
    DestinationReserved,
    DestinationActivated,
    SourceReleased,
    Completed,
}

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NormalizedHandoffConformance {
    pub engine_type: Backend,
    pub order: HandoffOrder,
    pub lifecycle: Vec<NormalizedHandoffEvent>,
    pub source_output_tokens: usize,
    pub destination_output_tokens: usize,
    pub completed_requests: usize,
    pub destination_stored: NormalizedStoredTiming,
    pub source_drained: bool,
    pub destination_drained: bool,
    pub driver_drained: bool,
}

#[doc(hidden)]
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct NormalizedStoredTiming {
    pub before_activation: usize,
    pub on_activation: usize,
    pub repeated_activation_hashes_after_activation: usize,
}

impl NormalizedHandoffConformance {
    #[doc(hidden)]
    pub fn validate(&self) -> Result<()> {
        let expected_order = match self.engine_type {
            Backend::Vllm | Backend::Trtllm => HandoffOrder::SourceFirst,
            Backend::Sglang => HandoffOrder::DestinationFirst,
        };
        if self.order != expected_order {
            bail!(
                "normalized handoff order mismatch: expected {expected_order:?}, got {:?}",
                self.order
            );
        }
        if self.lifecycle != expected_normalized_handoff(self.order) {
            bail!(
                "normalized handoff lifecycle mismatch: got {:?}",
                self.lifecycle
            );
        }
        if self.source_output_tokens != 1
            || self.destination_output_tokens != 2
            || self.completed_requests != 1
        {
            bail!("normalized handoff output/completion counts do not match the fixture");
        }
        if self.destination_stored.before_activation != 0
            || self.destination_stored.on_activation == 0
            || self
                .destination_stored
                .repeated_activation_hashes_after_activation
                != 0
        {
            bail!("normalized handoff destination KV visibility is invalid");
        }
        if !self.source_drained || !self.destination_drained || !self.driver_drained {
            bail!("normalized handoff did not drain");
        }
        Ok(())
    }
}

pub fn expected_normalized_handoff(order: HandoffOrder) -> &'static [NormalizedHandoffEvent] {
    use NormalizedHandoffEvent::*;
    match order {
        HandoffOrder::SourceFirst => &[
            SourceHeld,
            DestinationAccepted,
            DestinationReserved,
            DestinationActivated,
            SourceReleased,
            Completed,
        ],
        HandoffOrder::DestinationFirst => &[
            DestinationAccepted,
            DestinationReserved,
            SourceHeld,
            DestinationActivated,
            SourceReleased,
            Completed,
        ],
    }
}

#[derive(Default)]
struct ActionJournal {
    started: bool,
    next_id: u64,
    issued: FxHashMap<HandoffActionId, HandoffAction>,
    outcomes: FxHashMap<HandoffActionId, HandoffActionOutcome>,
}

#[derive(Default)]
struct SourceProgress {
    submit_issued: bool,
    submitted: bool,
    held: bool,
    transfer_timing: Option<HandoffTransferTiming>,
    release_issued: bool,
    cancel_issued: bool,
    cleanup_done: bool,
}

#[derive(Default)]
struct DestinationProgress {
    reserve_issued: bool,
    accepted: bool,
    reserved: bool,
    transferable_prompt_tokens: Option<usize>,
    activation_issued: bool,
    activation_applied: bool,
    cancel_issued: bool,
    cleanup_done: bool,
}

#[derive(Default)]
struct TransferProgress {
    issued: bool,
    scheduled: bool,
    completed: bool,
}

/// Pure state machine for one prefill-to-decode ownership handoff.
pub struct HandoffCoordinatorCore {
    handoff_id: HandoffId,
    order: HandoffOrder,
    mode: CoordinatorMode,
    actions: ActionJournal,
    source: SourceProgress,
    destination: DestinationProgress,
    transfer: TransferProgress,
    completion: Option<HandoffCompletion>,
    fallback_transfer_delay_ms: f64,
}

impl HandoffCoordinatorCore {
    pub fn new(handoff_id: HandoffId, order: HandoffOrder) -> Self {
        Self::new_with_fallback(handoff_id, order, 0.0)
    }

    pub fn new_with_fallback(
        handoff_id: HandoffId,
        order: HandoffOrder,
        fallback_transfer_delay_ms: f64,
    ) -> Self {
        debug_assert!(fallback_transfer_delay_ms.is_finite());
        debug_assert!(fallback_transfer_delay_ms >= 0.0);
        Self {
            handoff_id,
            order,
            mode: CoordinatorMode::Active,
            actions: ActionJournal::default(),
            source: SourceProgress::default(),
            destination: DestinationProgress::default(),
            transfer: TransferProgress::default(),
            completion: None,
            fallback_transfer_delay_ms,
        }
    }

    pub fn start(&mut self) -> Result<Vec<IssuedHandoffAction>> {
        // `started` alone is not sufficient. A terminal fact arriving before
        // `start()` runs `begin_cleanup` -> `advance_cleanup`, which sees neither
        // `submit_issued` nor `reserve_issued` and completes immediately, leaving
        // `mode == Complete` with `started == false`. Starting from there issues a
        // SubmitPrefill that every subsequent `on_fact` and `on_action_outcome`
        // then swallows through their `Complete` guards, so the prefill is never
        // released, cancelled, or completed and holds its in-flight slot and KV
        // blocks for the rest of the run.
        if self.actions.started || self.mode != CoordinatorMode::Active {
            return Ok(Vec::new());
        }
        self.actions.started = true;
        let action = match self.order {
            HandoffOrder::SourceFirst => self.issue_submit_prefill(),
            HandoffOrder::DestinationFirst => self.issue_reserve_destination(),
        };
        Ok(vec![action])
    }

    pub fn on_fact(&mut self, fact: HandoffFact) -> Result<Vec<IssuedHandoffAction>> {
        self.validate_handoff(fact.handoff_id())?;
        if self.mode != CoordinatorMode::Active {
            // A terminal fact can legitimately arrive *after* the handoff
            // already succeeded: ownership then rests solely with the decode
            // destination, which is still generating tokens. Reopen cleanup
            // for exactly that case. `advance_cleanup` then issues
            // `CancelDestination` alone -- a successful handoff already set
            // `source.cleanup_done` when `ReleaseSource` applied, so the
            // source guard suppresses a redundant `CancelSource`.
            //
            // Without this, a decode-phase terminal was silently inert: the
            // runtime recorded the terminal while the decode worker kept the
            // request in its running batch to natural completion, holding its
            // KV and in-flight slot for the whole decode.
            //
            // All three terminal facts, matching the Active arm below. Only
            // `Canceled` has a production producer that can land post-success
            // today, but a narrower guard here than in the Active arm is a
            // landmine for whichever path starts emitting the other two.
            if matches!(
                fact,
                HandoffFact::Canceled { .. }
                    | HandoffFact::Failed { .. }
                    | HandoffFact::TimedOut { .. }
            ) && self.mode == CoordinatorMode::Complete
                && self.completion == Some(HandoffCompletion::Success)
            {
                // Not `begin_cleanup`: that method treats `Complete` as
                // terminal and returns no actions.
                self.mode = CoordinatorMode::CleaningUp;
                return self.advance_cleanup();
            }
            return Ok(Vec::new());
        }
        match fact {
            HandoffFact::SourceHeld {
                transfer_timing, ..
            } => {
                if self.source.held {
                    return Ok(Vec::new());
                }
                if !self.source.submitted {
                    bail!("source held before prefill submission was acknowledged");
                }
                validate_transfer_timing(transfer_timing)?;
                self.source.held = true;
                self.source.transfer_timing = Some(transfer_timing);
                self.advance_active()
            }
            HandoffFact::DestinationReserved {
                transferable_prompt_tokens,
                ..
            } => {
                if self.destination.reserved {
                    return Ok(Vec::new());
                }
                if !self.destination.accepted {
                    bail!("destination reserved before ownership was accepted");
                }
                self.destination.reserved = true;
                self.destination.transferable_prompt_tokens = Some(transferable_prompt_tokens);
                self.advance_active()
            }
            HandoffFact::TransferCompleted { .. } => {
                if self.transfer.completed {
                    return Ok(Vec::new());
                }
                if !self.transfer.scheduled {
                    bail!("transfer completed before it was scheduled");
                }
                self.transfer.completed = true;
                self.advance_active()
            }
            HandoffFact::Failed { .. }
            | HandoffFact::TimedOut { .. }
            | HandoffFact::Canceled { .. } => self.begin_cleanup(),
        }
    }

    pub fn on_action_outcome(
        &mut self,
        action_id: HandoffActionId,
        outcome: HandoffActionOutcome,
    ) -> Result<Vec<IssuedHandoffAction>> {
        if self.mode == CoordinatorMode::Complete {
            return Ok(Vec::new());
        }
        let Some(action) = self.actions.issued.get(&action_id).copied() else {
            bail!("unknown handoff action {action_id:?}");
        };
        if let Some(previous) = self.actions.outcomes.get(&action_id) {
            if previous != &outcome {
                bail!("conflicting outcome for handoff action {action_id:?}");
            }
            return Ok(Vec::new());
        }
        if let HandoffActionOutcome::Failed(_) = outcome {
            if matches!(
                action,
                HandoffAction::CancelSource { .. } | HandoffAction::CancelDestination { .. }
            ) {
                bail!("handoff cleanup action {action_id:?} failed");
            }
            self.actions.outcomes.insert(action_id, outcome);
            return self.begin_cleanup();
        }

        // Validate before journalling. Recording first made a *rejected*
        // outcome permanent, so a re-report of the same action id took the
        // `outcomes` hit above and failed with "conflicting outcome" instead of
        // the true "invalid handoff action outcome" -- a misleading diagnostic
        // for the caller that reported the bad outcome in the first place.
        match action {
            HandoffAction::SubmitPrefill { .. } => {
                require_outcome(&outcome, &[HandoffActionOutcome::Submitted])?;
            }
            HandoffAction::ReserveDestination { .. } => {
                require_outcome(&outcome, &[HandoffActionOutcome::Accepted])?;
            }
            HandoffAction::StartTransfer { .. } => {
                require_outcome(&outcome, &[HandoffActionOutcome::Scheduled])?;
            }
            HandoffAction::ActivateDestination { .. } => {
                require_outcome(&outcome, &[HandoffActionOutcome::Applied])?;
            }
            HandoffAction::ReleaseSource { .. }
            | HandoffAction::CancelSource { .. }
            | HandoffAction::CancelDestination { .. } => {
                require_outcome(
                    &outcome,
                    &[HandoffActionOutcome::Applied, HandoffActionOutcome::Noop],
                )?;
            }
            // Terminal bookkeeping: `complete()` has already cleared the
            // journal, so this arm does not constrain its outcome.
            HandoffAction::Complete { .. } => {}
        }
        self.actions.outcomes.insert(action_id, outcome);

        match action {
            HandoffAction::SubmitPrefill { .. } => self.source.submitted = true,
            HandoffAction::ReserveDestination { .. } => self.destination.accepted = true,
            HandoffAction::StartTransfer { .. } => self.transfer.scheduled = true,
            HandoffAction::ActivateDestination { .. } => self.destination.activation_applied = true,
            HandoffAction::ReleaseSource { .. } | HandoffAction::CancelSource { .. } => {
                self.source.cleanup_done = true;
            }
            HandoffAction::CancelDestination { .. } => self.destination.cleanup_done = true,
            HandoffAction::Complete { .. } => return Ok(Vec::new()),
        }

        match self.mode {
            CoordinatorMode::Active => self.advance_active(),
            CoordinatorMode::CleaningUp => self.advance_cleanup(),
            CoordinatorMode::Complete => Ok(Vec::new()),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.mode == CoordinatorMode::Complete
    }

    pub fn completion(&self) -> Option<HandoffCompletion> {
        self.completion
    }

    fn advance_active(&mut self) -> Result<Vec<IssuedHandoffAction>> {
        if self.order == HandoffOrder::SourceFirst
            && self.source.held
            && !self.destination.reserve_issued
        {
            return Ok(vec![self.issue_reserve_destination()]);
        }
        if self.order == HandoffOrder::DestinationFirst
            && self.destination.reserved
            && !self.source.submit_issued
        {
            return Ok(vec![self.issue_submit_prefill()]);
        }
        if self.source.held && self.destination.reserved && !self.transfer.issued {
            // Errors, not `expect`: the `held`/`reserved` flags and these two
            // payloads are set together today (`on_fact`'s SourceHeld and
            // DestinationReserved arms), but that is a state-machine invariant
            // held at two other sites, not a property of this one. A future
            // edit that sets a flag from a second site would turn a
            // state-machine bug into a process abort on the DES hot path, while
            // every other failure in this file is a recoverable `anyhow` error.
            let Some(transfer_timing) = self.source.transfer_timing else {
                bail!("held source did not retain transfer timing");
            };
            let Some(transferable_prompt_tokens) = self.destination.transferable_prompt_tokens
            else {
                bail!("reserved destination did not report its transferable footprint");
            };
            // Validate the number actually scheduled. `validate_transfer_timing`
            // runs on `SourceHeld` and checks `full_prompt_delay_ms()`, but in
            // `TransferTimingMode::DestinationMissing` that quantity is never
            // used: `delay_ms` reads `transferable_prompt_tokens` instead, which
            // arrives on the *`DestinationReserved`* fact and is bounded by
            // nothing. It is block-rounded at both KV backends
            // (`fresh_len() * block_size`, `unpublished_pages.len() * page_size`)
            // so it legitimately exceeds `full_prompt_tokens` -- a 10-token
            // prompt reserves 12 -- and the delay is monotonic in it. A finite
            // full-prompt delay therefore does not imply a finite scheduled one.
            // The fallback is only `debug_assert`ed, so it needs the same guard.
            let delay_ms = transfer_timing
                .delay_ms(transferable_prompt_tokens)
                .unwrap_or(self.fallback_transfer_delay_ms);
            validate_transfer_delay_ms(Some(delay_ms))?;
            // Set only once the transfer is certain to be issued: latching it
            // ahead of the fallible steps above would leave a rejected handoff
            // permanently marked as having issued a transfer it never did.
            self.transfer.issued = true;
            return Ok(vec![self.issue(HandoffAction::StartTransfer {
                handoff_id: self.handoff_id,
                delay_ms,
            })]);
        }
        if self.transfer.completed && !self.destination.activation_issued {
            self.destination.activation_issued = true;
            return Ok(vec![self.issue(HandoffAction::ActivateDestination {
                handoff_id: self.handoff_id,
            })]);
        }
        if self.destination.activation_applied && !self.source.release_issued {
            self.source.release_issued = true;
            return Ok(vec![self.issue(HandoffAction::ReleaseSource {
                handoff_id: self.handoff_id,
            })]);
        }
        if self.source.cleanup_done {
            return Ok(vec![self.complete()]);
        }
        Ok(Vec::new())
    }

    fn begin_cleanup(&mut self) -> Result<Vec<IssuedHandoffAction>> {
        if self.mode == CoordinatorMode::Complete {
            return Ok(Vec::new());
        }
        self.mode = CoordinatorMode::CleaningUp;
        self.advance_cleanup()
    }

    fn advance_cleanup(&mut self) -> Result<Vec<IssuedHandoffAction>> {
        let mut actions = Vec::new();
        if self.source.submit_issued && !self.source.cancel_issued && !self.source.cleanup_done {
            self.source.cancel_issued = true;
            actions.push(self.issue(HandoffAction::CancelSource {
                handoff_id: self.handoff_id,
            }));
        }
        if self.destination.reserve_issued
            && !self.destination.cancel_issued
            && !self.destination.cleanup_done
        {
            self.destination.cancel_issued = true;
            actions.push(self.issue(HandoffAction::CancelDestination {
                handoff_id: self.handoff_id,
            }));
        }
        if actions.is_empty()
            && (!self.source.submit_issued || self.source.cleanup_done)
            && (!self.destination.reserve_issued || self.destination.cleanup_done)
        {
            actions.push(self.complete());
        }
        Ok(actions)
    }

    fn issue_submit_prefill(&mut self) -> IssuedHandoffAction {
        self.source.submit_issued = true;
        self.issue(HandoffAction::SubmitPrefill {
            handoff_id: self.handoff_id,
        })
    }

    fn issue_reserve_destination(&mut self) -> IssuedHandoffAction {
        self.destination.reserve_issued = true;
        self.issue(HandoffAction::ReserveDestination {
            handoff_id: self.handoff_id,
        })
    }

    fn complete(&mut self) -> IssuedHandoffAction {
        self.completion = Some(if self.mode == CoordinatorMode::CleaningUp {
            HandoffCompletion::Canceled
        } else {
            HandoffCompletion::Success
        });
        self.mode = CoordinatorMode::Complete;
        let action = self.issue(HandoffAction::Complete {
            handoff_id: self.handoff_id,
        });
        self.actions.issued.clear();
        self.actions.outcomes.clear();
        action
    }

    fn issue(&mut self, action: HandoffAction) -> IssuedHandoffAction {
        let id = HandoffActionId(self.actions.next_id);
        self.actions.next_id = self
            .actions
            .next_id
            .checked_add(1)
            .expect("handoff action ID overflow");
        let previous = self.actions.issued.insert(id, action);
        debug_assert!(previous.is_none());
        IssuedHandoffAction { id, action }
    }

    fn validate_handoff(&self, handoff_id: HandoffId) -> Result<()> {
        if handoff_id != self.handoff_id {
            bail!("fact belongs to a different handoff");
        }
        Ok(())
    }
}

pub fn validate_transfer_delay_ms(transfer_delay_ms: Option<f64>) -> Result<()> {
    let Some(delay_ms) = transfer_delay_ms else {
        return Ok(());
    };
    if !delay_ms.is_finite() || delay_ms < 0.0 {
        bail!("invalid handoff transfer delay {delay_ms}");
    }
    Ok(())
}

pub fn validate_transfer_timing(transfer_timing: HandoffTransferTiming) -> Result<()> {
    if let Some(bandwidth_gb_s) = transfer_timing.bandwidth_gb_s
        && (!bandwidth_gb_s.is_finite() || bandwidth_gb_s < 0.0)
    {
        bail!("invalid handoff transfer bandwidth {bandwidth_gb_s}");
    }
    validate_transfer_delay_ms(transfer_timing.full_prompt_delay_ms())
}

fn require_outcome(outcome: &HandoffActionOutcome, allowed: &[HandoffActionOutcome]) -> Result<()> {
    if allowed.contains(outcome) {
        return Ok(());
    }
    bail!("invalid handoff action outcome {outcome:?}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::TransferTimingMode;
    use uuid::Uuid;

    /// A terminal fact before `start()` completes cleanup immediately (nothing
    /// has been issued to clean up), leaving `mode == Complete` with
    /// `started == false`. `start()` guarded only on `started`, so it then issued
    /// a SubmitPrefill that every later `on_fact`/`on_action_outcome` swallows
    /// through their `Complete` guards -- the prefill is never released or
    /// cancelled and holds its slot and KV blocks for the rest of the run.
    #[test]
    fn starting_an_already_terminal_coordinator_issues_nothing() {
        let handoff_id = HandoffId::new(Uuid::from_u128(7));
        let mut coordinator =
            HandoffCoordinatorCore::new_with_fallback(handoff_id, HandoffOrder::SourceFirst, 1.0);

        coordinator
            .on_fact(HandoffFact::Canceled { handoff_id })
            .expect("a terminal fact before start must be accepted");

        assert!(
            coordinator.start().unwrap().is_empty(),
            "a completed coordinator must not issue a prefill it can never retire"
        );
    }

    fn start_transfer_delay(timing: HandoffTransferTiming, fallback_ms: f64) -> f64 {
        let handoff_id = HandoffId::new(Uuid::from_u128(1));
        let mut coordinator = HandoffCoordinatorCore::new_with_fallback(
            handoff_id,
            HandoffOrder::SourceFirst,
            fallback_ms,
        );
        let submit = coordinator.start().unwrap().remove(0);
        assert!(matches!(submit.action, HandoffAction::SubmitPrefill { .. }));
        assert!(
            coordinator
                .on_action_outcome(submit.id, HandoffActionOutcome::Submitted)
                .unwrap()
                .is_empty()
        );
        let reserve = coordinator
            .on_fact(HandoffFact::SourceHeld {
                handoff_id,
                transfer_timing: timing,
            })
            .unwrap()
            .remove(0);
        assert!(matches!(
            reserve.action,
            HandoffAction::ReserveDestination { .. }
        ));
        assert!(
            coordinator
                .on_action_outcome(reserve.id, HandoffActionOutcome::Accepted)
                .unwrap()
                .is_empty()
        );
        let transfer = coordinator
            .on_fact(HandoffFact::DestinationReserved {
                handoff_id,
                transferable_prompt_tokens: 3,
            })
            .unwrap()
            .remove(0);
        match transfer.action {
            HandoffAction::StartTransfer { delay_ms, .. } => delay_ms,
            action => panic!("expected transfer action, got {action:?}"),
        }
    }

    #[test]
    fn missing_or_zero_bandwidth_uses_configured_fallback() {
        let missing = HandoffTransferTiming {
            mode: TransferTimingMode::DestinationMissing,
            full_prompt_tokens: 10,
            kv_bytes_per_token: None,
            bandwidth_gb_s: None,
        };
        assert_eq!(start_transfer_delay(missing, 7.5), 7.5);

        let zero_bandwidth = HandoffTransferTiming {
            kv_bytes_per_token: Some(1024),
            bandwidth_gb_s: Some(0.0),
            ..missing
        };
        validate_transfer_timing(zero_bandwidth).unwrap();
        assert_eq!(start_transfer_delay(zero_bandwidth, 7.5), 7.5);
    }

    fn completed_coordinator(handoff_id: HandoffId) -> HandoffCoordinatorCore {
        let mut coordinator =
            HandoffCoordinatorCore::new_with_fallback(handoff_id, HandoffOrder::SourceFirst, 1.0);
        let submit = coordinator.start().unwrap().remove(0);
        coordinator
            .on_action_outcome(submit.id, HandoffActionOutcome::Submitted)
            .unwrap();
        let reserve = coordinator
            .on_fact(HandoffFact::SourceHeld {
                handoff_id,
                transfer_timing: HandoffTransferTiming {
                    mode: TransferTimingMode::DestinationMissing,
                    full_prompt_tokens: 4,
                    kv_bytes_per_token: None,
                    bandwidth_gb_s: None,
                },
            })
            .unwrap()
            .remove(0);
        coordinator
            .on_action_outcome(reserve.id, HandoffActionOutcome::Accepted)
            .unwrap();
        let transfer = coordinator
            .on_fact(HandoffFact::DestinationReserved {
                handoff_id,
                transferable_prompt_tokens: 4,
            })
            .unwrap()
            .remove(0);
        coordinator
            .on_action_outcome(transfer.id, HandoffActionOutcome::Scheduled)
            .unwrap();
        let activate = coordinator
            .on_fact(HandoffFact::TransferCompleted { handoff_id })
            .unwrap()
            .remove(0);
        let release = coordinator
            .on_action_outcome(activate.id, HandoffActionOutcome::Applied)
            .unwrap()
            .remove(0);
        coordinator
            .on_action_outcome(release.id, HandoffActionOutcome::Applied)
            .unwrap();
        assert!(coordinator.is_complete());
        assert_eq!(coordinator.completion(), Some(HandoffCompletion::Success));
        coordinator
    }

    /// After a successful handoff the destination alone owns the request and is
    /// still generating tokens, so every terminal fact must reopen cleanup to
    /// issue `CancelDestination`. `on_fact`'s Active arm already treats the
    /// three uniformly; the post-success reopen must not be narrower, or the
    /// runtime records a terminal while the decode worker runs to completion
    /// holding its KV and in-flight slot.
    #[test]
    fn every_terminal_fact_reopens_cleanup_after_a_successful_handoff() {
        let handoff_id = HandoffId::new(Uuid::from_u128(1));
        for fact in [
            HandoffFact::Canceled { handoff_id },
            HandoffFact::Failed { handoff_id },
            HandoffFact::TimedOut { handoff_id },
        ] {
            let mut coordinator = completed_coordinator(handoff_id);
            let actions = coordinator.on_fact(fact.clone()).unwrap();
            assert!(
                actions
                    .iter()
                    .any(|issued| matches!(issued.action, HandoffAction::CancelDestination { .. })),
                "{fact:?} did not reopen cleanup; got {actions:?}"
            );
        }
    }

    /// `validate_transfer_timing` gates `SourceHeld` on `full_prompt_delay_ms()`,
    /// but `DestinationMissing` mode schedules `delay_ms(transferable_prompt_tokens)`
    /// instead. The transferable footprint is block-rounded at both KV backends
    /// and so legitimately exceeds `full_prompt_tokens`, which means a finite
    /// full-prompt delay does not imply a finite scheduled delay. The scheduled
    /// number must be validated at its point of use, not ~1200 lines later in
    /// `start_transfer`, which aborts the whole replay instead of rejecting the
    /// fact at the boundary that admitted it.
    #[test]
    fn destination_missing_validates_the_delay_it_actually_schedules() {
        let handoff_id = HandoffId::new(Uuid::from_u128(11));
        let timing = HandoffTransferTiming {
            mode: TransferTimingMode::DestinationMissing,
            full_prompt_tokens: 1,
            kv_bytes_per_token: Some(1024),
            bandwidth_gb_s: Some(1e-300),
        };
        // The boundary guard passes: the full-prompt delay is finite.
        validate_transfer_timing(timing).expect("full-prompt delay is finite");

        let mut coordinator =
            HandoffCoordinatorCore::new_with_fallback(handoff_id, HandoffOrder::SourceFirst, 0.0);
        let submit = coordinator.start().unwrap().remove(0);
        coordinator
            .on_action_outcome(submit.id, HandoffActionOutcome::Submitted)
            .unwrap();
        let reserve = coordinator
            .on_fact(HandoffFact::SourceHeld {
                handoff_id,
                transfer_timing: timing,
            })
            .unwrap()
            .remove(0);
        coordinator
            .on_action_outcome(reserve.id, HandoffActionOutcome::Accepted)
            .unwrap();

        let error = coordinator
            .on_fact(HandoffFact::DestinationReserved {
                handoff_id,
                transferable_prompt_tokens: 1 << 60,
            })
            .expect_err("a non-finite scheduled delay must be rejected at the coordinator");
        assert!(
            error.to_string().contains("invalid handoff transfer delay"),
            "unexpected error: {error}"
        );
    }

    /// The fallback delay is only `debug_assert`ed at construction, so in a
    /// release build a non-finite one reaches the event heap unless the
    /// point-of-use guard covers the post-`unwrap_or` value.
    #[test]
    fn a_non_finite_fallback_delay_is_rejected_before_it_is_issued() {
        let timing = HandoffTransferTiming {
            mode: TransferTimingMode::DestinationMissing,
            full_prompt_tokens: 4,
            kv_bytes_per_token: None,
            bandwidth_gb_s: None,
        };
        let handoff_id = HandoffId::new(Uuid::from_u128(12));
        let mut coordinator =
            HandoffCoordinatorCore::new_with_fallback(handoff_id, HandoffOrder::SourceFirst, 0.0);
        // Assigned rather than constructed: `new_with_fallback` only
        // `debug_assert`s finiteness, so a debug test build cannot reach the
        // release-build state this guard exists for through the constructor.
        coordinator.fallback_transfer_delay_ms = f64::INFINITY;
        let submit = coordinator.start().unwrap().remove(0);
        coordinator
            .on_action_outcome(submit.id, HandoffActionOutcome::Submitted)
            .unwrap();
        let reserve = coordinator
            .on_fact(HandoffFact::SourceHeld {
                handoff_id,
                transfer_timing: timing,
            })
            .unwrap()
            .remove(0);
        coordinator
            .on_action_outcome(reserve.id, HandoffActionOutcome::Accepted)
            .unwrap();

        assert!(
            coordinator
                .on_fact(HandoffFact::DestinationReserved {
                    handoff_id,
                    transferable_prompt_tokens: 4,
                })
                .is_err()
        );
    }

    /// A rejected outcome must not be journalled. Recording before validating
    /// made the rejection permanent, so re-reporting the same action id
    /// reported "conflicting outcome" rather than the true cause.
    #[test]
    fn a_rejected_outcome_is_not_recorded_in_the_journal() {
        let handoff_id = HandoffId::new(Uuid::from_u128(13));
        let mut coordinator =
            HandoffCoordinatorCore::new_with_fallback(handoff_id, HandoffOrder::SourceFirst, 1.0);
        let submit = coordinator.start().unwrap().remove(0);

        for _ in 0..2 {
            let error = coordinator
                .on_action_outcome(submit.id, HandoffActionOutcome::Accepted)
                .expect_err("SubmitPrefill only accepts Submitted");
            assert!(
                error.to_string().contains("invalid handoff action outcome"),
                "unexpected error: {error}"
            );
        }

        // The rejected outcome left no trace, so the correct one still applies.
        assert!(
            coordinator
                .on_action_outcome(submit.id, HandoffActionOutcome::Submitted)
                .is_ok()
        );
    }

    #[test]
    fn negative_bandwidth_is_rejected() {
        let timing = HandoffTransferTiming {
            mode: TransferTimingMode::FullPrompt,
            full_prompt_tokens: 10,
            kv_bytes_per_token: Some(1024),
            bandwidth_gb_s: Some(-1.0),
        };
        assert!(validate_transfer_timing(timing).is_err());
    }
}
