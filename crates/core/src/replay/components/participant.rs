// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Externally owned reactive workload hosted by the native replay event loop.
//!
//! A [`WorkloadParticipant`] is polled by the real `AggRuntime`/`DisaggRuntime`
//! `run` loop through the replay admission queue. It owns its own workload
//! state and policy; replay owns the clock. All times are the runtime's
//! logical milliseconds (`f64`); a participant must never read wall time.
//!
//! Contract with the host loop:
//! - `next_wakeup_ms` is the earliest logical time at which `advance_to` could
//!   produce a submission. A participant made runnable by a callback (for
//!   example a terminal that unblocks a successor) reports a wakeup at or
//!   before the callback time. Return `None` while parked on an engine fact;
//!   a present deadline must be finite. This is a repeatable peek: observing it
//!   must not consume readiness or change the next result without a new fact.
//! - `advance_to(now_ms)` runs the workload until it parks and returns what it
//!   submitted. After it returns, `next_wakeup_ms` must be `None` or strictly
//!   later than `now_ms` unless more work was made runnable by a later callback;
//!   otherwise the host would revisit the same instant forever.
//! - `on_output_token`, `on_terminal` and `on_quiescent` only record facts,
//!   each stamped with the authoritative logical `now_ms`. They must not run
//!   workload logic; the host calls `advance_to` afterwards, at the same
//!   logical timestamp, in the same event-loop iteration and before it makes
//!   any further engine decision at that instant. A participant timer due at
//!   `T` wins ties: the host polls the participant (`advance_to(T)`) before it
//!   settles pass completions at `T`, so a cancel at `T` discards outputs of a
//!   pass completing at `T`. It polls again after completions, so
//!   `advance_to` may run more than once per instant and must be idempotent
//!   when nothing changed.
//! - `ParticipantStep::cancels` are applied by the host through the runtime's
//!   native cancel path before the same step's submissions are admitted and
//!   before the affected worker commits its next pass. A cancel of a request
//!   that is unknown or already terminal is a no-op (its terminal was already
//!   delivered). A request submitted in this same step cannot be canceled by
//!   it. Canceling delivers exactly one `on_terminal(Canceled)`.
//! - `on_terminal` is the client-visible end of a request; `on_quiescent` is
//!   the point at which the physical servers released it. Encoder batches,
//!   committed passes and disaggregated transfer/cleanup can outlive client
//!   cancellation. Each callback fires once. Submitted UUIDs identify one
//!   occurrence and must not be reused within a run, including after quiescence.
//! - The trait is deliberately not `Send`: a participant may hold
//!   thread-local (`Rc`, local executor) state and is constructed and run on
//!   the thread that calls `Replayer::run`.

use anyhow::Result;
use uuid::Uuid;

use crate::replay::ReplayTerminalStatus;
use crate::replay::protocol::DirectRequest;

/// Requests a participant submitted during one `advance_to` call.
#[derive(Debug, Default)]
pub struct ParticipantStep {
    /// Requests to admit now. Each must carry `uuid: Some(..)`, the identity
    /// later reported through `on_output_token` / `on_terminal`. The host sets
    /// the arrival timestamp to the logical `now_ms`.
    pub submits: Vec<DirectRequest>,
    /// Requests to cancel now, by the uuid they were submitted with.
    pub cancels: Vec<Uuid>,
}

pub trait WorkloadParticipant {
    /// Earliest logical time the participant can make progress, if any.
    fn next_wakeup_ms(&mut self) -> Option<f64>;

    /// Run the workload at logical time `now_ms` until it parks.
    fn advance_to(&mut self, now_ms: f64, cluster_in_flight: usize) -> Result<ParticipantStep>;

    /// Mailbox fact: `request_id` emitted `token_id` at logical `now_ms`.
    fn on_output_token(&mut self, request_id: Uuid, token_id: u32, now_ms: f64) -> Result<()>;

    /// Mailbox fact: the engine reported prompt cache-read usage. Absence of
    /// this callback is unknown usage, not a measured zero.
    fn on_prompt_cache_read_tokens(
        &mut self,
        _request_id: Uuid,
        _tokens: usize,
        _now_ms: f64,
    ) -> Result<()> {
        Ok(())
    }

    /// Mailbox fact: `request_id` reached a terminal state at `now_ms`.
    fn on_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
        status: ReplayTerminalStatus,
    ) -> Result<()>;

    /// Mailbox fact: the servers fully released `request_id` at `now_ms`.
    /// Default: ignore.
    fn on_quiescent(&mut self, _request_id: Uuid, _now_ms: f64) -> Result<()> {
        Ok(())
    }

    /// True once the workload will submit no more requests and nothing is parked.
    fn is_drained(&self) -> bool;

    /// Requests submitted or known so far, for progress reporting only.
    fn total_requests(&self) -> usize;
}
