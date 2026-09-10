// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`SteppableReplay`]: the offline replay runtimes exposed as passive,
//! externally clocked, dynamically fed engines.
//!
//! [`crate::replay::Replayer`] drives itself over a workload authored up front.
//! This seam inverts that. The caller owns the clock and the loop, submits
//! requests as they emerge, and advances the runtime one logical timestamp at a
//! time. Each step writes arrival, admission, token, and terminal facts into the
//! runtime's own collector and reports the per-request events the caller needs
//! to wake its own futures. Batching, KV cache, prefix caching, chunked
//! prefill, placement, and disaggregated handoff stay inside the runtimes; this
//! module adds nothing but the seam.
//!
//! Every implementor drives the same runtime code paths `run()` drives. The one
//! difference is the drain: a step that frees an in-flight slot returns to the
//! caller *before* the freed worker is committed to its next pass, so a
//! replacement submitted at that instant is batched at that instant — the
//! evaluate/update split of a hardware delta cycle, reproducing the in-drain
//! admission `run()` gets for free from its own arrival queue.
//!
//! That guarantee is unconditional, so the runtimes hold the instant on a freed
//! slot rather than on an observed token. The two are not the same predicate: a
//! terminal signal may carry no token at all — an admission rejection, or a
//! terminal whose last token was emitted by an earlier signal — and gating time
//! advance on tokens alone would let those terminals slip past their instant and
//! land their replacement one step late, which is the exact failure this seam
//! exists to remove.

use std::collections::VecDeque;

use uuid::Uuid;

use crate::replay::agg::AggRuntimeImpl;
use crate::replay::components::{AdmissionQueue, NoReplayMetadata, ReplayMode};
use crate::replay::core::NoEngineEvents;
use crate::replay::core::round_robin::AggregatedRoundRobinPlacement;
use crate::replay::engine::{ReplayEngineConfig, ReplayEngineFactory};
use crate::replay::protocol::DirectRequest;
use crate::replay::{ReplayReport, ReplayTerminalStatus, SlaThresholds, WorkerStage};

/// One per-request event produced by a [`SteppableReplay`] step.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineEvent {
    /// The request this event belongs to.
    pub uuid: Uuid,
    /// True when this event carries an output token. The caller gates
    /// first-token off the first such event. Always equals `token_id.is_some()`.
    pub emitted_token: bool,
    /// The exact output token ID when this is a token event. A driver that
    /// accumulates a conversation reconstructs the next turn's prompt from this
    /// value rather than inventing token identities.
    pub token_id: Option<u32>,
    /// Terminal classification, or `None` for a token-only event. Distinguishes
    /// rejection and failure from successful completion.
    pub terminal_status: Option<ReplayTerminalStatus>,
}

impl EngineEvent {
    /// The single construction point, so `emitted_token` cannot diverge from
    /// `token_id`.
    fn new(
        uuid: Uuid,
        token_id: Option<u32>,
        terminal_status: Option<ReplayTerminalStatus>,
    ) -> Self {
        Self {
            uuid,
            emitted_token: token_id.is_some(),
            token_id,
            terminal_status,
        }
    }

    /// One output token for `uuid`.
    pub fn token(uuid: Uuid, token_id: u32) -> Self {
        Self::new(uuid, Some(token_id), None)
    }

    /// A terminal event carrying no output token.
    pub fn terminal(uuid: Uuid, status: ReplayTerminalStatus) -> Self {
        Self::new(uuid, None, Some(status))
    }
}

/// The result of one runtime step.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepOutcome {
    /// Simulated time in milliseconds after this step. The caller advances its
    /// own clock to here.
    pub end_ms: f64,
    /// Per-request events emitted during this step, tokens first.
    pub events: Vec<EngineEvent>,
}

/// An offline replay runtime as a steppable, clock-injected, dynamically fed
/// engine. Every topology implements it, so one caller loop drives any of them.
pub trait SteppableReplay {
    /// Current simulated time in milliseconds. Submissions arrive here, and it
    /// is the unit every collector timestamp is expressed in.
    fn now_ms(&self) -> f64;

    /// Advance the simulated clock to `now_ms`, used when the caller skips an
    /// idle gap with no engine work. Monotonic: a time at or before the current
    /// one is a no-op.
    fn advance_now_ms(&mut self, now_ms: f64);

    /// Admit `request` at the current simulated time. The returned id
    /// correlates it with later [`EngineEvent`]s.
    fn submit(&mut self, request: DirectRequest) -> anyhow::Result<Uuid>;

    /// Cancel one live request. Returns its terminal event when cancellation
    /// won the race; already-terminal or unknown requests return `None`.
    fn cancel(&mut self, uuid: Uuid) -> anyhow::Result<Option<EngineEvent>>;

    /// Advance one logical timestamp of work.
    fn step(&mut self) -> anyhow::Result<StepOutcome> {
        self.step_until(f64::INFINITY)
    }

    /// Advance at most through `until_ms`, emitting events at or before that
    /// deadline. When the next internal event is later than the deadline the
    /// runtime stops at the deadline, so an external discrete-event driver can
    /// interleave its own arrivals and firing gates without changing batch
    /// composition.
    fn step_until(&mut self, until_ms: f64) -> anyhow::Result<StepOutcome>;

    /// Next simulated time at which the runtime can make progress, for the
    /// caller's discrete-event pump. The caller advances to
    /// `min(its own next deadline, this)` and then steps. `None` when idle.
    fn next_event_ms(&mut self) -> Option<f64>;

    /// True when no submitted request work remains.
    fn is_idle(&self) -> bool;

    /// Number of submitted requests that have not reached a terminal.
    fn in_flight(&self) -> usize;

    /// Retain the complete per-request causality records a raw-record exporter
    /// needs. Off by default because it is not free.
    fn set_capture_per_request(&mut self, capture: bool);

    /// Configure the goodput thresholds the report classifies against.
    fn set_sla_thresholds(&mut self, sla: SlaThresholds);

    /// Measured `(ttft_ms, mean_itl_ms)` for `uuid` once it has a first token.
    fn request_latencies(&self, uuid: Uuid) -> Option<(f64, f64)>;

    /// First scheduler admission `(at_ms, reused_input_tokens)` for `uuid`.
    fn request_admission(&self, uuid: Uuid) -> Option<(f64, usize)>;

    /// Output tokens actually emitted for `uuid`.
    fn actual_output_length(&self, uuid: Uuid) -> Option<usize>;

    /// Drain the accumulated measurements into a report stamped with `wall_ms`,
    /// leaving the runtime's collector empty.
    fn take_report(&mut self, wall_ms: f64) -> ReplayReport;
}

/// Concrete aggregated runtime behind the steppable seam.
type SteppableAggRuntime =
    AggRuntimeImpl<AggregatedRoundRobinPlacement<()>, NoEngineEvents, NoReplayMetadata>;

/// Externally clocked replay admits every request the caller submits; the
/// caller owns whatever concurrency limit it wants to impose.
fn steppable_mode() -> ReplayMode {
    ReplayMode::Concurrency {
        max_in_flight: usize::MAX,
    }
}

/// Tracks which submitted requests are still live so a step can report the
/// terminals reached during it.
///
/// The runtimes classify a terminal at several places — ordinary completion,
/// scheduler rejection, and handoff failure. Rather than fan an event out of
/// each, the seam asks the collector which of the requests it is still
/// following have acquired a terminal status. The set is bounded by the
/// caller's own concurrency, so each step costs one lookup per live request.
#[derive(Default)]
struct LiveRequests {
    uuids: Vec<Uuid>,
}

impl LiveRequests {
    fn insert(&mut self, uuid: Uuid) {
        self.uuids.push(uuid);
    }

    fn contains(&self, uuid: Uuid) -> bool {
        self.uuids.contains(&uuid)
    }

    fn len(&self) -> usize {
        self.uuids.len()
    }
}

/// Build the aggregated role factory one steppable runtime runs on.
fn aggregated_role_factory(
    engine: &ReplayEngineConfig,
    factory: &ReplayEngineFactory,
) -> anyhow::Result<crate::replay::ReplayRoleFactory> {
    Ok(factory.role_factory(engine, WorkerStage::Aggregated, false)?)
}

/// Aggregated multi-worker topology as a [`SteppableReplay`].
pub struct SteppableAgg {
    runtime: SteppableAggRuntime,
    live: LiveRequests,
}

impl SteppableAgg {
    /// Build an aggregated runtime with `num_workers` round-robin engines.
    pub fn new(
        engine: ReplayEngineConfig,
        factory: &ReplayEngineFactory,
        num_workers: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(num_workers > 0, "num_workers must be positive");
        let role_factory = aggregated_role_factory(&engine, factory)?;
        let runtime = SteppableAggRuntime::new_composed(
            role_factory,
            AdmissionQueue::new_requests(VecDeque::new(), steppable_mode()),
            num_workers,
            None,
            |dp_size, topology| Ok(AggregatedRoundRobinPlacement::new(dp_size, topology)),
        )?
        .into_steppable();
        Ok(Self {
            runtime,
            live: LiveRequests::default(),
        })
    }
}

impl SteppableReplay for SteppableAgg {
    fn now_ms(&self) -> f64 {
        self.runtime.now_ms()
    }

    fn advance_now_ms(&mut self, now_ms: f64) {
        if now_ms.is_finite() && now_ms > self.runtime.now_ms() {
            self.runtime.advance_now_ms(now_ms);
        }
    }

    fn submit(&mut self, request: DirectRequest) -> anyhow::Result<Uuid> {
        if let Some(uuid) = request.uuid
            && self.live.contains(uuid)
        {
            anyhow::bail!("steppable replay request {uuid} is already live");
        }
        let uuid = self.runtime.submit_dynamic(request)?;
        self.live.insert(uuid);
        Ok(uuid)
    }

    fn cancel(&mut self, uuid: Uuid) -> anyhow::Result<Option<EngineEvent>> {
        let status = self.runtime.cancel_dynamic(uuid)?;
        if status.is_some() {
            self.live.uuids.retain(|candidate| *candidate != uuid);
        }
        Ok(status.map(|status| EngineEvent::terminal(uuid, status)))
    }

    fn step_until(&mut self, until_ms: f64) -> anyhow::Result<StepOutcome> {
        let end_ms = self.runtime.step_dynamic_until(until_ms)?;
        let tokens = self.runtime.take_step_tokens();
        let mut events = tokens
            .into_iter()
            .map(|(uuid, token_id)| EngineEvent::token(uuid, token_id))
            .collect::<Vec<_>>();
        for (uuid, status) in self.runtime.take_step_terminals() {
            self.live.uuids.retain(|candidate| *candidate != uuid);
            events.push(EngineEvent::terminal(uuid, status));
        }
        Ok(StepOutcome { end_ms, events })
    }

    fn next_event_ms(&mut self) -> Option<f64> {
        self.runtime.next_timestamp()
    }

    fn is_idle(&self) -> bool {
        self.runtime.is_workload_done()
    }

    fn in_flight(&self) -> usize {
        self.live.len()
    }

    fn set_capture_per_request(&mut self, capture: bool) {
        self.runtime
            .collector_mut()
            .set_capture_per_request(capture);
    }

    fn set_sla_thresholds(&mut self, sla: SlaThresholds) {
        self.runtime.collector_mut().set_sla_thresholds(sla);
    }

    fn request_latencies(&self, uuid: Uuid) -> Option<(f64, f64)> {
        self.runtime.collector().request_latencies(uuid)
    }

    fn request_admission(&self, uuid: Uuid) -> Option<(f64, usize)> {
        self.runtime.collector().request_admission(uuid)
    }

    fn actual_output_length(&self, uuid: Uuid) -> Option<usize> {
        self.runtime.collector().actual_output_length(uuid)
    }

    fn take_report(&mut self, wall_ms: f64) -> ReplayReport {
        self.runtime.take_report_dynamic(wall_ms)
    }
}

/// Single-worker topology as a [`SteppableReplay`]: one aggregated engine with
/// no placement choice to make.
pub struct SteppableEngine {
    inner: SteppableAgg,
}

impl SteppableEngine {
    /// Build a one-worker aggregated runtime.
    pub fn new(engine: ReplayEngineConfig, factory: &ReplayEngineFactory) -> anyhow::Result<Self> {
        Ok(Self {
            inner: SteppableAgg::new(engine, factory, 1)?,
        })
    }
}

impl SteppableReplay for SteppableEngine {
    fn now_ms(&self) -> f64 {
        self.inner.now_ms()
    }

    fn advance_now_ms(&mut self, now_ms: f64) {
        self.inner.advance_now_ms(now_ms);
    }

    fn submit(&mut self, request: DirectRequest) -> anyhow::Result<Uuid> {
        self.inner.submit(request)
    }

    fn cancel(&mut self, uuid: Uuid) -> anyhow::Result<Option<EngineEvent>> {
        self.inner.cancel(uuid)
    }

    fn step_until(&mut self, until_ms: f64) -> anyhow::Result<StepOutcome> {
        self.inner.step_until(until_ms)
    }

    fn next_event_ms(&mut self) -> Option<f64> {
        self.inner.next_event_ms()
    }

    fn is_idle(&self) -> bool {
        self.inner.is_idle()
    }

    fn in_flight(&self) -> usize {
        self.inner.in_flight()
    }

    fn set_capture_per_request(&mut self, capture: bool) {
        self.inner.set_capture_per_request(capture);
    }

    fn set_sla_thresholds(&mut self, sla: SlaThresholds) {
        self.inner.set_sla_thresholds(sla);
    }

    fn request_latencies(&self, uuid: Uuid) -> Option<(f64, f64)> {
        self.inner.request_latencies(uuid)
    }

    fn request_admission(&self, uuid: Uuid) -> Option<(f64, usize)> {
        self.inner.request_admission(uuid)
    }

    fn actual_output_length(&self, uuid: Uuid) -> Option<usize> {
        self.inner.actual_output_length(uuid)
    }

    fn take_report(&mut self, wall_ms: f64) -> ReplayReport {
        self.inner.take_report(wall_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(uuid: u128, input_length: usize, max_output_tokens: usize) -> DirectRequest {
        DirectRequest {
            tokens: (0..input_length as u32).collect(),
            max_output_tokens,
            uuid: Some(Uuid::from_u128(uuid)),
            arrival_timestamp_ms: Some(0.0),
            ..Default::default()
        }
    }

    fn drain(engine: &mut dyn SteppableReplay) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        for _ in 0..10_000 {
            if engine.is_idle() {
                return events;
            }
            events.extend(engine.step().unwrap().events);
        }
        panic!("steppable replay did not drain");
    }

    #[test]
    fn aggregated_replay_emits_one_token_event_per_output_token() {
        let mut engine = SteppableAgg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
        )
        .unwrap();
        engine.set_capture_per_request(true);
        let uuid = engine.submit(request(1, 128, 16)).unwrap();

        let events = drain(&mut engine);

        assert_eq!(
            events.iter().filter(|event| event.emitted_token).count(),
            16
        );
        assert_eq!(
            events.last().map(|event| event.terminal_status),
            Some(Some(ReplayTerminalStatus::Completed))
        );
        assert_eq!(engine.in_flight(), 0);
        let report = engine.take_report(engine.now_ms());
        assert_eq!(report.request_counts.completed_requests, 1);
        assert_eq!(report.request_counts.total_output_tokens, 16);
        assert_eq!(uuid, Uuid::from_u128(1));
    }

    #[test]
    fn a_bounded_step_never_crosses_the_external_deadline() {
        let mut engine = SteppableAgg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
        )
        .unwrap();
        engine.submit(request(2, 128, 16)).unwrap();

        let deadline_ms = 0.000_001;
        let outcome = engine.step_until(deadline_ms).unwrap();

        assert!(outcome.end_ms <= deadline_ms);
        assert_eq!(engine.now_ms(), deadline_ms);
        assert!(!engine.is_idle());
    }

    #[test]
    fn a_terminal_without_a_token_holds_its_instant() {
        let mut config = ReplayEngineConfig::default();
        // Anything longer than this is rejected at admission, and an admission
        // rejection is a terminal signal that carries no token.
        config.rank.max_model_len = Some(256);
        let mut engine = SteppableAgg::new(config, &ReplayEngineFactory::new(), 1).unwrap();
        // Settle one request before the instant under test, leaving its sole
        // scheduler available for the tokenless terminal and replacement.
        engine.submit(request(4, 128, 16)).unwrap();
        drain(&mut engine);
        let instant_ms = engine.now_ms();

        // The worker is idle, so this request is admitted, rejected, and freed
        // inside one drain — with no token anywhere in the step.
        engine.submit(request(5, 4096, 16)).unwrap();
        let outcome = engine.step().unwrap();

        assert!(
            outcome
                .events
                .iter()
                .any(|event| event.terminal_status == Some(ReplayTerminalStatus::Rejected)),
            "expected the oversized request to be rejected: {:?}",
            outcome.events
        );
        assert!(!outcome.events.iter().any(|event| event.emitted_token));
        assert_eq!(
            outcome.end_ms, instant_ms,
            "a step that frees an in-flight slot must return its instant, so the \
             caller can submit a replacement before the freed worker is committed"
        );
        let replacement = engine.submit(request(6, 128, 16)).unwrap();
        engine.step_until(instant_ms).unwrap();
        assert_eq!(engine.request_admission(replacement).unwrap().0, instant_ms);
    }

    #[test]
    fn non_finite_clock_advance_is_ignored() {
        let mut engine = SteppableAgg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
        )
        .unwrap();
        engine.advance_now_ms(f64::INFINITY);
        assert_eq!(engine.now_ms(), 0.0);
    }

    #[test]
    fn cancellation_is_immediately_terminal() {
        fn assert_canceled(engine: &mut dyn SteppableReplay, uuid: u128) {
            let uuid = engine.submit(request(uuid, 128, 64)).unwrap();
            let event = engine.cancel(uuid).unwrap();
            assert_eq!(
                event.map(|event| event.terminal_status),
                Some(Some(ReplayTerminalStatus::Canceled))
            );
            assert_eq!(engine.in_flight(), 0);
            assert!(engine.cancel(uuid).unwrap().is_none());
        }

        let factory = ReplayEngineFactory::new();
        let mut aggregated = SteppableAgg::new(ReplayEngineConfig::default(), &factory, 1).unwrap();
        assert_canceled(&mut aggregated, 10);

        let mut single_worker =
            SteppableEngine::new(ReplayEngineConfig::default(), &factory).unwrap();
        assert_canceled(&mut single_worker, 11);
    }

    #[test]
    fn steppable_replay_rejects_duplicate_live_request_ids() {
        fn assert_rejected(engine: &mut dyn SteppableReplay) {
            let first = engine.submit(request(21, 128, 16)).unwrap();
            let error = engine.submit(request(21, 128, 16)).unwrap_err();

            assert!(error.to_string().contains("already live"), "{error}");
            assert_eq!(engine.in_flight(), 1);
            assert_eq!(
                engine
                    .cancel(first)
                    .unwrap()
                    .map(|event| event.terminal_status),
                Some(Some(ReplayTerminalStatus::Canceled))
            );
        }

        let factory = ReplayEngineFactory::new();
        let mut aggregated = SteppableAgg::new(ReplayEngineConfig::default(), &factory, 1).unwrap();
        assert_rejected(&mut aggregated);
    }

    #[test]
    fn steppable_constructors_reject_zero_workers() {
        let factory = ReplayEngineFactory::new();

        assert!(SteppableAgg::new(ReplayEngineConfig::default(), &factory, 0).is_err());
    }
}
