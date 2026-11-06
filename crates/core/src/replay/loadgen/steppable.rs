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
//!
//! Only a terminal frees this externally visible admission boundary. A
//! nonterminal token can commit the next pass before [`SteppableReplay::step`]
//! returns, so a request submitted at that same timestamp can join a later batch
//! than it would in [`crate::replay::Replayer::run`]. A caller that needs that
//! batch composition must submit its known same-timestamp arrivals before it
//! advances the steppable runtime.
//!
//! [`SteppableDisagg`] is stricter: it also holds the instant on a nonterminal
//! token, so work the token releases is submitted before the next pass commits.

use std::collections::{HashSet, VecDeque};

use uuid::Uuid;

use super::ReplayRequestPayload;
use crate::replay::agg::AggRuntimeImpl;
use crate::replay::components::{
    AdmissionQueue, NoReplayMetadata, ReplayAdmissionMetadata, ReplayEngineObservation, ReplayMode,
};
use crate::replay::core::round_robin::{AggregatedRoundRobinPlacement, PoolRoundRobinPlacement};
use crate::replay::core::{NoEngineEvents, PlacementPolicy, WorkerTopology};
use crate::replay::disagg::DisaggRuntimeImpl;
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
    /// correlates it with later [`EngineEvent`]s. Explicit UUIDs must be unique
    /// among live requests and measurements retained in the current report
    /// epoch. A successful [`Self::take_report`] permits reuse in the next epoch.
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
    /// leaving the runtime's collector empty. Requires an idle runtime. Aggregate
    /// rates cover simulated time through the current clock, including idle gaps
    /// since the preceding report (or time zero for the first report). Request
    /// timestamps remain absolute, and capture/SLA settings persist. Previously
    /// retained UUIDs may be reused; request queries then refer to the new epoch.
    fn take_report(&mut self, wall_ms: f64) -> anyhow::Result<ReplayReport>;
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

/// Tracks submitted requests until a terminal event removes them.
#[derive(Default)]
struct LiveRequests {
    uuids: HashSet<Uuid>,
}

impl LiveRequests {
    fn insert(&mut self, uuid: Uuid) {
        self.uuids.insert(uuid);
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
        if let Some(uuid) = request.uuid
            && self.runtime.collector().contains_request(uuid)
        {
            anyhow::bail!("steppable replay request {uuid} is already retained");
        }
        let uuid = self.runtime.submit_dynamic(request)?;
        self.live.insert(uuid);
        Ok(uuid)
    }

    fn cancel(&mut self, uuid: Uuid) -> anyhow::Result<Option<EngineEvent>> {
        let status = self.runtime.cancel_dynamic(uuid)?;
        if status.is_some() {
            self.live.uuids.remove(&uuid);
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
            self.live.uuids.remove(&uuid);
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

    fn take_report(&mut self, wall_ms: f64) -> anyhow::Result<ReplayReport> {
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

    fn take_report(&mut self, wall_ms: f64) -> anyhow::Result<ReplayReport> {
        self.inner.take_report(wall_ms)
    }
}

/// Disaggregated prefill/decode topology as a [`SteppableReplay`], driving the
/// same [`DisaggRuntimeImpl`] that `run()` drives: its prefill and decode
/// engines, KV handoff and transfer links, and collector. The seam adds no
/// workload policy of its own.
///
/// The placement policy is a parameter so a router that makes real KV-aware
/// placement decisions composes through [`Self::with_placements`] without a new
/// seam; the default is round-robin per pool.
///
/// Passive-step contract (see `DisaggRuntimeImpl::step_dynamic_until`): a step
/// returns at its current instant, before the freed or completing worker is
/// committed to its next pass, whenever it surfaced an output token or a
/// client terminal. A token can release dependent work and a terminal frees a
/// slot, so the caller's submissions at that instant join the next physical
/// batch. Virtual time never advances past an outstanding token or terminal,
/// nor past the next internal event. [`EngineEvent`] terminals are the client's
/// causal terminals; server-side cleanup (prefill release, transfer teardown)
/// keeps running afterwards and [`SteppableReplay::is_idle`] stays false until
/// it quiesces.
pub struct SteppableDisagg<
    PlacementPolicyImpl = PoolRoundRobinPlacement<()>,
    Observation = NoEngineEvents,
    Metadata = NoReplayMetadata,
> where
    Observation: ReplayEngineObservation,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl:
        PlacementPolicy<ReplayRequestPayload, Metadata = Metadata, Observation = Observation::Batch>,
{
    runtime: DisaggRuntimeImpl<PlacementPolicyImpl, Observation, Metadata>,
    live: LiveRequests,
}

impl SteppableDisagg {
    /// Build a round-robin disaggregated runtime with the given worker counts.
    pub fn new(
        engine: ReplayEngineConfig,
        factory: &ReplayEngineFactory,
        num_prefill_workers: usize,
        num_decode_workers: usize,
    ) -> anyhow::Result<Self> {
        Self::with_placements(
            engine,
            factory,
            num_prefill_workers,
            num_decode_workers,
            |_, prefill, _, decode| {
                Ok((
                    PoolRoundRobinPlacement::new(prefill),
                    PoolRoundRobinPlacement::new(decode),
                ))
            },
        )
    }
}

impl<PlacementPolicyImpl, Observation, Metadata>
    SteppableDisagg<PlacementPolicyImpl, Observation, Metadata>
where
    Observation: ReplayEngineObservation,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl:
        PlacementPolicy<ReplayRequestPayload, Metadata = Metadata, Observation = Observation::Batch>,
{
    /// Build a disaggregated runtime whose prefill and decode placements come
    /// from `create_placements(prefill_dp_size, prefill_topology,
    /// decode_dp_size, decode_topology)`.
    pub fn with_placements(
        engine: ReplayEngineConfig,
        factory: &ReplayEngineFactory,
        num_prefill_workers: usize,
        num_decode_workers: usize,
        create_placements: impl FnOnce(
            u32,
            Vec<WorkerTopology>,
            u32,
            Vec<WorkerTopology>,
        ) -> anyhow::Result<(PlacementPolicyImpl, PlacementPolicyImpl)>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            num_prefill_workers > 0 && num_decode_workers > 0,
            "disaggregated worker counts must be positive"
        );
        let config = crate::replay::OfflineDisaggReplayConfig {
            // `OfflineDisaggReplayConfig::{prefill,decode}_factory` ignore their
            // flag, so the observation adapter's KV-event requirement must be
            // applied here, exactly as `Replayer` does.
            prefill_factory: factory.role_factory(
                &engine,
                WorkerStage::Prefill,
                Observation::capture_engine_kv_events(WorkerStage::Prefill),
            )?,
            decode_factory: factory.role_factory(
                &engine,
                WorkerStage::Decode,
                Observation::capture_engine_kv_events(WorkerStage::Decode),
            )?,
            prefill_startup_time_ms: None,
            decode_startup_time_ms: None,
            num_prefill_workers,
            num_decode_workers,
            handoff_latency_ms: 0.0,
        };
        let runtime = DisaggRuntimeImpl::new_composed(
            &config,
            AdmissionQueue::new_requests(VecDeque::new(), steppable_mode()),
            false,
            create_placements,
        )?
        .into_steppable();
        Ok(Self {
            runtime,
            live: LiveRequests::default(),
        })
    }
}

impl<PlacementPolicyImpl, Observation, Metadata> SteppableReplay
    for SteppableDisagg<PlacementPolicyImpl, Observation, Metadata>
where
    Observation: ReplayEngineObservation,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl:
        PlacementPolicy<ReplayRequestPayload, Metadata = Metadata, Observation = Observation::Batch>,
{
    fn now_ms(&self) -> f64 {
        self.runtime.now_ms()
    }

    fn advance_now_ms(&mut self, now_ms: f64) {
        // Only an idle runtime has no internal event to skip over; with work
        // outstanding the caller must step to it instead.
        if now_ms.is_finite() && now_ms > self.runtime.now_ms() && self.runtime.is_workload_done() {
            self.runtime.advance_now_ms(now_ms);
        }
    }

    fn submit(&mut self, request: DirectRequest) -> anyhow::Result<Uuid> {
        if let Some(uuid) = request.uuid {
            if self.live.contains(uuid) {
                anyhow::bail!("steppable replay request {uuid} is already live");
            }
            if self.runtime.collector().contains_request(uuid) {
                anyhow::bail!("steppable replay request {uuid} is already retained");
            }
        }
        let uuid = self
            .runtime
            .submit_dynamic_payload(ReplayRequestPayload::materialized(request))?;
        self.live.insert(uuid);
        Ok(uuid)
    }

    fn cancel(&mut self, uuid: Uuid) -> anyhow::Result<Option<EngineEvent>> {
        let status = self.runtime.cancel_dynamic(uuid)?;
        if status.is_some() {
            self.live.uuids.remove(&uuid);
        }
        Ok(status.map(|status| EngineEvent::terminal(uuid, status)))
    }

    fn step_until(&mut self, until_ms: f64) -> anyhow::Result<StepOutcome> {
        let end_ms = self.runtime.step_dynamic_until(until_ms)?;
        let mut events = self
            .runtime
            .take_step_tokens()
            .into_iter()
            .map(|(uuid, token_id)| EngineEvent::token(uuid, token_id))
            .collect::<Vec<_>>();
        for (uuid, status) in self.runtime.take_step_terminals() {
            self.live.uuids.remove(&uuid);
            events.push(EngineEvent::terminal(uuid, status));
        }
        // Server-side quiescence needs no caller action in this seam.
        self.runtime.take_step_quiescent();
        Ok(StepOutcome { end_ms, events })
    }

    fn next_event_ms(&mut self) -> Option<f64> {
        self.runtime.next_dynamic_timestamp()
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

    fn take_report(&mut self, wall_ms: f64) -> anyhow::Result<ReplayReport> {
        self.runtime.take_report_dynamic(wall_ms)
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
        let report = engine.take_report(engine.now_ms()).unwrap();
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
    fn a_cascade_terminal_is_surfaced_at_the_held_instant() {
        let mut config = ReplayEngineConfig::default();
        config.rank.max_model_len = Some(256);
        let mut engine = SteppableAgg::new(config, &ReplayEngineFactory::new(), 1).unwrap();
        engine.submit(request(40, 128, 16)).unwrap();
        loop {
            if engine
                .step()
                .unwrap()
                .events
                .iter()
                .any(|event| event.terminal_status.is_some())
            {
                break;
            }
        }

        engine.submit(request(41, 4096, 16)).unwrap();
        let replacement = engine.submit(request(42, 128, 16)).unwrap();
        let outcome = loop {
            let outcome = engine.step().unwrap();
            if outcome
                .events
                .iter()
                .any(|event| event.terminal_status == Some(ReplayTerminalStatus::Rejected))
            {
                break outcome;
            }
        };

        assert!(
            outcome
                .events
                .iter()
                .any(|event| { event.terminal_status == Some(ReplayTerminalStatus::Rejected) })
        );
        assert_eq!(outcome.end_ms, engine.now_ms());
        assert!(engine.request_admission(replacement).is_some());
    }

    #[test]
    fn report_rejects_non_finite_wall_time() {
        let mut engine = SteppableAgg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
        )
        .unwrap();

        assert!(engine.take_report(f64::NAN).is_err());
    }

    #[test]
    fn report_requires_an_idle_engine() {
        let mut engine = SteppableAgg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
        )
        .unwrap();
        engine.submit(request(19, 128, 16)).unwrap();

        assert!(engine.take_report(0.0).is_err());
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
    fn submission_exposes_the_current_timestamp_as_the_next_event() {
        let mut engine = SteppableAgg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
        )
        .unwrap();
        let now_ms = engine.now_ms();

        engine.submit(request(20, 128, 16)).unwrap();

        assert_eq!(engine.next_event_ms(), Some(now_ms));
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
            assert!(engine.is_idle());
            assert_eq!(engine.next_event_ms(), None);
        }

        let factory = ReplayEngineFactory::new();
        let mut aggregated = SteppableAgg::new(ReplayEngineConfig::default(), &factory, 1).unwrap();
        assert_canceled(&mut aggregated, 10);

        let mut multi_worker =
            SteppableAgg::new(ReplayEngineConfig::default(), &factory, 2).unwrap();
        assert_canceled(&mut multi_worker, 12);

        let mut single_worker =
            SteppableEngine::new(ReplayEngineConfig::default(), &factory).unwrap();
        assert_canceled(&mut single_worker, 11);
    }

    #[test]
    fn canceling_a_started_final_pass_retires_engine_ownership() {
        let mut engine =
            SteppableEngine::new(ReplayEngineConfig::default(), &ReplayEngineFactory::new())
                .unwrap();
        let uuid = engine.submit(request(13, 128, 1)).unwrap();

        engine.step_until(0.0).unwrap();
        engine.cancel(uuid).unwrap();
        engine.step().unwrap();

        assert_eq!(engine.in_flight(), 0);
        assert!(engine.is_idle());
        assert_eq!(engine.next_event_ms(), None);
        assert!(engine.take_report(engine.now_ms()).is_ok());
    }

    #[test]
    fn failed_dynamic_submission_does_not_retain_the_request() {
        let mut engine =
            SteppableEngine::new(ReplayEngineConfig::default(), &ReplayEngineFactory::new())
                .unwrap();
        let mut invalid = request(14, 128, 1);
        invalid.preferred_dp_rank = Some(1);

        assert!(engine.submit(invalid).is_err());
        assert_eq!(engine.in_flight(), 0);
        assert!(engine.cancel(Uuid::from_u128(14)).unwrap().is_none());

        engine.submit(request(14, 128, 1)).unwrap();
        drain(&mut engine);
        let report = engine.take_report(engine.now_ms()).unwrap();
        assert_eq!(report.request_counts.num_requests, 1);
        assert_eq!(report.request_counts.completed_requests, 1);
    }

    #[test]
    fn report_duration_starts_at_the_previous_report_boundary() {
        let mut engine =
            SteppableEngine::new(ReplayEngineConfig::default(), &ReplayEngineFactory::new())
                .unwrap();
        engine.submit(request(15, 128, 1)).unwrap();
        drain(&mut engine);
        let first_end_ms = engine.now_ms();
        engine.take_report(first_end_ms).unwrap();

        engine.advance_now_ms(first_end_ms + 1_000.0);
        engine.submit(request(16, 128, 1)).unwrap();
        drain(&mut engine);
        let second_end_ms = engine.now_ms();
        let report = engine.take_report(second_end_ms).unwrap();

        assert!((report.throughput.duration_ms - (second_end_ms - first_end_ms)).abs() < 1e-9);
    }

    #[test]
    fn report_duration_includes_trailing_idle_time_in_each_epoch() {
        let mut engine = SteppableAgg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            2,
        )
        .unwrap();
        engine.set_capture_per_request(true);
        engine.set_sla_thresholds(SlaThresholds {
            e2e_ms: Some(f64::MAX),
            ..Default::default()
        });
        for ordinal in 1..=2 {
            let start_ms = engine.now_ms();
            engine.submit(request(ordinal, 128, 1)).unwrap();
            drain(&mut engine);
            let terminal_ms = engine.now_ms();
            engine.advance_now_ms(terminal_ms + 1_000.0);
            let elapsed_ms = engine.now_ms() - start_ms;
            let report = engine.take_report(0.0).unwrap();

            assert!((report.throughput.duration_ms - elapsed_ms).abs() < 1e-9);
            assert_eq!(report.per_request[0].terminal_time_ms, terminal_ms);
            assert!(
                (report.throughput.decode_worker_seconds - 2.0 * elapsed_ms / 1_000.0).abs() < 1e-9
            );
            assert!((report.throughput.request_throughput_rps - 1_000.0 / elapsed_ms).abs() < 1e-9);
            assert!(
                (report.goodput.unwrap().request_throughput_rps - 1_000.0 / elapsed_ms).abs()
                    < 1e-9
            );
        }
    }

    #[test]
    fn empty_report_epochs_include_idle_time_without_request_rates() {
        let mut engine =
            SteppableEngine::new(ReplayEngineConfig::default(), &ReplayEngineFactory::new())
                .unwrap();
        for end_ms in [1_000.0, 2_000.0] {
            engine.advance_now_ms(end_ms);
            let report = engine.take_report(0.0).unwrap();
            assert_eq!(report.throughput.duration_ms, 1_000.0);
            assert_eq!(report.throughput.decode_worker_seconds, 1.0);
            assert_eq!(report.request_counts.num_requests, 0);
            assert_eq!(report.throughput.request_throughput_rps, 0.0);
            assert_eq!(report.throughput.output_throughput_tok_s, 0.0);
        }
    }

    #[test]
    fn report_epochs_preserve_runtime_evidence_capture() {
        let mut engine = SteppableAgg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
        )
        .unwrap();
        engine.runtime = engine
            .runtime
            .with_capture_options(crate::replay::ReplayCaptureOptions {
                capture_canonical_evidence: true,
                capture_lifecycle_evidence: true,
                ..Default::default()
            });
        for ordinal in 1..=2 {
            engine.submit(request(ordinal, 128, 1)).unwrap();
            drain(&mut engine);
            let report = engine.take_report(0.0).unwrap();
            assert!(report.runtime_evidence.pressure.is_some());
            assert!(report.runtime_evidence.kv_ingest.is_some());
        }
    }

    #[test]
    fn a_report_drain_allows_uuid_reuse_with_fresh_measurements() {
        let mut engine = SteppableAgg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
        )
        .unwrap();
        let uuid = engine.submit(request(25, 128, 1)).unwrap();
        drain(&mut engine);
        let first = engine.take_report(0.0).unwrap();
        assert_eq!(first.request_counts.total_output_tokens, 1);
        assert!(engine.actual_output_length(uuid).is_none());
        assert!(engine.request_latencies(uuid).is_none());
        assert!(engine.request_admission(uuid).is_none());
        assert!(engine.cancel(uuid).unwrap().is_none());

        assert_eq!(engine.submit(request(25, 128, 3)).unwrap(), uuid);
        let events = drain(&mut engine);
        assert_eq!(events.iter().filter(|event| event.emitted_token).count(), 3);
        assert_eq!(
            events
                .iter()
                .filter(|event| event.terminal_status.is_some())
                .count(),
            1
        );
        let second = engine.take_report(0.0).unwrap();
        assert_eq!(second.request_counts.num_requests, 1);
        assert_eq!(second.request_counts.completed_requests, 1);
        assert_eq!(second.request_counts.total_output_tokens, 3);
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
    fn steppable_replay_rejects_a_completed_request_id_before_reporting() {
        let mut engine = SteppableAgg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
        )
        .unwrap();
        let uuid = engine.submit(request(22, 128, 16)).unwrap();

        drain(&mut engine);

        let error = engine.submit(request(22, 128, 16)).unwrap_err();
        assert!(error.to_string().contains("already retained"), "{error}");
        assert_eq!(engine.in_flight(), 0);

        let report = engine.take_report(engine.now_ms()).unwrap();
        assert_eq!(report.request_counts.num_requests, 1);
        assert_eq!(report.request_counts.completed_requests, 1);
        assert_eq!(uuid, Uuid::from_u128(22));
    }

    #[test]
    fn aggregated_report_drain_preserves_collector_configuration() {
        let mut engine = SteppableAgg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
        )
        .unwrap();
        engine.set_capture_per_request(true);
        engine.set_sla_thresholds(SlaThresholds {
            e2e_ms: Some(f64::MAX),
            ..Default::default()
        });

        engine.submit(request(23, 128, 16)).unwrap();
        drain(&mut engine);
        let first = engine.take_report(engine.now_ms()).unwrap();
        assert_eq!(first.per_request.len(), 1);
        assert!(first.goodput.is_some());

        engine.submit(request(24, 128, 16)).unwrap();
        drain(&mut engine);
        let second = engine.take_report(engine.now_ms()).unwrap();
        assert_eq!(second.per_request.len(), 1);
        assert!(second.goodput.is_some());
    }

    #[test]
    fn steppable_constructors_reject_zero_workers() {
        let factory = ReplayEngineFactory::new();

        assert!(SteppableAgg::new(ReplayEngineConfig::default(), &factory, 0).is_err());
    }

    #[test]
    fn disaggregated_replay_streams_tokens_and_one_terminal_through_the_handoff() {
        let mut engine = SteppableDisagg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            1,
        )
        .unwrap();
        engine.set_capture_per_request(true);
        let uuid = engine.submit(request(60, 128, 8)).unwrap();
        assert_eq!(engine.in_flight(), 1);

        let events = drain(&mut engine);

        assert_eq!(
            events.iter().filter(|event| event.emitted_token).count(),
            8
        );
        let terminals = events
            .iter()
            .filter(|event| event.terminal_status.is_some())
            .collect::<Vec<_>>();
        assert_eq!(terminals.len(), 1);
        assert_eq!(terminals[0].uuid, uuid);
        assert_eq!(
            terminals[0].terminal_status,
            Some(ReplayTerminalStatus::Completed)
        );
        assert_eq!(engine.in_flight(), 0);
        let report = engine.take_report(engine.now_ms()).unwrap();
        assert_eq!(report.request_counts.completed_requests, 1);
        assert_eq!(report.request_counts.total_output_tokens, 8);
        assert_eq!(report.per_request.len(), 1);
    }

    #[test]
    fn a_nonterminal_token_holds_its_instant_for_the_next_decision() {
        let mut engine = SteppableDisagg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            1,
        )
        .unwrap();
        engine.submit(request(61, 128, 16)).unwrap();
        let outcome = loop {
            let outcome = engine.step().unwrap();
            if outcome
                .events
                .iter()
                .any(|event| event.emitted_token && event.terminal_status.is_none())
            {
                break outcome;
            }
            assert!(!engine.is_idle(), "no nonterminal token was emitted");
        };

        assert_eq!(outcome.end_ms, engine.now_ms());
        // The commit half is still owed at this instant, so a child submitted
        // now joins the pass the worker starts next.
        assert_eq!(engine.next_event_ms(), Some(engine.now_ms()));
    }

    #[test]
    fn disaggregated_bounded_step_never_crosses_the_deadline_or_skips_time() {
        let mut engine = SteppableDisagg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            1,
        )
        .unwrap();
        engine.submit(request(62, 128, 16)).unwrap();

        let deadline_ms = 0.000_001;
        let outcome = engine.step_until(deadline_ms).unwrap();

        assert!(outcome.end_ms <= deadline_ms);
        assert!(!engine.is_idle());
        // Work is outstanding, so an external idle-gap skip must not move time.
        let before_ms = engine.now_ms();
        engine.advance_now_ms(before_ms + 1_000.0);
        assert_eq!(engine.now_ms(), before_ms);
    }

    #[test]
    fn disaggregated_cancellation_is_immediately_terminal_and_cleans_up() {
        let mut engine = SteppableDisagg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            1,
        )
        .unwrap();
        let uuid = engine.submit(request(63, 128, 64)).unwrap();
        let event = engine.cancel(uuid).unwrap();

        assert_eq!(
            event.map(|event| event.terminal_status),
            Some(Some(ReplayTerminalStatus::Canceled))
        );
        assert_eq!(engine.in_flight(), 0);
        assert!(engine.cancel(uuid).unwrap().is_none());
        let events = drain(&mut engine);
        assert!(
            !events
                .iter()
                .any(|event| event.uuid == uuid && event.terminal_status.is_some()),
            "a canceled request must not emit a second terminal: {events:?}"
        );
        assert!(engine.is_idle());
        let report = engine.take_report(engine.now_ms()).unwrap();
        assert_eq!(report.request_counts.num_requests, 1);
    }

    #[test]
    fn disaggregated_report_drain_allows_uuid_reuse_and_rejects_live_duplicates() {
        let mut engine = SteppableDisagg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            1,
        )
        .unwrap();
        let uuid = engine.submit(request(64, 128, 2)).unwrap();
        let error = engine.submit(request(64, 128, 2)).unwrap_err();
        assert!(error.to_string().contains("already live"), "{error}");
        drain(&mut engine);
        let error = engine.submit(request(64, 128, 2)).unwrap_err();
        assert!(error.to_string().contains("already retained"), "{error}");
        assert!(engine.take_report(f64::NAN).is_err());
        let first = engine.take_report(engine.now_ms()).unwrap();
        assert_eq!(first.request_counts.total_output_tokens, 2);
        assert!(engine.actual_output_length(uuid).is_none());

        assert_eq!(engine.submit(request(64, 128, 3)).unwrap(), uuid);
        drain(&mut engine);
        let second = engine.take_report(engine.now_ms()).unwrap();
        assert_eq!(second.request_counts.num_requests, 1);
        assert_eq!(second.request_counts.total_output_tokens, 3);
    }

    #[test]
    fn disaggregated_constructor_rejects_empty_pools() {
        let factory = ReplayEngineFactory::new();
        assert!(SteppableDisagg::new(ReplayEngineConfig::default(), &factory, 0, 1).is_err());
        assert!(SteppableDisagg::new(ReplayEngineConfig::default(), &factory, 1, 0).is_err());
    }

    fn fixed_timing_config(prefill_ms: f64, decode_ms: f64) -> ReplayEngineConfig {
        let mut config = ReplayEngineConfig::default();
        config.rank.timing_model = crate::engine::TimingModelConfig::Fixed {
            prefill_ms,
            decode_ms,
        };
        config
    }

    /// A parent's nonterminal token releases a successor. With a zero-length
    /// prefill and no handoff latency the successor reaches the decode pool at
    /// the token's own instant, so it can only join the decode worker's next
    /// pass if that pass was not committed before the caller submitted it.
    #[test]
    fn a_successor_submitted_after_a_nonterminal_token_joins_the_next_decode_pass() {
        const DECODE_MS: f64 = 5.0;
        let mut engine = SteppableDisagg::new(
            fixed_timing_config(0.0, DECODE_MS),
            &ReplayEngineFactory::new(),
            1,
            1,
        )
        .unwrap();
        engine.set_capture_per_request(true);
        let parent = engine.submit(request(70, 64, 6)).unwrap();

        let token_ms = loop {
            let outcome = engine.step().unwrap();
            assert!(
                outcome
                    .events
                    .iter()
                    .all(|event| event.terminal_status.is_none()),
                "the parent terminated before emitting a nonterminal token: {:?}",
                outcome.events
            );
            if outcome
                .events
                .iter()
                .any(|event| event.uuid == parent && event.emitted_token)
            {
                break outcome.end_ms;
            }
            assert!(!engine.is_idle(), "the parent never emitted a token");
        };
        // The parent is still live: this is a branch, not a terminal release.
        assert_eq!(engine.in_flight(), 1);
        assert!(engine.actual_output_length(parent).unwrap() < 6);
        assert_eq!(engine.now_ms(), token_ms);

        let mut successor = request(71, 64, 2);
        successor.tokens = (1_000..1_064).collect();
        let child = engine.submit(successor).unwrap();
        let held = engine.step_until(token_ms).unwrap();
        assert_eq!(held.end_ms, token_ms);
        assert!(held.events.is_empty(), "{:?}", held.events);
        assert_eq!(engine.request_admission(child).unwrap().0, token_ms);

        // Both requests decode in the one pass committed at the token instant.
        let next = engine.step_until(f64::INFINITY).unwrap();
        assert!(
            (next.end_ms - (token_ms + DECODE_MS)).abs() < 1e-9,
            "expected one decode pass after {token_ms}, ended at {}",
            next.end_ms
        );
        for uuid in [parent, child] {
            assert!(
                next.events
                    .iter()
                    .any(|event| event.uuid == uuid && event.emitted_token),
                "{uuid} missed the shared pass: {:?}",
                next.events
            );
        }

        drain(&mut engine);
        let report = engine.take_report(engine.now_ms()).unwrap();
        let record = report
            .per_request
            .iter()
            .find(|record| record.uuid == child.to_string())
            .expect("successor record");
        assert_eq!(record.first_admit_ms, Some(token_ms));
        assert!((record.first_token_ms.unwrap() - (token_ms + DECODE_MS)).abs() < 1e-9);
        assert_eq!(record.terminal_status, ReplayTerminalStatus::Completed);
    }

    /// Cancel while the KV payload is on the wire, using a real transfer
    /// bandwidth: the transfer's link share is released, no output or second
    /// terminal follows, and virtual time is not driven to the dead deadline.
    #[test]
    fn canceling_during_a_bandwidth_limited_transfer_retires_it() {
        let mut config = fixed_timing_config(1.0, 5.0);
        // 64 prompt tokens x 1 MB at 1 GB/s: the handoff alone takes >= 64 ms.
        config.rank.kv_transfer_bandwidth = Some(1.0);
        config.rank.kv_transfer_bytes_per_token = Some(1_000_000);
        let mut engine =
            SteppableDisagg::new(config, &ReplayEngineFactory::new(), 1, 1).unwrap();
        let uuid = engine.submit(request(72, 64, 4)).unwrap();

        // Prefill (1 ms) is done; the transfer is in flight.
        while engine.now_ms() < 10.0 {
            let outcome = engine.step_until(10.0).unwrap();
            assert!(outcome.events.is_empty(), "{:?}", outcome.events);
        }
        assert_eq!(engine.in_flight(), 1);
        assert!(
            engine.next_event_ms().unwrap() >= 64.0,
            "the handoff must be bandwidth-limited, next event {:?}",
            engine.next_event_ms()
        );

        let event = engine.cancel(uuid).unwrap().expect("cancel wins the race");
        assert_eq!(event.terminal_status, Some(ReplayTerminalStatus::Canceled));
        assert_eq!(engine.in_flight(), 0);
        assert!(engine.cancel(uuid).unwrap().is_none());

        let events = drain(&mut engine);
        assert!(
            events.iter().all(|event| event.uuid != uuid),
            "a canceled transfer must produce no further events: {events:?}"
        );
        assert!(
            engine.now_ms() < 30.0,
            "the canceled transfer's deadline must not drive time, now {}",
            engine.now_ms()
        );
        let report = engine.take_report(engine.now_ms()).unwrap();
        assert_eq!(report.request_counts.completed_requests, 0);
        assert_eq!(report.request_counts.total_output_tokens, 0);

        // The released link serves the next request normally.
        let successor = engine.submit(request(73, 64, 2)).unwrap();
        let events = drain(&mut engine);
        assert_eq!(
            events
                .iter()
                .filter(|event| event.uuid == successor && event.emitted_token)
                .count(),
            2
        );
        let report = engine.take_report(engine.now_ms()).unwrap();
        assert_eq!(report.request_counts.completed_requests, 1);
    }
}
