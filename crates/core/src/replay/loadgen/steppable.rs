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

use std::collections::{HashSet, VecDeque};

use uuid::Uuid;

use crate::replay::agg::AggRuntimeImpl;
use crate::replay::components::{
    AdmissionQueue, NoReplayMetadata, ReplayAdmissionMetadata, ReplayEngineObservation, ReplayMode,
};
use crate::replay::core::NoEngineEvents;
use crate::replay::core::round_robin::AggregatedRoundRobinPlacement;
use crate::replay::core::{PlacementPolicy, WorkerTopology};
use crate::replay::engine::{ReplayEngineConfig, ReplayEngineFactory, ReplayRoleFactory};
use crate::replay::loadgen::ReplayRequestPayload;
use crate::replay::protocol::DirectRequest;
use crate::replay::{
    ReplayCaptureOptions, ReplayReport, ReplayTerminalStatus, SlaThresholds, WorkerStage,
};

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

    /// SGLang's `/flush_cache` at the current instant: drop every cached KV
    /// page and reset the page allocator, keeping the runtime and its other
    /// scheduler state. Returns `false`, changing nothing, unless the runtime
    /// is idle, as SGLang does while requests are queued or running. The
    /// flush's KV events reach the placement policy at once. SGLang only.
    fn flush_cache(&mut self) -> anyhow::Result<bool> {
        anyhow::bail!("this steppable runtime has no cache flush")
    }
}

/// A type-erased placement policy for the aggregated steppable runtime,
/// parameterized on the observation and admission-metadata flavors.
///
/// `ObservationFlavor` (an engine observation *flavor*, e.g. `NoEngineEvents`
/// or a router's event adapter) and the trait's own `Observation` associated
/// type (that flavor's `::Batch`) are two different types one line apart;
/// named distinctly here so the projection is not mistaken for an identity.
pub type DynPlacement<ObservationFlavor, Metadata> = Box<
    dyn PlacementPolicy<
            ReplayRequestPayload,
            Metadata = Metadata,
            Observation = <ObservationFlavor as ReplayEngineObservation>::Batch,
        >,
>;

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

/// Build the aggregated role factory one steppable runtime runs on. The
/// observation flavor decides whether the engine retains and publishes KV
/// events for the placement policy.
fn aggregated_role_factory<O: ReplayEngineObservation>(
    engine: &ReplayEngineConfig,
    factory: &ReplayEngineFactory,
) -> anyhow::Result<ReplayRoleFactory> {
    Ok(factory.role_factory(
        engine,
        WorkerStage::Aggregated,
        O::capture_engine_kv_events(WorkerStage::Aggregated),
    )?)
}

/// Aggregated multi-worker topology as a [`SteppableReplay`].
///
/// The defaults are the round-robin runtime [`SteppableAgg::new`] builds;
/// [`SteppableAgg::with_placement`] builds one around a caller's policy.
pub struct SteppableAgg<
    P = AggregatedRoundRobinPlacement<()>,
    O = NoEngineEvents,
    M = NoReplayMetadata,
> where
    O: ReplayEngineObservation,
    M: ReplayAdmissionMetadata,
    P: PlacementPolicy<ReplayRequestPayload, Metadata = M, Observation = O::Batch>,
{
    runtime: AggRuntimeImpl<P, O, M>,
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
        let role_factory = aggregated_role_factory::<NoEngineEvents>(&engine, factory)?;
        let runtime = AggRuntimeImpl::new_composed(
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

impl<O, M> SteppableAgg<DynPlacement<O, M>, O, M>
where
    O: ReplayEngineObservation + 'static,
    M: ReplayAdmissionMetadata + 'static,
{
    /// Build an aggregated steppable runtime driven by a caller-supplied,
    /// type-erased placement policy. This is the injection point an
    /// out-of-crate policy uses; the runtime's generics stay private. When
    /// `O` captures engine KV events, the engine publishes them and they reach
    /// the policy's `observe`, as in trace-driven replay.
    pub fn with_placement(
        engine: ReplayEngineConfig,
        factory: &ReplayEngineFactory,
        num_workers: usize,
        make_placement: impl FnOnce(u32, Vec<WorkerTopology>) -> anyhow::Result<DynPlacement<O, M>>,
    ) -> anyhow::Result<Self> {
        Self::with_placement_and_capture_options(
            engine,
            factory,
            num_workers,
            ReplayCaptureOptions::default(),
            make_placement,
        )
    }

    /// As [`Self::with_placement`], additionally selecting the detailed
    /// capture the report carries, so a report through this seam can be
    /// compared byte for byte with a `Replayer::run()` report that enables
    /// canonical or lifecycle evidence.
    pub fn with_placement_and_capture_options(
        engine: ReplayEngineConfig,
        factory: &ReplayEngineFactory,
        num_workers: usize,
        capture: ReplayCaptureOptions,
        make_placement: impl FnOnce(u32, Vec<WorkerTopology>) -> anyhow::Result<DynPlacement<O, M>>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(num_workers > 0, "num_workers must be positive");
        let role_factory = aggregated_role_factory::<O>(&engine, factory)?;
        let runtime = AggRuntimeImpl::<DynPlacement<O, M>, O, M>::new_composed(
            role_factory,
            AdmissionQueue::new_requests(VecDeque::new(), steppable_mode()),
            num_workers,
            None,
            |dp_size, topology| {
                let mut placement = make_placement(dp_size, topology)?;
                // Worker-ready and topology-settled are raised only for dynamic
                // scaling, so a policy built here is never told its initial
                // topology is complete. A stateless policy does not care; a
                // stateful one holds every request in its pending queue
                // forever. Scoped to this constructor because a policy whose
                // `topology_settled` is a one-shot latch would otherwise have
                // that transition consumed before its real first settle.
                let released = placement.topology_settled(crate::replay::agg::REPLAY_EPOCH_MS)?;
                anyhow::ensure!(
                    released.is_empty(),
                    "placement released {} request(s) before any were submitted",
                    released.len()
                );
                Ok(placement)
            },
        )?
        .with_capture_options(capture)
        .into_steppable();
        Ok(Self {
            runtime,
            live: LiveRequests::default(),
        })
    }
}

impl<P, O, M> SteppableReplay for SteppableAgg<P, O, M>
where
    O: ReplayEngineObservation,
    M: ReplayAdmissionMetadata,
    P: PlacementPolicy<ReplayRequestPayload, Metadata = M, Observation = O::Batch>,
{
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

    fn flush_cache(&mut self) -> anyhow::Result<bool> {
        self.runtime.ensure_cache_flush_supported()?;
        if self.in_flight() > 0 {
            return Ok(false);
        }
        self.runtime.flush_cache_dynamic()
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

    fn flush_cache(&mut self) -> anyhow::Result<bool> {
        self.inner.flush_cache()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::replay::core::{EngineEventBatch, Placement, PlacementEffects};
    use crate::replay::protocol::ReplayRequestContext;

    /// An observation flavor that actually captures KV events, so a test can
    /// tell `with_placement` apart from a hardcoded no-capture role factory.
    /// `NoEngineEvents` cannot: its flag is already `false`.
    #[derive(Debug, Default)]
    struct CapturedKvEvents(Vec<crate::engine::KvEvent>);

    impl EngineEventBatch for CapturedKvEvents {
        fn is_empty(&self) -> bool {
            self.0.is_empty()
        }

        fn append(&mut self, mut other: Self) {
            self.0.append(&mut other.0);
        }
    }

    #[derive(Debug, Default)]
    struct CapturingObservation;

    impl ReplayEngineObservation for CapturingObservation {
        type Batch = CapturedKvEvents;

        const CAPTURE_ENGINE_KV_EVENTS: bool = true;

        fn observe_engine_events(
            _stage: WorkerStage,
            _worker_id: usize,
            _dp_rank: u32,
            events: Vec<crate::engine::KvEvent>,
        ) -> Self::Batch {
            CapturedKvEvents(events)
        }
    }

    #[derive(Default)]
    struct PolicyCalls {
        places: usize,
        terminals: usize,
        observed_batches: usize,
        session_ids: Vec<Option<String>>,
    }

    /// A stateful policy with the two properties the injection seam has to
    /// respect: it refuses to place before its topology is settled, and its
    /// `topology_settled` is a one-shot latch. Records its own calls so a test
    /// can confirm they arrived through the `Box`.
    struct LatchedPlacement<Events: EngineEventBatch> {
        inner: AggregatedRoundRobinPlacement<Events>,
        calls: Rc<RefCell<PolicyCalls>>,
        is_settled: bool,
    }

    impl<Events: EngineEventBatch> PlacementPolicy<ReplayRequestPayload> for LatchedPlacement<Events> {
        type Metadata = NoReplayMetadata;
        type Observation = Events;

        fn place(
            &mut self,
            request: &ReplayRequestPayload,
            metadata: Self::Metadata,
            session_id: Option<String>,
            now_ms: f64,
        ) -> anyhow::Result<PlacementEffects> {
            anyhow::ensure!(self.is_settled, "placed before the topology settled");
            let mut calls = self.calls.borrow_mut();
            calls.places += 1;
            calls.session_ids.push(session_id.clone());
            drop(calls);
            self.inner.place(request, metadata, session_id, now_ms)
        }

        fn observe(&mut self, observation: Events, now_ms: f64) -> anyhow::Result<Vec<Placement>> {
            if !observation.is_empty() {
                self.calls.borrow_mut().observed_batches += 1;
            }
            PlacementPolicy::<ReplayRequestPayload>::observe(&mut self.inner, observation, now_ms)
        }

        fn cancel_pending(&mut self, request_id: Uuid) -> bool {
            PlacementPolicy::<ReplayRequestPayload>::cancel_pending(&mut self.inner, request_id)
        }

        fn request_terminal(
            &mut self,
            request_id: Uuid,
            now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            self.calls.borrow_mut().terminals += 1;
            PlacementPolicy::<ReplayRequestPayload>::request_terminal(
                &mut self.inner,
                request_id,
                now_ms,
            )
        }

        fn prefill_completed(
            &mut self,
            request_id: Uuid,
            now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            PlacementPolicy::<ReplayRequestPayload>::prefill_completed(
                &mut self.inner,
                request_id,
                now_ms,
            )
        }

        fn pending_count(&self) -> usize {
            PlacementPolicy::<ReplayRequestPayload>::pending_count(&self.inner)
        }

        fn worker_ready(
            &mut self,
            worker: WorkerTopology,
            now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            PlacementPolicy::<ReplayRequestPayload>::worker_ready(&mut self.inner, worker, now_ms)
        }

        fn worker_draining(
            &mut self,
            worker: WorkerTopology,
            now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            PlacementPolicy::<ReplayRequestPayload>::worker_draining(
                &mut self.inner,
                worker,
                now_ms,
            )
        }

        fn worker_removed(
            &mut self,
            worker: WorkerTopology,
            now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            PlacementPolicy::<ReplayRequestPayload>::worker_removed(&mut self.inner, worker, now_ms)
        }

        fn topology_settled(&mut self, now_ms: f64) -> anyhow::Result<Vec<Placement>> {
            anyhow::ensure!(!self.is_settled, "topology_settled fired twice");
            self.is_settled = true;
            PlacementPolicy::<ReplayRequestPayload>::topology_settled(&mut self.inner, now_ms)
        }
    }

    fn latched<O: ReplayEngineObservation + 'static>(
        num_workers: usize,
        calls: &Rc<RefCell<PolicyCalls>>,
    ) -> SteppableAgg<DynPlacement<O, NoReplayMetadata>, O, NoReplayMetadata>
    where
        O::Batch: 'static,
    {
        let calls = Rc::clone(calls);
        SteppableAgg::with_placement(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            num_workers,
            move |dp_size, topology| {
                Ok(Box::new(LatchedPlacement {
                    inner: AggregatedRoundRobinPlacement::new(dp_size, topology),
                    calls,
                    is_settled: false,
                }) as DynPlacement<O, NoReplayMetadata>)
            },
        )
        .expect("engine builds")
    }

    /// An injected boxed policy receives the full lifecycle -- the settle
    /// handshake at construction, then placements and terminals -- through the
    /// `Box` forwarding impl.
    #[test]
    fn injected_boxed_placement_receives_the_full_lifecycle() {
        let calls = Rc::new(RefCell::new(PolicyCalls::default()));
        let mut engine = latched::<NoEngineEvents>(2, &calls);

        engine.submit(request(1, 64, 4)).expect("submit");
        engine.submit(request(2, 64, 4)).expect("submit");
        drain(&mut engine);

        let calls = calls.borrow();
        assert_eq!(calls.places, 2);
        assert_eq!(calls.terminals, 2);
    }

    /// KV events reach an injected policy when its observation flavor asks
    /// for them: the role factory is built with `O::capture_engine_kv_events`,
    /// so the engine retains and publishes events that arrive at `observe`.
    #[test]
    fn injected_placement_observes_kv_events_when_its_flavor_captures_them() {
        let calls = Rc::new(RefCell::new(PolicyCalls::default()));
        let mut engine = latched::<CapturingObservation>(1, &calls);

        engine.submit(request(1, 256, 8)).expect("submit");
        drain(&mut engine);

        assert!(
            calls.borrow().observed_batches > 0,
            "a capturing observation flavor produced no KV events for the policy"
        );
    }

    /// The seam changes nothing else: round-robin injected through
    /// `with_placement` gives every request the same admission, reuse, first
    /// token and terminal as the built-in round-robin runtime, with queueing
    /// and shared prefixes in play.
    #[test]
    fn injected_round_robin_matches_the_built_in_runtime() {
        fn timeline(engine: &mut dyn SteppableReplay) -> Vec<(u128, f64, usize, f64, f64)> {
            let mut terminals = std::collections::BTreeMap::new();
            for i in 0..24u128 {
                let family = (i % 3) as u32 * 10_000;
                engine
                    .submit(DirectRequest {
                        tokens: (0..(128 + (i as u32 * 37) % 300))
                            .map(|t| t + family)
                            .collect(),
                        max_output_tokens: 8 + (i as usize * 13) % 40,
                        uuid: Some(Uuid::from_u128(i + 1)),
                        arrival_timestamp_ms: Some(engine.now_ms()),
                        ..Default::default()
                    })
                    .unwrap();
                for _ in 0..3 {
                    for event in engine.step().unwrap().events {
                        if event.terminal_status.is_some() {
                            terminals.insert(event.uuid.as_u128(), engine.now_ms());
                        }
                    }
                }
            }
            for _ in 0..10_000 {
                if engine.is_idle() {
                    break;
                }
                for event in engine.step().unwrap().events {
                    if event.terminal_status.is_some() {
                        terminals.insert(event.uuid.as_u128(), engine.now_ms());
                    }
                }
            }
            assert!(engine.is_idle(), "steppable replay did not drain");
            (1..=24u128)
                .map(|id| {
                    let uuid = Uuid::from_u128(id);
                    let (admitted, reused) = engine.request_admission(uuid).unwrap();
                    let (ttft, _) = engine.request_latencies(uuid).unwrap();
                    (id, admitted, reused, ttft, terminals[&id])
                })
                .collect()
        }

        let mut config = ReplayEngineConfig::default();
        config.rank.max_num_seqs = 4;
        let mut built_in =
            SteppableAgg::new(config.clone(), &ReplayEngineFactory::new(), 2).unwrap();
        let calls = Rc::new(RefCell::new(PolicyCalls::default()));
        let calls_for_policy = Rc::clone(&calls);
        let mut injected = SteppableAgg::<
            DynPlacement<NoEngineEvents, NoReplayMetadata>,
            NoEngineEvents,
            NoReplayMetadata,
        >::with_placement(
            config,
            &ReplayEngineFactory::new(),
            2,
            move |dp_size, topology| {
                Ok(Box::new(LatchedPlacement {
                    inner: AggregatedRoundRobinPlacement::new(dp_size, topology),
                    calls: calls_for_policy,
                    is_settled: false,
                })
                    as DynPlacement<NoEngineEvents, NoReplayMetadata>)
            },
        )
        .unwrap();
        let expected = timeline(&mut built_in);
        assert_eq!(timeline(&mut injected), expected);
        assert!(
            expected
                .iter()
                .any(|(_, admitted, reused, _, _)| *admitted > 0.0 && *reused > 0)
        );
    }

    /// A dynamic submission's `replay_context.session_id` reaches placement,
    /// as it does for trace-driven admission.
    #[test]
    fn a_dynamic_submission_forwards_its_session_id_to_placement() {
        let calls = Rc::new(RefCell::new(PolicyCalls::default()));
        let mut engine = latched::<NoEngineEvents>(2, &calls);
        let mut with_session = request(1, 64, 2);
        with_session.replay_context = Some(ReplayRequestContext {
            authored_id: "req-1".into(),
            session_id: Some("conversation-7".into()),
            turn_index: Some(0),
            metadata: serde_json::Value::Null,
            prompt_token_source: Default::default(),
            agentic: None,
        });
        engine.submit(with_session).unwrap();
        engine.submit(request(2, 64, 2)).unwrap();
        drain(&mut engine);

        assert_eq!(
            calls.borrow().session_ids,
            vec![Some("conversation-7".to_string()), None]
        );
    }

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

    /// A cache flush is refused while a request is in flight. At idle it
    /// drops prefix reuse and keeps the runtime.
    #[test]
    fn a_cache_flush_drops_prefix_reuse() {
        let engine_config = ReplayEngineConfig {
            rank: crate::engine::EngineConfig {
                enable_prefix_caching: true,
                ..crate::engine::EngineConfig::for_backend(crate::engine::Backend::Sglang)
            },
            ..ReplayEngineConfig::default()
        };
        let mut engine = SteppableAgg::new(engine_config, &ReplayEngineFactory::new(), 1)
            .expect("engine builds");
        fn reused(engine: &mut dyn SteppableReplay, uuid: u128) -> usize {
            let at = engine.now_ms();
            engine
                .submit(DirectRequest {
                    arrival_timestamp_ms: Some(at),
                    ..request(uuid, 256, 4)
                })
                .unwrap();
            if uuid == 1 {
                assert!(
                    !engine.flush_cache().unwrap(),
                    "flushed with a request in flight"
                );
            }
            drain(engine);
            engine.request_admission(Uuid::from_u128(uuid)).unwrap().1
        }

        assert_eq!(reused(&mut engine, 1), 0);
        assert!(reused(&mut engine, 2) > 0, "the same prompt reused nothing");
        assert!(
            engine.flush_cache().unwrap(),
            "an idle runtime refused a flush"
        );
        assert_eq!(reused(&mut engine, 3), 0, "a prefix survived the flush");
    }

    #[test]
    fn a_cache_flush_is_sglang_only() {
        let mut engine =
            SteppableEngine::new(ReplayEngineConfig::default(), &ReplayEngineFactory::new())
                .unwrap();
        let error = engine.flush_cache().unwrap_err().to_string();
        assert!(error.contains("SGLang"), "{error}");
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
}
