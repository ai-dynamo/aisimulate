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

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use super::types::{ReplayHashIds, ReplayRequestPayload};

use uuid::Uuid;

use crate::replay::agg::AggRuntimeImpl;
use crate::replay::components::{AdmissionQueue, NoReplayMetadata, ReplayMode};
use crate::replay::core::NoEngineEvents;
use crate::replay::core::round_robin::AggregatedRoundRobinPlacement;
use crate::replay::engine::{ReplayEngineConfig, ReplayEngineFactory};
use crate::replay::protocol::DirectRequest;
use crate::replay::{ReplayReport, ReplayTerminalStatus, SlaThresholds, WorkerStage};

/// Safe owner of an immutable compact prompt hash array.
///
/// `hash_ids` must return the same identities throughout this owner's lifetime.
/// The replay retains an accepted owner until terminal, cancellation, or replay
/// destruction, even after worker admission has materialized its tokens. Hooks
/// must not panic or reenter the replay. Rejection never calls `on_accepted`.
pub trait CompactHashIdsLease: std::fmt::Debug + Send + Sync {
    fn hash_ids(&self) -> &[u32];

    fn on_accepted(&self) {}
}

/// A complete logical prompt represented by one identity per trace block.
/// Metadata is the current native request, with an empty `tokens` field; exact
/// output plans (including an explicitly empty plan) and all routing/provenance
/// fields pass unchanged through the existing deferred admission path.
#[derive(Debug)]
pub struct CompactDirectRequest {
    payload: ReplayRequestPayload,
}

impl CompactDirectRequest {
    pub fn new(
        request_metadata: DirectRequest,
        input_length: usize,
        hash_ids: Vec<u32>,
        trace_block_size: usize,
    ) -> anyhow::Result<Self> {
        Self::from_storage(
            request_metadata,
            input_length,
            ReplayHashIds::Owned(hash_ids),
            trace_block_size,
        )
    }

    pub fn from_hash_ids_lease(
        request_metadata: DirectRequest,
        input_length: usize,
        hash_ids: Arc<dyn CompactHashIdsLease>,
        trace_block_size: usize,
    ) -> anyhow::Result<Self> {
        Self::from_storage(
            request_metadata,
            input_length,
            ReplayHashIds::Leased(hash_ids),
            trace_block_size,
        )
    }

    fn from_storage(
        request_metadata: DirectRequest,
        input_length: usize,
        hash_ids: ReplayHashIds,
        trace_block_size: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            request_metadata.tokens.is_empty(),
            "compact request metadata must not contain tokens"
        );
        super::trace::validate_synthesizable_prompt(input_length, &hash_ids, trace_block_size)?;
        anyhow::ensure!(
            hash_ids.len() == input_length.div_ceil(trace_block_size),
            "compact request requires exactly one hash id per logical trace block"
        );
        Ok(Self {
            payload: ReplayRequestPayload::Deferred {
                request_metadata,
                input_length,
                hash_ids,
                trace_block_size,
            },
        })
    }

    fn into_parts(self) -> (ReplayRequestPayload, Option<Arc<dyn CompactHashIdsLease>>) {
        let owner = match &self.payload {
            ReplayRequestPayload::Deferred {
                hash_ids: ReplayHashIds::Leased(owner),
                ..
            } => Some(Arc::clone(owner)),
            _ => None,
        };
        (self.payload, owner)
    }
}

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

/// Failure of an atomic passive batch. Rejected batches do not mutate replay
/// state; runtime failures occur during stepping, never midway through enqueue.
#[derive(Debug)]
pub struct BatchSubmitError(anyhow::Error);

impl BatchSubmitError {
    pub fn is_poisoned(&self) -> bool {
        false
    }
}

impl std::fmt::Display for BatchSubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for BatchSubmitError {}

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

    /// Submit without expanding prompt tokens before server admission. Custom
    /// implementations are unsupported unless they explicitly implement this.
    fn submit_compact(&mut self, _request: CompactDirectRequest) -> anyhow::Result<Uuid> {
        anyhow::bail!("this steppable replay does not support compact submission")
    }

    /// Atomically accept a batch at the current clock, without executing it.
    fn submit_batch(
        &mut self,
        _requests: Vec<DirectRequest>,
    ) -> Result<Vec<Uuid>, BatchSubmitError> {
        Err(BatchSubmitError(anyhow::anyhow!(
            "this steppable replay does not support batch submission"
        )))
    }

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

    /// Maximum reuse observed across admissions, not the first-admission value.
    /// Available for live requests without enabling per-request capture.
    fn request_reused_input_tokens(&self, _uuid: Uuid) -> Option<usize> {
        None
    }

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
    uuids: HashMap<Uuid, Option<Arc<dyn CompactHashIdsLease>>>,
    pending: VecDeque<ReplayRequestPayload>,
}

impl LiveRequests {
    fn enqueue(
        &mut self,
        mut payload: ReplayRequestPayload,
        owner: Option<Arc<dyn CompactHashIdsLease>>,
    ) -> Uuid {
        let uuid = payload.metadata().uuid.unwrap_or_else(Uuid::new_v4);
        payload.metadata_mut().uuid = Some(uuid);
        self.pending.push_back(payload);
        self.uuids.insert(uuid, owner);
        if let Some(Some(owner)) = self.uuids.get(&uuid) {
            owner.on_accepted();
        }
        uuid
    }

    fn validate(
        &self,
        request: &DirectRequest,
        retained: bool,
        dp_size: u32,
        prefill_dp_size: u32,
        workers: usize,
    ) -> anyhow::Result<()> {
        if let Some(uuid) = request.uuid {
            anyhow::ensure!(
                !self.contains(uuid),
                "steppable replay request {uuid} is already live"
            );
            anyhow::ensure!(
                !retained,
                "steppable replay request {uuid} is already retained"
            );
        }
        anyhow::ensure!(
            request.preferred_dp_rank.is_none_or(|rank| rank < dp_size),
            "preferred attention-DP rank is out of range"
        );
        anyhow::ensure!(
            request
                .preferred_prefill_dp_rank
                .or(request.preferred_dp_rank)
                .is_none_or(|rank| rank < prefill_dp_size),
            "preferred prefill attention-DP rank is out of range"
        );
        if let Some(hint) = request.kv_hints {
            anyhow::ensure!(
                hint.source_worker < workers && hint.source_dp_rank < dp_size,
                "KV hint source is out of range"
            );
        }
        Ok(())
    }

    fn cancel_pending(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        collector: &mut crate::replay::TraceCollector,
    ) -> anyhow::Result<bool> {
        let Some(index) = self
            .pending
            .iter()
            .position(|payload| payload.metadata().uuid == Some(uuid))
        else {
            return Ok(false);
        };
        let request = &self.pending[index];
        collector.try_on_arrival(
            uuid,
            now_ms,
            request.input_length(),
            request.metadata().effective_max_output_tokens(),
        )?;
        if let Some(context) = request.metadata().replay_context.as_ref() {
            collector.on_request_context(uuid, context);
        }
        collector.on_terminal(uuid, now_ms, ReplayTerminalStatus::Canceled);
        self.pending.remove(index);
        self.uuids.remove(&uuid);
        Ok(true)
    }

    fn contains(&self, uuid: Uuid) -> bool {
        self.uuids.contains_key(&uuid)
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
    dp_size: u32,
    num_workers: usize,
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
            dp_size: engine.dp_size,
            num_workers,
        })
    }

    fn validate_submission(&self, request: &DirectRequest) -> anyhow::Result<()> {
        self.live.validate(
            request,
            request
                .uuid
                .is_some_and(|uuid| self.runtime.collector().contains_request(uuid)),
            self.dp_size,
            self.dp_size,
            self.num_workers,
        )
    }
}

impl SteppableReplay for SteppableAgg {
    fn now_ms(&self) -> f64 {
        self.runtime.now_ms()
    }

    fn advance_now_ms(&mut self, now_ms: f64) {
        if self.live.pending.is_empty() && now_ms.is_finite() && now_ms > self.runtime.now_ms() {
            self.runtime.advance_now_ms(now_ms);
        }
    }

    fn submit(&mut self, request: DirectRequest) -> anyhow::Result<Uuid> {
        self.validate_submission(&request)?;
        Ok(self
            .live
            .enqueue(ReplayRequestPayload::materialized(request), None))
    }

    fn submit_compact(&mut self, request: CompactDirectRequest) -> anyhow::Result<Uuid> {
        let (payload, owner) = request.into_parts();
        self.validate_submission(payload.metadata())?;
        Ok(self.live.enqueue(payload, owner))
    }

    fn submit_batch(
        &mut self,
        requests: Vec<DirectRequest>,
    ) -> Result<Vec<Uuid>, BatchSubmitError> {
        let mut ids = std::collections::HashSet::new();
        for request in &requests {
            self.validate_submission(request)
                .map_err(BatchSubmitError)?;
            if let Some(uuid) = request.uuid {
                if !ids.insert(uuid) {
                    return Err(BatchSubmitError(anyhow::anyhow!(
                        "duplicate request {uuid} in batch"
                    )));
                }
            }
        }
        Ok(requests
            .into_iter()
            .map(|request| {
                self.live
                    .enqueue(ReplayRequestPayload::materialized(request), None)
            })
            .collect())
    }

    fn cancel(&mut self, uuid: Uuid) -> anyhow::Result<Option<EngineEvent>> {
        if self
            .live
            .cancel_pending(uuid, self.runtime.now_ms(), self.runtime.collector_mut())?
        {
            return Ok(Some(EngineEvent::terminal(
                uuid,
                ReplayTerminalStatus::Canceled,
            )));
        }
        let status = self.runtime.cancel_dynamic(uuid)?;
        if status.is_some() {
            self.live.uuids.remove(&uuid);
        }
        Ok(status.map(|status| EngineEvent::terminal(uuid, status)))
    }

    fn step_until(&mut self, until_ms: f64) -> anyhow::Result<StepOutcome> {
        anyhow::ensure!(
            !until_ms.is_nan() && until_ms >= self.now_ms(),
            "step deadline precedes runtime time"
        );
        while let Some(payload) = self.live.pending.pop_front() {
            self.runtime.submit_dynamic_payload(payload)?;
        }
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
        if !self.live.pending.is_empty() {
            return Some(self.now_ms());
        }
        self.runtime.next_timestamp()
    }

    fn is_idle(&self) -> bool {
        self.live.pending.is_empty() && self.runtime.is_workload_done()
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

    fn request_reused_input_tokens(&self, uuid: Uuid) -> Option<usize> {
        self.runtime.collector().request_reused_input_tokens(uuid)
    }

    fn actual_output_length(&self, uuid: Uuid) -> Option<usize> {
        self.runtime.collector().actual_output_length(uuid)
    }

    fn take_report(&mut self, wall_ms: f64) -> anyhow::Result<ReplayReport> {
        anyhow::ensure!(self.is_idle(), "replay report requires an idle runtime");
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

    fn submit_compact(&mut self, request: CompactDirectRequest) -> anyhow::Result<Uuid> {
        self.inner.submit_compact(request)
    }

    fn submit_batch(
        &mut self,
        requests: Vec<DirectRequest>,
    ) -> Result<Vec<Uuid>, BatchSubmitError> {
        self.inner.submit_batch(requests)
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

    fn request_reused_input_tokens(&self, uuid: Uuid) -> Option<usize> {
        self.inner.request_reused_input_tokens(uuid)
    }

    fn actual_output_length(&self, uuid: Uuid) -> Option<usize> {
        self.inner.actual_output_length(uuid)
    }

    fn take_report(&mut self, wall_ms: f64) -> anyhow::Result<ReplayReport> {
        self.inner.take_report(wall_ms)
    }
}

type SteppableDisaggRuntime = crate::replay::disagg::DisaggRuntimeImpl<
    crate::replay::core::round_robin::PoolRoundRobinPlacement<()>,
    NoEngineEvents,
    NoReplayMetadata,
>;

/// Externally driven disaggregated replay using the current native handoff,
/// transfer, release, and worker runtime. Client terminal and server quiescence
/// remain distinct; compact owners survive both when cleanup is outstanding.
pub struct SteppableDisagg {
    runtime: SteppableDisaggRuntime,
    live: LiveRequests,
    retiring: HashMap<Uuid, Arc<dyn CompactHashIdsLease>>,
    dp_size: u32,
    prefill_dp_size: u32,
    num_decode_workers: usize,
}

impl SteppableDisagg {
    pub fn new(
        engine: ReplayEngineConfig,
        factory: &ReplayEngineFactory,
        num_prefill_workers: usize,
        num_decode_workers: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            num_prefill_workers > 0 && num_decode_workers > 0,
            "disaggregated worker counts must be positive"
        );
        let prefill_factory = factory.role_factory(&engine, WorkerStage::Prefill, false)?;
        let decode_factory = factory.role_factory(&engine, WorkerStage::Decode, false)?;
        let prefill_dp_size = prefill_factory.dp_size();
        let dp_size = decode_factory.dp_size();
        let config = crate::replay::OfflineDisaggReplayConfig {
            prefill_factory,
            decode_factory,
            prefill_startup_time_ms: None,
            decode_startup_time_ms: None,
            num_prefill_workers,
            num_decode_workers,
            handoff_latency_ms: 0.0,
        };
        let runtime = SteppableDisaggRuntime::new_composed(
            &config,
            AdmissionQueue::new_requests(VecDeque::new(), steppable_mode()),
            false,
            |_, prefill, _, decode| {
                Ok((
                    crate::replay::core::round_robin::PoolRoundRobinPlacement::new(prefill),
                    crate::replay::core::round_robin::PoolRoundRobinPlacement::new(decode),
                ))
            },
        )?
        .into_steppable();
        Ok(Self {
            runtime,
            live: LiveRequests::default(),
            retiring: HashMap::new(),
            dp_size,
            prefill_dp_size,
            num_decode_workers,
        })
    }

    fn validate_submission(&self, request: &DirectRequest) -> anyhow::Result<()> {
        self.live.validate(
            request,
            request
                .uuid
                .is_some_and(|uuid| self.runtime.collector().contains_request(uuid)),
            self.dp_size,
            self.prefill_dp_size,
            self.num_decode_workers,
        )
    }

    fn retire_owner(&mut self, uuid: Uuid) {
        if let Some(Some(owner)) = self.live.uuids.remove(&uuid) {
            self.retiring.insert(uuid, owner);
        }
    }

    fn release_quiescent_owners(&mut self) {
        for uuid in self.runtime.take_step_quiescent() {
            self.retiring.remove(&uuid);
        }
    }
}

impl SteppableReplay for SteppableDisagg {
    fn now_ms(&self) -> f64 {
        self.runtime.now_ms()
    }
    fn advance_now_ms(&mut self, now_ms: f64) {
        if self.live.pending.is_empty() && now_ms.is_finite() && now_ms > self.now_ms() {
            self.runtime.advance_now_ms(now_ms);
        }
    }
    fn submit(&mut self, request: DirectRequest) -> anyhow::Result<Uuid> {
        self.validate_submission(&request)?;
        Ok(self
            .live
            .enqueue(ReplayRequestPayload::materialized(request), None))
    }
    fn submit_compact(&mut self, request: CompactDirectRequest) -> anyhow::Result<Uuid> {
        let (payload, owner) = request.into_parts();
        self.validate_submission(payload.metadata())?;
        Ok(self.live.enqueue(payload, owner))
    }
    fn submit_batch(
        &mut self,
        requests: Vec<DirectRequest>,
    ) -> Result<Vec<Uuid>, BatchSubmitError> {
        let mut ids = std::collections::HashSet::new();
        for request in &requests {
            self.validate_submission(request)
                .map_err(BatchSubmitError)?;
            if let Some(uuid) = request.uuid {
                if !ids.insert(uuid) {
                    return Err(BatchSubmitError(anyhow::anyhow!(
                        "duplicate request {uuid} in batch"
                    )));
                }
            }
        }
        Ok(requests
            .into_iter()
            .map(|request| {
                self.live
                    .enqueue(ReplayRequestPayload::materialized(request), None)
            })
            .collect())
    }
    fn cancel(&mut self, uuid: Uuid) -> anyhow::Result<Option<EngineEvent>> {
        if self
            .live
            .cancel_pending(uuid, self.runtime.now_ms(), self.runtime.collector_mut())?
        {
            return Ok(Some(EngineEvent::terminal(
                uuid,
                ReplayTerminalStatus::Canceled,
            )));
        }
        let status = self.runtime.cancel_dynamic(uuid)?;
        if status.is_some() {
            self.retire_owner(uuid);
        }
        self.release_quiescent_owners();
        Ok(status.map(|status| EngineEvent::terminal(uuid, status)))
    }
    fn step_until(&mut self, until_ms: f64) -> anyhow::Result<StepOutcome> {
        anyhow::ensure!(
            !until_ms.is_nan() && until_ms >= self.now_ms(),
            "step deadline precedes runtime time"
        );
        while let Some(payload) = self.live.pending.pop_front() {
            self.runtime.submit_dynamic_payload(payload)?;
        }
        let end_ms = self.runtime.step_dynamic_until(until_ms)?;
        let mut events = self
            .runtime
            .take_step_tokens()
            .into_iter()
            .map(|(uuid, token_id)| EngineEvent::token(uuid, token_id))
            .collect::<Vec<_>>();
        for (uuid, status) in self.runtime.take_step_terminals() {
            self.retire_owner(uuid);
            events.push(EngineEvent::terminal(uuid, status));
        }
        self.release_quiescent_owners();
        Ok(StepOutcome { end_ms, events })
    }
    fn next_event_ms(&mut self) -> Option<f64> {
        if !self.live.pending.is_empty() {
            return Some(self.now_ms());
        }
        self.runtime.next_dynamic_timestamp()
    }
    fn is_idle(&self) -> bool {
        self.live.pending.is_empty() && self.runtime.is_workload_done()
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
    fn request_reused_input_tokens(&self, uuid: Uuid) -> Option<usize> {
        self.runtime.collector().request_reused_input_tokens(uuid)
    }
    fn actual_output_length(&self, uuid: Uuid) -> Option<usize> {
        self.runtime.collector().actual_output_length(uuid)
    }
    fn take_report(&mut self, wall_ms: f64) -> anyhow::Result<ReplayReport> {
        anyhow::ensure!(self.is_idle(), "replay report requires an idle runtime");
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

    #[derive(Debug)]
    struct TestLease {
        ids: Vec<u32>,
        accepted: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl CompactHashIdsLease for TestLease {
        fn hash_ids(&self) -> &[u32] {
            &self.ids
        }

        fn on_accepted(&self) {
            self.accepted
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[test]
    fn compact_geometry_is_checked_without_block_sized_scratch() {
        assert!(CompactDirectRequest::new(DirectRequest::default(), 1, vec![7], 0).is_err());
        assert!(CompactDirectRequest::new(DirectRequest::default(), 9, vec![7], 8).is_err());
        assert!(CompactDirectRequest::new(DirectRequest::default(), 1, vec![7, 8], 8).is_err());
        assert!(CompactDirectRequest::new(request(1, 1, 1), 1, vec![7], 8).is_err());
        let compact =
            CompactDirectRequest::new(DirectRequest::default(), 1, vec![7], usize::MAX).unwrap();
        assert_eq!(compact.payload.prompt_tokens(), vec![7]);
        assert_eq!(
            compact.payload.replay_hashes(u32::MAX).local_block_hashes,
            Vec::<u64>::new()
        );
        let empty = CompactDirectRequest::new(DirectRequest::default(), 0, vec![], 8).unwrap();
        assert!(empty.payload.prompt_tokens().is_empty());
    }

    #[test]
    fn deferred_hash_scratch_is_bounded_and_byte_exact_for_large_blocks() {
        for (input_length, ids, trace_block_size) in [
            (8194, vec![7], usize::MAX),
            (10013, vec![7, 9], 7000),
            (12310, vec![7, 7, 9], 5000),
        ] {
            let compact = CompactDirectRequest::new(
                DirectRequest::default(),
                input_length,
                ids,
                trace_block_size,
            )
            .unwrap();
            let tokens = compact.payload.prompt_tokens();
            for block_size in [1, 16, 4096, 4097, u32::MAX] {
                assert_eq!(
                    compact.payload.replay_hashes(block_size),
                    super::super::types::ReplayRequestHashes::from_tokens(&tokens, block_size)
                );
            }
        }
    }

    #[test]
    fn same_allocation_has_independent_concurrent_request_owners() {
        let mut engine =
            SteppableEngine::new(ReplayEngineConfig::default(), &ReplayEngineFactory::new())
                .unwrap();
        let accepted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let owner = Arc::new(TestLease {
            ids: vec![7],
            accepted: Arc::clone(&accepted),
        });
        let weak = Arc::downgrade(&owner);
        let first = engine
            .submit_compact(
                CompactDirectRequest::from_hash_ids_lease(request(750, 0, 1), 8, owner.clone(), 8)
                    .unwrap(),
            )
            .unwrap();
        engine
            .submit_compact(
                CompactDirectRequest::from_hash_ids_lease(request(751, 0, 1), 8, owner, 8).unwrap(),
            )
            .unwrap();
        assert_eq!(accepted.load(std::sync::atomic::Ordering::SeqCst), 2);
        engine.cancel(first).unwrap();
        assert!(weak.upgrade().is_some());
        drain(&mut engine);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn compact_metadata_preserves_explicit_empty_output_plan() {
        let metadata = DirectRequest {
            output_token_ids: Some(Vec::new()),
            priority: 7,
            strict_priority: 3,
            policy_class: Some("interactive".into()),
            ..request(752, 0, 99)
        };
        let compact = CompactDirectRequest::new(metadata, 8, vec![7], 8).unwrap();
        assert_eq!(
            compact.payload.metadata().output_token_ids,
            Some(Vec::new())
        );
        assert_eq!(compact.payload.metadata().effective_max_output_tokens(), 0);
        assert_eq!(compact.payload.metadata().priority, 7);
        assert_eq!(compact.payload.metadata().strict_priority, 3);
        assert_eq!(
            compact.payload.metadata().policy_class.as_deref(),
            Some("interactive")
        );
    }

    #[test]
    fn compact_lease_is_retained_until_terminal_cancel_or_destruction() {
        for mode in 0..3 {
            let accepted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let owner = Arc::new(TestLease {
                ids: vec![7, 7, 9],
                accepted: Arc::clone(&accepted),
            });
            let weak = Arc::downgrade(&owner);
            let metadata = DirectRequest {
                uuid: Some(Uuid::from_u128(700)),
                max_output_tokens: 3,
                output_token_ids: Some(vec![42, 43, 44]),
                ..Default::default()
            };
            let compact =
                CompactDirectRequest::from_hash_ids_lease(metadata, 17, owner, 8).unwrap();
            let mut engine =
                SteppableEngine::new(ReplayEngineConfig::default(), &ReplayEngineFactory::new())
                    .unwrap();
            let uuid = engine.submit_compact(compact).unwrap();
            assert_eq!(accepted.load(std::sync::atomic::Ordering::SeqCst), 1);
            assert!(weak.upgrade().is_some());
            engine.step_until(0.0).unwrap();
            assert!(
                weak.upgrade().is_some(),
                "materialization is not a lease terminal"
            );
            match mode {
                0 => {
                    let events = drain(&mut engine);
                    assert_eq!(
                        events
                            .iter()
                            .filter_map(|event| event.token_id)
                            .collect::<Vec<_>>(),
                        vec![42, 43, 44]
                    );
                }
                1 => {
                    engine.cancel(uuid).unwrap();
                }
                _ => drop(engine),
            }
            assert!(weak.upgrade().is_none());
        }
    }

    #[test]
    fn rejected_compact_submission_does_not_accept_or_retain_lease() {
        let accepted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let owner = Arc::new(TestLease {
            ids: vec![7],
            accepted: Arc::clone(&accepted),
        });
        let weak = Arc::downgrade(&owner);
        let mut metadata = request(701, 0, 1);
        metadata.preferred_dp_rank = Some(1);
        let compact = CompactDirectRequest::from_hash_ids_lease(metadata, 8, owner, 8).unwrap();
        let mut engine =
            SteppableEngine::new(ReplayEngineConfig::default(), &ReplayEngineFactory::new())
                .unwrap();
        assert!(engine.submit_compact(compact).is_err());
        assert_eq!(accepted.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(weak.upgrade().is_none());
        assert_eq!(engine.in_flight(), 0);
    }

    #[test]
    fn failed_atomic_batch_preserves_an_accepted_compact_owner() {
        for disagg in [false, true] {
            let factory = ReplayEngineFactory::new();
            let mut engine: Box<dyn SteppableReplay> = if disagg {
                Box::new(
                    SteppableDisagg::new(ReplayEngineConfig::default(), &factory, 1, 1).unwrap(),
                )
            } else {
                Box::new(SteppableEngine::new(ReplayEngineConfig::default(), &factory).unwrap())
            };
            let accepted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let owner = Arc::new(TestLease {
                ids: vec![7],
                accepted: Arc::clone(&accepted),
            });
            let weak = Arc::downgrade(&owner);
            let uuid = engine
                .submit_compact(
                    CompactDirectRequest::from_hash_ids_lease(request(710, 0, 1), 8, owner, 8)
                        .unwrap(),
                )
                .unwrap();
            engine.step_until(0.0).unwrap();
            let error = engine
                .submit_batch(vec![request(711, 8, 1), request(710, 8, 1)])
                .unwrap_err();
            assert!(!error.is_poisoned());
            assert_eq!(engine.in_flight(), 1);
            assert!(weak.upgrade().is_some());
            assert_eq!(accepted.load(std::sync::atomic::Ordering::SeqCst), 1);
            assert!(engine.cancel(Uuid::from_u128(711)).unwrap().is_none());
            assert!(
                engine
                    .submit_batch(vec![request(712, 8, 1), request(712, 8, 1)])
                    .is_err()
            );
            let mut invalid = request(714, 8, 1);
            invalid.preferred_dp_rank = Some(1);
            assert!(
                engine
                    .submit_batch(vec![request(713, 8, 1), invalid])
                    .is_err()
            );
            assert_eq!(engine.in_flight(), 1);
            engine.cancel(uuid).unwrap();
            drain(engine.as_mut());
            assert!(weak.upgrade().is_none());
            assert!(engine.is_idle());
            assert_eq!(
                engine.take_report(0.0).unwrap().request_counts.num_requests,
                1
            );
        }
    }

    #[test]
    fn compact_acceptance_does_not_enter_worker_or_materialize_prompt() {
        let mut engine = SteppableAgg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
        )
        .unwrap();
        engine
            .submit_compact(
                CompactDirectRequest::new(request(720, 0, 1), 17, vec![7, 7, 9], 8).unwrap(),
            )
            .unwrap();
        assert_eq!(engine.live.pending.len(), 1);
        assert!(engine.live.pending[0].materialized_tokens().is_none());
        assert!(
            engine
                .runtime
                .collector()
                .request_admission(Uuid::from_u128(720))
                .is_none()
        );
        assert_eq!(engine.next_event_ms(), Some(0.0));
        engine.advance_now_ms(100.0);
        assert_eq!(
            engine.now_ms(),
            0.0,
            "queued ingress owns the current timestamp"
        );
        drain(&mut engine);
    }

    #[test]
    fn maximum_reuse_accessor_is_not_first_admission_reuse() {
        let mut engine =
            SteppableEngine::new(ReplayEngineConfig::default(), &ReplayEngineFactory::new())
                .unwrap();
        let uuid = Uuid::from_u128(721);
        let collector = engine.inner.runtime.collector_mut();
        collector.try_on_arrival(uuid, 0.0, 32, 1).unwrap();
        collector.on_admit(uuid, 1.0, 8);
        collector.on_admit(uuid, 2.0, 24);
        assert_eq!(engine.request_admission(uuid), Some((1.0, 8)));
        assert_eq!(engine.request_reused_input_tokens(uuid), Some(24));
    }

    #[test]
    fn disaggregated_compact_and_materialized_preserve_causal_transcript() {
        fn run(compact: bool) -> Vec<(f64, EngineEvent)> {
            let mut engine = SteppableDisagg::new(
                ReplayEngineConfig::default(),
                &ReplayEngineFactory::new(),
                2,
                2,
            )
            .unwrap();
            for id in 730..733 {
                let metadata = DirectRequest {
                    output_token_ids: Some(vec![41, 42, 43]),
                    ..request(id, 0, 3)
                };
                if compact {
                    engine
                        .submit_compact(
                            CompactDirectRequest::new(metadata, 17, vec![7, 7, 9], 8).unwrap(),
                        )
                        .unwrap();
                } else {
                    let mut metadata = metadata;
                    metadata.tokens = [vec![7; 16], vec![9]].concat();
                    engine.submit(metadata).unwrap();
                }
            }
            let mut transcript = Vec::new();
            for _ in 0..1000 {
                if engine.is_idle() {
                    assert_eq!(engine.in_flight(), 0);
                    assert!(engine.retiring.is_empty());
                    let report = engine.take_report(0.0).unwrap();
                    assert_eq!(report.request_counts.completed_requests, 3);
                    return transcript;
                }
                let step = engine.step().unwrap();
                transcript.extend(step.events.into_iter().map(|event| (step.end_ms, event)));
            }
            panic!("disaggregated passive replay did not drain");
        }
        assert_eq!(run(true), run(false));
    }

    #[test]
    fn disaggregated_cancellation_retains_owner_until_server_quiescence() {
        let mut engine = SteppableDisagg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            1,
        )
        .unwrap();
        let accepted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let owner = Arc::new(TestLease {
            ids: vec![7; 16],
            accepted,
        });
        let weak = Arc::downgrade(&owner);
        let uuid = engine
            .submit_compact(
                CompactDirectRequest::from_hash_ids_lease(request(740, 0, 64), 128, owner, 8)
                    .unwrap(),
            )
            .unwrap();
        engine.step_until(0.0).unwrap();
        assert_eq!(
            engine.cancel(uuid).unwrap().unwrap().terminal_status,
            Some(ReplayTerminalStatus::Canceled)
        );
        assert_eq!(engine.in_flight(), 0);
        if !engine.is_idle() {
            assert!(weak.upgrade().is_some());
        }
        let events = drain(&mut engine);
        assert!(
            !events
                .iter()
                .any(|event| event.uuid == uuid && event.terminal_status.is_some()),
            "cancel terminal must be delivered once"
        );
        assert!(weak.upgrade().is_none());
        assert!(engine.retiring.is_empty());
        assert_eq!(engine.next_event_ms(), None);
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
}
