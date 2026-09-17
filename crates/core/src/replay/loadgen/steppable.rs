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

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use uuid::Uuid;

use crate::replay::agg::AggRuntimeImpl;
use crate::replay::components::{
    AdmissionQueue, NoReplayMetadata, ReplayAdmissionMetadata, ReplayEngineObservation, ReplayMode,
};
use crate::replay::core::NoEngineEvents;
use crate::replay::core::round_robin::{AggregatedRoundRobinPlacement, PoolRoundRobinPlacement};
use crate::replay::core::{PlacementBatchError, PlacementPolicy, WorkerTopology};
use crate::replay::disagg::DisaggRuntimeImpl;
use crate::replay::engine::{ReplayEngineConfig, ReplayEngineFactory, ReplayRoleFactory};
use crate::replay::loadgen::types::CompactHashIds;
use crate::replay::loadgen::{CompactHashIdsLease, ReplayRequestPayload};
use crate::replay::protocol::DirectRequest;
use crate::replay::{
    OfflineDisaggReplayConfig, ReplayCaptureOptions, ReplayReport, ReplayTerminalStatus,
    SlaThresholds, WorkerStage,
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

/// Error from an atomic direct-request batch submission.
///
/// A normal rejection leaves the replay unchanged and retryable. A poisoned
/// error means a provider/internal fault may have mutated hidden state; the
/// replay rejects every later fallible operation and must be destroyed.
pub type BatchSubmissionError = PlacementBatchError;

/// A full-prompt request in the compact representation emitted by the legacy
/// trace compiler. `hash_ids` contains one canonical identity per trace block;
/// token IDs are synthesized only when a worker admits the request.
#[derive(Debug)]
pub struct CompactDirectRequest {
    /// Request metadata. Its `tokens` field must be empty.
    pub request: DirectRequest,
    /// Logical prompt length in tokens.
    pub input_token_count: usize,
    /// Tokens represented by each element of `hash_ids`.
    pub trace_block_size: usize,
    /// Canonical trace-block identities.
    hash_ids: CompactHashIds,
}

impl CompactDirectRequest {
    /// Creates a compact request that owns its hash IDs, preserving the
    /// copied V1 compact-submission route.
    pub fn owned(
        request: DirectRequest,
        input_token_count: usize,
        trace_block_size: usize,
        hash_ids: Vec<u32>,
    ) -> Self {
        Self {
            request,
            input_token_count,
            trace_block_size,
            hash_ids: hash_ids.into(),
        }
    }

    /// Creates a compact request backed by a provider-owned lifecycle lease.
    ///
    /// The replay retains this lease until the request reaches a terminal
    /// state, is cancelled, or the replay is destroyed. This is the no-copy
    /// route for providers that can expose their compact IDs through a safe
    /// owner, such as an `Arc`-backed Dynamo request bundle.
    pub fn leased(
        request: DirectRequest,
        input_token_count: usize,
        trace_block_size: usize,
        hash_ids: Arc<dyn CompactHashIdsLease>,
    ) -> Self {
        Self {
            request,
            input_token_count,
            trace_block_size,
            hash_ids: CompactHashIds::Leased(hash_ids),
        }
    }

    fn into_payload(
        self,
    ) -> anyhow::Result<(ReplayRequestPayload, Option<Arc<dyn CompactHashIdsLease>>)> {
        anyhow::ensure!(
            self.request.tokens.is_empty(),
            "compact request must not include raw tokens"
        );
        anyhow::ensure!(
            self.trace_block_size > 0,
            "compact request trace_block_size must be positive"
        );
        let expected_blocks = self.input_token_count.div_ceil(self.trace_block_size);
        anyhow::ensure!(
            self.hash_ids.as_slice().len() == expected_blocks,
            "compact request has {} hash ids for {} tokens at trace block size {}; expected {expected_blocks}",
            self.hash_ids.as_slice().len(),
            self.input_token_count,
            self.trace_block_size,
        );
        let retained_lease = self.hash_ids.retained_lease();
        Ok((
            ReplayRequestPayload::deferred(
                self.request,
                self.input_token_count,
                self.hash_ids,
                self.trace_block_size,
            ),
            retained_lease,
        ))
    }
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
    ///
    /// A non-finite `now_ms` is a caller bug, not a no-op. It cannot be
    /// reported through this signature, so implementations log it and leave the
    /// clock untouched; the sibling [`Self::step_until`] rejects the same value
    /// with an error.
    fn advance_now_ms(&mut self, now_ms: f64);

    /// Admit `request` at the current simulated time. The returned id
    /// correlates it with later [`EngineEvent`]s. Explicit UUIDs must be unique
    /// among live requests and measurements retained in the current report
    /// epoch. A successful [`Self::take_report`] permits reuse in the next epoch.
    fn submit(&mut self, request: DirectRequest) -> anyhow::Result<Uuid>;

    /// Atomically admit a materialized direct-request batch at the current
    /// simulated time.
    ///
    /// This deliberately has no compact-request counterpart: compact input may
    /// borrow provider-owned storage, while this transaction owns only the
    /// already-materialized requests supplied by its caller. Implementations
    /// must either accept every request or leave the replay unchanged.
    fn submit_batch(
        &mut self,
        _requests: Vec<DirectRequest>,
    ) -> Result<Vec<Uuid>, BatchSubmissionError> {
        Err(BatchSubmissionError::unchanged(anyhow::anyhow!(
            "this steppable replay does not support atomic direct batch submission"
        )))
    }

    /// Admit a compact full-prompt trace at the current simulated time.
    fn submit_compact(&mut self, _request: CompactDirectRequest) -> anyhow::Result<Uuid> {
        anyhow::bail!("this steppable replay does not support compact trace submission")
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
    ///
    /// Thresholds that fail `SlaThresholds::validate` (non-finite, non-positive,
    /// or `e2e_ms` combined with `ttft_ms`/`itl_ms`) are refused and the previous
    /// thresholds are kept, so the steppable seam and `Replayer` agree on which
    /// SLAs are authorable. Like [`Self::advance_now_ms`], this signature has no
    /// error to return, so the refusal is logged.
    fn set_sla_thresholds(&mut self, sla: SlaThresholds);

    /// Measured `(ttft_ms, mean_itl_ms)` for `uuid` once it has a first token.
    fn request_latencies(&self, uuid: Uuid) -> Option<(f64, f64)>;

    /// First scheduler admission `(at_ms, reused_input_tokens)` for `uuid`.
    fn request_admission(&self, uuid: Uuid) -> Option<(f64, usize)>;

    /// Greatest prefix-cache reuse observed across all scheduler admissions
    /// for `uuid`. This remains distinct from [`Self::request_admission`],
    /// whose reuse value is deliberately pinned to the first admission.
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

/// A type-erased placement policy for the aggregated steppable runtime,
/// parameterized on the observation and admission-metadata flavors.
///
/// `ObservationFlavor` (an engine observation *flavor*, e.g. `NoEngineEvents`
/// or `RouterEventObservation`) and the trait's own `Observation` associated
/// type (that flavor's `::Batch`) are two different types one line apart;
/// named distinctly here so the projection is not mistaken for an identity.
pub type DynPlacement<ObservationFlavor, Metadata> = Box<
    dyn PlacementPolicy<
            ReplayRequestPayload,
            Metadata = Metadata,
            Observation = <ObservationFlavor as ReplayEngineObservation>::Batch,
        >,
>;
type SteppableDisaggRuntime<P, O, M> = DisaggRuntimeImpl<P, O, M>;

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
    compact_leases: HashMap<Uuid, Arc<dyn CompactHashIdsLease>>,
}

impl LiveRequests {
    fn insert(&mut self, uuid: Uuid, compact_lease: Option<Arc<dyn CompactHashIdsLease>>) {
        self.uuids.insert(uuid);
        if let Some(compact_lease) = compact_lease {
            compact_lease.on_accepted();
            self.compact_leases.insert(uuid, compact_lease);
        }
    }

    fn remove(&mut self, uuid: Uuid) {
        self.uuids.remove(&uuid);
        self.compact_leases.remove(&uuid);
    }

    fn contains(&self, uuid: Uuid) -> bool {
        self.uuids.contains(&uuid)
    }

    fn len(&self) -> usize {
        self.uuids.len()
    }
}

/// Aggregated multi-worker topology as a [`SteppableReplay`].
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
    engine_block_size: usize,
    poisoned: Option<String>,
}

/// The aggregated role factory both constructors below build identically,
/// parameterized only by the observation flavor that decides whether the
/// engine retains and publishes KV events.
fn aggregated_role_factory<O: ReplayEngineObservation>(
    factory: &ReplayEngineFactory,
    engine: &ReplayEngineConfig,
) -> anyhow::Result<ReplayRoleFactory> {
    Ok(factory.role_factory(
        engine,
        WorkerStage::Aggregated,
        O::capture_engine_kv_events(WorkerStage::Aggregated),
    )?)
}

impl<O, M> SteppableAgg<DynPlacement<O, M>, O, M>
where
    O: ReplayEngineObservation + 'static,
    M: ReplayAdmissionMetadata + 'static,
{
    /// Build an aggregated steppable runtime driven by a caller-supplied,
    /// type-erased placement policy. This is the injection point an
    /// out-of-crate policy uses; the runtime's generics stay private.
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
    /// capture the report carries.
    ///
    /// Without this the steppable seam could not enable canonical or
    /// lifecycle evidence at all, so every report through it had
    /// `runtime_evidence.pressure` and `.kv_ingest` unset. Both are
    /// `skip_serializing_if = "Option::is_none"`, so the canonical JSON
    /// changed shape rather than erroring, and a byte-exact comparison
    /// against a `Replayer::run()` result -- which can set these -- compared
    /// a record with evidence against one without.
    pub fn with_placement_and_capture_options(
        engine: ReplayEngineConfig,
        factory: &ReplayEngineFactory,
        num_workers: usize,
        capture: ReplayCaptureOptions,
        make_placement: impl FnOnce(u32, Vec<WorkerTopology>) -> anyhow::Result<DynPlacement<O, M>>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(num_workers > 0, "num_workers must be positive");
        let engine_block_size = engine.rank.block_size;
        let role_factory = aggregated_role_factory::<O>(factory, &engine)?;
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
            engine_block_size,
            poisoned: None,
        })
    }
}

impl SteppableAgg<AggregatedRoundRobinPlacement<()>, NoEngineEvents, NoReplayMetadata> {
    /// Build an aggregated runtime with `num_workers` round-robin engines.
    pub fn new(
        engine: ReplayEngineConfig,
        factory: &ReplayEngineFactory,
        num_workers: usize,
    ) -> anyhow::Result<Self> {
        Self::new_with_capture_options(
            engine,
            factory,
            num_workers,
            ReplayCaptureOptions::default(),
        )
    }

    /// As [`Self::new`], additionally selecting the detailed capture the
    /// report carries. See
    /// [`SteppableAgg::with_placement_and_capture_options`].
    pub fn new_with_capture_options(
        engine: ReplayEngineConfig,
        factory: &ReplayEngineFactory,
        num_workers: usize,
        capture: ReplayCaptureOptions,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(num_workers > 0, "num_workers must be positive");
        let engine_block_size = engine.rank.block_size;
        let role_factory = aggregated_role_factory::<NoEngineEvents>(factory, &engine)?;
        let runtime = AggRuntimeImpl::new_composed(
            role_factory,
            AdmissionQueue::new_requests(VecDeque::new(), steppable_mode()),
            num_workers,
            None,
            // Unlike `with_placement`, this never calls `topology_settled` on
            // the placement it builds. That's fine here specifically:
            // `AggregatedRoundRobinPlacement::topology_settled` always
            // returns `Ok(Vec::new())`, so it is the stateless case
            // `with_placement`'s own doc comment calls out -- there is no
            // pending-queue state that an unsettled initial topology could
            // leave stuck. An injected policy is not guaranteed that, which
            // is why `with_placement` calls it explicitly.
            |dp_size, topology| Ok(AggregatedRoundRobinPlacement::new(dp_size, topology)),
        )?
        .with_capture_options(capture)
        .into_steppable();
        Ok(Self {
            runtime,
            live: LiveRequests::default(),
            engine_block_size,
            poisoned: None,
        })
    }
}

impl<P, O, M> SteppableAgg<P, O, M>
where
    O: ReplayEngineObservation,
    M: ReplayAdmissionMetadata,
    P: PlacementPolicy<ReplayRequestPayload, Metadata = M, Observation = O::Batch>,
{
    fn ensure_healthy(&self) -> anyhow::Result<()> {
        if let Some(error) = &self.poisoned {
            anyhow::bail!("steppable replay is poisoned after atomic batch failure: {error}");
        }
        Ok(())
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
        if let Some(error) = &self.poisoned {
            tracing::error!(%error, "steppable replay ignored a clock advance after poison");
            return;
        }
        if !now_ms.is_finite() {
            // Not the sanctioned monotonic no-op: the contract admits a time
            // at or before the current one, which NaN and +-inf are not.
            // Folding them into the same silent no-op freezes the simulated
            // clock and yields a wrong report duration with no signal, while
            // `step_until` bails loudly on the very same value. This signature
            // has no error to return, so surface it the way `release_cap_slot`
            // does.
            tracing::error!(
                now_ms,
                "steppable replay ignored a non-finite clock advance"
            );
            return;
        }
        if now_ms > self.runtime.now_ms() {
            self.runtime.advance_now_ms(now_ms);
        }
    }

    fn submit(&mut self, request: DirectRequest) -> anyhow::Result<Uuid> {
        self.ensure_healthy()?;
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
        self.live.insert(uuid, None);
        Ok(uuid)
    }

    fn submit_batch(
        &mut self,
        requests: Vec<DirectRequest>,
    ) -> Result<Vec<Uuid>, BatchSubmissionError> {
        if let Err(error) = self.ensure_healthy() {
            return Err(BatchSubmissionError::poisoned(error));
        }
        // This seam admits requests only between externally-driven steps. No
        // worker pass can run while this method is evaluating, so all
        // externally-rejectable direct-request conflicts are decided before
        // the first admission. The provider always materializes UUIDs before
        // crossing this boundary; nevertheless reject a missing UUID here so
        // the all-or-nothing contract does not depend on that ABI detail.
        let mut batch_ids = HashSet::with_capacity(requests.len());
        for request in &requests {
            let uuid = request.uuid.ok_or_else(|| {
                BatchSubmissionError::unchanged(anyhow::anyhow!(
                    "atomic direct batch request is missing a UUID"
                ))
            })?;
            if !batch_ids.insert(uuid)
                || self.live.contains(uuid)
                || self.runtime.collector().contains_request(uuid)
            {
                return Err(BatchSubmissionError::unchanged(anyhow::anyhow!(
                    "steppable replay request {uuid} is already retained"
                )));
            }
        }

        let accepted = match self.runtime.submit_dynamic_batch(requests) {
            Ok(accepted) => accepted,
            Err(error) => {
                if error.is_poisoned() {
                    self.poisoned = Some(error.to_string());
                }
                return Err(error);
            }
        };
        for uuid in &accepted {
            self.live.insert(*uuid, None);
        }
        Ok(accepted)
    }

    fn submit_compact(&mut self, request: CompactDirectRequest) -> anyhow::Result<Uuid> {
        self.ensure_healthy()?;
        if let Some(uuid) = request.request.uuid
            && self.live.contains(uuid)
        {
            anyhow::bail!("steppable replay request {uuid} is already live");
        }
        if let Some(uuid) = request.request.uuid
            && self.runtime.collector().contains_request(uuid)
        {
            anyhow::bail!("steppable replay request {uuid} is already retained");
        }
        let (payload, compact_lease) = request.into_payload()?;
        let uuid = self
            .runtime
            .submit_dynamic_compact(payload, self.engine_block_size)?;
        self.live.insert(uuid, compact_lease);
        Ok(uuid)
    }

    fn cancel(&mut self, uuid: Uuid) -> anyhow::Result<Option<EngineEvent>> {
        self.ensure_healthy()?;
        let status = self.runtime.cancel_dynamic(uuid)?;
        if status.is_some() {
            self.runtime.discard_step_terminal(uuid);
            self.live.remove(uuid);
        }
        Ok(status.map(|status| EngineEvent::terminal(uuid, status)))
    }

    fn step_until(&mut self, until_ms: f64) -> anyhow::Result<StepOutcome> {
        self.ensure_healthy()?;
        let end_ms = self.runtime.step_dynamic_until(until_ms)?;
        let tokens = self.runtime.take_step_tokens();
        let mut events = tokens
            .into_iter()
            .map(|(uuid, token_id)| EngineEvent::token(uuid, token_id))
            .collect::<Vec<_>>();
        for (uuid, status) in self.runtime.take_step_terminals() {
            self.live.remove(uuid);
            events.push(EngineEvent::terminal(uuid, status));
        }
        Ok(StepOutcome { end_ms, events })
    }

    fn next_event_ms(&mut self) -> Option<f64> {
        if self.poisoned.is_some() {
            return None;
        }
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

    fn request_reused_input_tokens(&self, uuid: Uuid) -> Option<usize> {
        self.runtime.collector().request_reused_input_tokens(uuid)
    }

    fn actual_output_length(&self, uuid: Uuid) -> Option<usize> {
        self.runtime.collector().actual_output_length(uuid)
    }

    fn take_report(&mut self, wall_ms: f64) -> anyhow::Result<ReplayReport> {
        self.ensure_healthy()?;
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
        Self::new_with_capture_options(engine, factory, ReplayCaptureOptions::default())
    }

    /// As [`Self::new`], additionally selecting the detailed capture the
    /// report carries. See
    /// [`SteppableAgg::with_placement_and_capture_options`].
    pub fn new_with_capture_options(
        engine: ReplayEngineConfig,
        factory: &ReplayEngineFactory,
        capture: ReplayCaptureOptions,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            inner: SteppableAgg::new_with_capture_options(engine, factory, 1, capture)?,
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

    fn submit_batch(
        &mut self,
        requests: Vec<DirectRequest>,
    ) -> Result<Vec<Uuid>, BatchSubmissionError> {
        self.inner.submit_batch(requests)
    }

    fn submit_compact(&mut self, request: CompactDirectRequest) -> anyhow::Result<Uuid> {
        self.inner.submit_compact(request)
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

/// Disaggregated prefill/decode topology as a [`SteppableReplay`].
pub struct SteppableDisagg<
    P = PoolRoundRobinPlacement<()>,
    O = NoEngineEvents,
    M = NoReplayMetadata,
> where
    O: ReplayEngineObservation,
    M: ReplayAdmissionMetadata,
    P: PlacementPolicy<ReplayRequestPayload, Metadata = M, Observation = O::Batch>,
{
    runtime: SteppableDisaggRuntime<P, O, M>,
    live: LiveRequests,
    engine_block_size: usize,
}

impl SteppableDisagg<PoolRoundRobinPlacement<()>, NoEngineEvents, NoReplayMetadata> {
    /// Build round-robin prefill and decode pools.
    pub fn new(
        engine: ReplayEngineConfig,
        factory: &ReplayEngineFactory,
        prefill_workers: usize,
        decode_workers: usize,
    ) -> anyhow::Result<Self> {
        Self::new_with_capture_options(
            engine,
            factory,
            prefill_workers,
            decode_workers,
            ReplayCaptureOptions::default(),
        )
    }

    /// As [`Self::new`], additionally selecting the detailed capture the
    /// report carries. See
    /// [`SteppableAgg::with_placement_and_capture_options`].
    pub fn new_with_capture_options(
        engine: ReplayEngineConfig,
        factory: &ReplayEngineFactory,
        prefill_workers: usize,
        decode_workers: usize,
        capture: ReplayCaptureOptions,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(prefill_workers > 0, "num_prefill_workers must be positive");
        anyhow::ensure!(decode_workers > 0, "num_decode_workers must be positive");
        let engine_block_size = engine.rank.block_size;
        // `false`, not `O::capture_engine_kv_events(..)` the way `with_placement`
        // computes it on the aggregated side: this constructor has no
        // observation-flavor type parameter to read one from. `SteppableDisagg`
        // only ever builds `PoolRoundRobinPlacement`, with no injection entry
        // point analogous to `SteppableAgg::with_placement` yet, so there is no
        // caller today who could ask for KV events here. If disagg gains its
        // own `with_placement`, generalize this the same way that one did.
        let config = OfflineDisaggReplayConfig {
            prefill_factory: factory.role_factory(&engine, WorkerStage::Prefill, false)?,
            decode_factory: factory.role_factory(&engine, WorkerStage::Decode, false)?,
            prefill_startup_time_ms: None,
            decode_startup_time_ms: None,
            num_prefill_workers: prefill_workers,
            num_decode_workers: decode_workers,
            handoff_latency_ms: 0.0,
        };
        let runtime = SteppableDisaggRuntime::<
            PoolRoundRobinPlacement<()>,
            NoEngineEvents,
            NoReplayMetadata,
        >::new_composed(
            &config,
            AdmissionQueue::new_requests(VecDeque::new(), steppable_mode()),
            false,
            |_, prefill, _, decode| {
                Ok((
                    PoolRoundRobinPlacement::new(prefill),
                    PoolRoundRobinPlacement::new(decode),
                ))
            },
        )?
        .with_capture_options(capture)
        .into_steppable();
        Ok(Self {
            runtime,
            live: LiveRequests::default(),
            engine_block_size,
        })
    }
}

impl<O, M> SteppableDisagg<DynPlacement<O, M>, O, M>
where
    O: ReplayEngineObservation + 'static,
    M: ReplayAdmissionMetadata + 'static,
{
    /// Build a disaggregated runtime driven by separate caller-supplied
    /// prefill and decode placement policies.
    pub fn with_placements(
        engine: ReplayEngineConfig,
        factory: &ReplayEngineFactory,
        prefill_workers: usize,
        decode_workers: usize,
        make_placements: impl FnOnce(
            u32,
            Vec<WorkerTopology>,
            u32,
            Vec<WorkerTopology>,
        )
            -> anyhow::Result<(DynPlacement<O, M>, DynPlacement<O, M>)>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(prefill_workers > 0, "num_prefill_workers must be positive");
        anyhow::ensure!(decode_workers > 0, "num_decode_workers must be positive");
        let engine_block_size = engine.rank.block_size;
        let config = OfflineDisaggReplayConfig {
            prefill_factory: factory.role_factory(
                &engine,
                WorkerStage::Prefill,
                O::capture_engine_kv_events(WorkerStage::Prefill),
            )?,
            decode_factory: factory.role_factory(
                &engine,
                WorkerStage::Decode,
                O::capture_engine_kv_events(WorkerStage::Decode),
            )?,
            prefill_startup_time_ms: None,
            decode_startup_time_ms: None,
            num_prefill_workers: prefill_workers,
            num_decode_workers: decode_workers,
            handoff_latency_ms: 0.0,
        };
        let runtime = SteppableDisaggRuntime::<DynPlacement<O, M>, O, M>::new_composed(
            &config,
            AdmissionQueue::new_requests(VecDeque::new(), steppable_mode()),
            false,
            |prefill_dp_size, prefill, decode_dp_size, decode| {
                let (mut prefill_placement, mut decode_placement) =
                    make_placements(prefill_dp_size, prefill, decode_dp_size, decode)?;
                let prefill_released = prefill_placement.topology_settled(0.0)?;
                anyhow::ensure!(
                    prefill_released.is_empty(),
                    "prefill placement released {} request(s) before any were submitted",
                    prefill_released.len()
                );
                let decode_released = decode_placement.topology_settled(0.0)?;
                anyhow::ensure!(
                    decode_released.is_empty(),
                    "decode placement released {} request(s) before any were submitted",
                    decode_released.len()
                );
                Ok((prefill_placement, decode_placement))
            },
        )?
        .with_capture_options(ReplayCaptureOptions::default())
        .into_steppable();
        Ok(Self {
            runtime,
            live: LiveRequests::default(),
            engine_block_size,
        })
    }
}

impl<P, O, M> SteppableReplay for SteppableDisagg<P, O, M>
where
    O: ReplayEngineObservation,
    M: ReplayAdmissionMetadata,
    P: PlacementPolicy<ReplayRequestPayload, Metadata = M, Observation = O::Batch>,
{
    fn now_ms(&self) -> f64 {
        self.runtime.now_ms()
    }
    fn advance_now_ms(&mut self, now_ms: f64) {
        if !now_ms.is_finite() {
            // Not the sanctioned monotonic no-op: the contract admits a time
            // at or before the current one, which NaN and +-inf are not.
            // Folding them into the same silent no-op freezes the simulated
            // clock and yields a wrong report duration with no signal, while
            // `step_until` bails loudly on the very same value. This signature
            // has no error to return, so surface it the way `release_cap_slot`
            // does.
            tracing::error!(
                now_ms,
                "steppable replay ignored a non-finite clock advance"
            );
            return;
        }
        if now_ms > self.runtime.now_ms() {
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
        self.live.insert(uuid, None);
        Ok(uuid)
    }
    fn submit_compact(&mut self, request: CompactDirectRequest) -> anyhow::Result<Uuid> {
        if let Some(uuid) = request.request.uuid
            && self.live.contains(uuid)
        {
            anyhow::bail!("steppable replay request {uuid} is already live");
        }
        if let Some(uuid) = request.request.uuid
            && self.runtime.collector().contains_request(uuid)
        {
            anyhow::bail!("steppable replay request {uuid} is already retained");
        }
        let (payload, compact_lease) = request.into_payload()?;
        let uuid = self
            .runtime
            .submit_dynamic_compact(payload, self.engine_block_size)?;
        self.live.insert(uuid, compact_lease);
        Ok(uuid)
    }
    fn cancel(&mut self, uuid: Uuid) -> anyhow::Result<Option<EngineEvent>> {
        let status = self.runtime.cancel_dynamic(uuid)?;
        if status.is_some() {
            self.runtime.discard_step_terminal(uuid);
            self.live.remove(uuid);
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
            self.live.remove(uuid);
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
    fn request_reused_input_tokens(&self, uuid: Uuid) -> Option<usize> {
        self.runtime.collector().request_reused_input_tokens(uuid)
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
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::replay::components::NoReplayMetadata;
    use crate::replay::core::{
        EngineEventBatch, Placement, PlacementBatchEffects, PlacementBatchError,
        PlacementBatchRequest, PlacementDecision, PlacementEffects,
    };

    fn request(uuid: u128, input_length: usize, max_output_tokens: usize) -> DirectRequest {
        DirectRequest {
            tokens: (0..input_length as u32).collect(),
            max_output_tokens,
            uuid: Some(Uuid::from_u128(uuid)),
            arrival_timestamp_ms: Some(0.0),
            ..Default::default()
        }
    }

    /// An observation flavor that actually captures KV events, so a test can
    /// tell the generalized `with_placement` apart from the hardcoded
    /// no-capture path it replaced. `NoEngineEvents` cannot: its flag is
    /// already `false`, so it agrees with a broken implementation.
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
    }

    /// A stateful policy with the two properties the injection seam has to
    /// respect: it refuses to place before its topology is settled, and its
    /// `topology_settled` is a one-shot latch. Counts its own calls so a test
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
            self.calls.borrow_mut().places += 1;
            self.inner.place(request, metadata, session_id, now_ms)
        }

        fn place_batch(
            &mut self,
            requests: Vec<PlacementBatchRequest<'_, ReplayRequestPayload, Self::Metadata>>,
            now_ms: f64,
        ) -> Result<PlacementBatchEffects, PlacementBatchError> {
            if !self.is_settled {
                return Err(PlacementBatchError::unchanged(anyhow::anyhow!(
                    "placed before the topology settled"
                )));
            }
            self.calls.borrow_mut().places += requests.len();
            self.inner.place_batch(requests, now_ms)
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

    /// An injected boxed policy receives the full lifecycle -- the settle
    /// handshake at construction, then placements and terminals -- through the
    /// `Box` forwarding impl.
    #[test]
    fn injected_boxed_placement_receives_the_full_lifecycle() {
        let calls = Rc::new(RefCell::new(PolicyCalls::default()));
        let policy_calls = Rc::clone(&calls);
        let mut engine = SteppableAgg::<
            DynPlacement<NoEngineEvents, NoReplayMetadata>,
            NoEngineEvents,
            NoReplayMetadata,
        >::with_placement(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            2,
            move |dp_size, topology| {
                Ok(Box::new(LatchedPlacement {
                    inner: AggregatedRoundRobinPlacement::new(dp_size, topology),
                    calls: policy_calls,
                    is_settled: false,
                })
                    as DynPlacement<NoEngineEvents, NoReplayMetadata>)
            },
        )
        .expect("engine builds");

        engine.submit(request(1, 64, 4)).expect("submit");
        engine.submit(request(2, 64, 4)).expect("submit");
        drain(&mut engine);

        let calls = calls.borrow();
        assert_eq!(calls.places, 2);
        assert_eq!(calls.terminals, 2);
    }

    /// KV events reach an injected policy when its observation flavor asks for
    /// them.
    ///
    /// This is the behavior `with_placement`'s generalization exists to enable:
    /// the role factory is built with `O::capture_engine_kv_events(...)`, so an
    /// observation whose flag is `true` makes the engine retain and publish
    /// events that then arrive at `observe`. A test using `NoEngineEvents`
    /// cannot show this -- its flag is already `false`, so it passes just as
    /// happily against a hardcoded no-capture role factory.
    #[test]
    fn injected_placement_observes_kv_events_when_its_flavor_captures_them() {
        let calls = Rc::new(RefCell::new(PolicyCalls::default()));
        let policy_calls = Rc::clone(&calls);
        let mut engine = SteppableAgg::<
            DynPlacement<CapturingObservation, NoReplayMetadata>,
            CapturingObservation,
            NoReplayMetadata,
        >::with_placement(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            move |dp_size, topology| {
                Ok(Box::new(LatchedPlacement {
                    inner: AggregatedRoundRobinPlacement::new(dp_size, topology),
                    calls: policy_calls,
                    is_settled: false,
                })
                    as DynPlacement<CapturingObservation, NoReplayMetadata>)
            },
        )
        .expect("engine builds");

        engine.submit(request(1, 256, 8)).expect("submit");
        drain(&mut engine);

        assert!(
            calls.borrow().observed_batches > 0,
            "a capturing observation flavor produced no KV events for the policy"
        );
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
    fn compact_submit_defers_trace_tokens_until_worker_admission() {
        let mut engine =
            SteppableEngine::new(ReplayEngineConfig::default(), &ReplayEngineFactory::new())
                .expect("engine builds");
        let uuid = engine
            .submit_compact(CompactDirectRequest {
                request: DirectRequest {
                    max_output_tokens: 1,
                    uuid: Some(Uuid::from_u128(99)),
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
                input_token_count: 32,
                trace_block_size: 16,
                hash_ids: vec![11, 12].into(),
            })
            .expect("compact request submits");

        assert_eq!(uuid, Uuid::from_u128(99));
        let events = drain(&mut engine);
        assert_eq!(events.iter().filter(|event| event.emitted_token).count(), 1);
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
        let mut aggregated = SteppableAgg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
        )
        .unwrap();

        assert!(aggregated.take_report(f64::NAN).is_err());

        let mut disaggregated = SteppableDisagg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            1,
        )
        .unwrap();
        assert!(disaggregated.take_report(f64::NAN).is_err());
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

        let mut disaggregated = SteppableDisagg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            1,
        )
        .unwrap();
        disaggregated.submit(request(20, 128, 16)).unwrap();

        assert!(disaggregated.take_report(0.0).is_err());
    }

    #[test]
    fn non_finite_clock_advance_is_ignored() {
        let mut engine = SteppableAgg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
        )
        .unwrap();
        // NaN is the one that used to be indistinguishable from the sanctioned
        // monotonic no-op: `now_ms.is_finite() && now_ms > self.now_ms()` is
        // false for NaN for both reasons at once. It is still ignored -- that
        // is this test's authored contract -- but it is now logged rather than
        // silently folded into "the caller asked to go backwards".
        for now_ms in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            engine.advance_now_ms(now_ms);
            assert_eq!(engine.now_ms(), 0.0, "clock moved on {now_ms}");
        }

        // The finite path still advances.
        engine.advance_now_ms(5.0);
        assert_eq!(engine.now_ms(), 5.0);
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

        let mut disaggregated = SteppableDisagg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            1,
        )
        .unwrap();
        let now_ms = disaggregated.now_ms();
        disaggregated.submit(request(21, 128, 16)).unwrap();

        assert_eq!(disaggregated.next_event_ms(), Some(now_ms));
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

        let mut disaggregated =
            SteppableDisagg::new(ReplayEngineConfig::default(), &factory, 1, 1).unwrap();
        assert_canceled(&mut disaggregated, 13);
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
    fn disaggregated_report_duration_includes_trailing_idle_time() {
        let mut engine = SteppableDisagg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            1,
        )
        .unwrap();
        engine.submit(request(43, 128, 1)).unwrap();
        drain(&mut engine);
        engine.advance_now_ms(engine.now_ms() + 1_000.0);

        let report = engine.take_report(0.0).unwrap();
        assert!((report.throughput.duration_ms - engine.now_ms()).abs() < 1e-9);
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
        // Through the public constructor, not by assigning the private
        // `runtime` field: this test used to assert a property the public API
        // could not produce, and it only compiled because `mod tests` is
        // in-module. Every report through the seam therefore had
        // `runtime_evidence.pressure`/`.kv_ingest` unset, and because both are
        // `skip_serializing_if = "Option::is_none"` the canonical JSON changed
        // shape rather than erroring.
        let mut engine = SteppableAgg::new_with_capture_options(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            ReplayCaptureOptions {
                capture_canonical_evidence: true,
                capture_lifecycle_evidence: true,
                ..Default::default()
            },
        )
        .unwrap();
        for ordinal in 1..=2 {
            engine.submit(request(ordinal, 128, 1)).unwrap();
            drain(&mut engine);
            let report = engine.take_report(0.0).unwrap();
            assert!(report.runtime_evidence.pressure.is_some());
            assert!(report.runtime_evidence.kv_ingest.is_some());
        }
    }

    /// Every topology must be able to enable evidence capture, and the
    /// default constructors must keep it off.
    #[test]
    fn capture_options_are_reachable_from_every_steppable_constructor() {
        fn assert_captures(engine: &mut dyn SteppableReplay, uuid: u128, expected: bool) {
            engine.submit(request(uuid, 128, 1)).unwrap();
            drain(engine);
            let report = engine.take_report(0.0).unwrap();
            assert_eq!(report.runtime_evidence.pressure.is_some(), expected);
            assert_eq!(report.runtime_evidence.kv_ingest.is_some(), expected);
        }

        let capture = ReplayCaptureOptions {
            capture_canonical_evidence: true,
            capture_lifecycle_evidence: true,
            ..Default::default()
        };
        let config = ReplayEngineConfig::default;
        let factory = ReplayEngineFactory::new();

        let mut single =
            SteppableEngine::new_with_capture_options(config(), &factory, capture).unwrap();
        assert_captures(&mut single, 41, true);

        let mut disaggregated =
            SteppableDisagg::new_with_capture_options(config(), &factory, 1, 1, capture).unwrap();
        assert_captures(&mut disaggregated, 42, true);

        let mut defaulted = SteppableEngine::new(config(), &factory).unwrap();
        assert_captures(&mut defaulted, 43, false);
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
    fn materialized_direct_batch_rejects_duplicate_ids_without_admission() {
        let factory = ReplayEngineFactory::new();
        let mut engine = SteppableAgg::new(ReplayEngineConfig::default(), &factory, 1).unwrap();

        let error = engine
            .submit_batch(vec![request(20, 128, 16), request(20, 128, 16)])
            .unwrap_err();

        assert!(error.to_string().contains("already retained"), "{error}");
        assert_eq!(engine.in_flight(), 0);
        assert!(engine.is_idle());
        assert_eq!(
            engine.submit(request(20, 128, 16)).unwrap(),
            Uuid::from_u128(20)
        );
    }

    #[derive(Clone, Default)]
    struct RejectsSecondPlacement {
        placements: usize,
    }

    impl PlacementPolicy<ReplayRequestPayload> for RejectsSecondPlacement {
        type Metadata = NoReplayMetadata;
        type Observation = ();

        fn place(
            &mut self,
            request: &ReplayRequestPayload,
            _metadata: Self::Metadata,
            _session_id: Option<String>,
            _now_ms: f64,
        ) -> anyhow::Result<PlacementEffects> {
            self.placements += 1;
            anyhow::ensure!(self.placements != 2, "second placement rejected");
            Ok(PlacementEffects {
                decision: PlacementDecision::Immediate(Placement {
                    request_id: request.metadata().uuid.expect("test request has UUID"),
                    scheduler_id: 0,
                    reported_overlap_tokens: 0,
                    cache_sample: None,
                    placement_replica_id: None,
                }),
                released: Vec::new(),
            })
        }

        fn place_batch(
            &mut self,
            requests: Vec<PlacementBatchRequest<'_, ReplayRequestPayload, Self::Metadata>>,
            now_ms: f64,
        ) -> Result<PlacementBatchEffects, PlacementBatchError> {
            let mut staged = self.clone();
            let mut decisions = Vec::with_capacity(requests.len());
            let mut released = Vec::new();
            for request in requests {
                let effects = staged
                    .place(
                        request.request,
                        request.metadata,
                        request.session_id,
                        now_ms,
                    )
                    .map_err(PlacementBatchError::unchanged)?;
                decisions.push(effects.decision);
                released.extend(effects.released);
            }
            *self = staged;
            Ok(PlacementBatchEffects {
                decisions,
                released,
            })
        }

        fn observe(
            &mut self,
            _observation: Self::Observation,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn cancel_pending(&mut self, _request_id: Uuid) -> bool {
            false
        }

        fn request_terminal(
            &mut self,
            _request_id: Uuid,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn prefill_completed(
            &mut self,
            _request_id: Uuid,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn pending_count(&self) -> usize {
            0
        }

        fn worker_ready(
            &mut self,
            _worker: WorkerTopology,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn worker_draining(
            &mut self,
            _worker: WorkerTopology,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn worker_removed(
            &mut self,
            _worker: WorkerTopology,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn topology_settled(&mut self, _now_ms: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn stateful_batch_rejection_does_not_commit_the_successful_prefix() {
        let mut engine = SteppableAgg::<
            DynPlacement<NoEngineEvents, NoReplayMetadata>,
            NoEngineEvents,
            NoReplayMetadata,
        >::with_placement(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            |_dp_size, _topology| {
                Ok(Box::new(RejectsSecondPlacement::default())
                    as DynPlacement<NoEngineEvents, NoReplayMetadata>)
            },
        )
        .expect("engine builds");

        let error = engine
            .submit_batch(vec![request(30, 128, 1), request(31, 128, 1)])
            .expect_err("second stateful placement rejects the whole batch");

        assert!(!error.is_poisoned(), "staged rejection keeps replay usable");
        assert_eq!(engine.in_flight(), 0);
        assert_eq!(
            engine
                .submit(request(32, 128, 1))
                .expect("the live placement state did not commit the first call"),
            Uuid::from_u128(32)
        );
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

        // Every topology, matching `cancellation_is_immediately_terminal`:
        // the `already live` arm of `SteppableDisagg::submit` is a verbatim
        // copy of the aggregated one rather than a shared helper, so driving
        // only `SteppableAgg` left it with no coverage at all.
        let factory = ReplayEngineFactory::new();
        let mut aggregated = SteppableAgg::new(ReplayEngineConfig::default(), &factory, 1).unwrap();
        assert_rejected(&mut aggregated);

        let mut multi_worker =
            SteppableAgg::new(ReplayEngineConfig::default(), &factory, 2).unwrap();
        assert_rejected(&mut multi_worker);

        let mut single_worker =
            SteppableEngine::new(ReplayEngineConfig::default(), &factory).unwrap();
        assert_rejected(&mut single_worker);

        let mut disaggregated =
            SteppableDisagg::new(ReplayEngineConfig::default(), &factory, 1, 1).unwrap();
        assert_rejected(&mut disaggregated);
    }

    /// The steppable seam forwards caller-supplied floats straight to the
    /// collector, bypassing `SlaThresholds::validate` -- which `Replayer`
    /// enforces. A `NaN` bound made `ttft_ms > bound` false for every request,
    /// so goodput silently equalled total throughput.
    #[test]
    fn steppable_replay_refuses_invalid_sla_thresholds() {
        fn assert_refused(engine: &mut dyn SteppableReplay, uuid: u128) {
            engine.set_sla_thresholds(SlaThresholds {
                ttft_ms: Some(f64::NAN),
                ..Default::default()
            });
            engine.submit(request(uuid, 128, 4)).unwrap();
            drain(engine);
            let report = engine.take_report(engine.now_ms()).unwrap();
            assert!(
                report.goodput.is_none(),
                "an invalid SLA must leave goodput unset, not classify every request as good"
            );
        }

        let factory = ReplayEngineFactory::new();
        let mut aggregated = SteppableAgg::new(ReplayEngineConfig::default(), &factory, 1).unwrap();
        assert_refused(&mut aggregated, 31);

        let mut single_worker =
            SteppableEngine::new(ReplayEngineConfig::default(), &factory).unwrap();
        assert_refused(&mut single_worker, 32);

        let mut disaggregated =
            SteppableDisagg::new(ReplayEngineConfig::default(), &factory, 1, 1).unwrap();
        assert_refused(&mut disaggregated, 33);
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
    fn disaggregated_replay_rejects_a_completed_request_id_before_reporting() {
        let mut engine = SteppableDisagg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            1,
        )
        .unwrap();
        let uuid = engine.submit(request(44, 128, 1)).unwrap();
        drain(&mut engine);

        let error = engine.submit(request(44, 128, 1)).unwrap_err();
        assert!(error.to_string().contains("already retained"), "{error}");
        assert_eq!(engine.in_flight(), 0);

        let report = engine.take_report(engine.now_ms()).unwrap();
        assert_eq!(report.request_counts.num_requests, 1);
        assert_eq!(report.request_counts.completed_requests, 1);
        assert_eq!(uuid, Uuid::from_u128(44));
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
        assert!(SteppableDisagg::new(ReplayEngineConfig::default(), &factory, 0, 1).is_err());
        assert!(SteppableDisagg::new(ReplayEngineConfig::default(), &factory, 1, 0).is_err());
    }

    #[test]
    fn disaggregated_replay_delivers_one_terminal_per_submission() {
        let mut engine = SteppableDisagg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            1,
        )
        .unwrap();
        engine.submit(request(31, 128, 8)).unwrap();
        let events = drain(&mut engine);
        assert_eq!(
            events
                .iter()
                .filter(|event| event.terminal_status.is_some())
                .count(),
            1
        );
    }

    #[test]
    fn disaggregated_cancellation_is_not_reported_again_when_stepped() {
        let mut engine = SteppableDisagg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            1,
        )
        .unwrap();
        let uuid = engine.submit(request(32, 128, 8)).unwrap();

        assert_eq!(
            engine
                .cancel(uuid)
                .unwrap()
                .and_then(|event| event.terminal_status),
            Some(ReplayTerminalStatus::Canceled)
        );
        assert!(
            engine
                .step_until(f64::INFINITY)
                .unwrap()
                .events
                .iter()
                .all(|event| event.terminal_status.is_none())
        );
    }

    #[test]
    fn disaggregated_rejects_a_retained_terminal_request_id() {
        let mut engine = SteppableDisagg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            1,
        )
        .unwrap();
        let uuid = engine.submit(request(34, 128, 8)).unwrap();
        engine.cancel(uuid).unwrap();

        let error = engine.submit(request(34, 128, 8)).unwrap_err();
        assert!(error.to_string().contains("already retained"), "{error}");
        assert_eq!(engine.in_flight(), 0);
        assert_eq!(
            engine
                .take_report(engine.now_ms())
                .unwrap()
                .request_counts
                .num_requests,
            1
        );
        assert_eq!(engine.submit(request(34, 128, 8)).unwrap(), uuid);
    }

    #[test]
    fn disaggregated_cancellation_suppresses_busy_worker_output() {
        let mut engine = SteppableDisagg::new(
            ReplayEngineConfig::default(),
            &ReplayEngineFactory::new(),
            1,
            1,
        )
        .unwrap();
        let uuid = engine.submit(request(33, 128, 8)).unwrap();

        let outcome = engine.step().unwrap();
        assert!(
            outcome.events.is_empty(),
            "the first step should only start the pipeline: {outcome:?}"
        );
        assert_eq!(
            engine
                .cancel(uuid)
                .unwrap()
                .and_then(|event| event.terminal_status),
            Some(ReplayTerminalStatus::Canceled)
        );

        let mut events = Vec::new();
        for _ in 0..32 {
            if engine.is_idle() {
                break;
            }
            events.extend(engine.step().unwrap().events);
        }
        assert!(
            events.iter().all(|event| event.uuid != uuid),
            "canceled request emitted later worker output: {events:?}"
        );
        assert!(
            engine.is_idle(),
            "canceled replay did not drain; next event: {:?}",
            engine.next_event_ms()
        );
    }
}
