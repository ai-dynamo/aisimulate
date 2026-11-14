// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-neutral contracts for offline replay.
//!
//! Runtime adapters and concrete policies live one level up so this directory
//! can later become a standalone crate without deployment-specific APIs.

use anyhow::Result;
use uuid::Uuid;

use crate::replay::ReplayTerminalStatus;
use crate::replay::protocol::{AuthoredKvHints, DirectRequest};

pub mod round_robin;

pub trait RequestIdentity {
    fn request_id(&self) -> Option<Uuid>;

    /// An authored attention-DP preference. Policies that do not provide
    /// rank-affinity semantics may ignore it.
    fn preferred_dp_rank(&self) -> Option<u32> {
        None
    }

    /// An authored peer-fetch hint naming the worker and attention-DP rank that
    /// hold a prefix of the prompt. Policies that do not forward authored hints
    /// may ignore it.
    fn authored_kv_hints(&self) -> Option<AuthoredKvHints> {
        None
    }

    /// Prompt tokens the request brings to its rank, for load-balancing
    /// placement. Zero when unknown.
    fn load_tokens(&self) -> usize {
        0
    }
}

#[derive(Debug)]
pub struct ReadyArrival<Request, Metadata> {
    pub request: Request,
    pub arrival_time_ms: f64,
    pub metadata: Metadata,
    pub authored_request_id: Option<String>,
    pub play_id: Option<String>,
    pub dispatched_at_ms: f64,
    pub session_id: Option<String>,
    pub turn_index: Option<usize>,
}

pub trait AdmissionSource {
    type Request;
    type Metadata;

    fn next_ready_time_ms(&mut self) -> Option<f64>;
    fn drain_ready(
        &mut self,
        now_ms: f64,
        cluster_in_flight: usize,
    ) -> Result<Vec<ReadyArrival<Self::Request, Self::Metadata>>>;
    fn on_output_token(&mut self, request_id: Uuid, token_id: u32, now_ms: f64) -> Result<()>;
    fn on_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
        status: ReplayTerminalStatus,
    ) -> Result<()>;
    fn is_drained(&self) -> bool;
    fn total_requests(&self) -> usize;
}

/// A reactive workload owned outside Replay and fed through the same
/// admission seam as Replay's own workload driver.
///
/// Replay owns the clock. The next instant is the earliest of the next engine
/// event and the source's next wakeup ([`Self::next_wakeup_ms`]), a request to
/// be stepped then that may produce nothing: empty steps are legal. A step is
/// one [`Self::pop_ready`] call. Each instant `t` runs four phases:
///
/// 1. **Wakeup step**, only if `t` is the source's wakeup or feedback awaits
///    it: Replay steps the source once, before engine work at `t` settles. Its
///    submissions are scheduled arrivals, admitted exactly as native arrivals
///    at `t`.
/// 2. **Settle**: engine work due at `t` settles (pass ends, output tokens,
///    terminals), delivered through the feedback callbacks.
/// 3. **Reaction fixpoint**: Replay steps the source again while a step left
///    new feedback at `t`. These submissions are reactions.
/// 4. **Pass starts**: idle workers start passes one at a time in ascending
///    worker order, each worker's attention-DP ranks in ascending rank order
///    within its one pass, as native replay does. Feedback from a pass start,
///    such as an admission rejection, is delivered and the source stepped
///    before the next worker starts, so a reaction joins every pass not yet
///    started and none that has.
///
/// An instant whose only activity is an empty wakeup step leaves engine
/// state untouched, so idle wakeups never change engine timing.
///
/// Progress rules: after every step at `t` the next wakeup is `None` or finite
/// and strictly later than `t`; a source not drained, with nothing in flight
/// and no wakeup, has stalled. Both fail the run with
/// `ReplayError::WorkloadSource`, as do an error the source returns, a reused
/// request `uuid` and an out-of-range rank. A per-instant step budget
/// (`Replayer::with_max_source_steps_per_instant`, default 1,000,000) bounds
/// steps at one instant; exceeding it, which only a reaction loop can, fails
/// with `ReplayError::ResourceLimited` naming the budget and the instant.
///
/// The run takes ownership of the source, so it cannot borrow caller state;
/// share results with the caller through an owned handle such as
/// `Rc<RefCell<_>>`. The source need not be `Send`.
// Deliberately separate from `AdmissionSource`, the runtimes' internal view of
// their admission queue: that trait returns `ReadyArrival` plumbing and has no
// causal-terminal or quiescence split, while this one hands Replay plain
// `DirectRequest`s and reports the client-visible terminal separately from
// engine quiescence.
pub trait ReplayWorkloadSource {
    /// The instant at which the source next wants to be stepped, or `None`
    /// while it only waits for feedback. Replay reads it once before the first
    /// step, where it must be finite and no earlier than the run's first
    /// instant, and after every step at `t`, where it must be `None` or finite
    /// and strictly later than `t`. A wakeup does not promise work.
    fn next_wakeup_ms(&mut self) -> Option<f64>;

    /// One step at `now_ms`: the requests the source submits now, possibly
    /// none. Every request must carry a `uuid` that is unique across the whole
    /// run. Replay stamps each returned request's arrival as `now_ms`.
    ///
    /// `cluster_in_flight` is informational: the requests inside the cluster
    /// when this step is called (in the engine, or waiting on placement,
    /// encoding, tokenization or detokenization). It does not count requests
    /// this source has submitted that Replay still holds, such as this
    /// instant's scheduled arrivals before they are released. The source owns
    /// its concurrency, so a source that bounds its in-flight work counts its
    /// own outstanding submissions rather than gating on this argument.
    fn pop_ready(&mut self, now_ms: f64, cluster_in_flight: usize) -> Result<Vec<DirectRequest>>;

    /// One client-visible output token, produced at replay instant `now_ms`.
    fn on_output_token(&mut self, request_id: Uuid, token_id: u32, now_ms: f64) -> Result<()>;

    /// The request's client-visible terminal.
    fn on_causal_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
        status: ReplayTerminalStatus,
    ) -> Result<()>;

    /// The request no longer holds engine state. Defaults to a no-op.
    fn on_quiescent(&mut self, _request_id: Uuid, _now_ms: f64) -> Result<()> {
        Ok(())
    }

    /// True when the source will submit nothing more and awaits no feedback.
    fn is_drained(&self) -> bool;

    /// Progress-bar total; `0` when unknown.
    fn total_requests_hint(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacementCacheSample {
    /// Prefix blocks available on the selected worker.
    pub overlap_blocks: u32,
    /// Largest prefix overlap available on any eligible worker at selection time.
    pub best_available_overlap_blocks: u32,
    pub isl_blocks: u32,
}

/// Request-scoped KV hints a router attaches to a placement (vLLM `kv_hints`,
/// Dynamo `kv_hint`). Only `kv.fetch` is modeled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvHints {
    pub fetch: KvFetchAction,
}

/// `kv.fetch@1.0`: pull a prompt prefix from a peer rank's G2 tier.
///
/// The source stands in for the hint's `source_control_endpoint`, which
/// identifies one worker and attention-DP rank. It uses the scheduler-id space
/// of the stage that placed the request. The hint's block list is not carried
/// because it does not affect behavior with opportunistic query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvFetchAction {
    pub source_scheduler_id: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub request_id: Uuid,
    pub scheduler_id: usize,
    pub reported_overlap_tokens: usize,
    pub cache_sample: Option<PlacementCacheSample>,
    /// Hints for the receiving engine, or `None` when the policy selected no peer
    /// source. Router-driven policies should set it only toward workers whose
    /// [`WorkerTopology::router_hint`] is true. Replay resolves the source when it
    /// dispatches. It ignores a hint in a pool without KVCR (including the P/D
    /// decode pool), toward the receiving rank itself or toward a source that has
    /// left the pool, and fails for a source scheduler id that never existed.
    pub kv_hints: Option<KvHints>,
    /// Placement-policy replica that made the decision, when the policy has replicas.
    pub placement_replica_id: Option<usize>,
}

#[derive(Debug)]
pub enum PlacementDecision {
    Immediate(Placement),
    Queued,
}

#[derive(Debug)]
pub struct PlacementEffects {
    pub decision: PlacementDecision,
    pub released: Vec<Placement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerTopology {
    pub worker_id: usize,
    pub scheduler_ids: Vec<usize>,
    /// Whether the worker's engines accept KVCR router hints (vLLM
    /// `router_capabilities: ["router_hint"]`).
    pub router_hint: bool,
}

pub trait PlacementPolicy<Request> {
    type Metadata;
    type Observation;

    fn place(
        &mut self,
        request: &Request,
        metadata: Self::Metadata,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<PlacementEffects>;
    fn observe(&mut self, observation: Self::Observation, now_ms: f64) -> Result<Vec<Placement>>;
    /// Engine ownership was accepted: aggregated/prefill submission after
    /// tokenization, or decode destination reservation in P/D mode.
    /// Commit tentative policy state here. Failure aborts the consuming replay
    /// run, calls `dispatch_aborted`, and produces no report.
    fn dispatch_committed(&mut self, _request_id: Uuid, _now_ms: f64) -> Result<()> {
        Ok(())
    }
    /// Discard tentative state after rejected ownership or a failed commit.
    fn dispatch_aborted(&mut self, _request_id: Uuid, _now_ms: f64) -> Result<()> {
        Ok(())
    }
    /// Advance policy time and release work at an existing replay timestamp.
    /// Pending work must be unlocked by an engine event or another callback;
    /// this hook does not create independent timers or advance replay time.
    fn advance_clock(&mut self, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }
    /// Whether this policy needs a settled scheduler load snapshot now.
    /// The runtime asks only at visible decode-pass completion boundaries.
    fn needs_scheduler_load(&self, _scheduler_id: usize, _now_ms: f64, _terminal: bool) -> bool {
        false
    }
    /// Latest scheduler-reported tokens in use plus unallocated queued tokens,
    /// and running plus waiting requests. This is separate from optional KV events.
    fn observe_scheduler_load(
        &mut self,
        _scheduler_id: usize,
        _tokens: usize,
        _requests: usize,
        _now_ms: f64,
    ) {
    }
    fn cancel_pending(&mut self, request_id: Uuid) -> bool;
    /// Discard all request state, including tentative placement state when
    /// cancellation precedes engine dispatch.
    fn request_terminal(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>>;
    /// The request's prefill started: the engine that prefills it admitted it
    /// into a forward pass for the first time. That engine is the aggregated
    /// worker, or the prefill-pool worker in P/D mode; decode-pool admissions
    /// never call this. Like `prefill_completed`, it fires at most once per
    /// request, so readmission after a preemption or retraction does not call
    /// it again.
    ///
    /// `now_ms` is the start of that pass, reported as `first_admit_ms`, and
    /// `reused_input_tokens` is the prompt prefix the admission served from
    /// cache, reported as `first_admission_reused_input_tokens`. The call
    /// follows the pass's start-of-pass KV observations and precedes the
    /// request's `prefill_completed`. A request that ends before its prefill
    /// completes, or an aggregated request whose first output also ends it,
    /// gets `request_terminal` without `prefill_completed`. Returned
    /// placements are dispatched like those from `prefill_completed`.
    fn prefill_started(
        &mut self,
        _request_id: Uuid,
        _reused_input_tokens: usize,
        _now_ms: f64,
    ) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }
    fn prefill_completed(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>>;
    fn pending_count(&self) -> usize;
    fn worker_ready(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>>;
    fn worker_draining(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>>;
    fn worker_removed(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>>;
    fn topology_settled(&mut self, now_ms: f64) -> Result<Vec<Placement>>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct EngineProgress {
    pub(crate) made_progress: bool,
    pub(crate) had_raw_observations: bool,
}

pub trait EngineEventBatch: Default {
    fn is_empty(&self) -> bool;
    fn append(&mut self, other: Self);
}

impl EngineEventBatch for () {
    #[inline]
    fn is_empty(&self) -> bool {
        true
    }

    #[inline]
    fn append(&mut self, _other: Self) {}
}

#[derive(Debug, Default)]
pub struct NoEngineEvents;
