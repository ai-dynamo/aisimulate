// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-neutral contracts for offline replay.
//!
//! Runtime adapters and concrete policies live one level up so this directory
//! can later become a standalone crate without deployment-specific APIs.

use std::error::Error;
use std::fmt;

use anyhow::{Result, anyhow};
use uuid::Uuid;

use crate::replay::ReplayTerminalStatus;

pub mod round_robin;

pub trait RequestIdentity {
    fn request_id(&self) -> Option<Uuid>;

    /// An authored attention-DP preference. Policies that do not provide
    /// rank-affinity semantics may ignore it.
    fn preferred_dp_rank(&self) -> Option<u32> {
        None
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
    fn on_output_token(&mut self, request_id: Uuid, token_id: u32) -> Result<()>;
    fn on_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
        status: ReplayTerminalStatus,
    ) -> Result<()>;
    fn is_drained(&self) -> bool;
    fn total_requests(&self) -> usize;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacementCacheSample {
    /// Prefix blocks available on the selected worker.
    pub overlap_blocks: u32,
    /// Largest prefix overlap available on any eligible worker at selection time.
    pub best_available_overlap_blocks: u32,
    pub isl_blocks: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub request_id: Uuid,
    pub scheduler_id: usize,
    pub reported_overlap_tokens: usize,
    pub cache_sample: Option<PlacementCacheSample>,
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

/// One request borrowed by an explicit placement batch transaction.
///
/// Metadata and session ownership move into the transaction. The request
/// payload itself stays in the replay until the placement policy has accepted
/// the complete batch, avoiding a prompt clone at the validation boundary.
pub struct PlacementBatchRequest<'a, Request, Metadata> {
    pub request: &'a Request,
    pub metadata: Metadata,
    pub session_id: Option<String>,
}

/// Placement effects returned only after a complete batch transaction commits.
pub struct PlacementBatchEffects {
    pub decisions: Vec<PlacementDecision>,
    pub released: Vec<Placement>,
}

/// Failure from an explicit placement batch transaction.
///
/// `poisoned` distinguishes an ordinary rejection that left placement state
/// unchanged from a provider/internal fault that may have committed a prefix.
/// Callers must fail-stop the replay after the latter.
#[derive(Debug)]
pub struct PlacementBatchError {
    error: anyhow::Error,
    poisoned: bool,
}

impl PlacementBatchError {
    pub fn unchanged(error: impl Into<anyhow::Error>) -> Self {
        Self {
            error: error.into(),
            poisoned: false,
        }
    }

    pub fn poisoned(error: impl Into<anyhow::Error>) -> Self {
        Self {
            error: error.into(),
            poisoned: true,
        }
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn map_error(self, map: impl FnOnce(anyhow::Error) -> anyhow::Error) -> Self {
        Self {
            error: map(self.error),
            poisoned: self.poisoned,
        }
    }
}

impl fmt::Display for PlacementBatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(formatter)
    }
}

impl Error for PlacementBatchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.error.source()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerTopology {
    pub worker_id: usize,
    pub scheduler_ids: Vec<usize>,
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

    /// Place an externally atomic batch.
    ///
    /// Returning an unchanged error guarantees no placement mutation. A
    /// poisoned error reports that a provider or invariant fault may have
    /// committed a prefix and the enclosing replay must not be used again.
    /// There is deliberately no sequential default: only policies with a real
    /// transaction implementation support this operation.
    fn place_batch(
        &mut self,
        _requests: Vec<PlacementBatchRequest<'_, Request, Self::Metadata>>,
        _now_ms: f64,
    ) -> std::result::Result<PlacementBatchEffects, PlacementBatchError> {
        Err(PlacementBatchError::unchanged(anyhow!(
            "placement policy does not support atomic batch admission"
        )))
    }
    fn observe(&mut self, observation: Self::Observation, now_ms: f64) -> Result<Vec<Placement>>;
    fn cancel_pending(&mut self, request_id: Uuid) -> bool;
    fn request_terminal(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>>;
    fn prefill_completed(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>>;
    fn pending_count(&self) -> usize;
    fn worker_ready(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>>;
    fn worker_draining(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>>;
    fn worker_removed(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>>;
    fn topology_settled(&mut self, now_ms: f64) -> Result<Vec<Placement>>;
}

/// Forwarding impl so a boxed policy is itself a [`PlacementPolicy`], letting
/// a policy constructed outside this crate be injected as
/// `Box<dyn PlacementPolicy<Request>>`. `?Sized` admits `T = dyn
/// PlacementPolicy<Request>`. In particular, an unforwarded default-bodied
/// method would silently skip the inner policy's override rather than fail to
/// compile -- keep every method forwarded here.
impl<Request, T: PlacementPolicy<Request> + ?Sized> PlacementPolicy<Request> for Box<T> {
    type Metadata = T::Metadata;
    type Observation = T::Observation;

    fn place(
        &mut self,
        request: &Request,
        metadata: Self::Metadata,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<PlacementEffects> {
        (**self).place(request, metadata, session_id, now_ms)
    }
    fn place_batch(
        &mut self,
        requests: Vec<PlacementBatchRequest<'_, Request, Self::Metadata>>,
        now_ms: f64,
    ) -> std::result::Result<PlacementBatchEffects, PlacementBatchError> {
        (**self).place_batch(requests, now_ms)
    }
    fn observe(&mut self, observation: Self::Observation, now_ms: f64) -> Result<Vec<Placement>> {
        (**self).observe(observation, now_ms)
    }
    fn cancel_pending(&mut self, request_id: Uuid) -> bool {
        (**self).cancel_pending(request_id)
    }
    fn request_terminal(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        (**self).request_terminal(request_id, now_ms)
    }
    fn prefill_completed(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        (**self).prefill_completed(request_id, now_ms)
    }
    fn pending_count(&self) -> usize {
        (**self).pending_count()
    }
    fn worker_ready(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>> {
        (**self).worker_ready(worker, now_ms)
    }
    fn worker_draining(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>> {
        (**self).worker_draining(worker, now_ms)
    }
    fn worker_removed(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>> {
        (**self).worker_removed(worker, now_ms)
    }
    fn topology_settled(&mut self, now_ms: f64) -> Result<Vec<Placement>> {
        (**self).topology_settled(now_ms)
    }
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
