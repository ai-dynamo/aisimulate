// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::marker::PhantomData;

use anyhow::{Result, bail};
use uuid::Uuid;

use super::ReplayMode;
use crate::replay::core::{AdmissionSource as CoreAdmissionSource, ReadyArrival};
use crate::replay::loadgen::{ReplayRequestHashes, ReplayRequestPayload, WorkloadDriver};
use crate::replay::protocol::DirectRequest;

#[doc(hidden)]
pub trait ReplayAdmissionMetadata: Sized {
    fn from_hashes(hashes: Option<ReplayRequestHashes>) -> Self;
    fn for_prefill(self) -> Self;
    fn max_output_tokens_override(&self) -> Option<usize>;
    fn into_hashes(self) -> Option<ReplayRequestHashes>;
}

pub type NoReplayMetadata = ();

impl ReplayAdmissionMetadata for () {
    #[inline]
    fn from_hashes(_hashes: Option<ReplayRequestHashes>) -> Self {}

    #[inline]
    fn for_prefill(self) -> Self {}

    #[inline]
    fn max_output_tokens_override(&self) -> Option<usize> {
        None
    }

    #[inline]
    fn into_hashes(self) -> Option<ReplayRequestHashes> {
        None
    }
}

/// Replay's richer admission record. The placement-facing [`ReadyArrival`]
/// intentionally carries only policy metadata; this sidecar also retains the
/// workload-authored ready time and hashes needed by optional replay artifacts.
pub(crate) struct ReplayReadyArrival<Metadata> {
    pub(crate) request: ReplayRequestPayload,
    pub(crate) arrival_time_ms: f64,
    pub(crate) scheduled_ready_at_ms: f64,
    pub(crate) authored_request_id: Option<String>,
    pub(crate) play_id: Option<String>,
    pub(crate) dispatched_at_ms: f64,
    pub(crate) metadata: Metadata,
    pub(crate) replay_hashes: Option<ReplayRequestHashes>,
    pub(crate) session_id: Option<String>,
    pub(crate) turn_index: Option<usize>,
}

impl<Metadata> ReplayReadyArrival<Metadata> {
    fn into_core(self) -> ReadyArrival<ReplayRequestPayload, Metadata> {
        ReadyArrival {
            request: self.request,
            arrival_time_ms: self.arrival_time_ms,
            metadata: self.metadata,
            authored_request_id: self.authored_request_id,
            play_id: self.play_id,
            dispatched_at_ms: self.dispatched_at_ms,
            session_id: self.session_id,
            turn_index: self.turn_index,
        }
    }
}

#[allow(clippy::large_enum_variant)] // Boxing the workload adds measurable replay hot-path cost.
enum AdmissionSource {
    Requests(VecDeque<DirectRequest>),
    Workload(WorkloadDriver),
}

pub(crate) struct AdmissionQueue<Metadata = NoReplayMetadata> {
    source: AdmissionSource,
    mode: ReplayMode,
    metadata: PhantomData<Metadata>,
}

impl<Metadata: ReplayAdmissionMetadata> AdmissionQueue<Metadata> {
    pub(crate) fn new_requests(source: VecDeque<DirectRequest>, mode: ReplayMode) -> Self {
        Self {
            source: AdmissionSource::Requests(source),
            mode,
            metadata: PhantomData,
        }
    }

    pub(crate) fn new_workload(driver: WorkloadDriver, mode: ReplayMode) -> Self {
        Self {
            source: AdmissionSource::Workload(driver),
            mode,
            metadata: PhantomData,
        }
    }

    pub(crate) fn mode(&self) -> ReplayMode {
        self.mode
    }

    pub(crate) fn next_ready_time_ms(&mut self) -> Option<f64> {
        match (&self.mode, &mut self.source) {
            (ReplayMode::Trace, AdmissionSource::Requests(pending)) => pending
                .front()
                .and_then(|request| request.arrival_timestamp_ms)
                // Never advertise a non-finite arrival as a deadline. This value
                // is reduced with `f64::min`, which returns the non-NaN operand
                // -- so a NaN is silently erased from the choice -- and which
                // propagates `+inf`, advancing `now_ms` to infinity whenever no
                // other event is pending, poisoning every derived latency before
                // the drain's guard can fire. The drain arm below fails loudly on
                // the same request, and `is_drained` reports false while it is
                // still queued, so withholding the deadline cannot hide it.
                .filter(|arrival_ms| arrival_ms.is_finite()),
            (ReplayMode::Trace, AdmissionSource::Workload(driver)) => driver.next_ready_time_ms(),
            // Concurrency: the driver owns the session cap and gates admission, so defer to
            // it directly (no in-flight clamp needed here).
            (ReplayMode::Concurrency { .. }, AdmissionSource::Workload(driver)) => {
                driver.next_ready_time_ms()
            }
            (ReplayMode::Concurrency { .. }, AdmissionSource::Requests(_)) => None,
        }
    }

    /// Offline replay keeps full-prompt workload arrivals compact while they
    /// wait in an aggregated or prefill router queue. Legacy request queues
    /// and cumulative-delta workloads remain materialized because they do not
    /// have an independent compact prompt representation.
    pub(crate) fn drain_ready_compact(
        &mut self,
        now_ms: f64,
        cluster_in_flight: usize,
        retain_artifact_hashes: bool,
    ) -> Result<Vec<ReplayReadyArrival<Metadata>>> {
        self.drain_ready_compact_with(now_ms, cluster_in_flight, retain_artifact_hashes, |ready| {
            ready
        })
    }

    fn drain_ready_compact_with<T>(
        &mut self,
        now_ms: f64,
        cluster_in_flight: usize,
        retain_artifact_hashes: bool,
        mut map: impl FnMut(ReplayReadyArrival<Metadata>) -> T,
    ) -> Result<Vec<T>> {
        match (&self.mode, &mut self.source) {
            (ReplayMode::Trace, AdmissionSource::Requests(pending)) => {
                let mut ready = Vec::new();
                loop {
                    let Some(front) = pending.front() else {
                        break;
                    };
                    // A missing arrival_timestamp_ms is malformed trace data,
                    // not "not ready yet" -- both filter's None and a
                    // legitimate future timestamp end up as no-match here,
                    // and treating the two identically means a queue with a
                    // timestamp-less request at its front never makes
                    // progress again: this loop and next_ready_time_ms both
                    // report nothing to wait for, so nothing ever re-checks
                    // it. Fail closed instead of wedging the whole queue.
                    let Some(arrival_time_ms) = front.arrival_timestamp_ms else {
                        bail!(
                            "offline trace replay request {:?} is missing arrival_timestamp_ms",
                            front.uuid
                        );
                    };
                    // Checked before the readiness comparison so both
                    // non-finite shapes are caught on whichever drain first
                    // sees this request at the front. NaN fails every
                    // comparison, so `NaN > now_ms` is false and the request
                    // is admitted immediately carrying a NaN arrival that
                    // poisons its derived latencies -- `request_latencies`
                    // computes `(first_token_ms - arrival).max(0.0)`, and
                    // `f64::max` returns the non-NaN operand, so TTFT is
                    // silently reported as 0.0 instead of erroring. `+inf`
                    // takes the other branch and is never ready, while
                    // `next_ready_time_ms` keeps advertising it as a pending
                    // deadline. Same class as the ready-time guard in
                    // `WorkloadDriver`, which this admission source bypasses.
                    if !arrival_time_ms.is_finite() {
                        bail!(
                            "offline trace replay request {:?} has a non-finite arrival_timestamp_ms {arrival_time_ms}",
                            front.uuid
                        );
                    }
                    if arrival_time_ms > now_ms {
                        break;
                    }
                    // Cannot fail: `pending.front()` above returned `Some` and
                    // nothing between there and here touches the queue.
                    let Some(request) = pending.pop_front() else {
                        bail!("offline trace replay lost the front request while admitting it");
                    };
                    let (session_id, turn_index) = request
                        .replay_context
                        .as_ref()
                        .map(|context| (context.session_id.clone(), context.turn_index))
                        .unwrap_or_default();
                    ready.push(map(ReplayReadyArrival {
                        request: ReplayRequestPayload::materialized(request),
                        arrival_time_ms,
                        scheduled_ready_at_ms: arrival_time_ms,
                        authored_request_id: None,
                        play_id: None,
                        dispatched_at_ms: now_ms,
                        metadata: Metadata::from_hashes(None),
                        replay_hashes: None,
                        session_id,
                        turn_index,
                    }));
                }
                Ok(ready)
            }
            (ReplayMode::Trace, AdmissionSource::Workload(driver)) => Ok(driver
                .pop_ready_compact(now_ms, usize::MAX)
                .into_iter()
                .map(|ready| {
                    let session_id = ready.emit_session_metadata.then_some(ready.session_id);
                    let turn_index = ready.emit_session_metadata.then_some(ready.turn_index);
                    let replay_hashes = ready.replay_hashes;
                    let (metadata_hashes, replay_hashes) = if retain_artifact_hashes {
                        (replay_hashes.clone(), replay_hashes)
                    } else {
                        (replay_hashes, None)
                    };
                    map(ReplayReadyArrival {
                        request: ready.request,
                        arrival_time_ms: ready.scheduled_ready_at_ms,
                        scheduled_ready_at_ms: ready.scheduled_ready_at_ms,
                        authored_request_id: ready.authored_request_id,
                        play_id: ready.play_id,
                        dispatched_at_ms: ready.dispatched_at_ms,
                        metadata: Metadata::from_hashes(metadata_hashes),
                        replay_hashes,
                        session_id,
                        turn_index,
                    })
                })
                .collect()),
            (ReplayMode::Concurrency { max_in_flight }, AdmissionSource::Requests(pending)) => {
                let mut ready = Vec::new();
                let mut simulated_in_flight = cluster_in_flight;
                while simulated_in_flight < *max_in_flight {
                    let Some(mut request) = pending.pop_front() else {
                        break;
                    };
                    request.arrival_timestamp_ms = Some(now_ms);
                    let (session_id, turn_index) = request
                        .replay_context
                        .as_ref()
                        .map(|context| (context.session_id.clone(), context.turn_index))
                        .unwrap_or_default();
                    ready.push(map(ReplayReadyArrival {
                        request: ReplayRequestPayload::materialized(request),
                        arrival_time_ms: now_ms,
                        scheduled_ready_at_ms: now_ms,
                        authored_request_id: None,
                        play_id: None,
                        dispatched_at_ms: now_ms,
                        metadata: Metadata::from_hashes(None),
                        replay_hashes: None,
                        session_id,
                        turn_index,
                    }));
                    simulated_in_flight += 1;
                }
                Ok(ready)
            }
            (ReplayMode::Concurrency { .. }, AdmissionSource::Workload(driver)) => {
                // The driver owns the session cap and only ever holds active sessions'
                // turns in its heap, so drain everything ready in heap (i.e. limit=usize MAX).
                Ok(driver
                    .pop_ready_compact(now_ms, usize::MAX)
                    .into_iter()
                    .map(|ready| {
                        let session_id = ready.emit_session_metadata.then_some(ready.session_id);
                        let turn_index = ready.emit_session_metadata.then_some(ready.turn_index);
                        let replay_hashes = ready.replay_hashes;
                        let (metadata_hashes, replay_hashes) = if retain_artifact_hashes {
                            (replay_hashes.clone(), replay_hashes)
                        } else {
                            (replay_hashes, None)
                        };
                        map(ReplayReadyArrival {
                            request: ready.request,
                            arrival_time_ms: now_ms,
                            scheduled_ready_at_ms: ready.scheduled_ready_at_ms,
                            authored_request_id: ready.authored_request_id,
                            play_id: ready.play_id,
                            dispatched_at_ms: ready.dispatched_at_ms,
                            metadata: Metadata::from_hashes(metadata_hashes),
                            replay_hashes,
                            session_id,
                            turn_index,
                        })
                    })
                    .collect())
            }
        }
    }

    pub(crate) fn on_request_terminal(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        status: crate::replay::ReplayTerminalStatus,
    ) -> Result<()> {
        let AdmissionSource::Workload(driver) = &mut self.source else {
            return Ok(());
        };
        driver.on_terminal(uuid, now_ms, status)
    }

    pub(crate) fn on_request_causal_terminal(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        status: crate::replay::ReplayTerminalStatus,
    ) -> Result<()> {
        let AdmissionSource::Workload(driver) = &mut self.source else {
            return Ok(());
        };
        driver.on_causal_terminal(uuid, now_ms, status)
    }

    pub(crate) fn on_request_quiescent(&mut self, uuid: Uuid, now_ms: f64) -> Result<()> {
        let AdmissionSource::Workload(driver) = &mut self.source else {
            return Ok(());
        };
        driver.on_quiescent(uuid, now_ms)
    }

    pub(crate) fn on_output_token(&mut self, uuid: Uuid, token_id: u32) -> Result<()> {
        let AdmissionSource::Workload(driver) = &mut self.source else {
            return Ok(());
        };
        driver.on_output_token(uuid, token_id)
    }

    pub(crate) fn is_drained(&self) -> bool {
        match &self.source {
            AdmissionSource::Requests(pending) => pending.is_empty(),
            AdmissionSource::Workload(driver) => driver.is_drained(),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_workload(&self) -> bool {
        matches!(self.source, AdmissionSource::Workload(_))
    }

    pub(crate) fn total_requests(&self) -> usize {
        match &self.source {
            AdmissionSource::Requests(pending) => pending.len(),
            AdmissionSource::Workload(driver) => driver.total_turns(),
        }
    }

    pub(crate) fn agentic_trajectory_snapshot(
        &self,
    ) -> Option<crate::replay::loadgen::AgenticTrajectorySnapshot> {
        let AdmissionSource::Workload(driver) = &self.source else {
            return None;
        };
        driver.agentic_trajectory_snapshot()
    }

    pub(crate) fn agentic_graph_identity(
        &self,
    ) -> Option<crate::replay::loadgen::AgenticGraphIdentity> {
        let AdmissionSource::Workload(driver) = &self.source else {
            return None;
        };
        driver.agentic_graph_identity()
    }
}

impl<Metadata: ReplayAdmissionMetadata> CoreAdmissionSource for AdmissionQueue<Metadata> {
    type Request = ReplayRequestPayload;
    type Metadata = Metadata;

    fn next_ready_time_ms(&mut self) -> Option<f64> {
        AdmissionQueue::next_ready_time_ms(self)
    }

    fn drain_ready(
        &mut self,
        now_ms: f64,
        cluster_in_flight: usize,
    ) -> Result<Vec<ReadyArrival<Self::Request, Self::Metadata>>> {
        self.drain_ready_compact_with(
            now_ms,
            cluster_in_flight,
            false,
            ReplayReadyArrival::into_core,
        )
    }

    fn on_output_token(&mut self, request_id: Uuid, token_id: u32) -> Result<()> {
        AdmissionQueue::on_output_token(self, request_id, token_id)
    }

    fn on_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
        status: crate::replay::ReplayTerminalStatus,
    ) -> Result<()> {
        AdmissionQueue::on_request_terminal(self, request_id, now_ms, status)
    }

    fn is_drained(&self) -> bool {
        AdmissionQueue::is_drained(self)
    }

    fn total_requests(&self) -> usize {
        AdmissionQueue::total_requests(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A request without an arrival timestamp at the front of a Trace queue
    /// must fail the drain outright, not silently wedge it: filtering it out
    /// as "not ready yet" (the pre-fix shape) reports nothing to admit and
    /// nothing to wait for, so nothing ever re-checks it and every request
    /// behind it in the queue is dropped for the rest of the run.
    #[test]
    fn drain_fails_closed_on_a_front_request_missing_its_arrival_timestamp() {
        let malformed = DirectRequest {
            arrival_timestamp_ms: None,
            ..Default::default()
        };
        let mut queue = AdmissionQueue::<NoReplayMetadata>::new_requests(
            VecDeque::from([malformed]),
            ReplayMode::Trace,
        );

        let error = match queue.drain_ready_compact(0.0, 0, false) {
            Ok(_) => panic!("a missing arrival_timestamp_ms must fail closed, not silently wedge"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("arrival_timestamp_ms"), "{error}");
    }

    /// NaN is admitted immediately (`NaN > now_ms` is false) and then reads
    /// as a 0.0 TTFT downstream; `+inf` is never ready but is still
    /// advertised as a pending deadline. Both must fail the drain, the same
    /// way `WorkloadDriver` rejects a non-finite ready time.
    #[test]
    fn drain_fails_closed_on_a_non_finite_arrival_timestamp() {
        for arrival in [f64::NAN, f64::INFINITY] {
            let malformed = DirectRequest {
                arrival_timestamp_ms: Some(arrival),
                ..Default::default()
            };
            let mut queue = AdmissionQueue::<NoReplayMetadata>::new_requests(
                VecDeque::from([malformed]),
                ReplayMode::Trace,
            );

            let error = match queue.drain_ready_compact(0.0, 0, false) {
                Ok(_) => panic!("a non-finite arrival_timestamp_ms must fail closed: {arrival}"),
                Err(error) => error.to_string(),
            };
            assert!(error.contains("non-finite"), "{error}");
        }
    }

    /// The drain guard above landed without its peek-side counterpart, which the
    /// drain's own comment already described: `next_ready_time_ms` returned the
    /// arrival verbatim, so `+inf` was advertised as the run's next timestamp
    /// (advancing `now_ms` to infinity when nothing else was pending) and NaN was
    /// silently erased by the `f64::min` reduction.
    #[test]
    fn a_non_finite_arrival_is_never_advertised_as_a_deadline() {
        for arrival in [f64::NAN, f64::INFINITY] {
            let malformed = DirectRequest {
                arrival_timestamp_ms: Some(arrival),
                ..Default::default()
            };
            let mut queue = AdmissionQueue::<NoReplayMetadata>::new_requests(
                VecDeque::from([malformed]),
                ReplayMode::Trace,
            );

            assert_eq!(
                queue.next_ready_time_ms(),
                None,
                "{arrival} must not be advertised as a deadline"
            );
            // Withholding the deadline must not hide the request: it is still
            // queued, so the run cannot report the workload drained.
            assert!(!queue.is_drained());
        }

        // A well-formed arrival is still advertised.
        let mut queue = AdmissionQueue::<NoReplayMetadata>::new_requests(
            VecDeque::from([DirectRequest {
                arrival_timestamp_ms: Some(25.0),
                ..Default::default()
            }]),
            ReplayMode::Trace,
        );
        assert_eq!(queue.next_ready_time_ms(), Some(25.0));
    }
}
