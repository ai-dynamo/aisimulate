// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::marker::PhantomData;

use anyhow::Result;
use uuid::Uuid;

use super::ReplayMode;
use crate::replay::ReplayTerminalStatus;
use crate::replay::core::{AdmissionSource as CoreAdmissionSource, ReadyArrival};
use crate::replay::loadgen::{
    AgenticFeedbackBatch, AgenticOutputFeedback, AgenticTerminalFeedback, GeneratedRequests,
    ReplayRequestHashes, ReplayRequestPayload, WorkloadDriver,
};
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
    GeneratedRequests(GeneratedRequests),
    Workload {
        driver: WorkloadDriver,
        pending_agentic_feedback: PendingAgenticFeedback,
    },
}

/// Runtime feedback collected beside the workload driver until the current
/// logical timestamp is flushed as one deterministic batch.
#[derive(Default)]
struct PendingAgenticFeedback {
    output_tokens: Vec<AgenticOutputFeedback>,
    causal_terminals: Vec<AgenticTerminalFeedback>,
    quiescent_requests: Vec<Uuid>,
}

impl PendingAgenticFeedback {
    fn is_empty(&self) -> bool {
        self.output_tokens.is_empty()
            && self.causal_terminals.is_empty()
            && self.quiescent_requests.is_empty()
    }

    fn take_batch(&mut self, at_ms: f64) -> AgenticFeedbackBatch {
        AgenticFeedbackBatch {
            at_ms,
            output_tokens: std::mem::take(&mut self.output_tokens),
            causal_terminals: std::mem::take(&mut self.causal_terminals),
            quiescent_requests: std::mem::take(&mut self.quiescent_requests),
        }
    }
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

    pub(crate) fn new_generated_requests(source: GeneratedRequests, max_in_flight: usize) -> Self {
        Self {
            source: AdmissionSource::GeneratedRequests(source),
            mode: ReplayMode::Concurrency { max_in_flight },
            metadata: PhantomData,
        }
    }

    pub(crate) fn new_workload(driver: WorkloadDriver, mode: ReplayMode) -> Self {
        Self {
            source: AdmissionSource::Workload {
                driver,
                pending_agentic_feedback: PendingAgenticFeedback::default(),
            },
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
                .and_then(|request| request.arrival_timestamp_ms),
            (ReplayMode::Trace, AdmissionSource::Workload { driver, .. }) => {
                driver.next_ready_time_ms()
            }
            // Concurrency: the driver owns the session cap and gates admission, so defer to
            // it directly (no in-flight clamp needed here).
            (ReplayMode::Concurrency { .. }, AdmissionSource::Workload { driver, .. }) => {
                driver.next_ready_time_ms()
            }
            (ReplayMode::Concurrency { .. }, AdmissionSource::Requests(_))
            | (_, AdmissionSource::GeneratedRequests(_)) => None,
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
                    let arrival_ms = pending
                        .front()
                        .and_then(|request| request.arrival_timestamp_ms)
                        .filter(|arrival_ms| *arrival_ms <= now_ms);
                    let Some(arrival_time_ms) = arrival_ms else {
                        break;
                    };
                    let request = pending
                        .pop_front()
                        .expect("front request must exist when arrival is ready");
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
            (ReplayMode::Trace, AdmissionSource::Workload { driver, .. }) => Ok(driver
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
                Self::drain_concurrency_requests(
                    now_ms,
                    cluster_in_flight,
                    *max_in_flight,
                    || Ok(pending.pop_front()),
                    map,
                )
            }
            (
                ReplayMode::Concurrency { max_in_flight },
                AdmissionSource::GeneratedRequests(source),
            ) => Self::drain_concurrency_requests(
                now_ms,
                cluster_in_flight,
                *max_in_flight,
                || source.pop_front(),
                map,
            ),
            (ReplayMode::Trace, AdmissionSource::GeneratedRequests(_)) => {
                anyhow::bail!("generated requests require concurrency admission")
            }
            (ReplayMode::Concurrency { .. }, AdmissionSource::Workload { driver, .. }) => {
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

    fn drain_concurrency_requests<T>(
        now_ms: f64,
        cluster_in_flight: usize,
        max_in_flight: usize,
        mut next_request: impl FnMut() -> Result<Option<DirectRequest>>,
        mut map: impl FnMut(ReplayReadyArrival<Metadata>) -> T,
    ) -> Result<Vec<T>> {
        let mut ready = Vec::new();
        let mut simulated_in_flight = cluster_in_flight;
        while simulated_in_flight < max_in_flight {
            let Some(mut request) = next_request()? else {
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

    pub(crate) fn on_request_terminal(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        status: crate::replay::ReplayTerminalStatus,
    ) -> Result<()> {
        let AdmissionSource::Workload { driver, .. } = &mut self.source else {
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
        let AdmissionSource::Workload { driver, .. } = &mut self.source else {
            return Ok(());
        };
        driver.on_causal_terminal(uuid, now_ms, status)
    }

    pub(crate) fn on_request_quiescent(&mut self, uuid: Uuid, now_ms: f64) -> Result<()> {
        let AdmissionSource::Workload { driver, .. } = &mut self.source else {
            return Ok(());
        };
        driver.on_quiescent(uuid, now_ms)
    }

    pub(crate) fn on_output_token(&mut self, uuid: Uuid, token_id: u32) -> Result<()> {
        let AdmissionSource::Workload { driver, .. } = &mut self.source else {
            return Ok(());
        };
        driver.on_output_token(uuid, token_id)
    }

    fn is_agentic(&self) -> bool {
        matches!(&self.source, AdmissionSource::Workload { driver, .. } if driver.is_agentic())
    }

    pub(crate) fn defer_output_token(&mut self, uuid: Uuid, token_id: u32) -> Result<()> {
        if !self.is_agentic() {
            return CoreAdmissionSource::on_output_token(self, uuid, token_id);
        }
        let AdmissionSource::Workload {
            pending_agentic_feedback,
            ..
        } = &mut self.source
        else {
            unreachable!("only an agentic workload can buffer agentic feedback");
        };
        if let Some(existing) = pending_agentic_feedback
            .output_tokens
            .iter_mut()
            .find(|feedback| feedback.request_uuid == uuid)
        {
            existing.token_ids.push(token_id);
        } else {
            pending_agentic_feedback
                .output_tokens
                .push(AgenticOutputFeedback {
                    request_uuid: uuid,
                    token_ids: vec![token_id],
                });
        }
        Ok(())
    }

    pub(crate) fn defer_causal_terminal(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        status: ReplayTerminalStatus,
    ) -> Result<()> {
        if !self.is_agentic() {
            return self.on_request_causal_terminal(uuid, now_ms, status);
        }
        let AdmissionSource::Workload {
            pending_agentic_feedback,
            ..
        } = &mut self.source
        else {
            unreachable!("only an agentic workload can buffer agentic feedback");
        };
        pending_agentic_feedback
            .causal_terminals
            .push(AgenticTerminalFeedback {
                request_uuid: uuid,
                status,
            });
        Ok(())
    }

    pub(crate) fn defer_quiescent(&mut self, uuid: Uuid, now_ms: f64) -> Result<()> {
        if !self.is_agentic() {
            return self.on_request_quiescent(uuid, now_ms);
        }
        let AdmissionSource::Workload {
            pending_agentic_feedback,
            ..
        } = &mut self.source
        else {
            unreachable!("only an agentic workload can buffer agentic feedback");
        };
        pending_agentic_feedback.quiescent_requests.push(uuid);
        Ok(())
    }

    pub(crate) fn defer_terminal(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        status: ReplayTerminalStatus,
    ) -> Result<()> {
        if !self.is_agentic() {
            return CoreAdmissionSource::on_terminal(self, uuid, now_ms, status);
        }
        let AdmissionSource::Workload {
            pending_agentic_feedback,
            ..
        } = &mut self.source
        else {
            unreachable!("only an agentic workload can buffer agentic feedback");
        };
        pending_agentic_feedback
            .causal_terminals
            .push(AgenticTerminalFeedback {
                request_uuid: uuid,
                status,
            });
        pending_agentic_feedback.quiescent_requests.push(uuid);
        Ok(())
    }

    pub(crate) fn flush_agentic_feedback(&mut self, now_ms: f64) -> Result<bool> {
        let AdmissionSource::Workload {
            driver,
            pending_agentic_feedback,
        } = &mut self.source
        else {
            return Ok(false);
        };
        if pending_agentic_feedback.is_empty() {
            return Ok(false);
        }
        driver.apply_agentic_feedback_batch(pending_agentic_feedback.take_batch(now_ms))?;
        Ok(true)
    }

    pub(crate) fn is_drained(&self) -> bool {
        match &self.source {
            AdmissionSource::Requests(pending) => pending.is_empty(),
            AdmissionSource::GeneratedRequests(source) => source.remaining() == 0,
            AdmissionSource::Workload { driver, .. } => driver.is_drained(),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_workload(&self) -> bool {
        matches!(self.source, AdmissionSource::Workload { .. })
    }

    pub(crate) fn total_requests(&self) -> usize {
        match &self.source {
            AdmissionSource::Requests(pending) => pending.len(),
            AdmissionSource::GeneratedRequests(source) => source.remaining(),
            AdmissionSource::Workload { driver, .. } => driver.total_turns(),
        }
    }

    pub(crate) fn agentic_trajectory_snapshot(
        &self,
    ) -> Option<crate::replay::loadgen::AgenticTrajectorySnapshot> {
        let AdmissionSource::Workload { driver, .. } = &self.source else {
            return None;
        };
        driver.agentic_trajectory_snapshot()
    }

    pub(crate) fn agentic_graph_identity(
        &self,
    ) -> Option<crate::replay::loadgen::AgenticGraphIdentity> {
        let AdmissionSource::Workload { driver, .. } = &self.source else {
            return None;
        };
        driver.agentic_graph_identity()
    }

    pub(crate) fn agentic_lifecycle_transcript(
        &self,
    ) -> Option<crate::replay::loadgen::AgenticLifecycleTranscript> {
        let AdmissionSource::Workload { driver, .. } = &self.source else {
            return None;
        };
        driver.agentic_lifecycle_transcript()
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
mod generated_tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn generated_requests_only_fill_vacant_slots_and_keep_original_indices() {
        let indices = Arc::new(Mutex::new(Vec::new()));
        let observed = indices.clone();
        let source = GeneratedRequests::new(1_000_000, move |index| {
            observed.lock().unwrap().push(index);
            Ok(DirectRequest {
                tokens: vec![index as u32],
                uuid: Some(Uuid::from_u128(index as u128 + 10)),
                max_output_tokens: 3,
                arrival_timestamp_ms: Some(999.0),
                priority: 7,
                ..Default::default()
            })
        });
        let mut admission = AdmissionQueue::<()>::new_generated_requests(source, 2);
        assert!(indices.lock().unwrap().is_empty());
        assert_eq!(admission.total_requests(), 1_000_000);
        assert_eq!(admission.next_ready_time_ms(), None);
        assert!(indices.lock().unwrap().is_empty());

        let first = admission.drain_ready_compact(5.0, 0, false).unwrap();
        assert_eq!(*indices.lock().unwrap(), vec![0, 1]);
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].request.metadata().arrival_timestamp_ms, Some(5.0));
        assert_eq!(first[0].request.metadata().priority, 7);
        assert_eq!(first[0].request.metadata().uuid, Some(Uuid::from_u128(10)));
        assert!(
            admission
                .drain_ready_compact(6.0, 2, false)
                .unwrap()
                .is_empty()
        );
        assert_eq!(*indices.lock().unwrap(), vec![0, 1]);

        // Completion/cancellation of either live request leaves exactly one slot.
        let next = admission.drain_ready_compact(9.0, 1, false).unwrap();
        assert_eq!(next.len(), 1);
        assert_eq!(*indices.lock().unwrap(), vec![0, 1, 2]);
        assert_eq!(next[0].request.metadata().uuid, Some(Uuid::from_u128(12)));
        assert_eq!(next[0].request.metadata().arrival_timestamp_ms, Some(9.0));
        assert_eq!(admission.total_requests(), 999_997);
        assert!(!admission.is_drained());
        drop(admission);
        assert_eq!(
            Arc::strong_count(&indices),
            1,
            "dropping replay releases its source"
        );
    }

    #[test]
    fn generated_source_stops_at_end_and_propagates_generation_errors() {
        let mut admission = AdmissionQueue::<()>::new_generated_requests(
            GeneratedRequests::new(1, |_| Ok(DirectRequest::default())),
            4,
        );
        assert_eq!(
            admission.drain_ready_compact(0.0, 0, false).unwrap().len(),
            1
        );
        assert!(admission.is_drained());
        assert!(
            admission
                .drain_ready_compact(1.0, 0, false)
                .unwrap()
                .is_empty()
        );

        let mut failing = AdmissionQueue::<()>::new_generated_requests(
            GeneratedRequests::new(1, |_| anyhow::bail!("source failed")),
            1,
        );
        let error = failing.drain_ready_compact(0.0, 0, false).err().unwrap();
        assert_eq!(error.to_string(), "source failed");
    }
}
