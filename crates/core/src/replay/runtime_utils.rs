// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BinaryHeap;
#[cfg(test)]
use std::collections::VecDeque;

use super::components::ScheduledEngineCompletion;
use super::core::EngineEventBatch;
use super::events::{
    EnginePassCompletion, SimulationEvent, SimulationEventKind, SimulationWorkerStage,
};
use crate::engine::HandoffId;
#[cfg(test)]
use crate::replay::protocol::DirectRequest;

/// A failed dispatch may leave arbitrary policy and reporting side effects.
/// Cleanup is best-effort; no subsequent execution or report is trustworthy.
#[derive(Default)]
pub(super) struct DispatchFailure {
    reason: Option<String>,
}

impl DispatchFailure {
    pub(super) fn ensure_healthy(&self) -> anyhow::Result<()> {
        if let Some(reason) = &self.reason {
            anyhow::bail!(
                "replay is poisoned after a failed dispatch: {reason}; construct a new runtime"
            );
        }
        Ok(())
    }

    pub(super) fn record(
        &mut self,
        mut error: anyhow::Error,
        rollback: anyhow::Result<()>,
        abort: anyhow::Result<()>,
    ) -> anyhow::Error {
        if let Err(cleanup) = rollback {
            error = error.context(format!("engine dispatch rollback failed: {cleanup:#}"));
        }
        if let Err(cleanup) = abort {
            error = error.context(format!("placement dispatch_aborted failed: {cleanup:#}"));
        }
        let reason = format!("{error:#}");
        self.reason.get_or_insert_with(|| reason.clone());
        error.context(format!(
            "replay is poisoned after a failed dispatch: {reason}; construct a new runtime"
        ))
    }
}

/// Result of advancing a replay runtime to its next settled semantic boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ReplayStepOutcome {
    Settled { now_ms: f64 },
    Complete,
    TimeLimitReached { now_ms: f64 },
}

pub(super) fn next_timestamp(
    next_arrival_ms: Option<f64>,
    next_event_ms: Option<f64>,
) -> Option<f64> {
    match (next_arrival_ms, next_event_ms) {
        (Some(arrival_ms), Some(event_ms)) => Some(arrival_ms.min(event_ms)),
        (Some(arrival_ms), None) => Some(arrival_ms),
        (None, Some(event_ms)) => Some(event_ms),
        (None, None) => None,
    }
}

/// A same-time wakeup is allowed while draining immediate policy work, but
/// must be consumed before the timestamp is settled. Check each policy before
/// merging deadlines: `f64::min` can otherwise hide a NaN behind a valid event.
pub(super) fn validate_policy_wakeup(
    wakeup_ms: Option<f64>,
    now_ms: f64,
    role: &str,
    settled: bool,
) -> anyhow::Result<()> {
    if let Some(wakeup_ms) = wakeup_ms {
        anyhow::ensure!(
            wakeup_ms.is_finite() && wakeup_ms >= now_ms,
            "{role} placement policy wakeup must be finite and not precede replay time {now_ms}ms; got {wakeup_ms}ms"
        );
        anyhow::ensure!(
            !settled || wakeup_ms > now_ms,
            "{role} placement policy wakeup at {wakeup_ms}ms made no progress at replay time {now_ms}ms; a settled wakeup must be strictly in the future"
        );
    }
    Ok(())
}

/// Return the earliest scheduled event that can advance replay semantics.
///
/// At most one telemetry heartbeat is armed at a time. Temporarily removing
/// it lets capped/deadlock logic inspect the canonical next event in O(log n)
/// without allowing observation alone to keep or advance the simulation.
pub(super) fn next_non_telemetry_event_ms<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
) -> Option<f64> {
    if !events
        .peek()
        .is_some_and(|event| matches!(event.kind, SimulationEventKind::TelemetryTick))
    {
        return events.peek().map(|event| event.at_ms);
    }

    let telemetry = events
        .pop()
        .expect("peeked telemetry event must remain in the queue");
    debug_assert!(
        !events
            .peek()
            .is_some_and(|event| matches!(event.kind, SimulationEventKind::TelemetryTick)),
        "replay must arm at most one telemetry tick"
    );
    let next_ms = events.peek().map(|event| event.at_ms);
    events.push(telemetry);
    next_ms
}

#[cfg(test)]
pub(super) fn pop_next_trace_ready(
    pending: &mut VecDeque<DirectRequest>,
    now_ms: f64,
) -> Option<(DirectRequest, f64)> {
    let arrival_ms = pending
        .front()
        .and_then(|request| request.arrival_timestamp_ms)
        .filter(|arrival_ms| *arrival_ms <= now_ms)?;
    let request = pending
        .pop_front()
        .expect("front request must exist when arrival is ready");
    Some((request, arrival_ms))
}

#[cfg(test)]
pub(super) fn pop_next_concurrency_ready(
    pending: &mut VecDeque<DirectRequest>,
    now_ms: f64,
    cluster_in_flight: usize,
    max_in_flight: usize,
) -> Option<(DirectRequest, f64)> {
    if cluster_in_flight >= max_in_flight {
        return None;
    }
    let request = pending.pop_front()?;
    Some((request, now_ms))
}

pub(super) fn push_worker_completions<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    next_event_seq: &mut u64,
    scheduled: ScheduledEngineCompletion<Events>,
) {
    let ScheduledEngineCompletion { at_ms, completion } = scheduled;
    events.push(SimulationEvent {
        at_ms,
        seq_no: *next_event_seq,
        kind: SimulationEventKind::EnginePassCompletion(completion),
    });
    *next_event_seq = next_event_seq
        .checked_add(1)
        .expect("offline replay event sequence overflow");
}

pub(super) fn pop_ready_worker_completions<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    now_ms: f64,
) -> Option<EnginePassCompletion<Events>> {
    let event = events.peek()?;
    if event.at_ms != now_ms {
        return None;
    }
    if !matches!(event.kind, SimulationEventKind::EnginePassCompletion(_)) {
        return None;
    }
    let event = events.pop().expect("event must exist after peek");
    match event.kind {
        SimulationEventKind::EnginePassCompletion(completion) => Some(completion),
        SimulationEventKind::TransferComplete { .. }
        | SimulationEventKind::WorkerReady { .. }
        | SimulationEventKind::ScalingTick
        | SimulationEventKind::TelemetryTick => {
            unreachable!("peeked engine completion event must match popped event")
        }
    }
}

pub(super) fn push_transfer_complete<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    next_event_seq: &mut u64,
    at_ms: f64,
    handoff_id: HandoffId,
) {
    events.push(SimulationEvent {
        at_ms,
        seq_no: *next_event_seq,
        kind: SimulationEventKind::TransferComplete { handoff_id },
    });
    *next_event_seq += 1;
}

pub(super) fn pop_ready_transfer_complete<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    now_ms: f64,
) -> Option<HandoffId> {
    let event = events.peek()?;
    if event.at_ms != now_ms {
        return None;
    }
    let SimulationEventKind::TransferComplete { .. } = &event.kind else {
        return None;
    };
    let event = events.pop().expect("event must exist after peek");
    let SimulationEventKind::TransferComplete { handoff_id } = event.kind else {
        unreachable!("peeked decode handoff event must match popped event");
    };
    Some(handoff_id)
}

pub(super) fn push_worker_ready<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    next_event_seq: &mut u64,
    at_ms: f64,
    stage: SimulationWorkerStage,
    worker_id: usize,
) {
    events.push(SimulationEvent {
        at_ms,
        seq_no: *next_event_seq,
        kind: SimulationEventKind::WorkerReady { stage, worker_id },
    });
    *next_event_seq += 1;
}

pub(super) fn pop_ready_worker_ready<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    now_ms: f64,
) -> Option<(SimulationWorkerStage, usize)> {
    let event = events.peek()?;
    if event.at_ms != now_ms {
        return None;
    }
    let SimulationEventKind::WorkerReady { .. } = &event.kind else {
        return None;
    };
    let event = events.pop().expect("event must exist after peek");
    let SimulationEventKind::WorkerReady { stage, worker_id } = event.kind else {
        unreachable!("peeked worker ready event must match popped event");
    };
    Some((stage, worker_id))
}

pub(super) fn push_scaling_tick<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    next_event_seq: &mut u64,
    at_ms: f64,
) {
    events.push(SimulationEvent {
        at_ms,
        seq_no: *next_event_seq,
        kind: SimulationEventKind::ScalingTick,
    });
    *next_event_seq += 1;
}

/// Pop a `ScalingTick` scheduled for exactly `now_ms` (peek-and-pop-at-now, like the
/// other `pop_ready_*` helpers). Payload-free, so it returns whether one fired.
pub(super) fn pop_ready_scaling_tick<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    now_ms: f64,
) -> bool {
    let Some(event) = events.peek() else {
        return false;
    };
    if event.at_ms != now_ms {
        return false;
    }
    if !matches!(event.kind, SimulationEventKind::ScalingTick) {
        return false;
    }
    events.pop().expect("event must exist after peek");
    true
}

pub(super) fn push_telemetry_tick<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    next_event_seq: &mut u64,
    at_ms: f64,
) {
    events.push(SimulationEvent {
        at_ms,
        seq_no: *next_event_seq,
        kind: SimulationEventKind::TelemetryTick,
    });
    *next_event_seq += 1;
}

pub(super) fn pop_ready_telemetry_tick<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    now_ms: f64,
) -> bool {
    let Some(event) = events.peek() else {
        return false;
    };
    if event.at_ms != now_ms {
        return false;
    }
    if !matches!(event.kind, SimulationEventKind::TelemetryTick) {
        return false;
    }
    events.pop().expect("event must exist after peek");
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::generalized::PassId;
    use crate::replay::components::ScheduledEngineCompletion;
    use crate::replay::events::SimulationWorkerStage;
    use uuid::Uuid;

    fn direct_request(uuid: u128, arrival_timestamp_ms: Option<f64>) -> DirectRequest {
        DirectRequest {
            tokens: vec![1; 8],
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(uuid)),
            dp_rank: 0,
            arrival_timestamp_ms,
            ..Default::default()
        }
    }

    #[test]
    fn test_next_timestamp_matches_current_choice_logic() {
        assert_eq!(next_timestamp(Some(1.0), Some(2.0)), Some(1.0));
        assert_eq!(next_timestamp(Some(2.0), Some(1.0)), Some(1.0));
        assert_eq!(next_timestamp(Some(3.0), None), Some(3.0));
        assert_eq!(next_timestamp(None, Some(4.0)), Some(4.0));
        assert_eq!(next_timestamp(None, None), None);
    }

    #[test]
    fn next_non_telemetry_event_ignores_the_armed_heartbeat_without_consuming_it() {
        let mut events: BinaryHeap<SimulationEvent<()>> = BinaryHeap::new();
        let mut next_event_seq = 0;
        push_telemetry_tick(&mut events, &mut next_event_seq, 1.0);

        assert_eq!(next_non_telemetry_event_ms(&mut events), None);
        assert_eq!(events.len(), 1);
        assert_eq!(events.peek().unwrap().at_ms, 1.0);

        push_worker_ready(
            &mut events,
            &mut next_event_seq,
            5.0,
            SimulationWorkerStage::Aggregated,
            0,
        );

        assert_eq!(next_non_telemetry_event_ms(&mut events), Some(5.0));
        assert_eq!(events.len(), 2);
        assert_eq!(events.peek().unwrap().at_ms, 1.0);
        assert!(matches!(
            events.peek().unwrap().kind,
            SimulationEventKind::TelemetryTick
        ));
    }

    #[test]
    fn test_pop_next_trace_ready_releases_only_arrivals_at_or_before_now() {
        let mut pending = VecDeque::from(vec![
            direct_request(1, Some(1.0)),
            direct_request(2, Some(1.1)),
            direct_request(3, Some(2.0)),
        ]);

        let (request_1, arrival_1) = pop_next_trace_ready(&mut pending, 1.0).unwrap();
        assert_eq!(request_1.uuid, Some(Uuid::from_u128(1)));
        assert_eq!(arrival_1, 1.0);

        assert!(pop_next_trace_ready(&mut pending, 1.0).is_none());

        let (request_2, arrival_2) = pop_next_trace_ready(&mut pending, 1.1).unwrap();
        assert_eq!(request_2.uuid, Some(Uuid::from_u128(2)));
        assert_eq!(arrival_2, 1.1);
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn test_pop_next_concurrency_ready_stops_at_max_in_flight() {
        let mut pending = VecDeque::from(vec![direct_request(1, None), direct_request(2, None)]);

        assert!(pop_next_concurrency_ready(&mut pending, 5.0, 2, 2).is_none());

        let (request, arrival_ms) = pop_next_concurrency_ready(&mut pending, 5.0, 1, 2).unwrap();
        assert_eq!(request.uuid, Some(Uuid::from_u128(1)));
        assert_eq!(arrival_ms, 5.0);
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn test_worker_ready_push_pop_round_trip() {
        let mut events: BinaryHeap<SimulationEvent<()>> = BinaryHeap::new();
        let mut next_event_seq = 0;

        push_worker_ready(
            &mut events,
            &mut next_event_seq,
            100.0,
            SimulationWorkerStage::Aggregated,
            3,
        );

        // Not ready before the scheduled time.
        assert!(pop_ready_worker_ready(&mut events, 99.0).is_none());

        let (stage, worker_id) = pop_ready_worker_ready(&mut events, 100.0).unwrap();
        assert_eq!(stage, SimulationWorkerStage::Aggregated);
        assert_eq!(worker_id, 3);
        assert!(events.is_empty());
    }

    #[test]
    fn test_worker_ready_does_not_interfere_with_completion_pop() {
        let mut events: BinaryHeap<SimulationEvent<()>> = BinaryHeap::new();
        let mut next_event_seq = 0;

        push_worker_ready(
            &mut events,
            &mut next_event_seq,
            10.0,
            SimulationWorkerStage::Aggregated,
            1,
        );

        // pop_ready_worker_completions must return None (wrong event kind).
        assert!(pop_ready_worker_completions(&mut events, 10.0).is_none());
        // The event should still be in the heap.
        assert_eq!(events.len(), 1);
        // pop_ready_worker_ready should succeed.
        assert!(pop_ready_worker_ready(&mut events, 10.0).is_some());
    }

    #[test]
    fn same_timestamp_events_follow_semantic_phase_and_identity_order() {
        let mut events: BinaryHeap<SimulationEvent<()>> = BinaryHeap::new();
        let mut next_event_seq = 0;
        push_transfer_complete(
            &mut events,
            &mut next_event_seq,
            10.0,
            HandoffId::new(Uuid::from_u128(2)),
        );
        push_worker_ready(
            &mut events,
            &mut next_event_seq,
            10.0,
            SimulationWorkerStage::Decode,
            3,
        );
        push_worker_completions(
            &mut events,
            &mut next_event_seq,
            ScheduledEngineCompletion {
                at_ms: 10.0,
                completion: EnginePassCompletion::new(SimulationWorkerStage::Decode, 4, PassId(9)),
            },
        );
        push_worker_completions(
            &mut events,
            &mut next_event_seq,
            ScheduledEngineCompletion {
                at_ms: 10.0,
                completion: EnginePassCompletion::new(SimulationWorkerStage::Prefill, 2, PassId(3)),
            },
        );

        let first = pop_ready_worker_completions(&mut events, 10.0).unwrap();
        assert_eq!(first.stage, SimulationWorkerStage::Prefill);
        assert_eq!(first.worker_id, 2);
        let second = pop_ready_worker_completions(&mut events, 10.0).unwrap();
        assert_eq!(second.stage, SimulationWorkerStage::Decode);
        assert_eq!(second.worker_id, 4);
        assert_eq!(
            pop_ready_worker_ready(&mut events, 10.0),
            Some((SimulationWorkerStage::Decode, 3))
        );
        assert_eq!(
            pop_ready_transfer_complete(&mut events, 10.0),
            Some(HandoffId::new(Uuid::from_u128(2)))
        );
    }
}

// Minimal queued placement used to exercise scheduler wakeups through both engines.
#[cfg(test)]
pub(super) mod wakeup_test_policy {
    use super::super::core::{
        Placement, PlacementDecision, PlacementEffects, PlacementPolicy, WorkerTopology,
    };
    use super::super::loadgen::ReplayRequestPayload;
    use uuid::Uuid;

    pub(crate) struct WakeupPlacement {
        delay_ms: Option<f64>,
        wakeup_ms: Option<f64>,
        consume_wakeup: bool,
        pending: Vec<Uuid>,
    }

    impl WakeupPlacement {
        pub(crate) fn new(wakeup_ms: Option<f64>, consume_wakeup: bool) -> Self {
            Self {
                delay_ms: wakeup_ms,
                wakeup_ms: None,
                consume_wakeup,
                pending: Vec::new(),
            }
        }
        fn placement(request_id: Uuid) -> Placement {
            Placement {
                request_id,
                scheduler_id: 0,
                reported_overlap_tokens: 0,
                cache_sample: None,
                placement_replica_id: None,
            }
        }
    }

    impl PlacementPolicy<ReplayRequestPayload> for WakeupPlacement {
        type Metadata = ();
        type Observation = ();
        fn place(
            &mut self,
            request: &ReplayRequestPayload,
            _: (),
            _: Option<String>,
            now_ms: f64,
        ) -> anyhow::Result<PlacementEffects> {
            let id = request.metadata().uuid.unwrap();
            let decision = if let Some(delay_ms) = self.delay_ms {
                self.wakeup_ms = Some(now_ms + delay_ms);
                self.pending.push(id);
                PlacementDecision::Queued
            } else {
                PlacementDecision::Immediate(Self::placement(id))
            };
            Ok(PlacementEffects {
                decision,
                released: Vec::new(),
            })
        }
        fn next_wakeup_ms(&self) -> Option<f64> {
            if self.pending.is_empty() {
                None
            } else {
                self.wakeup_ms
            }
        }
        fn advance_clock(&mut self, now_ms: f64) -> anyhow::Result<Vec<Placement>> {
            if self.consume_wakeup
                && self.wakeup_ms.is_some_and(|wake| wake <= now_ms)
                && !self.pending.is_empty()
            {
                self.wakeup_ms = None;
                self.delay_ms = None;
                return Ok(self.pending.drain(..).map(Self::placement).collect());
            }
            Ok(Vec::new())
        }
        fn observe(&mut self, _: (), _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
        fn cancel_pending(&mut self, id: Uuid) -> bool {
            let before = self.pending.len();
            self.pending.retain(|pending| *pending != id);
            self.pending.len() != before
        }
        fn request_terminal(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
        fn prefill_completed(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
        fn pending_count(&self) -> usize {
            self.pending.len()
        }
        fn worker_ready(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
        fn worker_draining(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
        fn worker_removed(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
        fn topology_settled(&mut self, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
    }
}

#[cfg(test)]
pub(super) mod dispatch_failure_test_policy {
    use super::super::core::{
        Placement, PlacementDecision, PlacementEffects, PlacementPolicy, WorkerTopology,
    };
    use crate::replay::loadgen::ReplayRequestPayload;
    use std::sync::{Arc, Mutex};
    use uuid::Uuid;

    #[derive(Default)]
    pub struct FailingDispatchPlacement {
        pub fail_commit: bool,
        pub fail_abort: bool,
        pub calls: Arc<Mutex<Vec<&'static str>>>,
    }

    impl PlacementPolicy<ReplayRequestPayload> for FailingDispatchPlacement {
        type Metadata = ();
        type Observation = ();

        fn place(
            &mut self,
            request: &ReplayRequestPayload,
            _: (),
            _: Option<String>,
            _: f64,
        ) -> anyhow::Result<PlacementEffects> {
            Ok(PlacementEffects {
                decision: PlacementDecision::Immediate(Placement {
                    request_id: request.metadata().uuid.unwrap(),
                    scheduler_id: 0,
                    reported_overlap_tokens: 0,
                    cache_sample: None,
                    placement_replica_id: None,
                }),
                released: Vec::new(),
            })
        }
        fn dispatch_committed(&mut self, _: Uuid, _: f64) -> anyhow::Result<()> {
            self.calls.lock().unwrap().push("commit");
            anyhow::ensure!(!self.fail_commit, "injected commit failure");
            Ok(())
        }
        fn dispatch_aborted(&mut self, _: Uuid, _: f64) -> anyhow::Result<()> {
            self.calls.lock().unwrap().push("abort");
            anyhow::ensure!(!self.fail_abort, "injected abort failure");
            Ok(())
        }
        fn observe(&mut self, _: (), _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
        fn cancel_pending(&mut self, _: Uuid) -> bool {
            false
        }
        fn request_terminal(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
        fn prefill_completed(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
        fn pending_count(&self) -> usize {
            0
        }
        fn worker_ready(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
        fn worker_draining(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
        fn worker_removed(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
        fn topology_settled(&mut self, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
    }
}

#[cfg(test)]
mod dispatch_failure_tests {
    use super::DispatchFailure;

    #[test]
    fn dispatch_failure_keeps_primary_and_both_cleanup_diagnostics() {
        let mut failure = DispatchFailure::default();
        let error = failure.record(
            anyhow::anyhow!("original dispatch failure"),
            Err(anyhow::anyhow!("rollback failure")),
            Err(anyhow::anyhow!("abort failure")),
        );
        assert_eq!(error.root_cause().to_string(), "original dispatch failure");
        for message in [
            "original dispatch failure",
            "rollback failure",
            "abort failure",
        ] {
            assert!(error.to_string().contains(message));
            assert!(
                failure
                    .ensure_healthy()
                    .unwrap_err()
                    .to_string()
                    .contains(message)
            );
        }
    }
}
