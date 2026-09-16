// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::marker::PhantomData;

use crate::engine::generalized::PassId;
use crate::engine::{KvEvent, LifecycleEvent};

use super::core::{EngineEventBatch, EngineProgress};
use crate::engine::HandoffId;
use crate::replay::protocol::{ForwardPassSnapshot, OutputSignal};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum SimulationWorkerStage {
    Aggregated,
    Prefill,
    Decode,
}

impl From<SimulationWorkerStage> for crate::replay::WorkerStage {
    fn from(stage: SimulationWorkerStage) -> Self {
        match stage {
            SimulationWorkerStage::Aggregated => Self::Aggregated,
            SimulationWorkerStage::Prefill => Self::Prefill,
            SimulationWorkerStage::Decode => Self::Decode,
        }
    }
}

#[derive(Debug)]
pub(crate) struct WorkerCompletionPayload<Events: EngineEventBatch = ()> {
    pub(crate) stage: SimulationWorkerStage,
    pub(crate) worker_idx: usize,
    pub(crate) completed_requests: usize,
    pub(crate) output_signals: Vec<OutputSignal>,
    pub(crate) lifecycle_events: Vec<LifecycleEvent>,
    pub(crate) engine_events: Events,
    /// The grouped pass start is needed to normalize pass-end observations to
    /// pass-start visibility in detailed replay artifacts.
    pub(crate) pass_started_at_ms: f64,
    /// Raw pass-end KV events retained only for optional replay artifacts.
    pub(crate) artifact_pass_end_kv_events: Option<Box<[KvEvent]>>,
    pub(crate) progress: EngineProgress,
    pub(crate) fpm: Option<ForwardPassSnapshot>,
    pub(crate) accept_length_output_tokens: usize,
    pub(crate) accept_length_decode_forwards: usize,
}

/// One logical generalized-engine pass whose modeled completion boundary has
/// become visible to the replay event loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EnginePassCompletion<Events: EngineEventBatch = ()> {
    pub(crate) stage: SimulationWorkerStage,
    pub(crate) worker_id: usize,
    pub(crate) pass_id: PassId,
    events: PhantomData<fn() -> Events>,
}

impl<Events: EngineEventBatch> EnginePassCompletion<Events> {
    pub(crate) fn new(stage: SimulationWorkerStage, worker_id: usize, pass_id: PassId) -> Self {
        Self {
            stage,
            worker_id,
            pass_id,
            events: PhantomData,
        }
    }
}

#[derive(Debug)]
pub(crate) enum SimulationEventKind<Events: EngineEventBatch = ()> {
    EnginePassCompletion(EnginePassCompletion<Events>),
    TransferComplete {
        handoff_id: HandoffId,
    },
    WorkerReady {
        stage: SimulationWorkerStage,
        worker_id: usize,
    },
    /// A recurring scaling heartbeat. Payload-free: the scaling snapshot is
    /// gathered from live runtime state when the tick fires. Re-enqueues itself
    /// at the time the scaling policy returns.
    ScalingTick,
    /// A policy-neutral telemetry sample scheduled on the replay virtual clock.
    /// Payload-free: the settled snapshot is gathered when the event fires.
    TelemetryTick,
}

impl<Events: EngineEventBatch> SimulationEventKind<Events> {
    /// Canonical phase order among events at the same logical timestamp.
    /// Runtime drains each phase to a fixed point before telemetry observes the
    /// settled state and scaling makes a decision.
    fn ordering_rank(&self) -> u8 {
        match self {
            SimulationEventKind::EnginePassCompletion(_) => 0,
            SimulationEventKind::WorkerReady { .. } => 1,
            SimulationEventKind::TransferComplete { .. } => 2,
            SimulationEventKind::TelemetryTick => 3,
            SimulationEventKind::ScalingTick => 4,
        }
    }

    fn semantic_cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (
                SimulationEventKind::EnginePassCompletion(left),
                SimulationEventKind::EnginePassCompletion(right),
            ) => (right.stage, right.worker_id, right.pass_id.get()).cmp(&(
                left.stage,
                left.worker_id,
                left.pass_id.get(),
            )),
            (
                SimulationEventKind::WorkerReady {
                    stage: left_stage,
                    worker_id: left_worker,
                },
                SimulationEventKind::WorkerReady {
                    stage: right_stage,
                    worker_id: right_worker,
                },
            ) => (*right_stage, *right_worker).cmp(&(*left_stage, *left_worker)),
            (
                SimulationEventKind::TransferComplete { handoff_id: left },
                SimulationEventKind::TransferComplete { handoff_id: right },
            ) => right.cmp(left),
            _ => Ordering::Equal,
        }
    }
}

#[derive(Debug)]
pub(crate) struct SimulationEvent<Events: EngineEventBatch = ()> {
    pub(crate) at_ms: f64,
    pub(crate) seq_no: u64,
    pub(crate) kind: SimulationEventKind<Events>,
}

impl<Events: EngineEventBatch> PartialEq for SimulationEvent<Events> {
    fn eq(&self, other: &Self) -> bool {
        self.at_ms.to_bits() == other.at_ms.to_bits() && self.seq_no == other.seq_no
    }
}

impl<Events: EngineEventBatch> Eq for SimulationEvent<Events> {}

impl<Events: EngineEventBatch> PartialOrd for SimulationEvent<Events> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Events: EngineEventBatch> Ord for SimulationEvent<Events> {
    fn cmp(&self, other: &Self) -> Ordering {
        // `total_cmp`, not `partial_cmp`: this feeds a `BinaryHeap`, which requires a
        // total order agreeing with `Eq`. `partial_cmp(..).unwrap_or(Equal)` makes a NaN
        // timestamp compare equal to everything, silently corrupting heap ordering
        // instead of surfacing the bad input, and disagrees with the bitwise `PartialEq`
        // above on signed zero.
        other
            .at_ms
            .total_cmp(&self.at_ms)
            .then_with(|| other.kind.ordering_rank().cmp(&self.kind.ordering_rank()))
            .then_with(|| self.kind.semantic_cmp(&other.kind))
            .then_with(|| other.seq_no.cmp(&self.seq_no))
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    use super::{SimulationEvent, SimulationEventKind};

    fn tick(at_ms: f64, seq_no: u64) -> SimulationEvent {
        SimulationEvent {
            at_ms,
            seq_no,
            kind: SimulationEventKind::TelemetryTick,
        }
    }

    /// `BinaryHeap` requires `Ord` to be a total order that agrees with `Eq`.
    /// `PartialEq` here is bitwise, so NaN and signed zero are the two inputs
    /// where a non-total comparison silently disagrees with it.
    #[test]
    fn cmp_agrees_with_eq_on_nan_and_signed_zero() {
        let events = [tick(f64::NAN, 0), tick(-0.0, 0), tick(0.0, 0), tick(1.0, 0)];

        for left in &events {
            for right in &events {
                assert_eq!(
                    left.cmp(right) == Ordering::Equal,
                    left == right,
                    "cmp/eq disagree for at_ms {} vs {}",
                    left.at_ms,
                    right.at_ms
                );
            }
        }
    }

    /// A NaN timestamp must not make unrelated events compare equal to each
    /// other transitively, which is what corrupts heap ordering.
    #[test]
    fn nan_event_does_not_collapse_heap_ordering() {
        let mut heap: BinaryHeap<SimulationEvent> = BinaryHeap::new();
        heap.push(tick(3.0, 2));
        heap.push(tick(f64::NAN, 1));
        heap.push(tick(1.0, 0));
        heap.push(tick(2.0, 3));

        let popped: Vec<f64> = std::iter::from_fn(|| heap.pop())
            .map(|event| event.at_ms)
            .collect();

        assert_eq!(popped.len(), 4);
        assert_eq!(popped[..3], [1.0, 2.0, 3.0]);
        assert!(popped[3].is_nan(), "NaN must sort after every real time");
    }
}
