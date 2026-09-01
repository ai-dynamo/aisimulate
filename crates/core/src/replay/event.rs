// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::marker::PhantomData;

use crate::engine::generalized::PassId;
use crate::engine::{KvEvent, LifecycleEvent};

use super::core::{EngineEventBatch, EngineProgress};
use crate::engine::HandoffId;
use crate::replay::protocol::{ForwardPassSnapshot, OutputSignal};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Tie-breaker among events at the *same* `at_ms`: telemetry first observes
    /// fully settled workload state, then scaling makes a decision from that
    /// timestamp. `seq_no` is globally unique, so this only reorders control
    /// events relative to same-timestamp work.
    fn ordering_rank(&self) -> u8 {
        match self {
            SimulationEventKind::TelemetryTick => 1,
            SimulationEventKind::ScalingTick => 2,
            _ => 0,
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
        other
            .at_ms
            .partial_cmp(&self.at_ms)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.kind.ordering_rank().cmp(&self.kind.ordering_rank()))
            .then_with(|| other.seq_no.cmp(&self.seq_no))
    }
}
