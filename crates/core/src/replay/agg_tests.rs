// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use super::*;
use crate::engine::{EngineConfig, TimingModelConfig};
use crate::replay::components::NoReplayMetadata;
use crate::replay::core::NoEngineEvents;
use crate::replay::core::round_robin::AggregatedRoundRobinPlacement;
use crate::replay::engine::{ReplayEngineConfig, ReplayEngineFactory};
use crate::replay::scaling::ReplayScalingDecision;
use crate::replay::{ReplayTelemetryObserver, ReplayTelemetrySnapshot, WorkerStage};

type RoundRobinAggRuntime =
    AggRuntimeImpl<AggregatedRoundRobinPlacement<()>, NoEngineEvents, NoReplayMetadata>;

struct NoopTelemetryObserver;

impl ReplayTelemetryObserver for NoopTelemetryObserver {
    fn on_sample(&mut self, _snapshot: ReplayTelemetrySnapshot) -> anyhow::Result<()> {
        Ok(())
    }
}

fn request(uuid: u128, arrival_ms: f64) -> DirectRequest {
    DirectRequest {
        tokens: vec![1; 4],
        max_output_tokens: 1,
        uuid: Some(Uuid::from_u128(uuid)),
        arrival_timestamp_ms: Some(arrival_ms),
        ..Default::default()
    }
}

fn runtime(pending: VecDeque<DirectRequest>) -> RoundRobinAggRuntime {
    runtime_with_startup_time(pending, None)
}

fn runtime_with_startup_time(
    pending: VecDeque<DirectRequest>,
    startup_time_ms: Option<f64>,
) -> RoundRobinAggRuntime {
    let config = ReplayEngineConfig {
        rank: EngineConfig {
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 1.0,
                decode_ms: 1.0,
            },
            ..EngineConfig::default()
        },
        ..ReplayEngineConfig::default()
    };
    let role_factory = ReplayEngineFactory::new()
        .role_factory(&config, WorkerStage::Aggregated, false)
        .unwrap();
    RoundRobinAggRuntime::new_composed(
        role_factory,
        AdmissionQueue::new_requests(pending, ReplayMode::Trace),
        1,
        startup_time_ms,
        |dp_size, topology| Ok(AggregatedRoundRobinPlacement::new(dp_size, topology)),
    )
    .unwrap()
}

#[test]
fn telemetry_only_timestamps_do_not_enter_the_agg_semantic_drain() {
    let pending = VecDeque::from([request(1, 10.0)]);
    let (_, baseline_stats) = runtime(pending.clone()).run().unwrap();
    let (_, observed_stats) = runtime(pending)
        .with_telemetry_observer(1.0, Box::new(NoopTelemetryObserver))
        .run()
        .unwrap();

    assert!(baseline_stats.semantic_drain_count > 1);
    assert_eq!(
        observed_stats.semantic_drain_count, baseline_stats.semantic_drain_count,
        "telemetry-only heartbeats must not wake aggregate semantic replay work"
    );
}

struct MismatchedPlacement;

impl PlacementPolicy<ReplayRequestPayload> for MismatchedPlacement {
    type Metadata = NoReplayMetadata;
    type Observation = ();

    fn place(
        &mut self,
        _request: &ReplayRequestPayload,
        _metadata: Self::Metadata,
        _session_id: Option<String>,
        _now_ms: f64,
    ) -> anyhow::Result<crate::replay::PlacementEffects> {
        Ok(crate::replay::PlacementEffects {
            decision: PlacementDecision::Immediate(Placement {
                request_id: Uuid::from_u128(999),
                scheduler_id: 0,
                reported_overlap_tokens: 0,
                cache_sample: None,
                placement_replica_id: None,
            }),
            released: Vec::new(),
        })
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

#[test]
fn mismatched_placement_does_not_retain_arrival_or_offered_traffic() {
    let role_factory = ReplayEngineFactory::new()
        .role_factory(
            &ReplayEngineConfig::default(),
            WorkerStage::Aggregated,
            false,
        )
        .unwrap();
    let mut runtime =
        AggRuntimeImpl::<MismatchedPlacement, NoEngineEvents, NoReplayMetadata>::new_composed(
            role_factory,
            AdmissionQueue::new_requests(VecDeque::new(), ReplayMode::Trace),
            1,
            None,
            |_, _| Ok(MismatchedPlacement),
        )
        .unwrap()
        .into_steppable();
    let uuid = Uuid::from_u128(1);
    let error = runtime.submit_dynamic(request(1, 0.0)).unwrap_err();
    assert!(error.to_string().contains("while placing"), "{error}");
    assert!(!runtime.collector.contains_request(uuid));
    assert_eq!(runtime.traffic.drain_planner(1_000.0).num_req, 0);
    assert!(runtime.requests.is_empty());
    assert_eq!(runtime.cluster_in_flight(), 0);
    assert!(runtime.cancel_dynamic(uuid).unwrap().is_none());
    assert_eq!(
        runtime
            .take_report_dynamic(0.0)
            .unwrap()
            .request_counts
            .num_requests,
        0
    );
}

#[test]
fn agg_settled_steps_resume_without_changing_the_result() {
    let pending = VecDeque::from([request(1, 0.0), request(2, 3.0)]);
    let (continuous, _) = runtime(pending.clone()).run().unwrap();

    let mut stepped = runtime(pending);
    let mut settled_boundaries = 0;
    loop {
        match stepped.step().unwrap() {
            ReplayStepOutcome::Settled { .. } => settled_boundaries += 1,
            ReplayStepOutcome::Complete => break,
            ReplayStepOutcome::TimeLimitReached { .. } => panic!("unexpected time limit"),
        }
    }
    let (resumed, _) = stepped.run().unwrap();

    assert!(settled_boundaries >= 2);
    assert_eq!(
        serde_json::to_value(continuous.finish()).unwrap(),
        serde_json::to_value(resumed.finish()).unwrap()
    );
}

#[test]
fn agg_first_step_settles_initial_scaling_and_zero_delay_worker_startup() {
    struct ScaleAtZero;

    impl ReplayScalingPolicy for ScaleAtZero {
        fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
            Ok(0.0)
        }

        fn on_tick(
            &mut self,
            snapshot: ReplayScalingSnapshot,
        ) -> anyhow::Result<ReplayScalingDecision> {
            assert_eq!(snapshot.now_ms, 0.0);
            Ok(ReplayScalingDecision {
                target_decode: Some(2),
                ..Default::default()
            })
        }
    }

    let mut stepped = runtime_with_startup_time(VecDeque::from([request(1, 0.0)]), Some(0.0))
        .with_scaling_policy(Box::new(ScaleAtZero));

    assert_eq!(
        stepped.step().unwrap(),
        ReplayStepOutcome::Settled { now_ms: 0.0 }
    );
    assert_eq!(stepped.engine.active_group_ids(), vec![0, 1]);
    assert!(stepped.engine.starting_group_ids().is_empty());
    assert!(stepped.events.iter().all(|event| event.at_ms > 0.0));
    assert!(stepped.next_timestamps().1.unwrap() > 0.0);
}
