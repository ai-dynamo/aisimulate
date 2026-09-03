// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use super::*;
use crate::engine::{EngineConfig, TimingModelConfig};
use crate::replay::components::NoReplayMetadata;
use crate::replay::core::NoEngineEvents;
use crate::replay::core::round_robin::AggregatedRoundRobinPlacement;
use crate::replay::engine::{ReplayEngineConfig, ReplayEngineFactory};
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
        None,
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
