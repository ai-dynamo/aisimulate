// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM async scheduling in offline replay.

use std::collections::VecDeque;

use aisimulate_core::engine::{EngineConfig, TimingModelConfig};
use aisimulate_core::replay::{
    CURRENT_REPLAY_SPEC_VERSION, DirectRequest, PerRequestRecord, ReplayEngineConfig,
    ReplayEngineFactory, ReplayReport, ReplayRuntimeInput, ReplaySpec, ReplayTopology, Replayer,
    WorkerPoolSpec,
};
use uuid::Uuid;

/// One vLLM worker with `num_gpu_blocks` 4-token blocks. A pass with prefill
/// takes 100 ms and a decode pass 10 ms.
fn engine(num_gpu_blocks: usize) -> EngineConfig {
    EngineConfig {
        block_size: 4,
        num_gpu_blocks,
        max_num_batched_tokens: 64,
        timing_model: TimingModelConfig::Fixed {
            prefill_ms: 100.0,
            decode_ms: 10.0,
        },
        ..EngineConfig::default()
    }
}

/// Run `(id, prompt tokens, outputs, arrival)` requests.
fn run(engine: EngineConfig, requests: &[(u128, u32, usize, f64)]) -> ReplayReport {
    let spec = ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
        encoder: None,
        topology: ReplayTopology::Aggregated {
            workers: WorkerPoolSpec {
                initial_workers: 1,
                ..WorkerPoolSpec::default()
            },
        },
        engine: serde_json::to_value(ReplayEngineConfig {
            rank: engine,
            ..Default::default()
        })
        .unwrap(),
        adapters: Default::default(),
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests: Vec::new(),
    };
    let requests = requests
        .iter()
        .map(|&(id, prompt, outputs, arrival_ms)| DirectRequest {
            tokens: (0..prompt).map(|token| token + id as u32 * 1000).collect(),
            max_output_tokens: outputs,
            uuid: Some(Uuid::from_u128(id)),
            arrival_timestamp_ms: Some(arrival_ms),
            ..Default::default()
        })
        .collect::<VecDeque<_>>();
    Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Requests(requests))
        .run()
        .unwrap()
}

fn record(report: &ReplayReport, id: u128) -> &PerRequestRecord {
    let uuid = Uuid::from_u128(id).to_string();
    report
        .per_request
        .iter()
        .find(|record| record.uuid == uuid)
        .unwrap()
}

#[test]
fn a_request_waiting_only_for_freed_blocks_is_admitted_when_they_are_freed() {
    // R (24 tokens) prefills until 100 and finishes in its decode pass at
    // 110. W (16 tokens) arrives at 50 and fits only in R's blocks. vLLM frees
    // them after deciding the pass at 110, which has no other work, so it
    // decides again at once and admits W at 110.
    let report = run(engine(10), &[(1, 24, 2, 0.0), (2, 16, 2, 50.0)]);
    assert_eq!(record(&report, 2).first_admit_ms, Some(110.0));
}
