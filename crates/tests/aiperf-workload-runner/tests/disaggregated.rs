// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual prefill/decode deployment hosting shared AgentX workload semantics.

use aisimulate_core::{
    engine::{EngineConfig, TimingModelConfig},
    replay::{ReplayEngineConfig, ReplayEngineFactory, loadgen::SteppableDisagg},
};
use std::rc::Rc;

#[path = "support/run_fixture.rs"]
mod run_fixture;

#[cfg(feature = "kv-router-proof")]
#[test]
fn simulator_owned_agentx_uses_real_dynamo_kv_placement_for_both_pools() {
    use aisimulate_core::replay::{ReplayComposition, ReplayDeterminism};
    use dynamo_mocker::placement::{KvReplayComposition, MockerConfig};
    let directory = tempfile::tempdir().unwrap();
    let raw = run_fixture::branch_request(directory.path());
    let frame = aiperf_protocol::runner_v2::decode_run_request_v2(raw.as_bytes()).unwrap();
    let rank = EngineConfig {
        timing_model: TimingModelConfig::Fixed { prefill_ms: 1.0, decode_ms: 2.0 },
        block_size: 8, max_num_seqs: 64, enable_prefix_caching: true,
        enable_chunked_prefill: false, async_scheduling: Some(false),
        ..Default::default()
    };
    let mut prefill = MockerConfig::default();
    prefill.engine.engine = rank.clone();
    let decode = prefill.clone();
    let mut composition = KvReplayComposition::disaggregated(prefill, decode, 2, 2, None, None, None);
    composition.set_determinism(ReplayDeterminism::CanonicalV1).unwrap();
    let deployment = SteppableDisagg::<_, dynamo_mocker::placement::RouterEventObservation,
        dynamo_mocker::placement::KvReplayMetadata>::with_placements(
        ReplayEngineConfig { rank, ..Default::default() }, &ReplayEngineFactory::new(), 2, 2,
        |pdp, ptopology, ddp, dtopology| composition.create_disaggregated_placements(pdp, ptopology, ddp, dtopology),
    ).unwrap();
    let report = aisim_aiperf_workload_runner::execute_with_deployment(
        &frame.run, aiperf_clock_runtime::cancellation::RunCancellation::new(),
        Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        Box::new(deployment), 64, [7; 32],
    ).expect("actual Dynamo KV placement over disaggregated physical engines");
    assert_eq!(report["engine_drained"], true);
    assert_eq!(report["backend_report"]["completed_requests"], 4);
    for record in report["backend_per_request"].as_array().unwrap() {
        assert!(record["routing_history"].as_array().unwrap().len() >= 2);
        assert!(record["prefill_worker_idx"].is_number());
        assert!(record["decode_worker_idx"].is_number());
        assert_eq!(record["terminal_status"], "completed");
    }
}

#[test]
fn simulator_owned_agentx_branches_cross_actual_prefill_decode_handoff() {
    let directory = tempfile::tempdir().unwrap();
    let raw = run_fixture::branch_request(directory.path());
    let frame = aiperf_protocol::runner_v2::decode_run_request_v2(raw.as_bytes()).unwrap();
    let config = ReplayEngineConfig {
        rank: EngineConfig {
            timing_model: TimingModelConfig::Fixed { prefill_ms: 1.0, decode_ms: 2.0 },
            max_num_seqs: 64,
            enable_prefix_caching: false,
            enable_chunked_prefill: false,
            async_scheduling: Some(false),
            ..Default::default()
        },
        ..Default::default()
    };
    let deployment = SteppableDisagg::new(config, &ReplayEngineFactory::new(), 1, 1).unwrap();
    let report = aisim_aiperf_workload_runner::execute_with_deployment(
        &frame.run,
        aiperf_clock_runtime::cancellation::RunCancellation::new(),
        Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        Box::new(deployment), 64, [7; 32],
    ).expect("shared AgentX over real disaggregated deployment");
    assert_eq!(report["engine_drained"], true);
    assert_eq!(report["requests"].as_array().unwrap().len(), 4);
    let backend = &report["backend_report"];
    assert_eq!(backend["completed_requests"], 4);
    assert!(backend["prefill_worker_seconds"].as_f64().unwrap() > 0.0);
    assert!(backend["decode_worker_seconds"].as_f64().unwrap() > 0.0);
    let records = report["backend_per_request"].as_array().unwrap();
    assert_eq!(records.len(), 4);
    for record in records {
        assert_eq!(record["prefill_worker_idx"], 0);
        assert_eq!(record["decode_worker_idx"], 0);
        assert!(record["prefill_admit_ms"].is_number());
        assert!(record["decode_admit_ms"].is_number());
        assert!(record["source_held_ms"].is_number());
        assert!(record["destination_activated_ms"].is_number());
        assert_eq!(record["terminal_status"], "completed");
    }
}
