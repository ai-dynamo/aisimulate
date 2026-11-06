// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hash-native preparation exercises the actual shared AgentX workload without
//! any text tokenizer. The production participant owns a panic-on-use text seam.

use std::rc::Rc;

use aiperf_clock_runtime::cancellation::RunCancellation;
use aisim_aiperf_workload_runner::participant_adapter::{NativeInputProjection, native_hosted};
use aisimulate_core::engine::{EngineConfig, TimingModelConfig};
use aisimulate_core::replay::{
    ReplayCaptureOptions, ReplayEngineConfig, ReplayEngineFactory, ReplayRuntimeInput, ReplaySpec,
    ReplayTopology, Replayer,
};
use serde_json::{Value, json};

#[path = "support/run_fixture.rs"]
mod run_fixture;

fn run(frame: &str) -> Value {
    let run = aiperf_protocol::runner_v2::decode_run_request_v2(frame.as_bytes())
        .unwrap()
        .run;
    let (participant, handle) = native_hosted(
        &run,
        NativeInputProjection::TraceHashes,
        RunCancellation::new(),
        Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        Rc::new(aiperf_rng::identity::DeterministicIdentitySource::new(
            [17; 32],
        )),
    )
    .expect("direct shared preparation must not encode or decode text");
    let spec: ReplaySpec = serde_json::from_value(json!({
        "version":1,"topology":ReplayTopology::aggregated(1),
        "engine": ReplayEngineConfig { rank: EngineConfig {
            timing_model: TimingModelConfig::Fixed { prefill_ms:1.0, decode_ms:1.0 },
            block_size:8, max_num_seqs:32, enable_prefix_caching:true,
            enable_chunked_prefill:false, async_scheduling:Some(false), ..Default::default()
        }, ..Default::default() }, "requests":[]
    }))
    .unwrap();
    let report = Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::External(Box::new(participant)))
        .with_capture_options(ReplayCaptureOptions {
            capture_per_request: true,
            ..Default::default()
        })
        .run()
        .unwrap();
    let hosted = handle.into_report().unwrap();
    assert_eq!(
        report.per_request.len(),
        hosted["requests"].as_array().unwrap().len()
    );
    hosted
}

#[test]
fn direct_hashes_keep_authored_lengths_branches_and_sticky_marker_without_text() {
    let directory = tempfile::tempdir().unwrap();
    let mut frame = aiperf_protocol::runner_v2::decode_run_request_v2(
        run_fixture::branch_request(directory.path()).as_bytes(),
    )
    .unwrap();
    let Some(aiperf_config::model::dataset::Dataset::File(dataset)) = frame
        .run
        .cfg
        .datasets
        .as_mut()
        .and_then(|rows| rows.first_mut())
    else {
        panic!("file fixture")
    };
    dataset.synthesis.as_mut().unwrap()["cache_bust_target"] = json!("first_turn_prefix");
    let hosted = run(&serde_json::to_string(&frame).unwrap());
    let rows = hosted["requests"].as_array().unwrap();
    assert_eq!(rows.len(), 4);
    let lengths: Vec<_> = rows
        .iter()
        .map(|row| row["request"]["input_length"].as_u64().unwrap())
        .collect();
    assert_eq!(lengths.iter().filter(|length| **length == 16).count(), 2);
    assert_eq!(lengths.iter().filter(|length| **length == 24).count(), 2);
    for row in rows {
        assert_eq!(row["terminal"], "completed");
        assert!(row["input_token_ids"].as_array().is_some());
        assert_eq!(
            row["input_token_ids"].as_array().unwrap().len() as u64,
            row["request"]["input_length"].as_u64().unwrap()
        );
    }
    let root = &rows[0]["request"]["root_correlation_id"];
    assert!(
        rows.iter()
            .any(|row| row["request"]["parent_correlation_id"] != Value::Null)
    );
    assert!(
        rows.iter()
            .all(|row| &row["request"]["root_correlation_id"] == root)
    );
}
