// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct in-process library controls for the shared AgentX workload.

use serde_json::{Value, json};

#[path = "support/run_fixture.rs"]
mod run_fixture;
use run_fixture::{branch_request, request};

#[test]
fn direct_library_executes_native_scheduled_synthetic_and_conversation_sources() {
    let datasets = [
        json!({
            "type":"synthetic", "entries":1, "sampling":"sequential",
            "turns":{"value":2}, "turn_delay_ratio":1.0,
            "prompts":{"corpus":"sonnet","batch_size":1,"random_corpus_style":"vllm","isl":{"value":16},"osl":{"value":2}}
        }),
        json!({
            "type":"file", "format":"single_turn", "sampling":"sequential", "options":{},
            "records":[{"text":"first scheduled prompt","output_length":2},
                       {"text":"second scheduled prompt","output_length":2}]
        }),
        json!({
            "type":"file", "format":"multi_turn", "sampling":"sequential", "options":{},
            "records":[{"session_id":"authored-session","turns":[
                {"text":"first conversation prompt","output_length":2},
                {"text":"second conversation prompt","output_length":2,"delay":3.0}
            ]}]
        }),
    ];
    for dataset in datasets {
        let directory = tempfile::tempdir().unwrap();
        let mut frame = request(directory.path());
        frame["run"]["cfg"]["datasets"] = json!([dataset]);
        frame["run"]["cfg"]["finite_replay"] = Value::Null;
        frame["run"]["cfg"]["trajectory_start_min_ratio"] = json!(0.0);
        frame["run"]["cfg"]["trajectory_start_max_ratio"] = json!(0.0);
        frame["run"]["cfg"]["phases"] = json!([{
            "type":"concurrency", "name":"profiling", "exclude_from_results":false,
            "seamless":false, "concurrency":1, "requests":2, "grace_period":1.0
        }]);
        let native = aiperf_protocol::runner_v2::decode_run_request_v2(
            &serde_json::to_vec(&frame).unwrap(),
        ).unwrap();
        let report = aisim_aiperf_workload_runner::execute_with_identity_seed(
            &native.run,
            aiperf_clock_runtime::cancellation::RunCancellation::new(),
            std::rc::Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
            [41; 32],
            None,
        ).expect("shared native scheduled lowering and actual model execution");
        let rows = report["requests"].as_array().unwrap();
        assert_eq!(rows.len(), 2, "one request cap governs shared scheduling");
        assert_eq!(report["model_submitted_count"], 2);
        assert_eq!(report["engine_drained"], true);
        for row in rows {
            assert_eq!(row["status"], "settled");
            assert_eq!(row["request"]["body"]["model"], "model");
            assert!(!row["wire"]["body_bytes"].as_array().unwrap().is_empty());
            assert!(!row["input_token_ids"].as_array().unwrap().is_empty());
            assert_eq!(row["output_token_ids"].as_array().unwrap().len(), 2);
        }
        if frame["run"]["cfg"]["datasets"][0]["format"] == "multi_turn" {
            assert_eq!(rows[0]["request"]["correlation_id"], rows[1]["request"]["correlation_id"]);
            assert_eq!(rows[1]["request"]["turn_index"], 1);
            assert!(rows[1]["issue_ns"].as_i64().unwrap() >= rows[0]["end_ns"].as_i64().unwrap());
            assert!(rows[1]["request"]["body"]["messages"].as_array().unwrap().len() > 1);
        }
    }
}

#[test]
fn direct_library_executes_finite_branching_native_run_in_process() {
    let directory = tempfile::tempdir().unwrap();
    let frame = branch_request(directory.path());
    let request = aiperf_protocol::runner_v2::decode_run_request_v2(frame.as_bytes()).unwrap();
    let observer = std::rc::Rc::new(aiperf_timing::RecordingPhaseObserver::default());
    let report = aisim_aiperf_workload_runner::execute(
        &request.run,
        aiperf_clock_runtime::cancellation::RunCancellation::new(),
        observer.clone(),
    )
    .expect("direct simulator-owned library composition");
    let requests = report["requests"].as_array().unwrap();
    assert_eq!(requests.len(), 4, "finite native branch executes once");
    let children: Vec<&Value> = requests
        .iter()
        .filter(|row| !row["request"]["parent_correlation_id"].is_null())
        .collect();
    let parents: Vec<&Value> = requests
        .iter()
        .filter(|row| row["request"]["parent_correlation_id"].is_null())
        .collect();
    assert_eq!(children.len(), 2);
    assert_eq!(parents.len(), 2);
    let child_end = children
        .iter()
        .map(|row| row["end_ns"].as_i64().unwrap())
        .max()
        .unwrap();
    let join_start = parents
        .iter()
        .map(|row| row["start_ns"].as_i64().unwrap())
        .max()
        .unwrap();
    assert!(
        join_start >= child_end,
        "shared native join cannot cross child settlement"
    );
    assert_ne!(
        parents[0]["request"]["body"],
        children[0]["request"]["body"]
    );
    assert_eq!(report["engine_drained"], true);
    assert!(
        !observer.events().is_empty(),
        "native phase lifecycle must be observed directly"
    );
    assert!(
        !directory.path().join("aisim-library-proof.json").exists(),
        "direct composition leaves publication to the embedding simulator"
    );
}

#[test]
fn authored_corpus_selection_changes_actual_native_request_bodies() {
    let mut bodies = Vec::new();
    for corpus in ["coding", "sonnet"] {
        let directory = tempfile::tempdir().unwrap();
        let mut frame = request(directory.path());
        frame["run"]["cfg"]["datasets"][0]["prompts"]["corpus"] = json!(corpus);
        let native = aiperf_protocol::runner_v2::decode_run_request_v2(
            &serde_json::to_vec(&frame).unwrap(),
        )
        .unwrap();
        let report = aisim_aiperf_workload_runner::execute(
            &native.run,
            aiperf_clock_runtime::cancellation::RunCancellation::new(),
            std::rc::Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        )
        .unwrap_or_else(|error| panic!("corpus={corpus}: {error:#}"));
        bodies.push(report["requests"][0]["request"]["body"].clone());
    }
    assert_ne!(
        bodies[0], bodies[1],
        "authored corpus must reach native prompt synthesis"
    );
}
