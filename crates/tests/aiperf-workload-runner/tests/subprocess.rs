// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Proof crosses the actual independently built executable, not a linked test helper.

use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Write},
    process::{Command, Stdio},
};

fn runner_binary() -> std::path::PathBuf {
    let path = match option_env!("CARGO_BIN_EXE_aisim-aiperf-workload-runner") {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            let test = std::env::current_exe().expect("resolve running integration test");
            let dependencies = test
                .parent()
                .expect("integration test has target deps directory");
            let target = dependencies
                .parent()
                .expect("deps directory has target profile directory");
            target.join(format!(
                "aisim-aiperf-workload-runner{}",
                std::env::consts::EXE_SUFFIX
            ))
        }
    };
    assert!(
        path.is_file(),
        "independently built runner binary is missing: {}",
        path.display()
    );
    path
}

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
        frame["run"]["cfg"].as_object_mut().unwrap().remove("trajectory_start_min_ratio");
        frame["run"]["cfg"].as_object_mut().unwrap().remove("trajectory_start_max_ratio");
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
fn direct_library_executes_finite_branching_native_run_without_stdio_or_children() {
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
    assert_eq!(report["proof_rendezvous_entered"], false);
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
fn branching_finite_trace_keeps_native_child_body_and_join_frontier() {
    let directory = tempfile::tempdir().unwrap();
    let mut child = Command::new(runner_binary())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(input, "{}", branch_request(directory.path())).unwrap();
    input.flush().unwrap();
    let output = child.stdout.take().unwrap();
    let events: Vec<Value> = BufReader::new(output)
        .lines()
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
        .collect();
    assert!(child.wait().unwrap().success());
    drop(input);
    assert_eq!(
        events.last().unwrap()["outcome"],
        "succeeded",
        "{events:#?}"
    );
    let report: Value = serde_json::from_slice(
        &std::fs::read(directory.path().join("aisim-library-proof.json")).unwrap(),
    )
    .unwrap();
    let requests = report["requests"].as_array().unwrap();
    assert_eq!(
        requests.len(),
        4,
        "finite branch must execute once, without recycling"
    );
    let children: Vec<&Value> = requests
        .iter()
        .filter(|row| !row["request"]["parent_correlation_id"].is_null())
        .collect();
    assert_eq!(
        children.len(),
        2,
        "actual native branch ancestry must survive the execution seam"
    );
    let parents: Vec<&Value> = requests
        .iter()
        .filter(|row| row["request"]["parent_correlation_id"].is_null())
        .collect();
    assert_eq!(parents.len(), 2);
    let child_end = children
        .iter()
        .map(|row| row["end_ns"].as_i64().unwrap())
        .max()
        .unwrap();
    let parent_join = parents
        .iter()
        .map(|row| row["start_ns"].as_i64().unwrap())
        .max()
        .unwrap();
    assert!(
        parent_join >= child_end,
        "parent join crossed an unsettled native child"
    );
    assert!(requests.iter().all(|row| {
        row["request"]["body"]["messages"]
            .as_array()
            .is_some_and(|messages| !messages.is_empty())
    }));
    assert_ne!(
        parents[0]["request"]["body"], children[0]["request"]["body"],
        "distinct hash populations must retain distinct corpus prompts"
    );
    assert_eq!(report["engine_drained"], true);
}

#[test]
fn capabilities_are_explicit_and_do_not_claim_full_product_support() {
    let output = Command::new(runner_binary())
        .arg("capabilities")
        .output()
        .unwrap();
    assert!(output.status.success());
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["per_token_ipc"], false);
    assert_eq!(value["artifacts"], "proof_report_not_product_aiperf_v2");
}

#[test]
fn malformed_complete_request_is_rejected_by_native_codec() {
    let directory = tempfile::tempdir().unwrap();
    let mut frame = request(directory.path());
    frame["unexpected"] = json!(true);
    let mut child = Command::new(runner_binary())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(child.stdin.take().unwrap(), "{frame}").unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
}

#[test]
fn finite_native_workload_writes_body_bearing_proof_artifact() {
    let directory = tempfile::tempdir().unwrap();
    let mut child = Command::new(runner_binary())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(input, "{}", request(directory.path())).unwrap();
    input.flush().unwrap();
    // Keep stdin open: native clean-EOF cancellation must not race the run.
    let output = child.stdout.take().unwrap();
    let events: Vec<Value> = BufReader::new(output)
        .lines()
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
        .collect();
    assert!(child.wait().unwrap().success());
    drop(input);
    let terminal = events.last().unwrap();
    assert_eq!(terminal["event"], "run_terminal");
    assert_eq!(terminal["outcome"], "succeeded", "{events:#?}");
    let report: Value = serde_json::from_slice(
        &std::fs::read(directory.path().join("aisim-library-proof.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(report["engine_drained"], true);
    assert_eq!(report["requests"].as_array().unwrap().len(), 2);
    assert!(report["requests"][0]["request"]["body"]["messages"].is_array());
    assert!(
        events
            .iter()
            .any(|event| event["event"] == "phase_progress")
    );
}

#[test]
fn authored_corpus_selection_changes_actual_native_request_bodies() {
    let mut bodies = Vec::new();
    for corpus in ["coding", "sonnet"] {
        let directory = tempfile::tempdir().unwrap();
        let mut frame = request(directory.path());
        frame["run"]["cfg"]["datasets"][0]["prompts"]["corpus"] = json!(corpus);
        let mut child = Command::new(runner_binary())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut input = child.stdin.take().unwrap();
        writeln!(input, "{frame}").unwrap();
        input.flush().unwrap();
        let events: Vec<Value> = BufReader::new(child.stdout.take().unwrap())
            .lines()
            .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
            .collect();
        assert!(child.wait().unwrap().success());
        drop(input);
        assert_eq!(
            events.last().unwrap()["outcome"],
            "succeeded",
            "corpus={corpus}: {events:#?}"
        );
        let report: Value = serde_json::from_slice(
            &std::fs::read(directory.path().join("aisim-library-proof.json")).unwrap(),
        )
        .unwrap();
        bodies.push(report["requests"][0]["request"]["body"].clone());
    }
    assert_ne!(
        bodies[0], bodies[1],
        "authored corpus must reach native prompt synthesis"
    );
}

#[test]
fn artifact_collision_still_emits_exactly_one_failed_terminal() {
    let directory = tempfile::tempdir().unwrap();
    let artifact = directory.path().join("aisim-library-proof.json");
    std::fs::write(&artifact, b"existing-owned-output").unwrap();
    let mut child = Command::new(runner_binary())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(input, "{}", request(directory.path())).unwrap();
    input.flush().unwrap();
    let output = child.stdout.take().unwrap();
    let events: Vec<Value> = BufReader::new(output)
        .lines()
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
        .collect();
    assert!(child.wait().unwrap().success());
    drop(input);
    assert_eq!(
        events
            .iter()
            .filter(|event| event["event"] == "run_terminal")
            .count(),
        1
    );
    assert_eq!(events.last().unwrap()["outcome"], "failed");
    assert_eq!(events.last().unwrap()["stage"], "reporting");
    assert_eq!(std::fs::read(artifact).unwrap(), b"existing-owned-output");
}

#[test]
fn malformed_control_at_virtual_wait_cannot_be_reported_as_cancelled() {
    let directory = tempfile::tempdir().unwrap();
    let mut child = Command::new(runner_binary())
        .arg("--proof-wait-for-cancel")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(input, "{}", request(directory.path())).unwrap();
    input.flush().unwrap();
    let mut sent = false;
    let mut events = Vec::new();
    for line in BufReader::new(child.stdout.take().unwrap()).lines() {
        let event: Value = serde_json::from_str(&line.unwrap()).unwrap();
        if event["metrics"]["proof_rendezvous_waiting"] == 1.0 {
            writeln!(input,"{{\"protocol_version\":2,\"command_id\":\"invalid\",\"command\":{{\"type\":\"pause\"}}}}").unwrap();
            input.flush().unwrap();
            sent = true;
        }
        events.push(event);
    }
    assert!(child.wait().unwrap().success());
    drop(input);
    assert!(sent);
    assert_eq!(events.last().unwrap()["outcome"], "failed");
    assert_eq!(events.last().unwrap()["stage"], "protocol");
    assert_eq!(
        events
            .iter()
            .filter(|event| event["event"] == "run_terminal")
            .count(),
        1
    );
}

#[test]
fn framed_cancel_interrupts_committed_model_virtual_wait() {
    let directory = tempfile::tempdir().unwrap();
    let mut child = Command::new(runner_binary())
        .arg("--proof-wait-for-cancel")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(input, "{}", request(directory.path())).unwrap();
    input.flush().unwrap();
    let output = child.stdout.take().unwrap();
    let mut sent_cancel = false;
    let mut events = Vec::new();
    for line in BufReader::new(output).lines() {
        let event: Value = serde_json::from_str(&line.unwrap()).unwrap();
        if event["metrics"]["proof_rendezvous_waiting"] == 1.0 {
            assert!(
                events
                    .iter()
                    .any(|event: &Value| event["event"] == "run_started")
            );
            assert!(
                event["metrics"]["proof_pending_model_deadline_ns"]
                    .as_f64()
                    .unwrap()
                    > event["observed_at_ns"].as_f64().unwrap()
            );
            writeln!(
                input,
                "{}",
                json!({"protocol_version":2,"command_id":"stop-proof","command":{"type":"cancel"}})
            )
            .unwrap();
            input.flush().unwrap();
            sent_cancel = true;
        }
        events.push(event);
    }
    assert!(child.wait().unwrap().success());
    drop(input);
    assert!(
        sent_cancel,
        "never observed a suspended committed model pass"
    );
    assert!(
        events
            .iter()
            .any(|event| event["event"] == "control_result" && event["command_id"] == "stop-proof")
    );
    assert_eq!(events.last().unwrap()["outcome"], "cancelled");
    let report: Value = serde_json::from_slice(
        &std::fs::read(directory.path().join("aisim-library-proof.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(report["engine_drained"], true);
    assert_eq!(report["proof_rendezvous_entered"], true);
    assert_eq!(
        report["model_submitted_count"], 1,
        "the suspended native turn was actually submitted"
    );
    assert_eq!(
        report["model_cancelled_count"], 1,
        "actual Engine ownership must be cancelled"
    );
    let model = report["model_trace"].as_array().unwrap();
    assert!(model.iter().any(|event| event["event"] == "pass_started"));
    assert!(
        !model.iter().any(|event| event["event"] == "observed"),
        "suspended model output must not cross the cancelled replay frontier"
    );
    assert!(
        report["requests"].as_array().unwrap().len() < 2,
        "cancel advanced past the replay frontier"
    );
}
