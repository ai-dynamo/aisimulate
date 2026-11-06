// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Proof crosses the actual independently built executable, not a linked test helper.

use std::{io::{BufRead,BufReader,Write},process::{Command,Stdio}};
use serde_json::{Value,json};

fn request(directory: &std::path::Path) -> Value {
    let mut cfg = aiperf_config::model::BenchmarkConfig::default();
    cfg.datasets = Some(vec![serde_json::from_value(json!({"type":"file","format":"weka_trace","sampling":"sequential","prompts":{"corpus":"coding"},"synthesis":{"speedup_ratio":1.0,"prefix_len_multiplier":1.0,"prefix_root_multiplier":1,"prompt_len_multiplier":1.0,"output_len_multiplier":1.0,"cache_bust_target":"none"},"records":{"id":"finite-root","models":["model"],"block_size":8,"hash_id_scope":"local","requests":[{"t":0.0,"type":"n","model":"model","in":16,"out":2,"hash_ids":[1,2],"api_time":0.001},{"t":0.01,"type":"n","model":"model","in":24,"out":2,"hash_ids":[1,2,3],"api_time":0.001}]}})).unwrap()]);
    cfg.models = Some(serde_json::from_value(json!({"strategy":"round_robin","items":[{"name":"model"}]})).unwrap());
    cfg.tokenizer = Some(serde_json::from_value(json!({"name":"builtin"})).unwrap());
    cfg.phases = Some(serde_json::from_value(json!([{"type":"concurrency","name":"profiling","exclude_from_results":false,"seamless":false,"concurrency":1,"requests":2,"grace_period":1.0}])).unwrap());
    cfg.trajectory_start_min_ratio = Some(0.0);
    cfg.trajectory_start_max_ratio = Some(0.0);
    let run = aiperf_config::model::BenchmarkRun { benchmark_id:"independent-proof".into(),artifact_dir:directory.into(),cfg,cli_command:None,label:String::new(),random_seed:Some(42),planned_replay_traces:Default::default(),authored_corpus_bounds:None,sweep_id:None,trial:0,variation:None,variables:Default::default() };
    serde_json::to_value(aiperf_protocol::runner_v2::RunRequestV2 { protocol_version:2,run }).unwrap()
}

#[test]
fn capabilities_are_explicit_and_do_not_claim_full_product_support() {
    let output = Command::new(env!("CARGO_BIN_EXE_aisim-aiperf-workload-runner")).arg("capabilities").output().unwrap();
    assert!(output.status.success());
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["per_token_ipc"],false);
    assert_eq!(value["artifacts"],"proof_report_not_product_aiperf_v2");
}

#[test]
fn malformed_complete_request_is_rejected_by_native_codec() {
    let directory = tempfile::tempdir().unwrap();
    let mut frame = request(directory.path());
    frame["unexpected"] = json!(true);
    let mut child = Command::new(env!("CARGO_BIN_EXE_aisim-aiperf-workload-runner")).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    writeln!(child.stdin.take().unwrap(),"{frame}").unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
}

#[test]
fn finite_native_workload_writes_body_bearing_proof_artifact() {
    let directory = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_aisim-aiperf-workload-runner")).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(input,"{}",request(directory.path())).unwrap();
    input.flush().unwrap();
    // Keep stdin open: native clean-EOF cancellation must not race the run.
    let output = child.stdout.take().unwrap();
    let events: Vec<Value> = BufReader::new(output).lines().map(|line|serde_json::from_str(&line.unwrap()).unwrap()).collect();
    assert!(child.wait().unwrap().success());
    drop(input);
    let terminal = events.last().unwrap();
    assert_eq!(terminal["event"],"run_terminal");
    assert_eq!(terminal["outcome"],"succeeded","{events:#?}");
    let report: Value = serde_json::from_slice(&std::fs::read(directory.path().join("aisim-library-proof.json")).unwrap()).unwrap();
    assert_eq!(report["engine_drained"],true);
    assert_eq!(report["requests"].as_array().unwrap().len(),2);
    assert!(report["requests"][0]["request"]["body"]["messages"].is_array());
    assert!(events.iter().any(|event|event["event"] == "phase_progress"));
}

#[test]
fn framed_cancel_uses_native_control_result() {
    let directory = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_aisim-aiperf-workload-runner")).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(input,"{}",request(directory.path())).unwrap();
    writeln!(input,"{}",json!({"protocol_version":2,"command_id":"stop-proof","command":{"type":"cancel"}})).unwrap();
    input.flush().unwrap();
    let output = child.stdout.take().unwrap();
    let events: Vec<Value> = BufReader::new(output).lines().map(|line|serde_json::from_str(&line.unwrap()).unwrap()).collect();
    assert!(child.wait().unwrap().success());
    drop(input);
    assert!(events.iter().any(|event|event["event"] == "control_result" && event["command_id"] == "stop-proof"));
    assert_eq!(events.last().unwrap()["outcome"],"cancelled");
}
