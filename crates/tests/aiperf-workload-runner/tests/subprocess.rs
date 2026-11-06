// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Proof crosses the actual independently built executable, not a linked test helper.

use std::{io::{BufRead,BufReader,Write},process::{Command,Stdio}};
use serde_json::{Value,json};

fn runner_binary() -> std::path::PathBuf {
    let path = match option_env!("CARGO_BIN_EXE_aisim-aiperf-workload-runner") {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            let test = std::env::current_exe().expect("resolve running integration test");
            let dependencies = test.parent().expect("integration test has target deps directory");
            let target = dependencies.parent().expect("deps directory has target profile directory");
            target.join(format!("aisim-aiperf-workload-runner{}",std::env::consts::EXE_SUFFIX))
        }
    };
    assert!(path.is_file(),"independently built runner binary is missing: {}",path.display());
    path
}

fn request(directory: &std::path::Path) -> Value {
    let mut cfg = aiperf_config::model::BenchmarkConfig::default();
    // Populate EVERY nonnullable required section from the native direct-wire
    // inventory. Defaults remain owned by each public config type.
    cfg.endpoint = Some(serde_json::from_value(json!({
        "urls":["http://127.0.0.1:1"],"url_strategy":"round_robin","type":"chat","streaming":true,
        "allow_empty_content":false,"use_legacy_max_tokens":false,"use_server_token_count":false,
        "per_chunk_usage":false,"timeout_seconds":30.0,"connection_reuse":"pooled","ssl_verify":true,
        "connection_limit":64,"keepalive_timeout":15.0,"download_video_content":false,
        "polling_interval_seconds":1.0,"extra":{},"headers":{},"http2":false,
        "wait_for_model_timeout":0.0,"wait_for_model_interval":1.0,"wait_for_model_mode":"models","proxy_from_env":false
    })).expect("complete endpoint policy"));
    cfg.transport = Some(aiperf_config::model::transport::Transport::Http);
    cfg.runtime = Some(Default::default());
    cfg.metrics = Some(Default::default());
    cfg.artifacts = Some(aiperf_config::model::artifacts::Artifacts {inputs_path:"inputs.json".into(),..Default::default()});
    cfg.gpu_telemetry = Some(Default::default());
    cfg.server_metrics = Some(Default::default());
    cfg.network_latency = Some(Default::default());
    cfg.export = Some(serde_json::from_value(json!({
        "visibility":{"dev_mode":false,"show_internal_metrics":false,"show_experimental_metrics":false},
        "genai_perf":{"enabled":false,"json":false,"csv":false},
        "console_txt":{"enabled":false,"width":140,"title":"AISim proof","metrics":{},"model_names":[],"usage_diff_threshold_pct":10.0,"osl_mismatch_threshold_pct":5.0,"osl_mismatch_max_tokens":50.0},
        "server_metrics":{"json":false,"csv":false},"plugin_exporters":[]
    })).expect("complete disabled export policy"));
    cfg.datasets = Some(vec![serde_json::from_value(json!({"type":"file","format":"weka_trace","options":{},"sampling":"sequential","prompts":{"corpus":"coding"},"synthesis":{"speedup_ratio":1.0,"prefix_len_multiplier":1.0,"prefix_root_multiplier":1,"prompt_len_multiplier":1.0,"output_len_multiplier":1.0,"cache_bust_target":"none"},"records":{"id":"finite-root","models":["model"],"block_size":8,"hash_id_scope":"local","requests":[{"t":0.0,"type":"n","model":"model","in":16,"out":2,"hash_ids":[1,2],"api_time":0.001},{"t":0.01,"type":"n","model":"model","in":24,"out":2,"hash_ids":[1,2,3],"api_time":0.001}]}})).unwrap()]);
    cfg.models = Some(aiperf_config::model::models::Models {
        strategy:aiperf_config::model::models::ModelStrategy::RoundRobin,
        items:vec![aiperf_config::model::models::ModelItem { name:"model".into(),..Default::default() }],
    });
    cfg.tokenizer = Some(aiperf_config::model::tokenizer::Tokenizer {
        name:Some("builtin".into()),revision:"main".into(),trust_remote_code:false,
        apply_chat_template:false,server_url:None,per_model:Default::default(),
    });
    cfg.phases = Some(serde_json::from_value(json!([{"type":"concurrency","name":"profiling","exclude_from_results":false,"seamless":false,"concurrency":1,"grace_period":1.0}])).unwrap());
    cfg.finite_replay = Some(true);
    cfg.trajectory_start_min_ratio = Some(0.0);
    cfg.trajectory_start_max_ratio = Some(0.0);
    let run = aiperf_config::model::BenchmarkRun { benchmark_id:"independent-proof".into(),artifact_dir:directory.into(),cfg,cli_command:None,label:String::new(),random_seed:Some(42),planned_replay_traces:Default::default(),authored_corpus_bounds:None,sweep_id:None,trial:0,variation:None,variables:Default::default() };
    let frame = aiperf_protocol::runner_v2::RunRequestV2 { protocol_version:2,run };
    let encoded = serde_json::to_vec(&frame).expect("encode complete native request fixture");
    let decoded = aiperf_protocol::runner_v2::decode_run_request_v2(&encoded).expect("fixture must pass native strict codec before spawning a child");
    aiperf_engine_input::execution_input::factory_workload_input_resolved(&decoded.run).expect("fixture must pass native workload projection before spawning a child");
    serde_json::from_slice(&encoded).unwrap()
}

fn branch_request(directory:&std::path::Path) -> String {
    let initial = request(directory);
    let bytes = serde_json::to_vec(&initial).unwrap();
    let mut native = aiperf_protocol::runner_v2::decode_run_request_v2(&bytes).unwrap();
    // Finite native replay ends on selected-tree exhaustion, never on an
    // authored request/session/duration cutoff. Reuse the complete base phase.
    let Some(aiperf_config::model::dataset::Dataset::File(dataset)) = native.run.cfg.datasets.as_mut().and_then(|rows|rows.first_mut()) else { panic!("file dataset required"); };
    // Wide signed hashes are exact raw source, not serde_json::Value numbers.
    let raw = r#"{"id":"branch-root","models":["model"],"block_size":8,"hash_id_scope":"local","requests":[{"t":0.0,"type":"n","model":"model","in":16,"out":2,"hash_ids":[184467440737095516170,184467440737095516171],"api_time":0.001},{"t":0.002,"type":"subagent","agent_id":"child-proof","subagent_type":"Explore","duration_ms":5,"status":"completed","models":["model"],"requests":[{"t":0.002,"type":"n","model":"model","in":16,"out":2,"hash_ids":[-184467440737095516170,-184467440737095516171],"api_time":0.001},{"t":0.004,"type":"n","model":"model","in":24,"out":2,"hash_ids":[-184467440737095516170,-184467440737095516171,-184467440737095516172],"api_time":0.001}]},{"t":0.01,"type":"n","model":"model","in":24,"out":2,"hash_ids":[184467440737095516170,184467440737095516171,184467440737095516172],"api_time":0.001}]}"#;
    dataset.records = Some(serde_json::value::RawValue::from_string(raw.into()).unwrap());
    serde_json::to_string(&native).unwrap()
}

#[test]
fn direct_library_executes_finite_branching_native_run_without_stdio_or_children() {
    let directory = tempfile::tempdir().unwrap();
    let frame = branch_request(directory.path());
    let request = aiperf_protocol::runner_v2::decode_run_request_v2(frame.as_bytes()).unwrap();
    let observer = std::rc::Rc::new(aiperf_timing::RecordingPhaseObserver::default());
    let report = aisim_aiperf_workload_runner::execute(
        &request.run,aiperf_clock_runtime::cancellation::RunCancellation::new(),observer.clone(),
    ).expect("direct simulator-owned library composition");
    let requests = report["requests"].as_array().unwrap();
    assert_eq!(requests.len(),4,"finite native branch executes once");
    let children:Vec<&Value> = requests.iter().filter(|row|!row["request"]["parent_correlation_id"].is_null()).collect();
    let parents:Vec<&Value> = requests.iter().filter(|row|row["request"]["parent_correlation_id"].is_null()).collect();
    assert_eq!(children.len(),2);
    assert_eq!(parents.len(),2);
    let child_end = children.iter().map(|row|row["end_ns"].as_i64().unwrap()).max().unwrap();
    let join_start = parents.iter().map(|row|row["start_ns"].as_i64().unwrap()).max().unwrap();
    assert!(join_start >= child_end,"shared native join cannot cross child settlement");
    assert_ne!(parents[0]["request"]["body"],children[0]["request"]["body"]);
    assert_eq!(report["engine_drained"],true);
    assert_eq!(report["proof_rendezvous_entered"],false);
    assert!(!observer.events().is_empty(),"native phase lifecycle must be observed directly");
    assert!(!directory.path().join("aisim-library-proof.json").exists(),"direct composition leaves publication to the embedding simulator");
}

#[test]
fn branching_finite_trace_keeps_native_child_body_and_join_frontier() {
    let directory = tempfile::tempdir().unwrap();
    let mut child = Command::new(runner_binary()).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(input,"{}",branch_request(directory.path())).unwrap();
    input.flush().unwrap();
    let output = child.stdout.take().unwrap();
    let events:Vec<Value> = BufReader::new(output).lines().map(|line|serde_json::from_str(&line.unwrap()).unwrap()).collect();
    assert!(child.wait().unwrap().success());
    drop(input);
    assert_eq!(events.last().unwrap()["outcome"],"succeeded","{events:#?}");
    let report:Value = serde_json::from_slice(&std::fs::read(directory.path().join("aisim-library-proof.json")).unwrap()).unwrap();
    let requests = report["requests"].as_array().unwrap();
    assert_eq!(requests.len(),4,"finite branch must execute once, without recycling");
    let children:Vec<&Value> = requests.iter().filter(|row|!row["request"]["parent_correlation_id"].is_null()).collect();
    assert_eq!(children.len(),2,"actual native branch ancestry must survive the execution seam");
    let parents:Vec<&Value> = requests.iter().filter(|row|row["request"]["parent_correlation_id"].is_null()).collect();
    assert_eq!(parents.len(),2);
    let child_end = children.iter().map(|row|row["end_ns"].as_i64().unwrap()).max().unwrap();
    let parent_join = parents.iter().map(|row|row["start_ns"].as_i64().unwrap()).max().unwrap();
    assert!(parent_join >= child_end,"parent join crossed an unsettled native child");
    assert!(requests.iter().all(|row|row["request"]["body"]["messages"].as_array().is_some_and(|messages|!messages.is_empty())));
    assert_ne!(parents[0]["request"]["body"],children[0]["request"]["body"],"distinct hash populations must retain distinct corpus prompts");
    assert_eq!(report["engine_drained"],true);
}

#[test]
fn capabilities_are_explicit_and_do_not_claim_full_product_support() {
    let output = Command::new(runner_binary()).arg("capabilities").output().unwrap();
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
    let mut child = Command::new(runner_binary()).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    writeln!(child.stdin.take().unwrap(),"{frame}").unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
}

#[test]
fn finite_native_workload_writes_body_bearing_proof_artifact() {
    let directory = tempfile::tempdir().unwrap();
    let mut child = Command::new(runner_binary()).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
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
fn authored_corpus_selection_changes_actual_native_request_bodies() {
    let mut bodies = Vec::new();
    for corpus in ["coding","sonnet"] {
        let directory = tempfile::tempdir().unwrap();
        let mut frame = request(directory.path());
        frame["run"]["cfg"]["datasets"][0]["prompts"]["corpus"] = json!(corpus);
        let mut child = Command::new(runner_binary()).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        let mut input = child.stdin.take().unwrap();
        writeln!(input,"{frame}").unwrap();
        input.flush().unwrap();
        let events:Vec<Value> = BufReader::new(child.stdout.take().unwrap()).lines().map(|line|serde_json::from_str(&line.unwrap()).unwrap()).collect();
        assert!(child.wait().unwrap().success());
        drop(input);
        assert_eq!(events.last().unwrap()["outcome"],"succeeded","corpus={corpus}: {events:#?}");
        let report:Value = serde_json::from_slice(&std::fs::read(directory.path().join("aisim-library-proof.json")).unwrap()).unwrap();
        bodies.push(report["requests"][0]["request"]["body"].clone());
    }
    assert_ne!(bodies[0],bodies[1],"authored corpus must reach native prompt synthesis");
}

#[test]
fn artifact_collision_still_emits_exactly_one_failed_terminal() {
    let directory = tempfile::tempdir().unwrap();
    let artifact = directory.path().join("aisim-library-proof.json");
    std::fs::write(&artifact,b"existing-owned-output").unwrap();
    let mut child = Command::new(runner_binary()).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(input,"{}",request(directory.path())).unwrap();
    input.flush().unwrap();
    let output = child.stdout.take().unwrap();
    let events:Vec<Value> = BufReader::new(output).lines().map(|line|serde_json::from_str(&line.unwrap()).unwrap()).collect();
    assert!(child.wait().unwrap().success());
    drop(input);
    assert_eq!(events.iter().filter(|event|event["event"] == "run_terminal").count(),1);
    assert_eq!(events.last().unwrap()["outcome"],"failed");
    assert_eq!(events.last().unwrap()["stage"],"reporting");
    assert_eq!(std::fs::read(artifact).unwrap(),b"existing-owned-output");
}

#[test]
fn malformed_control_at_virtual_wait_cannot_be_reported_as_cancelled() {
    let directory = tempfile::tempdir().unwrap();
    let mut child = Command::new(runner_binary()).arg("--proof-wait-for-cancel").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(input,"{}",request(directory.path())).unwrap();
    input.flush().unwrap();
    let mut sent = false;
    let mut events = Vec::new();
    for line in BufReader::new(child.stdout.take().unwrap()).lines() {
        let event:Value = serde_json::from_str(&line.unwrap()).unwrap();
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
    assert_eq!(events.last().unwrap()["outcome"],"failed");
    assert_eq!(events.last().unwrap()["stage"],"protocol");
    assert_eq!(events.iter().filter(|event|event["event"] == "run_terminal").count(),1);
}

#[test]
fn framed_cancel_interrupts_committed_model_virtual_wait() {
    let directory = tempfile::tempdir().unwrap();
    let mut child = Command::new(runner_binary()).arg("--proof-wait-for-cancel").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(input,"{}",request(directory.path())).unwrap();
    input.flush().unwrap();
    let output = child.stdout.take().unwrap();
    let mut sent_cancel = false;
    let mut events = Vec::new();
    for line in BufReader::new(output).lines() {
        let event: Value = serde_json::from_str(&line.unwrap()).unwrap();
        if event["metrics"]["proof_rendezvous_waiting"] == 1.0 {
            assert!(events.iter().any(|event:&Value|event["event"] == "run_started"));
            assert!(event["metrics"]["proof_pending_model_deadline_ns"].as_f64().unwrap() > event["observed_at_ns"].as_f64().unwrap());
            writeln!(input,"{}",json!({"protocol_version":2,"command_id":"stop-proof","command":{"type":"cancel"}})).unwrap();
            input.flush().unwrap();
            sent_cancel = true;
        }
        events.push(event);
    }
    assert!(child.wait().unwrap().success());
    drop(input);
    assert!(sent_cancel,"never observed a suspended committed model pass");
    assert!(events.iter().any(|event|event["event"] == "control_result" && event["command_id"] == "stop-proof"));
    assert_eq!(events.last().unwrap()["outcome"],"cancelled");
    let report: Value = serde_json::from_slice(&std::fs::read(directory.path().join("aisim-library-proof.json")).unwrap()).unwrap();
    assert_eq!(report["engine_drained"],true);
    assert_eq!(report["proof_rendezvous_entered"],true);
    assert!(report["requests"].as_array().unwrap().len() < 2,"cancel advanced past the replay frontier");
}
