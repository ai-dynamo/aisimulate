// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Complete native run inputs shared by direct and stdio proofs.

use serde_json::{Value, json};

pub fn request(directory: &std::path::Path) -> Value {
    // Every nonnullable direct-wire section uses its owning native type.
    let cfg = aiperf_config::model::BenchmarkConfig {
        endpoint: Some(serde_json::from_value(json!({
            "urls":["http://127.0.0.1:1"],"url_strategy":"round_robin","type":"chat","streaming":true,
            "allow_empty_content":false,"use_legacy_max_tokens":false,"use_server_token_count":false,
            "per_chunk_usage":false,"timeout_seconds":30.0,"connection_reuse":"pooled","ssl_verify":true,
            "connection_limit":64,"keepalive_timeout":15.0,"download_video_content":false,
            "polling_interval_seconds":1.0,"extra":{},"headers":{},"http2":false,
            "wait_for_model_timeout":0.0,"wait_for_model_interval":5.0,"wait_for_model_mode":"inference","proxy_from_env":false
        })).expect("complete endpoint policy")),
        transport: Some(aiperf_config::model::transport::Transport::Http),
        runtime: Some(Default::default()),
        metrics: Some(Default::default()),
        artifacts: Some(aiperf_config::model::artifacts::Artifacts {
            inputs_path: "inputs.json".into(),
            ..Default::default()
        }),
        gpu_telemetry: Some(Default::default()),
        server_metrics: Some(Default::default()),
        network_latency: Some(Default::default()),
        export: Some(serde_json::from_value(json!({
            "visibility":{"dev_mode":false,"show_internal_metrics":false,"show_experimental_metrics":false},
            "genai_perf":{"enabled":false,"json":false,"csv":false},
            "console_txt":{"enabled":false,"width":140,"title":"AISim proof","metrics":{},"model_names":[],"usage_diff_threshold_pct":10.0,"osl_mismatch_threshold_pct":5.0,"osl_mismatch_max_tokens":50.0},
            "server_metrics":{"json":false,"csv":false},"plugin_exporters":[]
        })).expect("complete disabled export policy")),
        datasets: Some(vec![serde_json::from_value(json!({"type":"file","format":"weka_trace","options":{},"sampling":"sequential","prompts":{"corpus":"coding"},"synthesis":{"speedup_ratio":1.0,"prefix_len_multiplier":1.0,"prefix_root_multiplier":1,"prompt_len_multiplier":1.0,"output_len_multiplier":1.0,"cache_bust_target":"none"},"records":{"id":"finite-root","models":["model"],"block_size":8,"hash_id_scope":"local","requests":[{"t":0.0,"type":"n","model":"model","in":16,"out":2,"hash_ids":[1,2],"api_time":0.001},{"t":0.01,"type":"n","model":"model","in":24,"out":2,"hash_ids":[1,2,3],"api_time":0.001}]}})).unwrap()]),
        models: Some(aiperf_config::model::models::Models {
            strategy: aiperf_config::model::models::ModelStrategy::RoundRobin,
            items: vec![aiperf_config::model::models::ModelItem {
                name: "model".into(),
                ..Default::default()
            }],
        }),
        tokenizer: Some(aiperf_config::model::tokenizer::Tokenizer {
            name: Some("builtin".into()),
            revision: "main".into(),
            trust_remote_code: false,
            apply_chat_template: false,
            server_url: None,
            per_model: Default::default(),
        }),
        phases: Some(serde_json::from_value(json!([{"type":"concurrency","name":"profiling","exclude_from_results":false,"seamless":false,"concurrency":1,"grace_period":1.0}])).unwrap()),
        finite_replay: Some(true),
        trajectory_start_min_ratio: Some(0.0),
        trajectory_start_max_ratio: Some(0.0),
        ..Default::default()
    };
    let run = aiperf_config::model::BenchmarkRun {
        benchmark_id: "independent-proof".into(),
        artifact_dir: directory.into(),
        cfg,
        cli_command: None,
        label: String::new(),
        random_seed: Some(42),
        planned_replay_traces: Default::default(),
        authored_corpus_bounds: None,
        sweep_id: None,
        trial: 0,
        variation: None,
        variables: Default::default(),
    };
    let frame = aiperf_protocol::runner_v2::RunRequestV2 {
        protocol_version: 2,
        run,
    };
    let encoded = serde_json::to_vec(&frame).expect("encode complete native request fixture");
    let decoded = aiperf_protocol::runner_v2::decode_run_request_v2(&encoded)
        .expect("fixture must pass native strict codec before spawning a child");
    aiperf_engine_input::execution_input::factory_workload_input_resolved(&decoded.run)
        .expect("fixture must pass native workload projection before spawning a child");
    serde_json::from_slice(&encoded).unwrap()
}

pub fn branch_request(directory: &std::path::Path) -> String {
    let initial = request(directory);
    let bytes = serde_json::to_vec(&initial).unwrap();
    let mut native = aiperf_protocol::runner_v2::decode_run_request_v2(&bytes).unwrap();
    // Finite native replay ends on selected-tree exhaustion, never on an
    // authored request/session/duration cutoff. Reuse the complete base phase.
    let Some(aiperf_config::model::dataset::Dataset::File(dataset)) = native
        .run
        .cfg
        .datasets
        .as_mut()
        .and_then(|rows| rows.first_mut())
    else {
        panic!("file dataset required");
    };
    // Wide signed hashes are exact raw source, not serde_json::Value numbers.
    let raw = r#"{"id":"branch-root","models":["model"],"block_size":8,"hash_id_scope":"local","requests":[{"t":0.0,"type":"n","model":"model","in":16,"out":2,"hash_ids":[184467440737095516170,184467440737095516171],"api_time":0.001},{"t":0.002,"type":"subagent","agent_id":"child-proof","subagent_type":"Explore","duration_ms":5,"status":"completed","models":["model"],"requests":[{"t":0.002,"type":"n","model":"model","in":16,"out":2,"hash_ids":[-184467440737095516170,-184467440737095516171],"api_time":0.001},{"t":0.004,"type":"n","model":"model","in":24,"out":2,"hash_ids":[-184467440737095516170,-184467440737095516171,-184467440737095516172],"api_time":0.001}]},{"t":0.01,"type":"n","model":"model","in":24,"out":2,"hash_ids":[184467440737095516170,184467440737095516171,184467440737095516172],"api_time":0.001}]}"#;
    dataset.records = Some(serde_json::value::RawValue::from_string(raw.into()).unwrap());
    serde_json::to_string(&native).unwrap()
}
