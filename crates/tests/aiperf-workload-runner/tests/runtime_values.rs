// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw comparison evidence, not a numerical-equivalence assertion.

use aisimulate_core::engine::{EngineConfig, EngineFactory, TimingModelConfig};
use serde_json::{Value, json};
use std::{path::PathBuf, rc::Rc};

#[path = "support/run_fixture.rs"]
mod run_fixture;

fn identity_seed_from_hex(value: &str) -> [u8; 32] {
    assert_eq!(value.len(), 64, "identity seed must contain exactly 64 hexadecimal digits");
    assert!(value.bytes().all(|byte| byte.is_ascii_hexdigit()), "identity seed must be hexadecimal");
    std::array::from_fn(|index| u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).unwrap())
}

#[test]
fn capture_finite_branch_runtime_values() {
    let temporary = tempfile::tempdir().unwrap();
    let output_root = std::env::var_os("AISIM_RUNTIME_VALUES_OUTPUT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| temporary.path().to_owned());
    std::fs::create_dir_all(&output_root).unwrap();
    // A supplied native frame is consumed verbatim; do not rewrite its artifact
    // path, seed, dataset, model, phase, or other original configuration.
    let raw = match std::env::var_os("AISIM_RUNTIME_VALUES_RUN_REQUEST") {
        Some(path) => std::fs::read_to_string(path).unwrap(),
        None => run_fixture::branch_request(&output_root),
    };
    let request = aiperf_protocol::runner_v2::decode_run_request_v2(raw.as_bytes()).unwrap();
    let observer = Rc::new(aiperf_timing::RecordingPhaseObserver::default());
    // Only this capture harness reads the environment. The library receives an
    // explicit run-local owner, never ambient process identity configuration.
    let identity_seed_hex = std::env::var_os("AISIM_RUNTIME_VALUES_IDENTITY_SEED")
        .map(|value| value.into_string().expect("identity seed must be UTF-8 hexadecimal"));
    let identity_seed = identity_seed_hex.as_deref().map(identity_seed_from_hex);
    // Fixed timing overrides are opt-in to this evidence target only. Ordinary
    // library and executable defaults remain exactly as before.
    let override_config = std::env::var_os("AISIM_RUNTIME_VALUES_ENGINE_CONFIG")
        .map(|path| serde_json::from_slice::<EngineConfig>(&std::fs::read(path).unwrap()).unwrap());
    let (report, model_manifest) = match override_config {
        Some(config) => {
            let capacity = std::env::var("AISIM_RUNTIME_VALUES_ADMISSION_CAPACITY")
                .ok()
                .map(|value| value.parse::<usize>().unwrap())
                .unwrap_or(config.max_num_seqs);
            let manifest = json!({"engine_config":config,"admission_capacity":capacity,"selection":"explicit_test_override"});
            let factory = EngineFactory::new(config).unwrap();
            let report = match identity_seed {
                Some(seed) => aisim_aiperf_workload_runner::execute_with_identity_seed(
                    &request.run,
                    aiperf_clock_runtime::cancellation::RunCancellation::new(),
                    observer.clone(),
                    seed,
                    Some((&factory, capacity)),
                ),
                None => aisim_aiperf_workload_runner::execute_with_engine_factory(
                    &request.run,
                    aiperf_clock_runtime::cancellation::RunCancellation::new(),
                    observer.clone(),
                    &factory,
                    capacity,
                ),
            }.unwrap();
            (report, manifest)
        }
        None => {
            let config = EngineConfig {
                timing_model: TimingModelConfig::Fixed { prefill_ms: 1.0, decode_ms: 1.0 },
                max_num_seqs: 64,
                enable_prefix_caching: false,
                enable_chunked_prefill: false,
                async_scheduling: Some(false),
                ..Default::default()
            };
            let report = match identity_seed {
                Some(seed) => aisim_aiperf_workload_runner::execute_with_identity_seed(
                    &request.run,
                    aiperf_clock_runtime::cancellation::RunCancellation::new(),
                    observer.clone(),
                    seed,
                    None,
                ),
                None => aisim_aiperf_workload_runner::execute(
                    &request.run,
                    aiperf_clock_runtime::cancellation::RunCancellation::new(),
                    observer.clone(),
                ),
            }.unwrap();
            (report, json!({"engine_config":config,"admission_capacity":64,"selection":"unchanged_library_default"}))
        }
    };
    let manifest = json!({
        "format":"aisim-runtime-values-manifest-v1",
        "aisim_source":env!("AISIM_SOURCE_REVISION"),
        "aiperf_source":env!("AIPERF_SOURCE_REVISION"),
        "model":model_manifest,
        "identity":{"selection":if identity_seed.is_some() {"explicit_run_local_seed"} else {"os_random_v4"},"seed_hex":identity_seed_hex},

        "phase_observer_events":observer.events(),
        "claim":"raw captured runtime values; numerical equivalence not asserted",
    });
    // Publish through the same format owner as the original offline writer.
    // The byte comparator reads this file directly, without reserialization.
    let backend_report_bytes = aiperf_report_io::report::serialize_sorted_python_report_json(
        &report["backend_report"],
    ).unwrap();
    assert_ne!(report["backend_report"], Value::Null);
    std::fs::write(output_root.join("backend-report.json"), backend_report_bytes).unwrap();
    let mut per_request_bytes = Vec::new();
    for record in report["backend_per_request"].as_array().expect("backend request population") {
        serde_json::to_writer(&mut per_request_bytes, record).unwrap();
        per_request_bytes.push(b'\n');
    }
    std::fs::write(output_root.join("backend-per-request.jsonl"), per_request_bytes).unwrap();
    std::fs::write(output_root.join("run-request.json"), raw.as_bytes()).unwrap();
    std::fs::write(output_root.join("model-manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    std::fs::write(output_root.join("aisim-runtime-values-raw.json"), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("RUN_REQUEST {}", raw);
    println!("MODEL_MANIFEST {}", manifest);
    println!("RAW_RUNTIME_VALUES {}", report);
    let requests = report["requests"].as_array().unwrap();
    assert!(!requests.is_empty(), "capture must not pass vacuously");
    let phase_reports = report["native_phase_report"]["reports"].as_array()
        .or_else(|| report["native_phase_report"]["phases"].as_array())
        .expect("native phase report population");
    assert!(!phase_reports.is_empty());
    assert!(!observer.events().is_empty());
    for (index, row) in requests.iter().enumerate() {
        assert_eq!(row["issue_order"], index);
        assert!(row["issue_ns"].is_i64());
        if row["request"].get("node_id").is_some() {
            assert!(row["request"]["node_id"].is_string());
            assert!(row["request"]["conversation_id"].is_string());
        } else {
            // Synthetic and ordinary scheduled inputs legitimately have no replay provenance.
            assert!(row["request"].get("replay_source").is_some());
            assert!(row["runtime_occurrence"].is_u64());
        }
        let bytes: Vec<u8> = serde_json::from_value(row["wire"]["body_bytes"].clone()).unwrap();
        assert!(!bytes.is_empty());
        if let Some(text) = row["wire"]["body_utf8"].as_str() {
            assert_eq!(std::str::from_utf8(&bytes).unwrap(), text);
        }
        let decoded: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded, row["request"]["body"]);
        assert!(!row["input_token_ids"].as_array().unwrap().is_empty());
        assert_ne!(row["status"], "issued");
        assert_ne!(row["status"], "infer_pending");
    }
}
