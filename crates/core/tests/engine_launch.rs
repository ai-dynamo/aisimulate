// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aisimulate_core::engine::{Backend, EngineConfig, EngineLaunchConfig};
use serde_json::{Value, json};

#[test]
fn legacy_launch_converges_on_canonical_engine_defaults() {
    for (backend, block_size) in [("vllm", 64), ("sglang", 1), ("trtllm", 32)] {
        let config = EngineLaunchConfig::from_value(json!({
            "engine_type": backend, "block_size": 0, "max_model_len": 128,
            "max_num_seqs": null, "ais_nextn": 2,
            "ais_nextn_accept_rates": "0.5", "is_decode": true,
        }))
        .unwrap();
        assert_eq!(config.block_size, block_size);
        assert_eq!(config.max_model_len, Some(128));
        assert_eq!(config.max_num_seqs, usize::MAX);
        assert_eq!(config.aic_nextn_accept_rates.as_deref(), Some("0.5,0"));
        assert!(config.is_decode());
        assert_eq!(
            EngineLaunchConfig::from_value(serde_json::to_value(&config).unwrap()).unwrap(),
            config
        );
    }
    let sglang = EngineLaunchConfig::from_value(json!({
        "engine_type":"sglang", "sglang":{"page_size":4,"schedule_policy":"fcfs"}
    }))
    .unwrap();
    assert_eq!(sglang.block_size, 4);
    assert_eq!(
        EngineLaunchConfig::default().engine,
        EngineConfig::default()
    );
}

#[test]
fn engine_validation_is_available_to_external_consumers() {
    let mut config = EngineConfig::for_backend(Backend::Sglang);
    config.block_size = 4;
    config.sglang.chunked_prefill_size = 5;
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("divisible by block_size")
    );
    config.sglang.chunked_prefill_size = 8;
    config.validate().unwrap();
    config.max_model_len = Some(0);
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("max_model_len")
    );
}

#[test]
fn launch_rejects_unknown_fields_conflicting_aliases_and_invalid_controls() {
    for input in [
        json!({"unknown_engine_flag": true}),
        json!({"sglang":{"unknown_flag":null}}),
        json!({"trtllm":{"unknown_flag":null}}),
        json!({"engine_type":"vllm","backend":"sglang"}),
        json!({"engine_type":"sglang","block_size":4,"sglang":{"page_size":8}}),
        json!({"engine_type":"sglang","block_size":4,"sglang":{"chunked_prefill_size":5}}),
        json!({"engine_type":"trtllm","trtllm":{"capacity_scheduler_policy":"max_utilization"}}),
        json!({"ais_nextn":2,"aic_nextn":3}),
        json!({"ais_nextn":1,"decode_speedup_ratio":2}),
        json!({"ais_nextn_accept_rates":"0.5"}),
        json!({"ais_nextn":1,"ais_nextn_accept_rates":"nan"}),
        json!({"max_model_len":0}),
        json!({"num_gpu_blocks":0}),
        json!({"is_prefill":true,"is_decode":true}),
        json!({"enable_chunked_prefill":null}),
        json!({"dp_size":0}),
        json!({"tensor_parallel_size":0}),
        json!({"startup_time":-1}),
        json!({"gpu_memory_utilization":1.1}),
    ] {
        assert!(
            EngineLaunchConfig::from_value(input.clone()).is_err(),
            "accepted {input}"
        );
    }
}

#[test]
fn performance_identity_and_explicit_launch_topology_cannot_diverge() {
    let perf = json!({"model":"model", "system":"gpu", "backend":"vllm",
                      "worker_type":"aggregated", "tp":2, "attention_dp":2});
    let valid = EngineLaunchConfig::from_value(json!({"ais_perf_config":perf})).unwrap();
    assert_eq!(valid.tensor_parallel_size, 2);
    assert_eq!(valid.dp_size, 2);
    // A caller may intentionally compare one scheduler with another backend's timing.
    let alternate = EngineLaunchConfig::from_value(json!({
        "engine_type":"sglang", "ais_perf_config":perf
    }))
    .unwrap();
    assert_eq!(alternate.backend, Backend::Sglang);
    for extra in [
        json!({"tensor_parallel_size":1}),
        json!({"dp_size":1}),
        json!({"worker_type":"decode"}),
        json!({"timing_model":{"type":"fixed", "prefill_ms":1, "decode_ms":1}}),
    ] {
        let mut input = extra.as_object().unwrap().clone();
        input.insert("ais_perf_config".into(), perf.clone());
        assert!(EngineLaunchConfig::from_value(Value::Object(input)).is_err());
    }
    let mut pipeline = perf;
    pipeline["pp"] = json!(2);
    assert!(EngineLaunchConfig::from_value(json!({"ais_perf_config":pipeline})).is_err());
}
