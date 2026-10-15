// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aisimulate_core::engine::{Backend, EngineConfig, EngineLaunchConfig};
use serde_json::json;

#[test]
fn canonical_launch_preserves_capacity_intent_and_backend_defaults() {
    for (backend, block_size) in [("vllm", 64), ("sglang", 1), ("trtllm", 32)] {
        let config = EngineLaunchConfig::from_value(
            json!({"engine":{"backend":backend,"max_model_len":128}}),
        )
        .unwrap();
        assert_eq!(config.block_size, block_size);
        assert!(!config.num_gpu_blocks_is_explicit);
        let restored =
            EngineLaunchConfig::from_value(serde_json::to_value(&config).unwrap()).unwrap();
        assert_eq!(restored, config);
        let explicit = EngineLaunchConfig::from_value(
            json!({"engine":{"backend":backend,"num_gpu_blocks":16}}),
        )
        .unwrap();
        assert!(explicit.num_gpu_blocks_is_explicit);
    }
    for invalid in [
        json!({"engine_type":"vllm"}),
        json!({"engine":{"backend":"vllm","sglang":{"page_size":16}}}),
    ] {
        assert!(EngineLaunchConfig::from_value(invalid).is_err());
    }
}

#[test]
fn shared_engine_validation_and_estimation_limits_are_public() {
    let mut engine = EngineConfig::for_backend(Backend::Sglang);
    engine.block_size = 4;
    engine.sglang.chunked_prefill_size = 5;
    assert!(
        engine
            .validate()
            .unwrap_err()
            .to_string()
            .contains("divisible by block_size")
    );
    engine.sglang.chunked_prefill_size = 8;
    engine.validate().unwrap();
    let launch = EngineLaunchConfig::from_value(
        json!({"engine":{"max_num_seqs":null,"max_num_batched_tokens":null}}),
    )
    .unwrap();
    assert!(
        !launch
            .capacity_estimation_options()
            .contains_key("max_num_seqs")
    );
    assert!(
        !launch
            .capacity_estimation_options()
            .contains_key("max_num_batched_tokens")
    );
}

#[test]
fn performance_identity_and_launch_controls_are_validated_once() {
    let perf = json!({"model":"model","system":"gpu","backend":"vllm","worker_type":"aggregated","tp":2,"attention_dp":2});
    let input =
        json!({"engine":{"timing_model":{"type":"external","provider":"ais","config":perf}}});
    let config = EngineLaunchConfig::from_value(input.clone()).unwrap();
    assert_eq!(config.dp_size, 2);
    assert_eq!(config.tensor_parallel_size, 2);
    assert_eq!(
        serde_json::to_value(&config.timing_model).unwrap()["config"],
        perf
    );
    for override_ in [
        json!({"dp_size":1}),
        json!({"tensor_parallel_size":1}),
        json!({"startup_time":-1}),
        json!({"gpu_memory_utilization":1.1}),
    ] {
        let mut invalid = input.clone();
        invalid
            .as_object_mut()
            .unwrap()
            .extend(override_.as_object().unwrap().clone());
        assert!(EngineLaunchConfig::from_value(invalid).is_err());
    }
    let mut invalid = input;
    invalid["engine"]["timing_model"]["config"]["pp"] = json!(2);
    assert!(EngineLaunchConfig::from_value(invalid).is_err());
}
