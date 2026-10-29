// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::{Value, json};

fn config(speculation: Value, nextn: u32) -> Value {
    json!({
        "model": "Qwen/Qwen3-8B", "system": "h200_sxm", "backend": "vllm",
        "backend_version": "0.24.0", "worker_type": "aggregated",
        "estimation_mode": "op_level", "database_mode": "SOL",
        "speculation": speculation, "nextn": nextn,
    })
}

fn generic_config() -> bool {
    match std::env::var("AIS_CALLBACK_GENERIC_CONFIG")
        .unwrap()
        .as_str()
    {
        "1" => true,
        "0" => false,
        other => panic!("unexpected API shape: {other}"),
    }
}

fn assert_decode(config: Value, expected_depth: u32) {
    let callback = build_model(&config).unwrap();
    assert_eq!(callback.nextn, expected_depth);
    // Independent packing: two requests, context 129, width depth + 1.
    // Do not call the adapter's decode_metrics helper to form the oracle.
    let packed = 2 * (expected_depth + 1);
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_decode_requests: packed,
            sum_decode_kv_tokens: packed * 129,
            ..Default::default()
        },
        ..Default::default()
    };
    let expected = callback
        .model
        .estimate_forward_pass_time_ms(&[metrics])
        .unwrap()
        .unwrap();
    let actual = callback.predict_decode(2, 128, 2).unwrap();
    assert!(actual.is_finite() && actual > 0.0);
    assert!((actual - expected).abs() <= expected.abs() * 1e-12);
}

#[test]
fn autoregressive_and_legacy_depth_reach_actual_decode() {
    assert_decode(config(Value::Null, 0), 0);
    for depth in [1, 5] {
        assert_decode(config(Value::Null, depth), depth);
    }
}

#[test]
fn ngram_depth_reaches_actual_decode() {
    for depth in [1, 5] {
        assert_decode(
            config(
                json!({"kind": "ngram", "params": {"num_speculative_tokens": depth}}),
                0,
            ),
            depth,
        );
    }
}

#[test]
fn disabled_identity_obeys_selected_core_schema() {
    let value = config(json!({"kind": "none"}), 0);
    if generic_config() {
        assert_decode(value, 0);
        assert_decode(config(json!({"kind": "none", "params": {}}), 0), 0);
    } else {
        let error = build_model(&value)
            .err()
            .expect("legacy enum must reject none");
        assert!(error.to_string().contains("invalid AIS perf config"));
    }
}

#[test]
fn malformed_disabled_identity_and_legacy_conflict_are_rejected() {
    for spec in [
        json!({"kind": "none", "params": {"depth": 1}}),
        json!({"kind": "none", "draft_config": {}}),
        json!({"kind": "none", "draft_model_path": "must-not-be-loaded"}),
    ] {
        assert!(build_model(&config(spec, 0)).is_err());
    }
    assert!(build_model(&config(json!({"kind": "none"}), 1)).is_err());
}

#[test]
fn unsupported_scheme_and_invalid_ngram_depth_are_rejected() {
    let error = build_model(&config(json!({"kind": "mtp", "params": {"depth": 1}}), 0))
        .err()
        .expect("cost-only schemes must not enable replay");
    let expected = if generic_config() {
        "Dynamo replay requires ngram speculation or legacy nextn"
    } else {
        "invalid AIS perf config"
    };
    assert!(error.to_string().contains(expected), "{error}");
    for depth in [json!(0), json!(6), json!(1.5), json!(true), json!("1")] {
        assert!(
            build_model(&config(
                json!({"kind": "ngram", "params": {"num_speculative_tokens": depth}}),
                0
            ))
            .is_err()
        );
    }
}
