// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python wheel host for the published AISimulate core and native routing.

#[cfg(feature = "python")]
pub use aisimulate_core::ReplayExecutionResult;
pub use aisimulate_core::{engine, replay};
pub mod native_router;

#[cfg(feature = "python")]
use aisimulate_core::{
    execute_replay_with_composition, execute_replay_with_dispatch, register_python_module,
    replay_python_error, run_prepared_replay,
};
#[cfg(feature = "python")]
use anyhow::{Result, ensure};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use replay::{ReplayCaptureOptions, ReplayTelemetryObserver, Replayer};

#[cfg(feature = "python")]
fn execute_json(payload: &str, capture_artifacts: bool) -> Result<String> {
    use native_router::{
        NATIVE_DYNAMO_PROVIDER, NativeDynamoReplayComposition, NativeDynamoRouterConfig,
    };
    let mut routing_evidence = None;
    let mut result = execute_replay_with_dispatch(
        payload,
        capture_artifacts,
        |spec, factory, input, capture_artifacts, determinism, tokenization| {
            if spec.adapters.placement.provider != NATIVE_DYNAMO_PROVIDER {
                return run_prepared_replay(
                    spec,
                    factory,
                    input,
                    capture_artifacts,
                    determinism,
                    tokenization,
                );
            }
            if capture_artifacts {
                return Err(replay::ReplayError::InvalidSpec(
                    "native routing does not expose single-worker parity artifacts".into(),
                ));
            }
            let config = NativeDynamoRouterConfig::from_value(&spec.adapters.placement.config)
                .map_err(|error| replay::ReplayError::InvalidSpec(error.to_string()))?;
            let composition = NativeDynamoReplayComposition::from_config(config, None)
                .with_capture_decisions(spec.record_per_request);
            routing_evidence = Some(composition.routing_evidence());
            let mut replayer = Replayer::with_composition(spec, factory, composition)?
                .with_capture_options(ReplayCaptureOptions {
                    determinism,
                    ..Default::default()
                });
            if let Some(input) = input {
                replayer = replayer.with_runtime_input(input);
            }
            if let Some(tokenization) = tokenization {
                replayer = replayer.with_tokenization(
                    tokenization.workers,
                    tokenization.base_ms,
                    tokenization.per_input_token_ms,
                )?;
            }
            replayer.run().map(|report| (report, None))
        },
    )?;
    if let Some(evidence) = routing_evidence {
        result
            .report_fields
            .insert("routing_policy".into(), evidence.snapshot()?);
    }
    result.into_json()
}

/// Execute native routing inside AISimulate, retaining optional consumer observers
/// and scaling callbacks without transferring ownership of the simulation loop.
#[cfg(feature = "python")]
pub fn execute_native_router_replay(
    payload: &str,
    capture: ReplayCaptureOptions,
    scaling: Option<Box<dyn crate::replay::ReplayScalingPolicy>>,
    telemetry: Option<(f64, Box<dyn ReplayTelemetryObserver>)>,
    prefill_load_estimator: Option<crate::native_router::ReplayPrefillLoadEstimator>,
) -> Result<ReplayExecutionResult> {
    use crate::native_router::{
        NATIVE_DYNAMO_PROVIDER, NativeDynamoReplayComposition, NativeDynamoRouterConfig,
    };
    let envelope: serde_json::Value = serde_json::from_str(payload)?;
    let spec = envelope.get("spec").unwrap_or(&envelope);
    ensure!(
        spec.pointer("/adapters/placement/provider")
            .and_then(|v| v.as_str())
            == Some(NATIVE_DYNAMO_PROVIDER),
        "native routing requires the dynamo_kv_router placement provider"
    );
    let config = NativeDynamoRouterConfig::from_value(
        spec.pointer("/adapters/placement/config")
            .unwrap_or(&serde_json::Value::Null),
    )?;
    let capture_decisions = capture.effective_per_request()
        || spec
            .get("record_per_request")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
    let mut composition = NativeDynamoReplayComposition::from_config(config, scaling)
        .with_capture_decisions(capture_decisions);
    if let Some(estimator) = prefill_load_estimator {
        composition = composition.with_prefill_load_estimator(estimator);
    }
    let evidence = composition.routing_evidence();
    let mut result = execute_replay_with_composition(payload, composition, capture, telemetry)?;
    result
        .report_fields
        .insert("routing_policy".into(), evidence.snapshot()?);
    Ok(result)
}

#[cfg(feature = "python")]
#[pyfunction]
fn run_replay_json(py: Python<'_>, payload: &str) -> PyResult<String> {
    py.allow_threads(|| execute_json(payload, false))
        .map_err(replay_python_error)
}

#[cfg(feature = "python")]
#[pyfunction]
fn run_replay_with_artifacts_json(py: Python<'_>, payload: &str) -> PyResult<String> {
    py.allow_threads(|| execute_json(payload, true))
        .map_err(replay_python_error)
}

#[cfg(feature = "python")]
#[pyfunction]
fn native_router_available() -> bool {
    true
}

#[cfg(feature = "python")]
#[pymodule]
fn _runtime(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_python_module(module)?;
    module.add_function(wrap_pyfunction!(run_replay_json, module)?)?;
    module.add_function(wrap_pyfunction!(run_replay_with_artifacts_json, module)?)?;
    module.add_function(wrap_pyfunction!(native_router_available, module)?)?;
    Ok(())
}

#[cfg(all(test, feature = "python"))]
mod tests {
    use super::*;

    #[test]
    fn wheel_module_registers_core_api_and_dispatches_native_and_default_replay() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_runtime").unwrap();
            _runtime(&module).unwrap();
            assert!(
                module
                    .getattr("native_router_available")
                    .unwrap()
                    .call0()
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
            for function in [
                "canonical_timing_config_json",
                "engine_launch_from_replay_role_json",
                "run_replay_with_artifacts_json",
            ] {
                assert!(module.getattr(function).unwrap().is_callable());
            }
            let mut payload = serde_json::json!({
                "version": 1,
                "topology": {"kind": "aggregated", "workers": {"initial_workers": 1}},
                "engine": {"rank": {"block_size": 4, "num_gpu_blocks": 64,
                    "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}}},
                "adapters": {"placement": {"provider": "dynamo_kv_router", "config": {}},
                    "scaling": {"provider": "none"}},
                "requests": [{"id": "module-call", "arrival_time_ms": 0.0,
                    "input_tokens": 4, "input_token_ids": [1, 2, 3, 4], "output_tokens": 1}]
            });
            let run = |payload: &serde_json::Value| {
                let output = module
                    .getattr("run_replay_json")
                    .unwrap()
                    .call1((payload.to_string(),))
                    .unwrap()
                    .extract::<String>()
                    .unwrap();
                serde_json::from_str::<serde_json::Value>(&output).unwrap()
            };
            let native = run(&payload);
            assert_eq!(native["completed_requests"], 1);
            assert!(native.get("routing_policy").is_some());

            payload["adapters"]["placement"]["provider"] = "round_robin".into();
            let default = run(&payload);
            assert_eq!(default["completed_requests"], 1);
            assert!(default.get("routing_policy").is_none());
            assert_eq!(default["per_request"][0]["request_id"], "module-call");
        });
    }

    #[test]
    fn wheel_and_shared_native_executor_preserve_tokenization() {
        for disaggregated in [false, true] {
            for provider in ["round_robin", "dynamo_kv_router"] {
                let topology = if disaggregated {
                    serde_json::json!({"kind": "disaggregated", "prefill": {"initial_workers": 1},
                        "decode": {"initial_workers": 1}})
                } else {
                    serde_json::json!({"kind": "aggregated", "workers": {"initial_workers": 1}})
                };
                let mut payload = serde_json::json!({
                    "spec": {
                        "version": 1,
                        "topology": topology,
                        "engine": {"rank": {"block_size": 4, "num_gpu_blocks": 64,
                            "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}}},
                        "adapters": {"placement": {"provider": provider, "config": {}},
                            "scaling": {"provider": "none"}},
                        "requests": [{"id": "tokenized", "arrival_time_ms": 0.0,
                            "input_tokens": 4, "input_token_ids": [1, 2, 3, 4], "output_tokens": 1}]
                    },
                    "tokenization": {"workers": 1, "base_ms": 2.0, "per_input_token_ms": 0.5}
                });
                let mut reports = vec![
                    serde_json::from_str::<serde_json::Value>(
                        &execute_json(&payload.to_string(), false).unwrap(),
                    )
                    .unwrap(),
                ];
                if provider == "dynamo_kv_router" {
                    reports.push(
                        serde_json::from_str(
                            &execute_native_router_replay(
                                &payload.to_string(),
                                ReplayCaptureOptions::default(),
                                None,
                                None,
                                None,
                            )
                            .unwrap()
                            .into_json()
                            .unwrap(),
                        )
                        .unwrap(),
                    );
                } else if !disaggregated {
                    let artifacts: serde_json::Value =
                        serde_json::from_str(&execute_json(&payload.to_string(), true).unwrap())
                            .unwrap();
                    reports.push(artifacts["report"].clone());
                }
                for report in reports {
                    assert_eq!(report["completed_requests"], 1);
                    // One tokenizer takes 2 + 0.5 * 4 = 4 ms before engine admission.
                    assert_eq!(report["per_request"][0]["tokenized_ms"], 4.0);
                    assert_eq!(report["per_request"][0]["first_admit_ms"], 4.0);
                }
                payload["tokenization"]["workers"] = 0.into();
                assert!(
                    format!(
                        "{:#}",
                        execute_json(&payload.to_string(), false).unwrap_err()
                    )
                    .contains("tokenization workers must be positive")
                );
            }
        }
    }

    #[test]
    fn native_router_execution_is_owned_by_aisimulate() {
        let payload = serde_json::json!({
            "version": 1,
            "topology": {"kind": "aggregated", "workers": {"initial_workers": 2}},
            "engine": {"rank": {"block_size": 4, "num_gpu_blocks": 64,
                "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}}},
            "adapters": {"placement": {"provider": "dynamo_kv_router", "config": {
                "affinity": {"mode": "session", "ttl_seconds": 60.0}
            }}, "scaling": {"provider": "none"}},
            "requests": [
                {"id":"first", "arrival_time_ms":0.0, "input_tokens":4,
                    "input_token_ids":[1,2,3,4], "output_tokens":1, "session_id":"shared"},
                {"id":"second", "arrival_time_ms":10.0, "input_tokens":4,
                    "input_token_ids":[1,2,3,4], "output_tokens":1, "session_id":"shared"}
            ]
        })
        .to_string();
        {
            let report: serde_json::Value =
                serde_json::from_str(&execute_json(&payload, false).unwrap()).unwrap();
            let role = &report["routing_policy"]["roles"]["aggregated"];
            assert_eq!(role["native_policy"], "dynamo.DefaultWorkerSelector");
            assert_eq!(
                role["dynamo_revision"],
                crate::native_router::NATIVE_DYNAMO_REVISION
            );
            assert_eq!(role["decision_count"], 2);
            assert_eq!(role["post_dispatch_checks"], 2);
            assert!(role["physical_kv_events"].as_u64().unwrap() > 0);
            assert_eq!(role["decisions"][1]["binding_reused"], true);
            assert_eq!(
                role["decisions"][0]["worker_id"],
                role["decisions"][1]["worker_id"]
            );
            assert!(role["decisions"][1]["overlap_blocks"].as_u64().unwrap() > 0);
            assert_eq!(report["per_request"].as_array().unwrap().len(), 2);

            let mut no_capture: serde_json::Value = serde_json::from_str(&payload).unwrap();
            no_capture["record_per_request"] = false.into();
            let report: serde_json::Value =
                serde_json::from_str(&execute_json(&no_capture.to_string(), false).unwrap())
                    .unwrap();
            assert_eq!(
                report["routing_policy"]["roles"]["aggregated"]["decision_count"],
                2
            );
            assert!(
                report["routing_policy"]["roles"]["aggregated"]["decisions"]
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
        }
    }
}
