// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aisimulate_core::engine::{Backend, EngineConfig, TimingModel};
use aisimulate_core::replay::{
    ReplayAdapters, ReplayEngineConfig, ReplayError, ReplayRequest, ReplaySpec, ReplayTopology,
    WorkerPoolSpec, run_engine_replay_with_timing,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct TestTiming {
    invalid: Option<f64>,
    fail_prefill: bool,
    invalid_calls: AtomicUsize,
}

impl TimingModel for TestTiming {
    fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
        if self.fail_prefill
            && let Some(value) = self.invalid
        {
            self.invalid_calls.fetch_add(1, Ordering::Relaxed);
            return Ok(value);
        }
        Ok(1.0)
    }

    fn predict_decode_ms(&self, _: usize, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
        if !self.fail_prefill
            && let Some(value) = self.invalid
        {
            self.invalid_calls.fetch_add(1, Ordering::Relaxed);
            return Ok(value);
        }
        Ok(1.0)
    }
}

// Exercise the public runtime, not just the heap: an invalid timing result must
// propagate as an engine error before it can be normalized or scheduled. Finite
// controls prove the same request reaches completion through each configuration.
#[test]
fn public_replay_rejects_non_finite_timing_before_normalization() {
    for backend in [Backend::Vllm, Backend::Sglang, Backend::Trtllm] {
        for dp_size in [1, 2] {
            for disagg in [false, true] {
                for fail_prefill in [false, true] {
                    for invalid in [
                        None,
                        Some(f64::NAN),
                        Some(f64::INFINITY),
                        Some(f64::NEG_INFINITY),
                    ] {
                        let config = ReplayEngineConfig {
                            dp_size,
                            rank: EngineConfig {
                                backend,
                                ..EngineConfig::default()
                            },
                            ..ReplayEngineConfig::default()
                        };
                        let spec = ReplaySpec {
                            version: 1,
                            topology: if disagg {
                                ReplayTopology::Disaggregated {
                                    prefill: WorkerPoolSpec::default(),
                                    decode: WorkerPoolSpec::default(),
                                    handoff_latency_ms: 0.0,
                                }
                            } else {
                                ReplayTopology::aggregated(1)
                            },
                            engine: serde_json::to_value(config).unwrap(),
                            adapters: ReplayAdapters::default(),
                            max_sim_time_ms: None,
                            max_in_flight: None,
                            record_per_request: true,
                            sla: Default::default(),
                            requests: vec![ReplayRequest {
                                id: "non-finite-timing".into(),
                                arrival_time_ms: 0.0,
                                input_tokens: 4,
                                input_token_ids: Some(vec![1, 2, 3, 4]),
                                output_tokens: 3,
                                output_token_ids: None,
                                dp_rank: None,
                                prefill_dp_rank: None,
                                session_id: None,
                                turn_index: None,
                                metadata: serde_json::Value::Null,
                            }],
                        };
                        let timing = Arc::new(TestTiming {
                            invalid,
                            fail_prefill,
                            invalid_calls: AtomicUsize::new(0),
                        });
                        let result = run_engine_replay_with_timing(spec, timing.clone());
                        if invalid.is_some() {
                            assert!(
                                timing.invalid_calls.load(Ordering::Relaxed) > 0,
                                "provider not exercised: {backend:?} disagg={disagg} dp={dp_size} prefill={fail_prefill}: {result:?}"
                            );
                            let error = result.expect_err(
                                "invalid timing must fail instead of completing or hanging",
                            );
                            assert!(
                                matches!(error, ReplayError::Engine(_)),
                                "wrong error classification: {error}"
                            );
                            assert!(
                                error
                                    .to_string()
                                    .contains("timing provider returned non-finite duration"),
                                "wrong boundary: {error}"
                            );
                        } else {
                            let report = result.expect("finite timing control must complete");
                            assert_eq!(report.request_counts.num_requests, 1);
                            assert_eq!(report.request_counts.completed_requests, 1);
                        }
                    }
                }
            }
        }
    }
}
