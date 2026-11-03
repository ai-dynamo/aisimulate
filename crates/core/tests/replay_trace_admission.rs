// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aisimulate_core::engine::{Backend, EngineConfig, TimingModelConfig};
use aisimulate_core::replay::{
    CURRENT_REPLAY_SPEC_VERSION, DirectRequest, ReplayEngineConfig, ReplayEngineFactory,
    ReplayError, ReplayRuntimeInput, ReplaySpec, ReplayTopology, Replayer, WorkerPoolSpec,
    run_engine_handoff_conformance,
};
use uuid::Uuid;

fn engine(backend: Backend) -> ReplayEngineConfig {
    ReplayEngineConfig {
        rank: EngineConfig {
            block_size: 4,
            num_gpu_blocks: 64,
            max_num_seqs: 4,
            max_num_batched_tokens: 64,
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 1.0,
                decode_ms: 1.0,
            },
            ..EngineConfig::for_backend(backend)
        },
        ..Default::default()
    }
}

fn spec(backend: Backend, disagg: bool) -> ReplaySpec {
    ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
        encoder: None,
        topology: if disagg {
            ReplayTopology::Disaggregated {
                prefill: WorkerPoolSpec::default(),
                decode: WorkerPoolSpec::default(),
                handoff_latency_ms: 0.0,
            }
        } else {
            ReplayTopology::aggregated(1)
        },
        engine: serde_json::to_value(engine(backend)).unwrap(),
        adapters: Default::default(),
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests: Vec::new(),
    }
}

fn request(id: u128, arrival_timestamp_ms: Option<f64>) -> DirectRequest {
    DirectRequest {
        tokens: vec![1, 2, 3, 4],
        max_output_tokens: 2,
        uuid: Some(Uuid::from_u128(id)),
        arrival_timestamp_ms,
        ..Default::default()
    }
}

fn assert_admission_error(error: ReplayError, malformed: Option<f64>, queue_index: usize) {
    assert!(matches!(error, ReplayError::Invariant(_)), "{error}");
    let message = error.to_string();
    for expected in [
        "arrival timestamp".to_string(),
        malformed.map_or_else(|| "missing".to_string(), |value| value.to_string()),
        Uuid::from_u128(99).to_string(),
        format!("queue index {queue_index}"),
        "0 admitted at 0ms".to_string(),
    ] {
        assert!(
            message.contains(&expected),
            "missing {expected:?}: {message}"
        );
    }
}

#[test]
fn replay_consumers_reject_malformed_trace_arrivals_before_scheduling() {
    for backend in [Backend::Vllm, Backend::Sglang, Backend::Trtllm] {
        for disagg in [false, true] {
            for malformed in [
                None,
                Some(f64::NAN),
                Some(f64::INFINITY),
                Some(f64::NEG_INFINITY),
            ] {
                for valid_prefix in [false, true] {
                    let mut requests = std::collections::VecDeque::new();
                    if valid_prefix {
                        requests.push_back(request(1, Some(0.0)));
                    }
                    requests.push_back(request(99, malformed));
                    let error = Replayer::new(spec(backend, disagg), ReplayEngineFactory::new())
                        .unwrap()
                        .with_runtime_input(ReplayRuntimeInput::Requests(requests))
                        .run()
                        .expect_err("malformed arrival must surface through the runtime");
                    assert_admission_error(error, malformed, usize::from(valid_prefix));
                }
            }
        }
    }
}

#[test]
fn replay_consumers_admit_boundary_and_future_trace_arrivals_at_the_authored_time() {
    for backend in [Backend::Vllm, Backend::Sglang, Backend::Trtllm] {
        for disagg in [false, true] {
            let arrivals = [0.0, 10.0, 20.0];
            let requests = arrivals
                .iter()
                .enumerate()
                .map(|(index, &arrival)| request(index as u128 + 1, Some(arrival)))
                .collect();
            let report = Replayer::new(spec(backend, disagg), ReplayEngineFactory::new())
                .unwrap()
                .with_runtime_input(ReplayRuntimeInput::Requests(requests))
                .run()
                .unwrap();
            assert_eq!(report.request_counts.completed_requests, arrivals.len());
            assert_eq!(report.per_request.len(), arrivals.len());
            for (index, arrival) in arrivals.into_iter().enumerate() {
                let id = Uuid::from_u128(index as u128 + 1).to_string();
                let record = report
                    .per_request
                    .iter()
                    .find(|record| record.uuid == id)
                    .unwrap();
                assert_eq!(record.arrival_time_ms, arrival);
                assert_eq!(record.first_admit_ms, Some(arrival));
                assert!(record.terminal_time_ms > arrival);
            }
        }
    }
}

#[test]
fn handoff_conformance_rejects_malformed_trace_arrivals() {
    for backend in [Backend::Vllm, Backend::Sglang, Backend::Trtllm] {
        for malformed in [
            None,
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
        ] {
            let error = run_engine_handoff_conformance(
                engine(backend),
                ReplayEngineFactory::new(),
                request(99, malformed),
            )
            .expect_err("handoff conformance must propagate admission validation");
            assert_admission_error(error, malformed, 0);
        }
    }
}
