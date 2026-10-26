// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use aisimulate_core::engine::{Backend, EngineConfig, TimingModelConfig};
use aisimulate_core::replay::loadgen::{
    AGENTIC_MOONCAKE_SCHEMA, AGENTIC_MOONCAKE_VERSION, AgenticHashIdScope, AgenticMooncakeHeader,
    AgenticMooncakeRow, AgenticSourceProvenance, ValidatedAgenticGraph, WorkloadDriver,
};
use aisimulate_core::replay::{
    CURRENT_REPLAY_SPEC_VERSION, DirectRequest, PerRequestRecord, ReplayEngineConfig,
    ReplayEngineFactory, ReplayError, ReplayReport, ReplayRuntimeInput, ReplaySpec,
    ReplayTerminalStatus, ReplayTopology, Replayer, WorkerPoolSpec,
};
use uuid::Uuid;

const FIRST: u128 = 1;
const SECOND: u128 = 2;
const BACKENDS: [Backend; 3] = [Backend::Vllm, Backend::Sglang, Backend::Trtllm];

/// `(workers, base_ms, per_output_ms)`.
type Detokenization = (usize, f64, f64);

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

fn spec(backend: Backend, disagg: bool, max_in_flight: Option<usize>) -> ReplaySpec {
    let pool = WorkerPoolSpec {
        initial_workers: 1,
        ..WorkerPoolSpec::default()
    };
    ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
        encoder: None,
        topology: if disagg {
            ReplayTopology::Disaggregated {
                prefill: pool,
                decode: WorkerPoolSpec::default(),
                handoff_latency_ms: 0.0,
            }
        } else {
            ReplayTopology::Aggregated { workers: pool }
        },
        engine: serde_json::to_value(engine(backend)).unwrap(),
        adapters: Default::default(),
        max_sim_time_ms: None,
        max_in_flight,
        record_per_request: true,
        sla: Default::default(),
        requests: Vec::new(),
    }
}

fn request(
    id: u128,
    input_tokens: u32,
    output_tokens: usize,
    arrival_timestamp_ms: Option<f64>,
) -> DirectRequest {
    DirectRequest {
        tokens: (0..input_tokens)
            .map(|token| token + id as u32 * 1000)
            .collect(),
        max_output_tokens: output_tokens,
        uuid: Some(Uuid::from_u128(id)),
        arrival_timestamp_ms,
        ..Default::default()
    }
}

fn run(
    spec: ReplaySpec,
    requests: VecDeque<DirectRequest>,
    detokenization: Option<Detokenization>,
) -> Result<ReplayReport, ReplayError> {
    let mut replayer = Replayer::new(spec, ReplayEngineFactory::new())?
        .with_runtime_input(ReplayRuntimeInput::Requests(requests));
    if let Some((workers, base_ms, per_output_ms)) = detokenization {
        replayer = replayer.with_detokenization(workers, base_ms, per_output_ms)?;
    }
    replayer.run()
}

fn record(report: &ReplayReport, id: u128) -> &PerRequestRecord {
    let uuid = Uuid::from_u128(id).to_string();
    report
        .per_request
        .iter()
        .find(|record| record.uuid == uuid)
        .unwrap()
}

fn assert_close(actual: Option<f64>, expected: Option<f64>, context: &str) {
    let (actual, expected) = (actual.unwrap(), expected.unwrap());
    assert!(
        (actual - expected).abs() < 1e-9,
        "{context}: {actual} != {expected}"
    );
}

#[test]
fn a_lone_request_receives_each_output_after_detokenization() {
    for backend in BACKENDS {
        for disagg in [false, true] {
            let context = format!("{backend:?} disagg={disagg}");
            let requests = || VecDeque::from([request(FIRST, 8, 3, Some(0.0))]);
            let direct = run(spec(backend, disagg, None), requests(), None).unwrap();
            // Each pass's one output takes 0.5 + 0.25 on the detokenizer,
            // which is idle again before the next pass ends.
            let streamed = run(
                spec(backend, disagg, None),
                requests(),
                Some((1, 0.5, 0.25)),
            )
            .unwrap();
            let (direct, streamed) = (record(&direct, FIRST), record(&streamed, FIRST));
            assert_eq!(streamed.first_admit_ms, direct.first_admit_ms, "{context}");
            assert_close(
                streamed.first_token_ms,
                direct.first_token_ms.map(|ms| ms + 0.75),
                &context,
            );
            assert_close(
                streamed.last_token_ms,
                direct.last_token_ms.map(|ms| ms + 0.75),
                &context,
            );
            assert_close(
                Some(streamed.terminal_time_ms),
                streamed.last_token_ms,
                &context,
            );
            assert_close(
                streamed.ttft_ms,
                direct.ttft_ms.map(|ms| ms + 0.75),
                &context,
            );
            assert_close(
                streamed.e2e_latency_ms,
                direct.e2e_latency_ms.map(|ms| ms + 0.75),
                &context,
            );
        }
    }
}

#[test]
fn outputs_of_one_pass_reach_the_client_together() {
    // The second request's first token shares a pass with the first request's
    // decode, so both wait for one server to handle two outputs. With two
    // servers the requests belong to different ones and each handles one.
    for disagg in [false, true] {
        let context = format!("disagg={disagg}");
        let requests = || {
            VecDeque::from([
                request(FIRST, 4, 8, Some(0.0)),
                request(SECOND, 4, 2, Some(2.5)),
            ])
        };
        let direct = run(spec(Backend::Vllm, disagg, None), requests(), None).unwrap();
        for (workers, shared_pass_ms) in [(1, 0.5), (2, 0.25)] {
            let context = format!("{context} workers={workers}");
            let streamed = run(
                spec(Backend::Vllm, disagg, None),
                requests(),
                Some((workers, 0.0, 0.25)),
            )
            .unwrap();
            let (direct_second, streamed_second) =
                (record(&direct, SECOND), record(&streamed, SECOND));
            assert_eq!(
                streamed_second.first_admit_ms, direct_second.first_admit_ms,
                "{context}"
            );
            assert_close(
                streamed_second.first_token_ms,
                direct_second.first_token_ms.map(|ms| ms + shared_pass_ms),
                &context,
            );
            assert_close(
                record(&streamed, FIRST).first_token_ms,
                record(&direct, FIRST).first_token_ms.map(|ms| ms + 0.25),
                &context,
            );
        }
    }
}

#[test]
fn the_client_sends_its_next_request_when_it_receives_the_response() {
    for backend in BACKENDS {
        for disagg in [false, true] {
            let context = format!("{backend:?} disagg={disagg}");
            let requests =
                || VecDeque::from([request(FIRST, 8, 2, None), request(SECOND, 8, 2, None)]);
            let direct = run(spec(backend, disagg, Some(1)), requests(), None).unwrap();
            let direct_first = record(&direct, FIRST);
            assert_eq!(
                record(&direct, SECOND).arrival_time_ms,
                direct_first.terminal_time_ms,
                "{context}"
            );

            let streamed = run(
                spec(backend, disagg, Some(1)),
                requests(),
                Some((1, 0.5, 0.0)),
            )
            .unwrap();
            assert_eq!(streamed.request_counts.completed_requests, 2, "{context}");
            let first = record(&streamed, FIRST);
            assert_eq!(
                first.terminal_time_ms,
                direct_first.terminal_time_ms + 0.5,
                "{context}"
            );
            assert_eq!(
                record(&streamed, SECOND).arrival_time_ms,
                first.terminal_time_ms,
                "{context}"
            );
        }
    }
}

fn short_context_spec(disagg: bool, max_in_flight: Option<usize>) -> ReplaySpec {
    let mut spec = spec(Backend::Vllm, disagg, max_in_flight);
    spec.engine = serde_json::to_value(ReplayEngineConfig {
        rank: EngineConfig {
            max_model_len: Some(16),
            ..engine(Backend::Vllm).rank
        },
        ..engine(Backend::Vllm)
    })
    .unwrap();
    spec
}

#[test]
fn a_rejection_reaches_the_client_when_the_engine_rejects_it() {
    for disagg in [false, true] {
        let context = format!("disagg={disagg}");
        let requests =
            || VecDeque::from([request(FIRST, 20, 2, None), request(SECOND, 8, 2, None)]);
        let direct = run(short_context_spec(disagg, Some(1)), requests(), None).unwrap();
        let report = run(
            short_context_spec(disagg, Some(1)),
            requests(),
            Some((1, 0.5, 0.25)),
        )
        .unwrap();
        let (rejected, next) = (record(&report, FIRST), record(&report, SECOND));
        assert_eq!(
            rejected.terminal_status,
            ReplayTerminalStatus::Rejected,
            "{context}"
        );
        // A rejection carries no output, so it skips detokenization.
        assert_eq!(
            rejected.terminal_time_ms,
            record(&direct, FIRST).terminal_time_ms,
            "{context}"
        );
        assert_eq!(next.arrival_time_ms, rejected.terminal_time_ms, "{context}");
    }
}

#[test]
fn a_rejection_does_not_delay_the_outputs_of_its_pass() {
    for disagg in [false, true] {
        let context = format!("disagg={disagg}");
        let detokenization = Some((1, 0.5, 0.25));
        let alone = run(
            short_context_spec(disagg, None),
            VecDeque::from([request(FIRST, 8, 3, Some(0.0))]),
            detokenization,
        )
        .unwrap();
        // The second request exceeds the context and is rejected in the first
        // request's last pass.
        let with_rejection = run(
            short_context_spec(disagg, None),
            VecDeque::from([
                request(FIRST, 8, 3, Some(0.0)),
                request(SECOND, 20, 2, Some(1.5)),
            ]),
            detokenization,
        )
        .unwrap();
        assert_eq!(
            record(&with_rejection, SECOND).terminal_status,
            ReplayTerminalStatus::Rejected,
            "{context}"
        );
        let (alone, with_rejection) = (record(&alone, FIRST), record(&with_rejection, FIRST));
        assert_eq!(
            with_rejection.first_token_ms, alone.first_token_ms,
            "{context}"
        );
        assert_eq!(
            with_rejection.last_token_ms, alone.last_token_ms,
            "{context}"
        );
        assert_eq!(
            with_rejection.terminal_time_ms, alone.terminal_time_ms,
            "{context}"
        );
    }
}

#[test]
fn attention_dp_ranks_of_a_worker_share_its_detokenizer_servers() {
    let dp_spec = || {
        let mut spec = spec(Backend::Vllm, false, None);
        spec.engine = serde_json::to_value(ReplayEngineConfig {
            dp_size: 2,
            ..engine(Backend::Vllm)
        })
        .unwrap();
        spec
    };
    let requests = || {
        VecDeque::from([
            request(FIRST, 8, 2, Some(0.0)),
            request(SECOND, 8, 2, Some(0.0)),
        ])
    };
    let direct = run(dp_spec(), requests(), None).unwrap();
    assert_ne!(
        record(&direct, FIRST).routing_history[0].dp_rank,
        record(&direct, SECOND).routing_history[0].dp_rank
    );
    // Both ranks finish their first pass together. With one server the worker
    // handles one rank's output, then the other's; with two servers the
    // requests belong to different ones.
    for (workers, expected) in [(1, [1.0, 2.0]), (2, [1.0, 1.0])] {
        let streamed = run(dp_spec(), requests(), Some((workers, 0.0, 1.0))).unwrap();
        let mut delays = [FIRST, SECOND].map(|id| {
            record(&streamed, id).first_token_ms.unwrap()
                - record(&direct, id).first_token_ms.unwrap()
        });
        delays.sort_by(f64::total_cmp);
        assert_eq!(delays, expected, "workers={workers}");
    }
}

#[test]
fn zero_detokenization_cost_keeps_pass_end_delivery() {
    for backend in BACKENDS {
        for disagg in [false, true] {
            let requests =
                || VecDeque::from([request(FIRST, 8, 3, None), request(SECOND, 4, 2, None)]);
            let direct = run(spec(backend, disagg, Some(1)), requests(), None).unwrap();
            let zero = run(
                spec(backend, disagg, Some(1)),
                requests(),
                Some((1, 0.0, 0.0)),
            )
            .unwrap();
            assert_eq!(
                serde_json::to_value(&direct.per_request).unwrap(),
                serde_json::to_value(&zero.per_request).unwrap(),
                "{backend:?} disagg={disagg}"
            );
        }
    }
}

#[test]
fn a_response_not_received_by_the_time_cap_is_incomplete() {
    for disagg in [false, true] {
        let requests = || VecDeque::from([request(FIRST, 8, 2, Some(0.0))]);
        let direct = run(spec(Backend::Vllm, disagg, None), requests(), None).unwrap();
        let finished_ms = record(&direct, FIRST).terminal_time_ms;
        let mut capped = spec(Backend::Vllm, disagg, None);
        capped.max_sim_time_ms = Some(finished_ms + 0.25);
        let report = run(capped.clone(), requests(), Some((1, 0.5, 0.0))).unwrap();
        assert_eq!(
            report.request_counts.completed_requests, 0,
            "disagg={disagg}"
        );
        let report = run(capped, requests(), Some((1, 0.2, 0.0))).unwrap();
        assert_eq!(
            report.request_counts.completed_requests, 1,
            "disagg={disagg}"
        );
    }
}

#[test]
fn agentic_requests_complete_through_the_client_path() {
    for disagg in [false, true] {
        let graph = ValidatedAgenticGraph::from_agentic_mooncake_rows(
            AgenticMooncakeHeader {
                schema: AGENTIC_MOONCAKE_SCHEMA.into(),
                version: AGENTIC_MOONCAKE_VERSION,
                block_size: 4,
                hash_id_scope: AgenticHashIdScope::Local,
                source: AgenticSourceProvenance {
                    format: "self-authored".into(),
                    digest: "detokenization-agentic".into(),
                },
            },
            vec![AgenticMooncakeRow {
                request_id: "root".into(),
                play_id: "play".into(),
                session_id: "root".into(),
                model: "synthetic-model".into(),
                input_length: Some(8),
                output_length: Some(2),
                hash_ids: Some(vec![1, 2]),
                not_before_ms: 0.0,
                ..Default::default()
            }],
        )
        .unwrap();
        let driver = WorkloadDriver::new_agentic_trace_with_lanes(graph, 4, 1).unwrap();
        let report = Replayer::new(
            spec(Backend::Vllm, disagg, None),
            ReplayEngineFactory::new(),
        )
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Workload(driver))
        .with_detokenization(1, 0.5, 0.25)
        .unwrap()
        .run()
        .unwrap();
        assert_eq!(
            report.request_counts.completed_requests, 1,
            "disagg={disagg}"
        );
        let record = &report.per_request[0];
        assert_eq!(Some(record.terminal_time_ms), record.last_token_ms);
    }
}

#[test]
fn base_time_occupies_the_detokenizer() {
    // Each job takes 1.5 ms but the next output comes 1 ms later, so it queues.
    let requests = || VecDeque::from([request(FIRST, 8, 2, Some(0.0))]);
    let direct = run(spec(Backend::Vllm, false, None), requests(), None).unwrap();
    let report = run(
        spec(Backend::Vllm, false, None),
        requests(),
        Some((1, 1.5, 0.0)),
    )
    .unwrap();
    let (direct, queued) = (record(&direct, FIRST), record(&report, FIRST));
    let first_received = direct.first_token_ms.unwrap() + 1.5;
    assert_close(queued.first_token_ms, Some(first_received), "first");
    assert_close(
        queued.last_token_ms,
        Some(first_received.max(direct.last_token_ms.unwrap()) + 1.5),
        "last",
    );
    assert!(queued.last_token_ms.unwrap() - direct.last_token_ms.unwrap() > 1.5);
}

#[test]
fn invalid_detokenization_is_rejected() {
    let replayer =
        || Replayer::new(spec(Backend::Vllm, false, None), ReplayEngineFactory::new()).unwrap();
    for (workers, base_ms, per_output_ms) in [
        (0, 1.0, 0.0),
        (1, -1.0, 0.0),
        (1, 0.0, -1.0),
        (1, f64::NAN, 0.0),
        (1, 0.0, f64::INFINITY),
    ] {
        let error = replayer()
            .with_detokenization(workers, base_ms, per_output_ms)
            .err()
            .expect("invalid detokenization must be rejected");
        assert!(matches!(error, ReplayError::InvalidSpec(_)), "{error}");
    }
}
