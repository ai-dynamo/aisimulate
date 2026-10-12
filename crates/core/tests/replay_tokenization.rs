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
    ReplayEngineFactory, ReplayError, ReplayReport, ReplayRuntimeInput, ReplaySpec, ReplayTopology,
    Replayer, WorkerPoolSpec,
};
use uuid::Uuid;

const LONG: u128 = 1;
const SHORT: u128 = 2;
const BACKENDS: [Backend; 3] = [Backend::Vllm, Backend::Sglang, Backend::Trtllm];

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

fn spec(
    backend: Backend,
    disagg: bool,
    workers: usize,
    max_in_flight: Option<usize>,
) -> ReplaySpec {
    let pool = WorkerPoolSpec {
        initial_workers: workers,
        ..WorkerPoolSpec::default()
    };
    ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
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

fn request(id: u128, input_tokens: u32, arrival_timestamp_ms: Option<f64>) -> DirectRequest {
    DirectRequest {
        tokens: (0..input_tokens)
            .map(|token| token + id as u32 * 1000)
            .collect(),
        max_output_tokens: 2,
        uuid: Some(Uuid::from_u128(id)),
        arrival_timestamp_ms,
        ..Default::default()
    }
}

/// A long prompt authored before a short one at the same timestamp.
fn burst() -> VecDeque<DirectRequest> {
    VecDeque::from([request(LONG, 64, Some(0.0)), request(SHORT, 4, Some(0.0))])
}

fn run(
    spec: ReplaySpec,
    requests: VecDeque<DirectRequest>,
    tokenization: Option<(usize, f64, f64)>,
) -> Result<ReplayReport, ReplayError> {
    let mut replayer = Replayer::new(spec, ReplayEngineFactory::new())?
        .with_runtime_input(ReplayRuntimeInput::Requests(requests));
    if let Some((workers, base_ms, per_input_token_ms)) = tokenization {
        replayer = replayer.with_tokenization(workers, base_ms, per_input_token_ms)?;
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

#[test]
fn same_timestamp_requests_are_admitted_in_authored_order_without_tokenization() {
    for backend in BACKENDS {
        for disagg in [false, true] {
            let report = run(spec(backend, disagg, 1, None), burst(), None).unwrap();
            let (long, short) = (record(&report, LONG), record(&report, SHORT));
            assert_eq!(long.tokenized_ms, None);
            assert_eq!(short.tokenized_ms, None);
            assert_eq!(long.first_admit_ms, Some(0.0));
            assert!(
                short.first_admit_ms >= long.first_admit_ms,
                "{backend:?} disagg={disagg}"
            );
        }
    }
}

#[test]
fn one_tokenizer_serves_a_burst_in_arrival_order() {
    for backend in BACKENDS {
        for disagg in [false, true] {
            let report = run(spec(backend, disagg, 1, None), burst(), Some((1, 1.0, 0.5))).unwrap();
            assert_eq!(report.request_counts.completed_requests, 2);
            let (long, short) = (record(&report, LONG), record(&report, SHORT));
            // 1 + 0.5 * 64 = 33 ms, then the short request: 33 + 1 + 0.5 * 4 = 36 ms.
            assert_eq!(long.tokenized_ms, Some(33.0));
            assert_eq!(short.tokenized_ms, Some(36.0));
            assert_eq!(
                long.first_admit_ms,
                Some(33.0),
                "{backend:?} disagg={disagg}"
            );
            assert!(
                short.first_admit_ms.unwrap() >= 36.0,
                "{backend:?} disagg={disagg}"
            );
            // Latency metrics keep the authored arrival.
            assert_eq!(short.arrival_time_ms, 0.0);
            assert!(short.ttft_ms.unwrap() > 36.0);
        }
    }
}

#[test]
fn spare_tokenizers_let_shorter_prompts_reach_the_engine_first() {
    for backend in BACKENDS {
        for disagg in [false, true] {
            let report = run(spec(backend, disagg, 1, None), burst(), Some((2, 1.0, 0.5))).unwrap();
            let (long, short) = (record(&report, LONG), record(&report, SHORT));
            assert_eq!(long.tokenized_ms, Some(33.0));
            assert_eq!(short.tokenized_ms, Some(3.0));
            assert_eq!(
                short.first_admit_ms,
                Some(3.0),
                "{backend:?} disagg={disagg}"
            );
            assert_eq!(
                long.first_admit_ms,
                Some(33.0),
                "{backend:?} disagg={disagg}"
            );
        }
    }
}

#[test]
fn each_worker_has_its_own_tokenizer() {
    for disagg in [false, true] {
        // Round-robin sends the long and short requests to different workers.
        let report = run(
            spec(Backend::Vllm, disagg, 2, None),
            burst(),
            Some((1, 1.0, 0.5)),
        )
        .unwrap();
        let (long, short) = (record(&report, LONG), record(&report, SHORT));
        assert_eq!(long.tokenized_ms, Some(33.0));
        assert_eq!(short.tokenized_ms, Some(3.0));
    }
}

#[test]
fn zero_tokenization_cost_keeps_direct_dispatch() {
    for disagg in [false, true] {
        let direct = run(spec(Backend::Vllm, disagg, 1, None), burst(), None).unwrap();
        let zero = run(
            spec(Backend::Vllm, disagg, 1, None),
            burst(),
            Some((1, 0.0, 0.0)),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&direct.per_request).unwrap(),
            serde_json::to_value(&zero.per_request).unwrap()
        );
    }
}

#[test]
fn requests_being_tokenized_hold_a_concurrency_slot() {
    for disagg in [false, true] {
        let requests = VecDeque::from([request(LONG, 8, None), request(SHORT, 8, None)]);
        let report = run(
            spec(Backend::Vllm, disagg, 1, Some(1)),
            requests,
            Some((1, 5.0, 0.0)),
        )
        .unwrap();
        assert_eq!(report.request_counts.completed_requests, 2);
        let (first, second) = (record(&report, LONG), record(&report, SHORT));
        assert_eq!(first.tokenized_ms, Some(5.0));
        assert_eq!(second.arrival_time_ms, first.terminal_time_ms);
        assert_eq!(second.tokenized_ms, Some(first.terminal_time_ms + 5.0));
    }
}

#[test]
fn invalid_tokenization_is_rejected() {
    for (workers, base_ms, per_input_token_ms) in [
        (0, 1.0, 0.0),
        (1, -1.0, 0.0),
        (1, 0.0, -1.0),
        (1, f64::NAN, 0.0),
        (1, 0.0, f64::INFINITY),
    ] {
        let error = Replayer::new(
            spec(Backend::Vllm, false, 1, None),
            ReplayEngineFactory::new(),
        )
        .unwrap()
        .with_tokenization(workers, base_ms, per_input_token_ms)
        .err()
        .expect("invalid tokenization must be rejected");
        assert!(matches!(error, ReplayError::InvalidSpec(_)), "{error}");
    }
}

#[test]
fn attention_dp_ranks_of_a_worker_share_its_tokenizer() {
    let mut spec = spec(Backend::Vllm, false, 1, None);
    spec.engine = serde_json::to_value(ReplayEngineConfig {
        dp_size: 2,
        ..engine(Backend::Vllm)
    })
    .unwrap();
    let report = run(spec, burst(), Some((1, 1.0, 0.5))).unwrap();
    let (long, short) = (record(&report, LONG), record(&report, SHORT));
    // Placement spreads the burst across both ranks of the one logical worker,
    // which still has a single tokenizer server.
    assert_ne!(
        long.routing_history[0].dp_rank,
        short.routing_history[0].dp_rank
    );
    assert_eq!(long.tokenized_ms, Some(33.0));
    assert_eq!(short.tokenized_ms, Some(36.0));
}

#[test]
fn agentic_workloads_are_rejected() {
    let row = AgenticMooncakeRow {
        request_id: "root".into(),
        play_id: "play".into(),
        session_id: "session".into(),
        model: "synthetic-model".into(),
        input_length: Some(8),
        output_length: Some(1),
        hash_ids: Some(vec![1, 2]),
        ..Default::default()
    };
    let graph = ValidatedAgenticGraph::from_agentic_mooncake_rows(
        AgenticMooncakeHeader {
            schema: AGENTIC_MOONCAKE_SCHEMA.into(),
            version: AGENTIC_MOONCAKE_VERSION,
            block_size: 4,
            hash_id_scope: AgenticHashIdScope::Local,
            source: AgenticSourceProvenance {
                format: "self-authored".into(),
                digest: "tokenization-agentic-rejection".into(),
            },
        },
        vec![row],
    )
    .unwrap();
    let driver = WorkloadDriver::new_agentic_trace_with_lanes(graph, 4, 1).unwrap();
    let error = Replayer::new(
        spec(Backend::Vllm, false, 1, None),
        ReplayEngineFactory::new(),
    )
    .unwrap()
    .with_runtime_input(ReplayRuntimeInput::Workload(driver))
    .with_tokenization(1, 1.0, 0.0)
    .unwrap()
    .run()
    .expect_err("agentic tokenization must be rejected");
    assert!(matches!(error, ReplayError::InvalidSpec(_)), "{error}");
    assert!(error.to_string().contains("agentic"), "{error}");
}
