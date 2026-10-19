// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use aisimulate_core::engine::{Backend, EngineConfig, TimingModelConfig};
use aisimulate_core::replay::loadgen::{
    AGENTIC_MOONCAKE_SCHEMA, AGENTIC_MOONCAKE_VERSION, AgenticHashIdScope, AgenticMooncakeHeader,
    AgenticMooncakeRow, AgenticProfileOptions, AgenticSnapshotOptions, AgenticSourceProvenance,
    PreparedAgenticSnapshots, ValidatedAgenticGraph, WorkloadDriver,
};
use aisimulate_core::replay::{
    CURRENT_REPLAY_SPEC_VERSION, DirectRequest, PerRequestRecord, ReplayEngineConfig,
    ReplayEngineFactory, ReplayError, ReplayReport, ReplayRuntimeInput, ReplaySpec,
    ReplayTerminalStatus, ReplayTopology, Replayer, WorkerPoolSpec,
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

fn agentic_row(id: &str, input_blocks: u64, not_before_ms: f64) -> AgenticMooncakeRow {
    AgenticMooncakeRow {
        request_id: id.into(),
        play_id: "play".into(),
        session_id: id.into(),
        model: "synthetic-model".into(),
        input_length: Some(input_blocks as usize * 4),
        output_length: Some(2),
        hash_ids: Some(
            (0..input_blocks)
                .map(|block| block + input_blocks * 1000)
                .collect(),
        ),
        not_before_ms,
        ..Default::default()
    }
}

fn agentic_driver(
    rows: Vec<AgenticMooncakeRow>,
    profile: Option<AgenticProfileOptions>,
    warmup: bool,
) -> WorkloadDriver {
    let graph = ValidatedAgenticGraph::from_agentic_mooncake_rows(
        AgenticMooncakeHeader {
            schema: AGENTIC_MOONCAKE_SCHEMA.into(),
            version: AGENTIC_MOONCAKE_VERSION,
            block_size: 4,
            hash_id_scope: AgenticHashIdScope::Local,
            source: AgenticSourceProvenance {
                format: "self-authored".into(),
                digest: "tokenization-agentic".into(),
            },
        },
        rows,
    )
    .unwrap();
    let Some(profile) = profile else {
        return WorkloadDriver::new_agentic_trace_with_lanes(graph, 4, 1).unwrap();
    };
    let prepared = graph
        .prepare_snapshots(1, AgenticSnapshotOptions { seed: 42 })
        .unwrap();
    let play = prepared.context().prepare_play_from_start(0, 0).unwrap();
    let snapshots = PreparedAgenticSnapshots::from_plays(vec![play]).unwrap();
    let mut driver = if warmup {
        WorkloadDriver::new_agentic_warmup(snapshots, 4, true, 1.0).unwrap()
    } else {
        WorkloadDriver::new_agentic_snapshots(snapshots, 4, true, 1.0).unwrap()
    };
    driver.enable_agentic_profile(profile).unwrap();
    driver
}

fn run_agentic(
    spec: ReplaySpec,
    driver: WorkloadDriver,
    base_ms: f64,
    per_input_token_ms: f64,
) -> ReplayReport {
    Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Workload(driver))
        .with_tokenization(1, base_ms, per_input_token_ms)
        .unwrap()
        .run()
        .unwrap()
}

#[test]
fn agentic_requests_are_tokenized_before_the_engine() {
    for disagg in [false, true] {
        let driver = agentic_driver(vec![agentic_row("root", 2, 0.0)], None, false);
        let report = run_agentic(spec(Backend::Vllm, disagg, 1, None), driver, 5.0, 0.0);
        assert_eq!(report.request_counts.completed_requests, 1);
        let record = &report.per_request[0];
        assert_eq!(record.tokenized_ms, Some(record.arrival_time_ms + 5.0));
        assert_eq!(record.first_admit_ms, record.tokenized_ms);
    }
}

#[test]
fn an_agentic_profile_cancels_requests_still_being_tokenized() {
    for disagg in [false, true] {
        // At the 50 ms deadline "busy" holds a 200 ms prefill pass in its engine,
        // while "late" (arriving at 40 ms) is tokenized until 80 ms. Both are
        // canceled; "late" never reaches an engine.
        let driver = agentic_driver(
            vec![agentic_row("busy", 2, 0.0), agentic_row("late", 50, 40.0)],
            Some(AgenticProfileOptions {
                duration_seconds: 0.05,
                response_grace_seconds: 0.0,
                ..Default::default()
            }),
            false,
        );
        let mut spec = spec(Backend::Vllm, disagg, 1, None);
        spec.engine = serde_json::to_value(ReplayEngineConfig {
            rank: EngineConfig {
                timing_model: TimingModelConfig::Fixed {
                    prefill_ms: 200.0,
                    decode_ms: 1.0,
                },
                ..engine(Backend::Vllm).rank
            },
            ..engine(Backend::Vllm)
        })
        .unwrap();
        let report = run_agentic(spec, driver, 0.0, 0.1);
        let profile = report.agentic_profile.as_ref().unwrap();
        assert_eq!(profile.canceled_requests, 2, "{profile:?}");
        assert_eq!(profile.client_in_flight_requests, 0);
        let late = report
            .per_request
            .iter()
            .find(|record| record.request_id.as_deref() == Some("late"))
            .unwrap();
        assert_eq!(late.terminal_status, ReplayTerminalStatus::Canceled);
        assert_eq!(late.tokenized_ms, None);
        assert_eq!(late.first_admit_ms, None);
        assert_eq!(late.output_length, 0);
    }
}

#[test]
fn a_non_finite_tokenization_time_fails_the_run() {
    for disagg in [false, true] {
        let error = run(
            spec(Backend::Vllm, disagg, 1, None),
            burst(),
            Some((1, 0.0, f64::MAX)),
        )
        .expect_err("an infinite tokenization time must fail");
        assert!(error.to_string().contains("finite time"), "{error}");
    }
}

#[test]
fn agentic_warmup_waits_for_requests_being_tokenized() {
    for disagg in [false, true] {
        let driver = agentic_driver(
            vec![agentic_row("root", 2, 0.0)],
            Some(AgenticProfileOptions {
                duration_seconds: 0.1,
                ..Default::default()
            }),
            true,
        );
        let report = run_agentic(spec(Backend::Vllm, disagg, 1, None), driver, 5.0, 0.0);
        let phases = report
            .agentic_phases
            .as_ref()
            .expect("warmup records its preparation phase");
        assert!(phases.finished_at_ms.is_some(), "{phases:?}");
        assert!(phases.failure_reason.is_none(), "{phases:?}");
        // The preparation request finished only after its tokenization.
        assert!(phases.finished_at_ms.unwrap() > 5.0, "{phases:?}");
    }
}
