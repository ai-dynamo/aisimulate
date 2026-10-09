// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Functional AgentX qualification through exported native APIs. Fixed timing
//! makes these lifecycle checks reproducible; it does not qualify benchmark fidelity.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use aisimulate_core::engine::{Backend, EngineConfig, NativeHostOffloadConfig, TimingModelConfig};
use aisimulate_core::replay::loadgen::{
    AGENTIC_MOONCAKE_SCHEMA, AGENTIC_MOONCAKE_VERSION, AgenticDependency,
    AgenticDependencyRelation, AgenticDependencyTrigger, AgenticHashIdScope,
    AgenticLifecycleEventKind, AgenticMooncakeHeader, AgenticMooncakeRow, AgenticPlayStatus,
    AgenticProfileOptions, AgenticReplayPhase, AgenticSnapshotOptions, AgenticSourceProvenance,
    PreparedAgenticSnapshots, ValidatedAgenticGraph, WekaImporter, WorkloadDriver,
    load_agentic_mooncake, load_weka_agentic_graph,
};
use aisimulate_core::replay::{
    PerRequestRecord, ReplayCaptureOptions, ReplayDeterminism, ReplayEngineConfig,
    ReplayEngineFactory, ReplayReport, ReplayRuntimeInput, ReplaySpec, ReplayTerminalStatus,
    ReplayTopology, Replayer, WorkerPoolSpec,
};
use rstest::rstest;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/e2e/configs/unified_cli/fixtures/traces")
        .join(name)
}

fn materialize(source: &Path, destination: &Path) {
    let importer = WekaImporter::open(source).unwrap();
    let mut writer = BufWriter::new(File::create(destination).unwrap());
    serde_json::to_writer(&mut writer, importer.header()).unwrap();
    writeln!(writer).unwrap();
    importer
        .for_each_row(|row| {
            serde_json::to_writer(&mut writer, &row)?;
            writeln!(writer)?;
            Ok(())
        })
        .unwrap();
    writer.flush().unwrap();
}

fn run(
    graph: ValidatedAgenticGraph,
    backend: Backend,
    max_model_len: Option<usize>,
) -> ReplayReport {
    let driver = WorkloadDriver::new_agentic_trace_with_lanes(graph, 4, 1).unwrap();
    let spec = ReplaySpec {
        version: 1,
        encoder: None,
        topology: ReplayTopology::aggregated(1),
        engine: serde_json::to_value(ReplayEngineConfig {
            rank: EngineConfig {
                num_gpu_blocks: 64,
                block_size: 4,
                max_model_len,
                max_num_seqs: 4,
                max_num_batched_tokens: 64,
                aic_nextn: None,
                native_host_offload: None,
                timing_model: TimingModelConfig::Fixed {
                    prefill_ms: 2.0,
                    decode_ms: 1.0,
                },
                ..EngineConfig::for_backend(backend)
            },
            ..ReplayEngineConfig::default()
        })
        .unwrap(),
        adapters: Default::default(),
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests: Vec::new(),
    };
    Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Workload(driver))
        .with_capture_options(ReplayCaptureOptions {
            capture_per_request: true,
            determinism: ReplayDeterminism::CanonicalV1,
            ..Default::default()
        })
        .run()
        .unwrap()
}

fn assert_same_execution(expected: &ReplayReport, actual: &ReplayReport) {
    assert_eq!(actual.agentic_graph, expected.agentic_graph);
    let expected_lifecycle = expected.agentic_lifecycle.as_ref().unwrap();
    let actual_lifecycle = actual.agentic_lifecycle.as_ref().unwrap();
    assert_eq!(
        actual_lifecycle.to_jsonl().unwrap(),
        expected_lifecycle.to_jsonl().unwrap()
    );
    assert_eq!(
        actual_lifecycle.digest().unwrap(),
        expected_lifecycle.digest().unwrap()
    );
    assert_eq!(actual.agentic_play_outcomes, expected.agentic_play_outcomes);
    assert_eq!(
        serde_json::to_value(&actual.per_request).unwrap(),
        serde_json::to_value(&expected.per_request).unwrap()
    );
    // Wall time is host execution cost, not a simulated workload observation.
    assert_eq!(
        serde_json::to_value(actual.clone().with_wall_time_ms(0.0)).unwrap(),
        serde_json::to_value(expected.clone().with_wall_time_ms(0.0)).unwrap()
    );
}

fn qualify_paths(source: &Path, backend: Backend, max_model_len: Option<usize>) -> ReplayReport {
    let directory = tempfile::tempdir().unwrap();
    let materialized = directory.path().join("agentic-v2.jsonl");
    materialize(source, &materialized);
    let direct = load_weka_agentic_graph(source, Some(4)).unwrap();
    let reloaded = load_agentic_mooncake(&materialized, 4).unwrap();
    assert_eq!(direct.identity(), reloaded.identity());
    assert_eq!(direct.nodes(), reloaded.nodes());

    let expected_identity = direct.identity();
    let report = run(direct.clone(), backend, max_model_len);
    assert_eq!(report.agentic_graph.as_ref(), Some(&expected_identity));
    assert_same_execution(&report, &run(reloaded.clone(), backend, max_model_len));
    assert_same_execution(&report, &run(direct, backend, max_model_len));
    assert_same_execution(&report, &run(reloaded, backend, max_model_len));
    report
}

#[rstest]
#[case::overlapping_spawn("weka/a.json", 2, 1)]
#[case::relative_child_and_parent_resume("weka-relative.json", 4, 1)]
#[case::jsonl_one_lane_two_plays("weka-two-plays.jsonl", 3, 2)]
fn public_aggregated_weka_and_materialized_v2_have_identical_complete_lifecycles(
    #[case] name: &str,
    #[case] requests: usize,
    #[case] plays: usize,
    #[values(Backend::Vllm, Backend::Sglang)] backend: Backend,
) {
    let report = qualify_paths(&fixture(name), backend, None);
    assert_eq!(report.request_counts.num_requests, requests);
    assert_eq!(report.request_counts.completed_requests, requests);
    assert_eq!(report.per_request.len(), requests);
    assert!(report.per_request.iter().all(|request| {
        request.terminal_status == ReplayTerminalStatus::Completed
            && request.output_length == request.requested_output_length
    }));
    let outcomes = report.agentic_play_outcomes.as_ref().unwrap();
    assert_eq!(outcomes.len(), plays);
    assert!(outcomes.iter().all(|outcome| {
        outcome.status == AgenticPlayStatus::Completed && outcome.settled_at_ms.is_some()
    }));
    let lifecycle = report.agentic_lifecycle.as_ref().unwrap();
    assert_eq!(lifecycle.events.len(), requests * 3 + plays);
    assert_eq!(
        lifecycle.events[0].event,
        AgenticLifecycleEventKind::Dispatch
    );
    assert_eq!(lifecycle.events[0].at_ms, 0.0);
    assert_eq!(
        lifecycle.events.last().unwrap().event,
        AgenticLifecycleEventKind::PlayQuiescent
    );

    // A configured lane is a client session-tree slot: all requests in the
    // first play, including background children, must be terminal before reuse.
    // Resource settlement is tracked separately and need not precede dispatch.
    if plays == 2 {
        let first_terminals = lifecycle
            .events
            .iter()
            .enumerate()
            .filter(|(_, event)| {
                event.play_id == outcomes[0].play_id
                    && event.event == AgenticLifecycleEventKind::CausalTerminal
            })
            .collect::<Vec<_>>();
        assert!(!first_terminals.is_empty());
        let second_dispatch = lifecycle
            .events
            .iter()
            .position(|event| {
                event.play_id == outcomes[1].play_id
                    && event.event == AgenticLifecycleEventKind::Dispatch
            })
            .unwrap();
        assert!(
            first_terminals
                .iter()
                .all(|(index, _)| *index < second_dispatch)
        );
        assert_eq!(
            first_terminals
                .iter()
                .map(|(_, event)| event.at_ms)
                .max_by(f64::total_cmp)
                .unwrap(),
            lifecycle.events[second_dispatch].at_ms
        );
    }
}

#[test]
fn public_vllm_child_context_rejection_settles_the_play_and_skips_parent_resume() {
    // Exercise the shared context limit through a vLLM child request.
    let report = qualify_paths(&fixture("weka-relative.json"), Backend::Vllm, Some(6));
    assert_eq!(report.request_counts.num_requests, 3);
    assert_eq!(report.request_counts.completed_requests, 2);
    let outcome = &report.agentic_play_outcomes.as_ref().unwrap()[0];
    assert_eq!(outcome.status, AgenticPlayStatus::Failed);
    assert_eq!(outcome.failure_status, Some(ReplayTerminalStatus::Rejected));
    assert!(
        outcome
            .failure_request_id
            .as_ref()
            .unwrap()
            .ends_with("outer:1:inner:1")
    );
    assert!(outcome.settled_at_ms.is_some());
    let lifecycle = report.agentic_lifecycle.as_ref().unwrap();
    let skipped = lifecycle
        .events
        .iter()
        .filter(|event| event.event == AgenticLifecycleEventKind::Skipped)
        .collect::<Vec<_>>();
    assert_eq!(skipped.len(), 1);
    assert!(skipped[0].request_id.as_ref().unwrap().ends_with("outer:2"));
    assert_eq!(
        lifecycle.events.last().unwrap().event,
        AgenticLifecycleEventKind::PlayQuiescent
    );
}

// Self-authored graph: a child fills the device cache during the parent's tool
// gap, then a blocking join resumes the original prefix. The last turn changes
// every prefix hash. No source dataset or wall-clock benchmark is involved.
fn g2_graph() -> ValidatedAgenticGraph {
    let row = |id: &str, session: &str, start, hashes| AgenticMooncakeRow {
        request_id: id.into(),
        play_id: "g2-play".into(),
        session_id: session.into(),
        model: "synthetic-model".into(),
        input_length: Some(9),
        output_length: Some(1),
        hash_ids: Some(hashes),
        not_before_ms: start,
        recorded_api_time_ms: Some(2.0),
        ..Default::default()
    };
    let edge = |id: &str, relation, delay_ms| AgenticDependency {
        request_id: id.into(),
        trigger: AgenticDependencyTrigger::Completion,
        relation,
        delay_ms,
    };
    let seed = row("seed", "parent", 0.0, vec![1, 2, 3]);
    let mut child = row("evict", "child", 100.0, vec![11, 12, 13]);
    child.dependencies = vec![edge("seed", AgenticDependencyRelation::Spawn, 20.0)];
    let mut resumed = row("restore", "parent", 200.0, vec![1, 2, 3]);
    resumed.dependencies = vec![
        edge("seed", AgenticDependencyRelation::Sequence, 0.0),
        edge("evict", AgenticDependencyRelation::Join, 20.0),
    ];
    let mut busted = row("bust", "parent", 300.0, vec![21, 22, 23]);
    busted.dependencies = vec![edge("restore", AgenticDependencyRelation::Sequence, 20.0)];
    ValidatedAgenticGraph::from_agentic_mooncake_rows(
        AgenticMooncakeHeader {
            schema: AGENTIC_MOONCAKE_SCHEMA.into(),
            version: AGENTIC_MOONCAKE_VERSION,
            block_size: 4,
            hash_id_scope: AgenticHashIdScope::Local,
            source: AgenticSourceProvenance {
                format: "self-authored".into(),
                digest: "agentx-g2-fork-join-v1".into(),
            },
        },
        vec![seed, child, resumed, busted],
    )
    .unwrap()
}

fn g2_spec(disaggregated: bool, shared: bool, host_blocks: Option<usize>, h2d: f64) -> ReplaySpec {
    let rank = EngineConfig {
        // Every nine-token prompt needs all three G1 blocks. The child's
        // different prefix must evict both complete blocks of its parent.
        num_gpu_blocks: 3,
        block_size: 4,
        max_num_seqs: 1,
        max_num_batched_tokens: 16,
        kv_cache_bytes_per_token: Some(250_000),
        native_host_offload: host_blocks.map(|blocks| {
            let host = NativeHostOffloadConfig::new(blocks).with_bandwidths(1.0, h2d);
            if shared {
                host.cluster_shared("agentx-g2-tp1")
            } else {
                host
            }
        }),
        timing_model: TimingModelConfig::Fixed {
            prefill_ms: 2.0,
            decode_ms: 1.0,
        },
        ..EngineConfig::for_backend(Backend::Vllm)
    };
    ReplaySpec {
        encoder: None,
        version: 1,
        topology: if disaggregated {
            ReplayTopology::Disaggregated {
                prefill: WorkerPoolSpec::default(),
                decode: WorkerPoolSpec::default(),
                handoff_latency_ms: 1.0,
            }
        } else {
            ReplayTopology::aggregated(1)
        },
        engine: serde_json::to_value(ReplayEngineConfig {
            rank,
            ..Default::default()
        })
        .unwrap(),
        adapters: Default::default(),
        max_sim_time_ms: Some(10_000.0),
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests: Vec::new(),
    }
}

fn run_g2(spec: ReplaySpec, driver: WorkloadDriver) -> ReplayReport {
    Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Workload(driver))
        .with_capture_options(ReplayCaptureOptions {
            capture_per_request: true,
            determinism: ReplayDeterminism::CanonicalV1,
            ..Default::default()
        })
        .run()
        .unwrap()
}

fn g2_request<'a>(report: &'a ReplayReport, id: &str) -> &'a PerRequestRecord {
    report
        .per_request
        .iter()
        .find(|row| row.request_id.as_deref() == Some(id))
        .unwrap()
}

#[rstest]
fn agentx_g2_restores_evicted_parent_after_tool_gap_and_join(
    #[values(false, true)] disaggregated: bool,
    #[values(false, true)] shared: bool,
) {
    let run = |capacity, bandwidth| {
        run_g2(
            g2_spec(disaggregated, shared, capacity, bandwidth),
            WorkloadDriver::new_agentic_trace_with_lanes(g2_graph(), 4, 1).unwrap(),
        )
    };
    let hbm = run(None, 1.0);
    let offload = run(Some(8), 1.0);
    let slow = run(Some(8), 0.01);
    let too_small = run(Some(1), 1.0);
    for report in [&hbm, &offload, &slow, &too_small] {
        assert_eq!(report.request_counts.completed_requests, 4);
        assert_eq!(
            report.agentic_play_outcomes.as_ref().unwrap()[0].status,
            AgenticPlayStatus::Completed
        );
        let restored = g2_request(report, "restore");
        let child = g2_request(report, "evict");
        assert!(restored.dispatched_at_ms.unwrap() >= child.terminal_time_ms + 20.0);
        assert_eq!(g2_request(report, "bust").reused_input_tokens, 0);
    }
    let restored = g2_request(&offload, "restore");
    assert_eq!(restored.first_admission_g1_reused_input_tokens, Some(0));
    assert_eq!(restored.first_admission_host_reused_input_tokens, Some(8));
    assert_eq!(g2_request(&hbm, "restore").reused_input_tokens, 0);
    assert_eq!(
        g2_request(&too_small, "restore").first_admission_host_reused_input_tokens,
        Some(0)
    );
    assert!(offload.committed_prefill_tokens < hbm.committed_prefill_tokens);
    let admission_wait = |report: &ReplayReport| {
        let request = g2_request(report, "restore");
        request.first_admit_ms.unwrap() - request.dispatched_at_ms.unwrap()
    };
    // Two physical 1 MB blocks travel back from G2. A slower link must delay
    // actual admission, not merely change the declared cache-hit ratio.
    assert!(admission_wait(&slow) >= admission_wait(&offload) + 100.0);
    if shared {
        assert_eq!(offload.g2_domains.len(), 1);
        assert_eq!(offload.g2_domains[0].capacity_blocks, 8);
        assert!(offload.g2_domains[0].used_blocks <= 8);
    } else {
        assert!(offload.g2_domains.is_empty());
    }
    assert_same_execution(&offload, &run(Some(8), 1.0));
}

fn g2_snapshot_driver(warmup: bool, from_start: bool) -> WorkloadDriver {
    let prepared = g2_graph()
        .prepare_snapshots(1, AgenticSnapshotOptions { seed: 42 })
        .unwrap();
    let play = if from_start {
        prepared.context().prepare_play_from_start(0, 0).unwrap()
    } else {
        prepared.context().prepare_play(0, 0, Some(50.0)).unwrap()
    };
    let snapshots = PreparedAgenticSnapshots::from_plays(vec![play]).unwrap();
    if warmup {
        WorkloadDriver::new_agentic_warmup(snapshots, 4, true, 1.0).unwrap()
    } else {
        WorkloadDriver::new_agentic_snapshots(snapshots, 4, true, 1.0).unwrap()
    }
}

#[rstest]
fn agentx_g2_warmup_retains_host_state_without_measuring_preparation(
    #[values(false, true)] disaggregated: bool,
    #[values(false, true)] shared: bool,
) {
    let mut spec = g2_spec(disaggregated, shared, Some(8), 1.0);
    let mut config: ReplayEngineConfig = serde_json::from_value(spec.engine.clone()).unwrap();
    // The first primer's D2H remains in flight after the short warmup passes.
    config
        .rank
        .native_host_offload
        .as_mut()
        .unwrap()
        .d2h_bandwidth_gbps = 0.01;
    spec.engine = serde_json::to_value(config).unwrap();
    let cold = run_g2(spec.clone(), g2_snapshot_driver(false, false));
    let warm = run_g2(spec, g2_snapshot_driver(true, false));
    let phases = warm.agentic_phases.as_ref().unwrap();
    assert_eq!(phases.phase, AgenticReplayPhase::Profile);
    assert_eq!(phases.requests.len(), 11);
    assert_eq!(phases.lanes[0].warmup_completed, 10);
    let origin = phases.profile_start_ms.unwrap();
    let settled = phases
        .requests
        .iter()
        .filter_map(|row| row.quiescent_at_ms)
        .max_by(f64::total_cmp)
        .unwrap();
    assert!(
        origin > settled,
        "the barrier must wait for the outstanding D2H: {phases:?}"
    );
    assert_eq!(warm.request_counts.completed_requests, 3);
    assert!(
        warm.per_request
            .iter()
            .all(|row| row.agentic_phase == Some(AgenticReplayPhase::Profile))
    );
    assert_eq!(
        g2_request(&warm, "restore").first_admission_g1_reused_input_tokens,
        Some(0)
    );
    assert_eq!(
        g2_request(&warm, "restore").first_admission_host_reused_input_tokens,
        Some(8)
    );
    assert_eq!(
        g2_request(&cold, "restore").first_admission_host_reused_input_tokens,
        Some(0)
    );
    assert!(warm.committed_prefill_tokens < cold.committed_prefill_tokens);
    assert!(
        warm.per_request
            .iter()
            .all(|row| row.arrival_time_ms >= 0.0)
    );
}

#[rstest]
fn agentx_g2_recycled_plays_bust_cache_identity_and_close_admission_at_cutoff(
    #[values(false, true)] disaggregated: bool,
    #[values(false, true)] shared: bool,
) {
    let mut driver = g2_snapshot_driver(false, true);
    driver
        .enable_agentic_profile(AgenticProfileOptions {
            duration_seconds: 0.7,
            response_grace_seconds: 0.0,
            ..Default::default()
        })
        .unwrap();
    let mut spec = g2_spec(disaggregated, shared, Some(64), 0.01);
    spec.max_sim_time_ms = None;
    let report = run_g2(spec, driver);
    let profile = report.agentic_profile.as_ref().unwrap();
    assert!(profile.plays_started >= 2, "{profile:?}");
    assert_eq!(profile.client_in_flight_requests, 0);
    assert!(profile.admission_closed);
    let seeds = report
        .per_request
        .iter()
        .filter(|row| row.request_id.as_deref() == Some("seed"))
        .collect::<Vec<_>>();
    assert!(seeds.len() >= 2);
    for seed in seeds {
        assert_eq!(seed.first_admission_host_reused_input_tokens, Some(0));
        assert_eq!(seed.first_admission_g1_reused_input_tokens, Some(0));
    }
    assert!(
        report
            .per_request
            .iter()
            .all(|row| row.dispatched_at_ms.unwrap() < profile.admission_cutoff_ms.unwrap())
    );
}

#[rstest]
fn agentx_g2_profile_cancels_pending_host_restore_and_releases_server_resources(
    #[values(false, true)] disaggregated: bool,
    #[values(false, true)] shared: bool,
) {
    let mut driver = g2_snapshot_driver(false, true);
    driver
        .enable_agentic_profile(AgenticProfileOptions {
            // The restored parent arrives at 200 ms; its two-block H2D needs
            // another 200 ms. End profiling while it owns the G1 reservation.
            duration_seconds: 0.3,
            response_grace_seconds: 0.0,
            ..Default::default()
        })
        .unwrap();
    let mut spec = g2_spec(disaggregated, shared, Some(8), 0.01);
    spec.max_sim_time_ms = None;
    let report = run_g2(spec, driver);
    let profile = report.agentic_profile.as_ref().unwrap();
    assert_eq!(profile.canceled_requests, 1, "{profile:?}");
    assert_eq!(profile.client_in_flight_requests, 0);
    assert_eq!(profile.server_unsettled_requests, 0);
    assert_eq!(profile.finished_at_ms, Some(300.0));
    let restore = g2_request(&report, "restore");
    assert_eq!(restore.terminal_status, ReplayTerminalStatus::Canceled);
    assert!(restore.dispatched_at_ms.is_some());
    assert_eq!(restore.first_admit_ms, None);
    assert_eq!(restore.output_length, 0);
}
