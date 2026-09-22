// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Functional AgentX qualification through exported native APIs. Fixed timing
//! makes these lifecycle checks reproducible; it does not qualify benchmark fidelity.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use aisimulate_core::engine::{Backend, EngineConfig, TimingModelConfig};
use aisimulate_core::replay::loadgen::{
    AgenticLifecycleEventKind, AgenticPlayStatus, ValidatedAgenticGraph, WekaImporter,
    WorkloadDriver, load_agentic_mooncake, load_weka_agentic_graph,
};
use aisimulate_core::replay::{
    ReplayCaptureOptions, ReplayDeterminism, ReplayEngineConfig, ReplayEngineFactory, ReplayReport,
    ReplayRuntimeInput, ReplaySpec, ReplayTerminalStatus, ReplayTopology, Replayer,
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
