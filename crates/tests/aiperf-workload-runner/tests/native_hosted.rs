// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Production AgentX hosted by AISim's native Replayer event loop (no
//! `drive_sim*`, no `SteppableReplay`), aggregated and disaggregated, with the
//! old clock-hosted run as the tie-ordering oracle.

use aisimulate_core::engine::{EngineConfig, TimingModelConfig};
use aisimulate_core::replay::{
    ReplayCaptureOptions, ReplayDeterminism, ReplayEngineConfig, ReplayEngineFactory,
    ReplayReport, ReplayRuntimeInput, ReplaySpec, ReplayTopology, Replayer, WorkerPoolSpec,
};
use serde_json::{Value, json};
use std::{collections::BTreeSet, rc::Rc};

#[path = "support/run_fixture.rs"]
mod run_fixture;

const IDENTITY_SEED: [u8; 32] = [41; 32];

fn native_run(frame: &str) -> aiperf_config::model::BenchmarkRun {
    aiperf_protocol::runner_v2::decode_run_request_v2(frame.as_bytes())
        .expect("native strict codec")
        .run
}

/// Same fixed 1ms/1ms single-rank model the clock-hosted oracle builds.
fn spec(topology: ReplayTopology) -> ReplaySpec {
    ReplaySpec {
        version: 1,
        encoder: None,
        topology,
        engine: serde_json::to_value(ReplayEngineConfig {
            rank: EngineConfig {
                timing_model: TimingModelConfig::Fixed { prefill_ms: 1.0, decode_ms: 1.0 },
                max_num_seqs: 64,
                enable_prefix_caching: false,
                enable_chunked_prefill: false,
                async_scheduling: Some(false),
                ..Default::default()
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
    }
}

fn disaggregated() -> ReplayTopology {
    ReplayTopology::Disaggregated {
        prefill: WorkerPoolSpec::default(),
        decode: WorkerPoolSpec::default(),
        handoff_latency_ms: 0.0,
    }
}

fn host(run: &aiperf_config::model::BenchmarkRun, topology: ReplayTopology) -> (ReplayReport, Value) {
    let (participant, handle) = aisim_aiperf_workload_runner::participant_adapter::native_hosted(
        run,
        aiperf_clock_runtime::cancellation::RunCancellation::new(),
        Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        Rc::new(aiperf_rng::identity::DeterministicIdentitySource::new(IDENTITY_SEED)),
    )
    .expect("shared native lowering hosts the production plans");
    let report = Replayer::new(spec(topology), ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::External(Box::new(participant)))
        .with_capture_options(ReplayCaptureOptions {
            capture_per_request: true,
            determinism: ReplayDeterminism::CanonicalV1,
            ..Default::default()
        })
        .run()
        .expect("native Replayer event loop hosts AIPerf");
    (report, handle.into_report().expect("AIPerf phase report"))
}

fn ns_close(ns: &Value, ms: f64) -> bool {
    (ns.as_i64().expect("ns value") as f64 / 1_000_000.0 - ms).abs() < 1e-6
}

fn ten_request_frame(directory: &std::path::Path) -> String {
    // Spacing equals fixed completion time (1ms prefill + 1ms decode), so an
    // authored arrival coincides with an engine completion instant.
    let requests = (0..10)
        .map(|i| {
            json!({"t": i as f64 * 0.002, "type":"n", "model":"model", "in":16, "out":2,
                   "hash_ids":[1,2], "api_time":0.001})
        })
        .collect::<Vec<_>>();
    let raw = json!({"id":"ten-root","models":["model"],"block_size":8,"hash_id_scope":"local","requests":requests});
    let mut native: Value = serde_json::from_str(&run_fixture::branch_request(directory)).unwrap();
    native["run"]["cfg"]["datasets"][0]["records"] = raw;
    serde_json::to_string(&native).unwrap()
}

/// Lineage, ordering, and Replayer/AIPerf clock agreement for any topology.
fn assert_agentx_branch(report: &ReplayReport, hosted: &Value) {
    let rows = hosted["requests"].as_array().unwrap();
    assert_eq!(rows.len(), 4, "two parent and two child requests");
    for row in rows {
        assert_eq!(row["status"], "settled");
        assert_eq!(row["terminal"], "completed");
        assert_eq!(row["output_token_ids"].as_array().unwrap().len(), 2);
    }
    let correlations: BTreeSet<_> =
        rows.iter().map(|row| row["request"]["correlation_id"].as_str().unwrap()).collect();
    assert_eq!(correlations.len(), 2, "parent and child are distinct sessions");
    let (parents, children): (Vec<_>, Vec<_>) = rows.iter().partition(|row| {
        row["request"]["correlation_id"] == row["request"]["root_correlation_id"]
    });
    assert_eq!((parents.len(), children.len()), (2, 2));
    for child in &children {
        assert_eq!(
            child["request"]["parent_correlation_id"],
            parents[0]["request"]["correlation_id"]
        );
        assert!(child["issue_ns"].as_i64() >= parents[0]["issue_ns"].as_i64());
    }
    for group in [&parents, &children] {
        for pair in group.windows(2) {
            assert!(
                pair[1]["issue_ns"].as_i64() >= pair[0]["end_ns"].as_i64(),
                "a session's next turn is released only after its previous response"
            );
        }
    }
    // The Replayer's own per-request time axis must agree with what AIPerf saw.
    let engine = serde_json::to_value(&report.per_request).unwrap();
    let engine = engine.as_array().unwrap();
    assert_eq!(engine.len(), 4);
    for row in rows {
        let uuid = row["request"]["uuid"].as_str().unwrap();
        let record = engine
            .iter()
            .find(|record| record["uuid"] == uuid)
            .unwrap_or_else(|| panic!("engine has no record for {uuid}"));
        assert!(ns_close(&row["start_ns"], record["arrival_time_ms"].as_f64().unwrap()));
        assert!(ns_close(&row["end_ns"], record["terminal_time_ms"].as_f64().unwrap()));
        // Token reaction: AIPerf's first-content time is the engine's first token.
        assert!(ns_close(&row["first_content_ns"], record["first_token_ms"].as_f64().unwrap()));
        assert!(ns_close(
            &json!(row["first_ttft_ns"].as_i64().unwrap()),
            record["ttft_ms"].as_f64().unwrap()
        ));
    }
}

#[test]
fn native_replayer_hosts_production_agentx_parent_child_aggregated() {
    let directory = tempfile::tempdir().unwrap();
    let run = native_run(&run_fixture::branch_request(directory.path()));
    let (report, hosted) = host(&run, ReplayTopology::aggregated(1));
    assert_agentx_branch(&report, &hosted);
}

#[test]
fn native_replayer_hosts_production_agentx_parent_child_disaggregated() {
    let directory = tempfile::tempdir().unwrap();
    let run = native_run(&run_fixture::branch_request(directory.path()));
    let (report, hosted) = host(&run, disaggregated());
    assert_agentx_branch(&report, &hosted);
}

/// Tie oracle: the clock-hosted run (`drive_sim_with_source`: clock events and
/// ready futures before an equal-time engine pass) against the native loop
/// (participant timer before completions at T). Any divergence refutes exact
/// tie acceptance and is the finding, not a test to loosen.
fn timeline(rows: &[Value]) -> Vec<Value> {
    rows.iter()
        .map(|row| {
            json!({
                "correlation": row["request"]["correlation_id"],
                "turn": row["request"]["turn_index"],
                "issue_ns": row["issue_ns"],
                "start_ns": row["start_ns"],
                "first_content_ns": row["first_content_ns"],
                "end_ns": row["end_ns"],
                "output": row["output_token_ids"],
            })
        })
        .collect()
}

fn assert_tie_parity(frame: &str, label: &str) {
    let run = native_run(frame);
    let old = aisim_aiperf_workload_runner::execute_with_identity_seed(
        &run,
        aiperf_clock_runtime::cancellation::RunCancellation::new(),
        Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        IDENTITY_SEED,
        None,
    )
    .expect("clock-hosted oracle");
    let (_, native) = host(&run, ReplayTopology::aggregated(1));
    assert_eq!(
        timeline(native["requests"].as_array().unwrap()),
        timeline(old["requests"].as_array().unwrap()),
        "{label}: native loop diverged from the clock-hosted oracle"
    );
}

#[test]
fn tie_parity_four_request_branch_before_completion() {
    let directory = tempfile::tempdir().unwrap();
    assert_tie_parity(&run_fixture::branch_request(directory.path()), "4-request branch");
}

#[test]
fn tie_parity_ten_request_equal_instant_arrivals() {
    let directory = tempfile::tempdir().unwrap();
    assert_tie_parity(&ten_request_frame(directory.path()), "10-request equal-instant");
}
