// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hosted AgentX finite/recycle and system-idle-cap behavior not covered by
//! `native_hosted.rs` or `native_hosted_phases.rs`.
//!
//! Each frame is the production AIPerf lowering over a strict native
//! `RunRequestV2` hosted by AISim's native `Replayer`. The fixed 1 ms prefill /
//! 1 ms decode model makes every timestamp an exact integer nanosecond value,
//! so the assertions below are about actual dispatch instants rather than
//! tolerances. Wide signed WEKA hash IDs stay in `RawValue`; they are never
//! routed through `serde_json::Value`.

use aiperf_clock_runtime::cancellation::RunCancellation;
use aisimulate_core::engine::{EngineConfig, TimingModelConfig};
use aisimulate_core::replay::{
    ReplayCaptureOptions, ReplayDeterminism, ReplayEngineConfig, ReplayEngineFactory, ReplayReport,
    ReplayRuntimeInput, ReplaySpec, ReplayTopology, Replayer,
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    rc::Rc,
};

#[path = "support/run_fixture.rs"]
mod run_fixture;

const IDENTITY_SEED: [u8; 32] = [41; 32];

fn native_run(frame: &str) -> aiperf_config::model::BenchmarkRun {
    aiperf_protocol::runner_v2::decode_run_request_v2(frame.as_bytes())
        .expect("native strict codec")
        .run
}

/// Same fixed 1 ms/1 ms single-rank model the sibling candidate proofs use.
fn spec(topology: ReplayTopology) -> ReplaySpec {
    ReplaySpec {
        version: 1,
        encoder: None,
        topology,
        engine: serde_json::to_value(ReplayEngineConfig {
            rank: EngineConfig {
                timing_model: TimingModelConfig::Fixed {
                    prefill_ms: 1.0,
                    decode_ms: 1.0,
                },
                max_num_seqs: 64,
                block_size: 8,
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

fn host(
    run: &aiperf_config::model::BenchmarkRun,
    topology: ReplayTopology,
) -> (ReplayReport, Value) {
    let (participant, handle) = aisim_aiperf_workload_runner::participant_adapter::native_hosted(
        run,
        RunCancellation::new(),
        Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        Rc::new(aiperf_rng::identity::DeterministicIdentitySource::new(
            IDENTITY_SEED,
        )),
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

/// Replace the authored WEKA records on a strict frame without ever parsing
/// the wide i128 hash IDs into `Value`.
fn native_frame(
    directory: &Path,
    records: &str,
    finite_replay: bool,
    sessions: Option<u64>,
    system_idle_gap_cap_seconds: Option<f64>,
) -> String {
    let mut value = run_fixture::request(directory);
    value["run"]["cfg"]["finite_replay"] = json!(finite_replay);
    if let Some(sessions) = sessions {
        // A profiling session cap terminates a non-finite recycle run; finite
        // replay rejects authored cutoffs, so it stays absent there.
        value["run"]["cfg"]["phases"][0]["sessions"] = json!(sessions);
    }
    if let Some(cap) = system_idle_gap_cap_seconds {
        value["run"]["cfg"]["system_idle_gap_cap_seconds"] = json!(cap);
    }
    let bytes = serde_json::to_vec(&value).unwrap();
    let mut native = aiperf_protocol::runner_v2::decode_run_request_v2(&bytes)
        .expect("strict native frame before raw record substitution");
    let Some(aiperf_config::model::dataset::Dataset::File(dataset)) = native
        .run
        .cfg
        .datasets
        .as_mut()
        .and_then(|datasets| datasets.first_mut())
    else {
        panic!("file dataset fixture required");
    };
    dataset.records = Some(serde_json::value::RawValue::from_string(records.into()).unwrap());
    serde_json::to_string(&aiperf_protocol::runner_v2::RunRequestV2 {
        protocol_version: 2,
        run: native,
    })
    .unwrap()
}

/// Two distinct roots, each with two sequential parent turns and one spawned
/// child request. The wide signed hashes make the raw source text observable.
const RECYCLE_RECORDS: &str = r#"[
{"id":"tree-alpha","models":["model"],"block_size":8,"hash_id_scope":"local","requests":[
 {"t":0.0,"type":"n","model":"model","in":16,"out":2,"hash_ids":[184467440737095516170,184467440737095516171],"api_time":0.001},
 {"t":0.001,"type":"subagent","agent_id":"child-alpha","subagent_type":"Explore","duration_ms":5,"status":"completed","models":["model"],"requests":[
   {"t":0.001,"type":"n","model":"model","in":16,"out":2,"hash_ids":[-184467440737095516170,-184467440737095516171],"api_time":0.001}]},
 {"t":0.05,"type":"n","model":"model","in":24,"out":2,"hash_ids":[184467440737095516170,184467440737095516171,184467440737095516172],"api_time":0.001}]},
{"id":"tree-beta","models":["model"],"block_size":8,"hash_id_scope":"local","requests":[
 {"t":0.0,"type":"n","model":"model","in":16,"out":2,"hash_ids":[184467440737095516175,184467440737095516176],"api_time":0.001},
 {"t":0.001,"type":"subagent","agent_id":"child-beta","subagent_type":"Explore","duration_ms":5,"status":"completed","models":["model"],"requests":[
   {"t":0.001,"type":"n","model":"model","in":16,"out":2,"hash_ids":[-184467440737095516175,-184467440737095516176],"api_time":0.001}]},
 {"t":0.05,"type":"n","model":"model","in":24,"out":2,"hash_ids":[184467440737095516175,184467440737095516176,184467440737095516177],"api_time":0.001}]}
]"#;

/// One root with a 10 s recorded gap between its two turns. Without the system
/// idle cap the second turn waits for the recorded instant; with it, the
/// pending replay timer is pulled forward.
const IDLE_RECORDS: &str = r#"{"id":"idle-root","models":["model"],"block_size":8,"hash_id_scope":"local","requests":[
 {"t":0.0,"type":"n","model":"model","in":16,"out":2,"hash_ids":[184467440737095516170,184467440737095516171],"api_time":0.001},
 {"t":10.0,"type":"n","model":"model","in":24,"out":2,"hash_ids":[184467440737095516170,184467440737095516171,184467440737095516172],"api_time":0.001}
]}"#;

fn is_root_row(row: &Value) -> bool {
    row["request"]["turn_index"] == 0
        && row["request"]["correlation_id"] == row["request"]["root_correlation_id"]
}

fn root_issue(tree: &[&Value]) -> i64 {
    for &row in tree {
        if is_root_row(row) {
            return row["issue_ns"].as_i64().unwrap();
        }
    }
    panic!("tree has no root turn");
}

fn child_end(tree: &[&Value]) -> i64 {
    for &row in tree {
        if row["request"]["correlation_id"] != row["request"]["root_correlation_id"] {
            return row["end_ns"].as_i64().unwrap();
        }
    }
    panic!("tree has no subagent request");
}

fn tree_max_end(tree: &[&Value]) -> i64 {
    let mut max = i64::MIN;
    for &row in tree {
        max = max.max(row["end_ns"].as_i64().unwrap());
    }
    max
}

fn tree_max_issue(tree: &[&Value]) -> i64 {
    let mut max = i64::MIN;
    for &row in tree {
        max = max.max(row["issue_ns"].as_i64().unwrap());
    }
    max
}

/// A lane recycles only after every member of its current tree — including the
/// spawned child — is terminal. Both roots run once, then the authored session
/// stop refuses a third draw, so the request population is exact.
#[test]
fn native_recycle_waits_for_the_whole_tree_before_the_next_root() {
    let directory = tempfile::tempdir().unwrap();
    let frame = native_frame(directory.path(), RECYCLE_RECORDS, false, Some(2), None);
    let run = native_run(&frame);
    let (report, hosted) = host(&run, ReplayTopology::aggregated(1));
    println!("RECYCLE_RAW {hosted}");

    let rows = hosted["requests"].as_array().unwrap();
    assert_eq!(rows.len(), 6, "two trees x (two parent turns + one child)");
    assert_eq!(report.per_request.len(), 6);
    assert_eq!(
        hosted["native_phase_report"]["phases"][1]["requests_sent"], 6,
        "the sessions stop accepts exactly two whole trees"
    );

    let mut tree_ids = BTreeSet::new();
    let mut trees = BTreeMap::<String, Vec<&Value>>::new();
    for row in rows {
        assert_eq!(row["status"], "settled");
        assert_eq!(row["terminal"], "completed");
        let root = row["request"]["root_correlation_id"]
            .as_str()
            .unwrap()
            .to_owned();
        tree_ids.insert(root.clone());
        trees.entry(root).or_default().push(row);
    }
    assert_eq!(tree_ids.len(), 2, "exactly two distinct replayed roots");
    assert_eq!(trees.len(), 2);

    let mut ordered: Vec<&Vec<&Value>> = trees.values().collect();
    ordered.sort_by_key(|tree| root_issue(tree.as_slice()));
    let first = *ordered[0];
    let second = *ordered[1];

    for tree in [first, second] {
        assert_eq!(
            tree.len(),
            3,
            "whole-tree population per root: two parent turns and one child"
        );
        let parents = tree
            .iter()
            .filter(|row| {
                (**row)["request"]["correlation_id"] == (**row)["request"]["root_correlation_id"]
            })
            .count();
        assert_eq!(parents, 2);
        assert_eq!(tree.len() - parents, 1, "one spawned subagent request");
    }

    // Tree alpha: parent turn 0 ends at 2 ms, the child at 3 ms, and parent
    // turn 1 issues 50 ms after that return and ends at 54 ms. The recycled
    // root must not issue at either of the earlier boundaries.
    let first_child_end = child_end(first);
    let first_root_issue = root_issue(first);
    let first_last_issue = tree_max_issue(first);
    let first_max_end = tree_max_end(first);
    let second_root_issue = root_issue(second);
    assert!(second_root_issue > first_root_issue);
    assert!(
        second_root_issue > first_child_end,
        "recycle must wait for the spawned child, not only the parent's first turn"
    );
    assert!(
        second_root_issue > first_last_issue,
        "recycle must not start while the parent's second turn is still in flight"
    );
    assert_eq!(
        second_root_issue, first_max_end,
        "the drained tree is replaced at the drain instant (zero-delay recycle)"
    );
}

/// The phase-wide system idle cap pulls a pending replay timer to
/// `idle_start + cap` without rewriting the trace. Turn 0 ends at 2 ms; a
/// 250 ms cap therefore issues the recorded-at-10 s turn at 252 ms exactly.
#[test]
fn native_system_idle_cap_pulls_the_next_turn_to_the_cap_deadline() {
    let directory = tempfile::tempdir().unwrap();
    let frame = native_frame(directory.path(), IDLE_RECORDS, true, None, Some(0.25));
    let run = native_run(&frame);
    let (report, hosted) = host(&run, ReplayTopology::aggregated(1));
    println!("IDLE_CAP_RAW {hosted}");

    let rows = hosted["requests"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(report.per_request.len(), 2);
    for row in rows {
        assert_eq!(row["status"], "settled");
        assert_eq!(row["terminal"], "completed");
    }
    assert_eq!(rows[0]["issue_ns"], 0_i64);
    assert_eq!(
        rows[0]["end_ns"], 2_000_000_i64,
        "1 ms prefill + 1 ms decode"
    );
    assert_eq!(
        rows[1]["issue_ns"], 252_000_000_i64,
        "idle starts at the 2 ms terminal; 2 ms + the 250 ms cap"
    );
    assert_eq!(rows[1]["end_ns"], 254_000_000_i64);

    // Comparison control: the identical frame without the cap waits for the
    // recorded 10 s instant. The old clock-hosted path is used only for this
    // baseline; all behavioral claims above come from the native host.
    let control_run = native_run(&native_frame(
        directory.path(),
        IDLE_RECORDS,
        true,
        None,
        None,
    ));
    let control = aisim_aiperf_workload_runner::execute_with_identity_seed(
        &control_run,
        RunCancellation::new(),
        Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        IDENTITY_SEED,
        None,
    )
    .expect("clock-hosted uncapped control");
    let control_rows = control["requests"].as_array().unwrap();
    assert_eq!(control_rows.len(), 2);
    assert_eq!(
        control_rows[1]["issue_ns"], 10_002_000_000_i64,
        "without the cap the turn keeps its recorded 10 s end-to-start delay after the 2 ms return"
    );
}
