// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Production AgentX hosted by AISim's native Replayer event loop (no
//! `drive_sim*`, no `SteppableReplay`), aggregated and disaggregated, with the
//! old clock-hosted run as the tie-ordering oracle.

use aiperf_clock_runtime::cancellation::RunCancellation;
use aisimulate_core::engine::{EngineConfig, TimingModelConfig};
use aisimulate_core::replay::{
    ParticipantStep, ReplayCaptureOptions, ReplayDeterminism, ReplayEngineConfig,
    ReplayEngineFactory, ReplayReport, ReplayRuntimeInput, ReplaySpec, ReplayTerminalStatus,
    ReplayTopology, Replayer, WorkerPoolSpec, WorkloadParticipant,
};
use serde_json::{Value, json};
use std::{cell::RefCell, collections::BTreeSet, rc::Rc};
use uuid::Uuid;

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
                timing_model: TimingModelConfig::Fixed {
                    prefill_ms: 1.0,
                    decode_ms: 1.0,
                },
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

fn host(
    run: &aiperf_config::model::BenchmarkRun,
    topology: ReplayTopology,
) -> (ReplayReport, Value) {
    let (participant, handle) = aisim_aiperf_workload_runner::participant_adapter::native_hosted(
        run,
        aiperf_clock_runtime::cancellation::RunCancellation::new(),
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
    let correlations: BTreeSet<_> = rows
        .iter()
        .map(|row| row["request"]["correlation_id"].as_str().unwrap())
        .collect();
    assert_eq!(
        correlations.len(),
        2,
        "parent and child are distinct sessions"
    );
    let (parents, children): (Vec<_>, Vec<_>) = rows
        .iter()
        .partition(|row| row["request"]["correlation_id"] == row["request"]["root_correlation_id"]);
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
        assert!(ns_close(
            &row["start_ns"],
            record["arrival_time_ms"].as_f64().unwrap()
        ));
        assert!(ns_close(
            &row["end_ns"],
            record["terminal_time_ms"].as_f64().unwrap()
        ));
        // Token reaction: AIPerf's first-content time is the engine's first token.
        assert!(ns_close(
            &row["first_content_ns"],
            record["first_token_ms"].as_f64().unwrap()
        ));
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

/// Exact JSON comparison, including IEEE-754 bits rather than approximate or
/// numeric JSON equality. Every key and array member must be present on both sides.
fn assert_exact(actual: &Value, expected: &Value, path: &str) -> usize {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) if a.is_f64() && b.is_f64() => {
            assert_eq!(
                a.as_f64().unwrap().to_bits(),
                b.as_f64().unwrap().to_bits(),
                "{path}"
            );
            1
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path}: population");
            a.iter()
                .zip(b)
                .enumerate()
                .map(|(index, (a, b))| assert_exact(a, b, &format!("{path}/{index}")))
                .sum()
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(
                a.keys().collect::<Vec<_>>(),
                b.keys().collect::<Vec<_>>(),
                "{path}: keys"
            );
            a.iter()
                .map(|(key, a)| assert_exact(a, &b[key], &format!("{path}/{key}")))
                .sum()
        }
        _ => {
            assert_eq!(actual, expected, "{path}");
            0
        }
    }
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
    let (backend, native) = host(&run, ReplayTopology::aggregated(1));
    println!("{label} CLOCK_HOSTED_RAW {old}");
    println!("{label} NATIVE_HOSTED_RAW {native}");
    println!(
        "{label} NATIVE_BACKEND_RAW {}",
        serde_json::to_value(&backend).unwrap()
    );
    println!(
        "{label} NATIVE_BACKEND_REQUESTS {}",
        serde_json::to_value(&backend.per_request).unwrap()
    );
    assert!(!native["requests"].as_array().unwrap().is_empty());
    assert_exact(
        &native["requests"],
        &old["requests"],
        &format!("{label}/requests"),
    );
    assert_exact(
        &native["native_phase_report"],
        &old["native_phase_report"],
        &format!("{label}/native_phase_report"),
    );
    assert_exact(
        &native["virtual_end_ns"],
        &old["virtual_end_ns"],
        &format!("{label}/end"),
    );
    // The native Replayer measures host CPU processing time while the existing
    // clock-hosted proof stamps virtual time. Keep the raw witness above; only
    // the three explicitly host-processing fields are outside physical parity.
    let mut physical = serde_json::to_value(&backend).unwrap();
    let mut old_physical = old["backend_report"].clone();
    for key in [
        "wall_time_ms",
        "processed_tokens_per_s",
        "processed_output_tokens_per_s",
    ] {
        assert!(
            physical.as_object_mut().unwrap().remove(key).is_some(),
            "{key}"
        );
        assert!(
            old_physical.as_object_mut().unwrap().remove(key).is_some(),
            "{key}"
        );
    }
    let mut float_pairs = assert_exact(&physical, &old_physical, &format!("{label}/backend"));
    float_pairs += assert_exact(
        &serde_json::to_value(&backend.per_request).unwrap(),
        &old["backend_per_request"],
        &format!("{label}/backend_per_request"),
    );
    assert!(
        float_pairs > 0,
        "physical comparison must consume actual floats"
    );
    println!("{label} EXACT_PHYSICAL_FLOAT_PAIRS {float_pairs}");
}

#[test]
fn tie_parity_four_request_branch_before_completion() {
    let directory = tempfile::tempdir().unwrap();
    assert_tie_parity(
        &run_fixture::branch_request(directory.path()),
        "4-request branch",
    );
}

#[test]
fn tie_parity_ten_request_equal_instant_arrivals() {
    let directory = tempfile::tempdir().unwrap();
    assert_tie_parity(
        &ten_request_frame(directory.path()),
        "10-request equal-instant",
    );
}

fn join_frame(directory: &std::path::Path, child_output: usize) -> String {
    let mut frame = run_fixture::request(directory);
    frame["run"]["cfg"]["datasets"][0]["records"] = json!({
        "id":"join-proof", "models":["model"], "block_size":8, "hash_id_scope":"local",
        "requests":[
            {"t":0.0,"type":"n","model":"model","in":16,"out":2,"hash_ids":[1,2],"api_time":0.002},
            {"t":0.0,"type":"subagent","agent_id":"child-proof","subagent_type":"Explore",
             "duration_ms":3,"status":"completed","models":["model"],"requests":[
                {"t":0.0,"type":"n","model":"model","in":16,"out":child_output,"hash_ids":[4,5],"api_time":0.003}
             ]},
            {"t":0.102,"type":"n","model":"model","in":24,"out":2,"hash_ids":[1,2,3],"api_time":0.002}
        ]
    });
    serde_json::to_string(&frame).unwrap()
}

fn assert_join(child_output: usize, expected_join_ns: i64) {
    let directory = tempfile::tempdir().unwrap();
    let run = native_run(&join_frame(directory.path(), child_output));
    let (report, hosted) = host(&run, ReplayTopology::aggregated(1));
    println!("JOIN child_output={child_output} RAW {hosted}");
    let rows = hosted["requests"].as_array().unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(report.per_request.len(), 3);
    let (parents, children): (Vec<_>, Vec<_>) = rows
        .iter()
        .partition(|row| row["request"]["correlation_id"] == row["request"]["root_correlation_id"]);
    assert_eq!((parents.len(), children.len()), (2, 1));
    assert_eq!(
        children[0]["request"]["parent_correlation_id"],
        parents[0]["request"]["correlation_id"]
    );
    assert_eq!(parents[0]["issue_ns"], 0);
    assert_eq!(parents[0]["end_ns"], 2_000_000);
    assert_eq!(children[0]["issue_ns"], 0);
    assert_eq!(children[0]["end_ns"], child_output as i64 * 1_000_000);
    assert_eq!(parents[1]["issue_ns"], expected_join_ns);
    assert_eq!(parents[1]["end_ns"], expected_join_ns + 2_000_000);
    for row in rows {
        assert_eq!(row["status"], "settled");
        assert_eq!(row["terminal"], "completed");
        let backend = report
            .per_request
            .iter()
            .find(|record| record.uuid == row["request"]["uuid"].as_str().unwrap())
            .unwrap();
        assert!(ns_close(&row["issue_ns"], backend.arrival_time_ms));
        assert!(ns_close(&row["end_ns"], backend.terminal_time_ms));
        assert!(ns_close(
            &row["first_content_ns"],
            backend.first_token_ms.unwrap()
        ));
    }
}

#[test]
fn native_agentx_join_preserves_recorded_deadline_after_fast_child() {
    assert_join(3, 102_000_000);
}

#[test]
fn native_agentx_join_waits_for_live_child_after_recorded_deadline() {
    assert_join(210, 210_000_000);
}

#[derive(Default)]
struct ParticipantFacts {
    submits: Vec<(Uuid, f64)>,
    cancels: Vec<(Uuid, f64)>,
    tokens: Vec<(Uuid, f64)>,
    terminals: Vec<(Uuid, f64, ReplayTerminalStatus)>,
    quiescent: Vec<(Uuid, f64)>,
}

/// A host-owned cancellation deadline, forwarding the actual AgentX participant.
/// This owns neither workload policy nor physical advancement.
struct CancelAt {
    inner: aisim_aiperf_workload_runner::participant_adapter::AgentXParticipant,
    cancel: RunCancellation,
    at_ms: f64,
    fired: bool,
    facts: Rc<RefCell<ParticipantFacts>>,
}

impl WorkloadParticipant for CancelAt {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        let inner = self.inner.next_wakeup_ms();
        if self.fired {
            inner
        } else {
            Some(inner.map_or(self.at_ms, |time| time.min(self.at_ms)))
        }
    }
    fn advance_to(&mut self, now_ms: f64, in_flight: usize) -> anyhow::Result<ParticipantStep> {
        if !self.fired && now_ms >= self.at_ms {
            self.fired = true;
            self.cancel.cancel();
        }
        let step = self.inner.advance_to(now_ms, in_flight)?;
        let mut facts = self.facts.borrow_mut();
        facts.submits.extend(
            step.submits
                .iter()
                .map(|request| (request.uuid.unwrap(), now_ms)),
        );
        facts
            .cancels
            .extend(step.cancels.iter().map(|id| (*id, now_ms)));
        Ok(step)
    }
    fn on_output_token(&mut self, id: Uuid, token: u32, now_ms: f64) -> anyhow::Result<()> {
        self.facts.borrow_mut().tokens.push((id, now_ms));
        self.inner.on_output_token(id, token, now_ms)
    }
    fn on_terminal(
        &mut self,
        id: Uuid,
        now_ms: f64,
        status: ReplayTerminalStatus,
    ) -> anyhow::Result<()> {
        self.facts.borrow_mut().terminals.push((id, now_ms, status));
        self.inner.on_terminal(id, now_ms, status)
    }
    fn on_quiescent(&mut self, id: Uuid, now_ms: f64) -> anyhow::Result<()> {
        self.facts.borrow_mut().quiescent.push((id, now_ms));
        self.inner.on_quiescent(id, now_ms)
    }
    fn is_drained(&self) -> bool {
        self.inner.is_drained()
    }
    fn total_requests(&self) -> usize {
        self.inner.total_requests()
    }
}

fn assert_native_cancellation(at_ms: f64) {
    let directory = tempfile::tempdir().unwrap();
    let run = native_run(&join_frame(directory.path(), 3));
    let cancel = RunCancellation::new();
    let (inner, handle) = aisim_aiperf_workload_runner::participant_adapter::native_hosted(
        &run,
        cancel.clone(),
        Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        Rc::new(aiperf_rng::identity::DeterministicIdentitySource::new(
            IDENTITY_SEED,
        )),
    )
    .unwrap();
    let facts = Rc::new(RefCell::new(ParticipantFacts::default()));
    let participant = CancelAt {
        inner,
        cancel,
        at_ms,
        fired: false,
        facts: facts.clone(),
    };
    let mut replay = spec(ReplayTopology::aggregated(1));
    let mut engine: ReplayEngineConfig = serde_json::from_value(replay.engine.clone()).unwrap();
    engine.rank.timing_model = TimingModelConfig::Fixed {
        prefill_ms: 10.0,
        decode_ms: 10.0,
    };
    replay.engine = serde_json::to_value(engine).unwrap();
    let report = Replayer::new(replay, ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::External(Box::new(participant)))
        .with_capture_options(ReplayCaptureOptions {
            capture_per_request: true,
            ..Default::default()
        })
        .run()
        .unwrap();
    let hosted = handle.into_report().unwrap();
    println!("CANCEL at_ms={at_ms} HOSTED_RAW {hosted}");
    println!(
        "CANCEL at_ms={at_ms} BACKEND_RAW {}",
        serde_json::to_value(&report.per_request).unwrap()
    );
    let facts = facts.borrow();
    assert_eq!(facts.submits.len(), 2);
    assert_eq!(facts.cancels.len(), 2);
    assert_eq!(facts.terminals.len(), 2);
    assert_eq!(facts.quiescent.len(), 2);
    assert!(
        facts.tokens.is_empty(),
        "timer-first cancellation must discard first-output tie"
    );
    for (id, submitted_at) in &facts.submits {
        assert_eq!(*submitted_at, 0.0);
        assert_eq!(
            facts
                .cancels
                .iter()
                .filter(|(candidate, time)| candidate == id && *time == at_ms)
                .count(),
            1
        );
        assert_eq!(
            facts
                .terminals
                .iter()
                .filter(|(candidate, time, status)| candidate == id
                    && *time == at_ms
                    && *status == ReplayTerminalStatus::Canceled)
                .count(),
            1
        );
    }
    assert_eq!(report.per_request.len(), 2);
    for record in &report.per_request {
        assert_eq!(
            record.first_admit_ms,
            Some(0.0),
            "cancellation must occur inside an admitted pass"
        );
        assert_eq!(record.terminal_status, ReplayTerminalStatus::Canceled);
        assert_eq!(record.terminal_time_ms, at_ms);
        assert_eq!(record.output_length, 0);
        assert_eq!(record.first_token_ms, None);
    }
    let rows = hosted["requests"].as_array().unwrap();
    assert_eq!(rows.len(), 2, "cancel must prevent the parent join");
    for row in rows {
        assert_eq!(row["issue_ns"], 0);
        match row["status"].as_str().unwrap() {
            "future_dropped" => assert_eq!(row["drop_ns"], (at_ms * 1_000_000.0) as i64),
            "cancelled_none" => assert_eq!(row["return_ns"], (at_ms * 1_000_000.0) as i64),
            status => panic!("cancelled request fabricated settlement: {status}"),
        }
    }
    let phases = hosted["native_phase_report"]["phases"].as_array().unwrap();
    assert_eq!(phases.len(), 1);
    let phase = &phases[0];
    assert_eq!(phase["was_cancelled"], true);
    for key in [
        "requests_completed",
        "in_flight_requests",
        "pending_branch_work",
    ] {
        assert_eq!(phase[key], 0, "{key}");
    }
    assert_eq!(phase["forced_completion"], false);
    assert_eq!(phase["cancel_drain_timeout_triggered"], false);
}

#[test]
fn native_agentx_cancels_in_flight_requests_inside_committed_pass() {
    assert_native_cancellation(4.0);
}

#[test]
fn native_agentx_cancellation_timer_wins_first_output_tie() {
    assert_native_cancellation(10.0);
}
