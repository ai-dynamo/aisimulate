// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Candidate-path phase behavior not covered by `native_hosted.rs`: a nonzero
//! warmup primer handing off to profiling, byte-exact sticky cache-bust markers,
//! and prompt-cache reuse reaching the hosted participant.
//!
//! Every fixture is the production AIPerf lowering over a strict native frame
//! (`RunRequestV2` -> `factory_workload_input_resolved`) hosted by AISim's
//! native `Replayer`; no simulator workload generator and no private scheduler
//! is involved. Timing is the fixed 1 ms prefill / 1 ms decode model.

use aiperf_clock_runtime::cancellation::RunCancellation;
use aiperf_exchange_model::cache_bust::{CacheBustTarget, build_cache_bust_marker};
use aisimulate_core::engine::{EngineConfig, TimingModelConfig};
use aisimulate_core::replay::{
    ParticipantStep, ReplayCaptureOptions, ReplayDeterminism, ReplayEngineConfig,
    ReplayEngineFactory, ReplayReport, ReplayRuntimeInput, ReplaySpec, ReplayTerminalStatus,
    ReplayTopology, Replayer, WorkloadParticipant,
};
use serde_json::{Value, json};
use std::{cell::RefCell, collections::BTreeSet, path::Path, rc::Rc};
use uuid::Uuid;

#[path = "support/run_fixture.rs"]
mod run_fixture;

const IDENTITY_SEED: [u8; 32] = [41; 32];
const BENCHMARK_ID: &str = "independent-proof";
const ROOT_TRACE_ID: &str = "branch-root";

fn native_run(frame: &str) -> aiperf_config::model::BenchmarkRun {
    aiperf_protocol::runner_v2::decode_run_request_v2(frame.as_bytes())
        .expect("native strict codec")
        .run
}

/// Same fixed 1 ms/1 ms single-rank model the other candidate proofs build.
fn spec(topology: ReplayTopology, enable_prefix_caching: bool) -> ReplaySpec {
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
                enable_prefix_caching,
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
    let report = Replayer::new(spec(topology, false), ReplayEngineFactory::new())
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

/// A linear two-turn trace made non-finite so t* can slice it: a 0.5 start
/// ratio selects the turn at 5 ms, leaving turn 0 as the warmup primer and
/// turn 1 as the profiling resume point.
fn warmup_frame(directory: &Path) -> String {
    let mut native = run_fixture::request(directory);
    native["run"]["cfg"]["finite_replay"] = json!(false);
    native["run"]["cfg"]["trajectory_start_min_ratio"] = json!(0.5);
    native["run"]["cfg"]["trajectory_start_max_ratio"] = json!(0.5);
    // A non-finite pass needs its own stop; one session leaves the two-turn
    // trace intact instead of recycling another root.
    native["run"]["cfg"]["phases"][0]["sessions"] = json!(1);
    serde_json::to_string(&native).unwrap()
}

/// The existing branch fixture, with the production marker policy enabled.
fn cache_bust_frame(directory: &Path) -> String {
    let mut native: Value = serde_json::from_str(&run_fixture::branch_request(directory)).unwrap();
    native["run"]["cfg"]["datasets"][0]["synthesis"]["cache_bust_target"] =
        json!("first_turn_prefix");
    serde_json::to_string(&native).unwrap()
}

/// Forwarding participant that retains the prompt-cache-read facts AIPerf's
/// `Execution` observer receives from the Replayer.
struct CacheReadRecorder {
    inner: aisim_aiperf_workload_runner::participant_adapter::AgentXParticipant,
    reads: Rc<RefCell<Vec<(Uuid, usize, f64)>>>,
}

impl WorkloadParticipant for CacheReadRecorder {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        self.inner.next_wakeup_ms()
    }
    fn advance_to(&mut self, now_ms: f64, in_flight: usize) -> anyhow::Result<ParticipantStep> {
        self.inner.advance_to(now_ms, in_flight)
    }
    fn on_output_token(&mut self, id: Uuid, token: u32, now_ms: f64) -> anyhow::Result<()> {
        self.inner.on_output_token(id, token, now_ms)
    }
    fn on_terminal(
        &mut self,
        id: Uuid,
        now_ms: f64,
        status: ReplayTerminalStatus,
    ) -> anyhow::Result<()> {
        self.inner.on_terminal(id, now_ms, status)
    }
    fn on_prompt_cache_read_tokens(
        &mut self,
        id: Uuid,
        tokens: usize,
        now_ms: f64,
    ) -> anyhow::Result<()> {
        self.reads.borrow_mut().push((id, tokens, now_ms));
        self.inner.on_prompt_cache_read_tokens(id, tokens, now_ms)
    }
    fn on_quiescent(&mut self, id: Uuid, now_ms: f64) -> anyhow::Result<()> {
        self.inner.on_quiescent(id, now_ms)
    }
    fn is_drained(&self) -> bool {
        self.inner.is_drained()
    }
    fn total_requests(&self) -> usize {
        self.inner.total_requests()
    }
}

/// A nonzero warmup primer must actually dispatch, drain before profiling, and
/// resume the same trajectory rather than restarting it cold.
#[test]
fn native_warmup_primer_hands_off_to_profiling() {
    let directory = tempfile::tempdir().unwrap();
    let run = native_run(&warmup_frame(directory.path()));
    let (report, hosted) = host(&run, ReplayTopology::aggregated(1));
    println!("WARMUP_HANDOFF_RAW {hosted}");

    assert_eq!(report.per_request.len(), 2);
    let rows = hosted["requests"].as_array().unwrap();
    assert_eq!(rows.len(), 2, "one warmup primer and one profiling turn");

    let phases = hosted["native_phase_report"]["phases"].as_array().unwrap();
    assert_eq!(phases.len(), 2);
    assert_eq!(phases[0]["kind"], "warmup");
    assert_eq!(phases[0]["requests_sent"], 1);
    assert_eq!(phases[0]["requests_completed"], 1);
    assert_eq!(phases[1]["kind"], "profiling");
    assert_eq!(phases[1]["requests_sent"], 1);
    assert_eq!(phases[1]["requests_completed"], 1);

    let warmup = rows
        .iter()
        .find(|row| row["request"]["credit_phase"] == "warmup")
        .expect("warmup primer row");
    let profiling = rows
        .iter()
        .find(|row| row["request"]["credit_phase"] == "profiling")
        .expect("profiling resume row");

    // The primer is the pre-t* turn; profiling resumes at the next turn of the
    // same conversation and root correlation, not a new cold root.
    assert_eq!(warmup["request"]["turn_index"], 0);
    assert_eq!(profiling["request"]["turn_index"], 1);
    assert_eq!(
        profiling["request"]["conversation_id"],
        warmup["request"]["conversation_id"]
    );
    assert_eq!(
        profiling["request"]["root_correlation_id"],
        warmup["request"]["root_correlation_id"]
    );
    // Warmup turns are capped at one output token; profiling keeps the
    // authored cap. The phase barrier means the primer settles first.
    assert_eq!(warmup["request"]["max_output_tokens"], 1);
    assert_eq!(profiling["request"]["max_output_tokens"], 2);
    assert!(
        warmup["end_ns"].as_i64().unwrap() <= profiling["issue_ns"].as_i64().unwrap(),
        "profiling must not issue before the warmup phase drains"
    );
    // Full-history dispatch resumes from the primer's first message.
    assert_eq!(
        profiling["request"]["body"]["messages"][0],
        warmup["request"]["body"]["messages"][0]
    );
    for row in rows {
        assert_eq!(row["status"], "settled");
        assert_eq!(row["terminal"], "completed");
    }
}

/// The minted marker is one exact string per trajectory tree, stamped on every
/// turn (including the subagent branch), and placed as the byte prefix of the
/// first user message rather than in a re-derived or re-tokenized form.
#[test]
fn native_cache_bust_marker_is_byte_exact_and_sticky_across_the_tree() {
    let directory = tempfile::tempdir().unwrap();
    let run = native_run(&cache_bust_frame(directory.path()));
    let (report, hosted) = host(&run, ReplayTopology::aggregated(1));
    println!("CACHE_BUST_STICKY_RAW {hosted}");

    // First pass over `branch-root`, lane 0: the same arguments the production
    // `mint_marker` uses for a cold finite selection.
    let expected = build_cache_bust_marker(
        BENCHMARK_ID,
        0,
        0,
        ROOT_TRACE_ID,
        CacheBustTarget::FirstTurnPrefix,
    )
    .expect("first_turn_prefix always mints a marker");
    assert!(expected.starts_with("[rid:"));

    assert_eq!(report.per_request.len(), 4);
    let rows = hosted["requests"].as_array().unwrap();
    assert_eq!(rows.len(), 4, "two parent and two child requests");

    let mut markers = BTreeSet::new();
    let mut roots = BTreeSet::new();
    for row in rows {
        assert_eq!(row["request"]["cache_bust_target"], "first_turn_prefix");
        assert_eq!(
            row["request"]["cache_bust_marker"], expected,
            "every tree turn carries the exact minted bytes"
        );
        markers.insert(
            row["request"]["cache_bust_marker"]
                .as_str()
                .unwrap()
                .to_string(),
        );
        roots.insert(
            row["request"]["root_correlation_id"]
                .as_str()
                .unwrap()
                .to_string(),
        );
        let messages = row["request"]["body"]["messages"]
            .as_array()
            .expect("prepared body messages");
        assert!(
            messages.iter().any(|message| message["content"]
                .as_str()
                .is_some_and(|content| content.starts_with(expected.as_str()))),
            "marker must be the byte prefix of a message: {row}"
        );
        assert_eq!(row["status"], "settled");
    }
    assert_eq!(markers, BTreeSet::from([expected]));
    assert_eq!(roots.len(), 1, "parent and child share one trajectory tree");
}

/// With prefix caching enabled, the second turn reuses the first turn's leading
/// blocks; the engine's reuse record and the prompt-cache-read fact delivered
/// to the hosted participant must agree.
#[test]
fn native_prompt_cache_reuse_reaches_the_hosted_participant() {
    let directory = tempfile::tempdir().unwrap();
    let run = native_run(&serde_json::to_string(&run_fixture::request(directory.path())).unwrap());
    let reads = Rc::new(RefCell::new(Vec::new()));
    let (participant, handle) = aisim_aiperf_workload_runner::participant_adapter::native_hosted(
        &run,
        RunCancellation::new(),
        Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        Rc::new(aiperf_rng::identity::DeterministicIdentitySource::new(
            IDENTITY_SEED,
        )),
    )
    .expect("shared native lowering hosts the production plans");
    let participant = CacheReadRecorder {
        inner: participant,
        reads: reads.clone(),
    };
    let report = Replayer::new(
        spec(ReplayTopology::aggregated(1), true),
        ReplayEngineFactory::new(),
    )
    .unwrap()
    .with_runtime_input(ReplayRuntimeInput::External(Box::new(participant)))
    .with_capture_options(ReplayCaptureOptions {
        capture_per_request: true,
        determinism: ReplayDeterminism::CanonicalV1,
        ..Default::default()
    })
    .run()
    .expect("native Replayer event loop hosts AIPerf");
    let hosted = handle.into_report().expect("AIPerf phase report");
    println!("PREFIX_CACHE_RAW {hosted}");

    assert_eq!(report.per_request.len(), 2);
    let reused: Vec<_> = report
        .per_request
        .iter()
        .filter(|record| record.first_admission_reused_input_tokens > 0)
        .collect();
    assert_eq!(
        reused.len(),
        1,
        "the second turn reuses the first turn's leading blocks"
    );
    let reused = reused[0];
    assert!(reused.first_admission_reused_input_tokens > 0);

    // A zero-token cache-read signal is not reuse; only positive reuse proves
    // the prefix-cache path.
    let reads = reads.borrow();
    let positive: Vec<_> = reads.iter().filter(|(_, tokens, _)| *tokens > 0).collect();
    assert_eq!(
        positive.len(),
        1,
        "one positive prompt-cache-read fact reaches the hosted participant"
    );
    assert_eq!(positive[0].0.to_string(), reused.uuid);
    assert_eq!(positive[0].1, reused.first_admission_reused_input_tokens);
}
