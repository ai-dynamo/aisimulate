// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An external reactive workload drives Replay through the admission seam,
//! on aggregated and disaggregated topologies, with no Replay-authored requests.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::num::NonZeroU32;
use std::rc::Rc;

use aisimulate_core::engine::{Backend, EngineConfig, TimingModelConfig};
use aisimulate_core::replay::{
    CURRENT_REPLAY_SPEC_VERSION, DirectRequest, PerRequestRecord, ReplayCaptureOptions,
    ReplayComposition, ReplayDeterminism, ReplayEngineConfig, ReplayEngineFactory, ReplayError,
    ReplayReport, ReplayRequestContext, ReplayResult, ReplayRuntimeInput, ReplayScalingDecision,
    ReplayScalingPolicy, ReplayScalingSnapshot, ReplaySpec, ReplayTelemetrySampleKind,
    ReplayTerminalStatus, ReplayTopology, ReplayWorkloadSource, Replayer, RoundRobinComposition,
    WorkerPoolSpec, WorkerTopology,
};
use uuid::Uuid;

const SESSIONS: u128 = 3;
const TURNS: u128 = 4;
const OUTPUT_TOKENS: usize = 3;

#[derive(Default)]
struct Log {
    tokens: HashMap<Uuid, usize>,
    terminals: HashMap<Uuid, (f64, ReplayTerminalStatus)>,
    quiescent: HashMap<Uuid, f64>,
    pops: Vec<f64>,
}

/// Closed loop: a session's next turn is ready the instant its previous turn ends.
struct ClosedLoop {
    ready: Vec<(u128, u128)>,
    outstanding: HashMap<Uuid, (u128, u128)>,
    log: Rc<RefCell<Log>>,
}

fn id(session: u128, turn: u128) -> Uuid {
    Uuid::from_u128(session * 1_000 + turn + 1)
}

impl ClosedLoop {
    fn new(log: Rc<RefCell<Log>>) -> Self {
        Self {
            ready: (0..SESSIONS).map(|session| (session, 0)).collect(),
            outstanding: HashMap::new(),
            log,
        }
    }
}

impl ReplayWorkloadSource for ClosedLoop {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        // Turns become ready only in reaction to feedback, which the loop delivers
        // in the same pass it then drains; only the initial turns need a wakeup.
        (!self.ready.is_empty() && self.log.borrow().pops.is_empty()).then_some(0.0)
    }

    fn pop_ready(&mut self, now_ms: f64, _in_flight: usize) -> anyhow::Result<Vec<DirectRequest>> {
        self.log.borrow_mut().pops.push(now_ms);
        Ok(std::mem::take(&mut self.ready)
            .into_iter()
            .map(|(session, turn)| {
                let uuid = id(session, turn);
                self.outstanding.insert(uuid, (session, turn));
                DirectRequest {
                    tokens: (0..8)
                        .map(|token| (session * 100 + turn * 10 + token) as u32)
                        .collect(),
                    max_output_tokens: OUTPUT_TOKENS,
                    uuid: Some(uuid),
                    replay_context: Some(ReplayRequestContext {
                        session_id: Some(format!("s{session}")),
                        turn_index: Some(turn as usize),
                        ..Default::default()
                    }),
                    ..Default::default()
                }
            })
            .collect())
    }

    fn on_output_token(
        &mut self,
        request_id: Uuid,
        _token_id: u32,
        _now_ms: f64,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.log.borrow().terminals.contains_key(&request_id),
            "token after terminal for {request_id}"
        );
        *self.log.borrow_mut().tokens.entry(request_id).or_default() += 1;
        Ok(())
    }

    fn on_causal_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
        status: ReplayTerminalStatus,
    ) -> anyhow::Result<()> {
        let (session, turn) = self
            .outstanding
            .remove(&request_id)
            .ok_or_else(|| anyhow::anyhow!("unknown or duplicate terminal {request_id}"))?;
        self.log
            .borrow_mut()
            .terminals
            .insert(request_id, (now_ms, status));
        if turn + 1 < TURNS {
            self.ready.push((session, turn + 1));
        }
        Ok(())
    }

    fn on_quiescent(&mut self, request_id: Uuid, now_ms: f64) -> anyhow::Result<()> {
        self.log.borrow_mut().quiescent.insert(request_id, now_ms);
        Ok(())
    }

    fn is_drained(&self) -> bool {
        self.ready.is_empty() && self.outstanding.is_empty()
    }

    fn total_requests_hint(&self) -> usize {
        (SESSIONS * TURNS) as usize
    }
}

fn topologies() -> impl Iterator<Item = (Backend, bool)> {
    [Backend::Vllm, Backend::Sglang, Backend::Trtllm]
        .into_iter()
        .flat_map(|backend| [false, true].map(|disaggregated| (backend, disaggregated)))
}

fn spec_with(
    backend: Backend,
    disaggregated: bool,
    workers: usize,
    prefill_ms: f64,
    tune: impl FnOnce(&mut EngineConfig),
) -> ReplaySpec {
    let mut rank = EngineConfig::for_backend(backend);
    rank.block_size = 4;
    rank.num_gpu_blocks = 256;
    rank.max_num_seqs = 4;
    rank.max_num_batched_tokens = 64;
    rank.enable_prefix_caching = false;
    rank.timing_model = TimingModelConfig::Fixed {
        prefill_ms,
        decode_ms: 1.0,
    };
    if backend == Backend::Vllm && prefill_ms == 0.0 {
        rank.async_scheduling = Some(false);
    }
    tune(&mut rank);
    // Prefill and decode roles inherit the shared rank when left unset.
    let engine = ReplayEngineConfig {
        rank,
        ..Default::default()
    };
    let pool = WorkerPoolSpec {
        initial_workers: workers,
        ..Default::default()
    };
    ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
        encoder: None,
        topology: if disaggregated {
            ReplayTopology::Disaggregated {
                prefill: pool.clone(),
                decode: pool,
                handoff_latency_ms: 1.0,
            }
        } else {
            ReplayTopology::aggregated(workers)
        },
        engine: serde_json::to_value(engine).unwrap(),
        adapters: Default::default(),
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests: Vec::new(),
    }
}

fn spec(backend: Backend, disaggregated: bool) -> ReplaySpec {
    spec_with(backend, disaggregated, 1, 2.0, |_| {})
}

fn zero_prefill_spec(backend: Backend, disaggregated: bool) -> ReplaySpec {
    spec_with(backend, disaggregated, 1, 0.0, |_| {})
}

fn run_external(
    spec: ReplaySpec,
    source: impl ReplayWorkloadSource + 'static,
) -> ReplayResult<ReplayReport> {
    Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .run_with_source(Box::new(source))
}

fn run_native(spec: ReplaySpec, requests: impl Into<VecDeque<DirectRequest>>) -> ReplayReport {
    Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Requests(requests.into()))
        .run()
        .unwrap()
}

fn record(report: &ReplayReport, ordinal: u128) -> &PerRequestRecord {
    report
        .per_request
        .iter()
        .find(|record| record.uuid == Uuid::from_u128(ordinal).to_string())
        .unwrap()
}

fn direct_request(
    ordinal: u128,
    max_output_tokens: usize,
    arrival_ms: Option<f64>,
) -> DirectRequest {
    DirectRequest {
        tokens: vec![ordinal as u32; 8],
        uuid: Some(Uuid::from_u128(ordinal)),
        max_output_tokens,
        arrival_timestamp_ms: arrival_ms,
        ..Default::default()
    }
}

#[test]
fn external_closed_loop_reacts_at_the_same_instant_on_every_topology() {
    for (backend, disaggregated) in topologies() {
        let label = format!("{backend:?} disaggregated={disaggregated}");
        let log = Rc::new(RefCell::new(Log::default()));
        let report = run_external(spec(backend, disaggregated), ClosedLoop::new(log.clone()))
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        let log = log.borrow();
        let records: HashMap<String, _> = report
            .per_request
            .iter()
            .map(|record| (record.uuid.clone(), record))
            .collect();
        assert_eq!(records.len(), (SESSIONS * TURNS) as usize, "{label}");
        assert_eq!(log.terminals.len(), records.len(), "{label}");
        for session in 0..SESSIONS {
            for turn in 0..TURNS {
                let uuid = id(session, turn);
                let record = records[&uuid.to_string()];
                let (terminal_ms, status) = log.terminals[&uuid];
                assert_eq!(status, ReplayTerminalStatus::Completed, "{label} {uuid}");
                assert_eq!(terminal_ms, record.terminal_time_ms, "{label} {uuid}");
                assert_eq!(record.output_length, OUTPUT_TOKENS, "{label} {uuid}");
                assert_eq!(log.tokens[&uuid], record.output_length, "{label} {uuid}");
                assert!(
                    log.quiescent.contains_key(&uuid),
                    "{label} {uuid} quiescence"
                );
                if turn > 0 {
                    let previous = records[&id(session, turn - 1).to_string()];
                    assert_eq!(
                        record.arrival_time_ms, previous.terminal_time_ms,
                        "{label}: turn {turn} of session {session} must arrive at the \
                             instant its predecessor ended"
                    );
                } else {
                    assert_eq!(record.arrival_time_ms, 0.0, "{label} {uuid}");
                }
            }
        }
        assert!(
            log.pops.windows(2).all(|pair| pair[0] <= pair[1]),
            "{label}: drain instants must not regress"
        );
    }
}

#[test]
fn external_source_refuses_competing_inputs() {
    let log = Rc::new(RefCell::new(Log::default()));
    let mut with_cap = spec(Backend::Vllm, false);
    with_cap.max_in_flight = Some(2);
    let error = run_external(with_cap, ClosedLoop::new(log.clone())).unwrap_err();
    assert!(error.to_string().contains("max_in_flight"), "{error}");

    let mut with_limit = spec(Backend::Vllm, false);
    with_limit.max_sim_time_ms = Some(10.0);
    let error = run_external(with_limit, ClosedLoop::new(log)).unwrap_err();
    assert!(error.to_string().contains("max_sim_time_ms"), "{error}");
}

#[test]
fn replayer_stays_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Replayer>();
}

#[test]
fn external_source_refuses_detokenization_latency() {
    let log = Rc::new(RefCell::new(Log::default()));
    let error = Replayer::new(spec(Backend::Vllm, false), ReplayEngineFactory::new())
        .unwrap()
        .with_detokenization(1, 10.0, 0.0)
        .unwrap()
        .run_with_source(Box::new(ClosedLoop::new(log)))
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("detokenization latency"), "{message}");
    assert!(message.contains("with_detokenization"), "{message}");
}

/// Submits request 1, then request 2 as soon as request 1's first token arrives.
#[derive(Default)]
struct TokenReactive {
    started: bool,
    token_seen: bool,
    successor_emitted: bool,
    terminals: usize,
}

impl ReplayWorkloadSource for TokenReactive {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        (!self.started).then_some(0.0)
    }

    fn pop_ready(&mut self, _now_ms: f64, _in_flight: usize) -> anyhow::Result<Vec<DirectRequest>> {
        let ordinal: u128 = if !self.started {
            self.started = true;
            1
        } else if self.token_seen && !self.successor_emitted {
            self.successor_emitted = true;
            2
        } else {
            return Ok(Vec::new());
        };
        Ok(vec![token_reactive_request(ordinal, None)])
    }

    fn on_output_token(
        &mut self,
        request_id: Uuid,
        _token_id: u32,
        _now_ms: f64,
    ) -> anyhow::Result<()> {
        if request_id == Uuid::from_u128(1) {
            self.token_seen = true;
        }
        Ok(())
    }

    fn on_causal_terminal(
        &mut self,
        _request_id: Uuid,
        _now_ms: f64,
        _status: ReplayTerminalStatus,
    ) -> anyhow::Result<()> {
        self.terminals += 1;
        Ok(())
    }

    fn is_drained(&self) -> bool {
        self.terminals == 2
    }
}

fn token_reactive_request(ordinal: u128, arrival_ms: Option<f64>) -> DirectRequest {
    direct_request(ordinal, if ordinal == 1 { 3 } else { 1 }, arrival_ms)
}

fn admission_ms(report: &ReplayReport, ordinal: u128) -> (f64, f64) {
    let record = record(report, ordinal);
    let admitted = record.prefill_admit_ms.or(record.first_admit_ms).unwrap();
    (record.arrival_time_ms, admitted)
}

#[test]
fn token_triggered_submission_is_admitted_like_a_native_arrival() {
    let mut failures = Vec::new();
    for (backend, disaggregated) in topologies() {
        let external = run_external(
            zero_prefill_spec(backend, disaggregated),
            TokenReactive::default(),
        )
        .unwrap();
        let (arrival, external_admitted) = admission_ms(&external, 2);
        let first_token = record(&external, 1).first_token_ms;
        assert_eq!(Some(arrival), first_token, "{backend:?} {disaggregated}");
        let native = run_native(
            zero_prefill_spec(backend, disaggregated),
            [
                token_reactive_request(1, Some(0.0)),
                token_reactive_request(2, Some(arrival)),
            ],
        );
        let (_, native_admitted) = admission_ms(&native, 2);
        println!(
            "{backend:?} disaggregated={disaggregated}: arrival {arrival}; external admitted {external_admitted}, native admitted {native_admitted}"
        );
        if external_admitted != native_admitted {
            failures.push((backend, disaggregated, external_admitted, native_admitted));
        }
    }
    assert!(
        failures.is_empty(),
        "token-triggered submissions were admitted later than native arrivals: {failures:?}"
    );
}

#[test]
fn external_source_refuses_belady_and_a_second_request_input() {
    let log = Rc::new(RefCell::new(Log::default()));
    let mut belady = spec(Backend::Vllm, false);
    belady.engine["kv_eviction_policy"] = serde_json::json!("belady");
    let error = run_external(belady, ClosedLoop::new(log.clone())).unwrap_err();
    assert!(error.to_string().contains("belady"), "{error}");

    let mut authored = spec(Backend::Vllm, false);
    authored.requests = serde_json::from_value(serde_json::json!([{
        "id": "authored",
        "arrival_time_ms": 0.0,
        "input_tokens": 8,
        "output_tokens": 1,
    }]))
    .unwrap();
    let error = run_external(authored, ClosedLoop::new(log.clone())).unwrap_err();
    assert!(
        error.to_string().contains("another request input"),
        "{error}"
    );

    let error = Replayer::new(spec(Backend::Vllm, false), ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Requests(VecDeque::from([
            token_reactive_request(9, Some(0.0)),
        ])))
        .run_with_source(Box::new(ClosedLoop::new(log)))
        .unwrap_err();
    assert!(
        error.to_string().contains("another request input"),
        "{error}"
    );
}

/// A source that breaks one clause of the contract in a scripted way.
#[derive(Clone, Copy, Debug)]
enum Fault {
    /// Submits at 0 ms, then keeps its wakeup at the instant it was stepped.
    NonAdvancing,
    /// Submits at 5 ms, then wakes at 1 ms.
    Rewind,
    /// Submits at 0 ms, then wakes at a NaN instant.
    NonFinite,
    /// Submits at 0 ms, then wakes at an infinite instant.
    Infinite,
    /// Wakes first at -1 ms, before the run's first instant.
    InitialPast,
    /// Never wakes and never drains.
    Stall,
    /// Submits one request, then after it ends neither wakes nor drains.
    StallAfterWork,
    /// Resubmits the uuid of a request that already completed.
    ReusedUuid,
}

struct Faulty {
    fault: Fault,
    steps: usize,
    last_step_ms: f64,
    submitted: usize,
    outstanding: usize,
}

impl Faulty {
    fn new(fault: Fault) -> Self {
        Self {
            fault,
            steps: 0,
            last_step_ms: f64::NAN,
            submitted: 0,
            outstanding: 0,
        }
    }
}

impl ReplayWorkloadSource for Faulty {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        let first = self.steps == 0;
        match self.fault {
            Fault::NonAdvancing => Some(if first { 0.0 } else { self.last_step_ms }),
            Fault::Rewind => Some(if first { 5.0 } else { 1.0 }),
            Fault::NonFinite => Some(if first { 0.0 } else { f64::NAN }),
            Fault::Infinite => Some(if first { 0.0 } else { f64::INFINITY }),
            Fault::InitialPast => Some(-1.0),
            Fault::Stall => None,
            Fault::StallAfterWork | Fault::ReusedUuid => first.then_some(0.0),
        }
    }

    fn pop_ready(&mut self, now_ms: f64, _in_flight: usize) -> anyhow::Result<Vec<DirectRequest>> {
        self.steps += 1;
        self.last_step_ms = now_ms;
        // The uuid-reusing source submits its one uuid again once it ends.
        let limit = if matches!(self.fault, Fault::ReusedUuid) {
            2
        } else {
            1
        };
        if matches!(self.fault, Fault::Stall) || self.submitted == limit || self.outstanding > 0 {
            return Ok(Vec::new());
        }
        self.submitted += 1;
        self.outstanding += 1;
        Ok(vec![direct_request(1, 1, None)])
    }

    fn on_output_token(&mut self, _: Uuid, _: u32, _: f64) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_causal_terminal(
        &mut self,
        _: Uuid,
        _: f64,
        _: ReplayTerminalStatus,
    ) -> anyhow::Result<()> {
        self.outstanding -= 1;
        Ok(())
    }

    /// Every source but the stalling ones finishes cleanly once its requests
    /// complete, so a missing check yields a successful report.
    fn is_drained(&self) -> bool {
        match self.fault {
            Fault::Stall | Fault::StallAfterWork => false,
            Fault::ReusedUuid => self.submitted == 2 && self.outstanding == 0,
            _ => self.steps > 0 && self.outstanding == 0,
        }
    }
}

#[test]
fn external_source_contract_violations_fail_the_run() {
    for (fault, expected) in [
        (
            Fault::NonAdvancing,
            "returned 0 ms after a step at 0 ms; a wakeup must be None or finite and \
             strictly later",
        ),
        (Fault::Rewind, "returned 1 ms after a step at 5 ms"),
        (Fault::NonFinite, "returned NaN ms after a step at 0 ms"),
        (Fault::Infinite, "returned inf ms after a step at 0 ms"),
        (Fault::InitialPast, "returned -1 ms before the first step"),
        (
            Fault::Stall,
            "stalled at 0 ms: not drained, with nothing in flight and no wakeup",
        ),
        (Fault::StallAfterWork, "not drained, with nothing in flight"),
        (Fault::ReusedUuid, "reused request uuid"),
    ] {
        for disaggregated in [false, true] {
            let label = format!("{fault:?} disaggregated={disaggregated}");
            let error = run_external(spec(Backend::Vllm, disaggregated), Faulty::new(fault))
                .map(|report| format!("{label}: ran to a report: {:?}", report.request_counts))
                .unwrap_err();
            let message = format!("{error:#}");
            assert!(message.contains(expected), "{label}: {message}");
            assert!(
                matches!(error, ReplayError::WorkloadSource(_)),
                "{label}: a source fault must be classified as the source's: {error:?}"
            );
        }
    }
}

/// Telemetry must not sample between ready workers' pass starts.
#[test]
fn external_reaction_point_does_not_change_telemetry_samples() {
    let mut spec = zero_prefill_spec(Backend::Vllm, false);
    spec.topology = ReplayTopology::aggregated(2);
    let external = Replayer::new(spec.clone(), ReplayEngineFactory::new())
        .unwrap()
        .with_telemetry_capture(1.0)
        .unwrap()
        .run_with_source(Box::new(OpenLoop::new(
            VecDeque::from([
                direct_request(1, 1, Some(1.0)),
                direct_request(2, 1, Some(1.0)),
            ]),
            [],
        )))
        .unwrap();
    let native = Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_telemetry_capture(1.0)
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Requests(VecDeque::from([
            direct_request(1, 1, Some(1.0)),
            direct_request(2, 1, Some(1.0)),
        ])))
        .run()
        .unwrap();
    let samples = |report: &ReplayReport| {
        serde_json::to_value(&report.telemetry.as_ref().unwrap().samples).unwrap()
    };
    assert_eq!(samples(&external), samples(&native));
}

struct Failing {
    in_pop: bool,
    popped: bool,
}

impl ReplayWorkloadSource for Failing {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        (!self.popped).then_some(0.0)
    }

    fn pop_ready(&mut self, _now_ms: f64, _in_flight: usize) -> anyhow::Result<Vec<DirectRequest>> {
        anyhow::ensure!(!self.in_pop, "intentional source error");
        if self.popped {
            return Ok(Vec::new());
        }
        self.popped = true;
        Ok(vec![direct_request(1, 1, None)])
    }

    fn on_output_token(&mut self, _: Uuid, _: u32, _: f64) -> anyhow::Result<()> {
        anyhow::bail!("intentional source error")
    }

    fn on_causal_terminal(
        &mut self,
        _: Uuid,
        _: f64,
        _: ReplayTerminalStatus,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn is_drained(&self) -> bool {
        false
    }
}

#[test]
fn source_errors_are_classified_as_workload_source_failures() {
    for in_pop in [true, false] {
        for disaggregated in [false, true] {
            let label = format!("in_pop={in_pop} disaggregated={disaggregated}");
            let error = run_external(
                spec(Backend::Vllm, disaggregated),
                Failing {
                    in_pop,
                    popped: false,
                },
            )
            .unwrap_err();
            assert!(
                matches!(&error, ReplayError::WorkloadSource(message)
                    if message.contains("intentional source error")),
                "{label}: {error:?}"
            );
        }
    }
}

/// Session-tagged request for the rejection-reaction scenario. Request 1 is
/// too large for the engine's KV capacity and is rejected at admission.
fn session_request(ordinal: u128, session: &str, arrival_ms: Option<f64>) -> DirectRequest {
    DirectRequest {
        tokens: vec![ordinal as u32; if ordinal == 1 { 4_096 } else { 8 }],
        replay_context: Some(ReplayRequestContext {
            session_id: Some(session.into()),
            turn_index: Some(0),
            ..Default::default()
        }),
        ..direct_request(ordinal, 2, arrival_ms)
    }
}

/// Announces request 1 (session `a`, rejected) and request 2 (session `b`)
/// for `at_ms`, then reacts to request 1's rejection at that instant with the
/// `reactions` in session `b`, all at once or one per pop.
struct RejectionReactive {
    at_ms: f64,
    reactions: Vec<u128>,
    one_per_pop: bool,
    started: bool,
    rejected: bool,
    terminals: usize,
    /// Terminals the run ends with: requests 1 and 2 plus every reaction.
    total: usize,
}

impl RejectionReactive {
    fn new(at_ms: f64, reactions: Vec<u128>, one_per_pop: bool) -> Self {
        let total = 2 + reactions.len();
        Self {
            at_ms,
            reactions,
            one_per_pop,
            started: false,
            rejected: false,
            terminals: 0,
            total,
        }
    }
}

impl ReplayWorkloadSource for RejectionReactive {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        (!self.started).then_some(self.at_ms)
    }

    fn pop_ready(&mut self, now_ms: f64, _in_flight: usize) -> anyhow::Result<Vec<DirectRequest>> {
        if !self.started {
            if now_ms < self.at_ms {
                return Ok(Vec::new());
            }
            self.started = true;
            return Ok(vec![
                session_request(1, "a", None),
                session_request(2, "b", None),
            ]);
        }
        if !self.rejected || self.reactions.is_empty() {
            return Ok(Vec::new());
        }
        let count = if self.one_per_pop {
            1
        } else {
            self.reactions.len()
        };
        Ok(self
            .reactions
            .drain(..count)
            .map(|ordinal| session_request(ordinal, "b", None))
            .collect())
    }

    fn on_output_token(&mut self, _: Uuid, _: u32, _: f64) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_causal_terminal(
        &mut self,
        request_id: Uuid,
        _now_ms: f64,
        status: ReplayTerminalStatus,
    ) -> anyhow::Result<()> {
        if request_id == Uuid::from_u128(1) {
            anyhow::ensure!(
                status != ReplayTerminalStatus::Completed,
                "the oversized request must be rejected for this scenario"
            );
            self.rejected = true;
        }
        self.terminals += 1;
        Ok(())
    }

    fn is_drained(&self) -> bool {
        self.started && self.reactions.is_empty() && self.terminals == self.total
    }
}

fn two_worker_affinity_spec(backend: Backend, disaggregated: bool) -> ReplaySpec {
    let mut spec = spec(backend, disaggregated);
    spec.adapters.placement.config = serde_json::json!({ "session_affinity": true });
    match &mut spec.topology {
        ReplayTopology::Disaggregated { prefill, .. } => prefill.initial_workers = 2,
        topology => *topology = ReplayTopology::aggregated(2),
    }
    spec
}

/// A reaction to a rejection on one worker joins the pass another worker
/// starts at that same instant, as a native arrival would: on disaggregated
/// runs the rejection is delivered inside the prefill drive loop.
#[test]
fn rejection_reaction_joins_another_workers_same_instant_pass() {
    // (announcement instant, reactions, one reaction per pop)
    let scenarios: [(f64, &[u128], bool); 3] = [
        (0.0, &[3], false),
        // A fulfilled announcement must not make a later reaction look announced.
        (1.0, &[3], false),
        // Reactions returned one per pop each keep the hold.
        (1.0, &[3, 4], true),
    ];
    let mut failures = Vec::new();
    for (at_ms, reactions, one_per_pop) in scenarios {
        for (backend, disaggregated) in topologies() {
            let label = format!(
                "{backend:?} disaggregated={disaggregated} at={at_ms} reactions={reactions:?} \
                     one_per_pop={one_per_pop}"
            );
            let external = run_external(
                two_worker_affinity_spec(backend, disaggregated),
                RejectionReactive::new(at_ms, reactions.to_vec(), one_per_pop),
            )
            .unwrap_or_else(|error| panic!("{label}: {error}"));
            // Time-align each native successor with its external arrival so
            // the comparison isolates admission.
            let mut native_requests = vec![
                session_request(1, "a", Some(at_ms)),
                session_request(2, "b", Some(at_ms)),
            ];
            for &ordinal in reactions {
                let (arrival, _) = admission_ms(&external, ordinal);
                native_requests.push(session_request(ordinal, "b", Some(arrival)));
            }
            let native = run_native(
                two_worker_affinity_spec(backend, disaggregated),
                native_requests,
            );
            for &ordinal in reactions {
                let actual = admission_ms(&external, ordinal);
                let expected = admission_ms(&native, ordinal);
                println!("{label} request {ordinal}: external {actual:?}, native {expected:?}");
                if actual != expected {
                    failures.push((label.clone(), ordinal, actual, expected));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "rejection reactions were admitted later than native arrivals: {failures:#?}"
    );
}

/// Pre-pass dispatch exposes the rejection before the native telemetry boundary,
/// so its same-instant reaction must be included in that interval's arrivals.
#[test]
fn telemetry_counts_reactions_settled_before_its_native_drain_boundary() {
    for backend in [Backend::Vllm, Backend::Sglang, Backend::Trtllm] {
        let spec = two_worker_affinity_spec(backend, true);
        let external = Replayer::new(spec.clone(), ReplayEngineFactory::new())
            .unwrap()
            .with_telemetry_capture(1.0)
            .unwrap()
            .run_with_source(Box::new(RejectionReactive::new(1.0, vec![3], false)))
            .unwrap();
        let (arrival, _) = admission_ms(&external, 3);
        assert_eq!(arrival, 1.0, "{backend:?}: the reaction lands at 1 ms");
        let native = Replayer::new(spec, ReplayEngineFactory::new())
            .unwrap()
            .with_telemetry_capture(1.0)
            .unwrap()
            .with_runtime_input(ReplayRuntimeInput::Requests(VecDeque::from([
                session_request(1, "a", Some(1.0)),
                session_request(2, "b", Some(1.0)),
                session_request(3, "b", Some(arrival)),
            ])))
            .run()
            .unwrap();
        assert_eq!(
            admission_ms(&external, 3),
            admission_ms(&native, 3),
            "{backend:?}"
        );
        let arrivals_at_one = |report: &ReplayReport| {
            serde_json::to_value(&report.telemetry.as_ref().unwrap().samples)
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .find(|sample| sample["sampled_at_ms"] == 1.0)
                .unwrap()["traffic"]["arriving_requests"]
                .clone()
        };
        assert_eq!(arrivals_at_one(&external), 3, "{backend:?}");
        assert_eq!(arrivals_at_one(&native), 3, "{backend:?}");
    }
}

fn open_loop_spec(
    backend: Backend,
    disaggregated: bool,
    workers: usize,
    prefill_ms: f64,
    prefix_caching: bool,
) -> ReplaySpec {
    spec_with(backend, disaggregated, workers, prefill_ms, |rank| {
        rank.num_gpu_blocks = 64;
        rank.max_num_seqs = 3;
        rank.max_num_batched_tokens = 32;
        rank.enable_prefix_caching = prefix_caching;
    })
}

/// Bursts at identical instants, shared prefixes and varied lengths, so the
/// tight engine capacity queues work in placement.
fn open_loop_trace() -> VecDeque<DirectRequest> {
    let arrivals = [
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0, 5.0, 8.0,
    ];
    arrivals
        .iter()
        .enumerate()
        .map(|(index, &arrival_ms)| {
            let prefix = (index % 3) as u32 * 1_000;
            let len = 8 + (index * 5) % 20;
            DirectRequest {
                tokens: (0..len as u32)
                    .map(|token| {
                        if token < 8 {
                            prefix + token
                        } else {
                            50_000 + index as u32 * 100 + token
                        }
                    })
                    .collect(),
                max_output_tokens: 1 + (index * 7) % 6,
                uuid: Some(Uuid::from_u128(index as u128 + 1)),
                arrival_timestamp_ms: Some(arrival_ms),
                ..Default::default()
            }
        })
        .collect()
}

/// Replays a fixed open-loop trace, never reacting to feedback, optionally
/// also waking at `idle` instants with nothing to submit.
struct OpenLoop {
    pending: VecDeque<DirectRequest>,
    idle: VecDeque<f64>,
    outstanding: usize,
}

impl OpenLoop {
    fn new(pending: VecDeque<DirectRequest>, idle: impl IntoIterator<Item = f64>) -> Self {
        Self {
            pending,
            idle: idle.into_iter().collect(),
            outstanding: 0,
        }
    }
}

impl ReplayWorkloadSource for OpenLoop {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        let arrival = self
            .pending
            .front()
            .and_then(|request| request.arrival_timestamp_ms);
        match (arrival, self.idle.front().copied()) {
            (Some(arrival), Some(idle)) => Some(arrival.min(idle)),
            (arrival, idle) => arrival.or(idle),
        }
    }

    fn pop_ready(&mut self, now_ms: f64, _in_flight: usize) -> anyhow::Result<Vec<DirectRequest>> {
        while self.idle.front().is_some_and(|idle| *idle <= now_ms) {
            self.idle.pop_front();
        }
        let mut ready = Vec::new();
        while self
            .pending
            .front()
            .is_some_and(|request| request.arrival_timestamp_ms <= Some(now_ms))
        {
            ready.push(self.pending.pop_front().unwrap());
        }
        self.outstanding += ready.len();
        Ok(ready)
    }

    fn on_output_token(&mut self, _: Uuid, _: u32, _: f64) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_causal_terminal(
        &mut self,
        _: Uuid,
        _: f64,
        _: ReplayTerminalStatus,
    ) -> anyhow::Result<()> {
        self.outstanding -= 1;
        Ok(())
    }

    fn is_drained(&self) -> bool {
        self.pending.is_empty() && self.idle.is_empty() && self.outstanding == 0
    }
}

/// Round-robin placement with a scaling policy that wants a second prefill
/// worker whenever its tick sees prefill forward-pass snapshots.
#[derive(Default)]
struct FpmScaling;

impl ReplayComposition for FpmScaling {
    type Metadata = <RoundRobinComposition as ReplayComposition>::Metadata;
    type Observation = <RoundRobinComposition as ReplayComposition>::Observation;
    type AggregatedPlacement = <RoundRobinComposition as ReplayComposition>::AggregatedPlacement;
    type DisaggregatedPlacement =
        <RoundRobinComposition as ReplayComposition>::DisaggregatedPlacement;

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<Self::AggregatedPlacement> {
        RoundRobinComposition::default().create_aggregated_placement(dp_size, topology)
    }

    fn create_disaggregated_placements(
        &mut self,
        prefill_dp_size: u32,
        prefill_topology: Vec<WorkerTopology>,
        decode_dp_size: u32,
        decode_topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        RoundRobinComposition::default().create_disaggregated_placements(
            prefill_dp_size,
            prefill_topology,
            decode_dp_size,
            decode_topology,
        )
    }

    fn take_scaling_policy(&mut self) -> anyhow::Result<Option<Box<dyn ReplayScalingPolicy>>> {
        Ok(Some(Box::new(FpmScaling)))
    }
}

impl ReplayScalingPolicy for FpmScaling {
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
        Ok(1.0)
    }

    fn on_tick(
        &mut self,
        snapshot: ReplayScalingSnapshot,
    ) -> anyhow::Result<ReplayScalingDecision> {
        Ok(ReplayScalingDecision {
            target_prefill: Some(if snapshot.prefill_fpm.is_empty() {
                1
            } else {
                2
            }),
            target_decode: None,
            next_tick_ms: (snapshot.now_ms < 8.0).then_some(snapshot.now_ms + 0.5),
        })
    }
}

/// Scaling must see forward-pass snapshots at native drain boundaries.
#[test]
fn announced_work_under_scaling_matches_native() {
    let mut failures = Vec::new();
    for (backend, disaggregated) in topologies() {
        for startup_delay_ms in [0.0, 0.5, 2.0] {
            let label =
                format!("{backend:?} disaggregated={disaggregated} delay={startup_delay_ms}");
            let mut spec = spec_with(backend, disaggregated, 1, 0.0, |_| {});
            match &mut spec.topology {
                ReplayTopology::Disaggregated {
                    prefill, decode, ..
                } => {
                    prefill.startup_delay_ms = startup_delay_ms;
                    decode.startup_delay_ms = startup_delay_ms;
                }
                ReplayTopology::Aggregated { workers } => {
                    workers.startup_delay_ms = startup_delay_ms;
                }
            }
            let trace: VecDeque<_> = (1..=12)
                .map(|ordinal| {
                    direct_request(ordinal, 3, Some(if ordinal <= 2 { 1.0 } else { 1.5 }))
                })
                .collect();
            let external =
                Replayer::with_composition(spec.clone(), ReplayEngineFactory::new(), FpmScaling)
                    .unwrap()
                    .with_telemetry_capture(0.5)
                    .unwrap()
                    .run_with_source(Box::new(OpenLoop::new(trace.clone(), [])))
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
            let native = Replayer::with_composition(spec, ReplayEngineFactory::new(), FpmScaling)
                .unwrap()
                .with_telemetry_capture(0.5)
                .unwrap()
                .with_runtime_input(ReplayRuntimeInput::Requests(trace))
                .run()
                .unwrap();
            for ordinal in 1..=12 {
                let actual = serde_json::to_value(record(&external, ordinal)).unwrap();
                let expected = serde_json::to_value(record(&native, ordinal)).unwrap();
                if actual != expected {
                    failures.push(format!("{label} request {ordinal}: {actual} != {expected}"));
                    break;
                }
            }
            let samples = |report: &ReplayReport| {
                serde_json::to_value(&report.telemetry.as_ref().unwrap().samples).unwrap()
            };
            if samples(&external) != samples(&native) {
                failures.push(format!("{label}: telemetry samples differ"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "announced work diverged from native under scaling: {failures:#?}"
    );
}

#[test]
fn open_loop_trace_through_the_seam_matches_native_records() {
    let records = |report: &ReplayReport| {
        let mut records: Vec<_> = report
            .per_request
            .iter()
            .map(|record| serde_json::to_value(record).unwrap())
            .collect();
        records.sort_by_key(|record| record["uuid"].as_str().unwrap().to_string());
        records
    };
    let mut mismatches = Vec::new();
    for (backend, disaggregated) in topologies() {
        for workers in [1, 2] {
            for prefill_ms in [0.0, 2.0] {
                for prefix_caching in [false, true] {
                    let spec =
                        open_loop_spec(backend, disaggregated, workers, prefill_ms, prefix_caching);
                    let native = run_native(spec.clone(), open_loop_trace());
                    let native_records = records(&native);
                    for idle in [&[][..], &[0.5, 1.25, 3.0, 4.0, 6.5, 7.0, 9.0, 60.0]] {
                        let label = format!(
                            "{backend:?} disaggregated={disaggregated} workers={workers} \
                             prefill_ms={prefill_ms} prefix_caching={prefix_caching} \
                             idle={idle:?}"
                        );
                        let external = run_external(
                            spec.clone(),
                            OpenLoop::new(open_loop_trace(), idle.iter().copied()),
                        )
                        .unwrap_or_else(|error| panic!("{label}: {error:?}"));
                        let external_records = records(&external);
                        assert_eq!(
                            (native_records.len(), external_records.len()),
                            (open_loop_trace().len(), open_loop_trace().len()),
                            "{label}: every trace request must produce a record on both sides"
                        );
                        if native_records != external_records {
                            mismatches.push(label);
                        }
                    }
                }
            }
        }
    }
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// Canonical determinism rewrites request IDs to ordinals, but an external
/// source owns identity, so the combination is refused instead of bypassed.
#[test]
fn external_source_refuses_canonical_determinism() {
    let error = Replayer::new(spec(Backend::Vllm, false), ReplayEngineFactory::new())
        .unwrap()
        .with_capture_options(ReplayCaptureOptions {
            determinism: ReplayDeterminism::CanonicalV1,
            ..Default::default()
        })
        .run_with_source(Box::new(OpenLoop::new(
            VecDeque::from([
                direct_request(1, 1, Some(1.0)),
                direct_request(2, 1, Some(1.0)),
            ]),
            [],
        )))
        .unwrap_err();
    assert!(
        matches!(&error, ReplayError::InvalidSpec(message) if message.contains("CanonicalV1")),
        "{error:?}"
    );
}

/// Panics expose composition side effects before source validation.
struct HostileComposition;

impl ReplayComposition for HostileComposition {
    type Metadata = <RoundRobinComposition as ReplayComposition>::Metadata;
    type Observation = <RoundRobinComposition as ReplayComposition>::Observation;
    type AggregatedPlacement = <RoundRobinComposition as ReplayComposition>::AggregatedPlacement;
    type DisaggregatedPlacement =
        <RoundRobinComposition as ReplayComposition>::DisaggregatedPlacement;

    fn create_aggregated_placement(
        &mut self,
        _dp_size: u32,
        _topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<Self::AggregatedPlacement> {
        unreachable!("an invalid external-source run must not construct a runtime")
    }

    fn create_disaggregated_placements(
        &mut self,
        _prefill_dp_size: u32,
        _prefill_topology: Vec<WorkerTopology>,
        _decode_dp_size: u32,
        _decode_topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        unreachable!("an invalid external-source run must not construct a runtime")
    }

    fn set_determinism(&mut self, _determinism: ReplayDeterminism) -> ReplayResult<()> {
        panic!("an invalid external-source run must not configure its composition")
    }
}

#[test]
fn invalid_external_source_does_not_invoke_composition_determinism() {
    let mut invalid = spec(Backend::Vllm, false);
    invalid.max_in_flight = Some(1);
    let error = Replayer::with_composition(invalid, ReplayEngineFactory::new(), HostileComposition)
        .unwrap()
        .run_with_source(Box::new(OpenLoop::new(
            VecDeque::from([
                direct_request(1, 1, Some(1.0)),
                direct_request(2, 1, Some(1.0)),
            ]),
            [],
        )))
        .unwrap_err();
    assert!(
        matches!(&error, ReplayError::InvalidSpec(message) if message.contains("max_in_flight")),
        "{error:?}"
    );
}

/// Empty wakeups bracket one request and keep the source live after completion.
struct TimerDriven {
    wakeups: VecDeque<f64>,
    /// `(instant, requests submitted)` for every step.
    steps: Rc<RefCell<Vec<(f64, usize)>>>,
    submitted: bool,
    outstanding: usize,
}

impl TimerDriven {
    fn new(steps: Rc<RefCell<Vec<(f64, usize)>>>) -> Self {
        Self {
            wakeups: VecDeque::from([1.0, 2.0, 3.0, 40.0]),
            steps,
            submitted: false,
            outstanding: 0,
        }
    }
}

impl ReplayWorkloadSource for TimerDriven {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        self.wakeups.front().copied()
    }

    fn pop_ready(&mut self, now_ms: f64, _in_flight: usize) -> anyhow::Result<Vec<DirectRequest>> {
        while self.wakeups.front().is_some_and(|wakeup| *wakeup <= now_ms) {
            self.wakeups.pop_front();
        }
        let ready = if now_ms == 3.0 && !self.submitted {
            self.submitted = true;
            self.outstanding += 1;
            vec![direct_request(1, 3, None)]
        } else {
            Vec::new()
        };
        self.steps.borrow_mut().push((now_ms, ready.len()));
        Ok(ready)
    }

    fn on_output_token(&mut self, _: Uuid, _: u32, _: f64) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_causal_terminal(
        &mut self,
        _: Uuid,
        _: f64,
        _: ReplayTerminalStatus,
    ) -> anyhow::Result<()> {
        self.outstanding -= 1;
        Ok(())
    }

    fn is_drained(&self) -> bool {
        self.submitted && self.outstanding == 0 && self.wakeups.is_empty()
    }
}

#[test]
fn empty_wakeups_are_legal_and_leave_engine_timing_unchanged() {
    for (backend, disaggregated) in topologies() {
        let label = format!("{backend:?} disaggregated={disaggregated}");
        let steps = Rc::new(RefCell::new(Vec::new()));
        let external = run_external(
            spec(backend, disaggregated),
            TimerDriven::new(steps.clone()),
        )
        .unwrap_or_else(|error| panic!("{label}: {error:?}"));
        let native = run_native(
            spec(backend, disaggregated),
            [direct_request(1, 3, Some(3.0))],
        );
        assert_eq!(
            serde_json::to_value(record(&external, 1)).unwrap(),
            serde_json::to_value(record(&native, 1)).unwrap(),
            "{label}"
        );
        let steps = steps.borrow();
        let empty_wakeups: Vec<f64> = steps
            .iter()
            .filter(|(at_ms, submitted)| *submitted == 0 && [1.0, 2.0, 40.0].contains(at_ms))
            .map(|(at_ms, _)| *at_ms)
            .collect();
        assert_eq!(empty_wakeups, [1.0, 2.0, 40.0], "{label}: {steps:?}");
        assert_eq!(steps.iter().filter(|(_, n)| *n > 0).count(), 1, "{label}");
    }
}

/// A due periodic interval must not be relabeled Final when the source drains.
#[test]
fn final_empty_wakeup_emits_the_due_periodic_telemetry_sample() {
    for (backend, disaggregated) in topologies() {
        let label = format!("{backend:?} disaggregated={disaggregated}");
        let steps = Rc::new(RefCell::new(Vec::new()));
        let report = Replayer::new(spec(backend, disaggregated), ReplayEngineFactory::new())
            .unwrap()
            .with_telemetry_capture(10.0)
            .unwrap()
            .run_with_source(Box::new(TimerDriven::new(steps)))
            .unwrap_or_else(|error| panic!("{label}: {error:?}"));
        let telemetry = report.telemetry.as_ref().unwrap();
        let at_final_wakeup: Vec<_> = telemetry
            .samples
            .iter()
            .filter(|row| row.sample.sampled_at_ms == 40.0)
            .collect();
        assert_eq!(at_final_wakeup.len(), 1, "{label}");
        assert_eq!(
            at_final_wakeup[0].sample.kind,
            ReplayTelemetrySampleKind::Periodic,
            "{label}"
        );
    }
}

/// Resubmits a fresh, oversized request every time the last one is rejected,
/// all at the same instant, so the clock can never advance.
#[derive(Default)]
struct EndlessRetry {
    next: u128,
    started: bool,
    retry_pending: bool,
}

impl ReplayWorkloadSource for EndlessRetry {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        (!self.started).then_some(0.0)
    }

    fn pop_ready(&mut self, _now_ms: f64, _in_flight: usize) -> anyhow::Result<Vec<DirectRequest>> {
        if self.started && !self.retry_pending {
            return Ok(Vec::new());
        }
        self.started = true;
        self.retry_pending = false;
        self.next += 1;
        Ok(vec![DirectRequest {
            tokens: vec![1; 4_096],
            ..direct_request(self.next, 1, None)
        }])
    }

    fn on_output_token(&mut self, _: Uuid, _: u32, _: f64) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_causal_terminal(
        &mut self,
        _: Uuid,
        _: f64,
        status: ReplayTerminalStatus,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            status != ReplayTerminalStatus::Completed,
            "the oversized request must be rejected for this scenario"
        );
        self.retry_pending = true;
        Ok(())
    }

    fn is_drained(&self) -> bool {
        false
    }
}

#[test]
fn a_same_instant_reaction_loop_exhausts_the_step_budget() {
    for disaggregated in [false, true] {
        let error = Replayer::new(
            spec(Backend::Vllm, disaggregated),
            ReplayEngineFactory::new(),
        )
        .unwrap()
        .with_max_source_steps_per_instant(NonZeroU32::new(64).unwrap())
        .run_with_source(Box::new(EndlessRetry::default()))
        .unwrap_err();
        assert!(
            matches!(&error, ReplayError::ResourceLimited(message)
                if message.contains("step budget")
                    && message.contains("64 steps at instant 0 ms")),
            "disaggregated={disaggregated}: {error:?}"
        );
    }
}

struct RankedRequest {
    request: Option<DirectRequest>,
}

impl ReplayWorkloadSource for RankedRequest {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        self.request.is_some().then_some(0.0)
    }

    fn pop_ready(&mut self, _now_ms: f64, _in_flight: usize) -> anyhow::Result<Vec<DirectRequest>> {
        Ok(self.request.take().into_iter().collect())
    }

    fn on_output_token(&mut self, _: Uuid, _: u32, _: f64) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_causal_terminal(
        &mut self,
        _: Uuid,
        _: f64,
        _: ReplayTerminalStatus,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn is_drained(&self) -> bool {
        self.request.is_none()
    }
}

/// Ranks a source names are checked like spec ranks, before placement; like a
/// spec request, a prefill rank on an aggregated topology is ignored.
#[test]
fn source_request_ranks_are_validated_at_admission() {
    let ranked = |preferred: Option<u32>, prefill: Option<u32>| DirectRequest {
        preferred_dp_rank: preferred,
        preferred_prefill_dp_rank: prefill,
        ..direct_request(1, 1, None)
    };
    for (label, disaggregated, request, expected) in [
        (
            "aggregated rank out of range",
            false,
            ranked(Some(5), None),
            "rank 5 is out of range",
        ),
        (
            "prefill rank out of range",
            true,
            ranked(None, Some(5)),
            "rank 5 is out of range",
        ),
        (
            "decode rank out of range",
            true,
            ranked(Some(5), None),
            "rank 5 is out of range",
        ),
    ] {
        let error = run_external(
            spec(Backend::Vllm, disaggregated),
            RankedRequest {
                request: Some(request),
            },
        )
        .unwrap_err();
        assert!(
            matches!(&error, ReplayError::WorkloadSource(message) if message.contains(expected)),
            "{label}: {error:?}"
        );
    }
    let report = run_external(
        spec(Backend::Vllm, false),
        RankedRequest {
            request: Some(ranked(None, Some(7))),
        },
    )
    .expect("an aggregated run ignores a prefill rank, as for spec requests");
    assert_eq!(report.request_counts.completed_requests, 1);
}

/// One step: its instant, whether it had seen request 1's terminal, and the
/// ordinals it submitted.
type ProbeStep = (f64, bool, Vec<u128>);

/// Submits request 1 at 0 ms and request 2 at a wakeup at `wakeup_ms`, the
/// instant request 1 ends, and request 3 in reaction to request 1's terminal.
/// Logs, for every step, its instant and whether it had seen that terminal.
struct PhaseProbe {
    wakeup_ms: f64,
    next: VecDeque<(f64, u128)>,
    terminal_seen: bool,
    reacted: bool,
    terminals: usize,
    steps: Rc<RefCell<Vec<ProbeStep>>>,
}

impl ReplayWorkloadSource for PhaseProbe {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        self.next.front().map(|(at_ms, _)| *at_ms)
    }

    fn pop_ready(&mut self, now_ms: f64, _in_flight: usize) -> anyhow::Result<Vec<DirectRequest>> {
        let mut ordinals = Vec::new();
        while self.next.front().is_some_and(|(at_ms, _)| *at_ms <= now_ms) {
            ordinals.push(self.next.pop_front().unwrap().1);
        }
        if self.terminal_seen && !self.reacted {
            self.reacted = true;
            ordinals.push(3);
        }
        self.steps
            .borrow_mut()
            .push((now_ms, self.terminal_seen, ordinals.clone()));
        Ok(ordinals
            .into_iter()
            .map(|ordinal| direct_request(ordinal, 2, None))
            .collect())
    }

    fn on_output_token(&mut self, _: Uuid, _: u32, _: f64) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_causal_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
        _: ReplayTerminalStatus,
    ) -> anyhow::Result<()> {
        if request_id == Uuid::from_u128(1) {
            anyhow::ensure!(now_ms == self.wakeup_ms, "request 1 ended at {now_ms} ms");
            self.terminal_seen = true;
        }
        self.terminals += 1;
        Ok(())
    }

    fn is_drained(&self) -> bool {
        self.terminals == 3
    }
}

/// Coincident wakeups must submit before seeing the engine terminal; only the
/// following step can react to it.
#[test]
fn the_wakeup_step_precedes_settlement_and_reactions_follow_it() {
    for (backend, disaggregated) in topologies() {
        let label = format!("{backend:?} disaggregated={disaggregated}");
        let alone = run_native(
            spec(backend, disaggregated),
            [direct_request(1, 2, Some(0.0))],
        );
        let wakeup_ms = record(&alone, 1).terminal_time_ms;
        let steps = Rc::new(RefCell::new(Vec::new()));
        let external = run_external(
            spec(backend, disaggregated),
            PhaseProbe {
                wakeup_ms,
                next: VecDeque::from([(0.0, 1), (wakeup_ms, 2)]),
                terminal_seen: false,
                reacted: false,
                terminals: 0,
                steps: steps.clone(),
            },
        )
        .unwrap_or_else(|error| panic!("{label}: {error:?}"));
        let at_wakeup: Vec<_> = steps
            .borrow()
            .iter()
            .filter(|(at_ms, ..)| *at_ms == wakeup_ms)
            .cloned()
            .collect();
        assert_eq!(
            at_wakeup,
            [(wakeup_ms, false, vec![2]), (wakeup_ms, true, vec![3])],
            "{label}: the wakeup step precedes settlement and the reaction follows it"
        );
        let native = run_native(
            spec(backend, disaggregated),
            [
                direct_request(1, 2, Some(0.0)),
                direct_request(2, 2, Some(wakeup_ms)),
                direct_request(3, 2, Some(wakeup_ms)),
            ],
        );
        for ordinal in 1..=3 {
            assert_eq!(
                serde_json::to_value(record(&external, ordinal)).unwrap(),
                serde_json::to_value(record(&native, ordinal)).unwrap(),
                "{label} request {ordinal}"
            );
        }
    }
}

/// Two workers with session affinity: requests 1 and 2 pin session `a` to
/// worker 0 and `b` to worker 1. At 10 ms the source submits `burst`; the
/// first feedback (token or terminal) for trigger 11, on worker 0, brings a
/// reaction for worker 1, and that for trigger 21, on worker 1, a reaction
/// for worker 0. Both triggers' feedback comes from their pass starts.
struct PassStartOrder {
    burst: Vec<(u128, &'static str, usize)>,
    /// Terminals the run ends with: two pinning requests, the burst and two
    /// reactions.
    total: usize,
    steps: usize,
    terminals: usize,
    triggered: Vec<u128>,
    reactions: Vec<u128>,
    /// Every trigger, step submission and terminal, in order.
    log: Rc<RefCell<Vec<String>>>,
}

const TRIGGER_A: u128 = 11;
const TRIGGER_B: u128 = 21;
const REACTION_FOR_A: u128 = 13;
const REACTION_FOR_B: u128 = 23;

impl PassStartOrder {
    fn new(burst: Vec<(u128, &'static str, usize)>, log: Rc<RefCell<Vec<String>>>) -> Self {
        Self {
            total: burst.len() + 4,
            burst,
            steps: 0,
            terminals: 0,
            triggered: Vec::new(),
            reactions: Vec::new(),
            log,
        }
    }

    fn feedback(&mut self, ordinal: u128, now_ms: f64) {
        let reaction = match ordinal {
            TRIGGER_A => REACTION_FOR_B,
            TRIGGER_B => REACTION_FOR_A,
            _ => return,
        };
        if !self.triggered.contains(&ordinal) {
            self.triggered.push(ordinal);
            self.log
                .borrow_mut()
                .push(format!("trigger {ordinal} at {now_ms}"));
            self.reactions.push(reaction);
        }
    }
}

impl ReplayWorkloadSource for PassStartOrder {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        match self.steps {
            0 => Some(0.0),
            1 if self.terminals == 2 => Some(10.0),
            _ => None,
        }
    }

    fn pop_ready(&mut self, now_ms: f64, _in_flight: usize) -> anyhow::Result<Vec<DirectRequest>> {
        let requests: Vec<(u128, &str, usize)> = if self.steps == 0 {
            self.steps += 1;
            vec![(1, "a", 8), (2, "b", 8)]
        } else if self.steps == 1 && now_ms == 10.0 {
            self.steps += 1;
            std::mem::take(&mut self.burst)
        } else {
            self.reactions
                .drain(..)
                .map(|ordinal| {
                    let session = if ordinal == REACTION_FOR_A { "a" } else { "b" };
                    (ordinal, session, 8)
                })
                .collect()
        };
        if !requests.is_empty() {
            self.log.borrow_mut().push(format!(
                "submit {:?} at {now_ms}",
                requests
                    .iter()
                    .map(|(ordinal, ..)| *ordinal)
                    .collect::<Vec<_>>()
            ));
        }
        Ok(requests
            .into_iter()
            .map(|(ordinal, session, tokens)| DirectRequest {
                tokens: vec![ordinal as u32; tokens],
                replay_context: Some(ReplayRequestContext {
                    session_id: Some(session.into()),
                    turn_index: Some(0),
                    ..Default::default()
                }),
                ..direct_request(ordinal, 3, None)
            })
            .collect())
    }

    fn on_output_token(&mut self, request_id: Uuid, _: u32, now_ms: f64) -> anyhow::Result<()> {
        self.feedback(request_id.as_u128(), now_ms);
        Ok(())
    }

    fn on_causal_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
        _: ReplayTerminalStatus,
    ) -> anyhow::Result<()> {
        self.terminals += 1;
        self.feedback(request_id.as_u128(), now_ms);
        self.log
            .borrow_mut()
            .push(format!("terminal {} at {now_ms}", request_id.as_u128()));
        Ok(())
    }

    fn is_drained(&self) -> bool {
        self.terminals == self.total
    }
}

/// Zero-duration prefill exposes feedback at pass start. A reaction can join
/// worker 1's unstarted pass but cannot change worker 0's committed decode pass.
/// Disaggregated prefill feedback is covered by the rejection-reaction test.
#[test]
fn passes_start_in_ascending_worker_order_and_reactions_join_only_later_ones() {
    let mut failures = Vec::new();
    for backend in [Backend::Vllm, Backend::Sglang, Backend::Trtllm] {
        let label = format!("{backend:?}");
        let mut spec = zero_prefill_spec(backend, false);
        spec.adapters.placement.config = serde_json::json!({ "session_affinity": true });
        spec.topology = ReplayTopology::aggregated(2);
        let burst = vec![(TRIGGER_A, "a", 8), (TRIGGER_B, "b", 8)];
        let log = Rc::new(RefCell::new(Vec::new()));
        let report = run_external(spec, PassStartOrder::new(burst, log.clone()))
            .unwrap_or_else(|error| panic!("{label}: {error:?}"));
        let log = log.borrow();
        let position = |entry: String| log.iter().position(|logged| *logged == entry);
        let ordered = [
            position(format!("trigger {TRIGGER_A} at 10")),
            position(format!("submit [{REACTION_FOR_B}] at 10")),
            position(format!("trigger {TRIGGER_B} at 10")),
            position(format!("submit [{REACTION_FOR_A}] at 10")),
        ];
        let admitted_b = admission_ms(&report, REACTION_FOR_B);
        let admitted_a = admission_ms(&report, REACTION_FOR_A);
        println!("{label}: for b {admitted_b:?}, for a {admitted_a:?}; {log:?}");
        if ordered.iter().any(Option::is_none) || !ordered.is_sorted() {
            failures.push(format!(
                "{label}: worker 0 must start first and the source react before worker 1 \
                 starts: {log:?}"
            ));
        }
        if admitted_b != (10.0, 10.0) {
            failures.push(format!(
                "{label}: must join worker 1's pass: {admitted_b:?}"
            ));
        }
        if admitted_a.0 != 10.0 || admitted_a.1 <= 10.0 {
            failures.push(format!(
                "{label}: must miss worker 0's started pass: {admitted_a:?}"
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
