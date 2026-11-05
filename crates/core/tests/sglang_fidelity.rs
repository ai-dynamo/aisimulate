// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
// Modified behavioral adaptation of SGLang v0.5.22; see THIRD_PARTY_NOTICES.md.
// Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d

//! Deterministic offline regression scenarios across admission, P/D, cache salts,
//! and all HiCache write policies. Inputs are synthetic; no model download is needed.

use std::cell::RefCell;
use std::rc::Rc;

use aisimulate_core::engine::{
    Backend, EngineConfig, KvEvent, KvEventData, KvEventTier, SglangConfig, SglangHiCacheConfig,
    SglangHiCacheWritePolicy, TimingModelConfig,
};
use aisimulate_core::replay::loadgen::ReplayRequestPayload;
use aisimulate_core::replay::{
    AggregatedRoundRobinPlacement, CURRENT_REPLAY_SPEC_VERSION, EngineEventBatch, NoReplayMetadata,
    PerRequestRecord, Placement, PlacementEffects, PlacementPolicy, PoolRoundRobinPlacement,
    ReplayCaptureOptions, ReplayComposition, ReplayDeterminism, ReplayEngineConfig,
    ReplayEngineFactory, ReplayEngineObservation, ReplayReport, ReplayRequest, ReplayRoleConfig,
    ReplaySpec, ReplayTopology, Replayer, RequestIdentity, WorkerPool, WorkerPoolSpec, WorkerStage,
    WorkerTopology,
};
use uuid::Uuid;

const PAGE: usize = 64;
const PREFILL_MS: f64 = 10.0;
const DECODE_MS: f64 = 1.0;

// ---------------------------------------------------------------------------
// Recording composition: round-robin placement plus a log of every callback
// and every native KV event, stamped with the replay time it was observed at.
// ---------------------------------------------------------------------------

// Fields are compared through `PartialEq` and printed on failure.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
enum PolicyCall {
    Place {
        stage: &'static str,
        request: Option<Uuid>,
        at_ms: f64,
    },
    PrefillStarted {
        request: Uuid,
        reused: usize,
        at_ms: f64,
    },
    PrefillCompleted {
        stage: &'static str,
        request: Uuid,
        at_ms: f64,
    },
    Terminal {
        stage: &'static str,
        request: Uuid,
        at_ms: f64,
    },
}

#[derive(Debug, Clone, Default, PartialEq)]
struct Recording {
    calls: Vec<PolicyCall>,
    kv: Vec<(WorkerStage, f64, KvEvent)>,
}

impl Recording {
    fn stored_blocks(&self, stage: WorkerStage, from_ms: f64, to_ms: f64) -> usize {
        self.kv
            .iter()
            .filter(|(seen, at_ms, _)| *seen == stage && (from_ms..to_ms).contains(at_ms))
            .map(|(_, _, event)| match &event.data {
                KvEventData::Stored(stored) => stored.blocks.len(),
                KvEventData::Removed { .. } => 0,
            })
            .sum()
    }

    fn stored_events(&self, stage: WorkerStage) -> usize {
        self.kv
            .iter()
            .filter(|(seen, _, event)| {
                *seen == stage && matches!(event.data, KvEventData::Stored(_))
            })
            .count()
    }
}

type SharedRecording = Rc<RefCell<Recording>>;

#[derive(Debug, Default)]
struct KvBatch(Vec<(WorkerStage, KvEvent)>);

impl EngineEventBatch for KvBatch {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn append(&mut self, mut other: Self) {
        self.0.append(&mut other.0);
    }
}

struct KvObservation;

impl ReplayEngineObservation for KvObservation {
    type Batch = KvBatch;
    const CAPTURE_ENGINE_KV_EVENTS: bool = true;

    fn observe_engine_events(
        stage: WorkerStage,
        _worker_id: usize,
        _dp_rank: u32,
        events: Vec<KvEvent>,
    ) -> KvBatch {
        KvBatch(events.into_iter().map(|event| (stage, event)).collect())
    }
}

struct RecordingPlacement<P> {
    stage: &'static str,
    inner: P,
    log: SharedRecording,
}

impl<P> PlacementPolicy<ReplayRequestPayload> for RecordingPlacement<P>
where
    P: PlacementPolicy<ReplayRequestPayload, Metadata = (), Observation = KvBatch>,
{
    type Metadata = ();
    type Observation = KvBatch;

    fn place(
        &mut self,
        request: &ReplayRequestPayload,
        metadata: (),
        session_id: Option<String>,
        now_ms: f64,
    ) -> anyhow::Result<PlacementEffects> {
        self.log.borrow_mut().calls.push(PolicyCall::Place {
            stage: self.stage,
            request: request.request_id(),
            at_ms: now_ms,
        });
        self.inner.place(request, metadata, session_id, now_ms)
    }

    fn observe(&mut self, observation: KvBatch, now_ms: f64) -> anyhow::Result<Vec<Placement>> {
        self.log.borrow_mut().kv.extend(
            observation
                .0
                .iter()
                .map(|(stage, event)| (*stage, now_ms, event.clone())),
        );
        self.inner.observe(observation, now_ms)
    }

    fn dispatch_committed(&mut self, request_id: Uuid, now_ms: f64) -> anyhow::Result<()> {
        self.inner.dispatch_committed(request_id, now_ms)
    }

    fn dispatch_aborted(&mut self, request_id: Uuid, now_ms: f64) -> anyhow::Result<()> {
        self.inner.dispatch_aborted(request_id, now_ms)
    }

    fn advance_clock(&mut self, now_ms: f64) -> anyhow::Result<Vec<Placement>> {
        self.inner.advance_clock(now_ms)
    }

    fn cancel_pending(&mut self, request_id: Uuid) -> bool {
        self.inner.cancel_pending(request_id)
    }

    fn request_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        self.log.borrow_mut().calls.push(PolicyCall::Terminal {
            stage: self.stage,
            request: request_id,
            at_ms: now_ms,
        });
        self.inner.request_terminal(request_id, now_ms)
    }

    fn prefill_started(
        &mut self,
        request: Uuid,
        reused: usize,
        now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        self.log
            .borrow_mut()
            .calls
            .push(PolicyCall::PrefillStarted {
                request,
                reused,
                at_ms: now_ms,
            });
        self.inner.prefill_started(request, reused, now_ms)
    }

    fn prefill_completed(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        self.log
            .borrow_mut()
            .calls
            .push(PolicyCall::PrefillCompleted {
                stage: self.stage,
                request: request_id,
                at_ms: now_ms,
            });
        self.inner.prefill_completed(request_id, now_ms)
    }

    fn pending_count(&self) -> usize {
        self.inner.pending_count()
    }

    fn worker_ready(
        &mut self,
        worker: WorkerTopology,
        now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        self.inner.worker_ready(worker, now_ms)
    }

    fn worker_draining(
        &mut self,
        worker: WorkerTopology,
        now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        self.inner.worker_draining(worker, now_ms)
    }

    fn worker_removed(
        &mut self,
        worker: WorkerTopology,
        now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        self.inner.worker_removed(worker, now_ms)
    }

    fn topology_settled(&mut self, now_ms: f64) -> anyhow::Result<Vec<Placement>> {
        self.inner.topology_settled(now_ms)
    }
}

struct RecordingComposition {
    log: SharedRecording,
}

impl RecordingComposition {
    fn wrap<P>(&self, stage: &'static str, inner: P) -> RecordingPlacement<P> {
        RecordingPlacement {
            stage,
            inner,
            log: Rc::clone(&self.log),
        }
    }
}

impl ReplayComposition for RecordingComposition {
    type Metadata = NoReplayMetadata;
    type Observation = KvObservation;
    type AggregatedPlacement = RecordingPlacement<AggregatedRoundRobinPlacement<KvBatch>>;
    type DisaggregatedPlacement = RecordingPlacement<PoolRoundRobinPlacement<KvBatch>>;

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<Self::AggregatedPlacement> {
        Ok(self.wrap(
            "aggregated",
            AggregatedRoundRobinPlacement::new(dp_size, topology),
        ))
    }

    fn create_disaggregated_placements(
        &mut self,
        _prefill_dp_size: u32,
        prefill_topology: Vec<WorkerTopology>,
        _decode_dp_size: u32,
        decode_topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        Ok((
            self.wrap("prefill", PoolRoundRobinPlacement::new(prefill_topology)),
            self.wrap("decode", PoolRoundRobinPlacement::new(decode_topology)),
        ))
    }
}

// ---------------------------------------------------------------------------
// Trace and engine builders.
// ---------------------------------------------------------------------------

fn sglang_rank(num_gpu_blocks: usize) -> EngineConfig {
    EngineConfig {
        block_size: PAGE,
        num_gpu_blocks,
        max_num_seqs: 8,
        max_num_batched_tokens: 2048,
        emit_kv_events: true,
        timing_model: TimingModelConfig::Fixed {
            prefill_ms: PREFILL_MS,
            decode_ms: DECODE_MS,
        },
        sglang: SglangConfig {
            chunked_prefill_size: 512,
            max_prefill_tokens: 2048,
            // No output reservation for running requests, so the pressure phase
            // over-admits and must retract.
            schedule_conservativeness: 0.0,
            clip_max_new_tokens: 1,
            num_reserved_decode_tokens: 0,
            ..SglangConfig::default()
        },
        ..EngineConfig::for_backend(Backend::Sglang)
    }
}

fn prompt(seed: u32, len: usize) -> Vec<u32> {
    (0..len as u32).map(|offset| seed + offset).collect()
}

fn request(id: &str, arrival_time_ms: f64, input: Vec<u32>, output_tokens: usize) -> ReplayRequest {
    ReplayRequest {
        id: id.to_string(),
        arrival_time_ms,
        input_tokens: input.len(),
        input_token_ids: Some(input),
        output_tokens,
        output_token_ids: None,
        dp_rank: None,
        prefill_dp_rank: None,
        kv_hints: None,
        session_id: None,
        cache_salt: None,
        turn_index: None,
        metadata: serde_json::Value::Null,
    }
}

/// Pressure phase: three distinct 2-page prompts whose 400-token outputs
/// need 3 x 9 pages, more than the 16-page pool. Probe phase: two identical
/// 4-page prompts after the pool has drained, one second apart.
fn trace() -> Vec<ReplayRequest> {
    let mut requests: Vec<_> = (0..3)
        .map(|index| {
            request(
                &format!("pressure-{index}"),
                0.0,
                prompt(10_000 + index * 1_000, 2 * PAGE),
                400,
            )
        })
        .collect();
    let probe = prompt(50_000, 4 * PAGE);
    requests.push(request("probe-0", 5_000.0, probe.clone(), 8));
    requests.push(request("probe-1", 6_000.0, probe, 8));
    requests
}

fn spec(topology: ReplayTopology, engine: ReplayEngineConfig) -> ReplaySpec {
    ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
        topology,
        engine: serde_json::to_value(engine).unwrap(),
        adapters: Default::default(),
        max_sim_time_ms: Some(100_000.0),
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        encoder: None,
        requests: trace(),
    }
}

fn aggregated_spec() -> ReplaySpec {
    spec(
        ReplayTopology::aggregated(1),
        ReplayEngineConfig {
            rank: sglang_rank(16),
            ..ReplayEngineConfig::default()
        },
    )
}

fn disaggregated_spec() -> ReplaySpec {
    let role = |num_gpu_blocks| ReplayRoleConfig {
        rank: sglang_rank(num_gpu_blocks),
        ..ReplayRoleConfig::default()
    };
    spec(
        ReplayTopology::Disaggregated {
            prefill: WorkerPoolSpec::default(),
            decode: WorkerPoolSpec::default(),
            handoff_latency_ms: 1.0,
        },
        ReplayEngineConfig {
            rank: sglang_rank(64),
            prefill: Some(role(64)),
            decode: Some(role(16)),
            ..ReplayEngineConfig::default()
        },
    )
}

// ---------------------------------------------------------------------------
// Execution and shared checks.
// ---------------------------------------------------------------------------

fn run(spec: ReplaySpec) -> (ReplayReport, Recording) {
    let log = SharedRecording::default();
    let report = Replayer::with_composition(
        spec,
        ReplayEngineFactory::new(),
        RecordingComposition {
            log: Rc::clone(&log),
        },
    )
    .unwrap()
    .with_capture_options(ReplayCaptureOptions {
        capture_canonical_evidence: true,
        capture_per_request: true,
        determinism: ReplayDeterminism::CanonicalV1,
        ..ReplayCaptureOptions::default()
    })
    .run()
    .unwrap();
    let recording = std::mem::take(&mut *log.borrow_mut());
    (report, recording)
}

/// Run twice and require identical reports, per-request records and logs.
fn run_deterministic(spec: ReplaySpec) -> (ReplayReport, Recording) {
    let (first, first_log) = run(spec.clone());
    let (second, second_log) = run(spec);
    let summary = |report: &ReplayReport| {
        let mut report = report.clone();
        report.throughput.wall_time_ms = 0.0;
        serde_json::to_value(&report).unwrap()
    };
    assert_eq!(summary(&first), summary(&second));
    assert_eq!(
        serde_json::to_value(&first.per_request).unwrap(),
        serde_json::to_value(&second.per_request).unwrap()
    );
    assert_eq!(first_log, second_log);
    (first, first_log)
}

fn record<'a>(report: &'a ReplayReport, id: &str) -> &'a PerRequestRecord {
    report
        .per_request
        .iter()
        .find(|record| record.request_id.as_deref() == Some(id))
        .unwrap_or_else(|| panic!("no record for {id}"))
}

/// Every retraction is a pressure record of `pool` whose request was
/// readmitted, and readmissions are exactly those retractions.
fn assert_retractions(report: &ReplayReport, pool: WorkerPool) -> u64 {
    let pressure = report
        .runtime_evidence
        .pressure
        .as_ref()
        .expect("canonical evidence captures pressure");
    assert_eq!(pressure.vllm_preemptions_total, 0);
    assert_eq!(
        pressure.sglang_retractions_total,
        pressure.records.len() as u64
    );
    for retraction in &pressure.records {
        assert_eq!(retraction.pool, pool, "{retraction:?}");
        assert!(retraction.readmitted_at_ms.is_some(), "{retraction:?}");
    }
    let readmissions: usize = report
        .per_request
        .iter()
        .map(|record| record.readmission_count)
        .sum();
    assert_eq!(readmissions as u64, pressure.sglang_retractions_total);
    pressure.sglang_retractions_total
}

/// The probe requests run alone on a drained pool under fixed timing, so the
/// second one's timeline is the first one's shifted by their arrival gap, and
/// it reuses the capped whole-prompt match: `(256 - 1) / 64` pages.
fn assert_probes(report: &ReplayReport) {
    let first = record(report, "probe-0");
    let second = record(report, "probe-1");
    assert_eq!(first.reused_input_tokens, 0);
    assert_eq!(second.reused_input_tokens, 3 * PAGE);
    assert_eq!(second.first_admission_reused_input_tokens, 3 * PAGE);
    let shift = second.arrival_time_ms - first.arrival_time_ms;
    assert_eq!(
        second.first_token_ms,
        first.first_token_ms.map(|ms| ms + shift)
    );
    assert_eq!(second.terminal_time_ms, first.terminal_time_ms + shift);
    assert_eq!(second.output_length, 8);
}

fn assert_all_completed(report: &ReplayReport) {
    assert_eq!(
        report.request_counts.completed_requests,
        report.per_request.len()
    );
    for record in &report.per_request {
        let first_token = record.first_token_ms.expect("every request emits tokens");
        assert!(first_token >= record.arrival_time_ms, "{record:?}");
        assert!(record.terminal_time_ms >= first_token, "{record:?}");
        assert_eq!(record.output_length, record.requested_output_length);
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[test]
fn sglang_aggregated_replay_reports_reuse_retraction_kv_and_completion() {
    let (report, log) = run_deterministic(aggregated_spec());
    assert_all_completed(&report);
    assert_probes(&report);
    assert!(assert_retractions(&report, WorkerPool::Agg) > 0);

    // Fixed timing: the probe's first token comes from its prefill pass.
    let probe = record(&report, "probe-0");
    assert_eq!(
        probe.first_token_ms,
        Some(probe.arrival_time_ms + PREFILL_MS)
    );

    // probe-0 publishes its four prompt pages before probe-1 arrives.
    assert!(log.stored_blocks(WorkerStage::Aggregated, 5_000.0, 6_000.0) >= 4);
    assert!(
        log.kv
            .iter()
            .all(|(stage, _, _)| *stage == WorkerStage::Aggregated)
    );

    // Every request is placed once and reaches the policy as prefill-complete
    // and terminal exactly once.
    for kind in ["place", "prefill_started", "terminal"] {
        let count = log
            .calls
            .iter()
            .filter(|call| {
                matches!(
                    (kind, call),
                    ("place", PolicyCall::Place { .. })
                        | ("prefill_started", PolicyCall::PrefillStarted { .. })
                        | ("terminal", PolicyCall::Terminal { .. })
                )
            })
            .count();
        assert_eq!(count, 5, "{kind}");
    }
}

#[test]
fn sglang_disaggregated_replay_reports_reuse_retraction_kv_and_completion() {
    let (report, log) = run_deterministic(disaggregated_spec());
    assert_all_completed(&report);
    assert_probes(&report);
    // Destinations reserve only the 2-page prompt, so the 16-page decode pool
    // must retract once the three outputs grow.
    assert!(assert_retractions(&report, WorkerPool::Decode) > 0);

    for record in &report.per_request {
        let reserved = record.destination_reserved_ms.expect("reserved");
        let activated = record.destination_activated_ms.expect("activated");
        let decode_admitted = record.decode_admit_ms.expect("decode admitted");
        assert!(
            record.prefill_admit_ms.expect("prefilled") <= reserved,
            "{record:?}"
        );
        assert!(
            reserved <= activated && activated <= decode_admitted,
            "{record:?}"
        );
        assert!(decode_admitted <= record.terminal_time_ms, "{record:?}");
        assert_eq!(record.prefill_worker_idx, Some(0));
        assert_eq!(record.decode_worker_idx, Some(0));
    }
    // The default credits no decode-side reuse to a transferred prompt, even when the
    // decode radix already holds it (probe-1). Retracted requests are excluded:
    // their decode readmission can match their own released prefix.
    for id in ["probe-0", "probe-1"] {
        assert_eq!(
            record(&report, id).decode_reused_input_tokens,
            Some(0),
            "{id}"
        );
    }

    assert!(log.stored_blocks(WorkerStage::Prefill, 5_000.0, 6_000.0) >= 4);
    // Decode radix caching is off unless explicitly enabled.
    let decode_stores = log.stored_events(WorkerStage::Decode);
    assert!(
        decode_stores == 0,
        "decode cache defaults off: {decode_stores}"
    );

    let placed = |stage: &str| {
        log.calls
            .iter()
            .filter(|call| matches!(call, PolicyCall::Place { stage: seen, .. } if *seen == stage))
            .count()
    };
    assert_eq!((placed("prefill"), placed("decode")), (5, 5));
}

fn agg_with(rank: EngineConfig, requests: Vec<ReplayRequest>) -> ReplaySpec {
    let mut replay = spec(
        ReplayTopology::aggregated(1),
        ReplayEngineConfig {
            rank,
            ..Default::default()
        },
    );
    replay.requests = requests;
    replay
}

#[test]
fn salted_reuse_and_final_kv_survive_speculative_and_overlap_modes() {
    for overlap in [false, true] {
        for speculative in [false, true] {
            let mut rank = sglang_rank(64);
            rank.sglang.host_loop = overlap;
            if speculative {
                rank.aic_nextn = Some(2);
                rank.aic_nextn_accept_rates = Some("1.0,1.0".into());
            }
            let shared = prompt(1_000, 256);
            let mut requests = vec![
                request("a0", 0.0, shared.clone(), 4),
                request("a1", 100.0, shared.clone(), 4),
                request("b0", 200.0, shared, 4),
            ];
            for (req, salt) in requests.iter_mut().zip(["a", "a", "b"]) {
                req.cache_salt = Some(salt.into());
            }
            let input = prompt(20_000, 100);
            let outputs = prompt(30_000, 28);
            let mut first = request("finish", 300.0, input.clone(), 28);
            first.output_token_ids = Some(outputs.clone());
            requests.push(first);
            requests.push(request(
                "finish-probe",
                500.0,
                input
                    .into_iter()
                    .chain(outputs)
                    .chain(prompt(40_000, 72))
                    .collect(),
                1,
            ));
            let (report, log) = run_deterministic(agg_with(rank, requests));
            assert_all_completed(&report);
            assert_eq!(
                record(&report, "a1").first_admission_reused_input_tokens,
                192
            );
            assert_eq!(record(&report, "b0").first_admission_reused_input_tokens, 0);
            // All four modes stop at P+N-1 materialized tokens for this length stop.
            assert_eq!(
                record(&report, "finish-probe").first_admission_reused_input_tokens,
                64,
                "overlap={overlap} speculative={speculative}"
            );
            for row in &report.per_request {
                let uuid = Uuid::parse_str(&row.uuid).unwrap();
                let starts = log
                    .calls
                    .iter()
                    .filter_map(|call| match call {
                        PolicyCall::PrefillStarted {
                            request,
                            reused,
                            at_ms,
                        } if *request == uuid => Some((*reused, *at_ms)),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    starts,
                    vec![(
                        row.first_admission_reused_input_tokens,
                        row.first_admit_ms.unwrap()
                    )]
                );
            }
        }
    }
}

#[test]
fn whole_prompt_and_idle_reset_gate_later_arrivals() {
    let mut rank = sglang_rank(32);
    rank.sglang.schedule_conservativeness = 1.0;
    rank.sglang.clip_max_new_tokens = 1024;
    let replay = agg_with(
        rank,
        vec![
            request("long", 0.0, prompt(1_000, 768), 700),
            request("whole-prompt", 50.0, prompt(5_000, 1250), 64),
            request("after-idle", 3000.0, prompt(10_000, 1024), 896),
            request("reset-probe", 3040.0, prompt(15_000, 448), 64),
        ],
    );
    let (report, _) = run_deterministic(replay);
    assert_all_completed(&report);
    assert!(
        record(&report, "whole-prompt").first_admit_ms.unwrap()
            >= record(&report, "long").terminal_time_ms
    );
    assert!(
        record(&report, "reset-probe").first_admit_ms.unwrap()
            >= record(&report, "after-idle").terminal_time_ms
    );
}

#[test]
fn pd_decode_cache_flag_and_capacity_hold_sources_until_handoff() {
    for caching in [false, true] {
        let mut replay = disaggregated_spec();
        let mut engine: ReplayEngineConfig = serde_json::from_value(replay.engine).unwrap();
        let prefill = &mut engine.prefill.as_mut().unwrap().rank;
        prefill.max_num_seqs = 2;
        let decode = &mut engine.decode.as_mut().unwrap().rank;
        decode.num_gpu_blocks = 36;
        decode.sglang.num_reserved_decode_tokens = 512;
        decode.sglang.clip_max_new_tokens = 1024;
        decode.sglang.disaggregation_decode_enable_radix_cache = caching;
        replay.engine = serde_json::to_value(engine).unwrap();
        replay.requests = vec![
            request("held-0", 0.0, prompt(1_000, 512), 1000),
            request("held-1", 0.0, prompt(3_000, 512), 1000),
            request("held-2", 0.0, prompt(5_000, 256), 8),
            request("reuse-0", 5000.0, prompt(8_000, 256), 8),
            request("reuse-1", 5100.0, prompt(8_000, 256), 8),
        ];
        let (report, log) = run_deterministic(replay);
        assert_all_completed(&report);
        assert!(
            record(&report, "held-1").destination_reserved_ms.unwrap()
                >= record(&report, "held-0").terminal_time_ms
        );
        assert!(
            record(&report, "held-2").prefill_admit_ms.unwrap()
                >= record(&report, "held-0").source_released_ms.unwrap()
        );
        assert_eq!(
            record(&report, "reuse-1").decode_reused_input_tokens,
            Some(if caching { 256 } else { 0 })
        );
        assert_eq!(log.stored_events(WorkerStage::Decode) > 0, caching);
    }
}

fn hicache_replay(policy: SglangHiCacheWritePolicy, overlap: bool) -> ReplaySpec {
    let mut rank = sglang_rank(9);
    rank.kv_cache_bytes_per_token = Some(1024);
    rank.sglang.host_loop = overlap;
    rank.sglang.hicache = Some(SglangHiCacheConfig {
        write_policy: policy,
        d2h_bandwidth_gbps: 0.1,
        h2d_bandwidth_gbps: 0.1,
        ..SglangHiCacheConfig::new(32)
    });
    let warm = prompt(1_000, 384);
    let mut requests = vec![
        request("warm", 0.0, warm.clone(), 1),
        request("warm-again", 100.0, warm.clone(), 1),
        request("pressure", 200.0, prompt(3_000, 384), 1),
        request(
            "reload",
            300.0,
            warm.clone().into_iter().chain(prompt(2_000, 64)).collect(),
            1,
        ),
        request("other-tenant", 400.0, warm, 1),
    ];
    for row in &mut requests {
        row.cache_salt = Some("a".into());
    }
    requests.last_mut().unwrap().cache_salt = Some("b".into());
    agg_with(rank, requests)
}

#[test]
fn hicache_policies_reload_salted_prefixes_and_keep_tier_events_consistent() {
    for policy in [
        SglangHiCacheWritePolicy::WriteThrough,
        SglangHiCacheWritePolicy::WriteThroughSelective,
        SglangHiCacheWritePolicy::WriteBack,
    ] {
        for overlap in [false, true] {
            let (report, log) = run_deterministic(hicache_replay(policy, overlap));
            assert_all_completed(&report);
            assert!(
                record(&report, "reload")
                    .first_admission_host_reused_input_tokens
                    .unwrap()
                    > 0,
                "{policy:?} overlap={overlap}"
            );
            assert_eq!(
                record(&report, "other-tenant").first_admission_reused_input_tokens,
                0
            );
            let mut device_hashes = std::collections::HashSet::new();
            let mut host_hashes = std::collections::HashSet::new();
            for (_, _, event) in &log.kv {
                let resident = if event.tier == KvEventTier::Device {
                    &mut device_hashes
                } else {
                    &mut host_hashes
                };
                match &event.data {
                    KvEventData::Stored(stored) => {
                        assert!(matches!(stored.cache_salt.as_deref(), Some("a" | "b")));
                        resident.extend(stored.blocks.iter().map(|block| block.block_hash));
                    }
                    KvEventData::Removed { block_hashes } => {
                        for hash in block_hashes {
                            assert!(resident.remove(hash), "removal without store: {event:?}");
                        }
                    }
                }
            }
            assert!(!host_hashes.is_empty());
        }
    }
}

fn adversarial_spec(
    seed: u32,
    count: usize,
    policy: SglangHiCacheWritePolicy,
    overlap: bool,
) -> ReplaySpec {
    let mut rank = sglang_rank(32);
    rank.max_num_seqs = 16;
    rank.sglang.chunked_prefill_size = 256;
    rank.sglang.host_loop = overlap;
    rank.kv_cache_bytes_per_token = Some(1024);
    rank.sglang.hicache = Some(SglangHiCacheConfig {
        write_policy: policy,
        d2h_bandwidth_gbps: if seed % 3 == 0 { 0.0 } else { 0.25 },
        h2d_bandwidth_gbps: if seed % 5 == 0 { 0.0 } else { 0.125 },
        latency_to_first_byte_ms: (seed % 4) as f64,
        ..SglangHiCacheConfig::new(8 + (seed as usize % 5) * 16)
    });
    let mut state = u64::from(seed) + 1;
    let mut requests = Vec::with_capacity(count);
    for i in 0..count {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let family = (state >> 32) as u32 % 12;
        let length = 128 + ((state >> 40) as usize % 8) * 64;
        let mut row = request(
            &format!("r{i}"),
            (i / 8) as f64 * 250.0,
            prompt(10_000 + family * 10_000, length),
            1 + ((state >> 48) as usize % 128),
        );
        row.cache_salt = Some(format!("tenant-{}", family % 3));
        requests.push(row);
    }
    agg_with(rank, requests)
}

#[test]
fn bounded_adversarial_replays_drain_with_exact_outputs() {
    for seed in 0..4 {
        for policy in [
            SglangHiCacheWritePolicy::WriteThrough,
            SglangHiCacheWritePolicy::WriteThroughSelective,
            SglangHiCacheWritePolicy::WriteBack,
        ] {
            for overlap in [false, true] {
                eprintln!("adversarial seed={seed} policy={policy:?} overlap={overlap}");
                let (report, _) = run_deterministic(adversarial_spec(seed, 48, policy, overlap));
                assert_all_completed(&report);
                assert_eq!(
                    report.request_counts.completed_requests, 48,
                    "seed={seed} {policy:?} overlap={overlap}"
                );
            }
        }
    }
}

#[test]
fn hicache_backup_pressure_waits_for_completion() {
    for overlap in [false, true] {
        for decode_stall in [false, true] {
            let mut rank = sglang_rank(if decode_stall { 10 } else { 9 });
            rank.sglang.host_loop = overlap;
            rank.kv_cache_bytes_per_token = Some(1024);
            rank.sglang.hicache = Some(SglangHiCacheConfig {
                d2h_bandwidth_gbps: 0.0,
                h2d_bandwidth_gbps: 0.0,
                latency_to_first_byte_ms: if decode_stall { 1000.0 } else { 3.0 },
                ..SglangHiCacheConfig::new(32)
            });
            let (report, _) = run_deterministic(agg_with(
                rank,
                vec![
                    request("backup", 0.0, prompt(1000, 384), 1),
                    request(
                        "blocked",
                        11.0,
                        prompt(3000, 128),
                        if decode_stall { 256 } else { 1 },
                    ),
                ],
            ));
            assert_all_completed(&report);
            let blocked = record(&report, "blocked");
            if decode_stall {
                assert_eq!(blocked.first_admit_ms, Some(11.0));
                assert!(blocked.terminal_time_ms >= 1010.0);
            } else {
                assert_eq!(blocked.first_admit_ms, Some(13.0));
            }
        }
    }
}

/// Run explicitly with --ignored --nocapture. Optional result path makes this usable
/// with an interleaved wall-clock benchmark runner without adding a production API.
#[test]
#[ignore = "extended offline soak and profiling workload"]
fn extended_offline_soak_profile() {
    let count = std::env::var("AISIM_SOAK_REQUESTS")
        .map_or(Ok(1000), |value| value.parse::<usize>())
        .unwrap();
    let seeds = std::env::var("AISIM_SOAK_SEEDS")
        .map_or(Ok(8), |value| value.parse::<u32>())
        .unwrap();
    let mut completed = 0;
    let mut output_tokens = 0;
    let mut simulated_ms = 0.0;
    let mut reused = 0;
    let started = std::time::Instant::now();
    for seed in 0..seeds {
        for policy in [
            SglangHiCacheWritePolicy::WriteThrough,
            SglangHiCacheWritePolicy::WriteThroughSelective,
            SglangHiCacheWritePolicy::WriteBack,
        ] {
            for overlap in [false, true] {
                eprintln!("soak seed={seed} policy={policy:?} overlap={overlap} count={count}");
                let (report, _) = run(adversarial_spec(seed, count, policy, overlap));
                assert_all_completed(&report);
                assert_eq!(report.request_counts.completed_requests, count);
                completed += report.request_counts.completed_requests;
                for row in &report.per_request {
                    output_tokens += row.output_length;
                    reused += row.first_admission_reused_input_tokens;
                    simulated_ms += row.terminal_time_ms;
                }
            }
        }
    }
    let result = serde_json::json!({
        "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
        "completed": completed,
        "output_tokens": output_tokens,
        "first_admission_reused_tokens": reused,
        "terminal_ms_sum": simulated_ms,
        "requests_per_case": count,
        "seeds": seeds,
    });
    if let Ok(destination) = std::env::var("AISIM_SOAK_RESULT") {
        std::fs::write(destination, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    }
    println!("{result}");
}
