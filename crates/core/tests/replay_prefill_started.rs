// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `PlacementPolicy::prefill_started` reports each request's first prefill
//! admission as the report records it, and can release queued placements.

use std::cell::RefCell;
use std::rc::Rc;

use aisimulate_core::engine::{Backend, EngineConfig, TimingModelConfig};
use aisimulate_core::replay::loadgen::ReplayRequestPayload;
use aisimulate_core::replay::{
    AggregatedRoundRobinPlacement, CURRENT_REPLAY_SPEC_VERSION, NoEngineEvents, NoReplayMetadata,
    PerRequestRecord, Placement, PlacementDecision, PlacementEffects, PlacementPolicy,
    PoolRoundRobinPlacement, ReplayCaptureOptions, ReplayComposition, ReplayDeterminism,
    ReplayEngineConfig, ReplayEngineFactory, ReplayReport, ReplayRequest, ReplayRequestPool,
    ReplayRoutingOutcome, ReplaySpec, ReplayTopology, Replayer, WorkerPoolSpec, WorkerStage,
    WorkerTopology,
};
use uuid::Uuid;

/// One prefill callback. `started` holds `(reused_input_tokens, now_ms)` for
/// `prefill_started` and is `None` for `prefill_completed`.
#[derive(Debug, Clone, Copy)]
struct PrefillCallback {
    stage: WorkerStage,
    request_id: Uuid,
    started: Option<(usize, f64)>,
}

type CallbackLog = Rc<RefCell<Vec<PrefillCallback>>>;

/// Delegates to round-robin placement and records the prefill callbacks. With
/// `hold_first`, it queues the first request it places until another
/// request's prefill starts.
struct RecordingPlacement<P> {
    stage: WorkerStage,
    inner: P,
    log: CallbackLog,
    hold_first: bool,
    placed: usize,
    held: Option<Placement>,
}

impl<P> RecordingPlacement<P> {
    fn record(&self, request_id: Uuid, started: Option<(usize, f64)>) {
        self.log.borrow_mut().push(PrefillCallback {
            stage: self.stage,
            request_id,
            started,
        });
    }
}

impl<P> PlacementPolicy<ReplayRequestPayload> for RecordingPlacement<P>
where
    P: PlacementPolicy<ReplayRequestPayload, Metadata = (), Observation = ()>,
{
    type Metadata = ();
    type Observation = ();

    fn place(
        &mut self,
        request: &ReplayRequestPayload,
        metadata: (),
        session_id: Option<String>,
        now_ms: f64,
    ) -> anyhow::Result<PlacementEffects> {
        let effects = self.inner.place(request, metadata, session_id, now_ms)?;
        self.placed += 1;
        if !self.hold_first || self.placed > 1 {
            return Ok(effects);
        }
        let PlacementDecision::Immediate(placement) = effects.decision else {
            anyhow::bail!("round-robin placement queued the request this policy holds");
        };
        self.held = Some(placement);
        Ok(PlacementEffects {
            decision: PlacementDecision::Queued,
            released: effects.released,
        })
    }

    fn observe(&mut self, observation: (), now_ms: f64) -> anyhow::Result<Vec<Placement>> {
        self.inner.observe(observation, now_ms)
    }

    fn cancel_pending(&mut self, request_id: Uuid) -> bool {
        if self.held.is_some_and(|held| held.request_id == request_id) {
            self.held = None;
            return true;
        }
        self.inner.cancel_pending(request_id)
    }

    fn request_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        self.inner.request_terminal(request_id, now_ms)
    }

    fn prefill_started(
        &mut self,
        request_id: Uuid,
        reused_input_tokens: usize,
        now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        self.record(request_id, Some((reused_input_tokens, now_ms)));
        let mut released = self
            .inner
            .prefill_started(request_id, reused_input_tokens, now_ms)?;
        // The held request cannot start before its release, so this start
        // belongs to another request.
        released.extend(self.held.take());
        Ok(released)
    }

    fn prefill_completed(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        self.record(request_id, None);
        self.inner.prefill_completed(request_id, now_ms)
    }

    fn pending_count(&self) -> usize {
        self.inner.pending_count() + usize::from(self.held.is_some())
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
    log: CallbackLog,
    hold_first: bool,
}

impl RecordingComposition {
    fn placement<P>(&self, stage: WorkerStage, inner: P) -> RecordingPlacement<P> {
        RecordingPlacement {
            stage,
            inner,
            log: Rc::clone(&self.log),
            // The decode pool never sees a prefill start that would release it.
            hold_first: self.hold_first && stage != WorkerStage::Decode,
            placed: 0,
            held: None,
        }
    }
}

impl ReplayComposition for RecordingComposition {
    type Metadata = NoReplayMetadata;
    type Observation = NoEngineEvents;
    type AggregatedPlacement = RecordingPlacement<AggregatedRoundRobinPlacement<()>>;
    type DisaggregatedPlacement = RecordingPlacement<PoolRoundRobinPlacement<()>>;

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<Self::AggregatedPlacement> {
        Ok(self.placement(
            WorkerStage::Aggregated,
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
            self.placement(
                WorkerStage::Prefill,
                PoolRoundRobinPlacement::new(prefill_topology),
            ),
            self.placement(
                WorkerStage::Decode,
                PoolRoundRobinPlacement::new(decode_topology),
            ),
        ))
    }
}

/// Every (backend, disaggregated) pair whose P/D path the replay supports.
const CASES: [(Backend, bool); 5] = [
    (Backend::Vllm, false),
    (Backend::Vllm, true),
    (Backend::Sglang, false),
    (Backend::Sglang, true),
    (Backend::Trtllm, false),
];

fn spec(
    backend: Backend,
    disagg: bool,
    num_gpu_blocks: usize,
    requests: Vec<ReplayRequest>,
) -> ReplaySpec {
    let mut engine = ReplayEngineConfig {
        rank: EngineConfig {
            block_size: 4,
            num_gpu_blocks,
            max_num_seqs: 4,
            max_num_batched_tokens: 64,
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 1.0,
                decode_ms: 1.0,
            },
            ..EngineConfig::for_backend(backend)
        },
        ..Default::default()
    };
    // Tiny pools isolate callback lifecycle from the default 512-token decode headroom.
    engine.rank.sglang.num_reserved_decode_tokens = 0;
    ReplaySpec {
        encoder: None,
        version: CURRENT_REPLAY_SPEC_VERSION,
        topology: if disagg {
            ReplayTopology::Disaggregated {
                prefill: WorkerPoolSpec::default(),
                decode: WorkerPoolSpec::default(),
                handoff_latency_ms: 0.0,
            }
        } else {
            ReplayTopology::aggregated(1)
        },
        engine: serde_json::to_value(engine).unwrap(),
        adapters: Default::default(),
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests,
    }
}

fn request(
    id: &str,
    arrival_time_ms: f64,
    input_token_ids: Vec<u32>,
    output_tokens: usize,
) -> ReplayRequest {
    ReplayRequest {
        id: id.to_string(),
        arrival_time_ms,
        input_tokens: input_token_ids.len(),
        input_token_ids: Some(input_token_ids),
        output_tokens,
        output_token_ids: None,
        dp_rank: None,
        prefill_dp_rank: None,
        kv_hints: None,
        session_id: None,
        turn_index: None,
        metadata: serde_json::Value::Null,
    }
}

fn run(spec: ReplaySpec, hold_first: bool) -> (ReplayReport, Vec<PrefillCallback>) {
    let log = CallbackLog::default();
    let report = Replayer::with_composition(
        spec,
        ReplayEngineFactory::new(),
        RecordingComposition {
            log: Rc::clone(&log),
            hold_first,
        },
    )
    .unwrap()
    .with_capture_options(ReplayCaptureOptions {
        capture_per_request: true,
        determinism: ReplayDeterminism::CanonicalV1,
        ..Default::default()
    })
    .run()
    .unwrap();
    let log = log.take();
    (report, log)
}

fn record<'a>(report: &'a ReplayReport, id: &str) -> &'a PerRequestRecord {
    report
        .per_request
        .iter()
        .find(|record| record.request_id.as_deref() == Some(id))
        .unwrap_or_else(|| panic!("no record for {id}"))
}

fn uuid(record: &PerRequestRecord) -> Uuid {
    Uuid::parse_str(&record.uuid).unwrap()
}

/// The primer warms a prefix the probe reuses after the primer finishes.
fn warm_prefix_requests() -> Vec<ReplayRequest> {
    vec![
        request("primer", 0.0, (0..16).collect(), 4),
        request("cold", 0.5, (200..212).collect(), 4),
        request("probe", 30.0, (0..12).chain(100..104).collect(), 4),
    ]
}

/// Three requests whose decode outgrows a fourteen-block pool. vLLM and
/// SGLang readmit under pressure; TRT-LLM's guaranteed-no-evict policy waits.
fn pressure_requests() -> Vec<ReplayRequest> {
    (0..3)
        .map(|index| {
            let first = 100 * index;
            request(
                &format!("pressure-{index}"),
                0.0,
                (first..first + 12).collect(),
                20,
            )
        })
        .collect()
}

#[test]
fn prefill_started_reports_each_first_prefill_admission_once() {
    for (backend, disagg) in CASES {
        for (scenario, num_gpu_blocks, requests) in [
            ("warm-prefix", 64, warm_prefix_requests()),
            ("pressure", 14, pressure_requests()),
        ] {
            let case = format!("{backend:?} disagg={disagg} {scenario}");
            let count = requests.len();
            let mut replay = spec(backend, disagg, num_gpu_blocks, requests);
            if backend == Backend::Sglang && scenario == "pressure" {
                let mut engine: ReplayEngineConfig = serde_json::from_value(replay.engine).unwrap();
                // A short output forecast admits all rows before decode growth forces retraction.
                engine.rank.sglang.schedule_conservativeness = 0.0;
                engine.rank.sglang.clip_max_new_tokens = 1;
                replay.engine = serde_json::to_value(engine).unwrap();
            }
            let (report, log) = run(replay, false);
            assert_eq!(report.request_counts.completed_requests, count, "{case}");
            let prefill_stage = if disagg {
                WorkerStage::Prefill
            } else {
                WorkerStage::Aggregated
            };
            assert!(
                log.iter().all(|callback| callback.stage == prefill_stage),
                "{case}: only the pool that prefills sees prefill callbacks: {log:?}"
            );
            for record in &report.per_request {
                let id = uuid(record);
                let context = format!("{case} {:?}", record.request_id);
                let callbacks = log
                    .iter()
                    .filter(|callback| callback.request_id == id)
                    .map(|callback| callback.started)
                    .collect::<Vec<_>>();
                let [Some((reused_input_tokens, now_ms)), None] = callbacks[..] else {
                    panic!("{context}: expected one start, then one completion: {callbacks:?}");
                };
                assert_eq!(Some(now_ms), record.first_admit_ms, "{context}");
                assert_eq!(
                    reused_input_tokens, record.first_admission_reused_input_tokens,
                    "{context}"
                );
                if disagg {
                    assert_eq!(Some(now_ms), record.prefill_admit_ms, "{context}");
                }
            }
            if scenario == "warm-prefix" {
                assert!(
                    record(&report, "probe").first_admission_reused_input_tokens > 0,
                    "{case}: the probe must start from the primer's cached prefix"
                );
            } else {
                let readmitted = report
                    .per_request
                    .iter()
                    .flat_map(|record| &record.admission_history)
                    .any(|admission| admission.is_readmission);
                assert_eq!(
                    readmitted,
                    backend != Backend::Trtllm,
                    "{case}: pressure must exercise the backend's eviction policy"
                );
            }
        }
    }
}

#[test]
fn prefill_started_precedes_single_output_completion_across_chunks() {
    for (backend, disagg) in CASES {
        for prefill_ms in [0.0, 1.0] {
            let case = format!("{backend:?} disagg={disagg} prefill_ms={prefill_ms}");
            let mut replay = spec(
                backend,
                disagg,
                64,
                vec![request("single-output", 0.0, (0..12).collect(), 1)],
            );
            let mut engine: ReplayEngineConfig = serde_json::from_value(replay.engine).unwrap();
            engine.rank.max_num_batched_tokens = 4;
            engine.rank.sglang.chunked_prefill_size = 4;
            engine.rank.timing_model = TimingModelConfig::Fixed {
                prefill_ms,
                decode_ms: prefill_ms,
            };
            replay.engine = serde_json::to_value(engine).unwrap();
            let (report, log) = run(replay, false);
            assert_eq!(report.request_counts.completed_requests, 1, "{case}");
            let record = record(&report, "single-output");
            let callbacks = log.iter().map(|call| call.started).collect::<Vec<_>>();
            let start = Some((0, record.first_admit_ms.unwrap()));
            let expected = if disagg {
                vec![start, None]
            } else {
                vec![start]
            };
            assert_eq!(callbacks, expected, "{case}");
        }
    }
}

#[test]
fn prefill_started_releases_a_request_held_for_another_prefill() {
    for (backend, disagg) in CASES {
        let case = format!("{backend:?} disagg={disagg}");
        let requests = vec![
            request("held", 0.0, (0..8).collect(), 2),
            request("trigger", 5.0, (100..108).collect(), 2),
        ];
        let (report, log) = run(spec(backend, disagg, 64, requests), true);
        assert_eq!(report.request_counts.completed_requests, 2, "{case}");

        let trigger = record(&report, "trigger");
        let held = record(&report, "held");
        let trigger_start = trigger.first_admit_ms.unwrap();
        assert_eq!(trigger_start, 5.0, "{case}");
        let pool = if disagg {
            ReplayRequestPool::Prefill
        } else {
            ReplayRequestPool::Agg
        };
        let route = held
            .routing_history
            .iter()
            .find(|route| route.pool == pool)
            .unwrap();
        assert_eq!(route.outcome, ReplayRoutingOutcome::Queued, "{case}");
        assert_eq!(route.queue_entered_at_ms, Some(0.0), "{case}");
        assert_eq!(route.released_at_ms, Some(trigger_start), "{case}");
        assert!(held.first_admit_ms.unwrap() >= trigger_start, "{case}");

        let starts = log
            .iter()
            .filter(|callback| callback.started.is_some())
            .map(|callback| callback.request_id)
            .collect::<Vec<_>>();
        assert_eq!(starts, [uuid(trigger), uuid(held)], "{case}");
    }
}
