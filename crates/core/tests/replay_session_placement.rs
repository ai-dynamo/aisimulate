// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::rc::Rc;

use aisimulate_core::engine::{Backend, EngineConfig, KvEvent, KvEventData, TimingModelConfig};
use aisimulate_core::replay::loadgen::{
    MooncakeRow, ReplayRequestPayload, Trace, load_agentic_mooncake,
};
use aisimulate_core::replay::{
    AggregatedRoundRobinPlacement, CURRENT_REPLAY_SPEC_VERSION, EngineEventBatch, NoReplayMetadata,
    Placement, PlacementEffects, PlacementPolicy, PoolRoundRobinPlacement, ReplayComposition,
    ReplayEngineConfig, ReplayEngineFactory, ReplayEngineObservation, ReplayRequest,
    ReplayRuntimeInput, ReplaySpec, ReplayTopology, Replayer, WorkerPoolSpec, WorkerStage,
    WorkerTopology,
};
use uuid::Uuid;

type SeenSessions = Rc<RefCell<Vec<(&'static str, Option<String>)>>>;

/// Delegates to round-robin placement and records the session each call sees.
struct SessionRecordingPlacement<P> {
    stage: &'static str,
    inner: P,
    seen: SeenSessions,
    events: Rc<RefCell<Vec<(WorkerStage, KvEvent)>>>,
}

impl<P> PlacementPolicy<ReplayRequestPayload> for SessionRecordingPlacement<P>
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
        self.seen
            .borrow_mut()
            .push((self.stage, session_id.clone()));
        self.inner.place(request, metadata, session_id, now_ms)
    }

    fn observe(&mut self, observation: KvBatch, now_ms: f64) -> anyhow::Result<Vec<Placement>> {
        self.events
            .borrow_mut()
            .extend(observation.0.iter().cloned());
        self.inner.observe(observation, now_ms)
    }

    fn cancel_pending(&mut self, request_id: Uuid) -> bool {
        self.inner.cancel_pending(request_id)
    }

    fn request_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        self.inner.request_terminal(request_id, now_ms)
    }

    fn prefill_completed(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
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

struct SessionRecordingComposition {
    seen: SeenSessions,
    events: Rc<RefCell<Vec<(WorkerStage, KvEvent)>>>,
}

impl SessionRecordingComposition {
    fn placement<P>(&self, stage: &'static str, inner: P) -> SessionRecordingPlacement<P> {
        SessionRecordingPlacement {
            stage,
            inner,
            seen: Rc::clone(&self.seen),
            events: Rc::clone(&self.events),
        }
    }
}

impl ReplayComposition for SessionRecordingComposition {
    type Metadata = NoReplayMetadata;
    type Observation = KvObservation;
    type AggregatedPlacement = SessionRecordingPlacement<AggregatedRoundRobinPlacement<KvBatch>>;
    type DisaggregatedPlacement = SessionRecordingPlacement<PoolRoundRobinPlacement<KvBatch>>;

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<Self::AggregatedPlacement> {
        Ok(self.placement(
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
            self.placement("prefill", PoolRoundRobinPlacement::new(prefill_topology)),
            self.placement("decode", PoolRoundRobinPlacement::new(decode_topology)),
        ))
    }
}

fn spec(disagg: bool, max_in_flight: Option<usize>) -> ReplaySpec {
    let engine = ReplayEngineConfig {
        rank: EngineConfig {
            block_size: 4,
            num_gpu_blocks: 64,
            max_num_seqs: 4,
            max_num_batched_tokens: 64,
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 1.0,
                decode_ms: 1.0,
            },
            ..EngineConfig::for_backend(Backend::Vllm)
        },
        ..Default::default()
    };
    ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
        topology: if disagg {
            ReplayTopology::Disaggregated {
                prefill: WorkerPoolSpec::default(),
                decode: WorkerPoolSpec::default(),
                handoff_latency_ms: 0.0,
            }
        } else {
            ReplayTopology::aggregated(2)
        },
        engine: serde_json::to_value(engine).unwrap(),
        adapters: Default::default(),
        max_sim_time_ms: None,
        max_in_flight,
        record_per_request: true,
        sla: Default::default(),
        requests: Vec::new(),
    }
}

fn row(session_id: Option<&str>, timestamp: Option<f64>, hash_ids: Vec<u64>) -> MooncakeRow {
    MooncakeRow {
        session_id: session_id.map(str::to_string),
        input_length: Some(4 * hash_ids.len()),
        output_length: Some(2),
        hash_ids: Some(hash_ids),
        timestamp,
        delay: timestamp.is_none().then_some(1.0),
        ..Default::default()
    }
}

/// Two turns of authored session `a`, interleaved with rows on lines 2 and 4
/// that carry no session ID.
fn mixed_trace() -> Trace {
    Trace::from_mooncake_rows(
        vec![
            row(Some("a"), Some(0.0), vec![1, 2]),
            row(None, Some(1.0), vec![1, 3]),
            row(Some("a"), None, vec![1, 2, 4]),
            row(None, Some(2.0), vec![1, 5]),
        ],
        4,
    )
    .unwrap()
}

#[test]
fn placement_sees_only_authored_session_ids_in_open_and_closed_loop() {
    for disagg in [false, true] {
        for max_in_flight in [None, Some(2)] {
            let case = format!("disagg={disagg} max_in_flight={max_in_flight:?}");
            let driver = match max_in_flight {
                Some(limit) => mixed_trace().into_concurrency_driver_with_block_size(4, limit),
                None => mixed_trace().into_trace_driver_with_block_size(4),
            }
            .unwrap();
            let seen = SeenSessions::default();
            let events = Rc::new(RefCell::new(Vec::new()));
            let report = Replayer::with_composition(
                spec(disagg, max_in_flight),
                ReplayEngineFactory::new(),
                SessionRecordingComposition {
                    seen: Rc::clone(&seen),
                    events: Rc::clone(&events),
                },
            )
            .unwrap()
            .with_runtime_input(ReplayRuntimeInput::Workload(driver))
            .run()
            .unwrap();
            assert_eq!(report.request_counts.completed_requests, 4, "{case}");

            let stages: &[&str] = if disagg {
                &["prefill", "decode"]
            } else {
                &["aggregated"]
            };
            for stage in stages {
                let mut sessions = seen
                    .borrow()
                    .iter()
                    .filter(|(seen_stage, _)| seen_stage == stage)
                    .map(|(_, session_id)| session_id.clone())
                    .collect::<Vec<_>>();
                sessions.sort();
                assert_eq!(
                    sessions,
                    [None, None, Some("a".to_string()), Some("a".to_string())],
                    "{case} stage={stage}"
                );
            }

            // Legacy multi-turn sessions are normalized into ReplayRequestContext
            // at ingress, without using synthetic row IDs or inventing authored IDs.
            assert!(
                report
                    .per_request
                    .iter()
                    .all(|record| record.request_id.is_none())
            );
            let stores = events.borrow();
            let sessions: std::collections::BTreeSet<_> = stores
                .iter()
                .filter_map(|(_, event)| match &event.data {
                    KvEventData::Stored(stored) => Some(stored.session_id.as_deref()),
                    _ => None,
                })
                .collect();
            assert_eq!(sessions, [None, Some("a")].into_iter().collect(), "{case}");

            // Rows without a session ID keep their per-request identity.
            let mut identities = report
                .per_request
                .iter()
                .map(|record| (record.session_id.clone(), record.turn_index))
                .collect::<Vec<_>>();
            identities.sort();
            assert_eq!(
                identities,
                [
                    (Some("a".to_string()), Some(0)),
                    (Some("a".to_string()), Some(1)),
                    (Some("request_2".to_string()), Some(0)),
                    (Some("request_4".to_string()), Some(0)),
                ],
                "{case}"
            );
        }
    }
}

#[test]
fn session_only_replay_context_reaches_placement_reporting_and_stores() {
    for (input, expected) in [
        (Some("session-only"), Some("session-only")),
        (Some("  trimmed  "), Some("trimmed")),
        (Some("  "), None),
        (None, None),
    ] {
        for disagg in [false, true] {
            for max_in_flight in [None, Some(1)] {
                let request = serde_json::from_value(serde_json::json!({
                    "tokens": [1, 2, 3, 4, 5, 6, 7, 8],
                    "max_output_tokens": 2,
                    "uuid": Uuid::from_u128(1),
                    "dp_rank": 0,
                    "arrival_timestamp_ms": 0.0,
                    "replay_context": {"session_id": input}
                }))
                .unwrap();
                let seen = SeenSessions::default();
                let events = Rc::new(RefCell::new(Vec::new()));
                let report = Replayer::with_composition(
                    spec(disagg, max_in_flight),
                    ReplayEngineFactory::new(),
                    SessionRecordingComposition {
                        seen: Rc::clone(&seen),
                        events: Rc::clone(&events),
                    },
                )
                .unwrap()
                .with_runtime_input(ReplayRuntimeInput::Requests(vec![request].into()))
                .run()
                .unwrap();
                assert_eq!(report.request_counts.completed_requests, 1);
                assert_eq!(report.per_request[0].session_id.as_deref(), expected);
                assert_eq!(report.per_request[0].request_id, None);
                let seen = seen.borrow();
                assert_eq!(seen.len(), if disagg { 2 } else { 1 });
                assert!(seen.iter().all(|(_, id)| id.as_deref() == expected));
                let events = events.borrow();
                let stores: Vec<_> = events
                    .iter()
                    .filter_map(|(_, event)| match &event.data {
                        KvEventData::Stored(stored) => Some(stored),
                        _ => None,
                    })
                    .collect();
                assert!(!stores.is_empty());
                assert!(
                    stores
                        .iter()
                        .all(|stored| stored.session_id.as_deref() == expected)
                );
            }
        }
    }
}

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
        _worker: usize,
        _rank: u32,
        events: Vec<KvEvent>,
    ) -> KvBatch {
        KvBatch(events.into_iter().map(|event| (stage, event)).collect())
    }
}

#[test]
fn session_metadata_reaches_stores_without_changing_execution() {
    for backend in [Backend::Vllm, Backend::Sglang, Backend::Trtllm] {
        for disagg in [false, true] {
            if backend == Backend::Trtllm && disagg {
                continue;
            }
            let run = |attributed: bool| {
                let mut spec = spec(disagg, None);
                let mut config: ReplayEngineConfig = serde_json::from_value(spec.engine).unwrap();
                config.rank.backend = backend;
                config.rank.max_num_seqs = 1;
                config.rank.emit_kv_events = true;
                config.rank.emit_kv_token_ids = true;
                spec.engine = serde_json::to_value(config).unwrap();
                spec.requests = (0..3)
                    .map(|i| ReplayRequest {
                        id: format!("request-{i}"),
                        arrival_time_ms: 0.0,
                        input_tokens: 16,
                        input_token_ids: Some(vec![10 + i; 16]),
                        output_tokens: 8,
                        output_token_ids: Some(vec![20 + i; 8]),
                        dp_rank: None,
                        prefill_dp_rank: None,
                        turn_index: None,
                        metadata: serde_json::Value::Null,
                        session_id: (attributed && i < 2).then(|| format!("session-{i}")),
                    })
                    .collect();
                let events = Rc::new(RefCell::new(Vec::new()));
                let report = Replayer::with_composition(
                    spec,
                    ReplayEngineFactory::new(),
                    SessionRecordingComposition {
                        seen: SeenSessions::default(),
                        events: Rc::clone(&events),
                    },
                )
                .unwrap()
                .run()
                .unwrap();
                assert_eq!(report.request_counts.completed_requests, 3);
                let timings = report
                    .per_request
                    .iter()
                    .map(|r| (r.ttft_ms, r.e2e_latency_ms))
                    .collect::<Vec<_>>();
                let events = events.borrow().clone();
                (events, timings)
            };
            let (mut attributed, timings) = run(true);
            let (plain, plain_timings) = run(false);
            let stages: &[_] = if disagg {
                &[WorkerStage::Prefill, WorkerStage::Decode]
            } else {
                &[WorkerStage::Aggregated]
            };
            for stage in stages {
                let sessions: std::collections::BTreeSet<_> = attributed
                    .iter()
                    .filter(|(seen_stage, _)| seen_stage == stage)
                    .filter_map(|(_, event)| match &event.data {
                        KvEventData::Stored(store) => Some(store.session_id.clone()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    sessions,
                    [None, Some("session-0".into()), Some("session-1".into())]
                        .into_iter()
                        .collect(),
                    "{backend:?} stage={stage:?}"
                );
            }
            for (_, event) in &mut attributed {
                if let KvEventData::Stored(store) = &mut event.data {
                    for block in &store.blocks {
                        let tokens = block.token_ids.as_ref().expect("token capture enabled");
                        let request_index = tokens[0] % 10;
                        let expected =
                            (request_index < 2).then(|| format!("session-{request_index}"));
                        assert_eq!(
                            store.session_id, expected,
                            "store attributed to another request"
                        );
                    }
                    store.session_id = None;
                }
            }
            assert_eq!(
                attributed, plain,
                "session must not change KV identities or event order: {backend:?} disagg={disagg}"
            );
            assert_eq!(
                timings, plain_timings,
                "session must not change virtual timing: {backend:?} disagg={disagg}"
            );
        }
    }
}

#[test]
fn session_identity_survives_disabled_reporting_without_fabricating_anonymous_ids() {
    use std::io::Write;
    for agentic in [false, true] {
        for authored_session in [Some("authored-session"), None] {
            for disagg in [false, true] {
                let driver = if agentic {
                    let mut file = tempfile::NamedTempFile::new().unwrap();
                    writeln!(
                        file,
                        "{}",
                        serde_json::json!({
                            "request_id": "root", "session_id": authored_session,
                            "timestamp": 0.0, "input_length": 8,
                            "output_length": 2, "hash_ids": [1, 2]
                        })
                    )
                    .unwrap();
                    load_agentic_mooncake(file.path(), 4)
                        .unwrap()
                        .into_trace_driver_with_block_size(4)
                        .unwrap()
                } else {
                    Trace::from_mooncake_rows(vec![row(authored_session, Some(0.0), vec![1, 2])], 4)
                        .unwrap()
                        .into_trace_driver_with_block_size(4)
                        .unwrap()
                }
                .without_session_metadata();
                let seen = SeenSessions::default();
                let events = Rc::new(RefCell::new(Vec::new()));
                let report = Replayer::with_composition(
                    spec(disagg, None),
                    ReplayEngineFactory::new(),
                    SessionRecordingComposition {
                        seen: Rc::clone(&seen),
                        events: Rc::clone(&events),
                    },
                )
                .unwrap()
                .with_runtime_input(ReplayRuntimeInput::Workload(driver))
                .run()
                .unwrap();
                assert_eq!(report.request_counts.completed_requests, 1);
                assert_eq!(seen.borrow().len(), if disagg { 2 } else { 1 });
                assert!(
                    seen.borrow()
                        .iter()
                        .all(|(_, id)| id.as_deref() == authored_session),
                    "agentic={agentic}, authored={authored_session:?}"
                );
                let events = events.borrow();
                let stores: Vec<_> = events
                    .iter()
                    .filter_map(|(_, event)| match &event.data {
                        KvEventData::Stored(stored) => Some(stored),
                        _ => None,
                    })
                    .collect();
                assert!(!stores.is_empty());
                assert!(
                    stores
                        .iter()
                        .all(|stored| stored.session_id.as_deref() == authored_session),
                    "agentic={agentic}, authored={authored_session:?}"
                );
            }
        }
    }
}
