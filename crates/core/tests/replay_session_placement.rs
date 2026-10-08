// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::rc::Rc;

use aisimulate_core::engine::{Backend, EngineConfig, TimingModelConfig};
use aisimulate_core::replay::loadgen::{MooncakeRow, ReplayRequestPayload, Trace};
use aisimulate_core::replay::{
    AggregatedRoundRobinPlacement, CURRENT_REPLAY_SPEC_VERSION, NoEngineEvents, NoReplayMetadata,
    Placement, PlacementEffects, PlacementPolicy, PoolRoundRobinPlacement, ReplayComposition,
    ReplayEngineConfig, ReplayEngineFactory, ReplayRuntimeInput, ReplaySpec, ReplayTopology,
    Replayer, WorkerPoolSpec, WorkerTopology,
};
use uuid::Uuid;

type SeenSessions = Rc<RefCell<Vec<(&'static str, Option<String>)>>>;

/// Delegates to round-robin placement and records the session each call sees.
struct SessionRecordingPlacement<P> {
    stage: &'static str,
    inner: P,
    seen: SeenSessions,
}

impl<P> PlacementPolicy<ReplayRequestPayload> for SessionRecordingPlacement<P>
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
        self.seen
            .borrow_mut()
            .push((self.stage, session_id.clone()));
        self.inner.place(request, metadata, session_id, now_ms)
    }

    fn observe(&mut self, observation: (), now_ms: f64) -> anyhow::Result<Vec<Placement>> {
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
}

impl SessionRecordingComposition {
    fn placement<P>(&self, stage: &'static str, inner: P) -> SessionRecordingPlacement<P> {
        SessionRecordingPlacement {
            stage,
            inner,
            seen: Rc::clone(&self.seen),
        }
    }
}

impl ReplayComposition for SessionRecordingComposition {
    type Metadata = NoReplayMetadata;
    type Observation = NoEngineEvents;
    type AggregatedPlacement = SessionRecordingPlacement<AggregatedRoundRobinPlacement<()>>;
    type DisaggregatedPlacement = SessionRecordingPlacement<PoolRoundRobinPlacement<()>>;

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
            let report = Replayer::with_composition(
                spec(disagg, max_in_flight),
                ReplayEngineFactory::new(),
                SessionRecordingComposition {
                    seen: Rc::clone(&seen),
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
