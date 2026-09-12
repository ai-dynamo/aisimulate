// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use super::*;
use crate::engine::{EngineConfig, TimingModelConfig};
use crate::replay::components::NoReplayMetadata;
use crate::replay::core::NoEngineEvents;
use crate::replay::core::round_robin::AggregatedRoundRobinPlacement;
use crate::replay::engine::{ReplayEngineConfig, ReplayEngineFactory};
use crate::replay::{ReplayTelemetryObserver, ReplayTelemetrySnapshot, WorkerStage};

type RoundRobinAggRuntime =
    AggRuntimeImpl<AggregatedRoundRobinPlacement<()>, NoEngineEvents, NoReplayMetadata>;

struct NoopTelemetryObserver;

impl ReplayTelemetryObserver for NoopTelemetryObserver {
    fn on_sample(&mut self, _snapshot: ReplayTelemetrySnapshot) -> anyhow::Result<()> {
        Ok(())
    }
}

fn request(uuid: u128, arrival_ms: f64) -> DirectRequest {
    DirectRequest {
        tokens: vec![1; 4],
        max_output_tokens: 1,
        uuid: Some(Uuid::from_u128(uuid)),
        arrival_timestamp_ms: Some(arrival_ms),
        ..Default::default()
    }
}

fn runtime(pending: VecDeque<DirectRequest>) -> RoundRobinAggRuntime {
    let config = ReplayEngineConfig {
        rank: EngineConfig {
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 1.0,
                decode_ms: 1.0,
            },
            ..EngineConfig::default()
        },
        ..ReplayEngineConfig::default()
    };
    let role_factory = ReplayEngineFactory::new()
        .role_factory(&config, WorkerStage::Aggregated, false)
        .unwrap();
    RoundRobinAggRuntime::new_composed(
        role_factory,
        AdmissionQueue::new_requests(pending, ReplayMode::Trace),
        1,
        None,
        |dp_size, topology| Ok(AggregatedRoundRobinPlacement::new(dp_size, topology)),
    )
    .unwrap()
}

#[test]
fn telemetry_only_timestamps_do_not_enter_the_agg_semantic_drain() {
    let pending = VecDeque::from([request(1, 10.0)]);
    let (_, baseline_stats) = runtime(pending.clone()).run().unwrap();
    let (_, observed_stats) = runtime(pending)
        .with_telemetry_observer(1.0, Box::new(NoopTelemetryObserver))
        .run()
        .unwrap();

    // The floor only has to be above "the run did nothing": one request needs at
    // least an arrival drain and a completion drain, so anything greater than one
    // proves the baseline actually drove semantic work for the comparison below
    // to be meaningful.
    assert!(baseline_stats.semantic_drain_count > 1);
    assert_eq!(
        observed_stats.semantic_drain_count, baseline_stats.semantic_drain_count,
        "telemetry-only heartbeats must not wake aggregate semantic replay work"
    );
}

/// Places every request onto a scheduler id the engine does not have.
struct UnknownSchedulerPlacement;

impl PlacementPolicy<ReplayRequestPayload> for UnknownSchedulerPlacement {
    type Metadata = NoReplayMetadata;
    type Observation = ();

    fn place(
        &mut self,
        request: &ReplayRequestPayload,
        _metadata: Self::Metadata,
        _session_id: Option<String>,
        _now_ms: f64,
    ) -> anyhow::Result<crate::replay::PlacementEffects> {
        Ok(crate::replay::PlacementEffects {
            decision: PlacementDecision::Immediate(Placement {
                request_id: request.metadata().uuid.unwrap(),
                scheduler_id: 4_242,
                reported_overlap_tokens: 0,
                cache_sample: None,
                placement_replica_id: None,
            }),
            released: Vec::new(),
        })
    }

    fn observe(&mut self, _: (), _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn cancel_pending(&mut self, _: Uuid) -> bool {
        false
    }

    fn request_terminal(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn prefill_completed(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn pending_count(&self) -> usize {
        0
    }

    fn worker_ready(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn worker_draining(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn worker_removed(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn topology_settled(&mut self, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }
}

/// A placement naming an unknown scheduler must leave no accounting behind.
///
/// Same invariant as `mismatched_placement_does_not_retain_arrival_or_offered_traffic`,
/// and the same class of policy failure -- but the scheduler lookup was never
/// hoisted above the accounting, so it fired after `on_arrival`,
/// `on_request_context`, `traffic.on_arrival()` and `record_placement`'s
/// hit-rate sample had already landed.
#[test]
fn unknown_scheduler_placement_does_not_retain_arrival_or_offered_traffic() {
    let role_factory = ReplayEngineFactory::new()
        .role_factory(
            &ReplayEngineConfig::default(),
            WorkerStage::Aggregated,
            false,
        )
        .unwrap();
    let mut runtime =
        AggRuntimeImpl::<UnknownSchedulerPlacement, NoEngineEvents, NoReplayMetadata>::new_composed(
            role_factory,
            AdmissionQueue::new_requests(VecDeque::new(), ReplayMode::Trace),
            1,
            None,
            |_, _| Ok(UnknownSchedulerPlacement),
        )
        .unwrap()
        .into_steppable();

    let uuid = Uuid::from_u128(1);
    let error = runtime.submit_dynamic(request(1, 0.0)).unwrap_err();
    assert!(error.to_string().contains("unknown scheduler"), "{error}");
    assert!(!runtime.collector.contains_request(uuid));
    assert_eq!(runtime.traffic.drain_planner(1_000.0).num_req, 0);
    assert!(runtime.requests.is_empty());
    assert_eq!(runtime.cluster_in_flight(), 0);
}

/// Returns a corrupt `next_tick_ms`, standing in for a PyO3 scaling policy
/// whose own arithmetic produced `inf - inf`.
struct NonFiniteNextTickPolicy {
    next_tick_ms: f64,
}

impl crate::replay::ReplayScalingPolicy for NonFiniteNextTickPolicy {
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
        Ok(1.0)
    }

    fn on_tick(
        &mut self,
        _snapshot: crate::replay::ReplayScalingSnapshot,
    ) -> anyhow::Result<crate::replay::ReplayScalingDecision> {
        Ok(crate::replay::ReplayScalingDecision {
            next_tick_ms: Some(self.next_tick_ms),
            ..crate::replay::ReplayScalingDecision::default()
        })
    }
}

/// A non-finite `next_tick_ms` must fail loudly rather than quietly disabling
/// scaling and FPM collection for the rest of the run.
///
/// Only `None` means "stop re-arming"; the `<= now_ms` filter is a deliberate
/// spin guard. NaN and +-inf were silently folded into the same permanent stop
/// as an explicit `None`, so a policy arithmetic bug read back as a policy that
/// had simply chosen to stop scaling.
#[test]
fn a_non_finite_scaling_next_tick_is_rejected() {
    for next_tick_ms in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = runtime(VecDeque::from([request(1, 10.0)]))
            .with_scaling_policy(Box::new(NonFiniteNextTickPolicy { next_tick_ms }))
            .run()
            .unwrap_err();
        assert!(
            error.to_string().contains("non-finite next_tick_ms"),
            "expected a loud rejection for {next_tick_ms}, got: {error}"
        );
    }
}

struct MismatchedPlacement;

impl PlacementPolicy<ReplayRequestPayload> for MismatchedPlacement {
    type Metadata = NoReplayMetadata;
    type Observation = ();

    fn place(
        &mut self,
        _request: &ReplayRequestPayload,
        _metadata: Self::Metadata,
        _session_id: Option<String>,
        _now_ms: f64,
    ) -> anyhow::Result<crate::replay::PlacementEffects> {
        Ok(crate::replay::PlacementEffects {
            decision: PlacementDecision::Immediate(Placement {
                request_id: Uuid::from_u128(999),
                scheduler_id: 0,
                reported_overlap_tokens: 0,
                cache_sample: None,
                placement_replica_id: None,
            }),
            released: Vec::new(),
        })
    }

    fn observe(&mut self, _: (), _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn cancel_pending(&mut self, _: Uuid) -> bool {
        false
    }

    fn request_terminal(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn prefill_completed(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn pending_count(&self) -> usize {
        0
    }

    fn worker_ready(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn worker_draining(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn worker_removed(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn topology_settled(&mut self, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }
}

/// Holds every request in its pending queue and records which cancellation
/// callback the runtime delivers.
#[derive(Default)]
struct QueueingPlacement {
    pending: Vec<Uuid>,
    cancel_pending_calls: usize,
    request_terminal_calls: usize,
}

impl PlacementPolicy<ReplayRequestPayload> for QueueingPlacement {
    type Metadata = NoReplayMetadata;
    type Observation = ();

    fn place(
        &mut self,
        request: &ReplayRequestPayload,
        _metadata: Self::Metadata,
        _session_id: Option<String>,
        _now_ms: f64,
    ) -> anyhow::Result<crate::replay::PlacementEffects> {
        let request_id = request
            .metadata()
            .uuid
            .ok_or_else(|| anyhow::anyhow!("queueing placement requires a request UUID"))?;
        self.pending.push(request_id);
        Ok(crate::replay::PlacementEffects {
            decision: PlacementDecision::Queued,
            released: Vec::new(),
        })
    }

    fn observe(&mut self, _: (), _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn cancel_pending(&mut self, request_id: Uuid) -> bool {
        self.cancel_pending_calls += 1;
        let Some(index) = self.pending.iter().position(|id| *id == request_id) else {
            return false;
        };
        self.pending.remove(index);
        true
    }

    fn request_terminal(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
        self.request_terminal_calls += 1;
        Ok(Vec::new())
    }

    fn prefill_completed(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn pending_count(&self) -> usize {
        self.pending.len()
    }

    fn worker_ready(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn worker_draining(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn worker_removed(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn topology_settled(&mut self, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }
}

/// Cancelling a router-queued request must deliver `cancel_pending` alone.
///
/// A queued request was never released to a worker, so there is no placement to
/// retire. Aggregated cancel used to call `cancel_pending` and then
/// `request_terminal` unconditionally, which tells a load-tracking policy to
/// free a slot it never took. The disaggregated runtime already treats the two
/// callbacks as alternatives; the in-repo round-robin policies no-op both,
/// which is why only a recording policy exposes this.
#[test]
fn canceling_a_router_queued_request_does_not_also_deliver_request_terminal() {
    let role_factory = ReplayEngineFactory::new()
        .role_factory(
            &ReplayEngineConfig::default(),
            WorkerStage::Aggregated,
            false,
        )
        .unwrap();
    let mut runtime =
        AggRuntimeImpl::<QueueingPlacement, NoEngineEvents, NoReplayMetadata>::new_composed(
            role_factory,
            AdmissionQueue::new_requests(VecDeque::new(), ReplayMode::Trace),
            1,
            None,
            |_, _| Ok(QueueingPlacement::default()),
        )
        .unwrap()
        .into_steppable();

    let uuid = runtime.submit_dynamic(request(1, 0.0)).unwrap();
    assert_eq!(
        runtime.placement.pending_count(),
        1,
        "the request must be parked in the router queue"
    );

    assert_eq!(
        runtime.cancel_dynamic(uuid).unwrap(),
        Some(ReplayTerminalStatus::Canceled)
    );
    assert_eq!(runtime.placement.cancel_pending_calls, 1);
    assert_eq!(
        runtime.placement.request_terminal_calls, 0,
        "a request the policy never released must not receive a placement terminal"
    );
    assert_eq!(runtime.placement.pending_count(), 0);
}

/// Queues every request, and releases a scripted batch on the next `place`.
/// Like a real policy, it drops every released id out of its own pending set
/// before handing the batch back, so an abandoned release is invisible to
/// `pending_count` -- which is exactly what makes a dropped tail undercount
/// `cluster_in_flight`.
#[derive(Default)]
struct ScriptedReleasePlacement {
    pending: Vec<Uuid>,
    next_release: Vec<Placement>,
}

impl PlacementPolicy<ReplayRequestPayload> for ScriptedReleasePlacement {
    type Metadata = NoReplayMetadata;
    type Observation = ();

    fn place(
        &mut self,
        request: &ReplayRequestPayload,
        _metadata: Self::Metadata,
        _session_id: Option<String>,
        _now_ms: f64,
    ) -> anyhow::Result<crate::replay::PlacementEffects> {
        let request_id = request
            .metadata()
            .uuid
            .ok_or_else(|| anyhow::anyhow!("scripted placement requires a request UUID"))?;
        self.pending.push(request_id);
        let released = std::mem::take(&mut self.next_release);
        for placement in &released {
            self.pending.retain(|id| *id != placement.request_id);
        }
        Ok(crate::replay::PlacementEffects {
            decision: PlacementDecision::Queued,
            released,
        })
    }

    fn observe(&mut self, _: (), _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn cancel_pending(&mut self, request_id: Uuid) -> bool {
        let Some(index) = self.pending.iter().position(|id| *id == request_id) else {
            return false;
        };
        self.pending.remove(index);
        true
    }

    fn request_terminal(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn prefill_completed(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn pending_count(&self) -> usize {
        self.pending.len()
    }

    fn worker_ready(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn worker_draining(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn worker_removed(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn topology_settled(&mut self, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }
}

type ScriptedAggRuntime =
    AggRuntimeImpl<ScriptedReleasePlacement, NoEngineEvents, NoReplayMetadata>;

fn scripted_release_runtime() -> ScriptedAggRuntime {
    let role_factory = ReplayEngineFactory::new()
        .role_factory(
            &ReplayEngineConfig::default(),
            WorkerStage::Aggregated,
            false,
        )
        .unwrap();
    AggRuntimeImpl::<ScriptedReleasePlacement, NoEngineEvents, NoReplayMetadata>::new_composed(
        role_factory,
        AdmissionQueue::new_requests(VecDeque::new(), ReplayMode::Trace),
        1,
        None,
        |_, _| Ok(ScriptedReleasePlacement::default()),
    )
    .unwrap()
    .into_steppable()
}

fn placement(request_id: Uuid) -> Placement {
    Placement {
        request_id,
        scheduler_id: 0,
        reported_overlap_tokens: 0,
        cache_sample: Some(crate::replay::PlacementCacheSample {
            overlap_blocks: 2,
            best_available_overlap_blocks: 2,
            isl_blocks: 4,
        }),
        placement_replica_id: None,
    }
}

/// A failing placement in the middle of a released batch must not abandon the
/// entries behind it. The policy has already dropped every released id from its
/// pending set, so an abandoned tail is counted by neither `pending_count` nor
/// `engine.in_flight()` -- `cluster_in_flight` reads zero and `is_workload_done`
/// reports a finished run with requests still parked in `self.requests`.
#[test]
fn a_failing_release_does_not_abandon_the_rest_of_the_batch() {
    let mut runtime = scripted_release_runtime();
    let first = runtime.submit_dynamic(request(1, 0.0)).unwrap();
    let ghost = Uuid::from_u128(0xdead);

    runtime.placement.next_release = vec![placement(ghost), placement(first)];
    let error = runtime.submit_dynamic(request(2, 0.0)).unwrap_err();
    assert!(
        error.to_string().contains("missing queued request state"),
        "{error}"
    );

    assert_eq!(
        runtime.requests.get(&first).map(|state| state.phase),
        Some(crate::replay::state::AggRequestPhase::Running),
        "the placement behind the failing one must still have been dispatched"
    );
    assert_eq!(runtime.engine.in_flight(), 1);
    assert!(!runtime.is_workload_done());
}

/// The failing placement itself must contribute no report bytes. `record_placement`
/// folds a KV hit-rate sample into the traffic accumulator, so committing it
/// before the fallible steps meant a rejected release still moved the mean.
#[test]
fn a_failing_release_folds_no_kv_hit_rate_sample() {
    let mut runtime = scripted_release_runtime();
    let first = runtime.submit_dynamic(request(1, 0.0)).unwrap();

    // Releasing the same id twice: the second `take_queued_request` finds the
    // request already `Running` and bails.
    runtime.placement.next_release = vec![placement(first), placement(first)];
    let error = runtime.submit_dynamic(request(2, 0.0)).unwrap_err();
    assert!(
        error.to_string().contains("expected queued request state"),
        "{error}"
    );
    assert_eq!(
        runtime.traffic.drain_planner(1_000.0).hit_rate_count,
        1,
        "only the release that actually dispatched may contribute a hit-rate sample"
    );
}

/// Places every request immediately and fails every `observe`, standing in for
/// a boxed policy (the real Dynamo KV router arrives here as one) whose
/// observation handling raises. This runtime's own contract treats such errors
/// as survivable -- the caller keeps driving.
struct FailingObservePlacement;

impl PlacementPolicy<ReplayRequestPayload> for FailingObservePlacement {
    type Metadata = NoReplayMetadata;
    type Observation = ();

    fn place(
        &mut self,
        request: &ReplayRequestPayload,
        _metadata: Self::Metadata,
        _session_id: Option<String>,
        _now_ms: f64,
    ) -> anyhow::Result<crate::replay::PlacementEffects> {
        let request_id = request
            .metadata()
            .uuid
            .ok_or_else(|| anyhow::anyhow!("failing-observe placement requires a request UUID"))?;
        Ok(crate::replay::PlacementEffects {
            decision: PlacementDecision::Immediate(Placement {
                request_id,
                scheduler_id: 0,
                reported_overlap_tokens: 0,
                cache_sample: None,
                placement_replica_id: None,
            }),
            released: Vec::new(),
        })
    }

    fn observe(&mut self, _: (), _: f64) -> anyhow::Result<Vec<Placement>> {
        anyhow::bail!("placement policy observe failed")
    }

    fn cancel_pending(&mut self, _: Uuid) -> bool {
        false
    }

    fn request_terminal(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn prefill_completed(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn pending_count(&self) -> usize {
        0
    }

    fn worker_ready(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn worker_draining(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn worker_removed(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn topology_settled(&mut self, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }
}

/// A failure after the engine-side cancel must not strand a `Running` entry.
///
/// `apply_command` releases the request inside the engine; the steps after it
/// are survivable policy failures. Leaving the runtime's own entry behind made a
/// second `cancel_dynamic` take the `Running` arm again and re-issue
/// `CancelRequest` to an engine that no longer had the request -- accepted as
/// `CommandResult::Noop`, so one request received two `Canceled` terminals.
#[test]
fn a_failed_cancel_observation_does_not_strand_a_running_request() {
    let role_factory = ReplayEngineFactory::new()
        .role_factory(
            &ReplayEngineConfig::default(),
            WorkerStage::Aggregated,
            false,
        )
        .unwrap();
    let mut runtime =
        AggRuntimeImpl::<FailingObservePlacement, NoEngineEvents, NoReplayMetadata>::new_composed(
            role_factory,
            AdmissionQueue::new_requests(VecDeque::new(), ReplayMode::Trace),
            1,
            None,
            |_, _| Ok(FailingObservePlacement),
        )
        .unwrap()
        .into_steppable();

    let uuid = runtime.submit_dynamic(request(1, 0.0)).unwrap();
    let error = runtime.cancel_dynamic(uuid).unwrap_err();
    assert!(error.to_string().contains("observe failed"), "{error}");

    assert!(
        !runtime.requests.contains_key(&uuid),
        "the engine no longer owns the request, so neither may the runtime"
    );
    assert_eq!(
        runtime.cancel_dynamic(uuid).unwrap(),
        None,
        "a cancelled request must not be cancellable a second time"
    );
}

#[test]
fn mismatched_placement_does_not_retain_arrival_or_offered_traffic() {
    let role_factory = ReplayEngineFactory::new()
        .role_factory(
            &ReplayEngineConfig::default(),
            WorkerStage::Aggregated,
            false,
        )
        .unwrap();
    let mut runtime =
        AggRuntimeImpl::<MismatchedPlacement, NoEngineEvents, NoReplayMetadata>::new_composed(
            role_factory,
            AdmissionQueue::new_requests(VecDeque::new(), ReplayMode::Trace),
            1,
            None,
            |_, _| Ok(MismatchedPlacement),
        )
        .unwrap()
        .into_steppable();
    let uuid = Uuid::from_u128(1);
    let error = runtime.submit_dynamic(request(1, 0.0)).unwrap_err();
    assert!(error.to_string().contains("while placing"), "{error}");
    assert!(!runtime.collector.contains_request(uuid));
    assert_eq!(runtime.traffic.drain_planner(1_000.0).num_req, 0);
    assert!(runtime.requests.is_empty());
    assert_eq!(runtime.cluster_in_flight(), 0);
    assert!(runtime.cancel_dynamic(uuid).unwrap().is_none());
    assert_eq!(
        runtime
            .take_report_dynamic(0.0)
            .unwrap()
            .request_counts
            .num_requests,
        0
    );
}
