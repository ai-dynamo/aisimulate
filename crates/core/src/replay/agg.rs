// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::artifact::{ReplayArtifactRequest, ReplayArtifactSink};
pub(super) use super::components::ReplayMode;
use super::core::{
    AdmissionSource as CoreAdmissionSource, EngineEventBatch, Placement, PlacementDecision,
    PlacementPolicy, WorkerTopology,
};
use super::events::{SimulationEvent, SimulationWorkerStage, WorkerCompletionPayload};
use super::evidence::{
    KvIngestBoundary, ReplayEvidenceCollector, WorkerLifecycleTransition,
    WorkerLifecycleTransitionKind, WorkerPool, WorkerPoolState, common_origin,
};
use super::progress::ReplayProgress;
use super::runtime_utils::{
    ReplayStepOutcome, next_non_telemetry_event_ms, next_timestamp as choose_next_timestamp,
    pop_ready_scaling_tick, pop_ready_telemetry_tick, pop_ready_worker_completions,
    pop_ready_worker_ready, push_scaling_tick, push_telemetry_tick, push_worker_completions,
    push_worker_ready,
};
use super::scaling::{LatestFpmBuffer, ReplayScalingPolicy, ReplayScalingSnapshot};
use super::telemetry::{
    ReplaySchedulerIntervalMetrics, ReplayTelemetryObserver, ReplayTelemetryRuntime,
    ReplayTelemetrySampleKind, ReplayTelemetrySnapshot, ReplayTrafficMetricsSnapshot,
};
use super::{
    components::{
        AdmissionQueue, EngineComponent, EngineEffects, EnginePassMode, ReplayAdmissionMetadata,
        ReplayEngineObservation, ReplayReadyArrival, TrafficAccumulators,
    },
    state::AggRequestState,
};
use crate::engine::{Command, CommandResult};
use crate::replay::engine::ReplayRoleFactory;
use crate::replay::loadgen::{AgenticPreparationTransition, ReplayRequestPayload};
use crate::replay::protocol::{DirectRequest, ForwardPassSnapshot, OutputSignal};
use crate::replay::{ReplayCaptureOptions, ReplayRequestPool};
use crate::replay::{ReplayTerminalStatus, TraceCollector};
use anyhow::{Context, bail};
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;
use uuid::Uuid;

const MAX_CONSECUTIVE_INTERNAL_STEPS: usize = 1024;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct AggRuntimeStats {
    #[cfg(test)]
    semantic_drain_count: usize,
}

pub(crate) struct AggRuntimeImpl<PlacementPolicyImpl, Observation, Metadata>
where
    Observation: ReplayEngineObservation,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl: PlacementPolicy<ReplayRequestPayload, Metadata = Metadata, Observation = Observation::Batch>,
{
    now_ms: f64,
    dp_size: u32,
    next_event_seq: u64,
    next_scaling_tick_ordinal: u64,
    admission: AdmissionQueue<Metadata>,
    requests: FxHashMap<Uuid, AggRequestState>,
    engine: EngineComponent<Observation>,
    collector: TraceCollector,
    artifact_sink: Option<ReplayArtifactSink>,
    evidence: ReplayEvidenceCollector,
    events: BinaryHeap<SimulationEvent<Observation::Batch>>,
    placement: PlacementPolicyImpl,
    progress: ReplayProgress,
    stats: AggRuntimeStats,
    /// Latest forward pass metric per worker/rank since the previous scaling tick.
    fpm_buffer: LatestFpmBuffer,
    /// Traffic statistics accumulated between scaling ticks.
    traffic: TrafficAccumulators,
    /// Optional cap on simulated wall-clock time. When set, `run()` exits
    /// gracefully once the next scheduled timestamp exceeds this cap, leaving
    /// any in-flight requests as incomplete in the report.
    max_sim_time_ms: Option<f64>,
    /// Optional scaling component. When set, `run()` seeds recurring `ScalingTick` events.
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    /// Optional policy-neutral virtual-time telemetry sampler.
    telemetry: Option<ReplayTelemetryRuntime>,
    /// Whether to retain the latest FPM snapshot per worker/rank. Only the planner
    /// consumes them, so the plain `run()` path leaves this `false`.
    collect_fpm: bool,
    /// Output tokens observed since the last steppable step, in emission order.
    /// Only the steppable seam drains this; `run()` leaves it empty.
    step_tokens: Vec<(Uuid, u32)>,
    step_terminals: Vec<(Uuid, ReplayTerminalStatus)>,
    /// Whether a terminal settled during the current steppable step, freeing an
    /// in-flight slot. `step_tokens` alone cannot carry this fact: a terminal
    /// signal may carry no token — an admission rejection, or a terminal whose
    /// last token was emitted by an earlier signal — and the caller must still
    /// get that instant back to submit a replacement before time advances.
    step_freed_slot: bool,
    /// Whether the steppable seam's evaluate/commit delta cycle is in use.
    defer_drive: bool,
    /// Set between the halves of that delta cycle: the next step commits the
    /// admissions and worker drives the previous step deferred.
    drive_pending: bool,
    drive_started: bool,
    drive_finalized: bool,
    profile_observers_started: bool,
    profile_cancel_started: bool,
    profile_canceled_requests: usize,
    profile_unsettled_requests: usize,
}

impl<PlacementPolicyImpl, Observation, Metadata>
    AggRuntimeImpl<PlacementPolicyImpl, Observation, Metadata>
where
    Observation: ReplayEngineObservation,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl: PlacementPolicy<ReplayRequestPayload, Metadata = Metadata, Observation = Observation::Batch>,
{
    pub(crate) fn new_composed(
        factory: ReplayRoleFactory,
        admission: AdmissionQueue<Metadata>,
        num_workers: usize,
        startup_time_ms: Option<f64>,
        create_placement: impl FnOnce(u32, Vec<WorkerTopology>) -> anyhow::Result<PlacementPolicyImpl>,
    ) -> anyhow::Result<Self> {
        let progress = ReplayProgress::new(
            CoreAdmissionSource::total_requests(&admission),
            "offline replay",
        );
        let dp_size = factory.dp_size();
        let gpus_per_worker = factory.gpus_per_worker()?;
        let engine = EngineComponent::<Observation>::new_with_factory(
            SimulationWorkerStage::Aggregated,
            EnginePassMode::Visible,
            factory,
            num_workers,
            startup_time_ms,
        )?;
        let placement = create_placement(dp_size, engine.active_topology())?;

        // Aggregated replay has a single (decode) pool; record its GPUs/worker
        // so the report can express GPU-hours from the mocker's own parallelism.
        let mut collector = TraceCollector::default();
        collector.set_gpus_per_worker(0, gpus_per_worker);

        Ok(Self {
            now_ms: 0.0,
            dp_size,
            next_event_seq: 0,
            next_scaling_tick_ordinal: 0,
            admission,
            requests: FxHashMap::default(),
            engine,
            collector,
            artifact_sink: None,
            evidence: ReplayEvidenceCollector::default(),
            events: BinaryHeap::new(),
            placement,
            progress,
            stats: AggRuntimeStats::default(),
            fpm_buffer: LatestFpmBuffer::default(),
            traffic: TrafficAccumulators::new(),
            max_sim_time_ms: None,
            scaling_policy: None,
            telemetry: None,
            collect_fpm: false,
            step_tokens: Vec::new(),
            step_terminals: Vec::new(),
            step_freed_slot: false,
            defer_drive: false,
            drive_pending: false,
            drive_started: false,
            drive_finalized: false,
            profile_observers_started: false,
            profile_cancel_started: false,
            profile_canceled_requests: 0,
            profile_unsettled_requests: 0,
        })
    }

    pub(crate) fn with_sla_thresholds(mut self, sla: crate::replay::SlaThresholds) -> Self {
        self.collector.set_sla_thresholds(sla);
        self
    }

    /// Toggle per-request record capture on the underlying collector. When
    /// `true`, the final `ReplayReport` returned from `run()` will
    /// have `per_request` populated. Default `false` (cheap).
    pub(crate) fn with_per_request_records(mut self, capture: bool) -> Self {
        self.collector.set_capture_per_request(capture);
        self
    }

    pub(crate) fn with_capture_options(mut self, options: ReplayCaptureOptions) -> Self {
        self.evidence = ReplayEvidenceCollector::new(options);
        self
    }

    pub(crate) fn with_artifact_sink(mut self, sink: ReplayArtifactSink) -> Self {
        self.engine.set_artifact_kv_capture(true);
        self.engine
            .set_host_offload_observer(sink.host_offload_observer());
        self.artifact_sink = Some(sink);
        self
    }

    /// Cap the simulated wall-clock duration. After construction, call this to
    /// have `run()` stop gracefully once the simulated clock would exceed
    /// `ms`. Pass `None` to run to natural completion (the default).
    ///
    /// max_sim_time_ms is a **soft cap** on the scheduling loop, not a hard truncation
    /// of recorded work. When the next scheduled simulated timestamp would
    /// exceed the cap, the loop exits, but worker passes already in flight
    /// complete normally — even if their token timestamps land past `ms`.
    /// Requests that hadn't received their first token before the cap fired
    /// stay in the report as incomplete (`first_token_ms = None`,
    /// `e2e_latency_ms = None`). `report.duration_ms` may exceed `ms` by up
    /// to one in-flight pass's duration. Enforcing a precise cap would
    /// require plumbing a deadline into the worker / engine core; not worth
    /// it for the calibration use case this exists to serve.
    pub(crate) fn with_max_sim_time_ms(mut self, ms: Option<f64>) -> Self {
        self.max_sim_time_ms = ms;
        self
    }

    /// Attach a scaling policy and enable tick-scoped FPM collection.
    pub(crate) fn with_scaling_policy(mut self, policy: Box<dyn ReplayScalingPolicy>) -> Self {
        self.collect_fpm = true;
        for worker_id in self.engine.active_group_ids() {
            self.fpm_buffer
                .activate_worker(worker_id, self.dp_size, self.now_ms);
        }
        self.scaling_policy = Some(policy);
        self
    }

    pub(crate) fn with_telemetry_observer(
        mut self,
        sample_interval_ms: f64,
        observer: Box<dyn ReplayTelemetryObserver>,
    ) -> Self {
        debug_assert!(sample_interval_ms.is_finite() && sample_interval_ms > 0.0);
        self.engine.enable_telemetry();
        self.traffic.enable_telemetry();
        self.telemetry = Some(ReplayTelemetryRuntime::new(sample_interval_ms, observer));
        self
    }

    /// Count all requests currently consuming cluster capacity, including router-queued ones.
    pub(crate) fn cluster_in_flight(&self) -> usize {
        self.engine.in_flight() + self.placement.pending_count()
    }

    /// Preserve the live `(worker_id, dp_rank)` identity when forwarding a
    /// rank-local scheduler snapshot to the scaling policy.
    fn record_fpm(
        &mut self,
        rank_id: usize,
        mut snapshot: ForwardPassSnapshot,
    ) -> anyhow::Result<()> {
        let (worker_id, dp_rank) = self.engine.rank_identity(rank_id).ok_or_else(|| {
            anyhow::anyhow!("offline replay FPM references unknown rank scheduler {rank_id}")
        })?;
        snapshot.worker_id = worker_id.to_string();
        snapshot.dp_rank = dp_rank;
        self.fpm_buffer.insert(worker_id, snapshot, self.now_ms);
        Ok(())
    }

    /// Deliver a request to a worker and update the runtime's bookkeeping for that assignment.
    fn dispatch_to_worker(
        &mut self,
        request: DirectRequest,
        uuid: Uuid,
        worker_idx: usize,
    ) -> anyhow::Result<()> {
        self.engine.dispatch(worker_idx, request, self.now_ms)?;
        // Aggregated replay uses a single pool. Treat the assignment as the
        // decode_worker_idx so per-request records consistently carry the
        // worker that served the request; prefill_worker_idx stays None,
        // signaling "no separate prefill pool".
        self.collector.on_decode_assigned(uuid, worker_idx);
        Ok(())
    }

    fn record_placement(&mut self, placement: Placement) {
        if let Some(sample) = placement.cache_sample {
            self.traffic
                .on_admission(sample.overlap_blocks, sample.isl_blocks);
        }
    }

    /// Materialize policy-released admissions into concrete worker dispatches.
    fn dispatch_placements(&mut self, placements: Vec<Placement>) -> anyhow::Result<()> {
        for placement in placements {
            self.record_placement(placement);
            let uuid = placement.request_id;
            let (logical_worker_id, dp_rank) = self
                .engine
                .rank_identity(placement.scheduler_id)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "offline replay placement references unknown scheduler {}",
                        placement.scheduler_id
                    )
                })?;
            self.collector.on_route_released(
                uuid,
                ReplayRequestPool::Agg,
                self.now_ms,
                logical_worker_id,
                placement.scheduler_id,
                dp_rank,
                placement.reported_overlap_tokens,
                placement.cache_sample,
                placement.placement_replica_id,
            );
            let request = self
                .requests
                .get_mut(&uuid)
                .ok_or_else(|| {
                    anyhow::anyhow!("offline replay missing queued request state for {uuid}")
                })?
                .take_queued_request(uuid, placement.scheduler_id)?;
            self.dispatch_to_worker(request, uuid, placement.scheduler_id)?;
        }
        Ok(())
    }

    /// Admit one external request into the collector, optional router, and worker pool.
    fn assign_request(
        &mut self,
        mut request: ReplayRequestPayload,
        arrival_time_ms: f64,
        metadata: Metadata,
        session_id: Option<String>,
    ) -> anyhow::Result<Uuid> {
        let uuid = request.metadata().uuid.unwrap_or_else(Uuid::new_v4);
        let input_length = request.input_length();
        let output_length = request.metadata().effective_max_output_tokens();
        request.metadata_mut().uuid = Some(uuid);
        if matches!(self.admission.mode(), ReplayMode::Concurrency { .. }) {
            request.metadata_mut().arrival_timestamp_ms = Some(arrival_time_ms);
        }

        let effects = self
            .placement
            .place(&request, metadata, session_id, self.now_ms)?;
        if let PlacementDecision::Immediate(placement) = &effects.decision
            && placement.request_id != uuid
        {
            bail!(
                "offline placement returned request {} while placing {uuid}",
                placement.request_id
            );
        }
        self.collector
            .try_on_arrival(uuid, arrival_time_ms, input_length, output_length)?;
        if let Some(context) = request.metadata().replay_context.as_ref() {
            self.collector.on_request_context(uuid, context);
        }
        self.traffic.on_arrival();
        match effects.decision {
            PlacementDecision::Immediate(placement) => {
                self.record_placement(placement);
                let (logical_worker_id, dp_rank) = self
                    .engine
                    .rank_identity(placement.scheduler_id)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "offline replay placement references unknown scheduler {}",
                            placement.scheduler_id
                        )
                    })?;
                self.collector.on_route_immediate(
                    uuid,
                    ReplayRequestPool::Agg,
                    logical_worker_id,
                    placement.scheduler_id,
                    dp_rank,
                    placement.reported_overlap_tokens,
                    placement.cache_sample,
                    placement.placement_replica_id,
                );
                self.requests.insert(
                    uuid,
                    AggRequestState::new_running(
                        input_length,
                        output_length,
                        placement.scheduler_id,
                    ),
                );
                self.dispatch_to_worker(
                    request.into_direct_request(),
                    uuid,
                    placement.scheduler_id,
                )?;
            }
            PlacementDecision::Queued => {
                self.collector
                    .on_route_queued(uuid, ReplayRequestPool::Agg, self.now_ms);
                self.requests
                    .insert(uuid, AggRequestState::new_queued(request));
            }
        }
        self.dispatch_placements(effects.released)?;
        Ok(uuid)
    }

    /// Return true once no request work remains. Lingering worker/control tick
    /// events carry no work and do not
    /// keep the run alive — otherwise a recurring tick would never let `run()` exit.
    fn is_done(&self) -> bool {
        if self.admission.agentic_profile_client_complete(self.now_ms) {
            return true;
        }
        self.only_idle_events_remain()
            && self.cluster_in_flight() == 0
            && CoreAdmissionSource::is_drained(&self.admission)
            && self.engine.is_drained()
    }

    /// Return true once the request workload is complete, even if `WorkerReady`
    /// or control-tick events remain in the queue. Lingering startup events for
    /// workers that will never receive requests should not block completion.
    pub(crate) fn is_workload_done(&self) -> bool {
        !self.drive_pending
            && self.cluster_in_flight() == 0
            && CoreAdmissionSource::is_drained(&self.admission)
            && self.engine.is_drained()
            && self.only_idle_events_remain()
    }

    /// True if the event heap is empty or contains only "idle" events that carry no
    /// pending request work: `WorkerReady` or a re-armed control heartbeat.
    fn only_idle_events_remain(&self) -> bool {
        use super::events::SimulationEventKind;
        self.events.iter().all(|e| {
            matches!(
                e.kind,
                SimulationEventKind::WorkerReady { .. }
                    | SimulationEventKind::ScalingTick
                    | SimulationEventKind::TelemetryTick
            )
        })
    }

    /// Return both the next event including telemetry and the canonical next
    /// timestamp that can advance replay semantics. Independently modeled
    /// engine work, such as host transfers, is semantic for both views.
    fn next_timestamps(&mut self) -> (Option<f64>, Option<f64>) {
        // A step that freed an in-flight slot holds the instant so an
        // external caller can interleave a same-`now_ms` submission before
        // the freed worker commits to its next pass (the delta-cycle
        // invariant documented on `step_dynamic_until`). When the caller has
        // nothing to submit, this deadline is what tells it to call the
        // stepper again at the same instant rather than treating a stale
        // "no next event" reading as the end of the run.
        if self.drive_pending {
            return (Some(self.now_ms), Some(self.now_ms));
        }
        let profile_deadline =
            self.admission
                .agentic_profile_deadlines()
                .and_then(|(_, grace, cancel)| {
                    let deadline = if self.profile_cancel_started {
                        cancel
                    } else {
                        grace
                    };
                    (deadline > self.now_ms).then_some(deadline)
                });
        let next_arrival_ms = choose_next_timestamp(
            CoreAdmissionSource::next_ready_time_ms(&mut self.admission),
            profile_deadline,
        );
        let next_event_ms = self.events.peek().map(|event| event.at_ms);
        let next_canonical_event_ms = if self.telemetry.is_some() {
            next_non_telemetry_event_ms(&mut self.events)
        } else {
            next_event_ms
        };
        let next_internal_deadline_ms = self.engine.next_internal_deadline_ms();
        (
            choose_next_timestamp(
                choose_next_timestamp(next_arrival_ms, next_event_ms),
                next_internal_deadline_ms,
            ),
            choose_next_timestamp(
                choose_next_timestamp(next_arrival_ms, next_canonical_event_ms),
                next_internal_deadline_ms,
            ),
        )
    }

    /// Apply router-visible KV events at the phase chosen by the scheduler core.
    fn apply_engine_observations(
        &mut self,
        events: Observation::Batch,
        boundary: KvIngestBoundary,
    ) -> anyhow::Result<()> {
        if let Some(event_count) = Observation::kv_ingest_event_count(&events) {
            self.evidence.record_kv_ingest(
                WorkerPool::Agg,
                boundary,
                self.now_ms,
                event_count,
                |encoder| Observation::encode_kv_ingest(&events, encoder),
            )?;
        }
        let placements = self.placement.observe(events, self.now_ms)?;
        self.dispatch_placements(placements)
    }

    /// Consume one output signal, updating router state, collector state, and completion counts.
    fn process_output_signal(&mut self, signal: OutputSignal) -> anyhow::Result<()> {
        if !self.admission.is_agentic_preparing()
            && self.admission.knows_preparation_request(signal.uuid)
        {
            // Retired preparation identities never become profile requests,
            // including callbacks delivered after their quiescence barrier.
            return Ok(());
        }
        if let Some(token_id) = signal.token_id {
            self.admission.defer_output_token(signal.uuid, token_id)?;
            self.collector.on_token(signal.uuid, self.now_ms);
            if self.defer_drive {
                self.step_tokens.push((signal.uuid, token_id));
            }
        }
        if signal.completed {
            let status = if signal.rejected {
                ReplayTerminalStatus::Rejected
            } else {
                ReplayTerminalStatus::Completed
            };
            self.collector.on_terminal(signal.uuid, self.now_ms, status);
            if self.defer_drive {
                self.step_freed_slot = true;
                self.step_terminals.push((signal.uuid, status));
            }
            let placements = self.placement.request_terminal(signal.uuid, self.now_ms)?;
            let removed_state = self.requests.remove(&signal.uuid).ok_or_else(|| {
                anyhow::anyhow!("offline replay missing request state for {}", signal.uuid)
            })?;
            // Rejected requests never ran: keep them out of completed-request
            // shape and latency samples. Their offered demand was already
            // recorded at arrival, matching requests_started_total.
            if !signal.rejected {
                let latencies = self.collector.request_latencies(signal.uuid);
                let actual_output_tokens = self
                    .collector
                    .actual_output_length(signal.uuid)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "offline replay missing collector state for {}",
                            signal.uuid
                        )
                    })?;
                debug_assert!(actual_output_tokens <= removed_state.output_tokens);
                self.traffic.on_completion(
                    removed_state.input_tokens,
                    actual_output_tokens,
                    latencies,
                );
            }
            self.admission
                .defer_terminal(signal.uuid, self.now_ms, status)?;
            if !self.admission.knows_preparation_request(signal.uuid) {
                self.progress.inc_completed();
            }
            self.dispatch_placements(placements)?;
            return Ok(());
        }

        let already_marked = self
            .requests
            .get(&signal.uuid)
            .ok_or_else(|| {
                anyhow::anyhow!("offline replay missing request state for {}", signal.uuid)
            })?
            .prefill_completed;
        if already_marked {
            return Ok(());
        }

        self.requests
            .get_mut(&signal.uuid)
            .ok_or_else(|| {
                anyhow::anyhow!("offline replay missing request state for {}", signal.uuid)
            })?
            .prefill_completed = true;
        let placements = self.placement.prefill_completed(signal.uuid, self.now_ms)?;
        self.dispatch_placements(placements)?;

        Ok(())
    }

    /// Apply one completed pass: free request slots, publish KV events, and handle outputs.
    fn process_completed_pass(
        &mut self,
        _worker_idx: usize,
        _completed_requests: usize,
        output_signals: Vec<OutputSignal>,
        engine_events: Observation::Batch,
        accept_length_output_tokens: usize,
        accept_length_decode_forwards: usize,
    ) -> anyhow::Result<()> {
        self.apply_engine_observations(engine_events, KvIngestBoundary::PassEnd)?;
        self.traffic
            .on_accept_length_sample(accept_length_output_tokens, accept_length_decode_forwards);
        for signal in output_signals {
            self.process_output_signal(signal)?;
        }
        Ok(())
    }

    /// Drain all worker-completion events scheduled for the current logical timestamp.
    fn apply_worker_completions(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        // TODO: Same-time DP-rank completions settle in deterministic event order, and
        // each pass may drain router-pending work before its sibling ranks are applied.
        // Preserve this lower-rank-first tie-break for now. Atomic settlement requires
        // splitting router state mutation from pending-admission draining.
        while let Some(completion) = pop_ready_worker_completions(&mut self.events, self.now_ms) {
            for payload in self
                .engine
                .on_scheduled_completion(completion, self.now_ms)?
            {
                self.process_worker_completion_payload(payload)?;
            }
            changed = true;
        }

        Ok(changed)
    }

    fn process_worker_completion_payload(
        &mut self,
        payload: WorkerCompletionPayload<Observation::Batch>,
    ) -> anyhow::Result<()> {
        debug_assert_eq!(payload.stage, SimulationWorkerStage::Aggregated);
        if let Some(fpm) = &payload.fpm {
            self.collector
                .on_completed_prefill_work(fpm.sum_prefill_tokens);
        }
        if let Some(sink) = &self.artifact_sink {
            sink.record_pass_completion_kv_events(
                payload.pass_started_at_ms,
                self.now_ms,
                payload
                    .artifact_pass_end_kv_events
                    .as_deref()
                    .unwrap_or_default(),
            )?;
            sink.record_outputs(self.now_ms, &payload.output_signals)?;
        }
        if self.collect_fpm
            && let Some(fpm) = payload.fpm
        {
            self.record_fpm(payload.worker_idx, fpm)?;
        }
        self.process_completed_pass(
            payload.worker_idx,
            payload.completed_requests,
            payload.output_signals,
            payload.engine_events,
            payload.accept_length_output_tokens,
            payload.accept_length_decode_forwards,
        )
    }

    /// Release every admission made ready by the shared admission queue.
    fn release_ready_arrivals(&mut self) -> anyhow::Result<bool> {
        let mut released_any = false;
        let cluster_in_flight = self.cluster_in_flight();
        for ready in self.admission.drain_ready_compact(
            self.now_ms,
            cluster_in_flight,
            self.artifact_sink.is_some(),
        )? {
            let ReplayReadyArrival {
                request,
                arrival_time_ms,
                scheduled_ready_at_ms,
                authored_request_id,
                play_id,
                dispatched_at_ms,
                metadata,
                replay_hashes,
                session_id,
                turn_index,
            } = ready;
            let input_length = request.input_length();
            let output_length = request.metadata().effective_max_output_tokens();
            let session_metadata = session_id.clone().zip(turn_index);
            let uuid = self.assign_request(request, arrival_time_ms, metadata, session_id)?;
            if let (Some(request_id), Some(play_id)) = (authored_request_id, play_id) {
                self.collector
                    .on_agentic_metadata(uuid, request_id, play_id, dispatched_at_ms);
            }
            if let Some(sink) = &self.artifact_sink {
                sink.record_request(ReplayArtifactRequest {
                    request_id: uuid,
                    observed_at_ms: self.now_ms,
                    scheduled_ready_at_ms,
                    input_length,
                    output_length,
                    replay_hashes,
                })?;
            }
            if let Some((session_id, turn_index)) = session_metadata {
                self.collector
                    .on_session_metadata(uuid, session_id, turn_index);
            }
            released_any = true;
        }
        Ok(released_any)
    }

    /// Start passes on every idle worker that can make progress at the current timestamp.
    fn drive_ready_workers(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        loop {
            let effects = self
                .engine
                .drive_ready(self.now_ms, Some(&mut self.collector))?;
            if effects.is_empty() {
                return Ok(changed);
            }
            changed = true;
            self.handle_engine_effects(effects)?;
            if self.defer_drive && self.step_freed_slot {
                return Ok(changed);
            }
        }
    }

    /// Settle transfer and other deadline-driven engine work due at the
    /// current timestamp before releasing new arrivals to scheduler admission.
    fn apply_internal_work(&mut self) -> anyhow::Result<bool> {
        let effects = self.engine.process_internal_work(self.now_ms)?;
        if let (Some(sink), Some(events)) = (&self.artifact_sink, effects.artifact_kv_events) {
            sink.record_internal_kv_events(self.now_ms, &events)?;
        }
        if !effects.engine_events.is_empty() {
            self.apply_engine_observations(effects.engine_events, KvIngestBoundary::OffloadTick)?;
        }
        Ok(effects.made_progress)
    }

    fn settle_internal_work(
        &mut self,
        consecutive_internal_steps: &mut usize,
    ) -> anyhow::Result<bool> {
        let mut changed = false;
        while self.apply_internal_work()? {
            *consecutive_internal_steps = consecutive_internal_steps
                .checked_add(1)
                .context("internal-work convergence counter overflow")?;
            if *consecutive_internal_steps >= MAX_CONSECUTIVE_INTERNAL_STEPS {
                bail!(
                    "offline replay detected non-converging engine internal work at {} ms",
                    self.now_ms
                );
            }
            changed = true;
        }
        Ok(changed)
    }

    fn handle_engine_effects(
        &mut self,
        effects: EngineEffects<Observation::Batch>,
    ) -> anyhow::Result<()> {
        // Native scheduling is engine-owned now, so admissions cross the
        // generalized-engine boundary as pass-start effects. Preserve the old
        // scheduler contract by making them visible to Replay's collector at
        // the same virtual timestamp before any completion is processed.
        for admission in effects.admissions {
            let attribution = admission.cache_tier_attribution;
            self.collector.on_admit_with_tier_attribution(
                admission.uuid,
                self.now_ms,
                admission.reused_input_tokens,
                attribution,
            );
            self.collector.on_pool_admission_with_tier_attribution(
                admission.uuid,
                ReplayRequestPool::Agg,
                self.now_ms,
                admission.reused_input_tokens,
                attribution,
            );
            self.evidence
                .record_pressure_readmission(admission.uuid, WorkerPool::Agg, self.now_ms);
        }
        for pressure in effects.pressure_events {
            self.evidence.record_native_pressure(
                &mut self.collector,
                WorkerPool::Agg,
                pressure.worker_id,
                pressure.dp_rank,
                pressure.event,
            );
        }
        if let (Some(sink), Some(pass_start)) = (&self.artifact_sink, effects.artifact_pass_start) {
            sink.record_pass_start_kv_events(pass_start.at_ms, &pass_start.kv_events)?;
        }
        self.apply_engine_observations(effects.pass_start_events, KvIngestBoundary::PassStart)?;
        for payload in effects.immediate_completions {
            self.process_worker_completion_payload(payload)?;
        }
        if let Some(scheduled) = effects.scheduled_completion {
            push_worker_completions(&mut self.events, &mut self.next_event_seq, scheduled);
        }
        Ok(())
    }

    /// Activate workers whose startup period has elapsed at the current timestamp.
    fn apply_worker_ready_events(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        while let Some((stage, worker_id)) = pop_ready_worker_ready(&mut self.events, self.now_ms) {
            debug_assert_eq!(stage, SimulationWorkerStage::Aggregated);
            if self.engine.mark_worker_ready(worker_id) {
                if self.collect_fpm {
                    self.fpm_buffer
                        .activate_worker(worker_id, self.dp_size, self.now_ms);
                }
                let topology = self.engine.worker_topology(worker_id).ok_or_else(|| {
                    anyhow::anyhow!("ready worker {worker_id} has no engine topology")
                })?;
                let placements = self.placement.worker_ready(topology, self.now_ms)?;
                let mut released = placements
                    .iter()
                    .map(|placement| placement.request_id)
                    .collect::<Vec<_>>();
                self.dispatch_placements(placements)?;
                let placements = self.placement.topology_settled(self.now_ms)?;
                released.extend(placements.iter().map(|placement| placement.request_id));
                self.dispatch_placements(placements)?;
                let origin = self.evidence.startup_origin(WorkerPool::Agg, worker_id);
                let state = self.lifecycle_state();
                self.evidence.record_lifecycle_operation(
                    self.now_ms,
                    WorkerPool::Agg,
                    "worker_ready_event",
                    None,
                    origin,
                    vec![WorkerLifecycleTransition {
                        worker_id,
                        transition: WorkerLifecycleTransitionKind::WorkerReady,
                        prior_state: Some("starting"),
                        state: "active",
                        reason: None,
                        origin_operation_ordinal: origin,
                    }],
                    state,
                    released,
                );
                changed = true;
            }
            // If mark_worker_ready returned false the worker was cancelled
            // during startup (scale-down) — the stale event is silently ignored.
        }
        Ok(changed)
    }

    /// Repeatedly process all work that becomes possible without advancing logical time.
    fn drain_current_timestamp(&mut self) -> anyhow::Result<()> {
        #[cfg(test)]
        {
            self.stats.semantic_drain_count += 1;
        }
        let mut consecutive_internal_steps = 0usize;
        loop {
            let mut changed = false;
            // Settle idle deadlines first: a completion below may release a
            // queued placement and submit it to that same idle worker.
            changed |= self.settle_internal_work(&mut consecutive_internal_steps)?;
            let completed = self.apply_worker_completions()?;
            changed |= completed;
            if completed {
                // Pass completion can expose another deadline at this timestamp.
                changed |= self.settle_internal_work(&mut consecutive_internal_steps)?;
            }
            changed |= self.apply_worker_ready_events()?;
            changed |= self.admission.flush_agentic_runtime_feedback(self.now_ms)?;
            changed |= self.finish_agentic_preparation()?;
            changed |= self.admission.advance_agentic_profile(self.now_ms)?;
            changed |= self.cancel_expired_agentic_profile()?;
            changed |= self.admission.flush_agentic_runtime_feedback(self.now_ms)?;
            if self.admission.agentic_profile_client_complete(self.now_ms) {
                return Ok(());
            }
            changed |= self.release_ready_arrivals()?;
            if self.defer_drive && self.step_freed_slot {
                self.drive_pending = true;
                return Ok(());
            }
            changed |= self.drive_ready_workers()?;
            if self.defer_drive && self.step_freed_slot {
                self.drive_pending = true;
                return Ok(());
            }
            let removed = self
                .engine
                .try_remove_drained()
                .context("failed to remove drained aggregated workers")?;
            let mut released = Vec::new();
            for worker_id in &removed {
                let placements = self.placement.worker_removed(
                    WorkerTopology {
                        worker_id: *worker_id,
                        scheduler_ids: Vec::new(),
                    },
                    self.now_ms,
                )?;
                released.extend(placements.iter().map(|placement| placement.request_id));
                self.dispatch_placements(placements)?;
            }
            if !removed.is_empty() {
                let origin = common_origin(removed.iter().filter_map(|worker_id| {
                    self.evidence.drain_origin(WorkerPool::Agg, *worker_id)
                }));
                let transitions = removed
                    .iter()
                    .map(|worker_id| WorkerLifecycleTransition {
                        worker_id: *worker_id,
                        transition: WorkerLifecycleTransitionKind::WorkerRemoved,
                        prior_state: Some("draining"),
                        state: "removed",
                        reason: None,
                        origin_operation_ordinal: self
                            .evidence
                            .drain_origin(WorkerPool::Agg, *worker_id),
                    })
                    .collect();
                let state = self.lifecycle_state();
                self.evidence.record_lifecycle_operation(
                    self.now_ms,
                    WorkerPool::Agg,
                    "drain_settlement",
                    None,
                    origin,
                    transitions,
                    state,
                    released,
                );
            }
            changed |= !removed.is_empty();
            // Telemetry observes settled pre-decision state; scaling then fires
            // last and retains its existing controller semantics.
            if self.telemetry.is_some() {
                changed |= self.apply_telemetry_ticks()?;
            }
            if self.scaling_policy.is_some() {
                changed |= self.apply_scaling_ticks()?;
            }

            if !changed {
                break;
            }
        }

        Ok(())
    }

    fn publish_telemetry_sample(&mut self, kind: ReplayTelemetrySampleKind) -> anyhow::Result<()> {
        let Some(telemetry) = self.telemetry.as_ref() else {
            return Ok(());
        };
        let sample_ordinal = telemetry.next_sample_ordinal();
        let interval_start_ms = telemetry.interval_start_ms();
        let (decode_scheduler_metrics, decode_interval_metrics, traffic) = match kind {
            ReplayTelemetrySampleKind::Baseline => (
                self.engine.telemetry_gauges_snapshot(),
                ReplaySchedulerIntervalMetrics::default(),
                ReplayTrafficMetricsSnapshot::default(),
            ),
            ReplayTelemetrySampleKind::Periodic | ReplayTelemetrySampleKind::Final => {
                let (gauges, interval) = self.engine.take_telemetry_snapshot()?;
                (gauges, interval, self.traffic.drain_telemetry(self.now_ms))
            }
        };
        let snapshot = ReplayTelemetrySnapshot {
            sample_ordinal,
            kind,
            interval_start_ms,
            sampled_at_ms: self.now_ms,
            traffic,
            prefill_scheduler_metrics: Vec::new(),
            decode_scheduler_metrics,
            prefill_interval_metrics: ReplaySchedulerIntervalMetrics::default(),
            decode_interval_metrics,
            router_pending_prefill_requests: 0,
            router_pending_decode_requests: self.placement.pending_count(),
            active_prefill_ids: Vec::new(),
            active_decode_ids: self.engine.active_group_ids(),
            starting_prefill_ids: Vec::new(),
            starting_decode_ids: self.engine.starting_group_ids(),
            draining_prefill_ids: Vec::new(),
            draining_decode_ids: self.engine.draining_group_ids(),
        };

        let mut telemetry = self
            .telemetry
            .take()
            .expect("telemetry must remain attached while publishing");
        let result = telemetry.publish(snapshot);
        if result.is_ok() && kind != ReplayTelemetrySampleKind::Baseline {
            telemetry.close_interval(self.now_ms);
        }
        self.telemetry = Some(telemetry);
        result
    }

    /// Start observational/planner clocks only once the saved suffix can run.
    fn start_profile_observers(&mut self) -> anyhow::Result<()> {
        if !self.profile_observers_started {
            self.seed_first_telemetry_tick()?;
            self.seed_first_scaling_tick()?;
            self.profile_observers_started = true;
        }
        Ok(())
    }

    /// The client cancellation boundary is separate from a committed engine
    /// batch's completion. Native cancellation suppresses future delivery but
    /// does not rewind the already executed batch or claim server quiescence.
    fn cancel_expired_agentic_profile(&mut self) -> anyhow::Result<bool> {
        let Some((_, grace, _)) = self.admission.agentic_profile_deadlines() else {
            return Ok(false);
        };
        if self.profile_cancel_started || self.now_ms < grace {
            return Ok(false);
        }
        self.profile_cancel_started = true;
        let requests = self.admission.agentic_profile_pending_request_ids();
        for uuid in requests {
            let state = self
                .requests
                .get(&uuid)
                .context("profile cancellation lost request state")?;
            let scheduler_id = state.scheduler_id();
            let busy = if let Some(scheduler_id) = scheduler_id {
                self.engine.request_has_committed_pass(scheduler_id, uuid)?
            } else {
                false
            };
            if let Some(scheduler_id) = scheduler_id {
                let effects = self.engine.apply_command(
                    scheduler_id,
                    Command::CancelRequest {
                        request_id: uuid,
                        discard_pending_output: true,
                    },
                    self.now_ms,
                )?;
                self.apply_engine_observations(
                    effects.engine_events,
                    KvIngestBoundary::SchedulerCommand,
                )?;
            } else if !self.placement.cancel_pending(uuid) {
                bail!("profile queued request {uuid} was absent from its router");
            }
            self.collector
                .on_terminal(uuid, self.now_ms, ReplayTerminalStatus::Canceled);
            self.admission.defer_causal_terminal(
                uuid,
                self.now_ms,
                ReplayTerminalStatus::Canceled,
            )?;
            if busy {
                self.profile_unsettled_requests += 1;
            } else {
                self.admission.defer_quiescent(uuid, self.now_ms)?;
            }
            self.requests.remove(&uuid);
            self.profile_canceled_requests += 1;
            self.progress.inc_completed();
            let placements = self.placement.request_terminal(uuid, self.now_ms)?;
            self.dispatch_placements(placements)?;
        }
        Ok(true)
    }

    fn finish_agentic_preparation(&mut self) -> anyhow::Result<bool> {
        if !self.admission.is_agentic_preparing() || !self.engine.is_drained() {
            return Ok(false);
        }
        let Some(transition) =
            self.admission
                .finish_agentic_preparation(self.now_ms, &self.collector, || {
                    self.engine.reset_timing_evidence()
                })?
        else {
            return Ok(false);
        };

        // The preparation ledger owns these measurements. Keep the same live
        // engines, router and cache while beginning an empty profile epoch.
        // Discard preparation evidence with its measurements; profile
        // pressure/lifecycle ordinals and KV digests start here.
        let next_evidence = ReplayEvidenceCollector::new(self.evidence.options());
        self.collector
            .set_runtime_evidence(std::mem::replace(&mut self.evidence, next_evidence).finish());
        self.collector.take_report(self.now_ms);
        self.collector
            .set_g3_profile_baseline(self.engine.g3_stats());
        self.collector.set_agentic_phases(
            self.admission
                .agentic_phase_evidence()
                .expect("a preparation transition retains its audit evidence"),
        );
        self.traffic.drain_planner(self.now_ms);
        self.traffic.drain_telemetry(self.now_ms);
        self.engine.take_telemetry_snapshot()?;
        self.fpm_buffer = LatestFpmBuffer::default();
        if self.collect_fpm {
            for worker_id in self.engine.active_group_ids() {
                self.fpm_buffer
                    .activate_worker(worker_id, self.dp_size, self.now_ms);
            }
        }

        if transition == AgenticPreparationTransition::OpenProfile {
            if let Some(cap_ms) = &mut self.max_sim_time_ms {
                *cap_ms += self.now_ms;
                anyhow::ensure!(
                    cap_ms.is_finite(),
                    "profile time limit overflows runtime clock"
                );
            }
            self.start_profile_observers()?;
        }
        Ok(true)
    }

    /// Emit a gauge-only baseline and schedule the first periodic sample.
    fn seed_first_telemetry_tick(&mut self) -> anyhow::Result<()> {
        let Some(telemetry) = self.telemetry.as_mut() else {
            return Ok(());
        };
        telemetry.start_at(self.now_ms);
        self.publish_telemetry_sample(ReplayTelemetrySampleKind::Baseline)?;
        let at_ms = self
            .telemetry
            .as_ref()
            .expect("telemetry must remain attached")
            .next_periodic_at_ms()?;
        push_telemetry_tick(&mut self.events, &mut self.next_event_seq, at_ms);
        Ok(())
    }

    fn apply_telemetry_ticks(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        while pop_ready_telemetry_tick(&mut self.events, self.now_ms) {
            self.publish_telemetry_sample(ReplayTelemetrySampleKind::Periodic)?;
            changed = true;
            if !self.is_workload_done() {
                let next_ms = self
                    .telemetry
                    .as_ref()
                    .expect("telemetry must remain attached")
                    .next_periodic_at_ms()?;
                push_telemetry_tick(&mut self.events, &mut self.next_event_seq, next_ms);
            }
        }
        Ok(changed)
    }

    fn publish_final_telemetry_sample(&mut self) -> anyhow::Result<()> {
        if !self.profile_observers_started {
            return Ok(());
        }
        let Some(telemetry) = self.telemetry.as_ref() else {
            return Ok(());
        };
        if self.now_ms > telemetry.interval_start_ms()
            || self.traffic.telemetry_has_observations()
            || self.engine.telemetry_has_interval_observations()
        {
            self.publish_telemetry_sample(ReplayTelemetrySampleKind::Final)?;
        }
        Ok(())
    }

    /// Seed the first `ScalingTick` from the policy's requested start time (a
    /// non-finite time means "no tick" and is skipped).
    fn seed_first_scaling_tick(&mut self) -> anyhow::Result<()> {
        let Some(mut policy) = self.scaling_policy.take() else {
            return Ok(());
        };
        let first_ms = policy.initial_tick_ms();
        self.scaling_policy = Some(policy);
        let first_ms = first_ms?;
        if first_ms.is_finite() {
            let at_ms = first_ms.max(self.now_ms);
            push_scaling_tick(&mut self.events, &mut self.next_event_seq, at_ms);
        } else {
            // No tick will ever fire to drain the FPM buffer; stop collecting it.
            self.collect_fpm = false;
        }
        Ok(())
    }

    /// Fire every `ScalingTick`: gather a settled snapshot, call the policy,
    /// apply its decision, and re-arm.
    /// Agg routes all FPM through `decode_fpm` and ignores the prefill target.
    fn apply_scaling_ticks(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        while pop_ready_scaling_tick(&mut self.events, self.now_ms) {
            if self.is_workload_done() {
                continue;
            }
            let active_decode_ids = self.engine.active_group_ids();
            let starting_decode_ids = self.engine.starting_group_ids();
            let draining_decode_ids = self.engine.draining_group_ids();
            self.fpm_buffer
                .emit_idle_due(&active_decode_ids, self.dp_size, self.now_ms);
            let tick_ordinal = self.next_scaling_tick_ordinal;
            let snapshot = ReplayScalingSnapshot {
                tick_ordinal,
                now_ms: self.now_ms,
                prefill_fpm: Vec::new(),
                decode_fpm: self.fpm_buffer.take(),
                traffic: self.traffic.drain_planner(self.now_ms),
                active_prefill_ids: Vec::new(),
                active_decode_ids,
                starting_prefill_ids: Vec::new(),
                starting_decode_ids,
                draining_prefill_ids: Vec::new(),
                draining_decode_ids,
            };
            self.next_scaling_tick_ordinal = self
                .next_scaling_tick_ordinal
                .checked_add(1)
                .expect("replay scaling tick ordinal overflow");
            let Some(mut policy) = self.scaling_policy.take() else {
                bail!("scaling tick fired without a policy");
            };
            let decision = policy.on_tick(snapshot);
            self.scaling_policy = Some(policy);
            let decision = decision?;

            if let Some(target) = decision.target_decode {
                self.apply_scaling_with_tick(target, Some(tick_ordinal))?;
            }

            // Re-arm only into the strict, finite future and only while work
            // remains; otherwise no later tick will drain the FPM buffer, so stop
            // collecting it (prevents unbounded growth once the cadence stops).
            let next_tick = decision
                .next_tick_ms
                .filter(|next_ms| next_ms.is_finite() && *next_ms > self.now_ms);
            if let Some(next_ms) = next_tick
                && !self.is_workload_done()
            {
                push_scaling_tick(&mut self.events, &mut self.next_event_seq, next_ms);
            } else {
                self.collect_fpm = false;
            }
            changed = true;
        }
        Ok(changed)
    }

    // ------------------------------------------------------------------
    // Scaling integration used by the in-loop `ScalingTick` handler.
    // ------------------------------------------------------------------

    /// Advance the sim clock to `new_now_ms`, integrating provisioned
    /// worker-seconds over the interval just elapsed. `worker_count()` counts
    /// active + starting-up + draining workers, so this captures the startup
    /// ramp and the scale-down drain tail. Aggregated replay has no separate
    /// prefill pool, so it reports through the decode role (prefill = 0).
    pub(crate) fn advance_now_ms(&mut self, new_now_ms: f64) {
        let dt_ms = (new_now_ms - self.now_ms).max(0.0);
        if dt_ms > 0.0 {
            let decode_worker_seconds = self.engine.worker_count() as f64 * dt_ms / 1000.0;
            self.collector
                .add_worker_seconds(0.0, decode_worker_seconds);
        }
        self.now_ms = new_now_ms;
    }

    /// Advance to an observational heartbeat without waking semantic replay
    /// work. Entering `drain_current_timestamp` here would retry deferred
    /// workers and make native scheduler progress depend on sample cadence.
    fn sample_telemetry_only_timestamp(&mut self, at_ms: f64) -> anyhow::Result<()> {
        self.advance_now_ms(at_ms);
        let sampled = self.apply_telemetry_ticks()?;
        if !sampled {
            bail!("telemetry-only timestamp did not publish its scheduled sample");
        }
        Ok(())
    }

    fn apply_scaling_with_tick(
        &mut self,
        target_workers: usize,
        planner_tick_ordinal: Option<u64>,
    ) -> anyhow::Result<()> {
        if target_workers != self.engine.non_draining_group_count() {
            self.collector.clear_static_worker_count();
        }
        let starting_before = self.engine.starting_group_ids();
        let (added, newly_marked, removed) = self
            .engine
            .apply_target_count(target_workers)
            .with_context(|| {
                format!("failed to apply aggregated worker target {target_workers}")
            })?;
        let startup_delay_ms = self.engine.startup_time_ms();
        let mut released = Vec::new();

        for &id in &added {
            match startup_delay_ms {
                Some(delay) => {
                    push_worker_ready(
                        &mut self.events,
                        &mut self.next_event_seq,
                        self.now_ms + delay,
                        SimulationWorkerStage::Aggregated,
                        id,
                    );
                }
                None => {
                    if self.collect_fpm {
                        self.fpm_buffer
                            .activate_worker(id, self.dp_size, self.now_ms);
                    }
                    let topology = self
                        .engine
                        .worker_topology(id)
                        .ok_or_else(|| anyhow::anyhow!("new worker {id} has no engine topology"))?;
                    let placements = self.placement.worker_ready(topology, self.now_ms)?;
                    released.extend(placements.iter().map(|placement| placement.request_id));
                    self.dispatch_placements(placements)?;
                }
            }
        }

        for &id in &newly_marked {
            let topology = self.engine.worker_topology(id).unwrap_or(WorkerTopology {
                worker_id: id,
                scheduler_ids: Vec::new(),
            });
            let placements = self.placement.worker_draining(topology, self.now_ms)?;
            released.extend(placements.iter().map(|placement| placement.request_id));
            self.dispatch_placements(placements)?;
        }
        for &id in &removed {
            let placements = self.placement.worker_removed(
                WorkerTopology {
                    worker_id: id,
                    scheduler_ids: Vec::new(),
                },
                self.now_ms,
            )?;
            released.extend(placements.iter().map(|placement| placement.request_id));
            self.dispatch_placements(placements)?;
        }
        let placements = self.placement.topology_settled(self.now_ms)?;
        released.extend(placements.iter().map(|placement| placement.request_id));
        self.dispatch_placements(placements)?;
        self.record_scale_lifecycle(
            &added,
            &newly_marked,
            &removed,
            &starting_before,
            startup_delay_ms.is_some(),
            planner_tick_ordinal,
            released,
        );
        Ok(())
    }

    fn lifecycle_state(&self) -> WorkerPoolState {
        WorkerPoolState {
            active: self.engine.active_group_ids(),
            starting: self.engine.starting_group_ids(),
            draining: self.engine.draining_group_ids(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_scale_lifecycle(
        &mut self,
        added: &[usize],
        newly_draining: &[usize],
        removed: &[usize],
        starting_before: &[usize],
        delayed_startup: bool,
        planner_tick_ordinal: Option<u64>,
        released: Vec<Uuid>,
    ) {
        if !self.evidence.options().capture_lifecycle_evidence {
            return;
        }
        let mut transitions = added
            .iter()
            .map(|worker_id| WorkerLifecycleTransition {
                worker_id: *worker_id,
                transition: if delayed_startup {
                    WorkerLifecycleTransitionKind::WorkerStarting
                } else {
                    WorkerLifecycleTransitionKind::WorkerReady
                },
                prior_state: None,
                state: if delayed_startup {
                    "starting"
                } else {
                    "active"
                },
                reason: None,
                origin_operation_ordinal: None,
            })
            .collect::<Vec<_>>();
        transitions.extend(
            newly_draining
                .iter()
                .map(|worker_id| WorkerLifecycleTransition {
                    worker_id: *worker_id,
                    transition: WorkerLifecycleTransitionKind::WorkerDraining,
                    prior_state: Some("active"),
                    state: "draining",
                    reason: None,
                    origin_operation_ordinal: None,
                }),
        );
        transitions.extend(removed.iter().map(|worker_id| {
            let cancelled = starting_before.binary_search(worker_id).is_ok();
            WorkerLifecycleTransition {
                worker_id: *worker_id,
                transition: WorkerLifecycleTransitionKind::WorkerRemoved,
                prior_state: Some(if cancelled { "starting" } else { "draining" }),
                state: "removed",
                reason: cancelled.then_some("startup_cancelled"),
                origin_operation_ordinal: if cancelled {
                    self.evidence.startup_origin(WorkerPool::Agg, *worker_id)
                } else {
                    self.evidence.drain_origin(WorkerPool::Agg, *worker_id)
                },
            }
        }));
        let origin = common_origin(
            removed
                .iter()
                .filter(|worker_id| starting_before.binary_search(worker_id).is_ok())
                .filter_map(|worker_id| self.evidence.startup_origin(WorkerPool::Agg, *worker_id)),
        );
        let state = self.lifecycle_state();
        self.evidence.record_lifecycle_operation(
            self.now_ms,
            WorkerPool::Agg,
            if planner_tick_ordinal.is_some() {
                "planner_scale"
            } else {
                "manual_scale"
            },
            planner_tick_ordinal,
            origin,
            transitions,
            state,
            released,
        );
    }

    fn ensure_drive_started(&mut self) -> anyhow::Result<bool> {
        if self.drive_started {
            return Ok(false);
        }
        if self.max_sim_time_ms.is_some() && self.admission.agentic_profile_report().is_some() {
            bail!("agentic_profile cannot be combined with max_sim_time_ms");
        }
        if let Some(cap_ms) = self.max_sim_time_ms
            && (!cap_ms.is_finite() || cap_ms < 0.0)
        {
            bail!("max_sim_time_ms must be a finite, non-negative value; got {cap_ms}");
        }
        self.drain_current_timestamp()?;
        if self.admission.agentic_phase_evidence().is_none() {
            self.start_profile_observers()?;
        }
        // Keep the baseline before the first scaling decision, but settle any
        // tick seeded at this instant before exposing a settled step boundary.
        if !self.is_done()
            && self
                .events
                .peek()
                .is_some_and(|event| event.at_ms <= self.now_ms)
        {
            self.drain_current_timestamp()?;
        }
        self.drive_started = true;
        Ok(true)
    }

    /// Advance to the next fully settled semantic timestamp without rebuilding
    /// engine, router, placement, or KV state.
    pub(crate) fn step(&mut self) -> anyhow::Result<ReplayStepOutcome> {
        let just_started = self.ensure_drive_started()?;
        if self.is_done() {
            return Ok(ReplayStepOutcome::Complete);
        }
        if just_started {
            return Ok(ReplayStepOutcome::Settled {
                now_ms: self.now_ms,
            });
        }

        loop {
            let (next_timestamp_ms, canonical_timestamp_ms) = self.next_timestamps();
            let Some(canonical_timestamp_ms) = canonical_timestamp_ms else {
                // Aggregated workers have no external handoff dependency. If
                // the event queue is empty while an engine still owns a
                // request, the preceding zero-duration pass lost its terminal
                // effect (or can otherwise never wake) and is a liveness
                // invariant failure, not ordinary quiescence.
                if self.engine.has_runnable_worker() || self.engine.in_flight() > 0 {
                    bail!(
                        "offline replay detected an effect-free zero-duration pass with {} in-flight requests remaining",
                        self.cluster_in_flight()
                    );
                }
                bail!(
                    "offline replay reached a dead end with {} in-flight requests remaining",
                    self.cluster_in_flight()
                );
            };
            if let Some(cap_ms) = self.max_sim_time_ms
                && !self.admission.is_agentic_preparing()
                && canonical_timestamp_ms > cap_ms
            {
                return Ok(ReplayStepOutcome::TimeLimitReached {
                    now_ms: self.now_ms,
                });
            }
            let next_timestamp_ms = next_timestamp_ms
                .expect("canonical replay activity must have a next scheduled timestamp");
            if next_timestamp_ms < canonical_timestamp_ms {
                self.sample_telemetry_only_timestamp(next_timestamp_ms)?;
                continue;
            }
            self.advance_now_ms(next_timestamp_ms);
            self.drain_current_timestamp()?;
            return Ok(if self.is_done() {
                ReplayStepOutcome::Complete
            } else {
                ReplayStepOutcome::Settled {
                    now_ms: self.now_ms,
                }
            });
        }
    }

    fn run_to_completion(&mut self) -> anyhow::Result<()> {
        while let ReplayStepOutcome::Settled { .. } = self.step()? {}
        if !self.drive_finalized {
            self.publish_final_telemetry_sample()?;
            self.drive_finalized = true;
        }
        Ok(())
    }

    /// Run the aggregated offline replay until all arrivals and worker work are exhausted.
    /// If `max_sim_time_ms` is set, exits gracefully when the next scheduled
    /// timestamp would exceed that cap; in-flight requests at that point are
    /// reported as incomplete.
    pub(crate) fn run(mut self) -> anyhow::Result<(TraceCollector, AggRuntimeStats)> {
        if self.admission.is_agentic_preparing() {
            self.collector.begin_batch_preparation_reporting();
        } else {
            self.collector.begin_batch_reporting();
        }
        self.run_to_completion()?;

        self.progress.finish();
        if let Some(snapshot) = self.admission.agentic_trajectory_snapshot() {
            self.collector.set_agentic_trajectory(snapshot);
        }
        if let Some(identity) = self.admission.agentic_graph_identity() {
            self.collector.set_agentic_graph(identity);
        }
        self.collector.g3_offload = self.engine.g3_stats();
        if let Some(snapshots) = self.admission.agentic_snapshot_evidence() {
            self.collector.set_agentic_snapshots(snapshots);
        }
        if let Some(phases) = self.admission.agentic_phase_evidence() {
            self.collector.set_agentic_phases(phases);
        }
        if let Some(mut profile) = self.admission.agentic_profile_report() {
            profile.finished_at_ms = Some(self.now_ms);
            profile.canceled_requests = self.profile_canceled_requests;
            profile.unsettled_server_requests =
                self.profile_unsettled_requests + self.requests.len();
            self.collector.set_agentic_profile(profile);
        }
        if let Some(transcript) = self.admission.agentic_lifecycle_transcript() {
            self.collector.set_agentic_lifecycle(transcript);
        }
        if let Some(outcomes) = self.admission.agentic_play_outcomes() {
            self.collector.set_agentic_play_outcomes(outcomes);
        }
        self.collector.set_runtime_evidence(self.evidence.finish());
        self.collector.prepare_batch_report()?;
        Ok((self.collector, self.stats))
    }
}

#[cfg(test)]
#[path = "agg_tests.rs"]
mod tests;

#[cfg(test)]
mod agentic_warmup_tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::engine::{Backend, EngineConfig, NativeHostOffloadConfig, TimingModelConfig};
    use crate::replay::WorkerStage;
    use crate::replay::components::{NoReplayMetadata, ReplayMode};
    use crate::replay::core::NoEngineEvents;
    use crate::replay::core::round_robin::AggregatedRoundRobinPlacement;
    use crate::replay::engine::{ReplayEngineConfig, ReplayEngineFactory};
    use crate::replay::loadgen::{
        AGENTIC_MOONCAKE_SCHEMA, AGENTIC_MOONCAKE_VERSION, AgenticDependency,
        AgenticDependencyRelation, AgenticDependencyTrigger, AgenticHashIdScope,
        AgenticMooncakeHeader, AgenticMooncakeRow, AgenticReplayPhase, AgenticSnapshotOptions,
        AgenticSourceProvenance, PreparedAgenticSnapshots, ValidatedAgenticGraph, WorkloadDriver,
    };
    use crate::replay::scaling::ReplayScalingDecision;

    type Runtime =
        AggRuntimeImpl<AggregatedRoundRobinPlacement<()>, NoEngineEvents, NoReplayMetadata>;

    fn runtime(backend: Backend, blocks: usize, prefill_ms: f64) -> Runtime {
        runtime_with_offload(backend, blocks, prefill_ms, false)
    }

    fn runtime_with_offload(
        backend: Backend,
        blocks: usize,
        prefill_ms: f64,
        offload: bool,
    ) -> Runtime {
        runtime_with_factory(
            backend,
            blocks,
            prefill_ms,
            offload,
            ReplayEngineFactory::new(),
        )
    }

    fn runtime_with_factory(
        backend: Backend,
        blocks: usize,
        prefill_ms: f64,
        offload: bool,
        factory: ReplayEngineFactory,
    ) -> Runtime {
        let row = |id: &str, start, input_length, hashes| AgenticMooncakeRow {
            request_id: id.into(),
            play_id: "play".into(),
            session_id: "conversation".into(),
            model: "model".into(),
            input_length: Some(input_length),
            output_length: Some(1),
            hash_ids: Some(hashes),
            not_before_ms: start,
            recorded_api_time_ms: Some(5.0),
            ..Default::default()
        };
        let history = row("history", 0.0, 128, vec![10, 20]);
        let mut future = row("future", 100.0, 192, vec![10, 20, 30]);
        future.dependencies = vec![AgenticDependency {
            request_id: "history".into(),
            trigger: AgenticDependencyTrigger::Completion,
            relation: AgenticDependencyRelation::Sequence,
            delay_ms: 0.0,
        }];
        let graph = ValidatedAgenticGraph::from_agentic_mooncake_rows(
            AgenticMooncakeHeader {
                schema: AGENTIC_MOONCAKE_SCHEMA.into(),
                version: AGENTIC_MOONCAKE_VERSION,
                block_size: 64,
                hash_id_scope: AgenticHashIdScope::Local,
                source: AgenticSourceProvenance {
                    format: "self-authored".into(),
                    digest: "warmup-runtime-v1".into(),
                },
            },
            vec![history, future],
        )
        .unwrap();
        let prepared = graph
            .prepare_snapshots(1, AgenticSnapshotOptions { seed: 42 })
            .unwrap();
        let play = prepared.context().prepare_play(0, 0, Some(50.0)).unwrap();
        let driver = WorkloadDriver::new_agentic_warmup(
            PreparedAgenticSnapshots::from_plays(vec![play]).unwrap(),
            64,
            true,
            1.0,
        )
        .unwrap();
        let config = ReplayEngineConfig {
            rank: EngineConfig {
                block_size: 64,
                num_gpu_blocks: blocks,
                max_model_len: (blocks == 1 && backend == Backend::Vllm).then_some(64),
                kv_cache_bytes_per_token: offload.then_some(1),
                native_host_offload: offload
                    .then(|| NativeHostOffloadConfig::new(64).with_bandwidths(0.000001, 0.000001)),
                timing_model: TimingModelConfig::Fixed {
                    prefill_ms,
                    decode_ms: 0.0,
                },
                ..EngineConfig::for_backend(backend)
            },
            ..Default::default()
        };
        let factory = factory
            .role_factory(&config, WorkerStage::Aggregated, false)
            .unwrap();
        Runtime::new_composed(
            factory,
            AdmissionQueue::new_workload(driver, ReplayMode::Trace),
            1,
            None,
            |dp, topology| Ok(AggregatedRoundRobinPlacement::new(dp, topology)),
        )
        .unwrap()
        .with_per_request_records(true)
    }

    #[test]
    fn profile_recycles_lanes_and_stops_new_arrivals_at_cutoff() {
        for backend in [Backend::Vllm, Backend::Sglang] {
            let mut replay = runtime(backend, 1024, 3.0);
            replay
                .admission
                .enable_agentic_profile(crate::replay::loadgen::AgenticProfileOptions {
                    duration_seconds: 0.25,
                    response_grace_seconds: 0.03,
                    ..Default::default()
                })
                .unwrap();
            let report = replay.run().unwrap().0.finish();
            let profile = report.agentic_profile.as_ref().unwrap();
            let origin = profile.profile_start_ms.unwrap();
            let cutoff = profile.admission_cutoff_ms.unwrap();
            assert!(profile.plays_started >= 3, "{backend:?}: {profile:?}");
            assert!(profile.client_completed_plays >= 2);
            assert_eq!(cutoff - origin, 250.0);
            assert!(
                report
                    .per_request
                    .iter()
                    .all(|request| request.arrival_time_ms + origin < cutoff)
            );
            assert_eq!(profile.client_in_flight_requests, 0);
            assert!(profile.admission_closed);
        }
    }

    #[rstest::rstest]
    fn profile_grace_includes_return_and_zero_grace_cancels_busy_pass(
        #[values(0.0, 10.0)] cancel_drain_seconds: f64,
    ) {
        for backend in [Backend::Vllm, Backend::Sglang] {
            for grace in [0.0, 0.03] {
                let mut replay = runtime(backend, 1024, 20.0);
                replay
                    .admission
                    .enable_agentic_profile(crate::replay::loadgen::AgenticProfileOptions {
                        duration_seconds: 0.051,
                        response_grace_seconds: grace,
                        cancel_drain_seconds,
                        ..Default::default()
                    })
                    .unwrap();
                let report = replay.run().unwrap().0.finish();
                let profile = report.agentic_profile.as_ref().unwrap();
                assert_eq!(report.per_request.len(), 1);
                assert!(!profile.cancel_drain_timed_out);
                assert_eq!(profile.client_in_flight_requests, 0);
                assert_eq!(
                    profile.cancel_drain_deadline_ms,
                    profile
                        .response_grace_deadline_ms
                        .map(|at| at + cancel_drain_seconds * 1000.0)
                );
                if grace == 0.0 {
                    assert_eq!(
                        report.per_request[0].terminal_status,
                        ReplayTerminalStatus::Canceled
                    );
                    assert_eq!(profile.canceled_requests, 1);
                    assert_eq!(profile.unsettled_server_requests, 1);
                    assert_eq!(profile.finished_at_ms, profile.admission_cutoff_ms);
                } else {
                    assert_eq!(
                        report.per_request[0].terminal_status,
                        ReplayTerminalStatus::Completed
                    );
                    assert!(profile.finished_at_ms.unwrap() > profile.admission_cutoff_ms.unwrap());
                    assert_eq!(profile.observation_duration_ms, Some(20.0));
                    assert_eq!(profile.canceled_requests, 0);
                }
            }
        }
    }

    #[test]
    fn profile_summary_and_detailed_metrics_match_including_grace() {
        for backend in [Backend::Vllm, Backend::Sglang] {
            let run = |capture| {
                let mut replay = runtime(backend, 1024, 20.0).with_per_request_records(capture);
                replay
                    .admission
                    .enable_agentic_profile(crate::replay::loadgen::AgenticProfileOptions {
                        duration_seconds: 0.051,
                        response_grace_seconds: 0.03,
                        ..Default::default()
                    })
                    .unwrap();
                replay.run().unwrap().0.finish()
            };
            assert_eq!(
                serde_json::to_value(run(false)).unwrap(),
                serde_json::to_value(run(true)).unwrap()
            );
        }
    }

    #[derive(Default)]
    struct EpochTiming {
        evidence: Mutex<crate::engine::TimingEvidenceSummary>,
        resets: std::sync::atomic::AtomicUsize,
    }

    impl crate::engine::TimingModel for EpochTiming {
        fn predict_prefill_ms(
            &self,
            _batch: usize,
            input: usize,
            _prefix: usize,
        ) -> anyhow::Result<f64> {
            use crate::engine::{
                TimingEvidenceSource, TimingOperationEvidence, TimingPhaseEvidence,
            };
            self.evidence.lock().unwrap().prefill.try_accumulate(
                TimingPhaseEvidence::try_from_operations(vec![TimingOperationEvidence::new(
                    format!("input-{input}"),
                    10.0,
                    Some(input as f64 * 10.0),
                    TimingEvidenceSource::Silicon,
                )?])?,
            )?;
            Ok(10.0)
        }

        fn predict_decode_ms(
            &self,
            _batch: usize,
            _active: usize,
            _context: usize,
            _total: usize,
        ) -> anyhow::Result<f64> {
            Ok(1.0)
        }

        fn evidence_summary(&self) -> Option<crate::engine::TimingEvidenceSummary> {
            Some(self.evidence.lock().unwrap().clone())
        }

        fn reset_evidence(&self) -> anyhow::Result<()> {
            *self.evidence.lock().unwrap() = Default::default();
            self.resets
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
    }

    struct UnsupportedEvidenceReset;

    impl crate::engine::TimingModel for UnsupportedEvidenceReset {
        fn predict_prefill_ms(
            &self,
            _batch: usize,
            _input: usize,
            _prefix: usize,
        ) -> anyhow::Result<f64> {
            Ok(10.0)
        }

        fn predict_decode_ms(
            &self,
            _batch: usize,
            _active: usize,
            _context: usize,
            _total: usize,
        ) -> anyhow::Result<f64> {
            Ok(1.0)
        }

        fn evidence_summary(&self) -> Option<crate::engine::TimingEvidenceSummary> {
            Some(Default::default())
        }
    }

    #[test]
    fn warmup_power_epoch_contains_only_profile_predictions() {
        use crate::engine::TimingModel;
        for backend in [Backend::Vllm, Backend::Sglang] {
            let timing = Arc::new(EpochTiming::default());
            let (collector, _) = runtime_with_factory(
                backend,
                64,
                10.0,
                false,
                ReplayEngineFactory::with_timing_model(timing.clone()),
            )
            .run()
            .unwrap();
            let report = collector.finish();
            assert_eq!(report.request_counts.completed_requests, 1);
            assert_eq!(report.agentic_phases.unwrap().requests.len(), 11);
            assert_eq!(timing.resets.load(std::sync::atomic::Ordering::Relaxed), 1);
            let summary = timing.evidence_summary().unwrap();
            assert_eq!(summary.prefill.latency_ms, 10.0);
            assert_eq!(summary.prefill.energy_wms, Some(1920.0));
            assert_eq!(summary.prefill.operations.len(), 1);
            assert_eq!(summary.prefill.operations[0].name, "input-192");
        }
    }

    #[test]
    fn failed_power_reset_does_not_open_or_shift_profile() {
        let mut replay = runtime_with_factory(
            Backend::Vllm,
            64,
            10.0,
            false,
            ReplayEngineFactory::with_timing_model(Arc::new(UnsupportedEvidenceReset)),
        )
        .with_max_sim_time_ms(Some(80.0));
        let error = loop {
            match replay.step() {
                Ok(ReplayStepOutcome::Settled { .. }) => {}
                Err(error) => break error,
                other => panic!("preparation unexpectedly completed: {other:?}"),
            }
        };
        assert!(
            error
                .to_string()
                .contains("does not support resetting its measurement epoch")
        );
        assert!(replay.admission.is_agentic_preparing());
        assert!(replay.engine.is_drained());
        assert!(!replay.profile_observers_started);
        assert_eq!(replay.max_sim_time_ms, Some(80.0));
        let evidence = replay.admission.agentic_phase_evidence().unwrap();
        assert_eq!(evidence.profile_start_ms, None);
        assert!(
            evidence
                .requests
                .iter()
                .all(|request| request.quiescent_at_ms.is_some())
        );
        assert!(replay.collector.contains_request(evidence.requests[0].uuid));
        assert!(replay.finish_agentic_preparation().is_err());
        assert_eq!(replay.admission.agentic_phase_evidence().unwrap(), evidence);
    }

    #[test]
    fn aborted_preparation_clears_accumulated_power_evidence() {
        use crate::engine::TimingModel;
        let timing = Arc::new(EpochTiming::default());
        let mut replay = runtime_with_factory(
            Backend::Vllm,
            64,
            10.0,
            false,
            ReplayEngineFactory::with_timing_model(timing.clone()),
        );
        // Finish the first primer and let the next preparation request enter
        // the native engine before cancellation aborts the preparation phase.
        loop {
            replay.step().unwrap();
            if replay.admission.agentic_phase_evidence().unwrap().requests[0]
                .quiescent_at_ms
                .is_some()
            {
                break;
            }
        }
        assert!(timing.evidence_summary().unwrap().prefill.latency_ms > 0.0);
        let uuid = *replay.requests.keys().next().unwrap();
        assert_eq!(
            replay.cancel_dynamic(uuid).unwrap(),
            Some(ReplayTerminalStatus::Canceled)
        );
        let (collector, _) = replay.run().unwrap();
        let report = collector.finish();
        assert_eq!(
            report.agentic_phases.unwrap().phase,
            AgenticReplayPhase::Aborted
        );
        assert_eq!(report.request_counts.num_requests, 0);
        assert_eq!(timing.evidence_summary(), Some(Default::default()));
        assert_eq!(timing.resets.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    struct Samples(Arc<Mutex<Vec<ReplayTelemetrySnapshot>>>);

    impl ReplayTelemetryObserver for Samples {
        fn on_sample(&mut self, sample: ReplayTelemetrySnapshot) -> anyhow::Result<()> {
            self.0.lock().unwrap().push(sample);
            Ok(())
        }
    }

    struct ScalingTicks(Arc<Mutex<Vec<ReplayScalingSnapshot>>>);

    impl ReplayScalingPolicy for ScalingTicks {
        fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
            Ok(0.0)
        }

        fn on_tick(
            &mut self,
            snapshot: ReplayScalingSnapshot,
        ) -> anyhow::Result<ReplayScalingDecision> {
            self.0.lock().unwrap().push(snapshot);
            Ok(ReplayScalingDecision::default())
        }
    }

    #[test]
    fn batch_warmup_summary_preserves_preparation_admissions_and_profile_epoch() {
        for backend in [Backend::Vllm, Backend::Sglang] {
            let run = |capture| {
                runtime(backend, 64, 10.0)
                    .with_per_request_records(capture)
                    .run()
                    .unwrap()
                    .0
                    .finish()
            };
            let detailed = run(true);
            let summary = run(false);
            assert!(summary.per_request.is_empty());
            assert!(
                summary
                    .agentic_phases
                    .as_ref()
                    .unwrap()
                    .requests
                    .iter()
                    .all(|request| request.first_admit_ms.is_some())
            );
            assert_eq!(
                serde_json::to_value(summary).unwrap(),
                serde_json::to_value(detailed).unwrap(),
                "{backend:?} batch summary must retain the same preparation evidence and profile metrics"
            );
        }
    }

    #[test]
    fn warmup_time_is_excluded_from_caps_metrics_telemetry_and_scaling() {
        for backend in [Backend::Vllm, Backend::Sglang] {
            let samples = Arc::new(Mutex::new(Vec::new()));
            let ticks = Arc::new(Mutex::new(Vec::new()));
            let (collector, _) = runtime(backend, 64, 10.0)
                // Preparation alone takes 110 ms, longer than this profile cap.
                .with_max_sim_time_ms(Some(80.0))
                .with_telemetry_observer(7.0, Box::new(Samples(samples.clone())))
                .with_scaling_policy(Box::new(ScalingTicks(ticks.clone())))
                .run()
                .unwrap();
            let report = collector.finish();
            let phases = report.agentic_phases.as_ref().unwrap();
            let start = phases.profile_start_ms.unwrap();
            assert!(start > 80.0, "{backend:?}: {phases:?}");
            assert_eq!(phases.requests.len(), 11);
            assert_eq!(report.request_counts.num_requests, 1);
            assert_eq!(report.request_counts.completed_requests, 1);
            assert_eq!(report.request_counts.total_output_tokens, 1);
            assert_eq!(report.per_request[0].arrival_time_ms, 50.0);
            assert_eq!(
                report.per_request[0].agentic_phase,
                Some(AgenticReplayPhase::Profile)
            );
            // The fixture keeps 50 ms of snapshot wait, then uses 10 ms prefill
            // and zero decode time on both backends.
            let duration_ms = 60.0;
            assert_eq!(report.throughput.duration_ms, duration_ms);
            assert!((report.throughput.decode_worker_seconds - duration_ms / 1000.0).abs() < 1e-9);

            let samples = samples.lock().unwrap();
            assert_eq!(samples[0].kind, ReplayTelemetrySampleKind::Baseline);
            assert_eq!(samples[0].sampled_at_ms, start);
            assert!(
                samples
                    .iter()
                    .all(|sample| sample.interval_start_ms >= start)
            );
            assert_eq!(
                samples
                    .iter()
                    .map(|sample| sample.traffic.arriving_requests)
                    .sum::<usize>(),
                1
            );
            assert_eq!(
                samples
                    .iter()
                    .map(|sample| sample.traffic.completed_requests)
                    .sum::<usize>(),
                1
            );
            let ticks = ticks.lock().unwrap();
            assert_eq!(ticks.len(), 1);
            assert_eq!(ticks[0].now_ms, start);
            assert_eq!(ticks[0].traffic.num_req, 0);
            assert_eq!(ticks[0].traffic.shape_count, 0);
            assert!(ticks[0].decode_fpm.is_empty());
        }
    }

    #[test]
    fn rejected_preparation_returns_invalid_evidence_without_profile_measurements() {
        {
            let backend = Backend::Vllm;
            let samples = Arc::new(Mutex::new(Vec::new()));
            let (collector, _) = runtime(backend, 1, 10.0)
                .with_telemetry_observer(7.0, Box::new(Samples(samples.clone())))
                .run()
                .unwrap();
            let report = collector.finish();
            let phases = report.agentic_phases.unwrap();
            assert_eq!(phases.phase, AgenticReplayPhase::Aborted);
            assert_eq!(phases.profile_start_ms, None);
            assert_eq!(
                phases
                    .requests
                    .iter()
                    .filter(|request| request.dispatched_at_ms.is_some())
                    .count(),
                1
            );
            assert_eq!(
                phases.requests[0].terminal_status,
                Some(ReplayTerminalStatus::Rejected)
            );
            assert!(phases.requests[0].quiescent_at_ms.is_some());
            assert_eq!(report.request_counts.num_requests, 0);
            assert_eq!(report.request_counts.completed_requests, 0);
            assert!(report.per_request.is_empty());
            assert_eq!(report.throughput.duration_ms, 0.0);
            assert_eq!(report.throughput.decode_worker_seconds, 0.0);
            assert!(samples.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn late_preparation_signals_keep_their_phase_and_do_not_enter_profile_measurements() {
        for backend in [Backend::Vllm, Backend::Sglang] {
            let mut replay = runtime(backend, 64, 10.0);
            while replay.admission.is_agentic_preparing() {
                replay.step().unwrap();
            }
            let phases = replay.admission.agentic_phase_evidence().unwrap();
            let uuid = phases.requests[0].uuid;
            for completed in [false, true] {
                replay
                    .process_output_signal(OutputSignal {
                        uuid,
                        token_id: Some(999),
                        completed,
                        rejected: false,
                        handoff_delay_ms: None,
                        cached_tokens: None,
                    })
                    .unwrap();
            }
            // Only known preparation identities are ignored; arbitrary native
            // callbacks still expose a runtime bookkeeping error.
            assert!(
                replay
                    .process_output_signal(OutputSignal {
                        uuid: Uuid::from_u128(99_999),
                        token_id: None,
                        completed: true,
                        rejected: false,
                        handoff_delay_ms: None,
                        cached_tokens: None,
                    })
                    .is_err()
            );
            assert_eq!(replay.admission.agentic_phase_evidence().unwrap(), phases);
            assert!(!replay.collector.contains_request(uuid));
            let (collector, _) = replay.run().unwrap();
            let report = collector.finish();
            assert_eq!(report.request_counts.num_requests, 1);
            assert_eq!(report.request_counts.completed_requests, 1);
            assert_eq!(report.request_counts.total_output_tokens, 1);
        }
    }

    #[test]
    fn preparation_barrier_starts_new_runtime_evidence_ordinals_and_digests() {
        use crate::engine::{PressureEvent, PressureKind, PressureState};

        let record = |replay: &mut Runtime, uuid, cause| {
            replay.evidence.record_native_pressure(
                &mut replay.collector,
                WorkerPool::Agg,
                0,
                0,
                PressureEvent {
                    at_ms: replay.now_ms,
                    kind: PressureKind::VllmPreemption,
                    request_id: uuid,
                    state_before: PressureState::default(),
                    state_after: PressureState::default(),
                    request_active_blocks_before: 1,
                    logical_available_blocks_before: Some(0),
                    required_blocks_before: Some(1),
                },
            );
            replay.evidence.record_lifecycle_operation(
                replay.now_ms,
                WorkerPool::Agg,
                cause,
                None,
                None,
                vec![WorkerLifecycleTransition {
                    worker_id: 0,
                    transition: WorkerLifecycleTransitionKind::WorkerReady,
                    prior_state: Some("starting"),
                    state: "active",
                    reason: None,
                    origin_operation_ordinal: None,
                }],
                replay.lifecycle_state(),
                Vec::new(),
            );
            replay
                .evidence
                .record_kv_ingest(
                    WorkerPool::Agg,
                    KvIngestBoundary::PassEnd,
                    replay.now_ms,
                    0,
                    |_| Ok(()),
                )
                .unwrap();
        };
        let options = ReplayCaptureOptions {
            capture_per_request: true,
            capture_lifecycle_evidence: true,
            capture_canonical_evidence: true,
            ..Default::default()
        };
        let mut replay = runtime(Backend::Vllm, 64, 10.0).with_capture_options(options);
        replay.step().unwrap();
        let primer = replay.admission.agentic_phase_evidence().unwrap().requests[0].uuid;
        // Feed each optional evidence boundary while native preparation is
        // live. These records must not become evidence of measured traffic.
        record(&mut replay, primer, "preparation_fixture");
        while replay.admission.is_agentic_preparing() {
            replay.step().unwrap();
        }
        assert_eq!(replay.evidence.options(), options);
        while replay.requests.is_empty() {
            replay.step().unwrap();
        }
        let profile_uuid = *replay.requests.keys().next().unwrap();
        let profile_at_ms = replay.now_ms;
        record(&mut replay, profile_uuid, "profile_fixture");
        let (collector, _) = replay.run().unwrap();
        let report = collector.finish();
        assert_eq!(report.per_request[0].pressure_record_ordinals, vec![0]);
        let evidence = report.runtime_evidence;
        let pressure = evidence.pressure.unwrap();
        assert_eq!(pressure.vllm_preemptions_total, 1);
        assert_eq!(pressure.records.len(), 1);
        assert_eq!(pressure.records[0].request_uuid, profile_uuid.to_string());
        assert_eq!(pressure.records[0].pressure_ordinal, 0);
        assert_eq!(pressure.records[0].at_ms, profile_at_ms);
        assert_eq!(evidence.lifecycle_operations.len(), 1);
        assert_eq!(evidence.lifecycle_operations[0].cause, "profile_fixture");
        assert_eq!(evidence.lifecycle_operations[0].operation_ordinal, 0);
        assert_eq!(
            evidence.lifecycle_operations[0].transitions[0].origin_operation_ordinal,
            Some(0)
        );
        let kv = evidence.kv_ingest.unwrap();
        assert_eq!(kv.batches, 1);
        assert_eq!(kv.boundaries["pass_end"].first_at_ms, profile_at_ms);
        assert_eq!(kv.boundaries["pass_end"].last_at_ms, profile_at_ms);
        let mut reference = ReplayEvidenceCollector::new(options);
        reference
            .record_kv_ingest(
                WorkerPool::Agg,
                KvIngestBoundary::PassEnd,
                profile_at_ms,
                0,
                |_| Ok(()),
            )
            .unwrap();
        assert_eq!(kv, reference.finish().kv_ingest.unwrap());
    }

    #[test]
    fn barrier_waits_for_host_offload_work_after_preparation_requests_settle() {
        let (collector, _) = runtime_with_offload(Backend::Vllm, 64, 10.0, true)
            .run()
            .unwrap();
        let report = collector.finish();
        let phases = report.agentic_phases.unwrap();
        let last_request_settlement = phases
            .requests
            .iter()
            .filter_map(|request| request.quiescent_at_ms)
            .max_by(f64::total_cmp)
            .unwrap();
        assert!(phases.profile_start_ms.unwrap() > last_request_settlement);
        assert_eq!(report.request_counts.completed_requests, 1);
        assert_eq!(report.per_request[0].arrival_time_ms, 50.0);
        assert_eq!(
            report.per_request[0].admission_history[0].reused_input_tokens,
            128
        );
    }
}

#[cfg(test)]
mod host_offload_tests {
    use super::*;
    use crate::engine::{EngineConfig, NativeHostOffloadConfig, TimingModelConfig};
    use crate::replay::components::NoReplayMetadata;
    use crate::replay::engine::{ReplayEngineConfig, ReplayEngineFactory};
    use crate::replay::loadgen::ReplayRequestPayload;
    use crate::replay::{NoEngineEvents, PlacementEffects, WorkerStage};
    use std::collections::VecDeque;

    const SEED: Uuid = Uuid::from_u128(1);
    const EVICT: Uuid = Uuid::from_u128(2);
    const RESTORE: Uuid = Uuid::from_u128(3);
    const COMPLETING_A: Uuid = Uuid::from_u128(4);
    const QUEUED_FOR_B: Uuid = Uuid::from_u128(5);

    struct ReleaseToIdleWorker {
        pending: Option<Uuid>,
    }

    impl PlacementPolicy<ReplayRequestPayload> for ReleaseToIdleWorker {
        type Metadata = NoReplayMetadata;
        type Observation = ();

        fn place(
            &mut self,
            request: &ReplayRequestPayload,
            _metadata: Self::Metadata,
            _session_id: Option<String>,
            _now_ms: f64,
        ) -> anyhow::Result<PlacementEffects> {
            let request_id = request
                .metadata()
                .uuid
                .expect("test requests have stable IDs");
            if request_id == QUEUED_FOR_B {
                assert!(self.pending.replace(request_id).is_none());
                return Ok(PlacementEffects {
                    decision: PlacementDecision::Queued,
                    released: Vec::new(),
                });
            }
            let scheduler_id =
                usize::from(request_id == SEED || request_id == EVICT || request_id == RESTORE);
            Ok(PlacementEffects {
                decision: PlacementDecision::Immediate(Placement {
                    request_id,
                    scheduler_id,
                    reported_overlap_tokens: 0,
                    cache_sample: None,
                    placement_replica_id: None,
                }),
                released: Vec::new(),
            })
        }

        fn observe(
            &mut self,
            _observation: Self::Observation,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn cancel_pending(&mut self, request_id: Uuid) -> bool {
            if self.pending == Some(request_id) {
                self.pending = None;
                true
            } else {
                false
            }
        }

        fn request_terminal(
            &mut self,
            request_id: Uuid,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            if request_id != COMPLETING_A {
                return Ok(Vec::new());
            }
            Ok(self
                .pending
                .take()
                .into_iter()
                .map(|request_id| Placement {
                    request_id,
                    scheduler_id: 1,
                    reported_overlap_tokens: 0,
                    cache_sample: None,
                    placement_replica_id: None,
                })
                .collect())
        }

        fn prefill_completed(
            &mut self,
            _request_id: Uuid,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn pending_count(&self) -> usize {
            usize::from(self.pending.is_some())
        }

        fn worker_ready(
            &mut self,
            _worker: WorkerTopology,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn worker_draining(
            &mut self,
            _worker: WorkerTopology,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn worker_removed(
            &mut self,
            _worker: WorkerTopology,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn topology_settled(&mut self, _now_ms: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
    }

    fn request(uuid: Uuid, at_ms: f64, tokens: Vec<u32>) -> DirectRequest {
        DirectRequest {
            tokens,
            max_output_tokens: 0,
            uuid: Some(uuid),
            arrival_timestamp_ms: Some(at_ms),
            ..DirectRequest::default()
        }
    }

    #[test]
    fn idle_internal_deadline_settles_before_completion_releases_placement() {
        let config = ReplayEngineConfig {
            rank: EngineConfig {
                num_gpu_blocks: 1,
                block_size: 4,
                max_num_seqs: 2,
                max_num_batched_tokens: 8,
                enable_prefix_caching: true,
                kv_cache_bytes_per_token: Some(250_000),
                native_host_offload: Some(
                    NativeHostOffloadConfig::new(2).with_bandwidths(1.0, 1.0),
                ),
                timing_model: TimingModelConfig::Fixed {
                    prefill_ms: 1.0,
                    decode_ms: 0.0,
                },
                ..EngineConfig::default()
            },
            ..ReplayEngineConfig::default()
        };
        let factory = ReplayEngineFactory::new()
            .role_factory(&config, WorkerStage::Aggregated, false)
            .unwrap();
        let admission = AdmissionQueue::new_requests(
            VecDeque::from([
                request(SEED, 0.0, vec![1, 2, 3, 4]),
                request(EVICT, 2.0, vec![5, 6, 7, 8]),
                request(RESTORE, 4.0, vec![1, 2, 3, 4]),
                request(COMPLETING_A, 4.0, vec![11, 12, 13, 14]),
                request(QUEUED_FOR_B, 4.0, vec![21, 22, 23, 24]),
            ]),
            ReplayMode::Trace,
        );
        let runtime = AggRuntimeImpl::<ReleaseToIdleWorker, NoEngineEvents, ()>::new_composed(
            factory,
            admission,
            2,
            None,
            move |_dp_size, topology| {
                assert_eq!(
                    topology
                        .iter()
                        .map(|worker| worker.scheduler_ids.as_slice())
                        .collect::<Vec<_>>(),
                    vec![&[0][..], &[1][..]]
                );
                Ok(ReleaseToIdleWorker { pending: None })
            },
        )
        .unwrap()
        .with_per_request_records(true);

        let (collector, _) = runtime.run().unwrap();
        let report = collector.finish();
        assert_eq!(report.request_counts.completed_requests, 5);
        let restore = report
            .per_request
            .iter()
            .find(|request| request.uuid == RESTORE.to_string())
            .expect("restore request record");
        assert_eq!(restore.decode_worker_idx, Some(1));
        assert_eq!(restore.first_admit_ms, Some(5.0));
        assert_eq!(restore.first_admission_host_reused_input_tokens, Some(4));
        assert_eq!(restore.terminal_status, ReplayTerminalStatus::Completed);
        let released = report
            .per_request
            .iter()
            .find(|request| request.uuid == QUEUED_FOR_B.to_string())
            .expect("released request record");
        assert_eq!(released.decode_worker_idx, Some(1));
        assert_eq!(released.first_admit_ms, Some(5.0));
        assert_eq!(released.terminal_status, ReplayTerminalStatus::Completed);
    }
}

impl<PlacementPolicyImpl, Observation, Metadata>
    AggRuntimeImpl<PlacementPolicyImpl, Observation, Metadata>
where
    Observation: ReplayEngineObservation,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl: PlacementPolicy<ReplayRequestPayload, Metadata = Metadata, Observation = Observation::Batch>,
{
    /// Put this runtime under external control: the caller owns the clock and
    /// the loop, and the drain splits into an evaluate half and a commit half
    /// (see [`Self::step_dynamic_until`]).
    pub(crate) fn into_steppable(mut self) -> Self {
        self.defer_drive = true;
        self
    }

    /// Current simulated time in milliseconds.
    pub(crate) fn now_ms(&self) -> f64 {
        self.now_ms
    }

    /// Pick the next canonical logical timestamp (excluding pure-telemetry
    /// ticks), for the external steppable driver.
    pub(crate) fn next_timestamp(&mut self) -> Option<f64> {
        self.next_timestamps().1
    }

    /// Shared read access to the accumulating measurements.
    pub(crate) fn collector(&self) -> &TraceCollector {
        &self.collector
    }

    /// Mutable access for the steppable seam's capture and SLA configuration.
    pub(crate) fn collector_mut(&mut self) -> &mut TraceCollector {
        &mut self.collector
    }

    /// Drain the output tokens observed since the previous call.
    pub(crate) fn take_step_tokens(&mut self) -> Vec<(Uuid, u32)> {
        std::mem::take(&mut self.step_tokens)
    }

    pub(crate) fn take_step_terminals(&mut self) -> Vec<(Uuid, ReplayTerminalStatus)> {
        std::mem::take(&mut self.step_terminals)
    }

    /// Admit `request` at the current simulated time. The returned id
    /// correlates the request with later measurements.
    pub(crate) fn submit_dynamic(&mut self, request: DirectRequest) -> anyhow::Result<Uuid> {
        let arrival_time_ms = self.now_ms;
        let uuid = self.assign_request(
            ReplayRequestPayload::materialized(request),
            arrival_time_ms,
            Metadata::from_hashes(None),
            None,
        )?;
        if self.defer_drive {
            self.drive_pending = true;
        }
        Ok(uuid)
    }

    /// Cancel a dynamically admitted request and return its terminal status
    /// when it was still owned by this replay.
    pub(crate) fn cancel_dynamic(
        &mut self,
        uuid: Uuid,
    ) -> anyhow::Result<Option<ReplayTerminalStatus>> {
        let Some((phase, scheduler_id)) = self
            .requests
            .get(&uuid)
            .map(|state| (state.phase, state.scheduler_id()))
        else {
            return Ok(None);
        };

        match phase {
            super::state::AggRequestPhase::QueuedAtRouter => {
                if !self.placement.cancel_pending(uuid) {
                    bail!("offline replay queued request {uuid} was absent from its router");
                }
            }
            super::state::AggRequestPhase::Running => {
                let scheduler_id = scheduler_id.ok_or_else(|| {
                    anyhow::anyhow!("offline replay running request {uuid} has no scheduler")
                })?;
                let effects = self.engine.apply_command(
                    scheduler_id,
                    Command::CancelRequest {
                        request_id: uuid,
                        discard_pending_output: true,
                    },
                    self.now_ms,
                )?;
                if !matches!(effects.result, CommandResult::Applied | CommandResult::Noop) {
                    bail!(
                        "offline replay cancellation for {uuid} returned an unexpected scheduler result"
                    );
                }
                self.apply_engine_observations(
                    effects.engine_events,
                    KvIngestBoundary::SchedulerCommand,
                )?;
            }
        }

        self.collector
            .on_terminal(uuid, self.now_ms, ReplayTerminalStatus::Canceled);
        self.requests.remove(&uuid).ok_or_else(|| {
            anyhow::anyhow!("offline replay lost request state while canceling {uuid}")
        })?;
        CoreAdmissionSource::on_terminal(
            &mut self.admission,
            uuid,
            self.now_ms,
            ReplayTerminalStatus::Canceled,
        )?;
        if !self.admission.knows_preparation_request(uuid) {
            self.progress.inc_completed();
        }
        let placements = self.placement.request_terminal(uuid, self.now_ms)?;
        self.dispatch_placements(placements)?;
        if self.cluster_in_flight() == 0
            && CoreAdmissionSource::is_drained(&self.admission)
            && self.engine.is_drained()
        {
            self.drive_pending = false;
        }
        Ok(Some(ReplayTerminalStatus::Canceled))
    }

    /// Completion-only slice of [`Self::drain_current_timestamp`]: settle every
    /// worker completion ready at the current instant without admitting
    /// arrivals or starting new passes.
    fn drain_completions(&mut self) -> anyhow::Result<()> {
        while self.apply_worker_completions()? {}
        Ok(())
    }

    /// Evaluate half of the delta cycle. Settle completions first; if one freed
    /// an in-flight slot, arm the commit half and return `true` so the caller
    /// sees the terminal and can submit a replacement at this same instant,
    /// before the freed worker is committed to its next pass. That reproduces
    /// `run()`'s in-drain admission, where `release_ready_arrivals` runs inside
    /// the same drain. Otherwise complete the full drain here.
    fn evaluate_completions_and_maybe_defer(&mut self) -> anyhow::Result<bool> {
        let before_in_flight = self.cluster_in_flight();
        self.drain_completions()?;
        if self.cluster_in_flight() < before_in_flight {
            self.drive_pending = true;
            return Ok(true);
        }
        self.drain_current_timestamp()?;
        Ok(false)
    }

    /// Advance at most through `until_ms`, returning the simulated time the
    /// step ended at. A non-finite `until_ms` runs to the next event whenever
    /// one exists.
    ///
    /// # Delta-cycle invariant
    ///
    /// Every decrease of [`Self::cluster_in_flight`] within a step sets
    /// `step_freed_slot`, and a set `step_freed_slot` forbids time advance.
    /// The caller's step loop is therefore the outer delta loop: this function
    /// unrolls the fixpoint exactly once per call, and the caller supplies the
    /// remaining rounds. A cascade at a single instant — an admission whose
    /// pass is rejected inline, freeing another slot at the same `now_ms` —
    /// cannot diverge for that reason: it holds the instant like any other
    /// terminal, and the next call resumes at the same `now_ms`.
    ///
    /// The `debug_assert!` below is the belt. `step_freed_slot` is set at the
    /// terminal rather than at the accounting, so a future decrement path that
    /// bypasses the terminal would break the invariant silently instead of
    /// failing a test.
    pub(crate) fn step_dynamic_until(&mut self, until_ms: f64) -> anyhow::Result<f64> {
        if until_ms.is_nan() || until_ms < self.now_ms {
            bail!(
                "aggregated step deadline {until_ms}ms precedes runtime time {}ms",
                self.now_ms
            );
        }
        self.step_freed_slot = false;
        let entry_in_flight = self.cluster_in_flight();
        if self.drive_pending {
            self.drive_pending = false;
            self.drain_current_timestamp()?;
        } else if self.evaluate_completions_and_maybe_defer()? {
            return Ok(self.now_ms);
        }

        // Hold the instant whenever this step surfaced anything the caller must
        // react to. A freed slot counts even when it produced no token: the
        // delta cycle's guarantee is that a step which frees a slot returns
        // before the freed worker is committed to its next pass.
        if self.step_tokens.is_empty()
            && !self.step_freed_slot
            && let Some(next_ms) = self.next_timestamp()
        {
            debug_assert!(
                self.cluster_in_flight() >= entry_in_flight,
                "in-flight fell from {entry_in_flight} to {} without setting \
                 step_freed_slot; a step that frees a slot must hold its instant",
                self.cluster_in_flight()
            );
            if next_ms <= until_ms {
                self.advance_now_ms(next_ms);
                if self.evaluate_completions_and_maybe_defer()? {
                    return Ok(self.now_ms);
                }
            } else if until_ms.is_finite() {
                self.advance_now_ms(until_ms);
            }
        }
        Ok(self.now_ms)
    }

    /// Drain the accumulated measurements into a report stamped with `wall_ms`,
    /// leaving this runtime's collector empty. G3 counters remain cumulative
    /// over the reusable runtime, like its retained cache contents.
    pub(crate) fn take_report_dynamic(
        &mut self,
        wall_ms: f64,
    ) -> anyhow::Result<crate::replay::ReplayReport> {
        anyhow::ensure!(wall_ms.is_finite(), "replay report wall_ms must be finite");
        anyhow::ensure!(
            self.is_workload_done(),
            "replay report requires an idle runtime"
        );
        let next_evidence = ReplayEvidenceCollector::new(self.evidence.options());
        self.collector
            .set_runtime_evidence(std::mem::replace(&mut self.evidence, next_evidence).finish());
        self.collector.g3_offload = self.engine.g3_stats();
        if let Some(phases) = self.admission.agentic_phase_evidence() {
            self.collector.set_agentic_phases(phases);
        }
        Ok(self
            .collector
            .take_report(self.now_ms)
            .with_wall_time_ms(wall_ms))
    }
}
