// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Preparation traffic is excluded from the formal measurement cohort. The
//! default repeats saved prefixes; the explicit live policy advances the DAG.
//!
//! Live policy adapted from the Apache-2.0 NVIDIA AIPerf implementation:
//! https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/strategies/agentic_replay.py
//! Modified for native deterministic scheduling and server-settlement barriers.
//!
//! Methodology reference for one-token primers and ten warmups per lane:
//! https://github.com/SemiAnalysisAI/InferenceX-app/blob/9bb7b13eb4985217a6282f340459fd5948613276/packages/app/src/components/datasets/agentx-methodology-article.tsx
//! This module implements those requirements using AISimulate's prepared plays
//! and native completion feedback. Repeating saved prefixes without advancing
//! the frontier is an AISimulate policy. See docs/replay/agentic/warmup.md for the
//! reference scope, upstream licenses, and separate AIPerf snapshot comparison.

use anyhow::{Context, Result, ensure};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::driver::{AgenticRuntimeFeedback, deferred_request_with_hashes};
use super::{AgenticPlaySnapshot, CompactReadyTurn, PreparedAgenticSnapshots};
use crate::replay::protocol::{DirectRequest, ReplayPromptTokenSource, ReplayRequestContext};
use crate::replay::{AgenticRuntimeIdentity, ReplayTerminalStatus};

pub const AGENTIC_PHASE_SCHEMA_V1: &str = "aisimulate.agentic.phases.v1";
pub const AGENTIC_WARMUP_REQUESTS_PER_LANE: usize = 10;
const PREPARATION_UUID_BASE: u128 = 1_u128 << 127;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgenticWarmupPolicy {
    #[default]
    RepeatedPrefixes,
    LiveTrajectoryV1,
}

impl AgenticWarmupPolicy {
    fn is_repeated(&self) -> bool {
        *self == Self::RepeatedPrefixes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgenticReplayPhase {
    Primer,
    Warmup,
    Profile,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgenticPreparationTransition {
    OpenProfile,
    Aborted,
}

/// Request phase is fixed at preparation, including for later server events.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AgenticPhaseRequest {
    pub uuid: Uuid,
    pub phase: AgenticReplayPhase,
    pub lane_id: usize,
    pub source_request_id: String,
    pub identity: AgenticRuntimeIdentity,
    pub input_length: usize,
    pub max_output_tokens: usize,
    /// Tokens covered by cacheable complete prompt blocks, not resident KV or a hit.
    /// Same-length admission recomputes the final input token, so its reuse is
    /// at most `floor((input_length - 1) / engine_block_size) * engine_block_size`
    /// for nonempty inputs, even if every complete prompt block is cached.
    pub expected_full_block_tokens: usize,
    pub dispatched_at_ms: Option<f64>,
    pub terminal_status: Option<ReplayTerminalStatus>,
    pub causal_terminal_at_ms: Option<f64>,
    pub quiescent_at_ms: Option<f64>,
    pub first_admit_ms: Option<f64>,
    pub first_admission_reused_input_tokens: Option<usize>,
    pub observed_output_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AgenticPhaseLane {
    pub lane_id: usize,
    pub play_id: String,
    pub primers_expected: usize,
    pub primers_completed: usize,
    pub warmup_expected: usize,
    pub warmup_completed: usize,
    pub requests_quiescent: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AgenticPhaseEvidence {
    pub schema: &'static str,
    pub phase: AgenticReplayPhase,
    pub barrier_condition: &'static str,
    pub profile_start_ms: Option<f64>,
    pub finished_at_ms: Option<f64>,
    pub failure_request_id: Option<Uuid>,
    pub failure_reason: Option<String>,
    pub lanes: Vec<AgenticPhaseLane>,
    pub requests: Vec<AgenticPhaseRequest>,
    #[serde(skip_serializing_if = "AgenticWarmupPolicy::is_repeated")]
    pub warmup_policy: AgenticWarmupPolicy,
}

#[derive(Debug)]
struct PreparationLane {
    snapshot: AgenticPlaySnapshot,
    source_indices: Vec<usize>,
    request_indices: Vec<usize>,
    next: usize,
    in_flight: Option<Uuid>,
    ready_at_ms: f64,
    primer_times_ms: Vec<f64>,
}

#[derive(Debug)]
pub(super) struct AgenticPreparation {
    lanes: Vec<PreparationLane>,
    requests: Vec<AgenticPhaseRequest>,
    index_by_uuid: FxHashMap<Uuid, usize>,
    failure_request_id: Option<Uuid>,
    failure_reason: Option<String>,
    transition: Option<AgenticPreparationTransition>,
    finished_at_ms: Option<f64>,
    last_event_at_ms: f64,
    engine_block_size: u32,
    include_replay_hashes: bool,
    policy: AgenticWarmupPolicy,
    live_started: bool,
    live_counts: Vec<usize>,
    live_requests: FxHashSet<Uuid>,
}

impl AgenticPreparation {
    pub(super) fn new(
        prepared: &PreparedAgenticSnapshots,
        engine_block_size: u32,
        include_replay_hashes: bool,
    ) -> Result<Self> {
        Self::with_policy(
            prepared,
            engine_block_size,
            include_replay_hashes,
            AgenticWarmupPolicy::RepeatedPrefixes,
            10_000.0,
        )
    }

    pub(super) fn with_policy(
        prepared: &PreparedAgenticSnapshots,
        engine_block_size: u32,
        include_replay_hashes: bool,
        policy: AgenticWarmupPolicy,
        primer_idle_cap_ms: f64,
    ) -> Result<Self> {
        ensure!(
            engine_block_size > 0,
            "warmup engine block size must be positive"
        );
        let mut this = Self {
            lanes: Vec::new(),
            requests: Vec::new(),
            index_by_uuid: FxHashMap::default(),
            failure_request_id: None,
            failure_reason: None,
            transition: None,
            finished_at_ms: None,
            last_event_at_ms: 0.0,
            engine_block_size,
            include_replay_hashes,
            policy,
            live_started: false,
            live_counts: vec![0; prepared.context().lane_count()],
            live_requests: FxHashSet::default(),
        };
        for snapshot in &prepared.plays {
            // Snapshot UUIDs use checked arithmetic. Check the separate domain
            // explicitly instead of assuming a hash cannot collide.
            for request in &snapshot.evidence.requests {
                let index = snapshot.source_index(&request.source_request_id)?;
                ensure!(
                    snapshot.request_uuid(index)?.as_u128() < PREPARATION_UUID_BASE,
                    "snapshot request UUID overlaps the preparation UUID domain"
                );
            }
            let primers = snapshot
                .evidence
                .primers
                .iter()
                .map(|primer| snapshot.source_index(&primer.source_request_id))
                .collect::<Result<Vec<_>>>()?;
            let warmup_sources = if primers.is_empty() {
                let first = snapshot
                    .evidence
                    .requests
                    .iter()
                    .filter(|request| !request.historical)
                    .min_by(|left, right| {
                        left.recorded_start_ms.total_cmp(&right.recorded_start_ms)
                    })
                    .context("warmup snapshot has no retained request")?;
                vec![snapshot.source_index(&first.source_request_id)?]
            } else {
                primers.clone()
            };
            let mut source_indices = primers.clone();
            if policy == AgenticWarmupPolicy::RepeatedPrefixes {
                source_indices.extend(
                    (0..AGENTIC_WARMUP_REQUESTS_PER_LANE)
                        .map(|ordinal| warmup_sources[ordinal % warmup_sources.len()]),
                );
            }
            let mut request_indices = Vec::with_capacity(source_indices.len());
            for (ordinal, &source_index) in source_indices.iter().enumerate() {
                let index = this.requests.len();
                let uuid = Uuid::from_u128(
                    PREPARATION_UUID_BASE
                        .checked_add(index as u128)
                        .context("preparation request UUID overflow")?,
                );
                let phase = if ordinal < primers.len() {
                    AgenticReplayPhase::Primer
                } else {
                    AgenticReplayPhase::Warmup
                };
                let phase_name = if phase == AgenticReplayPhase::Primer {
                    "primer"
                } else {
                    "warmup"
                };
                let node = &snapshot.context.graph.nodes[source_index];
                super::trace::validate_synthesizable_prompt_len(
                    node.input_length,
                    node.hash_ids.len(),
                    snapshot.context.graph.block_size,
                )?;
                let mut identity = snapshot.identity_at(source_index);
                identity.request_id =
                    format!("{}:{phase_name}:{ordinal}", snapshot.evidence.play_id);
                this.requests.push(AgenticPhaseRequest {
                    uuid,
                    phase,
                    lane_id: snapshot.evidence.lane_id,
                    source_request_id: node.request_id.clone(),
                    identity,
                    input_length: node.input_length,
                    max_output_tokens: 1,
                    expected_full_block_tokens: node.input_length / engine_block_size as usize
                        * engine_block_size as usize,
                    dispatched_at_ms: None,
                    terminal_status: None,
                    causal_terminal_at_ms: None,
                    quiescent_at_ms: None,
                    first_admit_ms: None,
                    first_admission_reused_input_tokens: None,
                    observed_output_tokens: 0,
                });
                this.index_by_uuid.insert(uuid, index);
                request_indices.push(index);
            }
            this.lanes.push(PreparationLane {
                snapshot: snapshot.clone(),
                source_indices,
                request_indices,
                next: 0,
                in_flight: None,
                ready_at_ms: 0.0,
                primer_times_ms: Vec::new(),
            });
        }
        if policy == AgenticWarmupPolicy::LiveTrajectoryV1 {
            ensure!(
                this.lanes.len() == prepared.context().lane_count(),
                "live warmup requires a complete initial lane cohort"
            );
            this.lanes
                .sort_by_key(|lane| lane.snapshot.evidence.lane_id);
            let mut max_lead_ms = 0.0_f64;
            for lane in &mut this.lanes {
                lane.primer_times_ms = lane
                    .source_indices
                    .iter()
                    .map(|&index| {
                        let start = lane.snapshot.context.recorded_start_ms(index);
                        let lead = (lane.snapshot.evidence.t_star_ms - start)
                            .max(0.0)
                            .min(primer_idle_cap_ms);
                        max_lead_ms = max_lead_ms.max(lead);
                        lead
                    })
                    .collect();
            }
            for lane in &mut this.lanes {
                for lead in &mut lane.primer_times_ms {
                    *lead = max_lead_ms - *lead;
                }
            }
        }
        Ok(this)
    }

    pub(super) fn live_active(&self) -> bool {
        self.live_started && self.transition.is_none() && self.failure_request_id.is_none()
    }

    pub(super) fn uses_live_policy(&self) -> bool {
        self.policy == AgenticWarmupPolicy::LiveTrajectoryV1
    }

    pub(super) fn is_live_request(&self, uuid: Uuid) -> bool {
        self.live_requests.contains(&uuid)
    }

    pub(super) fn lane_has_budget(&self, lane: usize) -> bool {
        self.live_active() && self.live_counts[lane] < AGENTIC_WARMUP_REQUESTS_PER_LANE
    }

    pub(super) fn ready_to_start_live(&self) -> bool {
        self.policy == AgenticWarmupPolicy::LiveTrajectoryV1
            && !self.live_started
            && self.failure_request_id.is_none()
            && self
                .requests
                .iter()
                .all(|request| request.quiescent_at_ms.is_some())
    }

    pub(super) fn start_live(&mut self, now_ms: f64) {
        self.live_started = true;
        self.last_event_at_ms = now_ms;
    }

    pub(super) fn record_live_dispatch(
        &mut self,
        turn: &CompactReadyTurn,
        lane: usize,
        input_length: usize,
        identity: AgenticRuntimeIdentity,
    ) {
        debug_assert!(self.lane_has_budget(lane));
        let index = self.requests.len();
        self.lanes[lane].request_indices.push(index);
        self.index_by_uuid.insert(turn.request_uuid, index);
        self.live_requests.insert(turn.request_uuid);
        self.live_counts[lane] += 1;
        self.requests.push(AgenticPhaseRequest {
            uuid: turn.request_uuid,
            phase: AgenticReplayPhase::Warmup,
            lane_id: lane,
            source_request_id: turn.authored_request_id.clone().unwrap_or_default(),
            identity,
            input_length,
            max_output_tokens: 1,
            expected_full_block_tokens: input_length / self.engine_block_size as usize
                * self.engine_block_size as usize,
            dispatched_at_ms: Some(turn.dispatched_at_ms),
            terminal_status: None,
            causal_terminal_at_ms: None,
            quiescent_at_ms: None,
            first_admit_ms: None,
            first_admission_reused_input_tokens: None,
            observed_output_tokens: 0,
        });
    }

    pub(super) fn contains(&self, uuid: Uuid) -> bool {
        self.index_by_uuid.contains_key(&uuid)
    }

    pub(super) fn ordinal(&self, uuid: Uuid) -> Option<usize> {
        self.index_by_uuid.get(&uuid).copied()
    }

    pub(super) fn transition(&self) -> Option<AgenticPreparationTransition> {
        self.transition
    }

    pub(super) fn next_ready_time_ms(&self) -> Option<f64> {
        if self.live_started || self.failure_request_id.is_some() || self.transition.is_some() {
            return None;
        }
        self.lanes
            .iter()
            .filter(|lane| lane.in_flight.is_none() && lane.next < lane.request_indices.len())
            .map(|lane| {
                if self.policy == AgenticWarmupPolicy::LiveTrajectoryV1 {
                    lane.primer_times_ms[lane.next..]
                        .iter()
                        .copied()
                        .min_by(f64::total_cmp)
                        .unwrap()
                } else {
                    lane.ready_at_ms
                }
            })
            .min_by(f64::total_cmp)
    }

    pub(super) fn pop_ready(
        &mut self,
        now_ms: f64,
        limit: usize,
        emit_session_metadata: bool,
    ) -> Vec<CompactReadyTurn> {
        if self.live_started || self.failure_request_id.is_some() || self.transition.is_some() {
            return Vec::new();
        }
        let mut ready = Vec::new();
        if self.policy == AgenticWarmupPolicy::LiveTrajectoryV1 && !self.live_started {
            // Independent predecessor streams on one lane may prime concurrently.
            // Select each due primer once; a completion is not a lane-serial gate.
            for lane in &mut self.lanes {
                while lane.next < lane.request_indices.len() && ready.len() < limit {
                    // Per-lane source order is not time order. Find the earliest
                    // remaining primer while keeping deterministic ties.
                    let best = (lane.next..lane.request_indices.len())
                        .min_by(|&a, &b| {
                            lane.primer_times_ms[a].total_cmp(&lane.primer_times_ms[b])
                        })
                        .unwrap();
                    if lane.primer_times_ms[best] > now_ms {
                        break;
                    }
                    lane.source_indices.swap(lane.next, best);
                    lane.request_indices.swap(lane.next, best);
                    lane.primer_times_ms.swap(lane.next, best);
                    lane.ready_at_ms = lane.primer_times_ms[lane.next];
                    ready.push(Self::dispatch_preparation(
                        lane,
                        &mut self.requests,
                        now_ms,
                        emit_session_metadata,
                        self.include_replay_hashes,
                        self.engine_block_size,
                    ));
                    lane.in_flight = None;
                }
            }
            return ready;
        }
        for lane in &mut self.lanes {
            if ready.len() == limit {
                break;
            }
            if lane.in_flight.is_some()
                || lane.next == lane.request_indices.len()
                || lane.ready_at_ms > now_ms
            {
                continue;
            }
            ready.push(Self::dispatch_preparation(
                lane,
                &mut self.requests,
                now_ms,
                emit_session_metadata,
                self.include_replay_hashes,
                self.engine_block_size,
            ));
        }
        ready
    }

    fn dispatch_preparation(
        lane: &mut PreparationLane,
        requests: &mut [AgenticPhaseRequest],
        now_ms: f64,
        emit_session_metadata: bool,
        include_replay_hashes: bool,
        engine_block_size: u32,
    ) -> CompactReadyTurn {
        let ordinal = lane.next;
        let request_index = lane.request_indices[ordinal];
        let source_index = lane.source_indices[ordinal];
        let evidence = &mut requests[request_index];
        let source = &lane.snapshot.context.graph.nodes[source_index];
        let metadata = DirectRequest {
            uuid: Some(evidence.uuid),
            max_output_tokens: 1,
            // A source prefill-only request has no output plan. Its warmup
            // emits a fixed execution-only token without consuming the
            // corpus RNG or appending output to another request's input.
            output_token_ids: Some(vec![
                lane.snapshot.context.outputs[source_index]
                    .first()
                    .copied()
                    .unwrap_or(0),
            ]),
            arrival_timestamp_ms: Some(lane.ready_at_ms),
            priority: source.priority,
            strict_priority: source.strict_priority,
            policy_class: source.policy_class.clone(),
            replay_context: Some(ReplayRequestContext {
                authored_id: evidence.source_request_id.clone(),
                session_id: Some(evidence.identity.conversation_id.clone()),
                turn_index: Some(ordinal),
                metadata: serde_json::json!({"agentic_phase": evidence.phase}),
                prompt_token_source: ReplayPromptTokenSource::Materialized,
                agentic: Some(evidence.identity.clone()),
            }),
            ..Default::default()
        };
        let (request, replay_hashes) = deferred_request_with_hashes(
            metadata,
            source.input_length,
            lane.snapshot.token_ids(source_index),
            lane.snapshot.context.graph.block_size,
            include_replay_hashes.then_some(engine_block_size),
        );
        evidence.dispatched_at_ms = Some(now_ms);
        lane.in_flight = Some(evidence.uuid);
        lane.next += 1;
        CompactReadyTurn {
            request_uuid: evidence.uuid,
            authored_request_id: Some(evidence.source_request_id.clone()),
            play_id: Some(evidence.identity.play_id.clone()),
            dispatched_at_ms: now_ms,
            session_id: evidence.identity.conversation_id.clone(),
            turn_index: ordinal,
            replay_key: None,
            scheduled_ready_at_ms: lane.ready_at_ms,
            replay_hashes,
            emit_session_metadata,
            synthetic_session_id: false,
            request,
        }
    }

    fn request(&self, uuid: Uuid) -> Result<&AgenticPhaseRequest> {
        let request = &self.requests[*self
            .index_by_uuid
            .get(&uuid)
            .with_context(|| format!("unknown preparation request {uuid}"))?];
        ensure!(
            request.dispatched_at_ms.is_some(),
            "preparation request {uuid} was not dispatched"
        );
        Ok(request)
    }

    fn validate_time(&self, now_ms: f64) -> Result<()> {
        ensure!(
            now_ms.is_finite() && now_ms >= self.last_event_at_ms,
            "preparation feedback timestamp must be finite and nondecreasing"
        );
        Ok(())
    }

    pub(super) fn validate_boundary_time(&self, now_ms: f64) -> Result<()> {
        self.validate_time(now_ms)
    }

    pub(super) fn validate_feedback(&self, feedback: &AgenticRuntimeFeedback) -> Result<()> {
        self.validate_time(feedback.at_ms)?;
        let mut terminal = FxHashSet::default();
        let mut outputs = FxHashSet::default();
        let mut quiescent = FxHashSet::default();
        for output in &feedback.output_tokens {
            if !self.contains(output.request_uuid) {
                continue;
            }
            let request = self.request(output.request_uuid)?;
            ensure!(
                feedback.at_ms >= request.dispatched_at_ms.unwrap(),
                "preparation output precedes dispatch"
            );
            ensure!(
                outputs.insert(output.request_uuid),
                "duplicate preparation output group"
            );
            ensure!(
                request.terminal_status.is_none(),
                "preparation output arrived after terminal"
            );
            ensure!(
                request.observed_output_tokens + output.token_ids.len() <= 1,
                "preparation request emitted more than one output token"
            );
        }
        for event in &feedback.causal_terminals {
            if !self.contains(event.request_uuid) {
                continue;
            }
            let request = self.request(event.request_uuid)?;
            ensure!(
                terminal.insert(event.request_uuid) && request.terminal_status.is_none(),
                "duplicate preparation causal terminal"
            );
            ensure!(
                feedback.at_ms >= request.dispatched_at_ms.unwrap(),
                "preparation terminal precedes dispatch"
            );
        }
        for &uuid in &feedback.quiescent_requests {
            if !self.contains(uuid) {
                continue;
            }
            let request = self.request(uuid)?;
            ensure!(
                quiescent.insert(uuid) && request.quiescent_at_ms.is_none(),
                "duplicate preparation quiescence"
            );
            ensure!(
                request.terminal_status.is_some() || terminal.contains(&uuid),
                "preparation quiescence precedes causal terminal"
            );
        }
        Ok(())
    }

    pub(super) fn on_output(&mut self, uuid: Uuid) -> Result<()> {
        let request = self.request(uuid)?;
        ensure!(
            request.terminal_status.is_none() && request.observed_output_tokens == 0,
            "preparation output exceeds its one-token contract or follows terminal"
        );
        self.requests[self.index_by_uuid[&uuid]].observed_output_tokens += 1;
        Ok(())
    }

    pub(super) fn on_terminal(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        status: ReplayTerminalStatus,
    ) -> Result<()> {
        self.validate_time(now_ms)?;
        let request = self.request(uuid)?;
        ensure!(
            request.terminal_status.is_none(),
            "duplicate preparation causal terminal"
        );
        ensure!(
            now_ms >= request.dispatched_at_ms.unwrap(),
            "preparation terminal precedes dispatch"
        );
        let index = self.index_by_uuid[&uuid];
        self.requests[index].terminal_status = Some(status);
        self.requests[index].causal_terminal_at_ms = Some(now_ms);
        self.last_event_at_ms = now_ms;
        if self.failure_request_id.is_none() {
            if status != ReplayTerminalStatus::Completed {
                self.failure_request_id = Some(uuid);
                self.failure_reason = Some(format!("preparation request ended with {status:?}"));
            } else if self.requests[index].observed_output_tokens != 1 {
                self.failure_request_id = Some(uuid);
                self.failure_reason =
                    Some("preparation request completed without exactly one output token".into());
            }
        }
        if self.policy == AgenticWarmupPolicy::LiveTrajectoryV1 {
            return Ok(());
        }
        let lane = self
            .lanes
            .iter_mut()
            .find(|lane| lane.in_flight == Some(uuid))
            .expect("dispatched preparation request retains its lane until terminal");
        lane.in_flight = None;
        lane.ready_at_ms = now_ms;
        Ok(())
    }

    pub(super) fn on_quiescent(&mut self, uuid: Uuid, now_ms: f64) -> Result<()> {
        self.validate_time(now_ms)?;
        let request = self.request(uuid)?;
        ensure!(
            request.causal_terminal_at_ms.is_some_and(|at| at <= now_ms),
            "preparation quiescence precedes causal terminal"
        );
        ensure!(
            request.quiescent_at_ms.is_none(),
            "duplicate preparation quiescence"
        );
        self.requests[self.index_by_uuid[&uuid]].quiescent_at_ms = Some(now_ms);
        self.last_event_at_ms = now_ms;
        Ok(())
    }

    pub(super) fn ready_transition(
        &self,
        now_ms: f64,
    ) -> Result<Option<AgenticPreparationTransition>> {
        self.validate_time(now_ms)?;
        if self.transition.is_some() {
            return Ok(None);
        }
        if self
            .requests
            .iter()
            .any(|request| request.dispatched_at_ms.is_some() && request.quiescent_at_ms.is_none())
        {
            return Ok(None);
        }
        if self.failure_request_id.is_some() {
            return Ok(Some(AgenticPreparationTransition::Aborted));
        }
        if (self.policy == AgenticWarmupPolicy::RepeatedPrefixes
            && self
                .lanes
                .iter()
                .all(|lane| lane.next == lane.request_indices.len()))
            || (self.live_started
                && self
                    .live_counts
                    .iter()
                    .all(|&count| count == AGENTIC_WARMUP_REQUESTS_PER_LANE))
        {
            return Ok(Some(AgenticPreparationTransition::OpenProfile));
        }
        Ok(None)
    }

    pub(super) fn finish(&mut self, transition: AgenticPreparationTransition, now_ms: f64) {
        self.transition = Some(transition);
        self.finished_at_ms = Some(now_ms);
        self.last_event_at_ms = now_ms;
    }

    pub(super) fn dispatched_ids(&self) -> Vec<Uuid> {
        self.requests
            .iter()
            .filter(|request| request.dispatched_at_ms.is_some())
            .map(|request| request.uuid)
            .collect()
    }

    pub(super) fn record_admission(&mut self, uuid: Uuid, at_ms: f64, reused: usize) -> Result<()> {
        let request = self.request(uuid)?;
        ensure!(
            at_ms.is_finite() && at_ms >= request.dispatched_at_ms.unwrap(),
            "preparation admission must follow dispatch at a finite time"
        );
        ensure!(
            reused <= request.input_length,
            "preparation reuse exceeds original input length"
        );
        if request.first_admit_ms.is_some() {
            return Ok(());
        }
        let request = &mut self.requests[self.index_by_uuid[&uuid]];
        request.first_admit_ms = Some(at_ms);
        request.first_admission_reused_input_tokens = Some(reused);
        Ok(())
    }

    pub(super) fn evidence(&self) -> AgenticPhaseEvidence {
        let phase = match self.transition {
            Some(AgenticPreparationTransition::OpenProfile) => AgenticReplayPhase::Profile,
            Some(AgenticPreparationTransition::Aborted) => AgenticReplayPhase::Aborted,
            None if self.requests.iter().any(|request| {
                request.phase == AgenticReplayPhase::Primer && request.terminal_status.is_none()
            }) =>
            {
                AgenticReplayPhase::Primer
            }
            None => AgenticReplayPhase::Warmup,
        };
        AgenticPhaseEvidence {
            warmup_policy: self.policy,
            schema: AGENTIC_PHASE_SCHEMA_V1,
            phase,
            barrier_condition: "all_preparation_succeeded_and_server_quiescent",
            profile_start_ms: (self.transition == Some(AgenticPreparationTransition::OpenProfile))
                .then_some(self.finished_at_ms)
                .flatten(),
            finished_at_ms: self.finished_at_ms,
            failure_request_id: self.failure_request_id,
            failure_reason: self.failure_reason.clone(),
            lanes: self
                .lanes
                .iter()
                .map(|lane| {
                    let requests = lane
                        .request_indices
                        .iter()
                        .map(|&index| &self.requests[index])
                        .collect::<Vec<_>>();
                    AgenticPhaseLane {
                        lane_id: lane.snapshot.evidence.lane_id,
                        play_id: lane.snapshot.evidence.play_id.clone(),
                        primers_expected: requests
                            .iter()
                            .filter(|r| r.phase == AgenticReplayPhase::Primer)
                            .count(),
                        primers_completed: requests
                            .iter()
                            .filter(|r| {
                                r.phase == AgenticReplayPhase::Primer
                                    && r.terminal_status == Some(ReplayTerminalStatus::Completed)
                                    && r.observed_output_tokens == 1
                            })
                            .count(),
                        warmup_expected: AGENTIC_WARMUP_REQUESTS_PER_LANE,
                        warmup_completed: requests
                            .iter()
                            .filter(|r| {
                                r.phase == AgenticReplayPhase::Warmup
                                    && r.terminal_status == Some(ReplayTerminalStatus::Completed)
                                    && r.observed_output_tokens == 1
                            })
                            .count(),
                        requests_quiescent: requests
                            .iter()
                            .filter(|r| r.quiescent_at_ms.is_some())
                            .count(),
                    }
                })
                .collect(),
            requests: self.requests.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::loadgen::{
        AGENTIC_MOONCAKE_SCHEMA, AGENTIC_MOONCAKE_VERSION, AgenticDependency,
        AgenticDependencyRelation, AgenticDependencyTrigger, AgenticHashIdScope,
        AgenticMooncakeHeader, AgenticMooncakeRow, AgenticOutputFeedback, AgenticSnapshotOptions,
        AgenticSourceProvenance, AgenticTerminalFeedback, ValidatedAgenticGraph, WorkloadDriver,
    };

    fn prepared(lanes: usize, cut: f64) -> PreparedAgenticSnapshots {
        let row =
            |id: &str, conversation: &str, start: f64, input_length, hashes| AgenticMooncakeRow {
                request_id: id.into(),
                play_id: "source".into(),
                session_id: conversation.into(),
                model: "model".into(),
                input_length: Some(input_length),
                output_length: Some(3),
                hash_ids: Some(hashes),
                not_before_ms: start,
                recorded_api_time_ms: Some(5.0),
                ..Default::default()
            };
        let edge = |source: &str, relation, trigger| AgenticDependency {
            request_id: source.into(),
            relation,
            trigger,
            delay_ms: 0.0,
        };
        let first = row("a-history", "main", 0.0, 128, vec![10, 20]);
        let mut child = row("b-history", "child", 20.0, 191, vec![10, 20, 30]);
        child.dependencies = vec![edge(
            "a-history",
            AgenticDependencyRelation::Spawn,
            AgenticDependencyTrigger::Dispatch,
        )];
        let mut next = row("a-future", "main", 100.0, 192, vec![10, 20, 40]);
        next.dependencies = vec![edge(
            "a-history",
            AgenticDependencyRelation::Sequence,
            AgenticDependencyTrigger::Completion,
        )];
        let mut next_child = row("b-future", "child", 140.0, 192, vec![10, 20, 30]);
        next_child.dependencies = vec![edge(
            "b-history",
            AgenticDependencyRelation::Sequence,
            AgenticDependencyTrigger::Completion,
        )];
        let mut join = row("join", "main", 200.0, 256, vec![10, 20, 40, 50]);
        join.dependencies = vec![
            edge(
                "a-future",
                AgenticDependencyRelation::Sequence,
                AgenticDependencyTrigger::Completion,
            ),
            edge(
                "b-future",
                AgenticDependencyRelation::Join,
                AgenticDependencyTrigger::Completion,
            ),
        ];
        let graph = ValidatedAgenticGraph::from_agentic_mooncake_rows(
            AgenticMooncakeHeader {
                schema: AGENTIC_MOONCAKE_SCHEMA.into(),
                version: AGENTIC_MOONCAKE_VERSION,
                block_size: 64,
                hash_id_scope: AgenticHashIdScope::Local,
                source: AgenticSourceProvenance {
                    format: "self-authored".into(),
                    digest: "phase-fixture-v1".into(),
                },
            },
            vec![first, child, next, next_child, join],
        )
        .unwrap();
        let sampled = graph
            .prepare_snapshots(lanes, AgenticSnapshotOptions::seeded(42))
            .unwrap();
        PreparedAgenticSnapshots::from_plays(
            (0..lanes)
                .map(|lane| sampled.context().prepare_play(lane, 0, Some(cut)).unwrap())
                .collect(),
        )
        .unwrap()
    }

    fn live_driver(lanes: usize, cut: f64) -> WorkloadDriver {
        let mut driver = WorkloadDriver::new_agentic_warmup_with_policy(
            prepared(lanes, cut),
            64,
            true,
            1.0,
            AgenticWarmupPolicy::LiveTrajectoryV1,
            10_000.0,
        )
        .unwrap();
        driver
            .enable_agentic_profile(super::super::AgenticProfileOptions::default())
            .unwrap();
        driver
    }

    #[test]
    fn live_warmup_advances_dag_recycles_and_preserves_the_formal_frontier() {
        let mut driver = live_driver(1, 50.0);
        let mut now = 0.0;
        let mut source_ids = Vec::new();
        let mut late_quiescence = Vec::new();
        for _ in 0..30 {
            driver.finish_agentic_preparation(now).unwrap();
            driver.advance_agentic_profile(now).unwrap();
            let next = driver
                .next_ready_time_ms()
                .expect("preparation owns a concrete next dispatch");
            now = now.max(next);
            let ready = driver.pop_ready_compact(now, 100);
            assert!(!ready.is_empty());
            for turn in &ready {
                assert!(turn.request.materialized_tokens().is_none());
                assert_eq!(turn.request.metadata().max_output_tokens, 1);
                assert_eq!(
                    turn.request
                        .metadata()
                        .output_token_ids
                        .as_ref()
                        .unwrap()
                        .len(),
                    1
                );
                assert!(driver.knows_preparation_request(turn.request_uuid));
                source_ids.push(turn.authored_request_id.clone().unwrap());
                driver.on_output_token(turn.request_uuid, 0).unwrap();
            }
            now += 1.0;
            let counts = driver.agentic_phase_evidence().unwrap();
            let last = counts
                .requests
                .iter()
                .filter(|r| r.phase == AgenticReplayPhase::Warmup)
                .count()
                == 10;
            for turn in ready {
                driver
                    .on_causal_terminal(turn.request_uuid, now, ReplayTerminalStatus::Completed)
                    .unwrap();
                if last {
                    late_quiescence.push(turn.request_uuid);
                } else {
                    driver.on_quiescent(turn.request_uuid, now).unwrap();
                }
            }
            if last {
                break;
            }
        }
        assert_eq!(
            source_ids,
            [
                "a-history",
                "b-history", // globally spread primers
                "a-future",
                "b-future",
                "join", // live sampled suffix
                "a-history",
                "b-history",
                "a-future",
                "b-future",
                "join", // first recycle
                "a-history",
                "b-history", // second recycle reaches quota
            ]
        );
        assert!(!late_quiescence.is_empty());
        assert_eq!(
            driver.next_ready_time_ms(),
            None,
            "quota-paused requests must not spin"
        );
        assert_eq!(driver.finish_agentic_preparation(now).unwrap(), None);
        assert!(driver.pop_ready_compact(now, 100).is_empty());
        now += 100.0;
        for uuid in late_quiescence {
            driver.on_quiescent(uuid, now).unwrap();
        }
        assert_eq!(
            driver.finish_agentic_preparation(now).unwrap(),
            Some(AgenticPreparationTransition::OpenProfile)
        );
        let evidence = driver.agentic_phase_evidence().unwrap();
        assert_eq!(
            evidence.warmup_policy,
            AgenticWarmupPolicy::LiveTrajectoryV1
        );
        assert_eq!(evidence.lanes[0].primers_completed, 2);
        assert_eq!(evidence.lanes[0].warmup_completed, 10);
        assert_eq!(evidence.lanes[0].requests_quiescent, 12);
        let profile = driver.agentic_profile_report().unwrap();
        assert_eq!(profile.corpus_cursor, 3);
        assert_eq!(profile.issued_requests, 0);
        assert_eq!(profile.plays_started, 1);
        let first = driver.pop_ready(now, 100);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].authored_request_id.as_deref(), Some("a-future"));
        assert_eq!(first[0].request.max_output_tokens, 3);
        assert!(!driver.knows_preparation_request(first[0].request_uuid));
        assert_eq!(
            driver.next_ready_time_ms(),
            Some(now + 40.0),
            "barrier waiting must not consume source-clock spacing"
        );
        let identity = first[0]
            .request
            .replay_context
            .as_ref()
            .unwrap()
            .agentic
            .as_ref()
            .unwrap();
        assert_eq!(
            identity.cache_id,
            evidence.requests.last().unwrap().identity.cache_id
        );
    }

    #[test]
    fn live_primers_are_concurrent_and_globally_aligned() {
        let mut driver = live_driver(2, 50.0);
        let first = driver.pop_ready(0.0, 100);
        assert_eq!(first.len(), 2);
        assert!(
            first
                .iter()
                .all(|turn| turn.authored_request_id.as_deref() == Some("a-history"))
        );
        assert_eq!(driver.next_ready_time_ms(), Some(20.0));
        let second = driver.pop_ready(20.0, 100);
        assert_eq!(
            second.len(),
            2,
            "child primers need not wait for their lane's root primer response"
        );
        assert!(
            second
                .iter()
                .all(|turn| turn.authored_request_id.as_deref() == Some("b-history"))
        );
        for turn in first.into_iter().chain(second) {
            driver.on_output_token(turn.request_uuid, 0).unwrap();
            driver
                .on_terminal(turn.request_uuid, 30.0, ReplayTerminalStatus::Completed)
                .unwrap();
        }
        assert_eq!(driver.finish_agentic_preparation(30.0).unwrap(), None);
        let live = driver.pop_ready(30.0, 100);
        assert_eq!(live.len(), 4);
        assert!(live.iter().all(|turn| matches!(
            turn.authored_request_id.as_deref(),
            Some("a-future" | "b-future")
        )));
    }

    #[test]
    fn failed_live_warmup_drains_and_aborts_before_profile() {
        let mut driver = live_driver(1, 0.0);
        driver.finish_agentic_preparation(0.0).unwrap(); // no primers
        let ready = driver.pop_ready(0.0, 100);
        assert_eq!(ready.len(), 2);
        driver
            .on_causal_terminal(ready[0].request_uuid, 1.0, ReplayTerminalStatus::Failed)
            .unwrap();
        driver.on_output_token(ready[1].request_uuid, 0).unwrap();
        driver
            .on_causal_terminal(ready[1].request_uuid, 1.0, ReplayTerminalStatus::Completed)
            .unwrap();
        assert_eq!(driver.finish_agentic_preparation(1.0).unwrap(), None);
        assert!(driver.pop_ready(1.0, 100).is_empty());
        for turn in ready {
            driver.on_quiescent(turn.request_uuid, 5.0).unwrap();
        }
        assert_eq!(
            driver.finish_agentic_preparation(5.0).unwrap(),
            Some(AgenticPreparationTransition::Aborted)
        );
        assert_eq!(
            driver.agentic_profile_report().unwrap().profile_start_ms,
            None
        );
    }

    fn linear_live_rows(nodes: usize, rootless: bool) -> Vec<AgenticMooncakeRow> {
        (0..nodes)
            .map(|index| AgenticMooncakeRow {
                request_id: format!("n{index}"),
                play_id: "linear".into(),
                session_id: if rootless && index > 0 {
                    "child"
                } else {
                    "main"
                }
                .into(),
                model: "model".into(),
                input_length: Some(64),
                output_length: Some(3),
                hash_ids: Some(vec![index as u64 + 1]),
                not_before_ms: index as f64,
                recorded_api_time_ms: Some(0.1),
                dependencies: if index == 0 {
                    vec![]
                } else {
                    vec![AgenticDependency {
                        request_id: format!("n{}", index - 1),
                        delay_ms: 100.0,
                        relation: if rootless && index == 1 {
                            AgenticDependencyRelation::Spawn
                        } else {
                            AgenticDependencyRelation::Sequence
                        },
                        trigger: if rootless && index == 1 {
                            AgenticDependencyTrigger::Dispatch
                        } else {
                            AgenticDependencyTrigger::Completion
                        },
                    }]
                },
                ..Default::default()
            })
            .collect()
    }

    fn rows_live_driver(
        rows: Vec<AgenticMooncakeRow>,
        lanes: usize,
        cut: f64,
        speedup: f64,
    ) -> WorkloadDriver {
        let graph = ValidatedAgenticGraph::from_agentic_mooncake_rows(
            AgenticMooncakeHeader {
                schema: AGENTIC_MOONCAKE_SCHEMA.into(),
                version: AGENTIC_MOONCAKE_VERSION,
                block_size: 64,
                hash_id_scope: AgenticHashIdScope::Local,
                source: AgenticSourceProvenance {
                    format: "self-authored".into(),
                    digest: "live-linear".into(),
                },
            },
            rows,
        )
        .unwrap();
        let prepared = graph
            .prepare_snapshots(lanes, AgenticSnapshotOptions::seeded(42))
            .unwrap();
        let plays = (0..lanes)
            .map(|lane| prepared.context().prepare_play(lane, 0, Some(cut)).unwrap())
            .collect();
        let mut driver = WorkloadDriver::new_agentic_warmup_with_policy(
            PreparedAgenticSnapshots::from_plays(plays).unwrap(),
            64,
            true,
            speedup,
            AgenticWarmupPolicy::LiveTrajectoryV1,
            10_000.0,
        )
        .unwrap();
        driver
            .enable_agentic_profile(super::super::AgenticProfileOptions::default())
            .unwrap();
        driver
    }

    fn linear_live_driver(nodes: usize, rootless: bool) -> WorkloadDriver {
        rows_live_driver(
            linear_live_rows(nodes, rootless),
            1,
            if rootless { 0.5 } else { 0.0 },
            1.0,
        )
    }

    fn finish_ten_live_rounds(driver: &mut WorkloadDriver, lanes: usize) -> f64 {
        driver.finish_agentic_preparation(0.0).unwrap();
        for round in 0..10 {
            let now = round as f64;
            driver.advance_agentic_profile(now).unwrap();
            let ready = driver.pop_ready(now, lanes);
            assert_eq!(ready.len(), lanes);
            for turn in ready {
                driver.on_output_token(turn.request_uuid, 0).unwrap();
                driver
                    .on_terminal(
                        turn.request_uuid,
                        now + 1.0,
                        ReplayTerminalStatus::Completed,
                    )
                    .unwrap();
            }
        }
        assert_eq!(
            driver.finish_agentic_preparation(10.0).unwrap(),
            Some(AgenticPreparationTransition::OpenProfile)
        );
        10.0
    }

    #[test]
    fn live_handoff_idle_cap_does_not_restore_future_continuation_timestamps() {
        let mut rows = linear_live_rows(13, false);
        rows[11].not_before_ms = 601_000.0;
        rows[11].dependencies[0].delay_ms = 600_000.0;
        rows[12].not_before_ms = 605_001.0;
        rows[12].dependencies[0].delay_ms = 4_000.0;
        let mut driver = rows_live_driver(rows, 2, 0.0, 1.0);
        let start = finish_ten_live_rounds(&mut driver, 2);
        let frontier = driver.pop_ready(start, 2);
        assert_eq!(frontier.len(), 2);
        assert!(
            frontier
                .iter()
                .all(|turn| turn.authored_request_id.as_deref() == Some("n10"))
        );
        // Keep the second lane busy so only the 300-second per-tree guard fires.
        driver
            .on_terminal(
                frontier[0].request_uuid,
                start + 1.0,
                ReplayTerminalStatus::Completed,
            )
            .unwrap();
        let capped = start + 1.0 + 300_000.0;
        assert_eq!(driver.next_ready_time_ms(), Some(capped));
        driver.advance_agentic_profile(capped).unwrap();
        let delayed = driver.pop_ready(capped, 1).pop().unwrap();
        assert_eq!(delayed.authored_request_id.as_deref(), Some("n11"));
        driver
            .on_terminal(
                delayed.request_uuid,
                capped + 1.0,
                ReplayTerminalStatus::Completed,
            )
            .unwrap();
        let continuation = capped + 1.0 + 4_000.0;
        assert_eq!(driver.next_ready_time_ms(), Some(continuation));
        let ready = driver.pop_ready(continuation, 1).pop().unwrap();
        assert_eq!(ready.authored_request_id.as_deref(), Some("n12"));
        assert_eq!(
            driver.agentic_profile_report().unwrap().idle_shifts.len(),
            1
        );
    }

    #[rstest::rstest]
    #[case::join(AgenticDependencyRelation::Join)]
    #[case::replay_barrier(AgenticDependencyRelation::ReplayBarrier)]
    fn live_handoff_preserves_blocked_frontier_but_not_later_turns(
        #[case] relation: AgenticDependencyRelation,
    ) {
        let mut rows = linear_live_rows(12, false);
        rows[10].not_before_ms = 5_000.0;
        rows[10].recorded_api_time_ms = Some(10_000.0);
        rows[10].dependencies.push(AgenticDependency {
            request_id: "child".into(),
            delay_ms: 0.0,
            relation,
            trigger: AgenticDependencyTrigger::Completion,
        });
        rows[11].not_before_ms = 19_000.0;
        rows[11].dependencies[0].delay_ms = 4_000.0;
        let mut child = rows[0].clone();
        child.request_id = "child".into();
        child.session_id = "child".into();
        child.not_before_ms = 1_000.0;
        child.dependencies = vec![AgenticDependency {
            request_id: "n9".into(),
            delay_ms: 100.0,
            relation: AgenticDependencyRelation::Spawn,
            trigger: AgenticDependencyTrigger::Dispatch,
        }];
        rows.push(child.clone());
        child.request_id = "future-child".into();
        child.session_id = "future-child".into();
        child.not_before_ms = 21_000.0;
        child.dependencies[0].request_id = "n11".into();
        child.dependencies[0].delay_ms = 2_000.0;
        rows.push(child);

        let mut driver = rows_live_driver(rows, 1, 0.0, 1.0);
        let start = finish_ten_live_rounds(&mut driver, 1);
        let ready = driver.pop_ready(start, 1).pop().unwrap();
        assert_eq!(ready.authored_request_id.as_deref(), Some("child"));
        driver
            .on_terminal(
                ready.request_uuid,
                start + 1.0,
                ReplayTerminalStatus::Completed,
            )
            .unwrap();
        // The parent was already waiting at a join/barrier when warmup ended:
        // keep its recorded deadline even if the remaining child finishes early.
        let parent_at = start + 4_000.0;
        assert_eq!(driver.next_ready_time_ms(), Some(parent_at));
        let parent = driver.pop_ready(parent_at, 1).pop().unwrap();
        assert_eq!(parent.authored_request_id.as_deref(), Some("n10"));
        driver
            .on_terminal(
                parent.request_uuid,
                parent_at + 1.0,
                ReplayTerminalStatus::Completed,
            )
            .unwrap();
        let next_at = parent_at + 1.0 + 4_000.0;
        assert_eq!(driver.next_ready_time_ms(), Some(next_at));
        let next = driver.pop_ready(next_at, 1).pop().unwrap();
        assert_eq!(next.authored_request_id.as_deref(), Some("n11"));
        // A child whose spawning turn was still blocked at handoff is not a
        // pending handoff stream. Its timer starts only on actual dispatch.
        assert_eq!(driver.next_ready_time_ms(), Some(next_at + 2_000.0));
        let spawned = driver.pop_ready(next_at + 2_000.0, 1).pop().unwrap();
        assert_eq!(spawned.authored_request_id.as_deref(), Some("future-child"));
    }

    #[test]
    fn live_handoff_fresh_root_preserves_other_lanes_scaled_frontier_offset() {
        let mut rows = linear_live_rows(10, false);
        let mut other = linear_live_rows(12, false);
        for row in &mut other {
            row.play_id = "other".into();
            row.request_id = format!("other-{}", row.request_id);
            for edge in &mut row.dependencies {
                edge.request_id = format!("other-{}", edge.request_id);
            }
        }
        rows.extend(other);
        let mut driver = rows_live_driver(rows, 2, 0.0, 2.0);
        let start = finish_ten_live_rounds(&mut driver, 2);
        // One lane finished exactly at its quota. Its new incarnation anchors
        // the phase at zero, without also bursting the other lane's frontier.
        let root = driver.pop_ready(start, 2);
        assert_eq!(root.len(), 1);
        assert_eq!(root[0].authored_request_id.as_deref(), Some("n0"));
        assert_eq!(driver.next_ready_time_ms(), Some(start + 5.0));
        let other = driver.pop_ready(start + 5.0, 2);
        assert_eq!(other.len(), 1);
        assert_eq!(other[0].authored_request_id.as_deref(), Some("other-n10"));
    }

    #[test]
    fn zero_time_live_warmup_is_bounded_and_failed_reset_is_transactional() {
        let mut driver = linear_live_driver(1, false);
        driver.finish_agentic_preparation(0.0).unwrap();
        for _ in 0..10 {
            driver.advance_agentic_profile(0.0).unwrap();
            let ready = driver.pop_ready(0.0, 10);
            assert_eq!(ready.len(), 1);
            driver.on_output_token(ready[0].request_uuid, 0).unwrap();
            driver
                .on_terminal(ready[0].request_uuid, 0.0, ReplayTerminalStatus::Completed)
                .unwrap();
        }
        let snapshots = serde_json::to_value(driver.agentic_snapshot_evidence()).unwrap();
        let profile = serde_json::to_value(driver.agentic_profile_report()).unwrap();
        let lifecycle = driver.agentic_lifecycle_transcript();
        let phase = driver.agentic_phase_evidence().unwrap();
        assert!(
            driver
                .finish_agentic_preparation_with(0.0, || anyhow::bail!("reset failed"))
                .is_err()
        );
        assert_eq!(
            serde_json::to_value(driver.agentic_snapshot_evidence()).unwrap(),
            snapshots
        );
        assert_eq!(
            serde_json::to_value(driver.agentic_profile_report()).unwrap(),
            profile
        );
        assert_eq!(driver.agentic_lifecycle_transcript(), lifecycle);
        assert_eq!(driver.agentic_phase_evidence().unwrap(), phase);
        assert_eq!(
            driver.finish_agentic_preparation(0.0).unwrap(),
            Some(AgenticPreparationTransition::OpenProfile)
        );
        assert_eq!(driver.agentic_profile_report().unwrap().corpus_cursor, 11);
        let formal = driver.pop_ready(0.0, 10);
        assert_eq!(formal.len(), 1);
        assert_eq!(formal[0].request.max_output_tokens, 3);
        assert!(!driver.knows_preparation_request(formal[0].request_uuid));
    }

    #[test]
    fn rootless_live_frontier_advances_and_recycles_without_replaying_history() {
        let mut driver = linear_live_driver(3, true);
        driver.finish_agentic_preparation(0.0).unwrap();
        let mut ids = Vec::new();
        for iteration in 0..10 {
            let now = iteration as f64;
            driver.advance_agentic_profile(now).unwrap();
            let ready = driver.pop_ready(now, 1).pop().unwrap();
            ids.push(ready.authored_request_id.clone().unwrap());
            driver.on_output_token(ready.request_uuid, 0).unwrap();
            driver
                .on_terminal(
                    ready.request_uuid,
                    now + 1.0,
                    ReplayTerminalStatus::Completed,
                )
                .unwrap();
        }
        assert_eq!(
            ids,
            ["n1", "n2", "n0", "n1", "n2", "n0", "n1", "n2", "n0", "n1"]
        );
        assert_eq!(
            driver.finish_agentic_preparation(10.0).unwrap(),
            Some(AgenticPreparationTransition::OpenProfile)
        );
        let ready = driver.pop_ready(10.0, 1).pop().unwrap();
        assert_eq!(ready.authored_request_id.as_deref(), Some("n2"));
        assert_eq!(ready.request.max_output_tokens, 3);
    }

    #[test]
    fn aborted_live_warmup_has_no_formal_trajectory_cohort() {
        let mut driver = linear_live_driver(1, false);
        driver.finish_agentic_preparation(0.0).unwrap();
        for now in [0.0, 1.0] {
            driver.advance_agentic_profile(now).unwrap();
            let turn = driver.pop_ready(now, 1).pop().unwrap();
            driver.on_output_token(turn.request_uuid, 0).unwrap();
            driver
                .on_terminal(
                    turn.request_uuid,
                    now + 1.0,
                    ReplayTerminalStatus::Completed,
                )
                .unwrap();
        }
        driver.advance_agentic_profile(2.0).unwrap();
        let turn = driver.pop_ready(2.0, 1).pop().unwrap();
        driver
            .on_terminal(turn.request_uuid, 3.0, ReplayTerminalStatus::Failed)
            .unwrap();
        assert_eq!(
            driver.finish_agentic_preparation(3.0).unwrap(),
            Some(AgenticPreparationTransition::Aborted)
        );
        assert_eq!(
            driver
                .agentic_trajectory_snapshot()
                .unwrap()
                .total_trajectories,
            0
        );
        assert!(driver.agentic_play_outcomes().unwrap().is_empty());
        assert_eq!(driver.agentic_profile_report().unwrap().plays_started, 0);
        assert_eq!(driver.agentic_phase_evidence().unwrap().requests.len(), 3);
    }

    #[test]
    fn full_snapshot_prefixes_repeat_ten_times_without_advancing_the_saved_frontier() {
        let prepared = prepared(2, 50.0);
        let mut cold =
            WorkloadDriver::new_agentic_snapshots(prepared.clone(), 48, true, 2.0).unwrap();
        let mut driver =
            WorkloadDriver::new_agentic_warmup(prepared.clone(), 48, true, 2.0).unwrap();
        let saved = driver.agentic_snapshot_evidence().unwrap().to_vec();
        let mut uuids = FxHashSet::default();
        for ordinal in 0..12 {
            let now = ordinal as f64;
            assert_eq!(driver.next_ready_time_ms(), Some(now));
            let ready = driver.pop_ready_compact(now, 100);
            assert_eq!(ready.len(), 2);
            for (lane, ready) in ready.into_iter().enumerate() {
                assert!(ready.request.metadata().tokens.is_empty());
                assert!(ready.request.materialized_tokens().is_none());
                assert!(ready.replay_hashes.is_some());
                let ready = ready.into_ready_turn();
                assert!(uuids.insert(ready.request_uuid));
                assert!(ready.request_uuid.as_u128() >= PREPARATION_UUID_BASE);
                let source = if ordinal % 2 == 0 {
                    "b-history"
                } else {
                    "a-history"
                };
                // Primers are in snapshot conversation order (child, main),
                // and the ten subsequent requests cycle that exact list.
                let input_length = if source == "a-history" { 128 } else { 191 };
                assert_eq!(ready.authored_request_id.as_deref(), Some(source));
                assert_eq!(
                    ready.request.tokens,
                    prepared.plays[lane]
                        .materialize_prefix(source, input_length)
                        .unwrap()
                );
                assert_eq!(ready.request.max_output_tokens, 1);
                assert_eq!(ready.request.output_token_ids.as_ref().unwrap().len(), 1);
                let identity = ready.request.replay_context.unwrap().agentic.unwrap();
                let original = prepared.plays[lane].identity(source).unwrap();
                assert_ne!(identity.request_id, original.request_id);
                assert_eq!(identity.cache_id, original.cache_id);
                assert_eq!(identity.conversation_id, original.conversation_id);
                driver.on_output_token(ready.request_uuid, 0).unwrap();
                driver.on_complete(ready.request_uuid, now + 1.0).unwrap();
            }
            assert_eq!(driver.agentic_snapshot_evidence().unwrap(), saved);
            assert!(
                driver
                    .agentic_lifecycle_transcript()
                    .unwrap()
                    .events
                    .is_empty()
            );
            if ordinal < 11 {
                assert_eq!(driver.finish_agentic_preparation(now + 1.0).unwrap(), None);
            }
        }
        assert_eq!(
            driver.finish_agentic_preparation(12.0).unwrap(),
            Some(AgenticPreparationTransition::OpenProfile)
        );
        assert_eq!(driver.finish_agentic_preparation(12.0).unwrap(), None);
        assert_eq!(
            driver.next_ready_time_ms(),
            Some(cold.next_ready_time_ms().unwrap() + 12.0)
        );
        let profile = driver.pop_ready(37.0, 100);
        let expected = cold.pop_ready(25.0, 100);
        assert_eq!(profile.len(), expected.len());
        for (actual, expected) in profile.iter().zip(&expected) {
            assert_eq!(actual.request_uuid, expected.request_uuid);
            assert_eq!(actual.request.tokens, expected.request.tokens);
            assert_eq!(
                actual.request.output_token_ids,
                expected.request.output_token_ids
            );
            assert_eq!(
                actual.scheduled_ready_at_ms,
                expected.scheduled_ready_at_ms + 12.0
            );
            assert_eq!(actual.request.max_output_tokens, 3);
        }
        let evidence = driver.agentic_phase_evidence().unwrap();
        assert_eq!(evidence.phase, AgenticReplayPhase::Profile);
        assert_eq!(evidence.profile_start_ms, Some(12.0));
        assert!(evidence.lanes.iter().all(|lane| lane.primers_completed == 2
            && lane.warmup_completed == 10
            && lane.requests_quiescent == 12));
        assert!(evidence.requests.iter().all(|request| request.expected_full_block_tokens == request.input_length / 48 * 48));
    }

    #[test]
    fn causal_completion_releases_zero_idle_work_but_barrier_waits_for_all_quiescence() {
        let mut driver =
            WorkloadDriver::new_agentic_warmup(prepared(1, 0.0), 64, true, 1.0).unwrap();
        let mut outstanding = Vec::new();
        for ordinal in 0..10 {
            let ready = driver.pop_ready(ordinal as f64, 1).pop().unwrap();
            assert_eq!(ready.authored_request_id.as_deref(), Some("a-history"));
            assert_eq!(ready.request.tokens.len(), 128);
            driver.on_output_token(ready.request_uuid, 0).unwrap();
            driver
                .on_causal_terminal(
                    ready.request_uuid,
                    ordinal as f64 + 1.0,
                    ReplayTerminalStatus::Completed,
                )
                .unwrap();
            outstanding.push(ready.request_uuid);
        }
        assert!(driver.pop_ready(1000.0, 100).is_empty());
        assert_eq!(driver.finish_agentic_preparation(10.0).unwrap(), None);
        for uuid in &outstanding[..9] {
            driver.on_quiescent(*uuid, 10.0).unwrap();
        }
        assert_eq!(driver.finish_agentic_preparation(10.0).unwrap(), None);
        driver.on_quiescent(outstanding[9], 15.0).unwrap();
        assert_eq!(
            driver.finish_agentic_preparation(15.0).unwrap(),
            Some(AgenticPreparationTransition::OpenProfile)
        );
        assert_eq!(driver.pop_ready(15.0, 10).len(), 1);
        let evidence = driver.agentic_phase_evidence().unwrap();
        assert_eq!(evidence.lanes[0].primers_expected, 0);
        assert_eq!(evidence.lanes[0].warmup_completed, 10);
    }

    #[test]
    fn preparation_failure_drains_issued_work_and_never_opens_profile() {
        for status in [
            ReplayTerminalStatus::Rejected,
            ReplayTerminalStatus::Failed,
            ReplayTerminalStatus::Canceled,
        ] {
            let mut driver =
                WorkloadDriver::new_agentic_warmup(prepared(2, 50.0), 64, true, 1.0).unwrap();
            let ready = driver.pop_ready(0.0, 2);
            driver
                .on_causal_terminal(ready[0].request_uuid, 1.0, status)
                .unwrap();
            assert!(driver.pop_ready(1.0, 10).is_empty());
            assert_eq!(driver.finish_agentic_preparation(1.0).unwrap(), None);
            driver.on_quiescent(ready[0].request_uuid, 2.0).unwrap();
            driver.on_output_token(ready[1].request_uuid, 0).unwrap();
            driver.on_complete(ready[1].request_uuid, 3.0).unwrap();
            assert_eq!(
                driver.finish_agentic_preparation(3.0).unwrap(),
                Some(AgenticPreparationTransition::Aborted)
            );
            assert!(driver.is_drained());
            assert_eq!(driver.next_ready_time_ms(), None);
            assert!(driver.pop_ready(1000.0, 10).is_empty());
            let evidence = driver.agentic_phase_evidence().unwrap();
            assert_eq!(evidence.phase, AgenticReplayPhase::Aborted);
            assert_eq!(evidence.failure_request_id, Some(ready[0].request_uuid));
            assert_eq!(evidence.profile_start_ms, None);
            assert_eq!(evidence.finished_at_ms, Some(3.0));
            assert!(
                driver
                    .agentic_lifecycle_transcript()
                    .unwrap()
                    .events
                    .is_empty()
            );
        }
    }

    #[test]
    fn preparation_feedback_validation_is_atomic_and_preserves_the_first_admission() {
        let mut driver =
            WorkloadDriver::new_agentic_warmup(prepared(2, 50.0), 64, true, 1.0).unwrap();
        let ready = driver.pop_ready(0.0, 2);
        let first = ready[0].request_uuid;
        let second = ready[1].request_uuid;
        driver.record_preparation_admission(first, 0.5, 64).unwrap();
        driver.record_preparation_admission(first, 1.0, 0).unwrap();
        let initial = driver.agentic_phase_evidence().unwrap();
        for bad in [
            AgenticRuntimeFeedback {
                at_ms: 1.0,
                causal_terminals: vec![
                    AgenticTerminalFeedback {
                        request_uuid: first,
                        status: ReplayTerminalStatus::Completed,
                    },
                    AgenticTerminalFeedback {
                        request_uuid: first,
                        status: ReplayTerminalStatus::Completed,
                    },
                ],
                ..Default::default()
            },
            AgenticRuntimeFeedback {
                at_ms: 1.0,
                causal_terminals: vec![AgenticTerminalFeedback {
                    request_uuid: first,
                    status: ReplayTerminalStatus::Completed,
                }],
                quiescent_requests: vec![second],
                ..Default::default()
            },
            AgenticRuntimeFeedback {
                at_ms: 1.0,
                causal_terminals: vec![
                    AgenticTerminalFeedback {
                        request_uuid: first,
                        status: ReplayTerminalStatus::Completed,
                    },
                    AgenticTerminalFeedback {
                        request_uuid: Uuid::nil(),
                        status: ReplayTerminalStatus::Completed,
                    },
                ],
                ..Default::default()
            },
            AgenticRuntimeFeedback {
                at_ms: 1.0,
                output_tokens: vec![AgenticOutputFeedback {
                    request_uuid: first,
                    token_ids: vec![1, 2],
                }],
                ..Default::default()
            },
        ] {
            assert!(driver.apply_agentic_runtime_feedback(bad).is_err());
            assert_eq!(driver.agentic_phase_evidence().unwrap(), initial);
        }
        driver
            .apply_agentic_runtime_feedback(AgenticRuntimeFeedback {
                at_ms: 2.0,
                output_tokens: vec![AgenticOutputFeedback {
                    request_uuid: first,
                    token_ids: vec![0],
                }],
                causal_terminals: vec![AgenticTerminalFeedback {
                    request_uuid: first,
                    status: ReplayTerminalStatus::Completed,
                }],
                quiescent_requests: vec![first],
                ..Default::default()
            })
            .unwrap();
        let after = driver.agentic_phase_evidence().unwrap();
        assert_eq!(after.requests[0].first_admit_ms, Some(0.5));
        assert_eq!(
            after.requests[0].first_admission_reused_input_tokens,
            Some(64)
        );
        assert!(
            driver
                .apply_agentic_runtime_feedback(AgenticRuntimeFeedback {
                    at_ms: 1.0,
                    causal_terminals: vec![AgenticTerminalFeedback {
                        request_uuid: second,
                        status: ReplayTerminalStatus::Completed
                    }],
                    ..Default::default()
                })
                .is_err()
        );
        assert_eq!(driver.agentic_phase_evidence().unwrap(), after);
    }

    #[test]
    fn zero_duration_prefill_only_source_warms_one_token_and_preserves_its_profile_output() {
        let graph = ValidatedAgenticGraph::from_agentic_mooncake_rows(
            AgenticMooncakeHeader {
                schema: AGENTIC_MOONCAKE_SCHEMA.into(),
                version: AGENTIC_MOONCAKE_VERSION,
                block_size: 64,
                hash_id_scope: AgenticHashIdScope::Local,
                source: AgenticSourceProvenance {
                    format: "self-authored".into(),
                    digest: "phase-prefill-only-v1".into(),
                },
            },
            vec![AgenticMooncakeRow {
                request_id: "prefill-only".into(),
                play_id: "play".into(),
                session_id: "conversation".into(),
                model: "model".into(),
                input_length: Some(126),
                output_length: Some(0),
                hash_ids: Some(vec![10, 20]),
                not_before_ms: 500.0,
                ..Default::default()
            }],
        )
        .unwrap();
        let snapshots = graph
            .prepare_snapshots(1, AgenticSnapshotOptions::seeded(17))
            .unwrap();
        let original = snapshots.plays[0]
            .materialize_prefix("prefill-only", 126)
            .unwrap();
        let mut driver = WorkloadDriver::new_agentic_warmup(snapshots, 64, true, 3.0).unwrap();
        let mut preparation_ids = Vec::new();
        for ordinal in 0..10 {
            let ready = driver.pop_ready(ordinal as f64, 1).pop().unwrap();
            assert_eq!(ready.request.tokens, original);
            assert_eq!(ready.request.max_output_tokens, 1);
            assert_eq!(ready.request.output_token_ids, Some(vec![0]));
            driver.on_output_token(ready.request_uuid, 0).unwrap();
            driver
                .on_complete(ready.request_uuid, ordinal as f64 + 1.0)
                .unwrap();
            preparation_ids.push(ready.request_uuid);
        }
        assert_eq!(
            driver.finish_agentic_preparation(10.0).unwrap(),
            Some(AgenticPreparationTransition::OpenProfile)
        );
        let before_late_event = driver.agentic_phase_evidence().unwrap();
        assert_eq!(before_late_event.lanes[0].primers_expected, 0);
        assert!(
            before_late_event
                .requests
                .iter()
                .all(|r| r.phase == AgenticReplayPhase::Warmup && r.observed_output_tokens == 1)
        );
        // Already-settled preparation work cannot become profiling feedback.
        assert!(driver.on_complete(preparation_ids[0], 11.0).is_err());
        assert!(driver.on_output_token(preparation_ids[0], 0).is_err());
        assert_eq!(driver.agentic_phase_evidence().unwrap(), before_late_event);
        let profile = driver.pop_ready(10.0, 1).pop().unwrap();
        assert_eq!(profile.request.tokens, original);
        assert_eq!(profile.request.max_output_tokens, 0);
        assert_eq!(profile.request.output_token_ids, Some(vec![]));
    }

    #[test]
    fn preparation_rejects_invalid_blocks_and_reuse_without_consuming_saved_state() {
        let prepared = prepared(1, 50.0);
        assert!(WorkloadDriver::new_agentic_warmup(prepared.clone(), 0, true, 1.0).is_err());
        if usize::BITS > 32 {
            assert!(
                WorkloadDriver::new_agentic_warmup(
                    prepared.clone(),
                    u32::MAX as usize + 1,
                    true,
                    1.0
                )
                .is_err()
            );
        }
        let mut driver = WorkloadDriver::new_agentic_warmup(prepared, 64, true, 1.0).unwrap();
        let ready = driver.pop_ready(2.0, 1).pop().unwrap();
        let evidence = driver.agentic_phase_evidence().unwrap();
        assert!(
            driver
                .record_preparation_admission(ready.request_uuid, 1.0, 0)
                .is_err()
        );
        assert!(
            driver
                .record_preparation_admission(ready.request_uuid, f64::NAN, 0)
                .is_err()
        );
        assert!(
            driver
                .record_preparation_admission(
                    ready.request_uuid,
                    2.0,
                    ready.request.tokens.len() + 1
                )
                .is_err()
        );
        assert!(
            driver
                .record_preparation_admission(Uuid::nil(), 2.0, 0)
                .is_err()
        );
        assert_eq!(driver.agentic_phase_evidence().unwrap(), evidence);
    }

    #[test]
    fn successful_terminal_without_the_required_output_invalidates_preparation_after_drain() {
        let mut driver =
            WorkloadDriver::new_agentic_warmup(prepared(1, 0.0), 64, true, 1.0).unwrap();
        let ready = driver.pop_ready(0.0, 1).pop().unwrap();
        driver
            .on_causal_terminal(ready.request_uuid, 1.0, ReplayTerminalStatus::Completed)
            .unwrap();
        assert_eq!(driver.finish_agentic_preparation(1.0).unwrap(), None);
        assert!(driver.pop_ready(1.0, 1).is_empty());
        driver.on_quiescent(ready.request_uuid, 2.0).unwrap();
        assert_eq!(
            driver.finish_agentic_preparation(2.0).unwrap(),
            Some(AgenticPreparationTransition::Aborted)
        );
        let evidence = driver.agentic_phase_evidence().unwrap();
        assert_eq!(
            evidence.requests[0].terminal_status,
            Some(ReplayTerminalStatus::Completed)
        );
        assert_eq!(evidence.lanes[0].warmup_completed, 0);
        assert!(
            evidence
                .failure_reason
                .as_deref()
                .unwrap()
                .contains("exactly one output token")
        );
    }

    #[test]
    fn overflowing_profile_activation_does_not_commit_the_phase_transition() {
        let mut driver =
            WorkloadDriver::new_agentic_warmup(prepared(1, 50.0), 64, true, 1e-306).unwrap();
        for ordinal in 0..12 {
            let at_ms = if ordinal == 0 { 0.0 } else { 1e308 };
            let request = driver.pop_ready(at_ms, 1).pop().unwrap();
            driver.on_output_token(request.request_uuid, 0).unwrap();
            driver.on_complete(request.request_uuid, 1e308).unwrap();
        }
        let preparation = driver.agentic_phase_evidence().unwrap();
        let saved = driver.agentic_snapshot_evidence().unwrap().to_vec();
        assert!(
            driver
                .finish_agentic_preparation(1e308)
                .unwrap_err()
                .to_string()
                .contains("overflows saved snapshot timing")
        );
        assert_eq!(driver.agentic_phase_evidence().unwrap(), preparation);
        assert_eq!(driver.agentic_snapshot_evidence().unwrap(), saved);
        assert!(
            driver
                .agentic_lifecycle_transcript()
                .unwrap()
                .events
                .is_empty()
        );
        assert!(driver.is_agentic_preparing());
    }
}
