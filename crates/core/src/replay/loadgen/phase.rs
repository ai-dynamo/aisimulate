// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Preparation traffic is separate from the saved graph frontier. Warmup repeats
//! full snapshot prefixes; it never advances a source request or its output RNG.
//!
//! Methodology reference for one-token primers and ten warmups per lane:
//! https://github.com/SemiAnalysisAI/InferenceX-app/blob/9bb7b13eb4985217a6282f340459fd5948613276/packages/app/src/components/datasets/agentx-methodology-article.tsx
//! This module implements those requirements using AISimulate's prepared plays
//! and native completion feedback. Repeating saved prefixes without advancing
//! the frontier is an AISimulate policy. See docs/agentic-warmup.md for the
//! reference scope, upstream licenses, and separate AIPerf snapshot comparison.

use anyhow::{Context, Result, ensure};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;
use uuid::Uuid;

use super::driver::{AgenticRuntimeFeedback, deferred_request_with_hashes};
use super::{AgenticPlaySnapshot, CompactReadyTurn, PreparedAgenticSnapshots};
use crate::replay::protocol::{DirectRequest, ReplayPromptTokenSource, ReplayRequestContext};
use crate::replay::{AgenticRuntimeIdentity, ReplayTerminalStatus};

pub const AGENTIC_PHASE_SCHEMA_V1: &str = "aisimulate.agentic.phases.v1";
pub const AGENTIC_WARMUP_REQUESTS_PER_LANE: usize = 10;
const PREPARATION_UUID_BASE: u128 = 1_u128 << 127;

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
}

#[derive(Debug)]
struct PreparationLane {
    snapshot: AgenticPlaySnapshot,
    source_indices: Vec<usize>,
    request_indices: Vec<usize>,
    next: usize,
    in_flight: Option<Uuid>,
    ready_at_ms: f64,
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
}

impl AgenticPreparation {
    pub(super) fn new(
        prepared: &PreparedAgenticSnapshots,
        engine_block_size: u32,
        include_replay_hashes: bool,
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
            source_indices.extend(
                (0..AGENTIC_WARMUP_REQUESTS_PER_LANE)
                    .map(|ordinal| warmup_sources[ordinal % warmup_sources.len()]),
            );
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
                super::trace::validate_synthesizable_prompt(
                    node.input_length,
                    &snapshot.token_ids(source_index)?,
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
            });
        }
        Ok(this)
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
        if self.failure_request_id.is_some() || self.transition.is_some() {
            return None;
        }
        self.lanes
            .iter()
            .filter(|lane| lane.in_flight.is_none() && lane.next < lane.request_indices.len())
            .map(|lane| lane.ready_at_ms)
            .min_by(f64::total_cmp)
    }

    pub(super) fn pop_ready(
        &mut self,
        now_ms: f64,
        limit: usize,
        emit_session_metadata: bool,
    ) -> Vec<CompactReadyTurn> {
        if self.failure_request_id.is_some() || self.transition.is_some() {
            return Vec::new();
        }
        let mut ready = Vec::new();
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
            let ordinal = lane.next;
            let request_index = lane.request_indices[ordinal];
            let source_index = lane.source_indices[ordinal];
            let evidence = &mut self.requests[request_index];
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
                lane.snapshot
                    .token_ids(source_index)
                    .expect("validated preparation token identities"),
                lane.snapshot.context.graph.block_size,
                self.include_replay_hashes.then_some(self.engine_block_size),
            );
            evidence.dispatched_at_ms = Some(now_ms);
            lane.in_flight = Some(evidence.uuid);
            lane.next += 1;
            ready.push(CompactReadyTurn {
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
                request,
            });
        }
        ready
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
        if self
            .lanes
            .iter()
            .all(|lane| lane.next == lane.request_indices.len())
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
                        .map(|&i| &self.requests[i])
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
            .prepare_snapshots(lanes, AgenticSnapshotOptions { seed: 42 })
            .unwrap();
        PreparedAgenticSnapshots::from_plays(
            (0..lanes)
                .map(|lane| sampled.context().prepare_play(lane, 0, Some(cut)).unwrap())
                .collect(),
        )
        .unwrap()
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
            .prepare_snapshots(1, AgenticSnapshotOptions { seed: 17 })
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
