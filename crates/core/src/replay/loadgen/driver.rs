// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;
use uuid::Uuid;

use super::trace::{synthesize_validated_trace_tokens, validate_synthesizable_prompt};
use super::types::{
    AgenticDependencyRelation, AgenticDependencyTrigger, AgenticGraphIdentity, AgenticPlayOutcome,
    AgenticPlayStatus, AgenticTrace, AgenticTrajectorySnapshot, CompactReadyTurn, ReadyTurn,
    ReplayRequestHashes, ReplayRequestPayload, Trace,
};
use super::{
    AgenticPlay, AgenticReplayContext, AgenticSnapshotEvidence, PreparedAgenticSnapshots,
    SYNTHETIC_OUTPUT_SEED, planned_output_token_ids,
};
use crate::engine::belady::{SequenceHash, input_sequence_hashes};
use crate::replay::ReplayTerminalStatus;
use crate::replay::protocol::{
    AgenticRuntimeIdentity, DirectRequest, ReplayPromptTokenSource, ReplayRequestContext,
};

pub const AGENTIC_LIFECYCLE_SCHEMA_V1: &str = "aisimulate.agentic.lifecycle.v1";
const AGENTIC_LIFECYCLE_DIGEST_DOMAIN_V1: &[u8] = b"aisimulate-agentic-lifecycle-v1\0";

/// Output progress for one request inside a same-timestamp feedback batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgenticOutputFeedback {
    pub request_uuid: Uuid,
    pub token_ids: Vec<u32>,
}

/// Causal terminal state for one request inside a same-timestamp batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgenticTerminalFeedback {
    pub request_uuid: Uuid,
    pub status: ReplayTerminalStatus,
}

/// Agentic workload feedback produced by the runtime at one logical timestamp.
///
/// The runtime owns `at_ms`. The driver canonicalizes request order by graph
/// ordinal and applies output progress, causal terminals, then quiescence.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgenticRuntimeFeedback {
    pub at_ms: f64,
    pub output_tokens: Vec<AgenticOutputFeedback>,
    pub causal_terminals: Vec<AgenticTerminalFeedback>,
    pub quiescent_requests: Vec<Uuid>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgenticLifecycleEventKind {
    Dispatch,
    CausalTerminal,
    Quiescent,
    Skipped,
    PlayQuiescent,
}

/// Canonical workload-level lifecycle record produced by the agentic driver.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AgenticLifecycleEvent {
    pub schema: &'static str,
    pub ordinal: u64,
    pub at_ms: f64,
    pub play_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    pub event: AgenticLifecycleEventKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<ReplayTerminalStatus>,
}

/// Byte-stable lifecycle evidence for one preloaded graph execution.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgenticLifecycleTranscript {
    pub events: Vec<AgenticLifecycleEvent>,
}

impl AgenticLifecycleTranscript {
    pub fn to_jsonl(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        for event in &self.events {
            serde_json::to_writer(&mut bytes, event)
                .context("serializing agentic lifecycle event")?;
            bytes.push(b'\n');
        }
        Ok(bytes)
    }

    pub fn digest(&self) -> Result<String> {
        let bytes = self.to_jsonl()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(AGENTIC_LIFECYCLE_DIGEST_DOMAIN_V1);
        hasher.update(&bytes);
        Ok(hasher.finalize().to_hex().to_string())
    }
}

#[derive(Debug, Clone, Copy)]
struct AgenticLifecycleRecord {
    ordinal: u64,
    at_ms: f64,
    play_index: usize,
    node_index: Option<usize>,
    event: AgenticLifecycleEventKind,
    status: Option<ReplayTerminalStatus>,
}

#[derive(Debug)]
enum SchedulingPolicy {
    Trace,
    Concurrency(ConcurrencyState),
    Agentic(Box<AgenticState>),
}

#[derive(Debug)]
struct ConcurrencyState {
    max_active_sessions: usize,
    next_pending_session: usize,
    active_sessions: usize,
}

#[derive(Debug)]
struct AgenticState {
    identities: Vec<AgenticRuntimeIdentity>,
    node_states: Vec<AgenticNodeState>,
    remaining_dependencies: Vec<usize>,
    authored_not_before_ms: Vec<f64>,
    ready_after_ms: Vec<f64>,
    dispatch_dependents: Vec<Vec<AgenticDependentEdge>>,
    completion_dependents: Vec<Vec<AgenticDependentEdge>>,
    node_to_play: Vec<usize>,
    plays: Vec<AgenticPlayState>,
    lanes: Vec<AgenticLaneState>,
    lifecycle: Vec<AgenticLifecycleRecord>,
    next_lifecycle_ordinal: u64,
    last_runtime_feedback_at_ms: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AgenticNodeState {
    Blocked,
    Ready,
    Emitted,
    Completed,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Copy)]
struct AgenticDependentEdge {
    target_node: usize,
    delay_ms: f64,
}

#[derive(Debug)]
struct AgenticPlayState {
    play_id: String,
    nodes: Vec<usize>,
    root_nodes: Vec<usize>,
    lane_index: Option<usize>,
    pending_terminals: usize,
    emitted_in_flight: usize,
    completed_nodes: usize,
    failed: bool,
    client_finished: bool,
    quiescent: bool,
    root_dispatch_ms: Option<f64>,
    max_terminal_ms: Option<f64>,
    quiescent_at_ms: Option<f64>,
    primary_failure: Option<AgenticFailureRecord>,
}

#[derive(Debug, Clone, Copy)]
struct AgenticFailureRecord {
    node_index: usize,
    at_ms: f64,
    status: ReplayTerminalStatus,
}

#[derive(Debug)]
struct AgenticLaneState {
    plays: Vec<usize>,
    next_play: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptMode {
    Full,
    DeltaCumulative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnOutcome {
    Completed,
    Rejected,
    Cancelled,
    Failed,
}

#[derive(Debug)]
struct TurnResolution {
    session_index: usize,
    outcome: TurnOutcome,
    session_ended: bool,
}

#[derive(Debug)]
struct SessionRuntime {
    session_id: String,
    turns: Vec<TurnRuntime>,
    cumulative_tokens: Vec<u32>,
    next_turn_index: usize,
    next_ready_at_ms: Option<f64>,
    in_flight: Option<Uuid>,
}

#[derive(Debug)]
enum PromptTokens {
    // Full-prompt traces stay in their compact on-disk representation until
    // dispatch. Delta-cumulative traces remain eager because later turns append
    // generated output to already-materialized session history.
    Deferred {
        input_length: usize,
        hash_ids: Vec<u32>,
    },
    Materialized(Vec<u32>),
}

impl PromptTokens {
    fn deferred(input_length: usize, hash_ids: Vec<u32>, trace_block_size: usize) -> Result<Self> {
        validate_synthesizable_prompt(input_length, &hash_ids, trace_block_size)?;
        Ok(Self::Deferred {
            input_length,
            hash_ids,
        })
    }

    fn input_length(&self) -> usize {
        match self {
            Self::Deferred { input_length, .. } => *input_length,
            Self::Materialized(tokens) => tokens.len(),
        }
    }

    fn take_deferred(&mut self) -> (usize, Vec<u32>) {
        match self {
            Self::Deferred {
                input_length,
                hash_ids,
            } => (*input_length, std::mem::take(hash_ids)),
            Self::Materialized(_) => {
                unreachable!("full-prompt turns must retain their deferred representation")
            }
        }
    }

    fn materialized(&self) -> &[u32] {
        match self {
            Self::Deferred { .. } => {
                unreachable!("delta-cumulative prompts are materialized during driver setup")
            }
            Self::Materialized(tokens) => tokens,
        }
    }
}

#[derive(Debug)]
struct TurnRuntime {
    request_id: Option<String>,
    play_id: Option<String>,
    replay_key: Option<String>,
    prompt_tokens: PromptTokens,
    max_output_tokens: usize,
    output_token_ids: Option<Vec<u32>>,
    delay_after_previous_ms: f64,
    priority: i32,
    strict_priority: u32,
    policy_class: Option<String>,
    // Canonical capture assigns ordinals; Belady may instead reserve an opaque
    // UUID here so the forecast and eventual causal admission share an identity.
    deterministic_request_id: Option<Uuid>,
}

#[derive(Debug, Clone, Copy)]
struct InFlightTurn {
    session_index: usize,
    turn_index: usize,
    emitted_output_tokens: usize,
}

#[derive(Debug, Clone, Copy)]
struct ReadySession {
    ready_at_ms: f64,
    session_index: usize,
    turn_index: usize,
}

impl PartialEq for ReadySession {
    fn eq(&self, other: &Self) -> bool {
        self.ready_at_ms.to_bits() == other.ready_at_ms.to_bits()
            && self.session_index == other.session_index
            && self.turn_index == other.turn_index
    }
}

impl Eq for ReadySession {}

impl Ord for ReadySession {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .ready_at_ms
            .total_cmp(&self.ready_at_ms)
            .then_with(|| other.session_index.cmp(&self.session_index))
            .then_with(|| other.turn_index.cmp(&self.turn_index))
    }
}

impl PartialOrd for ReadySession {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl SchedulingPolicy {
    fn schedules_sequential_turns(&self) -> bool {
        !matches!(self, Self::Agentic(_))
    }

    fn arrival_timestamp_ms(&self, scheduled_ready_at_ms: f64) -> Option<f64> {
        match self {
            Self::Concurrency(_) => None,
            Self::Trace | Self::Agentic(_) => Some(scheduled_ready_at_ms),
        }
    }

    fn dispatch_limit(&self, requested: usize, in_flight: usize) -> usize {
        match self {
            Self::Concurrency(state) => {
                requested.min(state.max_active_sessions.saturating_sub(in_flight))
            }
            Self::Trace | Self::Agentic(_) => requested,
        }
    }

    fn at_dispatch_capacity(&self, in_flight: usize) -> bool {
        matches!(
            self,
            Self::Concurrency(state) if in_flight >= state.max_active_sessions
        )
    }
}

impl ConcurrencyState {
    fn new(max_active_sessions: usize) -> Self {
        Self {
            max_active_sessions,
            next_pending_session: 0,
            active_sessions: 0,
        }
    }

    fn activate_pending(
        &mut self,
        sessions: &mut [SessionRuntime],
        ready_sessions: &mut BinaryHeap<ReadySession>,
        now_ms: f64,
    ) {
        while self.active_sessions < self.max_active_sessions
            && self.next_pending_session < sessions.len()
        {
            let session_index = self.next_pending_session;
            self.next_pending_session += 1;
            let session = &mut sessions[session_index];
            let turn_index = session.next_turn_index;
            session.next_ready_at_ms = Some(now_ms);
            ready_sessions.push(ReadySession {
                ready_at_ms: now_ms,
                session_index,
                turn_index,
            });
            self.active_sessions += 1;
        }
    }

    fn on_session_finished(
        &mut self,
        sessions: &mut [SessionRuntime],
        ready_sessions: &mut BinaryHeap<ReadySession>,
        now_ms: f64,
    ) {
        self.active_sessions = self.active_sessions.saturating_sub(1);
        self.activate_pending(sessions, ready_sessions, now_ms);
    }
}

impl AgenticState {
    fn record_lifecycle(
        &mut self,
        at_ms: f64,
        play_index: usize,
        node_index: Option<usize>,
        event: AgenticLifecycleEventKind,
        status: Option<ReplayTerminalStatus>,
    ) {
        let ordinal = self.next_lifecycle_ordinal;
        self.next_lifecycle_ordinal = self
            .next_lifecycle_ordinal
            .checked_add(1)
            .expect("agentic lifecycle ordinal overflow");
        self.lifecycle.push(AgenticLifecycleRecord {
            ordinal,
            at_ms,
            play_index,
            node_index,
            event,
            status,
        });
    }

    fn activate_play(
        &mut self,
        play_index: usize,
        start_ms: f64,
        sessions: &mut [SessionRuntime],
        ready_sessions: &mut BinaryHeap<ReadySession>,
    ) {
        let play = &self.plays[play_index];
        let root_not_before_ms = play
            .root_nodes
            .iter()
            .map(|root| self.authored_not_before_ms[*root])
            .min_by(f64::total_cmp)
            .expect("validated agentic play has a root");
        for &node_index in &play.nodes {
            self.ready_after_ms[node_index] =
                start_ms + (self.authored_not_before_ms[node_index] - root_not_before_ms).max(0.0);
        }
        for &root_node in &play.root_nodes {
            Self::schedule_node(
                root_node,
                self.ready_after_ms[root_node],
                &mut self.node_states,
                sessions,
                ready_sessions,
            );
        }
    }

    fn release_dispatch_dependents(
        &mut self,
        sessions: &mut [SessionRuntime],
        ready_sessions: &mut BinaryHeap<ReadySession>,
        source_node: usize,
        now_ms: f64,
    ) {
        Self::release_edges(
            &self.dispatch_dependents[source_node],
            &mut self.node_states,
            &mut self.remaining_dependencies,
            &mut self.ready_after_ms,
            sessions,
            ready_sessions,
            now_ms,
        );
    }

    fn release_completion_dependents(
        &mut self,
        sessions: &mut [SessionRuntime],
        ready_sessions: &mut BinaryHeap<ReadySession>,
        source_node: usize,
        now_ms: f64,
    ) {
        Self::release_edges(
            &self.completion_dependents[source_node],
            &mut self.node_states,
            &mut self.remaining_dependencies,
            &mut self.ready_after_ms,
            sessions,
            ready_sessions,
            now_ms,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn release_edges(
        edges: &[AgenticDependentEdge],
        node_states: &mut [AgenticNodeState],
        remaining_dependencies: &mut [usize],
        ready_after_ms: &mut [f64],
        sessions: &mut [SessionRuntime],
        ready_sessions: &mut BinaryHeap<ReadySession>,
        now_ms: f64,
    ) {
        for edge in edges {
            let target = edge.target_node;
            if node_states[target] != AgenticNodeState::Blocked {
                continue;
            }
            remaining_dependencies[target] = remaining_dependencies[target]
                .checked_sub(1)
                .expect("compiled agentic dependency must fire exactly once");
            ready_after_ms[target] = ready_after_ms[target].max(now_ms + edge.delay_ms);
            if remaining_dependencies[target] != 0 {
                continue;
            }
            Self::schedule_node(
                target,
                ready_after_ms[target],
                node_states,
                sessions,
                ready_sessions,
            );
        }
    }

    fn schedule_node(
        node_index: usize,
        ready_at_ms: f64,
        node_states: &mut [AgenticNodeState],
        sessions: &mut [SessionRuntime],
        ready_sessions: &mut BinaryHeap<ReadySession>,
    ) {
        if node_states[node_index] != AgenticNodeState::Blocked {
            return;
        }
        let session = &mut sessions[node_index];
        if session.in_flight.is_some() || session.next_turn_index >= session.turns.len() {
            return;
        }
        node_states[node_index] = AgenticNodeState::Ready;
        session.next_ready_at_ms = Some(ready_at_ms);
        ready_sessions.push(ReadySession {
            ready_at_ms,
            session_index: node_index,
            turn_index: 0,
        });
    }

    fn on_node_emitted(&mut self, node_index: usize, now_ms: f64) {
        debug_assert_eq!(self.node_states[node_index], AgenticNodeState::Ready);
        self.node_states[node_index] = AgenticNodeState::Emitted;
        let play = &mut self.plays[self.node_to_play[node_index]];
        play.pending_terminals += 1;
        play.emitted_in_flight += 1;
        if play.root_nodes.contains(&node_index) {
            play.root_dispatch_ms = Some(
                play.root_dispatch_ms
                    .map_or(now_ms, |seen| seen.min(now_ms)),
            );
        }
        self.record_lifecycle(
            now_ms,
            self.node_to_play[node_index],
            Some(node_index),
            AgenticLifecycleEventKind::Dispatch,
            None,
        );
    }

    fn on_node_terminal(
        &mut self,
        sessions: &mut [SessionRuntime],
        ready_sessions: &mut BinaryHeap<ReadySession>,
        node_index: usize,
        now_ms: f64,
        outcome: TurnOutcome,
    ) {
        let play_index = self.node_to_play[node_index];
        let status = match outcome {
            TurnOutcome::Completed => ReplayTerminalStatus::Completed,
            TurnOutcome::Rejected => ReplayTerminalStatus::Rejected,
            TurnOutcome::Cancelled => ReplayTerminalStatus::Canceled,
            TurnOutcome::Failed => ReplayTerminalStatus::Failed,
        };
        self.record_lifecycle(
            now_ms,
            play_index,
            Some(node_index),
            AgenticLifecycleEventKind::CausalTerminal,
            Some(status),
        );
        let was_failed = self.plays[play_index].failed;
        {
            let play = &mut self.plays[play_index];
            play.pending_terminals = play
                .pending_terminals
                .checked_sub(1)
                .expect("a dispatched agentic node becomes terminal exactly once");
            play.max_terminal_ms =
                Some(play.max_terminal_ms.map_or(now_ms, |seen| seen.max(now_ms)));
        }

        match outcome {
            TurnOutcome::Completed if !was_failed => {
                self.node_states[node_index] = AgenticNodeState::Completed;
                self.plays[play_index].completed_nodes += 1;
                self.release_completion_dependents(sessions, ready_sessions, node_index, now_ms);
            }
            TurnOutcome::Completed => {
                self.node_states[node_index] = AgenticNodeState::Completed;
                self.plays[play_index].completed_nodes += 1;
            }
            TurnOutcome::Rejected | TurnOutcome::Cancelled | TurnOutcome::Failed => {
                self.node_states[node_index] = AgenticNodeState::Failed;
                let play = &mut self.plays[play_index];
                play.failed = true;
                let candidate = AgenticFailureRecord {
                    node_index,
                    at_ms: now_ms,
                    status,
                };
                if play.primary_failure.is_none_or(|current| {
                    candidate
                        .at_ms
                        .total_cmp(&current.at_ms)
                        .then_with(|| candidate.node_index.cmp(&current.node_index))
                        == Ordering::Less
                }) {
                    play.primary_failure = Some(candidate);
                }
                if !was_failed {
                    let play_nodes = self.plays[play_index].nodes.clone();
                    for pending_node in play_nodes {
                        if matches!(
                            self.node_states[pending_node],
                            AgenticNodeState::Blocked | AgenticNodeState::Ready
                        ) {
                            self.node_states[pending_node] = AgenticNodeState::Skipped;
                            let session = &mut sessions[pending_node];
                            session.next_ready_at_ms = None;
                            session.next_turn_index = session.turns.len();
                            self.record_lifecycle(
                                now_ms,
                                play_index,
                                Some(pending_node),
                                AgenticLifecycleEventKind::Skipped,
                                None,
                            );
                        }
                    }
                }
            }
        }

        let play = &self.plays[play_index];
        let all_completed = !play.failed && play.completed_nodes == play.nodes.len();
        // A failed play skips undispatched work, but every dispatched request,
        // including background children, still owns client work until terminal.
        // Successful plays must also finish delayed and blocked authored nodes.
        if (all_completed || play.failed) && play.pending_terminals == 0 {
            self.release_lane(play_index, now_ms, sessions, ready_sessions);
        }
    }

    fn on_node_quiescent(&mut self, node_index: usize, now_ms: f64) -> Result<()> {
        let play_index = self.node_to_play[node_index];
        self.record_lifecycle(
            now_ms,
            play_index,
            Some(node_index),
            AgenticLifecycleEventKind::Quiescent,
            None,
        );
        let play = &mut self.plays[play_index];
        play.emitted_in_flight = play
            .emitted_in_flight
            .checked_sub(1)
            .context("agentic play settlement count underflow")?;
        if !play.quiescent && play.client_finished && play.emitted_in_flight == 0 {
            play.quiescent = true;
            play.quiescent_at_ms = Some(now_ms);
            self.record_lifecycle(
                now_ms,
                play_index,
                None,
                AgenticLifecycleEventKind::PlayQuiescent,
                None,
            );
        }
        Ok(())
    }

    fn release_lane(
        &mut self,
        play_index: usize,
        now_ms: f64,
        sessions: &mut [SessionRuntime],
        ready_sessions: &mut BinaryHeap<ReadySession>,
    ) {
        if self.plays[play_index].client_finished {
            return;
        }
        self.plays[play_index].client_finished = true;
        let Some(lane_index) = self.plays[play_index].lane_index else {
            return;
        };
        let lane = &mut self.lanes[lane_index];
        lane.next_play += 1;
        let Some(&next_play_index) = lane.plays.get(lane.next_play) else {
            return;
        };
        self.activate_play(next_play_index, now_ms, sessions, ready_sessions);
    }

    fn trajectory_snapshot(&self) -> AgenticTrajectorySnapshot {
        let mut e2e_latencies_ms = Vec::new();
        for play in &self.plays {
            if play.failed || play.completed_nodes != play.nodes.len() {
                continue;
            }
            let (Some(root_dispatch_ms), Some(max_terminal_ms)) =
                (play.root_dispatch_ms, play.max_terminal_ms)
            else {
                continue;
            };
            e2e_latencies_ms.push(max_terminal_ms - root_dispatch_ms);
        }
        AgenticTrajectorySnapshot {
            total_trajectories: self.plays.len(),
            completed_trajectories: e2e_latencies_ms.len(),
            e2e_latencies_ms,
        }
    }

    fn play_outcomes(&self, sessions: &[SessionRuntime]) -> Vec<AgenticPlayOutcome> {
        self.plays
            .iter()
            .map(|play| {
                let status = if !play.quiescent {
                    AgenticPlayStatus::Incomplete
                } else if play.failed {
                    AgenticPlayStatus::Failed
                } else {
                    AgenticPlayStatus::Completed
                };
                AgenticPlayOutcome {
                    play_id: play.play_id.clone(),
                    status,
                    causal_terminal_ms: play
                        .primary_failure
                        .map(|failure| failure.at_ms)
                        .or(play.max_terminal_ms),
                    settled_at_ms: play.quiescent_at_ms,
                    failure_request_id: play.primary_failure.map(|failure| {
                        sessions[failure.node_index].turns[0]
                            .request_id
                            .clone()
                            .expect("agentic node must retain its authored request ID")
                    }),
                    failure_status: play.primary_failure.map(|failure| failure.status),
                }
            })
            .collect()
    }
}

#[derive(Debug)]
pub struct WorkloadDriver {
    policy: SchedulingPolicy,
    prompt_mode: PromptMode,
    emit_session_metadata: bool,
    trace_block_size: usize,
    engine_block_size: u32,
    include_replay_hashes: bool,
    agentic_graph_identity: Option<AgenticGraphIdentity>,
    agentic_replay_context: Option<Arc<AgenticReplayContext>>,
    agentic_snapshots: Option<Vec<AgenticSnapshotEvidence>>,
    sessions: Vec<SessionRuntime>,
    in_flight: FxHashMap<Uuid, InFlightTurn>,
    agentic_settling: FxHashMap<Uuid, usize>,
    ready_sessions: BinaryHeap<ReadySession>,
}

impl WorkloadDriver {
    /// Execute the retained request suffix with a cold runtime. This consumes
    /// prepared logical state; it does not execute primers or restore native KV.
    /// Source history was compiled before this view and is never submitted.
    pub fn new_agentic_snapshots(
        prepared: PreparedAgenticSnapshots,
        engine_block_size: usize,
        include_replay_hashes: bool,
        speedup: f64,
    ) -> Result<Self> {
        if !speedup.is_finite() || speedup <= 0.0 {
            bail!("snapshot speedup must be finite and greater than zero");
        }
        let context = Arc::clone(&prepared.context);
        let mut view = AgenticTrace {
            block_size: context.graph.block_size,
            source: context.graph.source.clone(),
            graph_digest: context.graph.graph_digest.clone(),
            nodes: Vec::new(),
            plays: Vec::new(),
        };
        let mut source_nodes = Vec::new();
        for (cohort_index, snapshot) in prepared.plays.iter().enumerate() {
            let mut nodes = Vec::new();
            let mut root_nodes = Vec::new();
            for request in &snapshot.evidence.requests {
                if request.historical {
                    continue;
                }
                let source_index = snapshot.source_index(&request.source_request_id)?;
                let mut node = context.graph.nodes[source_index].clone();
                node.request_id = request.identity.request_id.clone();
                node.play_id = snapshot.evidence.play_id.clone();
                node.source_play_ordinal = Some(cohort_index);
                node.session_id = request.identity.conversation_id.clone();
                node.not_before_ms = request.remaining_delay_ms / speedup;
                node.dependencies.retain(|edge| {
                    request
                        .pending_dependencies
                        .contains(&snapshot.instance_request_id(&edge.request_id))
                });
                for edge in &mut node.dependencies {
                    edge.request_id = snapshot.instance_request_id(&edge.request_id);
                    edge.delay_ms /= speedup;
                }
                if !node.not_before_ms.is_finite()
                    || node.dependencies.iter().any(|e| !e.delay_ms.is_finite())
                {
                    bail!("snapshot speedup overflows replay timing");
                }
                let index = view.nodes.len();
                if node.dependencies.is_empty() {
                    root_nodes.push(index);
                }
                nodes.push(index);
                view.nodes.push(node);
                source_nodes.push((cohort_index, source_index));
            }
            if nodes.is_empty() || root_nodes.is_empty() {
                bail!("snapshot has no executable request frontier");
            }
            view.plays.push(AgenticPlay {
                play_id: snapshot.evidence.play_id.clone(),
                source_play_ordinal: Some(cohort_index),
                nodes,
                root_nodes,
            });
        }
        // The existing executor compiles the retained dependency structure. Its
        // temporary numbering is replaced by the full prepared context below.
        // No lane activation may renormalize the restored initial timers.
        let mut driver = Self::new_agentic_trace_with_options(
            view,
            engine_block_size,
            include_replay_hashes,
            None,
        )?;
        let SchedulingPolicy::Agentic(state) = &mut driver.policy else {
            unreachable!()
        };
        for (index, (cohort, source_index)) in source_nodes.into_iter().enumerate() {
            let snapshot = &prepared.plays[cohort];
            let source = &context.graph.nodes[source_index];
            let turn = &mut driver.sessions[index].turns[0];
            turn.prompt_tokens = PromptTokens::deferred(
                source.input_length,
                snapshot.token_ids(source_index)?,
                context.graph.block_size,
            )?;
            turn.output_token_ids = Some(context.outputs[source_index].clone());
            turn.deterministic_request_id = Some(snapshot.request_uuid(source_index)?);
            // Keep authored correlation separate from incarnation identity.
            turn.request_id = Some(source.request_id.clone());
            state.identities[index] = snapshot.identity_at(source_index);
        }
        for play in &mut state.plays {
            // Suffix duration is anchored at activation, even for a rootless
            // background frontier. No historical dispatch event is fabricated.
            play.root_dispatch_ms = Some(0.0);
        }
        driver.agentic_graph_identity = Some(context.graph.identity());
        driver.agentic_snapshots = Some(prepared.snapshots().to_vec());
        driver.agentic_replay_context = Some(context);
        Ok(driver)
    }

    pub fn is_agentic(&self) -> bool {
        matches!(self.policy, SchedulingPolicy::Agentic(_))
    }

    pub fn new_trace(trace: Trace, engine_block_size: usize) -> Result<Self> {
        Self::new(
            trace,
            engine_block_size,
            SchedulingPolicy::Trace,
            PromptMode::Full,
            true,
        )
    }

    pub fn new_trace_without_replay_hashes(
        trace: Trace,
        engine_block_size: usize,
        accumulate_session_deltas: bool,
    ) -> Result<Self> {
        trace.validate_for_trace_mode()?;
        let prompt_mode = if accumulate_session_deltas {
            PromptMode::DeltaCumulative
        } else {
            PromptMode::Full
        };
        Self::new(
            trace,
            engine_block_size,
            SchedulingPolicy::Trace,
            prompt_mode,
            false,
        )
    }

    pub fn new_trace_accumulating_deltas(trace: Trace, engine_block_size: usize) -> Result<Self> {
        Self::new(
            trace,
            engine_block_size,
            SchedulingPolicy::Trace,
            PromptMode::DeltaCumulative,
            true,
        )
    }

    /// Build a closed-loop concurrency driver. `max_in_flight` is the *session* cap
    /// (depth-first): a session holds its slot across all turns + think-time, and new
    /// sessions are admitted only while fewer than `max_in_flight` are active.
    pub fn new_concurrency(
        trace: Trace,
        engine_block_size: usize,
        max_in_flight: usize,
    ) -> Result<Self> {
        Self::new(
            trace,
            engine_block_size,
            SchedulingPolicy::Concurrency(ConcurrencyState::new(max_in_flight)),
            PromptMode::Full,
            true,
        )
    }

    pub fn new_concurrency_without_replay_hashes(
        trace: Trace,
        engine_block_size: usize,
        max_in_flight: usize,
        accumulate_session_deltas: bool,
    ) -> Result<Self> {
        trace.validate_for_concurrency_mode()?;
        let prompt_mode = if accumulate_session_deltas {
            PromptMode::DeltaCumulative
        } else {
            PromptMode::Full
        };
        Self::new(
            trace,
            engine_block_size,
            SchedulingPolicy::Concurrency(ConcurrencyState::new(max_in_flight)),
            prompt_mode,
            false,
        )
    }

    pub fn new_concurrency_accumulating_deltas(
        trace: Trace,
        engine_block_size: usize,
        max_in_flight: usize,
    ) -> Result<Self> {
        Self::new(
            trace,
            engine_block_size,
            SchedulingPolicy::Concurrency(ConcurrencyState::new(max_in_flight)),
            PromptMode::DeltaCumulative,
            true,
        )
    }

    pub fn new_agentic_trace(trace: AgenticTrace, engine_block_size: usize) -> Result<Self> {
        Self::new_agentic_trace_with_options(trace, engine_block_size, true, None)
    }

    pub fn new_agentic_trace_without_replay_hashes(
        trace: AgenticTrace,
        engine_block_size: usize,
    ) -> Result<Self> {
        Self::new_agentic_trace_with_options(trace, engine_block_size, false, None)
    }

    pub fn new_agentic_trace_with_lanes(
        trace: AgenticTrace,
        engine_block_size: usize,
        agentic_lanes: usize,
    ) -> Result<Self> {
        if agentic_lanes == 0 {
            bail!("agentic_lanes must be greater than 0");
        }
        Self::new_agentic_trace_with_options(trace, engine_block_size, true, Some(agentic_lanes))
    }

    #[allow(clippy::needless_range_loop)] // The index updates both play and lane tables.
    pub fn new_agentic_trace_with_options(
        trace: AgenticTrace,
        engine_block_size: usize,
        include_replay_hashes: bool,
        agentic_lanes: Option<usize>,
    ) -> Result<Self> {
        if engine_block_size == 0 {
            bail!("engine_block_size must be greater than 0");
        }
        if agentic_lanes == Some(0) {
            bail!("agentic_lanes must be greater than 0");
        }
        let engine_block_size_u32 =
            u32::try_from(engine_block_size).context("engine_block_size does not fit in u32")?;
        let agentic_graph_identity = trace.identity();
        let trace_block_size = trace.block_size;
        let mut hash_id_interner = FxHashMap::default();
        let mut next_hash_id = 0_u32;
        let mut index_by_id = FxHashMap::default();
        for (node_index, node) in trace.nodes.iter().enumerate() {
            index_by_id.insert(node.request_id.clone(), node_index);
        }
        let root_id_by_play = trace
            .plays
            .iter()
            .map(|play| {
                let root_id = (play.root_nodes.len() == 1)
                    .then(|| trace.nodes[play.root_nodes[0]].request_id.clone());
                (play.play_id.clone(), root_id)
            })
            .collect::<FxHashMap<_, _>>();
        let mut identities = trace
            .nodes
            .iter()
            .map(|node| {
                let parent_id = node
                    .dependencies
                    .iter()
                    .filter(|dependency| dependency.relation == AgenticDependencyRelation::Spawn)
                    .min_by_key(|dependency| index_by_id[&dependency.request_id])
                    .map(|dependency| dependency.request_id.clone());
                AgenticRuntimeIdentity {
                    request_id: node.request_id.clone(),
                    play_id: node.play_id.clone(),
                    conversation_id: node.session_id.clone(),
                    lane_id: None,
                    root_id: root_id_by_play.get(&node.play_id).cloned().flatten(),
                    parent_id,
                    cache_id: None,
                }
            })
            .collect::<Vec<_>>();
        let mut dispatch_dependents = vec![Vec::new(); trace.nodes.len()];
        let mut completion_dependents = vec![Vec::new(); trace.nodes.len()];
        let mut remaining_dependencies = Vec::with_capacity(trace.nodes.len());
        let mut authored_not_before_ms = Vec::with_capacity(trace.nodes.len());
        let mut sessions = Vec::with_capacity(trace.nodes.len());
        let mut output_rng = StdRng::seed_from_u64(SYNTHETIC_OUTPUT_SEED);

        for (node_index, node) in trace.nodes.into_iter().enumerate() {
            for dependency in &node.dependencies {
                let source_node = *index_by_id
                    .get(&dependency.request_id)
                    .expect("validated agentic dependency must exist");
                let edge = AgenticDependentEdge {
                    target_node: node_index,
                    delay_ms: dependency.delay_ms,
                };
                match dependency.trigger {
                    AgenticDependencyTrigger::Dispatch => {
                        dispatch_dependents[source_node].push(edge)
                    }
                    AgenticDependencyTrigger::Completion => {
                        completion_dependents[source_node].push(edge)
                    }
                }
            }
            remaining_dependencies.push(node.dependencies.len());
            authored_not_before_ms.push(node.not_before_ms);

            let hash_ids = node
                .hash_ids
                .into_iter()
                .map(|hash_id| {
                    if let Some(&interned) = hash_id_interner.get(&hash_id) {
                        return Ok(interned);
                    }
                    let interned = next_hash_id;
                    next_hash_id = next_hash_id
                        .checked_add(1)
                        .context("trace contains more unique hash IDs than u32 can represent")?;
                    hash_id_interner.insert(hash_id, interned);
                    Ok(interned)
                })
                .collect::<Result<Vec<_>>>()?;

            let prompt_tokens =
                PromptTokens::deferred(node.input_length, hash_ids, trace_block_size)?;
            let output_token_ids = Some(planned_output_token_ids(
                node.output_token_ids,
                node.max_output_tokens,
                &mut output_rng,
            ));
            // WorkloadDriver is deliberately model-neutral. Callers must define
            // and report how source-model provenance is projected onto the
            // configured execution timing model before handing the graph here.
            let deterministic_request_id = Uuid::from_u128(
                u128::try_from(node_index)
                    .expect("usize always fits in u128")
                    .checked_add(1)
                    .context("agentic request UUID ordinal overflow")?,
            );
            sessions.push(SessionRuntime {
                session_id: node.session_id,
                turns: vec![TurnRuntime {
                    request_id: Some(node.request_id),
                    play_id: Some(node.play_id),
                    replay_key: node.replay_key,
                    prompt_tokens,
                    max_output_tokens: node.max_output_tokens,
                    output_token_ids,
                    delay_after_previous_ms: 0.0,
                    priority: node.priority,
                    strict_priority: node.strict_priority,
                    policy_class: node.policy_class,
                    deterministic_request_id: Some(deterministic_request_id),
                }],
                cumulative_tokens: Vec::new(),
                next_turn_index: 0,
                next_ready_at_ms: None,
                in_flight: None,
            });
        }

        let mut node_to_play = vec![usize::MAX; sessions.len()];
        let mut plays = trace
            .plays
            .into_iter()
            .enumerate()
            .map(|(play_index, play)| {
                for &node_index in &play.nodes {
                    node_to_play[node_index] = play_index;
                }
                AgenticPlayState {
                    play_id: play.play_id,
                    nodes: play.nodes,
                    root_nodes: play.root_nodes,
                    lane_index: None,
                    pending_terminals: 0,
                    emitted_in_flight: 0,
                    completed_nodes: 0,
                    failed: false,
                    client_finished: false,
                    quiescent: false,
                    root_dispatch_ms: None,
                    max_terminal_ms: None,
                    quiescent_at_ms: None,
                    primary_failure: None,
                }
            })
            .collect::<Vec<_>>();
        debug_assert!(node_to_play.iter().all(|play| *play != usize::MAX));

        let mut lanes = Vec::new();
        if let Some(lane_count) = agentic_lanes {
            lanes = (0..lane_count)
                .map(|_| AgenticLaneState {
                    plays: Vec::new(),
                    next_play: 0,
                })
                .collect();
            for play_index in 0..plays.len() {
                let lane_index = play_index % lane_count;
                plays[play_index].lane_index = Some(lane_index);
                lanes[lane_index].plays.push(play_index);
            }
            for (node_index, identity) in identities.iter_mut().enumerate() {
                identity.lane_id = Some(format!(
                    "lane:{}",
                    plays[node_to_play[node_index]].lane_index.unwrap()
                ));
            }
        }

        let mut state = AgenticState {
            identities,
            node_states: vec![AgenticNodeState::Blocked; sessions.len()],
            remaining_dependencies,
            ready_after_ms: authored_not_before_ms.clone(),
            authored_not_before_ms,
            dispatch_dependents,
            completion_dependents,
            node_to_play,
            plays,
            lanes,
            lifecycle: Vec::new(),
            next_lifecycle_ordinal: 0,
            last_runtime_feedback_at_ms: None,
        };
        let mut ready_sessions = BinaryHeap::new();
        if state.lanes.is_empty() {
            for play in &state.plays {
                for &root_node in &play.root_nodes {
                    AgenticState::schedule_node(
                        root_node,
                        state.ready_after_ms[root_node],
                        &mut state.node_states,
                        &mut sessions,
                        &mut ready_sessions,
                    );
                }
            }
        } else {
            let initial_plays = state
                .lanes
                .iter()
                .filter_map(|lane| lane.plays.first().copied())
                .collect::<Vec<_>>();
            for play_index in initial_plays {
                state.activate_play(play_index, 0.0, &mut sessions, &mut ready_sessions);
            }
        }

        Ok(Self {
            policy: SchedulingPolicy::Agentic(Box::new(state)),
            prompt_mode: PromptMode::Full,
            emit_session_metadata: true,
            trace_block_size,
            engine_block_size: engine_block_size_u32,
            include_replay_hashes,
            agentic_graph_identity: Some(agentic_graph_identity),
            agentic_replay_context: None,
            agentic_snapshots: None,
            sessions,
            in_flight: FxHashMap::default(),
            agentic_settling: FxHashMap::default(),
            ready_sessions,
        })
    }

    fn new(
        trace: Trace,
        engine_block_size: usize,
        policy: SchedulingPolicy,
        prompt_mode: PromptMode,
        include_replay_hashes: bool,
    ) -> Result<Self> {
        if engine_block_size == 0 {
            bail!("engine_block_size must be greater than 0");
        }
        let engine_block_size_u32 =
            u32::try_from(engine_block_size).context("engine_block_size does not fit in u32")?;
        let trace_block_size = trace.block_size;
        let is_concurrency = matches!(&policy, SchedulingPolicy::Concurrency(_));
        let mut output_rng = StdRng::seed_from_u64(SYNTHETIC_OUTPUT_SEED);
        let sessions: Vec<SessionRuntime> = trace
            .sessions
            .into_iter()
            .map(|session| -> Result<SessionRuntime> {
                let next_ready_at_ms = if is_concurrency {
                    None
                } else {
                    Some(session.first_arrival_timestamp_ms.unwrap_or(0.0))
                };
                let turns = session
                    .turns
                    .into_iter()
                    .map(|mut turn| -> Result<TurnRuntime> {
                        let prompt_tokens = match prompt_mode {
                            PromptMode::Full => PromptTokens::deferred(
                                turn.input_length,
                                std::mem::take(&mut turn.hash_ids),
                                trace_block_size,
                            )?,
                            PromptMode::DeltaCumulative => PromptTokens::Materialized(
                                turn.synthesize_tokens(trace_block_size)?,
                            ),
                        };
                        let output_token_ids = Some(planned_output_token_ids(
                            turn.output_token_ids,
                            turn.max_output_tokens,
                            &mut output_rng,
                        ));
                        Ok(TurnRuntime {
                            request_id: None,
                            play_id: None,
                            prompt_tokens,
                            replay_key: turn.replay_key,
                            max_output_tokens: turn.max_output_tokens,
                            output_token_ids,
                            delay_after_previous_ms: turn.delay_after_previous_ms,
                            priority: turn.priority,
                            strict_priority: turn.strict_priority,
                            policy_class: turn.policy_class,
                            deterministic_request_id: None,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let cumulative_capacity = if prompt_mode == PromptMode::DeltaCumulative {
                    turns
                        .iter()
                        .map(|turn| {
                            turn.prompt_tokens.input_length()
                                + turn
                                    .output_token_ids
                                    .as_ref()
                                    .map_or(0, |output| output.len())
                        })
                        .sum()
                } else {
                    0
                };
                Ok(SessionRuntime {
                    session_id: session.session_id,
                    turns,
                    cumulative_tokens: Vec::with_capacity(cumulative_capacity),
                    next_turn_index: 0,
                    next_ready_at_ms,
                    in_flight: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let ready_sessions = sessions
            .iter()
            .enumerate()
            .filter_map(|(session_index, session)| {
                Some(ReadySession {
                    ready_at_ms: session.next_ready_at_ms?,
                    session_index,
                    turn_index: session.next_turn_index,
                })
            })
            .collect();

        let mut driver = Self {
            policy,
            prompt_mode,
            emit_session_metadata: true,
            trace_block_size,
            engine_block_size: engine_block_size_u32,
            include_replay_hashes,
            agentic_graph_identity: None,
            agentic_replay_context: None,
            agentic_snapshots: None,
            sessions,
            in_flight: FxHashMap::default(),
            agentic_settling: FxHashMap::default(),
            ready_sessions,
        };
        if let SchedulingPolicy::Concurrency(state) = &mut driver.policy {
            state.activate_pending(&mut driver.sessions, &mut driver.ready_sessions, 0.0);
        }
        Ok(driver)
    }

    /// Use stable monotonically increasing UUIDs for canonical replay.
    /// Callers must opt in through [`crate::replay::ReplayDeterminism::CanonicalV1`].
    pub fn with_deterministic_request_ids(mut self, first_id: u128) -> Self {
        self.set_deterministic_request_ids(first_id);
        self
    }

    pub(crate) fn set_deterministic_request_ids(&mut self, first_id: u128) {
        // Prepared UUIDs address the complete play, including omitted history.
        // Canonical replay must not renumber the retained suffix or a new play.
        if self.agentic_replay_context.is_some() {
            return;
        }
        let mut next_id = first_id;
        for session in &mut self.sessions {
            for turn in &mut session.turns {
                turn.deterministic_request_id = Some(Uuid::from_u128(next_id));
                next_id = next_id
                    .checked_add(1)
                    .expect("deterministic replay request UUID overflow");
            }
        }
    }

    /// Snapshot fixed input demand without executing any workload lifecycle.
    /// Reserving opaque IDs is the only mutation: arrivals, outputs, cursors,
    /// and readiness remain owned by the causal driver. Output plans are
    /// intentionally excluded; this forecast is not actual future cache reuse.
    pub(crate) fn prepare_belady_requests(
        &mut self,
        engine_block_size: usize,
    ) -> Result<Vec<(Uuid, Vec<SequenceHash>)>> {
        if !matches!(self.policy, SchedulingPolicy::Trace) || self.prompt_mode != PromptMode::Full {
            bail!(
                "belady requires a full-prompt, fixed-arrival trace driver; concurrency, agentic, and delta workloads are unsupported"
            );
        }
        if usize::try_from(self.engine_block_size)? != engine_block_size {
            bail!("belady workload engine block size must match the replay engine");
        }
        if !self.in_flight.is_empty()
            || self.ready_sessions.len() != self.sessions.len()
            || self.sessions.iter().any(|session| {
                session.turns.len() != 1
                    || session.next_turn_index != 0
                    || session.in_flight.is_some()
                    || session.next_ready_at_ms.is_none()
                    || !session.cumulative_tokens.is_empty()
            })
        {
            bail!("belady requires a pristine trace driver with one turn per session");
        }
        if self.sessions.iter().any(|session| {
            let arrival = session.next_ready_at_ms.expect("validated ready session");
            !arrival.is_finite() || arrival < 0.0
        }) {
            bail!("belady requires finite, nonnegative trace arrival times");
        }

        // Match ReadySession's arrival/source order, not UUID order. A future
        // occurrence is global demand even if routing later selects a different
        // worker; predicting that placement is deliberately outside this oracle.
        let mut order = (0..self.sessions.len()).collect::<Vec<_>>();
        order.sort_by(|&left, &right| {
            self.sessions[left]
                .next_ready_at_ms
                .unwrap()
                .total_cmp(&self.sessions[right].next_ready_at_ms.unwrap())
                .then_with(|| left.cmp(&right))
        });
        let mut forecast = Vec::with_capacity(order.len());
        for session_index in order {
            let turn = &mut self.sessions[session_index].turns[0];
            let request_id = *turn
                .deterministic_request_id
                .get_or_insert_with(Uuid::new_v4);
            let PromptTokens::Deferred {
                input_length,
                hash_ids,
            } = &turn.prompt_tokens
            else {
                unreachable!("full-prompt turns retain deferred input tokens");
            };
            // Use the same normalization as eventual admission, including when
            // source trace blocks and native engine blocks have different sizes.
            // Only this one expanded prompt is live; retain hashes in the oracle.
            let tokens =
                synthesize_validated_trace_tokens(*input_length, hash_ids, self.trace_block_size);
            forecast.push((
                request_id,
                input_sequence_hashes(&tokens, engine_block_size),
            ));
        }
        Ok(forecast)
    }

    fn request_uuid(&self, _session_index: usize, _turn_index: usize) -> Uuid {
        if let Some(request_id) =
            self.sessions[_session_index].turns[_turn_index].deterministic_request_id
        {
            return request_id;
        }

        Uuid::new_v4()
    }

    pub fn without_session_metadata(mut self) -> Self {
        self.emit_session_metadata = false;
        self
    }

    /// Failure-path companion: release a cap slot and terminate the owning session.
    /// No-op if `on_complete` already ran. Used when a request task is cancelled
    /// or panics before reaching `on_complete`.
    ///
    /// Terminating the session (marking it exhausted) prevents `run_workload` from
    /// deadlocking: `pop_ready` skips sessions with `in_flight.is_some()`, so a
    /// leaked session would leave `is_drained` stuck at `false` forever.
    pub fn release_cap_slot(&mut self, request_uuid: Uuid, now_ms: f64) {
        let Ok(Some(resolution)) = self.resolve_turn(request_uuid, now_ms, TurnOutcome::Cancelled)
        else {
            return;
        };
        self.apply_resolution(resolution, now_ms);
    }

    pub fn pop_ready(&mut self, now_ms: f64, limit: usize) -> Vec<ReadyTurn> {
        self.pop_ready_compact(now_ms, limit)
            .into_iter()
            .map(CompactReadyTurn::into_ready_turn)
            .collect()
    }

    #[doc(hidden)]
    pub fn pop_ready_compact(&mut self, now_ms: f64, limit: usize) -> Vec<CompactReadyTurn> {
        let effective_limit = self.policy.dispatch_limit(limit, self.in_flight.len());
        if effective_limit == 0 {
            return Vec::new();
        }

        let mut emitted = Vec::new();
        while emitted.len() < effective_limit {
            let Some(ready_session) = self.ready_sessions.pop() else {
                break;
            };
            if ready_session.ready_at_ms > now_ms {
                self.ready_sessions.push(ready_session);
                break;
            }

            let session_index = ready_session.session_index;
            let Some((turn_index, scheduled_ready_at_ms)) = self
                .sessions
                .get(session_index)
                .filter(|session| {
                    session.in_flight.is_none()
                        && session.next_turn_index == ready_session.turn_index
                        && session.next_ready_at_ms == Some(ready_session.ready_at_ms)
                })
                .map(|session| {
                    (
                        session.next_turn_index,
                        session
                            .next_ready_at_ms
                            .expect("ready session must have a timestamp"),
                    )
                })
            else {
                continue;
            };
            let request_uuid = self.request_uuid(session_index, turn_index);
            let agentic_identity = match &self.policy {
                SchedulingPolicy::Agentic(state) => Some(state.identities[session_index].clone()),
                SchedulingPolicy::Trace | SchedulingPolicy::Concurrency(_) => None,
            };
            let session = &mut self.sessions[session_index];
            let turn = &mut session.turns[turn_index];
            let replay_context = turn.request_id.as_ref().zip(turn.play_id.as_ref()).map(
                |(request_id, _play_id)| ReplayRequestContext {
                    authored_id: request_id.clone(),
                    session_id: Some(session.session_id.clone()),
                    turn_index: None,
                    metadata: serde_json::Value::Null,
                    prompt_token_source: ReplayPromptTokenSource::Materialized,
                    agentic: agentic_identity,
                },
            );
            let arrival_timestamp_ms = self.policy.arrival_timestamp_ms(scheduled_ready_at_ms);
            let (request, replay_hashes) = match self.prompt_mode {
                PromptMode::Full => {
                    let (input_length, hash_ids) = turn.prompt_tokens.take_deferred();
                    let request_metadata = DirectRequest {
                        tokens: Vec::new(),
                        max_output_tokens: turn.max_output_tokens,
                        output_token_ids: turn.output_token_ids.take(),
                        uuid: Some(request_uuid),
                        dp_rank: 0,
                        preferred_dp_rank: None,
                        preferred_prefill_dp_rank: None,
                        arrival_timestamp_ms,
                        priority: turn.priority,
                        strict_priority: turn.strict_priority,
                        policy_class: turn.policy_class.clone(),
                        replay_context: replay_context.clone(),
                    };
                    let request = ReplayRequestPayload::deferred(
                        request_metadata,
                        input_length,
                        hash_ids,
                        self.trace_block_size,
                    );
                    // The router needs engine-block hashes at arrival, but it
                    // does not need to retain the expanded prompt. Materialize
                    // once transiently for hashing, then keep only the compact
                    // payload until a worker admission.
                    // TODO: Derive engine-block hashes directly from the compact
                    // trace blocks so immediate dispatch does not materialize
                    // the prompt once for routing and again for admission.
                    // Preserve `ReplayRequestHashes::from_tokens` semantics when
                    // trace and engine block sizes differ.
                    let replay_hashes = self.include_replay_hashes.then(|| {
                        let request_tokens = request.prompt_tokens();
                        ReplayRequestHashes::from_tokens(&request_tokens, self.engine_block_size)
                    });
                    (request, replay_hashes)
                }
                PromptMode::DeltaCumulative => {
                    session
                        .cumulative_tokens
                        .extend_from_slice(turn.prompt_tokens.materialized());
                    let request_tokens = session.cumulative_tokens.clone();
                    let replay_hashes = self.include_replay_hashes.then(|| {
                        ReplayRequestHashes::from_tokens(&request_tokens, self.engine_block_size)
                    });
                    let request = ReplayRequestPayload::materialized(DirectRequest {
                        tokens: request_tokens,
                        max_output_tokens: turn.max_output_tokens,
                        output_token_ids: turn.output_token_ids.clone(),
                        uuid: Some(request_uuid),
                        dp_rank: 0,
                        preferred_dp_rank: None,
                        preferred_prefill_dp_rank: None,
                        arrival_timestamp_ms,
                        priority: turn.priority,
                        strict_priority: turn.strict_priority,
                        policy_class: turn.policy_class.clone(),
                        replay_context,
                    });
                    (request, replay_hashes)
                }
            };
            session.in_flight = Some(request_uuid);
            session.next_ready_at_ms = None;
            self.in_flight.insert(
                request_uuid,
                InFlightTurn {
                    session_index,
                    turn_index,
                    emitted_output_tokens: 0,
                },
            );
            emitted.push(CompactReadyTurn {
                request_uuid,
                authored_request_id: turn.request_id.clone(),
                play_id: turn.play_id.clone(),
                dispatched_at_ms: now_ms,
                session_id: session.session_id.clone(),
                turn_index,
                replay_key: turn.replay_key.clone(),
                scheduled_ready_at_ms,
                replay_hashes,
                emit_session_metadata: self.emit_session_metadata,
                request,
            });
            if let SchedulingPolicy::Agentic(state) = &mut self.policy {
                state.on_node_emitted(session_index, now_ms);
                state.release_dispatch_dependents(
                    &mut self.sessions,
                    &mut self.ready_sessions,
                    session_index,
                    now_ms,
                );
            }
        }
        emitted
    }

    pub fn on_output_token(&mut self, request_uuid: Uuid, token_id: u32) -> Result<()> {
        if self.prompt_mode == PromptMode::Full {
            return Ok(());
        }
        let in_flight = self
            .in_flight
            .get(&request_uuid)
            .copied()
            .ok_or_else(|| anyhow!("unknown workload request output for {request_uuid}"))?;

        let turn = &self.sessions[in_flight.session_index].turns[in_flight.turn_index];
        let planned_output_tokens = turn
            .output_token_ids
            .as_ref()
            .expect("delta turns must have planned output tokens");
        let expected_token = planned_output_tokens
            .get(in_flight.emitted_output_tokens)
            .ok_or_else(|| {
                anyhow!(
                    "workload request {request_uuid} emitted more than {} planned output tokens",
                    planned_output_tokens.len()
                )
            })?;
        if token_id != *expected_token {
            bail!(
                "workload request {request_uuid} emitted token {token_id} at position {}, expected {}",
                in_flight.emitted_output_tokens,
                expected_token
            );
        }

        let in_flight = self
            .in_flight
            .get_mut(&request_uuid)
            .expect("validated in-flight request must still exist");
        in_flight.emitted_output_tokens = in_flight
            .emitted_output_tokens
            .checked_add(1)
            .context("workload emitted output token count overflow")?;
        Ok(())
    }

    fn agentic_node_ordinal(&self, request_uuid: Uuid) -> Result<usize> {
        self.in_flight
            .get(&request_uuid)
            .map(|turn| turn.session_index)
            .or_else(|| self.agentic_settling.get(&request_uuid).copied())
            .ok_or_else(|| anyhow!("unknown agentic workload request {request_uuid}"))
    }

    /// Apply every workload-visible effect at one runtime-owned timestamp.
    ///
    /// Callers may collect effects in engine-specific order. This boundary
    /// makes that order unobservable by sorting requests by immutable graph
    /// ordinal, then applying output progress, causal terminals, and finally
    /// resource quiescence. Batch timestamps must be nondecreasing; a rejected
    /// batch leaves request state, dependencies, lifecycle, and clock unchanged.
    pub fn apply_agentic_runtime_feedback(
        &mut self,
        mut feedback: AgenticRuntimeFeedback,
    ) -> Result<()> {
        if !feedback.at_ms.is_finite() || feedback.at_ms < 0.0 {
            bail!(
                "agentic runtime feedback timestamp must be finite and non-negative; got {}",
                feedback.at_ms
            );
        }
        let SchedulingPolicy::Agentic(state) = &self.policy else {
            bail!("agentic runtime feedback requires an agentic workload driver");
        };
        if let Some(last_at_ms) = state.last_runtime_feedback_at_ms
            && feedback.at_ms < last_at_ms
        {
            bail!(
                "agentic runtime feedback timestamp regressed from {last_at_ms} ms to {} ms",
                feedback.at_ms
            );
        }

        let ordinal_by_uuid = feedback
            .output_tokens
            .iter()
            .map(|output| output.request_uuid)
            .chain(
                feedback
                    .causal_terminals
                    .iter()
                    .map(|terminal| terminal.request_uuid),
            )
            .chain(feedback.quiescent_requests.iter().copied())
            .map(|uuid| Ok((uuid, self.agentic_node_ordinal(uuid)?)))
            .collect::<Result<FxHashMap<_, _>>>()?;
        feedback
            .output_tokens
            .sort_by_key(|output| ordinal_by_uuid[&output.request_uuid]);
        feedback
            .causal_terminals
            .sort_by_key(|terminal| ordinal_by_uuid[&terminal.request_uuid]);
        feedback
            .quiescent_requests
            .sort_by_key(|uuid| ordinal_by_uuid[uuid]);

        for pair in feedback.output_tokens.windows(2) {
            if pair[0].request_uuid == pair[1].request_uuid {
                bail!(
                    "agentic request {} has duplicate output groups at {} ms",
                    pair[0].request_uuid,
                    feedback.at_ms
                );
            }
        }
        for pair in feedback.causal_terminals.windows(2) {
            if pair[0].request_uuid == pair[1].request_uuid {
                bail!(
                    "agentic request {} has duplicate causal terminals at {} ms",
                    pair[0].request_uuid,
                    feedback.at_ms
                );
            }
        }
        for pair in feedback.quiescent_requests.windows(2) {
            if pair[0] == pair[1] {
                bail!(
                    "agentic request {} has duplicate quiescence at {} ms",
                    pair[0],
                    feedback.at_ms
                );
            }
        }

        // Agentic graphs use full prompts, for which output feedback is a
        // no-op. Validate every fallible lifecycle transition before applying
        // any terminal: a later invalid transition must not commit earlier
        // completions, release dependencies, or consume pending cleanup.
        debug_assert_eq!(self.prompt_mode, PromptMode::Full);
        let mut becoming_terminal = FxHashSet::default();
        for terminal in &feedback.causal_terminals {
            let outcome = match terminal.status {
                ReplayTerminalStatus::Completed => TurnOutcome::Completed,
                ReplayTerminalStatus::Rejected => TurnOutcome::Rejected,
                ReplayTerminalStatus::Canceled => TurnOutcome::Cancelled,
                ReplayTerminalStatus::Failed => TurnOutcome::Failed,
            };
            if self.agentic_settling.contains_key(&terminal.request_uuid)
                || self
                    .validate_turn_resolution(terminal.request_uuid, outcome)?
                    .is_none()
            {
                bail!(
                    "agentic request {} received duplicate causal terminal",
                    terminal.request_uuid
                );
            }
            becoming_terminal.insert(terminal.request_uuid);
        }
        for request_uuid in &feedback.quiescent_requests {
            if !self.agentic_settling.contains_key(request_uuid)
                && !becoming_terminal.contains(request_uuid)
            {
                bail!("agentic request {request_uuid} became quiescent before its causal terminal");
            }
        }

        for output in feedback.output_tokens {
            for token_id in output.token_ids {
                self.on_output_token(output.request_uuid, token_id)?;
            }
        }
        for terminal in feedback.causal_terminals {
            self.on_causal_terminal(terminal.request_uuid, feedback.at_ms, terminal.status)?;
        }
        for request_uuid in feedback.quiescent_requests {
            self.on_quiescent(request_uuid, feedback.at_ms)?;
        }
        let SchedulingPolicy::Agentic(state) = &mut self.policy else {
            unreachable!("feedback application retains the agentic scheduling policy");
        };
        state.last_runtime_feedback_at_ms = Some(feedback.at_ms);
        Ok(())
    }

    pub fn on_complete(&mut self, request_uuid: Uuid, now_ms: f64) -> Result<()> {
        self.on_terminal(request_uuid, now_ms, ReplayTerminalStatus::Completed)
    }

    pub fn on_terminal(
        &mut self,
        request_uuid: Uuid,
        now_ms: f64,
        status: ReplayTerminalStatus,
    ) -> Result<()> {
        self.on_causal_terminal(request_uuid, now_ms, status)?;
        self.on_quiescent(request_uuid, now_ms)
    }

    /// Resolve the logical request outcome and update client scheduling.
    ///
    /// Disaggregated replay calls this as soon as final decode succeeds or a
    /// failure is observed. A play releases its client lane once all required
    /// requests are terminal, independently of the later [`Self::on_quiescent`]
    /// callbacks for P/D resources and coordinator actions. Aggregated replay uses
    /// [`Self::on_terminal`], where both events coincide.
    #[allow(clippy::collapsible_if)] // Keep agentic-only duplicate detection explicit.
    pub fn on_causal_terminal(
        &mut self,
        request_uuid: Uuid,
        now_ms: f64,
        status: ReplayTerminalStatus,
    ) -> Result<()> {
        let outcome = match status {
            ReplayTerminalStatus::Completed => TurnOutcome::Completed,
            ReplayTerminalStatus::Rejected => TurnOutcome::Rejected,
            ReplayTerminalStatus::Canceled => TurnOutcome::Cancelled,
            ReplayTerminalStatus::Failed => TurnOutcome::Failed,
        };
        let is_agentic = matches!(self.policy, SchedulingPolicy::Agentic(_));
        let Some(resolution) = self.resolve_turn(request_uuid, now_ms, outcome)? else {
            if is_agentic {
                bail!("agentic request {request_uuid} received duplicate causal terminal");
            }
            return Ok(());
        };
        if is_agentic {
            if self
                .agentic_settling
                .insert(request_uuid, resolution.session_index)
                .is_some()
            {
                bail!("agentic request {request_uuid} received duplicate causal terminal");
            }
        }
        self.apply_resolution(resolution, now_ms);
        Ok(())
    }

    /// Mark an already-terminal agentic request free of runtime-owned state.
    pub fn on_quiescent(&mut self, request_uuid: Uuid, now_ms: f64) -> Result<()> {
        let SchedulingPolicy::Agentic(state) = &mut self.policy else {
            return Ok(());
        };
        let Some(node_index) = self.agentic_settling.remove(&request_uuid) else {
            bail!("agentic request {request_uuid} became quiescent before its causal terminal");
        };
        state.on_node_quiescent(node_index, now_ms)
    }

    fn validate_turn_resolution(
        &self,
        request_uuid: Uuid,
        outcome: TurnOutcome,
    ) -> Result<Option<InFlightTurn>> {
        let Some(in_flight) = self.in_flight.get(&request_uuid).copied() else {
            return match outcome {
                TurnOutcome::Completed | TurnOutcome::Rejected | TurnOutcome::Failed => Err(
                    anyhow!("unknown workload request completion for {request_uuid}"),
                ),
                TurnOutcome::Cancelled => Ok(None),
            };
        };
        let session = self
            .sessions
            .get(in_flight.session_index)
            .ok_or_else(|| anyhow!("unknown workload session {}", in_flight.session_index))?;
        session.turns.get(in_flight.turn_index).ok_or_else(|| {
            anyhow!(
                "unknown workload turn {} for session {}",
                in_flight.turn_index,
                session.session_id
            )
        })?;
        if session.in_flight != Some(request_uuid) {
            bail!(
                "session {} resolution for {} does not match in-flight request {:?}",
                session.session_id,
                request_uuid,
                session.in_flight
            );
        }
        if session.next_turn_index != in_flight.turn_index {
            bail!(
                "session {} resolution for turn {} does not match next turn {}",
                session.session_id,
                in_flight.turn_index,
                session.next_turn_index
            );
        }

        if outcome == TurnOutcome::Rejected && in_flight.emitted_output_tokens != 0 {
            bail!(
                "rejected workload request {request_uuid} emitted {} output tokens",
                in_flight.emitted_output_tokens
            );
        }
        if matches!(outcome, TurnOutcome::Completed | TurnOutcome::Rejected) {
            in_flight
                .turn_index
                .checked_add(1)
                .context("workload turn index overflow")?;
        }
        Ok(Some(in_flight))
    }

    fn resolve_turn(
        &mut self,
        request_uuid: Uuid,
        now_ms: f64,
        outcome: TurnOutcome,
    ) -> Result<Option<TurnResolution>> {
        let Some(in_flight) = self.validate_turn_resolution(request_uuid, outcome)? else {
            return Ok(None);
        };
        let session = &self.sessions[in_flight.session_index];
        let turn = &session.turns[in_flight.turn_index];
        let completed_output_tokens = (outcome == TurnOutcome::Completed
            && self.prompt_mode == PromptMode::DeltaCumulative)
            .then(|| {
                let planned_output_tokens = turn
                    .output_token_ids
                    .as_ref()
                    .expect("delta turns must have planned output tokens");
                planned_output_tokens[..in_flight.emitted_output_tokens].to_vec()
            });
        let (next_turn_index, next_ready_at_ms, session_ended) = match outcome {
            TurnOutcome::Completed | TurnOutcome::Rejected => {
                let next_turn_index = in_flight
                    .turn_index
                    .checked_add(1)
                    .context("workload turn index overflow")?;
                let has_more_turns = self.policy.schedules_sequential_turns()
                    && next_turn_index < session.turns.len();
                let next_ready_at_ms = has_more_turns
                    .then(|| now_ms + session.turns[next_turn_index].delay_after_previous_ms);
                (next_turn_index, next_ready_at_ms, !has_more_turns)
            }
            TurnOutcome::Cancelled | TurnOutcome::Failed => (session.turns.len(), None, true),
        };

        self.in_flight
            .remove(&request_uuid)
            .expect("validated in-flight request must still exist");
        let session = &mut self.sessions[in_flight.session_index];
        session.in_flight = None;
        session.next_turn_index = next_turn_index;
        session.next_ready_at_ms = next_ready_at_ms;
        if session_ended {
            session.cumulative_tokens = Vec::new();
        }
        if next_ready_at_ms.is_some()
            && let Some(output_tokens) = completed_output_tokens
        {
            session.cumulative_tokens.extend(output_tokens);
        }
        if let Some(ready_at_ms) = next_ready_at_ms {
            self.ready_sessions.push(ReadySession {
                ready_at_ms,
                session_index: in_flight.session_index,
                turn_index: next_turn_index,
            });
        }

        Ok(Some(TurnResolution {
            session_index: in_flight.session_index,
            outcome,
            session_ended,
        }))
    }

    fn apply_resolution(&mut self, resolution: TurnResolution, now_ms: f64) {
        match &mut self.policy {
            SchedulingPolicy::Trace => {}
            SchedulingPolicy::Concurrency(state) => {
                if resolution.session_ended {
                    state.on_session_finished(&mut self.sessions, &mut self.ready_sessions, now_ms);
                }
            }
            SchedulingPolicy::Agentic(state) => {
                state.on_node_terminal(
                    &mut self.sessions,
                    &mut self.ready_sessions,
                    resolution.session_index,
                    now_ms,
                    resolution.outcome,
                );
            }
        }
    }

    pub fn next_ready_time_ms(&mut self) -> Option<f64> {
        if self.policy.at_dispatch_capacity(self.in_flight.len()) {
            return None;
        }
        loop {
            let ready_session = *self.ready_sessions.peek()?;
            let session = &self.sessions[ready_session.session_index];
            if session.in_flight.is_some()
                || session.next_turn_index != ready_session.turn_index
                || session.next_ready_at_ms != Some(ready_session.ready_at_ms)
            {
                self.ready_sessions.pop();
                continue;
            }
            return Some(ready_session.ready_at_ms);
        }
    }

    pub fn is_drained(&self) -> bool {
        self.in_flight.is_empty()
            && self.agentic_settling.is_empty()
            // Failed plays can leave queued entries for skipped nodes. Only a
            // session with unfinished turns proves that work remains.
            && !self.ready_sessions.peek().is_some_and(|ready| {
                let session = &self.sessions[ready.session_index];
                session.next_turn_index < session.turns.len()
            })
            && self
                .sessions
                .iter()
                .all(|session| session.next_turn_index >= session.turns.len())
    }

    pub fn total_turns(&self) -> usize {
        self.sessions
            .iter()
            .map(|session| session.turns.len())
            .sum()
    }

    pub fn agentic_trajectory_snapshot(&self) -> Option<AgenticTrajectorySnapshot> {
        match &self.policy {
            SchedulingPolicy::Agentic(state) => Some(state.trajectory_snapshot()),
            SchedulingPolicy::Trace | SchedulingPolicy::Concurrency(_) => None,
        }
    }

    pub fn agentic_graph_identity(&self) -> Option<AgenticGraphIdentity> {
        self.agentic_graph_identity.clone()
    }

    pub fn agentic_snapshot_evidence(&self) -> Option<&[AgenticSnapshotEvidence]> {
        self.agentic_snapshots.as_deref()
    }

    pub fn agentic_replay_context(&self) -> Option<&Arc<AgenticReplayContext>> {
        self.agentic_replay_context.as_ref()
    }

    pub fn agentic_play_outcomes(&self) -> Option<Vec<AgenticPlayOutcome>> {
        let SchedulingPolicy::Agentic(state) = &self.policy else {
            return None;
        };
        Some(state.play_outcomes(&self.sessions))
    }

    pub fn agentic_lifecycle_transcript(&self) -> Option<AgenticLifecycleTranscript> {
        let SchedulingPolicy::Agentic(state) = &self.policy else {
            return None;
        };
        Some(AgenticLifecycleTranscript {
            events: state
                .lifecycle
                .iter()
                .map(|record| AgenticLifecycleEvent {
                    schema: AGENTIC_LIFECYCLE_SCHEMA_V1,
                    ordinal: record.ordinal,
                    at_ms: record.at_ms,
                    play_id: state.plays[record.play_index].play_id.clone(),
                    request_id: record.node_index.map(|node_index| {
                        self.sessions[node_index].turns[0]
                            .request_id
                            .clone()
                            .expect("agentic node must retain its authored request ID")
                    }),
                    event: record.event,
                    status: record.status,
                })
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::loadgen::{
        AgenticDependency, AgenticDependencyRelation, AgenticDependencyTrigger, AgenticNode,
        AgenticPlay, AgenticSourceProvenance, AgenticTrace, SessionTrace, Trace, TurnTrace,
    };

    fn agentic_node(
        request_id: &str,
        play_id: &str,
        not_before_ms: f64,
        dependencies: Vec<AgenticDependency>,
    ) -> AgenticNode {
        AgenticNode {
            request_id: request_id.into(),
            play_id: play_id.into(),
            session_id: play_id.into(),
            input_length: 2,
            max_output_tokens: 1,
            hash_ids: vec![1, 2],
            not_before_ms,
            dependencies,
            ..Default::default()
        }
    }

    fn agentic_trace(nodes: Vec<AgenticNode>) -> AgenticTrace {
        let mut play_nodes: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        for (node_index, node) in nodes.iter().enumerate() {
            play_nodes
                .entry(node.play_id.clone())
                .or_default()
                .push(node_index);
        }
        let mut plays = play_nodes
            .into_iter()
            .map(|(play_id, node_indices)| AgenticPlay {
                root_nodes: node_indices
                    .iter()
                    .copied()
                    .filter(|node_index| nodes[*node_index].dependencies.is_empty())
                    .collect(),
                play_id,
                source_play_ordinal: None,
                nodes: node_indices,
            })
            .collect::<Vec<_>>();
        plays.sort_by(|left, right| left.play_id.cmp(&right.play_id));
        AgenticTrace {
            block_size: 1,
            source: AgenticSourceProvenance {
                format: "test".into(),
                digest: "fixture".into(),
            },
            graph_digest: "fixture".into(),
            nodes,
            plays,
        }
    }

    fn dependency(
        request_id: &str,
        trigger: AgenticDependencyTrigger,
        delay_ms: f64,
        relation: AgenticDependencyRelation,
    ) -> AgenticDependency {
        AgenticDependency {
            request_id: request_id.into(),
            trigger,
            delay_ms,
            relation,
        }
    }

    fn assert_deterministic_output_plan(
        mut first_driver: WorkloadDriver,
        mut second_driver: WorkloadDriver,
        expected_len: usize,
    ) {
        let first = first_driver.pop_ready(0.0, usize::MAX);
        let second = second_driver.pop_ready(0.0, usize::MAX);

        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        assert_eq!(
            first[0].request.output_token_ids,
            second[0].request.output_token_ids
        );
        assert_eq!(
            first[0].request.output_token_ids.as_ref().map(Vec::len),
            Some(expected_len)
        );
    }

    #[test]
    fn hash_free_admission_preserves_request_without_router_metadata() {
        let trace = Trace {
            block_size: 2,
            sessions: vec![SessionTrace {
                session_id: "a".into(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![TurnTrace {
                    input_length: 4,
                    max_output_tokens: 1,
                    hash_ids: vec![10, 11],
                    ..Default::default()
                }],
            }],
        };
        let mut with_hashes = WorkloadDriver::new_trace(trace.clone(), 2).unwrap();
        let mut without_hashes =
            WorkloadDriver::new_trace_without_replay_hashes(trace, 2, false).unwrap();

        let with_hashes = with_hashes.pop_ready(0.0, 1).pop().unwrap();
        let without_hashes = without_hashes.pop_ready(0.0, 1).pop().unwrap();

        assert!(with_hashes.replay_hashes.is_some());
        assert!(without_hashes.replay_hashes.is_none());
        assert_eq!(without_hashes.request.tokens, with_hashes.request.tokens);
        assert_eq!(
            without_hashes.request.output_token_ids,
            with_hashes.request.output_token_ids
        );
    }

    fn two_session_trace() -> Trace {
        Trace {
            block_size: 1,
            sessions: vec![
                SessionTrace {
                    session_id: "a".into(),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: vec![
                        TurnTrace {
                            input_length: 2,
                            max_output_tokens: 1,
                            hash_ids: vec![1, 2],
                            delay_after_previous_ms: 0.0,
                            ..Default::default()
                        },
                        TurnTrace {
                            input_length: 2,
                            max_output_tokens: 1,
                            hash_ids: vec![3, 4],
                            delay_after_previous_ms: 5.0,
                            ..Default::default()
                        },
                    ],
                },
                SessionTrace {
                    session_id: "b".into(),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: vec![TurnTrace {
                        input_length: 2,
                        max_output_tokens: 1,
                        hash_ids: vec![5, 6],
                        delay_after_previous_ms: 0.0,
                        ..Default::default()
                    }],
                },
            ],
        }
    }

    fn belady_flat_trace() -> Trace {
        Trace {
            block_size: 5,
            sessions: [10.0, 1.0, 1.0]
                .into_iter()
                .enumerate()
                .map(|(index, arrival)| SessionTrace {
                    session_id: format!("s{index}"),
                    first_arrival_timestamp_ms: Some(arrival),
                    turns: vec![TurnTrace {
                        input_length: 7 + index,
                        max_output_tokens: 3,
                        hash_ids: vec![11, 20 + index as u32],
                        ..Default::default()
                    }],
                })
                .collect(),
        }
    }

    #[test]
    fn belady_forecast_preserves_causal_driver_and_actual_block_identities() {
        let trace = belady_flat_trace();
        let mut baseline = WorkloadDriver::new_trace(trace.clone(), 3)
            .unwrap()
            .with_deterministic_request_ids(1);
        let mut prepared = WorkloadDriver::new_trace(trace, 3)
            .unwrap()
            .with_deterministic_request_ids(1);
        let ready_before = prepared.ready_sessions.clone().into_sorted_vec();
        let forecast = prepared.prepare_belady_requests(3).unwrap();
        assert_eq!(
            forecast.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            [2, 3, 1].map(Uuid::from_u128)
        );
        assert_eq!(forecast, prepared.prepare_belady_requests(3).unwrap());
        assert_eq!(
            ready_before,
            prepared.ready_sessions.clone().into_sorted_vec()
        );
        assert!(prepared.in_flight.is_empty());
        assert!(prepared.sessions.iter().all(|session| {
            session.next_turn_index == 0
                && session.in_flight.is_none()
                && matches!(
                    session.turns[0].prompt_tokens,
                    PromptTokens::Deferred { .. }
                )
        }));
        assert!(prepared.pop_ready(0.0, usize::MAX).is_empty());
        let mut observed = Vec::new();
        for at_ms in [1.0, 10.0] {
            let expected = baseline.pop_ready(at_ms, usize::MAX);
            let actual = prepared.pop_ready(at_ms, usize::MAX);
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert_eq!(
                    serde_json::to_value(&actual.request).unwrap(),
                    serde_json::to_value(&expected.request).unwrap()
                );
                assert_eq!(actual.replay_hashes, expected.replay_hashes);
                observed.push((
                    actual.request_uuid,
                    actual.replay_hashes.unwrap().sequence_hashes,
                ));
                prepared.on_complete(actual.request_uuid, at_ms).unwrap();
                baseline.on_complete(expected.request_uuid, at_ms).unwrap();
            }
        }
        assert_eq!(forecast, observed);
        assert!(prepared.is_drained());
    }

    #[test]
    fn belady_forecast_reserves_opaque_ids_without_changing_output_plans() {
        let mut driver =
            WorkloadDriver::new_trace_without_replay_hashes(belady_flat_trace(), 3, false).unwrap();
        let output_plans = driver
            .sessions
            .iter()
            .map(|session| session.turns[0].output_token_ids.clone())
            .collect::<Vec<_>>();
        let forecast = driver.prepare_belady_requests(3).unwrap();
        assert_eq!(forecast, driver.prepare_belady_requests(3).unwrap());
        assert_eq!(
            output_plans,
            driver
                .sessions
                .iter()
                .map(|session| session.turns[0].output_token_ids.clone())
                .collect::<Vec<_>>()
        );
        let first = driver.pop_ready(1.0, 1).pop().unwrap();
        assert_eq!(first.request_uuid, forecast[0].0);
        assert_eq!(
            input_sequence_hashes(&first.request.tokens, 3),
            forecast[0].1
        );
        assert!(first.replay_hashes.is_none());
    }

    #[test]
    fn belady_forecast_rejects_nonstatic_or_already_started_drivers() {
        let mut multi = WorkloadDriver::new_trace(two_session_trace(), 1).unwrap();
        assert!(multi.prepare_belady_requests(1).is_err());
        let trace = belady_flat_trace();
        let mut delta = WorkloadDriver::new_trace_accumulating_deltas(trace.clone(), 3).unwrap();
        assert!(delta.prepare_belady_requests(3).is_err());
        let mut closed = WorkloadDriver::new_concurrency(trace.clone(), 3, 1).unwrap();
        assert!(closed.prepare_belady_requests(3).is_err());
        let mut driver = WorkloadDriver::new_trace(trace, 3).unwrap();
        assert!(driver.prepare_belady_requests(4).is_err());
        assert_eq!(driver.pop_ready(1.0, 1).len(), 1);
        assert!(driver.prepare_belady_requests(3).is_err());
        let mut agentic = WorkloadDriver::new_agentic_trace(
            agentic_trace(vec![agentic_node("a", "p", 0.0, Vec::new())]),
            1,
        )
        .unwrap();
        assert!(agentic.prepare_belady_requests(1).is_err());
    }

    /// A: 2 turns (turn-1 has a 5ms think-time). B, C: 1 turn each. Used for the cap>1
    /// transition / cancellation tests (w/ a third session pending behind a cap of 2).
    fn three_session_trace() -> Trace {
        let mut trace = two_session_trace();
        trace.sessions.push(SessionTrace {
            session_id: "c".into(),
            first_arrival_timestamp_ms: Some(0.0),
            turns: vec![TurnTrace {
                input_length: 2,
                max_output_tokens: 1,
                hash_ids: vec![7, 8],
                delay_after_previous_ms: 0.0,
                ..Default::default()
            }],
        });
        trace
    }

    #[test]
    fn full_prompts_remain_deferred_until_dispatch() {
        let mut driver = WorkloadDriver::new_trace(two_session_trace(), 1).unwrap();

        assert!(driver.sessions.iter().all(|session| {
            session
                .turns
                .iter()
                .all(|turn| matches!(turn.prompt_tokens, PromptTokens::Deferred { .. }))
        }));

        let ready = driver.pop_ready(0.0, 1);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].request.tokens, vec![1, 2]);
        assert!(ready[0].replay_hashes.is_some());
    }

    #[test]
    fn compact_dispatch_does_not_retain_materialized_prompt() {
        let mut driver = WorkloadDriver::new_trace(two_session_trace(), 1).unwrap();

        let mut ready = driver.pop_ready_compact(0.0, 1);

        assert_eq!(ready.len(), 1);
        let request = ready.pop().expect("one compact request").request;
        assert_eq!(request.input_length(), 2);
        assert!(request.metadata().tokens.is_empty());
        assert!(request.materialized_tokens().is_none());
        assert_eq!(request.into_direct_request().tokens, vec![1, 2]);
    }

    #[test]
    fn delta_cumulative_prompts_remain_materialized_during_setup() {
        let driver =
            WorkloadDriver::new_concurrency_accumulating_deltas(two_session_trace(), 1, 1).unwrap();

        assert!(driver.sessions.iter().all(|session| {
            session
                .turns
                .iter()
                .all(|turn| matches!(turn.prompt_tokens, PromptTokens::Materialized(_)))
        }));
    }

    #[test]
    fn deferred_prompt_validation_preserves_setup_errors() {
        let trace = Trace {
            block_size: 4,
            sessions: vec![SessionTrace {
                session_id: "invalid".into(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![TurnTrace {
                    input_length: 5,
                    max_output_tokens: 1,
                    hash_ids: vec![1],
                    ..Default::default()
                }],
            }],
        };

        let error = WorkloadDriver::new_trace(trace, 4).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("input_length 5 exceeds synthesized capacity 4")
        );
    }

    #[test]
    fn unknown_completion_preserves_in_flight_state() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();
        let admitted = driver.pop_ready(0.0, usize::MAX);
        let request_uuid = admitted[0].request_uuid;
        let session_index = driver.in_flight[&request_uuid].session_index;

        let error = driver.on_complete(Uuid::new_v4(), 1.0).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("unknown workload request completion")
        );
        assert!(driver.in_flight.contains_key(&request_uuid));
        assert_eq!(driver.sessions[session_index].in_flight, Some(request_uuid));
    }

    #[test]
    fn unknown_cancellation_is_noop() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();
        let admitted = driver.pop_ready(0.0, usize::MAX);
        let request_uuid = admitted[0].request_uuid;
        let session_index = driver.in_flight[&request_uuid].session_index;

        driver.release_cap_slot(Uuid::new_v4(), 1.0);

        assert!(driver.in_flight.contains_key(&request_uuid));
        assert_eq!(driver.sessions[session_index].in_flight, Some(request_uuid));
    }

    #[test]
    fn inconsistent_session_mapping_preserves_in_flight_entry() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();
        let admitted = driver.pop_ready(0.0, usize::MAX);
        let request_uuid = admitted[0].request_uuid;
        let session_index = driver.in_flight[&request_uuid].session_index;
        driver.sessions[session_index].in_flight = Some(Uuid::new_v4());

        let error = driver.on_complete(request_uuid, 1.0).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("does not match in-flight request")
        );
        assert!(driver.in_flight.contains_key(&request_uuid));
        assert_eq!(driver.sessions[session_index].next_turn_index, 0);
    }

    #[test]
    fn cap_clamps_pop_ready_when_limit_is_unbounded() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(first.len(), 1);
        let second = driver.pop_ready(0.0, usize::MAX);
        assert!(
            second.is_empty(),
            "cap should block dispatch while slot is held"
        );
    }

    #[test]
    fn pop_ready_admits_next_turn_after_on_complete() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();

        let admitted = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(admitted.len(), 1);
        let uuid = admitted[0].request_uuid;
        driver.on_complete(uuid, 10.0).unwrap();

        // next admitted turn is *this* session's turn-1
        // (ready at completion 10 + think-time 5 = 15)
        let next = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].turn_index, 1);
        assert_ne!(next[0].request_uuid, uuid);
    }

    #[test]
    fn concurrency_is_depth_first_holding_slot_across_think_time() {
        // Session A: 2 turns (turn-1 has a 5ms think-time). Session B: 1 turn. cap = 1.
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();

        // A.turn0 admitted; B is pending (not activated — cap is 1).
        let a0 = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(a0.len(), 1);
        assert_eq!(a0[0].turn_index, 0);
        let a0_uuid = a0[0].request_uuid;
        driver.on_complete(a0_uuid, 10.0).unwrap();

        // During A's think-time (turn-1 ready at 10+5=15), B must NOT slip in: A holds the slot.
        assert!(
            driver.pop_ready(10.0, usize::MAX).is_empty(),
            "B must not be admitted while A holds its slot in think-time"
        );

        // A.turn1 dispatches before B ever starts (depth-first).
        let a1 = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(a1.len(), 1);
        assert_eq!(a1[0].turn_index, 1);
        driver.on_complete(a1[0].request_uuid, 20.0).unwrap();

        // Only now that A is fully done is B activated.
        let b0 = driver.pop_ready(20.0, usize::MAX);
        assert_eq!(b0.len(), 1);
        assert_eq!(b0[0].turn_index, 0);
        assert_ne!(b0[0].request_uuid, a0_uuid);
        assert!(!driver.is_drained(), "B still in flight");
        driver.on_complete(b0[0].request_uuid, 30.0).unwrap();
        assert!(driver.is_drained());
    }

    #[test]
    fn concurrency_cap2_admits_pending_when_active_session_finishes() {
        // cap = 2: A (2 turns) and B (1 turn) start active; C (1 turn) is pending.
        let mut driver = WorkloadDriver::new_concurrency(three_session_trace(), 1, 2).unwrap();

        // Initial cohort: A.t0 and B.t0 (the cap-2 set); C stays pending.
        let first = driver.pop_ready(0.0, usize::MAX);
        let mut ids: Vec<&str> = first.iter().map(|r| r.session_id.as_str()).collect();
        ids.sort();
        assert_eq!(
            ids,
            vec!["a", "b"],
            "cap-2 admits exactly A and B; C pending"
        );
        let a0 = first
            .iter()
            .find(|r| r.session_id == "a")
            .unwrap()
            .request_uuid;
        let b0 = first
            .iter()
            .find(|r| r.session_id == "b")
            .unwrap()
            .request_uuid;

        // A finishes turn-0 → enters think-time (A.t1 ready at 10+5=15); A keeps its slot.
        driver.on_complete(a0, 10.0).unwrap();
        // B finishes its only turn → frees a slot → C is activated.
        driver.on_complete(b0, 10.0).unwrap();

        // At t=10 only C is admittable (its freed slot); A is mid-think-time and retains
        // its slot — neither dropped nor re-admitted early.
        let at_10 = driver.pop_ready(10.0, usize::MAX);
        assert_eq!(at_10.len(), 1, "only C is admittable at t=10");
        assert_eq!(at_10[0].session_id, "c");
        assert_eq!(at_10[0].turn_index, 0);

        // A's retained slot resumes once its think-time elapses (t=15), proving it was
        // never evicted by C's admission.
        let at_15 = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(at_15.len(), 1);
        assert_eq!(
            (at_15[0].session_id.as_str(), at_15[0].turn_index),
            ("a", 1)
        );
    }

    #[test]
    fn release_cap_slot_terminates_inflight_session_and_admits_pending() {
        // Mirrors an online InFlightGuard drop (cancellation), which calls release_cap_slot.
        // cap = 2: A (2 turns) + B (1 turn) active, C (1 turn) pending. A is in think-time,
        // B is in flight and gets cancelled.
        let mut driver = WorkloadDriver::new_concurrency(three_session_trace(), 1, 2).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        let a0 = first
            .iter()
            .find(|r| r.session_id == "a")
            .unwrap()
            .request_uuid;
        let b0 = first
            .iter()
            .find(|r| r.session_id == "b")
            .unwrap()
            .request_uuid;

        // A → think-time (A.t1 ready at 15), retains its slot.
        driver.on_complete(a0, 10.0).unwrap();
        // B cancelled in flight: the online guard drop releases B's slot and terminates it.
        driver.release_cap_slot(b0, 10.0);

        // B's freed slot admits C; A's continuation is untouched.
        let at_10 = driver.pop_ready(10.0, usize::MAX);
        assert_eq!(at_10.len(), 1);
        assert_eq!(
            at_10[0].session_id, "c",
            "C admitted into the slot freed by B's cancellation"
        );
        driver.on_complete(at_10[0].request_uuid, 12.0).unwrap();

        // A's continuation survived the cancellation and resumes after its think-time.
        let a1 = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(a1.len(), 1);
        assert_eq!((a1[0].session_id.as_str(), a1[0].turn_index), ("a", 1));
        driver.on_complete(a1[0].request_uuid, 20.0).unwrap();

        // A (2 turns), B (cancelled/terminated), C (1 turn) all resolved → drained.
        assert!(driver.is_drained());
    }

    #[test]
    fn next_ready_time_ms_returns_none_at_cap() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();

        let admitted = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(admitted.len(), 1);

        assert!(
            driver.next_ready_time_ms().is_none(),
            "expected None while at cap even with ready sessions queued"
        );

        driver.on_complete(admitted[0].request_uuid, 10.0).unwrap();
        assert!(
            driver.next_ready_time_ms().is_some(),
            "expected readiness after a slot is freed"
        );
    }

    #[test]
    fn uncapped_concurrency_admits_all_sessions_up_to_caller_limit() {
        // usize::MAX cap == effectively uncapped: every session is activated, so the
        // caller's pop_ready limit is the only bound.
        let mut driver =
            WorkloadDriver::new_concurrency(two_session_trace(), 1, usize::MAX).unwrap();

        let admitted = driver.pop_ready(0.0, 5);
        assert_eq!(
            admitted.len(),
            2,
            "both sessions should admit when uncapped"
        );
        assert!(driver.next_ready_time_ms().is_none());
    }

    #[test]
    fn release_cap_slot_is_noop_after_on_complete() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();

        let admitted = driver.pop_ready(0.0, usize::MAX);
        let uuid = admitted[0].request_uuid;
        driver.on_complete(uuid, 5.0).unwrap();

        // release_cap_slot after on_complete is a no-op (the in-flight entry is already
        // gone), so it must NOT double-decrement active_sessions. The session still holds its
        // slot for turn-1 (ready at 5 + think-time 5 = 10)
        driver.release_cap_slot(uuid, 5.0);

        let next = driver.pop_ready(10.0, usize::MAX);
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].turn_index, 1);
        assert_ne!(next[0].request_uuid, uuid);
    }

    #[test]
    fn release_cap_slot_recovers_cap_when_on_complete_was_skipped() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();

        let admitted = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(admitted.len(), 1);

        driver.release_cap_slot(admitted[0].request_uuid, 0.0);

        let next = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(
            next.len(),
            1,
            "cap slot should be available after release_cap_slot"
        );
    }

    #[test]
    fn release_cap_slot_terminates_session_so_is_drained_completes() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();

        let admitted = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(admitted.len(), 1);
        let stuck_uuid = admitted[0].request_uuid;

        driver.release_cap_slot(stuck_uuid, 0.0);

        let neighbor = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(
            neighbor.len(),
            1,
            "other session must still be admissible after its neighbor was terminated"
        );
        driver.on_complete(neighbor[0].request_uuid, 1.0).unwrap();

        assert!(
            driver.is_drained(),
            "is_drained must become true so run_workload can exit"
        );
    }

    #[test]
    fn full_prompt_modes_plan_missing_output_token_ids_deterministically() {
        let trace = Trace {
            block_size: 1,
            sessions: vec![SessionTrace {
                session_id: "a".into(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![TurnTrace {
                    input_length: 2,
                    max_output_tokens: 3,
                    hash_ids: vec![10, 11],
                    ..Default::default()
                }],
            }],
        };
        assert_deterministic_output_plan(
            WorkloadDriver::new_trace(trace.clone(), 1).unwrap(),
            WorkloadDriver::new_trace(trace, 1).unwrap(),
            3,
        );

        let mut node = agentic_node("r1", "a", 0.0, Vec::new());
        node.max_output_tokens = 3;
        node.hash_ids = vec![10, 11];
        let trace = agentic_trace(vec![node]);
        assert_deterministic_output_plan(
            WorkloadDriver::new_agentic_trace(trace.clone(), 1).unwrap(),
            WorkloadDriver::new_agentic_trace(trace, 1).unwrap(),
            3,
        );
    }

    #[test]
    fn accumulating_delta_mode_includes_previous_output_tokens() {
        let trace = Trace {
            block_size: 4,
            sessions: vec![SessionTrace {
                session_id: "a".into(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    TurnTrace {
                        input_length: 6,
                        max_output_tokens: 2,
                        output_token_ids: Some(vec![20, 21]),
                        replay_key: None,
                        hash_ids: vec![10, 11],
                        delay_after_previous_ms: 0.0,
                        priority: 3,
                        strict_priority: 4,
                        policy_class: None,
                    },
                    TurnTrace {
                        input_length: 3,
                        max_output_tokens: 1,
                        output_token_ids: None,
                        replay_key: None,
                        hash_ids: vec![12],
                        delay_after_previous_ms: 5.0,
                        priority: -2,
                        strict_priority: 7,
                        policy_class: None,
                    },
                ],
            }],
        };
        let mut driver = WorkloadDriver::new_concurrency_accumulating_deltas(trace, 4, 1).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].request.tokens, vec![10, 10, 10, 10, 11, 11]);
        assert_eq!(first[0].request.output_token_ids, Some(vec![20, 21]));
        assert_eq!(first[0].request.priority, 3);
        assert_eq!(first[0].request.strict_priority, 4);
        driver.on_output_token(first[0].request_uuid, 20).unwrap();
        driver.on_output_token(first[0].request_uuid, 21).unwrap();
        driver.on_complete(first[0].request_uuid, 10.0).unwrap();

        let second = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(second.len(), 1);
        assert_eq!(
            second[0].request.tokens,
            vec![10, 10, 10, 10, 11, 11, 20, 21, 12, 12, 12]
        );
        assert_eq!(second[0].request.priority, -2);
        assert_eq!(second[0].request.strict_priority, 7);
    }

    #[test]
    fn accumulating_delta_mode_plans_missing_output_token_ids() {
        let trace = Trace {
            block_size: 1,
            sessions: vec![SessionTrace {
                session_id: "a".into(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    TurnTrace {
                        input_length: 2,
                        max_output_tokens: 3,
                        hash_ids: vec![10, 11],
                        ..Default::default()
                    },
                    TurnTrace {
                        input_length: 1,
                        max_output_tokens: 1,
                        hash_ids: vec![12],
                        ..Default::default()
                    },
                ],
            }],
        };
        let mut driver = WorkloadDriver::new_concurrency_accumulating_deltas(trace, 1, 1).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(first.len(), 1);
        let planned_output = first[0]
            .request
            .output_token_ids
            .clone()
            .expect("delta replay should plan synthetic outputs");
        assert_eq!(planned_output.len(), 3);
        for &token_id in &planned_output {
            driver
                .on_output_token(first[0].request_uuid, token_id)
                .unwrap();
        }
        driver.on_complete(first[0].request_uuid, 1.0).unwrap();

        let second = driver.pop_ready(1.0, usize::MAX);
        assert_eq!(second.len(), 1);
        let mut expected = vec![10, 11];
        expected.extend(planned_output);
        expected.push(12);
        assert_eq!(second[0].request.tokens, expected);
        assert_eq!(
            second[0].request.output_token_ids.as_ref().map(Vec::len),
            Some(1)
        );
    }

    #[test]
    fn accumulating_delta_mode_appends_only_emitted_output_tokens() {
        let trace = Trace {
            block_size: 1,
            sessions: vec![SessionTrace {
                session_id: "a".into(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    TurnTrace {
                        input_length: 1,
                        max_output_tokens: 3,
                        output_token_ids: Some(vec![20, 21, 22]),
                        hash_ids: vec![10],
                        ..Default::default()
                    },
                    TurnTrace {
                        input_length: 1,
                        max_output_tokens: 1,
                        hash_ids: vec![12],
                        ..Default::default()
                    },
                ],
            }],
        };
        let mut driver = WorkloadDriver::new_concurrency_accumulating_deltas(trace, 1, 1).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        driver.on_output_token(first[0].request_uuid, 20).unwrap();
        driver.on_output_token(first[0].request_uuid, 21).unwrap();
        driver.on_complete(first[0].request_uuid, 1.0).unwrap();

        let second = driver.pop_ready(1.0, usize::MAX);
        assert_eq!(second[0].request.tokens, vec![10, 20, 21, 12]);
    }

    #[test]
    fn accumulating_delta_mode_does_not_append_rejected_output_tokens() {
        let trace = Trace {
            block_size: 1,
            sessions: vec![SessionTrace {
                session_id: "a".into(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    TurnTrace {
                        input_length: 1,
                        max_output_tokens: 2,
                        output_token_ids: Some(vec![20, 21]),
                        hash_ids: vec![10],
                        ..Default::default()
                    },
                    TurnTrace {
                        input_length: 1,
                        max_output_tokens: 1,
                        hash_ids: vec![12],
                        ..Default::default()
                    },
                ],
            }],
        };
        let mut driver = WorkloadDriver::new_concurrency_accumulating_deltas(trace, 1, 1).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        driver
            .on_terminal(first[0].request_uuid, 1.0, ReplayTerminalStatus::Rejected)
            .unwrap();

        let second = driver.pop_ready(1.0, usize::MAX);
        assert_eq!(second[0].request.tokens, vec![10, 12]);
    }

    #[test]
    fn ending_one_delta_session_preserves_another_sessions_partial_output_history() {
        for terminal in [ReplayTerminalStatus::Canceled, ReplayTerminalStatus::Failed] {
            let trace = Trace {
                block_size: 1,
                sessions: [10, 30]
                    .into_iter()
                    .map(|token| SessionTrace {
                        session_id: token.to_string(),
                        first_arrival_timestamp_ms: Some(0.0),
                        turns: vec![
                            TurnTrace {
                                input_length: 1,
                                max_output_tokens: 2,
                                hash_ids: vec![token],
                                output_token_ids: Some(vec![token + 1, token + 2]),
                                ..Default::default()
                            },
                            TurnTrace {
                                input_length: 1,
                                max_output_tokens: 0,
                                hash_ids: vec![token + 3],
                                delay_after_previous_ms: 5.0,
                                ..Default::default()
                            },
                        ],
                    })
                    .collect(),
            };
            let mut driver = WorkloadDriver::new_trace_accumulating_deltas(trace, 1).unwrap();
            let first = driver.pop_ready(0.0, usize::MAX);
            assert_eq!(first.len(), 2);
            let ending = first
                .iter()
                .find(|turn| turn.request.tokens == [10])
                .unwrap();
            let surviving = first
                .iter()
                .find(|turn| turn.request.tokens == [30])
                .unwrap();
            driver.on_output_token(ending.request_uuid, 11).unwrap();
            driver.on_output_token(surviving.request_uuid, 31).unwrap();
            driver
                .on_terminal(ending.request_uuid, 1.0, terminal)
                .unwrap();
            let ended_session = driver
                .sessions
                .iter()
                .find(|s| s.session_id == "10")
                .unwrap();
            assert_eq!(ended_session.cumulative_tokens.capacity(), 0);
            driver.on_complete(surviving.request_uuid, 2.0).unwrap();
            assert!(!driver.is_drained());
            assert_eq!(driver.next_ready_time_ms(), Some(7.0));
            assert!(driver.pop_ready(6.0, usize::MAX).is_empty());

            let last = driver.pop_ready(7.0, usize::MAX);
            assert_eq!(last.len(), 1);
            assert_eq!(last[0].request.tokens, vec![30, 31, 33]);
            driver.on_complete(last[0].request_uuid, 8.0).unwrap();
            assert!(driver.is_drained());
            assert!(
                driver
                    .sessions
                    .iter()
                    .all(|s| s.cumulative_tokens.capacity() == 0)
            );
            assert!(driver.pop_ready(1_000.0, usize::MAX).is_empty());
        }
    }

    #[test]
    fn agentic_mode_releases_turn_after_dependency_completion_plus_delay() {
        let trace = agentic_trace(vec![
            agentic_node("r1", "play", 0.0, Vec::new()),
            agentic_node(
                "r2",
                "play",
                12.0,
                vec![dependency(
                    "r1",
                    AgenticDependencyTrigger::Completion,
                    5.0,
                    AgenticDependencyRelation::Sequence,
                )],
            ),
        ]);
        let mut driver = WorkloadDriver::new_agentic_trace(trace, 1).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].scheduled_ready_at_ms, 0.0);
        assert!(driver.pop_ready(14.0, usize::MAX).is_empty());

        driver.on_complete(first[0].request_uuid, 10.0).unwrap();
        assert_eq!(driver.next_ready_time_ms(), Some(15.0));
        assert!(driver.pop_ready(14.0, usize::MAX).is_empty());
        let second = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].scheduled_ready_at_ms, 15.0);
    }

    #[test]
    fn agentic_dispatch_child_follows_parent_emission_in_the_same_pass() {
        let trace = agentic_trace(vec![
            agentic_node("parent", "play", 0.0, Vec::new()),
            agentic_node(
                "child",
                "play",
                0.0,
                vec![dependency(
                    "parent",
                    AgenticDependencyTrigger::Dispatch,
                    0.0,
                    AgenticDependencyRelation::Spawn,
                )],
            ),
        ]);
        let mut driver = WorkloadDriver::new_agentic_trace(trace, 1).unwrap();

        let emitted = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(emitted.len(), 2);
        assert_eq!(emitted[0].authored_request_id.as_deref(), Some("parent"));
        assert_eq!(emitted[1].authored_request_id.as_deref(), Some("child"));
    }

    #[test]
    fn agentic_request_ids_are_stable_graph_ordinals() {
        let trace = agentic_trace(vec![
            agentic_node("parent", "play", 0.0, Vec::new()),
            agentic_node(
                "child",
                "play",
                0.0,
                vec![dependency(
                    "parent",
                    AgenticDependencyTrigger::Dispatch,
                    0.0,
                    AgenticDependencyRelation::Spawn,
                )],
            ),
        ]);

        let ids = (0..2)
            .map(|_| {
                WorkloadDriver::new_agentic_trace(trace.clone(), 1)
                    .unwrap()
                    .pop_ready(0.0, usize::MAX)
                    .into_iter()
                    .map(|turn| turn.request_uuid)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        assert_eq!(ids[0], ids[1]);
        assert_eq!(ids[0], vec![Uuid::from_u128(1), Uuid::from_u128(2)]);
    }

    #[test]
    fn agentic_dispatch_carries_the_stable_identity_envelope() {
        let trace = agentic_trace(vec![agentic_node("root", "play-a", 0.0, Vec::new())]);
        let mut driver = WorkloadDriver::new_agentic_trace_with_lanes(trace, 1, 1).unwrap();

        let ready = driver.pop_ready(0.0, 1).pop().unwrap();
        let context = ready.request.replay_context.as_ref().unwrap();
        let identity = context.agentic.as_ref().unwrap();

        assert_eq!(context.authored_id, "root");
        assert_eq!(identity.request_id, "root");
        assert_eq!(identity.play_id, "play-a");
        assert_eq!(identity.conversation_id, "play-a");
        assert_eq!(identity.lane_id.as_deref(), Some("lane:0"));
        assert_eq!(identity.root_id.as_deref(), Some("root"));
        assert_eq!(identity.parent_id, None);
    }

    #[test]
    fn same_timestamp_feedback_is_canonicalized_by_graph_ordinal() {
        let trace = agentic_trace(vec![
            agentic_node("a", "play", 0.0, Vec::new()),
            agentic_node("b", "play", 0.0, Vec::new()),
        ]);

        let run = |reverse: bool| {
            let mut driver = WorkloadDriver::new_agentic_trace(trace.clone(), 1).unwrap();
            let ready = driver.pop_ready(0.0, usize::MAX);
            let mut terminals = vec![
                AgenticTerminalFeedback {
                    request_uuid: ready[0].request_uuid,
                    status: ReplayTerminalStatus::Completed,
                },
                AgenticTerminalFeedback {
                    request_uuid: ready[1].request_uuid,
                    status: ReplayTerminalStatus::Failed,
                },
            ];
            let mut quiescent = ready
                .iter()
                .map(|turn| turn.request_uuid)
                .collect::<Vec<_>>();
            if reverse {
                terminals.reverse();
                quiescent.reverse();
            }
            driver
                .apply_agentic_runtime_feedback(AgenticRuntimeFeedback {
                    at_ms: 10.0,
                    output_tokens: Vec::new(),
                    causal_terminals: terminals,
                    quiescent_requests: quiescent,
                })
                .unwrap();
            driver.agentic_lifecycle_transcript().unwrap()
        };

        let authored = run(false);
        let reversed = run(true);
        assert_eq!(authored.to_jsonl().unwrap(), reversed.to_jsonl().unwrap());
        assert_eq!(authored.digest().unwrap(), reversed.digest().unwrap());
        assert_eq!(
            authored
                .events
                .iter()
                .filter(|event| event.event == AgenticLifecycleEventKind::CausalTerminal)
                .map(|event| event.request_id.as_deref().unwrap())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn agentic_feedback_rejects_time_regression_without_advancing_dependencies_or_clock() {
        let trace = agentic_trace(vec![
            agentic_node("clock", "play", 0.0, Vec::new()),
            agentic_node("parent", "play", 0.0, Vec::new()),
            agentic_node(
                "child",
                "play",
                0.0,
                vec![dependency(
                    "parent",
                    AgenticDependencyTrigger::Completion,
                    2.0,
                    AgenticDependencyRelation::Sequence,
                )],
            ),
        ]);
        let mut driver = WorkloadDriver::new_agentic_trace(trace, 1).unwrap();
        let ready = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(ready.len(), 2);
        let clock = ready[0].request_uuid;
        let parent = ready[1].request_uuid;
        let completed = |request_uuid| AgenticTerminalFeedback {
            request_uuid,
            status: ReplayTerminalStatus::Completed,
        };
        driver
            .apply_agentic_runtime_feedback(AgenticRuntimeFeedback {
                at_ms: 10.0,
                causal_terminals: vec![completed(clock)],
                quiescent_requests: vec![clock],
                ..Default::default()
            })
            .unwrap();
        let transcript_before = driver.agentic_lifecycle_transcript().unwrap();

        let error = driver
            .apply_agentic_runtime_feedback(AgenticRuntimeFeedback {
                at_ms: 5.0,
                causal_terminals: vec![completed(parent)],
                quiescent_requests: vec![parent],
                ..Default::default()
            })
            .unwrap_err();
        assert!(error.to_string().contains("regressed from 10 ms to 5 ms"));
        assert_eq!(
            driver.agentic_lifecycle_transcript().unwrap(),
            transcript_before
        );
        assert_eq!(driver.next_ready_time_ms(), None);
        assert!(driver.in_flight.contains_key(&parent));
        assert!(!driver.agentic_settling.contains_key(&parent));

        // A known request with invalid lifecycle feedback reaches application
        // validation, but must not make the next valid t=10 batch look stale.
        let error = driver
            .apply_agentic_runtime_feedback(AgenticRuntimeFeedback {
                at_ms: 20.0,
                quiescent_requests: vec![parent],
                ..Default::default()
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("became quiescent before its causal terminal")
        );
        assert_eq!(
            driver.agentic_lifecycle_transcript().unwrap(),
            transcript_before
        );
        assert_eq!(driver.next_ready_time_ms(), None);

        // A second successful batch at the same timestamp remains legal.
        driver
            .apply_agentic_runtime_feedback(AgenticRuntimeFeedback {
                at_ms: 10.0,
                causal_terminals: vec![completed(parent)],
                ..Default::default()
            })
            .unwrap();
        assert_eq!(driver.next_ready_time_ms(), Some(12.0));
        assert!(driver.pop_ready(11.0, usize::MAX).is_empty());
        let child = driver.pop_ready(12.0, usize::MAX);
        assert_eq!(child.len(), 1);
        assert_eq!(child[0].authored_request_id.as_deref(), Some("child"));
        driver
            .apply_agentic_runtime_feedback(AgenticRuntimeFeedback {
                at_ms: 12.0,
                causal_terminals: vec![completed(child[0].request_uuid)],
                quiescent_requests: vec![child[0].request_uuid],
                ..Default::default()
            })
            .unwrap();
        assert!(!driver.is_drained(), "parent cleanup is still pending");
        driver
            .apply_agentic_runtime_feedback(AgenticRuntimeFeedback {
                at_ms: 15.0,
                quiescent_requests: vec![parent],
                ..Default::default()
            })
            .unwrap();
        assert!(driver.is_drained());
        assert!(
            driver
                .agentic_lifecycle_transcript()
                .unwrap()
                .events
                .windows(2)
                .all(|events| events[0].at_ms <= events[1].at_ms)
        );
    }

    #[rstest::rstest]
    #[case::premature_quiescence(false)]
    #[case::duplicate_terminal(true)]
    fn agentic_rejected_mixed_feedback_preserves_state(#[case] duplicate_terminal: bool) {
        let trace = agentic_trace(vec![
            agentic_node("a-parent", "play", 0.0, Vec::new()),
            agentic_node("b-stale", "play", 0.0, Vec::new()),
            agentic_node("c-peer", "play", 0.0, Vec::new()),
            agentic_node(
                "d-child",
                "play",
                0.0,
                vec![dependency(
                    "a-parent",
                    AgenticDependencyTrigger::Completion,
                    2.0,
                    AgenticDependencyRelation::Sequence,
                )],
            ),
        ]);
        let mut driver = WorkloadDriver::new_agentic_trace_with_lanes(trace, 1, 1).unwrap();
        let ready = driver.pop_ready(0.0, usize::MAX);
        let parent = ready[0].request_uuid;
        let stale = ready[1].request_uuid;
        let peer = ready[2].request_uuid;
        let completed = |request_uuid| AgenticTerminalFeedback {
            request_uuid,
            status: ReplayTerminalStatus::Completed,
        };
        driver
            .apply_agentic_runtime_feedback(AgenticRuntimeFeedback {
                at_ms: 10.0,
                causal_terminals: vec![completed(stale)],
                ..Default::default()
            })
            .unwrap();
        let state_before = format!("{driver:?}");
        let mut invalid = AgenticRuntimeFeedback {
            at_ms: 20.0,
            causal_terminals: vec![completed(parent)],
            quiescent_requests: vec![parent, stale],
            ..Default::default()
        };
        if duplicate_terminal {
            invalid.causal_terminals.push(completed(stale));
        } else {
            invalid.quiescent_requests.push(peer);
        }

        assert!(driver.apply_agentic_runtime_feedback(invalid).is_err());
        // Include request/session state, dependency counters, lanes, cleanup,
        // output progress, lifecycle records, and the successful batch clock.
        assert_eq!(format!("{driver:?}"), state_before);
        assert_eq!(driver.next_ready_time_ms(), None);

        driver
            .apply_agentic_runtime_feedback(AgenticRuntimeFeedback {
                at_ms: 10.0,
                causal_terminals: vec![completed(parent), completed(peer)],
                quiescent_requests: vec![parent, stale, peer],
                ..Default::default()
            })
            .unwrap();
        assert_eq!(driver.next_ready_time_ms(), Some(12.0));
        let child = driver.pop_ready(12.0, 1).pop().unwrap();
        assert_eq!(child.authored_request_id.as_deref(), Some("d-child"));
        driver
            .apply_agentic_runtime_feedback(AgenticRuntimeFeedback {
                at_ms: 12.0,
                causal_terminals: vec![completed(child.request_uuid)],
                quiescent_requests: vec![child.request_uuid],
                ..Default::default()
            })
            .unwrap();
        assert!(driver.is_drained());
        assert!(
            driver
                .agentic_lifecycle_transcript()
                .unwrap()
                .events
                .windows(2)
                .all(|events| events[0].at_ms <= events[1].at_ms)
        );
    }

    #[test]
    fn same_timestamp_primary_failure_uses_graph_ordinal_not_status_severity() {
        let trace = agentic_trace(vec![
            agentic_node("first", "play", 0.0, Vec::new()),
            agentic_node("second", "play", 0.0, Vec::new()),
            agentic_node(
                "blocked-child",
                "play",
                0.0,
                vec![dependency(
                    "first",
                    AgenticDependencyTrigger::Completion,
                    0.0,
                    AgenticDependencyRelation::Sequence,
                )],
            ),
        ]);
        let mut driver = WorkloadDriver::new_agentic_trace(trace, 1).unwrap();
        let ready = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(ready.len(), 2);

        driver
            .apply_agentic_runtime_feedback(AgenticRuntimeFeedback {
                at_ms: 10.0,
                output_tokens: Vec::new(),
                causal_terminals: vec![
                    AgenticTerminalFeedback {
                        request_uuid: ready[1].request_uuid,
                        status: ReplayTerminalStatus::Rejected,
                    },
                    AgenticTerminalFeedback {
                        request_uuid: ready[0].request_uuid,
                        status: ReplayTerminalStatus::Failed,
                    },
                ],
                quiescent_requests: vec![ready[1].request_uuid, ready[0].request_uuid],
            })
            .unwrap();

        let outcome = driver.agentic_play_outcomes().unwrap().pop().unwrap();
        assert_eq!(outcome.status, AgenticPlayStatus::Failed);
        assert_eq!(outcome.failure_request_id.as_deref(), Some("first"));
        assert_eq!(outcome.failure_status, Some(ReplayTerminalStatus::Failed));
        assert_eq!(outcome.causal_terminal_ms, Some(10.0));
        assert_eq!(outcome.settled_at_ms, Some(10.0));
        assert!(
            driver
                .agentic_lifecycle_transcript()
                .unwrap()
                .events
                .iter()
                .any(|event| event.event == AgenticLifecycleEventKind::Skipped
                    && event.request_id.as_deref() == Some("blocked-child"))
        );
    }

    #[test]
    fn failed_play_drains_after_cleanup_without_waiting_for_queued_siblings() {
        let trace = agentic_trace(vec![
            agentic_node("root", "play", 0.0, Vec::new()),
            agentic_node("future-sibling", "play", 1_000_000.0, Vec::new()),
        ]);
        let mut driver = WorkloadDriver::new_agentic_trace(trace, 1).unwrap();
        let root = driver.pop_ready(0.0, usize::MAX).pop().unwrap();
        driver
            .on_causal_terminal(root.request_uuid, 1.0, ReplayTerminalStatus::Failed)
            .unwrap();
        assert!(!driver.is_drained(), "engine cleanup is still outstanding");
        driver.on_quiescent(root.request_uuid, 2.0).unwrap();
        // Check before dispatch/next_ready_time_ms can discard the skipped entry.
        assert!(driver.is_drained());
        let outcome = driver.agentic_play_outcomes().unwrap().pop().unwrap();
        assert_eq!(outcome.status, AgenticPlayStatus::Failed);
        assert_eq!(outcome.settled_at_ms, Some(2.0));
        assert!(driver.pop_ready(1_000_000.0, usize::MAX).is_empty());
    }

    #[test]
    fn every_non_success_terminal_fails_and_settles_the_play() {
        for status in [
            ReplayTerminalStatus::Rejected,
            ReplayTerminalStatus::Canceled,
            ReplayTerminalStatus::Failed,
        ] {
            let trace = agentic_trace(vec![
                agentic_node("root", "play", 0.0, Vec::new()),
                agentic_node(
                    "child",
                    "play",
                    0.0,
                    vec![dependency(
                        "root",
                        AgenticDependencyTrigger::Completion,
                        0.0,
                        AgenticDependencyRelation::Sequence,
                    )],
                ),
            ]);
            let mut driver = WorkloadDriver::new_agentic_trace(trace, 1).unwrap();
            assert_eq!(
                driver.agentic_play_outcomes().unwrap()[0].status,
                AgenticPlayStatus::Incomplete
            );
            let root = driver.pop_ready(0.0, 1).pop().unwrap();
            driver.on_terminal(root.request_uuid, 5.0, status).unwrap();

            let outcome = &driver.agentic_play_outcomes().unwrap()[0];
            assert_eq!(outcome.status, AgenticPlayStatus::Failed);
            assert_eq!(outcome.failure_request_id.as_deref(), Some("root"));
            assert_eq!(outcome.failure_status, Some(status));
            assert!(driver.is_drained());
        }
    }

    #[test]
    fn agentic_failed_parent_skips_unemitted_descendants() {
        let trace = agentic_trace(vec![
            agentic_node("parent", "play", 0.0, Vec::new()),
            agentic_node(
                "child",
                "play",
                0.0,
                vec![dependency(
                    "parent",
                    AgenticDependencyTrigger::Completion,
                    0.0,
                    AgenticDependencyRelation::Sequence,
                )],
            ),
        ]);
        let mut driver = WorkloadDriver::new_agentic_trace(trace, 1).unwrap();

        let parent = driver.pop_ready(0.0, usize::MAX);
        driver
            .on_terminal(parent[0].request_uuid, 10.0, ReplayTerminalStatus::Failed)
            .unwrap();
        assert!(driver.pop_ready(100.0, usize::MAX).is_empty());
        assert!(driver.is_drained());
        let snapshot = driver.agentic_trajectory_snapshot().unwrap();
        assert_eq!(snapshot.total_trajectories, 1);
        assert_eq!(snapshot.completed_trajectories, 0);
    }

    #[test]
    fn agentic_lane_recycles_at_client_terminal_but_drain_waits_for_cleanup() {
        for status in [
            ReplayTerminalStatus::Completed,
            ReplayTerminalStatus::Rejected,
            ReplayTerminalStatus::Canceled,
            ReplayTerminalStatus::Failed,
        ] {
            for cleanup_at_ms in [20.0, 40.0] {
                let trace = agentic_trace(vec![
                    agentic_node("a", "play-a", 0.0, Vec::new()),
                    agentic_node("b", "play-b", 0.0, Vec::new()),
                ]);
                let mut driver = WorkloadDriver::new_agentic_trace_with_lanes(trace, 1, 1).unwrap();
                let first = driver.pop_ready(0.0, 1).pop().unwrap();
                driver
                    .on_causal_terminal(first.request_uuid, 10.0, status)
                    .unwrap();

                let outcome = &driver.agentic_play_outcomes().unwrap()[0];
                assert_eq!(outcome.status, AgenticPlayStatus::Incomplete);
                assert_eq!(outcome.causal_terminal_ms, Some(10.0));
                assert_eq!(outcome.settled_at_ms, None);
                assert_eq!(driver.next_ready_time_ms(), Some(10.0));
                let second = driver.pop_ready(10.0, 1).pop().unwrap();
                assert_eq!(second.authored_request_id.as_deref(), Some("b"));
                assert_eq!(second.dispatched_at_ms, 10.0);
                driver.on_complete(second.request_uuid, 12.0).unwrap();
                assert!(!driver.is_drained(), "play-a still owns server cleanup");
                let failed = status != ReplayTerminalStatus::Completed;
                let mut expected_outcomes = vec![
                    AgenticPlayOutcome {
                        play_id: "play-a".into(),
                        status: AgenticPlayStatus::Incomplete,
                        causal_terminal_ms: Some(10.0),
                        settled_at_ms: None,
                        failure_request_id: failed.then(|| "a".into()),
                        failure_status: failed.then_some(status),
                    },
                    AgenticPlayOutcome {
                        play_id: "play-b".into(),
                        status: AgenticPlayStatus::Completed,
                        causal_terminal_ms: Some(12.0),
                        settled_at_ms: Some(12.0),
                        failure_request_id: None,
                        failure_status: None,
                    },
                ];
                assert_eq!(driver.agentic_play_outcomes().unwrap(), expected_outcomes);

                driver
                    .on_quiescent(first.request_uuid, cleanup_at_ms)
                    .unwrap();
                assert!(driver.is_drained());
                expected_outcomes[0].status = if failed {
                    AgenticPlayStatus::Failed
                } else {
                    AgenticPlayStatus::Completed
                };
                expected_outcomes[0].settled_at_ms = Some(cleanup_at_ms);
                assert_eq!(driver.agentic_play_outcomes().unwrap(), expected_outcomes);
                let transcript = driver.agentic_lifecycle_transcript().unwrap();
                let settlements = transcript
                    .events
                    .iter()
                    .filter(|event| event.event == AgenticLifecycleEventKind::PlayQuiescent)
                    .map(|event| (event.play_id.as_str(), event.at_ms))
                    .collect::<Vec<_>>();
                assert_eq!(
                    settlements,
                    vec![("play-b", 12.0), ("play-a", cleanup_at_ms)]
                );
            }
        }
    }

    #[test]
    fn late_quiescence_cannot_release_a_lane_owned_by_the_next_play() {
        let trace = agentic_trace(vec![
            agentic_node("a", "play-a", 0.0, Vec::new()),
            agentic_node("b", "play-b", 0.0, Vec::new()),
            agentic_node("c", "play-c", 0.0, Vec::new()),
        ]);
        let mut driver = WorkloadDriver::new_agentic_trace_with_lanes(trace, 1, 1).unwrap();
        let first = driver.pop_ready(0.0, 1).pop().unwrap();
        driver
            .on_causal_terminal(first.request_uuid, 10.0, ReplayTerminalStatus::Completed)
            .unwrap();
        let second = driver.pop_ready(10.0, 1).pop().unwrap();

        driver.on_quiescent(first.request_uuid, 20.0).unwrap();
        assert!(driver.pop_ready(20.0, usize::MAX).is_empty());
        driver
            .on_causal_terminal(second.request_uuid, 30.0, ReplayTerminalStatus::Completed)
            .unwrap();
        let third = driver.pop_ready(30.0, 1).pop().unwrap();
        assert_eq!(third.authored_request_id.as_deref(), Some("c"));
        driver.on_complete(third.request_uuid, 31.0).unwrap();
        assert!(!driver.is_drained());
        driver.on_quiescent(second.request_uuid, 35.0).unwrap();
        assert!(driver.is_drained());
    }

    #[test]
    fn agentic_lane_waits_for_background_terminals_even_after_root_failure() {
        for status in [
            ReplayTerminalStatus::Completed,
            ReplayTerminalStatus::Rejected,
            ReplayTerminalStatus::Canceled,
            ReplayTerminalStatus::Failed,
        ] {
            let trace = agentic_trace(vec![
                agentic_node("root", "play-a", 0.0, Vec::new()),
                agentic_node(
                    "background",
                    "play-a",
                    0.0,
                    vec![dependency(
                        "root",
                        AgenticDependencyTrigger::Dispatch,
                        0.0,
                        AgenticDependencyRelation::Spawn,
                    )],
                ),
                agentic_node("next", "play-b", 0.0, Vec::new()),
            ]);
            let mut driver = WorkloadDriver::new_agentic_trace_with_lanes(trace, 1, 1).unwrap();
            let ready = driver.pop_ready(0.0, usize::MAX);
            assert_eq!(ready.len(), 2);
            driver
                .on_terminal(ready[0].request_uuid, 10.0, status)
                .unwrap();
            assert!(driver.pop_ready(10.0, usize::MAX).is_empty());
            let outcome = &driver.agentic_play_outcomes().unwrap()[0];
            assert_eq!(outcome.status, AgenticPlayStatus::Incomplete);
            assert_eq!(outcome.causal_terminal_ms, Some(10.0));
            assert_eq!(outcome.settled_at_ms, None);

            driver
                .on_causal_terminal(ready[1].request_uuid, 15.0, ReplayTerminalStatus::Completed)
                .unwrap();
            let next = driver.pop_ready(15.0, 1).pop().unwrap();
            assert_eq!(next.authored_request_id.as_deref(), Some("next"));
            assert_eq!(next.dispatched_at_ms, 15.0);
            driver.on_complete(next.request_uuid, 16.0).unwrap();
            assert!(!driver.is_drained());
            let failed = status != ReplayTerminalStatus::Completed;
            // A failed play retains its primary failure at 10 ms, even though
            // its background request holds the client lane until 15 ms.
            let mut expected_outcomes = vec![
                AgenticPlayOutcome {
                    play_id: "play-a".into(),
                    status: AgenticPlayStatus::Incomplete,
                    causal_terminal_ms: Some(if failed { 10.0 } else { 15.0 }),
                    settled_at_ms: None,
                    failure_request_id: failed.then(|| "root".into()),
                    failure_status: failed.then_some(status),
                },
                AgenticPlayOutcome {
                    play_id: "play-b".into(),
                    status: AgenticPlayStatus::Completed,
                    causal_terminal_ms: Some(16.0),
                    settled_at_ms: Some(16.0),
                    failure_request_id: None,
                    failure_status: None,
                },
            ];
            assert_eq!(driver.agentic_play_outcomes().unwrap(), expected_outcomes);
            driver.on_quiescent(ready[1].request_uuid, 20.0).unwrap();
            assert!(driver.is_drained());
            expected_outcomes[0].status = if failed {
                AgenticPlayStatus::Failed
            } else {
                AgenticPlayStatus::Completed
            };
            expected_outcomes[0].settled_at_ms = Some(20.0);
            assert_eq!(driver.agentic_play_outcomes().unwrap(), expected_outcomes);
        }
    }

    #[test]
    fn agentic_lane_waits_for_delayed_authored_requests() {
        let trace = agentic_trace(vec![
            agentic_node("root", "play-a", 0.0, Vec::new()),
            agentic_node(
                "delayed",
                "play-a",
                0.0,
                vec![dependency(
                    "root",
                    AgenticDependencyTrigger::Completion,
                    5.0,
                    AgenticDependencyRelation::Sequence,
                )],
            ),
            agentic_node("next", "play-b", 0.0, Vec::new()),
        ]);
        let mut driver = WorkloadDriver::new_agentic_trace_with_lanes(trace, 1, 1).unwrap();
        let root = driver.pop_ready(0.0, 1).pop().unwrap();
        driver.on_complete(root.request_uuid, 10.0).unwrap();
        assert!(driver.pop_ready(14.0, usize::MAX).is_empty());
        let delayed = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(delayed.len(), 1);
        assert_eq!(delayed[0].authored_request_id.as_deref(), Some("delayed"));
        driver.on_complete(delayed[0].request_uuid, 20.0).unwrap();
        let next = driver.pop_ready(20.0, 1).pop().unwrap();
        assert_eq!(next.authored_request_id.as_deref(), Some("next"));
        driver.on_complete(next.request_uuid, 21.0).unwrap();
        assert!(driver.is_drained());
    }

    #[test]
    fn agentic_lane_rebases_descendant_timing_to_recycled_play_start() {
        let trace = agentic_trace(vec![
            agentic_node("first", "play-a", 0.0, Vec::new()),
            agentic_node("second-root", "play-b", 1_000.0, Vec::new()),
            agentic_node(
                "second-child",
                "play-b",
                1_005.0,
                vec![dependency(
                    "second-root",
                    AgenticDependencyTrigger::Dispatch,
                    0.0,
                    AgenticDependencyRelation::Spawn,
                )],
            ),
        ]);
        let mut driver = WorkloadDriver::new_agentic_trace_with_lanes(trace, 1, 1).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        driver
            .on_causal_terminal(first[0].request_uuid, 10.0, ReplayTerminalStatus::Completed)
            .unwrap();
        let second_root = driver.pop_ready(10.0, usize::MAX);
        assert_eq!(second_root.len(), 1);
        assert_eq!(
            second_root[0].authored_request_id.as_deref(),
            Some("second-root")
        );
        assert_eq!(second_root[0].scheduled_ready_at_ms, 10.0);
        assert!(driver.pop_ready(14.9, usize::MAX).is_empty());
        let second_child = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(second_child.len(), 1);
        assert_eq!(
            second_child[0].authored_request_id.as_deref(),
            Some("second-child")
        );
        driver.on_quiescent(first[0].request_uuid, 20.0).unwrap();
    }

    #[test]
    fn agentic_join_uses_the_slowest_completion_constraint() {
        let spawn = |request_id| {
            agentic_node(
                request_id,
                "play",
                0.0,
                vec![dependency(
                    "root",
                    AgenticDependencyTrigger::Dispatch,
                    0.0,
                    AgenticDependencyRelation::Spawn,
                )],
            )
        };
        let trace = agentic_trace(vec![
            spawn("a"),
            spawn("b"),
            agentic_node(
                "join",
                "play",
                1.0,
                vec![
                    dependency(
                        "a",
                        AgenticDependencyTrigger::Completion,
                        2.0,
                        AgenticDependencyRelation::Join,
                    ),
                    dependency(
                        "b",
                        AgenticDependencyTrigger::Completion,
                        2.0,
                        AgenticDependencyRelation::Join,
                    ),
                ],
            ),
            agentic_node("root", "play", 0.0, Vec::new()),
        ]);
        let mut driver = WorkloadDriver::new_agentic_trace(trace, 1).unwrap();
        let initial = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(initial.len(), 3);
        let a = initial
            .iter()
            .find(|turn| turn.authored_request_id.as_deref() == Some("a"))
            .unwrap();
        let b = initial
            .iter()
            .find(|turn| turn.authored_request_id.as_deref() == Some("b"))
            .unwrap();
        driver.on_complete(a.request_uuid, 10.0).unwrap();
        assert!(driver.next_ready_time_ms().is_none());
        driver.on_complete(b.request_uuid, 30.0).unwrap();
        assert_eq!(driver.next_ready_time_ms(), Some(32.0));
    }
}
