// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! InferenceX-style setup for the agentic workload driver.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;

impl AgenticReplayConfig {
    pub(super) fn validate(self, play_count: usize) -> Result<Self> {
        if self.lanes == 0 {
            bail!("agentic replay lanes must be greater than 0");
        }
        if play_count == 0 {
            bail!("agentic replay requires at least one play");
        }
        if self.lanes > play_count {
            bail!(
                "agentic replay lanes ({}) cannot exceed available plays ({play_count})",
                self.lanes
            );
        }
        if !self.start_min_ratio.is_finite()
            || !self.start_max_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.start_min_ratio)
            || !(0.0..=1.0).contains(&self.start_max_ratio)
            || self.start_min_ratio > self.start_max_ratio
        {
            bail!(
                "agentic replay start ratios must satisfy 0 <= min <= max <= 1; got {}..{}",
                self.start_min_ratio,
                self.start_max_ratio
            );
        }
        for (name, value) in [
            ("profile_duration_ms", self.profile_duration_ms),
            ("post_profile_grace_ms", self.post_profile_grace_ms),
            ("trace_idle_gap_cap_ms", self.trace_idle_gap_cap_ms),
            ("system_idle_gap_cap_ms", self.system_idle_gap_cap_ms),
        ] {
            if !value.is_finite() || value < 0.0 {
                bail!("{name} must be finite and non-negative; got {value}");
            }
        }
        if !(self.profile_duration_ms + self.post_profile_grace_ms).is_finite() {
            bail!("profile_duration_ms plus post_profile_grace_ms must be finite");
        }
        Ok(self)
    }
}

impl AgenticReplayState {
    pub(super) fn new(config: AgenticReplayConfig, next_cache_bust_hash_id: u32) -> Self {
        Self {
            config,
            phase: AgenticReplayPhase::Priming,
            primer_sessions: FxHashMap::default(),
            primers_in_flight: 0,
            warmup_issued_by_lane: vec![0; config.lanes],
            profile_start_ms: None,
            profile_deadline_ms: None,
            next_request_id: 1,
            next_cache_bust_hash_id,
            system_idle_since_ms: 0.0,
            total_trajectories: 0,
            completed_trajectories: 0,
            e2e_latencies_ms: Vec::new(),
        }
    }
}

impl AgenticState {
    pub(super) fn initialize_replay(
        &mut self,
        sessions: &mut Vec<SessionRuntime>,
        ready_sessions: &mut BinaryHeap<ReadySession>,
    ) {
        let config = self.replay.as_ref().expect("replay state exists").config;
        let mut rng = StdRng::seed_from_u64(config.random_seed);
        let initial_plays = self
            .lanes
            .iter()
            .filter_map(|lane| lane.plays.first().copied())
            .collect::<Vec<_>>();
        let mut primers = Vec::new();
        for play_index in initial_plays {
            let play = &self.plays[play_index];
            let start_ms = play
                .nodes
                .iter()
                .map(|node| self.authored_not_before_ms[*node])
                .min_by(f64::total_cmp)
                .expect("validated agentic play has nodes");
            let end_ms = play
                .nodes
                .iter()
                .map(|node| self.authored_not_before_ms[*node])
                .max_by(f64::total_cmp)
                .expect("validated agentic play has nodes");
            let ratio = if config.start_min_ratio == config.start_max_ratio {
                config.start_min_ratio
            } else {
                rng.random_range(config.start_min_ratio..config.start_max_ratio)
            };
            let t_star_ms = start_ms + (end_ms - start_ms) * ratio;

            let mut latest_history = FxHashMap::<String, usize>::default();
            let mut live_sessions = FxHashMap::<String, ()>::default();
            for &node_index in &play.nodes {
                let session_id = sessions[node_index].session_id.clone();
                if self.authored_not_before_ms[node_index] < t_star_ms {
                    let replace = latest_history.get(&session_id).is_none_or(|previous| {
                        self.authored_not_before_ms[*previous]
                            < self.authored_not_before_ms[node_index]
                    });
                    if replace {
                        latest_history.insert(session_id, node_index);
                    }
                } else {
                    live_sessions.insert(session_id, ());
                }
            }
            for session_id in live_sessions.keys() {
                let Some(&source_node) = latest_history.get(session_id) else {
                    continue;
                };
                let mut turn = sessions[source_node].turns[0].clone();
                turn.request_id = turn.request_id.map(|id| format!("{id}:snapshot-primer"));
                turn.max_output_tokens = 1;
                turn.output_token_ids = turn
                    .output_token_ids
                    .map(|tokens| tokens.into_iter().take(1).collect());
                turn.deterministic_request_id = None;
                let primer_session = sessions.len();
                sessions.push(SessionRuntime {
                    session_id: sessions[source_node].session_id.clone(),
                    turns: vec![turn],
                    cumulative_tokens: Vec::new(),
                    next_turn_index: 0,
                    next_ready_at_ms: None,
                    in_flight: None,
                });
                let lead_ms = (t_star_ms - self.authored_not_before_ms[source_node])
                    .max(0.0)
                    .min(config.system_idle_gap_cap_ms);
                primers.push((primer_session, source_node, lead_ms));
            }
            self.activate_snapshot_play(play_index, t_star_ms, 0.0, sessions, ready_sessions);
        }

        let max_lead_ms = primers
            .iter()
            .map(|(_, _, lead_ms)| *lead_ms)
            .max_by(f64::total_cmp)
            .unwrap_or(0.0);
        for (primer_session, source_node, lead_ms) in primers {
            let ready_at_ms = max_lead_ms - lead_ms;
            sessions[primer_session].next_ready_at_ms = Some(ready_at_ms);
            ready_sessions.push(ReadySession {
                ready_at_ms,
                session_index: primer_session,
                turn_index: 0,
            });
            self.replay
                .as_mut()
                .expect("replay state exists")
                .primer_sessions
                .insert(primer_session, source_node);
        }
        let primer_count = self
            .replay
            .as_ref()
            .expect("replay state exists")
            .primer_sessions
            .len();
        self.replay
            .as_mut()
            .expect("replay state exists")
            .primers_in_flight = primer_count;
        if primer_count == 0 {
            self.replay.as_mut().expect("replay state exists").phase = AgenticReplayPhase::Warmup;
            self.maybe_start_profiling(0.0, sessions, ready_sessions);
        }
    }

    pub(super) fn refresh_cache_bust_hash_ids(
        &mut self,
        play_index: usize,
        sessions: &[SessionRuntime],
    ) {
        let Some(replay) = &mut self.replay else {
            return;
        };
        let mut marker_by_session = FxHashMap::default();
        for &node_index in &self.plays[play_index].nodes {
            let marker = *marker_by_session
                .entry(sessions[node_index].session_id.as_str())
                .or_insert_with(|| {
                    let marker = replay.next_cache_bust_hash_id;
                    replay.next_cache_bust_hash_id = replay
                        .next_cache_bust_hash_id
                        .checked_add(1)
                        .expect("agentic replay cache-bust hash id overflow");
                    marker
                });
            self.cache_bust_hash_ids[node_index] = Some(marker);
        }
    }

    fn activate_snapshot_play(
        &mut self,
        play_index: usize,
        t_star_ms: f64,
        start_ms: f64,
        sessions: &mut [SessionRuntime],
        ready_sessions: &mut BinaryHeap<ReadySession>,
    ) {
        self.refresh_cache_bust_hash_ids(play_index, sessions);
        let play = &mut self.plays[play_index];
        play.emitted_in_flight = 0;
        play.completed_nodes = 0;
        play.failed = false;
        play.quiescent = false;
        play.root_dispatch_ms = None;
        play.max_terminal_ms = None;
        play.measured_occurrence = false;
        if let Some(lane_index) = play.lane_index {
            self.lanes[lane_index].active_play = Some(play_index);
        }

        for &node_index in &play.nodes {
            self.remaining_dependencies[node_index] = self.initial_dependencies[node_index];
            self.ready_after_ms[node_index] =
                start_ms + (self.authored_not_before_ms[node_index] - t_star_ms).max(0.0);
            let session = &mut sessions[node_index];
            session.next_ready_at_ms = None;
            session.in_flight = None;
            if self.authored_not_before_ms[node_index] < t_star_ms {
                self.node_states[node_index] = AgenticNodeState::Completed;
                session.next_turn_index = session.turns.len();
                play.completed_nodes += 1;
            } else {
                self.node_states[node_index] = AgenticNodeState::Blocked;
                session.next_turn_index = 0;
            }
        }

        for &source_node in &play.nodes {
            if self.node_states[source_node] != AgenticNodeState::Completed {
                continue;
            }
            for edge in self.dispatch_dependents[source_node]
                .iter()
                .chain(&self.completion_dependents[source_node])
            {
                self.remaining_dependencies[edge.target_node] = self.remaining_dependencies
                    [edge.target_node]
                    .checked_sub(1)
                    .expect("snapshot history dependency must fire exactly once");
            }
        }

        for &node_index in &play.nodes {
            if self.node_states[node_index] == AgenticNodeState::Blocked
                && self.remaining_dependencies[node_index] == 0
            {
                Self::schedule_node(
                    node_index,
                    start_ms,
                    &mut self.node_states,
                    sessions,
                    ready_sessions,
                );
            }
        }
    }

    pub(super) fn is_primer_session(&self, session_index: usize) -> bool {
        self.replay
            .as_ref()
            .is_some_and(|replay| replay.primer_sessions.contains_key(&session_index))
    }

    pub(super) fn can_emit(&self, session_index: usize, now_ms: f64) -> bool {
        let Some(replay) = &self.replay else {
            return true;
        };
        let is_primer = replay.primer_sessions.contains_key(&session_index);
        match replay.phase {
            AgenticReplayPhase::Priming => is_primer,
            AgenticReplayPhase::Warmup => {
                if is_primer {
                    return false;
                }
                let lane = self.plays[self.node_to_play[session_index]]
                    .lane_index
                    .expect("agentic replay node must belong to a lane");
                replay.warmup_issued_by_lane[lane] < replay.config.warmup_requests_per_lane
            }
            AgenticReplayPhase::Profiling => {
                !is_primer
                    && replay
                        .profile_deadline_ms
                        .is_none_or(|deadline| now_ms < deadline)
            }
            AgenticReplayPhase::Draining => false,
        }
    }

    pub(super) fn mark_emitted(&mut self, session_index: usize, now_ms: f64, measured: bool) {
        if self.is_primer_session(session_index) {
            return;
        }
        if let Some(replay) = &mut self.replay
            && replay.phase == AgenticReplayPhase::Warmup
        {
            let lane = self.plays[self.node_to_play[session_index]]
                .lane_index
                .expect("agentic replay node must belong to a lane");
            replay.warmup_issued_by_lane[lane] += 1;
        }
        self.on_node_emitted(session_index, now_ms, measured);
    }

    pub(super) fn on_primer_quiescent(
        &mut self,
        now_ms: f64,
        sessions: &mut [SessionRuntime],
        ready_sessions: &mut BinaryHeap<ReadySession>,
    ) {
        let replay = self
            .replay
            .as_mut()
            .expect("primer requires agentic replay state");
        replay.primers_in_flight = replay.primers_in_flight.saturating_sub(1);
        if replay.phase == AgenticReplayPhase::Priming && replay.primers_in_flight == 0 {
            replay.phase = AgenticReplayPhase::Warmup;
            for (node_index, session) in
                sessions.iter_mut().enumerate().take(self.node_states.len())
            {
                if self.node_states[node_index] != AgenticNodeState::Ready {
                    continue;
                }
                session.next_ready_at_ms = Some(now_ms);
                ready_sessions.push(ReadySession {
                    ready_at_ms: now_ms,
                    session_index: node_index,
                    turn_index: 0,
                });
            }
            self.maybe_start_profiling(now_ms, sessions, ready_sessions);
        }
    }

    pub(super) fn maybe_start_profiling(
        &mut self,
        now_ms: f64,
        sessions: &mut [SessionRuntime],
        ready_sessions: &mut BinaryHeap<ReadySession>,
    ) {
        let Some(replay) = &self.replay else {
            return;
        };
        if replay.phase != AgenticReplayPhase::Warmup
            || replay
                .warmup_issued_by_lane
                .iter()
                .any(|count| *count < replay.config.warmup_requests_per_lane)
            || self.plays.iter().any(|play| play.emitted_in_flight != 0)
        {
            return;
        }

        let duration_ms = replay.config.profile_duration_ms;
        let mut resume = Vec::new();
        for (node_index, session) in sessions.iter().enumerate().take(self.node_states.len()) {
            if self.node_states[node_index] == AgenticNodeState::Ready
                && session.in_flight.is_none()
                && session.next_turn_index < session.turns.len()
            {
                resume.push((
                    node_index,
                    (self.ready_after_ms[node_index] - now_ms).max(0.0),
                ));
            }
        }
        let minimum_delay = resume
            .iter()
            .map(|(_, delay)| *delay)
            .min_by(f64::total_cmp)
            .unwrap_or(0.0);

        {
            let replay = self.replay.as_mut().expect("replay state still exists");
            replay.phase = if duration_ms == 0.0 {
                AgenticReplayPhase::Draining
            } else {
                AgenticReplayPhase::Profiling
            };
            replay.profile_start_ms = Some(now_ms);
            replay.profile_deadline_ms = Some(now_ms + duration_ms);
            replay.system_idle_since_ms = now_ms;
        }

        let inactive_lanes = self
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(lane_index, lane)| lane.active_play.is_none().then_some(lane_index))
            .collect::<Vec<_>>();
        for lane_index in inactive_lanes {
            let lane = &mut self.lanes[lane_index];
            lane.next_play %= lane.plays.len();
            let play_index = lane.plays[lane.next_play];
            self.activate_play(play_index, now_ms, sessions, ready_sessions);
        }
        for play in &mut self.plays {
            if play.quiescent || play.lane_index.is_none() || play.measured_occurrence {
                continue;
            }
            play.measured_occurrence = true;
            play.root_dispatch_ms = None;
            play.max_terminal_ms = None;
            self.replay
                .as_mut()
                .expect("replay state still exists")
                .total_trajectories += 1;
        }
        for (node_index, delay) in resume {
            let ready_at_ms = now_ms + (delay - minimum_delay).max(0.0);
            self.ready_after_ms[node_index] = ready_at_ms;
            sessions[node_index].next_ready_at_ms = Some(ready_at_ms);
            ready_sessions.push(ReadySession {
                ready_at_ms,
                session_index: node_index,
                turn_index: 0,
            });
        }
    }

    pub(super) fn stop_profiling_if_due(&mut self, now_ms: f64, sessions: &mut [SessionRuntime]) {
        let Some(replay) = &mut self.replay else {
            return;
        };
        if replay.phase != AgenticReplayPhase::Profiling
            || replay
                .profile_deadline_ms
                .is_none_or(|deadline| now_ms < deadline)
        {
            return;
        }
        replay.phase = AgenticReplayPhase::Draining;
        for (node_index, session) in sessions.iter_mut().enumerate().take(self.node_states.len()) {
            if matches!(
                self.node_states[node_index],
                AgenticNodeState::Blocked | AgenticNodeState::Ready
            ) {
                self.node_states[node_index] = AgenticNodeState::Skipped;
                session.next_ready_at_ms = None;
                session.next_turn_index = session.turns.len();
            }
        }
    }

    pub(super) fn enforce_idle_caps(
        &mut self,
        sessions: &mut [SessionRuntime],
        ready_sessions: &mut BinaryHeap<ReadySession>,
        driver_in_flight: bool,
    ) {
        let Some(replay) = &self.replay else {
            return;
        };
        if replay.phase != AgenticReplayPhase::Profiling {
            return;
        }
        let trace_cap_ms = replay.config.trace_idle_gap_cap_ms;
        let system_cap_ms = replay.config.system_idle_gap_cap_ms;

        let mut shifts = Vec::new();
        for play_index in self.lanes.iter().filter_map(|lane| lane.active_play) {
            let play = &self.plays[play_index];
            if play.quiescent || play.emitted_in_flight != 0 {
                continue;
            }
            let earliest = play
                .nodes
                .iter()
                .filter_map(|node| sessions[*node].next_ready_at_ms)
                .min_by(f64::total_cmp);
            let Some(earliest) = earliest else {
                continue;
            };
            let idle_since = play
                .max_terminal_ms
                .or(replay.profile_start_ms)
                .unwrap_or(0.0);
            let capped = idle_since + trace_cap_ms;
            if earliest > capped {
                shifts.push((play_index, earliest - capped));
            }
        }
        for (play_index, shift) in shifts {
            for &node_index in &self.plays[play_index].nodes {
                let Some(ready_at_ms) = sessions[node_index].next_ready_at_ms else {
                    continue;
                };
                let shifted = ready_at_ms - shift;
                sessions[node_index].next_ready_at_ms = Some(shifted);
                self.ready_after_ms[node_index] = shifted;
                ready_sessions.push(ReadySession {
                    ready_at_ms: shifted,
                    session_index: node_index,
                    turn_index: 0,
                });
            }
        }

        if driver_in_flight {
            return;
        }
        let earliest = self
            .lanes
            .iter()
            .filter_map(|lane| lane.active_play)
            .flat_map(|play_index| self.plays[play_index].nodes.iter().copied())
            .filter_map(|node_index| sessions[node_index].next_ready_at_ms)
            .min_by(f64::total_cmp);
        let Some(earliest) = earliest else {
            return;
        };
        let idle_since = self
            .replay
            .as_ref()
            .expect("replay state still exists")
            .system_idle_since_ms;
        let capped = idle_since + system_cap_ms;
        if earliest <= capped {
            return;
        }
        let shift = earliest - capped;
        for play_index in self.lanes.iter().filter_map(|lane| lane.active_play) {
            for &node_index in &self.plays[play_index].nodes {
                let session = &mut sessions[node_index];
                let Some(ready_at_ms) = session.next_ready_at_ms else {
                    continue;
                };
                let shifted = ready_at_ms - shift;
                session.next_ready_at_ms = Some(shifted);
                self.ready_after_ms[node_index] = shifted;
                ready_sessions.push(ReadySession {
                    ready_at_ms: shifted,
                    session_index: node_index,
                    turn_index: 0,
                });
            }
        }
    }
}
