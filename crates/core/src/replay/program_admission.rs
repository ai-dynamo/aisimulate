// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified behavioral model derived from ai-dynamo/dynamo at
// 65a27b73b1aa35c0655e3eb0fbc50e327e2aa5d9:
// components/src/dynamo/thunderagent_router/{router,program_state}.py.
// Source: https://github.com/ai-dynamo/dynamo/tree/65a27b73b1aa35c0655e3eb0fbc50e327e2aa5d9/components/src/dynamo/thunderagent_router
// This is an offline logical-budget model, not a live Dynamo Router adapter.

use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use anyhow::{Result, bail, ensure};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::components::NoReplayMetadata;
use super::core::round_robin::PoolRoundRobinPlacement;
use super::core::{
    NoEngineEvents, Placement, PlacementDecision, PlacementEffects, PlacementPolicy, WorkerTopology,
};
use super::loadgen::ReplayRequestPayload;
use super::{ReplayComposition, ReplayError, ReplayResult, ReplaySpec, ReplayTopology};

/// Logical tokens per replica must come from the serving configuration, not
/// from a per-GPU byte count or observed physical prefix-cache occupancy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProgramAdmissionConfig {
    pub capacity_tokens_per_worker: usize,
    pub admission_control: bool,
    pub pause_threshold: f64,
    pub pause_target: f64,
    pub resume_hysteresis: f64,
    pub resume_timeout_seconds: f64,
    pub scheduler_interval_seconds: f64,
    pub acting_token_weight: f64,
    pub acting_decay_tau_seconds: f64,
    pub buffer_per_program: usize,
    pub soft_demote_threshold: f64,
    pub soft_demote_priority_jump: i32,
    pub resume_priority_boost: i32,
}

impl Default for ProgramAdmissionConfig {
    fn default() -> Self {
        Self {
            capacity_tokens_per_worker: 0,
            admission_control: true,
            pause_threshold: 0.95,
            pause_target: 0.80,
            resume_hysteresis: 0.10,
            resume_timeout_seconds: 1800.0,
            scheduler_interval_seconds: 5.0,
            acting_token_weight: 1.0,
            acting_decay_tau_seconds: 1.0,
            buffer_per_program: 100,
            soft_demote_threshold: 0.80,
            soft_demote_priority_jump: -2,
            resume_priority_boost: 1,
        }
    }
}

impl ProgramAdmissionConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.capacity_tokens_per_worker > 0,
            "capacity_tokens_per_worker must be positive"
        );
        for (name, value) in [
            ("pause_threshold", self.pause_threshold),
            ("pause_target", self.pause_target),
            ("resume_hysteresis", self.resume_hysteresis),
            ("soft_demote_threshold", self.soft_demote_threshold),
            ("acting_token_weight", self.acting_token_weight),
        ] {
            ensure!(
                value.is_finite() && (0.0..=1.0).contains(&value),
                "{name} must be finite and in [0,1]"
            );
        }
        ensure!(
            self.pause_threshold > 0.0 && self.pause_target <= self.pause_threshold,
            "pause_target must not exceed a positive pause_threshold"
        );
        ensure!(
            self.resume_hysteresis < self.pause_threshold,
            "resume_hysteresis must be below pause_threshold"
        );
        ensure!(
            self.soft_demote_threshold <= self.pause_threshold,
            "soft_demote_threshold must not exceed pause_threshold"
        );
        for (name, value) in [
            ("resume_timeout_seconds", self.resume_timeout_seconds),
            (
                "scheduler_interval_seconds",
                self.scheduler_interval_seconds,
            ),
            ("acting_decay_tau_seconds", self.acting_decay_tau_seconds),
        ] {
            ensure!(
                value.is_finite() && value > 0.0 && (value * 1000.0).is_finite(),
                "{name} must be finite and positive"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Default, Serialize)]
struct Counters {
    queued_requests: u64,
    pauses: u64,
    marked_for_pause: u64,
    resumes: u64,
    forced_resumes: u64,
    migrations: u64,
    ended_programs: u64,
    scheduler_ticks: u64,
    admission_wait_ms: f64,
    first_turn_admission_wait_ms: f64,
    continuation_admission_wait_ms: f64,
}

struct Program {
    key: String,
    worker: Option<usize>,
    last_worker: Option<usize>,
    tokens: usize,
    steps: usize,
    paused: bool,
    marked: bool,
    acting_since: Option<f64>,
    demoted_until: f64,
    request: Option<Uuid>,
    waiting_since: Option<f64>,
    boosted: bool,
}

/// Only compact identities and logical token counts are retained here. Prompt
/// expansion and physical prefix-cache ownership remain in the native engine.
pub struct ProgramAdmissionPlacement {
    config: ProgramAdmissionConfig,
    workers: BTreeMap<usize, usize>,
    used: BTreeMap<usize, f64>,
    programs: FxHashMap<usize, Program>,
    by_key: FxHashMap<String, usize>,
    requests: FxHashMap<Uuid, usize>,
    paused: BTreeMap<usize, u64>,
    next_pause_ordinal: u64,
    deadlines: BTreeSet<(u64, usize)>,
    next_id: usize,
    next_tick_ms: f64,
    counters: Counters,
}

impl ProgramAdmissionPlacement {
    pub fn new(
        config: ProgramAdmissionConfig,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            dp_size == 1,
            "program admission currently requires attention DP=1"
        );
        let mut workers = BTreeMap::new();
        for worker in topology {
            ensure!(
                worker.scheduler_ids.len() == 1,
                "program admission requires one scheduler per worker"
            );
            workers.insert(worker.worker_id, worker.scheduler_ids[0]);
        }
        ensure!(
            !workers.is_empty(),
            "program admission requires ready static workers"
        );
        let used = workers.keys().map(|&id| (id, 0.0)).collect();
        let next_tick_ms = config.scheduler_interval_seconds * 1000.0;
        Ok(Self {
            config,
            workers,
            used,
            programs: FxHashMap::default(),
            by_key: FxHashMap::default(),
            requests: FxHashMap::default(),
            paused: BTreeMap::new(),
            next_pause_ordinal: 0,
            deadlines: BTreeSet::new(),
            next_id: 0,
            next_tick_ms,
            counters: Counters::default(),
        })
    }

    fn contribution(&self, program: &Program) -> f64 {
        if program.paused {
            return 0.0;
        }
        let weight = if program.acting_since.is_some() {
            self.config.acting_token_weight
        } else {
            1.0
        };
        (program.tokens as f64 * weight).floor() + self.config.buffer_per_program as f64
    }

    fn remove_budget(&mut self, id: usize) {
        let p = &self.programs[&id];
        if let Some(worker) = p.worker {
            let value = self.contribution(p);
            *self.used.get_mut(&worker).unwrap() -= value;
        }
    }

    fn add_budget(&mut self, id: usize) {
        let p = &self.programs[&id];
        if let Some(worker) = p.worker {
            let value = self.contribution(p);
            *self.used.get_mut(&worker).unwrap() += value;
        }
    }

    fn placement(&self, id: usize) -> Placement {
        let p = &self.programs[&id];
        Placement {
            request_id: p.request.unwrap(),
            scheduler_id: self.workers[&p.worker.unwrap()],
            reported_overlap_tokens: 0,
            cache_sample: None,
            placement_replica_id: None,
        }
    }

    fn pause(&mut self, id: usize) {
        self.remove_budget(id);
        let p = self.programs.get_mut(&id).unwrap();
        p.paused = true;
        p.worker = None;
        p.marked = false;
        self.paused.insert(id, self.next_pause_ordinal);
        self.next_pause_ordinal += 1;
        self.counters.pauses += 1;
    }

    fn resume(&mut self, id: usize, worker: usize, now_ms: f64, forced: bool) -> Option<Placement> {
        let p = self.programs.get_mut(&id).unwrap();
        if p.last_worker.is_some_and(|previous| previous != worker) {
            self.counters.migrations += 1;
        }
        p.worker = Some(worker);
        p.last_worker = Some(worker);
        p.paused = false;
        p.boosted = p.waiting_since.is_some();
        if let Some(since) = p.waiting_since.take() {
            self.deadlines.remove(&(
                (since + self.config.resume_timeout_seconds * 1000.0).to_bits(),
                id,
            ));
            self.counters.admission_wait_ms += now_ms - since;
            if p.steps <= 1 {
                self.counters.first_turn_admission_wait_ms += now_ms - since;
            } else {
                self.counters.continuation_admission_wait_ms += now_ms - since;
            }
        }
        self.paused.remove(&id);
        self.counters.resumes += 1;
        self.counters.forced_resumes += u64::from(forced);
        self.add_budget(id);
        self.programs[&id].request.map(|_| self.placement(id))
    }

    fn tick(&mut self, now_ms: f64) -> Vec<Placement> {
        self.counters.scheduler_ticks += 1;
        let capacity = self.config.capacity_tokens_per_worker as f64;
        for p in self.programs.values_mut() {
            if let Some(worker) = p.worker {
                let utilization = self.used[&worker] / capacity;
                if self.config.soft_demote_threshold <= utilization
                    && utilization < self.config.pause_threshold
                    && !p.marked
                {
                    p.demoted_until = now_ms + self.config.scheduler_interval_seconds * 1500.0;
                }
            }
        }
        // Match ThunderAgent: resume only previously paused programs first.
        // Programs paused below must wait for a later tick (or their timeout).
        let ceiling = capacity * (self.config.pause_threshold - self.config.resume_hysteresis);
        // Positive finite f64 bits preserve numeric order. The worker-ID tie
        // uses Reverse so equal remaining capacities prefer the smaller ID.
        let mut remaining: BinaryHeap<_> = self
            .used
            .iter()
            .map(|(&w, &used)| (w, (ceiling.floor() - used).max(0.0)))
            .filter(|(_, free)| *free > self.config.buffer_per_program as f64)
            .map(|(w, free)| (free.to_bits(), std::cmp::Reverse(w)))
            .collect();
        let mut paused: Vec<_> = self.paused.keys().copied().collect();
        paused.sort_by_key(|id| {
            let p = &self.programs[id];
            (
                if p.steps <= 1 {
                    1
                } else if p.acting_since.is_none() {
                    0
                } else {
                    2
                },
                p.tokens,
                self.paused[id],
            )
        });
        let total_capacity: f64 = remaining
            .iter()
            .map(|(bits, _)| f64::from_bits(*bits))
            .sum();
        let mut reserved = 0.0;
        paused.retain(|id| {
            let required = self.programs[id].tokens as f64 + self.config.buffer_per_program as f64;
            if reserved + required <= total_capacity {
                reserved += required;
                true
            } else {
                false
            }
        });
        paused.sort_by_key(|id| std::cmp::Reverse(self.programs[id].tokens));
        let mut released = Vec::new();
        for id in paused {
            let required = self.programs[&id].tokens as f64 + self.config.buffer_per_program as f64;
            if let Some(&(bits, std::cmp::Reverse(worker))) = remaining.peek() {
                let free = f64::from_bits(bits);
                if required <= free {
                    remaining.pop();
                    let updated = free - required;
                    if updated > self.config.buffer_per_program as f64 {
                        remaining.push((updated.to_bits(), std::cmp::Reverse(worker)));
                    }
                    if let Some(placement) = self.resume(id, worker, now_ms, false) {
                        released.push(placement);
                    }
                }
            }
        }
        // Build each worker's candidate sets once per tick. Sorting candidates
        // avoids the live implementation's repeated full-table minimum scans.
        let mut candidates: BTreeMap<usize, (Vec<usize>, Vec<usize>)> = BTreeMap::new();
        for (&id, p) in &self.programs {
            if let Some(worker) = p.worker {
                if !p.marked {
                    let pair = candidates.entry(worker).or_default();
                    if p.acting_since.is_some() {
                        pair.0.push(id);
                    } else {
                        pair.1.push(id);
                    }
                }
            }
        }
        for (worker, (mut acting, mut reasoning)) in candidates {
            if self.used[&worker] <= capacity * self.config.pause_threshold {
                continue;
            }
            let key = |id: &usize| (self.programs[id].tokens, *id);
            acting.sort_by_key(key);
            reasoning.sort_by_key(key);
            for id in acting {
                if self.used[&worker] <= capacity * self.config.pause_target {
                    break;
                }
                self.pause(id);
            }
            if self.used[&worker] > capacity * self.config.pause_target {
                for id in reasoning {
                    self.programs.get_mut(&id).unwrap().marked = true;
                    self.counters.marked_for_pause += 1;
                }
            }
        }
        released
    }
}

impl PlacementPolicy<ReplayRequestPayload> for ProgramAdmissionPlacement {
    type Metadata = NoReplayMetadata;
    type Observation = ();

    fn place(
        &mut self,
        request: &ReplayRequestPayload,
        _: (),
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<PlacementEffects> {
        let interval = self.config.scheduler_interval_seconds * 1000.0;
        let next_tick = ((now_ms / interval).floor() + 1.0) * interval;
        let timeout = now_ms + self.config.resume_timeout_seconds * 1000.0;
        ensure!(
            now_ms.is_finite()
                && now_ms >= 0.0
                && timeout.is_finite()
                && timeout > now_ms
                && next_tick.is_finite()
                && next_tick > now_ms,
            "program admission deadlines must advance finite virtual time"
        );
        let uuid = request
            .metadata()
            .uuid
            .ok_or_else(|| anyhow::anyhow!("program request must have a UUID"))?;
        let key = request
            .metadata()
            .replay_context
            .as_ref()
            .and_then(|c| c.agentic.as_ref())
            .map(|a| a.play_id.clone())
            .or(session_id)
            .ok_or_else(|| {
                anyhow::anyhow!("program admission requires a program/session identity")
            })?;
        ensure!(
            !self.requests.contains_key(&uuid),
            "duplicate program request UUID"
        );
        if let Some(&id) = self.by_key.get(&key) {
            ensure!(
                self.programs[&id].request.is_none(),
                "program admission supports sequential programs; overlapping requests in program {key}"
            );
        }
        if self.programs.is_empty() && self.next_tick_ms <= now_ms {
            self.next_tick_ms = next_tick;
        }
        let id = if let Some(&id) = self.by_key.get(&key) {
            self.remove_budget(id);
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.by_key.insert(key.clone(), id);
            self.programs.insert(
                id,
                Program {
                    key,
                    worker: None,
                    last_worker: None,
                    tokens: 0,
                    steps: 0,
                    paused: false,
                    marked: false,
                    acting_since: None,
                    demoted_until: 0.0,
                    request: None,
                    waiting_since: None,
                    boosted: false,
                },
            );
            id
        };
        let p = self.programs.get_mut(&id).unwrap();
        p.steps += 1;
        p.tokens = request.input_length();
        p.acting_since = None;
        p.request = Some(uuid);
        p.boosted = false;
        self.requests.insert(uuid, id);
        if p.worker.is_none() && !p.paused {
            let required = p.tokens as f64 + self.config.buffer_per_program as f64;
            let worker = self
                .used
                .iter()
                .filter(|(_, used)| {
                    !self.config.admission_control
                        || (self.paused.is_empty()
                            && self.config.capacity_tokens_per_worker as f64 - **used >= required)
                })
                .min_by(|(wa, a), (wb, b)| a.total_cmp(b).then(wa.cmp(wb)))
                .map(|(&w, _)| w);
            let p = self.programs.get_mut(&id).unwrap();
            p.worker = worker;
            p.last_worker = worker;
            if worker.is_none() {
                p.paused = true;
                self.paused.insert(id, self.next_pause_ordinal);
                self.next_pause_ordinal += 1;
            }
        }
        self.add_budget(id);
        if self.programs[&id].paused {
            self.programs.get_mut(&id).unwrap().waiting_since = Some(now_ms);
            self.deadlines.insert((
                (now_ms + self.config.resume_timeout_seconds * 1000.0).to_bits(),
                id,
            ));
            self.counters.queued_requests += 1;
            Ok(PlacementEffects {
                decision: PlacementDecision::Queued,
                released: Vec::new(),
            })
        } else {
            Ok(PlacementEffects {
                decision: PlacementDecision::Immediate(self.placement(id)),
                released: Vec::new(),
            })
        }
    }
    fn observe(&mut self, _: (), _: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }
    fn cancel_pending(&mut self, uuid: Uuid) -> bool {
        let Some(&id) = self.requests.get(&uuid) else {
            return false;
        };
        if self.programs[&id].waiting_since.is_none() {
            return false;
        }
        let key = self.programs[&id].key.clone();
        self.program_finished(&key, 0.0)
            .expect("program cleanup is infallible");
        true
    }
    fn output_token(&mut self, uuid: Uuid) {
        if let Some(&id) = self.requests.get(&uuid) {
            let p = self.programs.get_mut(&id).unwrap();
            p.tokens += 1;
            if let Some(worker) = p.worker {
                *self.used.get_mut(&worker).unwrap() += 1.0;
            }
        }
    }
    fn request_terminal(&mut self, uuid: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        if let Some(id) = self.requests.remove(&uuid) {
            self.remove_budget(id);
            let p = self.programs.get_mut(&id).unwrap();
            p.request = None;
            p.acting_since = Some(now_ms);
            self.add_budget(id);
            if self.programs[&id].marked {
                self.pause(id);
            }
        }
        Ok(Vec::new())
    }
    fn prefill_completed(&mut self, _: Uuid, _: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }
    fn pending_count(&self) -> usize {
        self.deadlines.len()
    }
    fn tracks_program_completion(&self) -> bool {
        true
    }
    fn program_finished(&mut self, key: &str, _: f64) -> Result<Vec<Placement>> {
        if let Some(id) = self.by_key.remove(key) {
            self.remove_budget(id);
            let p = self.programs.remove(&id).unwrap();
            if let Some(uuid) = p.request {
                self.requests.remove(&uuid);
            }
            if let Some(since) = p.waiting_since {
                self.deadlines.remove(&(
                    (since + self.config.resume_timeout_seconds * 1000.0).to_bits(),
                    id,
                ));
            }
            self.paused.remove(&id);
            self.counters.ended_programs += 1;
        }
        Ok(Vec::new())
    }
    fn next_wakeup_ms(&self) -> Option<f64> {
        if !self.config.admission_control || self.programs.is_empty() {
            return None;
        }
        Some(self.deadlines.first().map_or(self.next_tick_ms, |(at, _)| {
            f64::from_bits(*at).min(self.next_tick_ms)
        }))
    }
    fn on_wakeup(&mut self, now_ms: f64) -> Result<Vec<Placement>> {
        let mut released = Vec::new();
        if self.next_tick_ms <= now_ms {
            let interval = self.config.scheduler_interval_seconds * 1000.0;
            let next = ((now_ms / interval).floor() + 1.0) * interval;
            ensure!(
                next.is_finite() && next > now_ms,
                "program admission scheduler interval cannot advance virtual time"
            );
            released.extend(self.tick(now_ms));
            self.next_tick_ms = next;
        }
        // Decay is needed only for force-resume selection. One table pass
        // computes all worker totals, instead of one table scan per worker.
        if self
            .deadlines
            .first()
            .is_some_and(|(at, _)| f64::from_bits(*at) <= now_ms)
        {
            let mut decayed: BTreeMap<_, f64> = self.workers.keys().map(|&w| (w, 0.0)).collect();
            for p in self.programs.values() {
                if let Some(worker) = p.worker {
                    let tokens = p.acting_since.map_or(p.tokens as f64, |since| {
                        (p.tokens as f64
                            * 2.0_f64.powf(
                                -(now_ms - since) / (self.config.acting_decay_tau_seconds * 1000.0),
                            ))
                        .floor()
                    });
                    *decayed.get_mut(&worker).unwrap() +=
                        tokens + self.config.buffer_per_program as f64;
                }
            }
            while let Some(&(at, id)) = self.deadlines.first() {
                if f64::from_bits(at) > now_ms {
                    break;
                }
                let worker = decayed
                    .iter()
                    .min_by(|(wa, a), (wb, b)| a.total_cmp(b).then(wa.cmp(wb)))
                    .map(|(&w, _)| w)
                    .unwrap();
                let charge =
                    self.programs[&id].tokens as f64 + self.config.buffer_per_program as f64;
                if let Some(placement) = self.resume(id, worker, now_ms, true) {
                    released.push(placement);
                }
                *decayed.get_mut(&worker).unwrap() += charge;
            }
        }
        Ok(released)
    }
    fn priority_adjustment(&self, uuid: Uuid, now_ms: f64) -> i32 {
        self.requests.get(&uuid).map_or(0, |id| {
            let p = &self.programs[id];
            (if p.boosted {
                self.config.resume_priority_boost
            } else {
                0
            })
            .saturating_add(if p.demoted_until > now_ms {
                self.config.soft_demote_priority_jump
            } else {
                0
            })
        })
    }
    fn diagnostics(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "provider": "program_admission", "qualification": "behavioral_model",
            "backend_priority_effects": "not_modeled_fcfs",
            "budget_basis": "logical_program_tokens_without_prefix_deduplication",
            "config": self.config, "counters": self.counters,
            "remaining_programs": self.programs.len(), "pending_requests": self.deadlines.len(),
            "logical_used_tokens_per_worker": self.used,
        }))
    }
    fn worker_ready(&mut self, _: WorkerTopology, _: f64) -> Result<Vec<Placement>> {
        bail!("program admission does not support worker startup or scaling")
    }
    fn worker_draining(&mut self, _: WorkerTopology, _: f64) -> Result<Vec<Placement>> {
        bail!("program admission does not support worker draining")
    }
    fn worker_removed(&mut self, _: WorkerTopology, _: f64) -> Result<Vec<Placement>> {
        bail!("program admission does not support worker removal")
    }
    fn topology_settled(&mut self, _: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }
}

pub struct ProgramAdmissionComposition(pub ProgramAdmissionConfig);

impl ReplayComposition for ProgramAdmissionComposition {
    type Metadata = NoReplayMetadata;
    type Observation = NoEngineEvents;
    type AggregatedPlacement = ProgramAdmissionPlacement;
    type DisaggregatedPlacement = PoolRoundRobinPlacement<()>;
    fn validate_spec(&self, spec: &ReplaySpec) -> ReplayResult<()> {
        self.0
            .validate()
            .map_err(|e| ReplayError::InvalidSpec(e.to_string()))?;
        let ReplayTopology::Aggregated { workers } = &spec.topology else {
            return Err(ReplayError::InvalidSpec(
                "program admission requires aggregated topology".into(),
            ));
        };
        if spec.adapters.placement.provider != "program_admission"
            || spec.adapters.scaling.provider != "none"
            || workers.startup_delay_ms != 0.0
        {
            return Err(ReplayError::InvalidSpec("program admission requires its explicit provider and static ready workers without scaling".into()));
        }
        let engine = super::engine::ReplayEngineConfig::parse(&spec.engine)?;
        if engine.dp_size != 1
            || engine.rank.native_host_offload.is_some()
            || engine.rank.g3_offload.is_some()
            || engine.rank.aic_nextn.is_some()
        {
            return Err(ReplayError::InvalidSpec("program admission requires attention DP=1 without host offload or speculative decoding".into()));
        }
        Ok(())
    }
    fn requires_workload_driver(&self) -> bool {
        true
    }
    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> Result<Self::AggregatedPlacement> {
        ProgramAdmissionPlacement::new(self.0.clone(), dp_size, topology)
    }
    fn create_disaggregated_placements(
        &mut self,
        _: u32,
        _: Vec<WorkerTopology>,
        _: u32,
        _: Vec<WorkerTopology>,
    ) -> Result<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        bail!("program admission requires aggregated topology")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::DirectRequest;

    fn policy(workers: usize) -> ProgramAdmissionPlacement {
        ProgramAdmissionPlacement::new(
            ProgramAdmissionConfig {
                capacity_tokens_per_worker: 100,
                buffer_per_program: 0,
                scheduler_interval_seconds: 0.01,
                resume_timeout_seconds: 0.025,
                ..Default::default()
            },
            1,
            (0..workers)
                .map(|id| WorkerTopology {
                    worker_id: id,
                    scheduler_ids: vec![id],
                })
                .collect(),
        )
        .unwrap()
    }
    fn request(uuid: u128, tokens: usize) -> ReplayRequestPayload {
        ReplayRequestPayload::materialized(DirectRequest {
            tokens: vec![1; tokens],
            uuid: Some(Uuid::from_u128(uuid)),
            max_output_tokens: 1,
            ..Default::default()
        })
    }
    fn place(
        p: &mut ProgramAdmissionPlacement,
        uuid: u128,
        key: &str,
        tokens: usize,
        now: f64,
    ) -> PlacementDecision {
        p.place(&request(uuid, tokens), (), Some(key.into()), now)
            .unwrap()
            .decision
    }
    fn worker(decision: PlacementDecision) -> usize {
        let PlacementDecision::Immediate(p) = decision else {
            panic!("expected immediate placement")
        };
        p.scheduler_id
    }

    #[test]
    fn affinity_survives_tool_wait_and_program_end_releases_budget() {
        let mut p = policy(2);
        assert_eq!(worker(place(&mut p, 1, "a", 60, 0.0)), 0);
        p.output_token(Uuid::from_u128(1));
        p.request_terminal(Uuid::from_u128(1), 1.0).unwrap();
        assert_eq!(p.used[&0], 61.0);
        assert_eq!(worker(place(&mut p, 2, "b", 60, 2.0)), 1);
        assert_eq!(worker(place(&mut p, 3, "a", 65, 5.0)), 0);
        p.request_terminal(Uuid::from_u128(3), 6.0).unwrap();
        p.program_finished("a", 6.0).unwrap();
        assert_eq!(p.used[&0], 0.0);
        assert_eq!(p.programs.len(), 1);
        assert_eq!(p.requests.len(), 1);
    }

    #[test]
    fn first_turn_waits_then_resumes_on_tick_after_program_end() {
        let mut p = policy(1);
        place(&mut p, 1, "a", 60, 0.0);
        assert!(matches!(
            place(&mut p, 2, "b", 60, 0.0),
            PlacementDecision::Queued
        ));
        p.request_terminal(Uuid::from_u128(1), 1.0).unwrap();
        p.program_finished("a", 1.0).unwrap();
        let released = p.on_wakeup(10.0).unwrap();
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].request_id, Uuid::from_u128(2));
        assert_eq!(p.pending_count(), 0);
        assert_eq!(p.counters.admission_wait_ms, 10.0);
    }

    #[test]
    fn reasoning_pauses_only_after_terminal_then_migrates_to_fit() {
        let mut p = policy(2);
        place(&mut p, 1, "a", 60, 0.0);
        place(&mut p, 2, "b", 20, 0.0);
        for _ in 0..40 {
            p.output_token(Uuid::from_u128(1));
        }
        p.on_wakeup(10.0).unwrap();
        assert!(p.programs[&0].marked);
        assert!(!p.programs[&0].paused);
        p.request_terminal(Uuid::from_u128(1), 11.0).unwrap();
        assert!(p.programs[&0].paused);
        assert_eq!(p.used[&0], 0.0);
        assert!(matches!(
            place(&mut p, 3, "a", 60, 12.0),
            PlacementDecision::Queued
        ));
        assert!(matches!(
            place(&mut p, 4, "c", 80, 12.0),
            PlacementDecision::Queued
        ));
        let released = p.on_wakeup(20.0).unwrap();
        assert_eq!(released.len(), 2);
        assert_eq!(released[0].request_id, Uuid::from_u128(4));
        assert_eq!(released[0].scheduler_id, 0);
        assert_eq!(released[1].request_id, Uuid::from_u128(3));
        assert_eq!(released[1].scheduler_id, 1);
        assert_eq!(p.counters.migrations, 1);
    }

    #[test]
    fn timeout_is_a_real_deadline_even_without_engine_work() {
        let mut p = policy(1);
        assert!(matches!(
            place(&mut p, 1, "oversize", 120, 0.0),
            PlacementDecision::Queued
        ));
        assert!(p.on_wakeup(10.0).unwrap().is_empty());
        assert!(p.on_wakeup(20.0).unwrap().is_empty());
        assert_eq!(p.next_wakeup_ms(), Some(25.0));
        let released = p.on_wakeup(25.0).unwrap();
        assert_eq!(released.len(), 1);
        assert_eq!(p.counters.forced_resumes, 1);
        assert_eq!(p.counters.admission_wait_ms, 25.0);
        assert_eq!(p.used[&0], 120.0);
    }

    fn overloaded_worker_with_acting_program() -> ProgramAdmissionPlacement {
        let mut p = policy(2);
        assert_eq!(worker(place(&mut p, 1, "acting", 60, 0.0)), 0);
        assert_eq!(worker(place(&mut p, 2, "finishes", 60, 0.0)), 1);
        assert_eq!(worker(place(&mut p, 3, "reasoning", 30, 0.0)), 0);
        for _ in 0..10 {
            p.output_token(Uuid::from_u128(3));
        }
        p.request_terminal(Uuid::from_u128(1), 1.0).unwrap();
        assert_eq!(p.used[&0], 100.0);
        p
    }

    #[test]
    fn tick_does_not_resume_a_program_paused_in_the_same_tick() {
        let mut p = overloaded_worker_with_acting_program();
        p.request_terminal(Uuid::from_u128(2), 1.0).unwrap();
        p.program_finished("finishes", 1.0).unwrap();

        assert!(p.on_wakeup(10.0).unwrap().is_empty());
        assert!(p.programs[&0].paused);
        assert_eq!(p.used[&0], 40.0);
        assert_eq!(p.used[&1], 0.0);
        assert_eq!(p.counters.pauses, 1);
        assert_eq!(p.counters.resumes, 0);
        assert_eq!(p.counters.migrations, 0);

        // An empty destination does not bypass the next policy tick.
        assert!(matches!(
            place(&mut p, 4, "acting", 60, 11.0),
            PlacementDecision::Queued
        ));
        let released = p.on_wakeup(20.0).unwrap();
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].request_id, Uuid::from_u128(4));
        assert_eq!(released[0].scheduler_id, 1);
        assert_eq!(p.counters.admission_wait_ms, 9.0);
        assert_eq!(p.counters.migrations, 1);
    }

    #[test]
    fn tick_resumes_existing_waiters_before_creating_new_pauses() {
        let mut p = overloaded_worker_with_acting_program();
        assert!(matches!(
            place(&mut p, 4, "waiting", 80, 1.0),
            PlacementDecision::Queued
        ));
        p.request_terminal(Uuid::from_u128(2), 2.0).unwrap();
        p.program_finished("finishes", 2.0).unwrap();

        let released = p.on_wakeup(10.0).unwrap();
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].request_id, Uuid::from_u128(4));
        assert_eq!(released[0].scheduler_id, 1);
        assert!(p.programs[&0].paused);
        assert_eq!(p.used[&0], 40.0);
        assert_eq!(p.used[&1], 80.0);
        assert_eq!(p.counters.resumes, 1);
        assert_eq!(p.counters.pauses, 1);
    }

    #[test]
    fn equal_size_resume_preserves_pause_order_instead_of_program_birth_order() {
        let mut p = policy(1);
        place(&mut p, 1, "older-program", 60, 0.0);
        p.request_terminal(Uuid::from_u128(1), 1.0).unwrap();
        place(&mut p, 2, "running", 30, 2.0);
        assert!(matches!(
            place(&mut p, 3, "earlier-waiter", 60, 3.0),
            PlacementDecision::Queued
        ));
        for _ in 0..10 {
            p.output_token(Uuid::from_u128(2));
        }
        p.on_wakeup(10.0).unwrap();
        assert!(p.programs[&0].paused);
        p.request_terminal(Uuid::from_u128(2), 11.0).unwrap();
        p.program_finished("running", 11.0).unwrap();
        let released = p.on_wakeup(20.0).unwrap();
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].request_id, Uuid::from_u128(3));
        assert!(p.programs[&0].paused);
    }

    #[test]
    fn invalid_overlap_is_transactional_and_cancel_removes_deadline() {
        let mut p = policy(1);
        place(&mut p, 1, "a", 120, 0.0);
        let before = p.diagnostics();
        assert!(p.place(&request(2, 10), (), Some("a".into()), 2.0).is_err());
        assert_eq!(before, p.diagnostics());
        assert!(p.cancel_pending(Uuid::from_u128(1)));
        assert_eq!(p.pending_count(), 0);
        assert_eq!(p.next_wakeup_ms(), None);
        assert!(p.by_key.is_empty() && p.requests.is_empty() && p.paused.is_empty());
    }

    #[test]
    fn late_first_arrival_and_affinity_only_have_no_stale_timers() {
        let mut p = policy(1);
        place(&mut p, 1, "late", 120, 103.0);
        assert_eq!(p.next_wakeup_ms(), Some(110.0));
        let mut p = policy(1);
        p.config.admission_control = false;
        assert_eq!(worker(place(&mut p, 1, "a", 120, 0.0)), 0);
        assert_eq!(worker(place(&mut p, 2, "b", 120, 0.0)), 0);
        assert_eq!(p.next_wakeup_ms(), None);
    }

    #[test]
    fn runtime_closes_the_loop_and_retires_every_program() {
        use crate::engine::{EngineConfig, TimingModelConfig};
        use crate::replay::WorkerStage;
        use crate::replay::agg::AggRuntimeImpl;
        use crate::replay::components::{AdmissionQueue, ReplayMode};
        use crate::replay::engine::{ReplayEngineConfig, ReplayEngineFactory};
        use crate::replay::loadgen::{SessionTrace, Trace, TurnTrace, WorkloadDriver};
        let trace = Trace {
            block_size: 2,
            sessions: (0..2)
                .map(|id| SessionTrace {
                    session_id: format!("program-{id}"),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: (0..2)
                        .map(|turn| TurnTrace {
                            input_length: 60,
                            max_output_tokens: 2,
                            hash_ids: vec![id + 1; 30],
                            delay_after_previous_ms: if turn == 0 { 0.0 } else { 5.0 },
                            ..Default::default()
                        })
                        .collect(),
                })
                .collect(),
        };
        let driver = WorkloadDriver::new_concurrency_without_replay_hashes(trace, 2, 2, false)
            .unwrap()
            .with_deterministic_request_ids(1);
        let engine_config = ReplayEngineConfig {
            rank: EngineConfig {
                block_size: 2,
                timing_model: TimingModelConfig::Fixed {
                    prefill_ms: 1.0,
                    decode_ms: 1.0,
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let factory = ReplayEngineFactory::new()
            .role_factory(&engine_config, WorkerStage::Aggregated, false)
            .unwrap();
        let runtime =
            AggRuntimeImpl::<ProgramAdmissionPlacement, NoEngineEvents, ()>::new_composed(
                factory,
                AdmissionQueue::new_workload(driver, ReplayMode::Concurrency { max_in_flight: 2 }),
                1,
                None,
                |dp, topology| ProgramAdmissionPlacement::new(policy(1).config, dp, topology),
            )
            .unwrap()
            .with_per_request_records(true);
        let (collector, _) = runtime.run().unwrap();
        let report = collector.finish();
        assert_eq!(report.request_counts.num_requests, 4);
        assert_eq!(report.request_counts.completed_requests, 4);
        assert_eq!(report.request_counts.total_output_tokens, 8);
        let diag = report.placement_diagnostics.unwrap();
        assert_eq!(diag["remaining_programs"], 0);
        assert_eq!(diag["pending_requests"], 0);
        assert_eq!(diag["counters"]["ended_programs"], 2);
        assert_eq!(diag["counters"]["queued_requests"], 1);
        assert_eq!(diag["logical_used_tokens_per_worker"]["0"], 0.0);
    }
}
