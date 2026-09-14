// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-boundary snapshots, not physical engine checkpoints.
//! The behavioral reference is AIPerf's trajectory_source.py at
//! https://github.com/ai-dynamo/aiperf/tree/7db2ba37a62aa80c882bc90eaf61cc8073e2387b.
//! Sampling and token allocation below are AISimulate-owned algorithms.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{
    AgenticDependencyRelation, AgenticDependencyTrigger, AgenticGraphIdentity, AgenticNode,
    ReplayRequestHashes, SYNTHETIC_OUTPUT_SEED, ValidatedAgenticGraph, planned_output_token_ids,
};
use crate::replay::AgenticRuntimeIdentity;

pub const AGENTIC_SNAPSHOT_SCHEMA_V1: &str = "aisimulate.agentic.snapshot.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgenticSnapshotOptions {
    pub seed: u64,
}

/// Recorded provenance and the independent benchmark request boundary.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AgenticSnapshotRequest {
    pub source_request_id: String,
    pub identity: AgenticRuntimeIdentity,
    pub recorded_start_ms: f64,
    pub recorded_end_ms: Option<f64>,
    pub historical: bool,
    /// Source-clock delay; execution divides this by the configured speedup.
    pub remaining_delay_ms: f64,
    pub pending_dependencies: Vec<String>,
}

/// A description only: physical primer execution belongs to phase orchestration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AgenticPrimer {
    pub source_request_id: String,
    pub request_id: String,
    pub conversation_id: String,
    pub input_length: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AgenticSnapshotEvidence {
    pub schema: &'static str,
    pub graph: AgenticGraphIdentity,
    pub seed: u64,
    pub lane_id: usize,
    pub play_ordinal: u64,
    pub source_play_id: String,
    pub play_id: String,
    pub cache_id: String,
    pub t_star_ms: f64,
    pub recorded_start_ms: f64,
    pub recorded_last_start_ms: f64,
    pub requests: Vec<AgenticSnapshotRequest>,
    pub primers: Vec<AgenticPrimer>,
}

/// Immutable corpus compilation retained by all views of a replay instance.
/// Token ranges are functions of lane/ordinal, never runtime completion order.
#[derive(Debug)]
pub struct AgenticReplayContext {
    pub(super) graph: ValidatedAgenticGraph,
    pub(super) lanes: usize,
    seed: u64,
    ranks: Vec<Vec<u32>>,
    index_by_id: FxHashMap<String, usize>,
    node_to_play: Vec<usize>,
    topological_rank: Vec<usize>,
    recorded_intervals: Vec<(f64, Option<f64>)>,
    pub(super) outputs: Vec<Vec<u32>>,
    token_stride: u64,
    request_stride: u128,
}

#[derive(Debug, Clone)]
pub struct AgenticPlaySnapshot {
    pub(super) context: Arc<AgenticReplayContext>,
    pub(super) source_play_index: usize,
    pub(super) slot: u64,
    pub(super) evidence: AgenticSnapshotEvidence,
}

#[derive(Debug, Clone)]
pub struct PreparedAgenticSnapshots {
    pub(super) context: Arc<AgenticReplayContext>,
    pub(super) plays: Vec<AgenticPlaySnapshot>,
    evidence: Vec<AgenticSnapshotEvidence>,
}

impl ValidatedAgenticGraph {
    pub fn prepare_snapshots(
        &self,
        lanes: usize,
        options: AgenticSnapshotOptions,
    ) -> Result<PreparedAgenticSnapshots> {
        if lanes == 0 || self.plays.is_empty() {
            bail!("agentic snapshots require positive lanes and a nonempty corpus");
        }
        let index_by_id = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.request_id.clone(), i))
            .collect::<FxHashMap<_, _>>();
        let mut remaining = self
            .nodes
            .iter()
            .map(|n| n.dependencies.len())
            .collect::<Vec<_>>();
        let mut dependents = vec![Vec::new(); self.nodes.len()];
        for (target, node) in self.nodes.iter().enumerate() {
            for edge in &node.dependencies {
                dependents[index_by_id[&edge.request_id]].push(target);
            }
        }
        let mut ready = remaining
            .iter()
            .enumerate()
            .filter_map(|(i, &n)| (n == 0).then_some(i))
            .collect::<BTreeSet<_>>();
        let mut topological_rank = vec![0; self.nodes.len()];
        let mut ordinal = 0;
        while let Some(source) = ready.pop_first() {
            topological_rank[source] = ordinal;
            ordinal += 1;
            for &target in &dependents[source] {
                remaining[target] -= 1;
                if remaining[target] == 0 {
                    ready.insert(target);
                }
            }
        }
        if ordinal != self.nodes.len() {
            bail!("snapshot source graph contains a dependency cycle");
        }
        let mut ranks = vec![Vec::new(); self.nodes.len()];
        let mut token_stride = 1_u64;
        let mut node_to_play = vec![0; self.nodes.len()];
        for (play_index, play) in self.plays.iter().enumerate() {
            let mut interner = FxHashMap::default();
            for &index in &play.nodes {
                node_to_play[index] = play_index;
                for hash in &self.nodes[index].hash_ids {
                    let next = u32::try_from(interner.len()).context(
                        "source play contains more unique hash IDs than u32 can represent",
                    )?;
                    ranks[index].push(*interner.entry(*hash).or_insert(next));
                }
            }
            token_stride = token_stride.max(interner.len() as u64);
        }
        let recorded_intervals = self
            .nodes
            .iter()
            .map(|node| {
                let (start, end) = node
                    .recorded_interval_ms
                    .map(|(start, end)| (start, Some(end)))
                    .unwrap_or_else(|| {
                        (
                            node.not_before_ms,
                            node.recorded_api_time_ms.map(|d| node.not_before_ms + d),
                        )
                    });
                if !start.is_finite()
                    || start < 0.0
                    || end.is_some_and(|end| !end.is_finite() || end < start)
                {
                    bail!("recorded API interval overflow or invalid interval");
                }
                Ok((start, end))
            })
            .collect::<Result<Vec<_>>>()?;
        // Preserve the complete corpus's existing output RNG consumption.
        let mut rng = StdRng::seed_from_u64(SYNTHETIC_OUTPUT_SEED);
        let outputs = self
            .nodes
            .iter()
            .map(|node| {
                planned_output_token_ids(
                    node.output_token_ids.clone(),
                    node.max_output_tokens,
                    &mut rng,
                )
            })
            .collect();
        let context = Arc::new(AgenticReplayContext {
            graph: self.clone(),
            lanes,
            seed: options.seed,
            ranks,
            outputs,
            index_by_id,
            node_to_play,
            topological_rank,
            recorded_intervals,
            token_stride,
            request_stride: self.nodes.len() as u128 + 1,
        });
        let plays = (0..lanes)
            .map(|lane| context.prepare_play(lane, 0, None))
            .collect::<Result<Vec<_>>>()?;
        PreparedAgenticSnapshots::from_plays(plays)
    }
}

impl AgenticReplayContext {
    pub fn graph_identity(&self) -> AgenticGraphIdentity {
        self.graph.identity()
    }
    pub fn lane_count(&self) -> usize {
        self.lanes
    }

    /// Prepare a deterministic corpus-cycle instance. `None` samples an initial
    /// cut; `Some(0.0)` starts a normalized Weka play from turn zero. Ordinals
    /// must advance on lane reuse, including while old server work still exists.
    pub fn prepare_play(
        self: &Arc<Self>,
        lane_id: usize,
        play_ordinal: u64,
        cut_ms: Option<f64>,
    ) -> Result<AgenticPlaySnapshot> {
        if lane_id >= self.lanes {
            bail!("snapshot lane is outside the configured lane count");
        }
        let slot = play_ordinal
            .checked_mul(u64::try_from(self.lanes)?)
            .and_then(|x| x.checked_add(lane_id as u64))
            .context("play ordinal overflow")?;
        let range_end = slot
            .checked_add(1)
            .and_then(|s| s.checked_mul(self.token_stride))
            .context("play token range overflow")?;
        if range_end > u64::from(u32::MAX) + 1 {
            bail!("play token identity range exceeds u32 capacity; identities cannot be recycled");
        }
        let source_play_index = (slot % self.graph.plays.len() as u64) as usize;
        let play = &self.graph.plays[source_play_index];
        let first = play
            .nodes
            .iter()
            .map(|&i| self.recorded_intervals[i].0)
            .min_by(f64::total_cmp)
            .context("source play has no requests")?;
        let last = play
            .nodes
            .iter()
            .map(|&i| self.recorded_intervals[i].0)
            .max_by(f64::total_cmp)
            .expect("nonempty play");
        // Versioned, platform-independent uniform draw: the upper 53 bits of a
        // domain-separated digest map exactly onto the binary64 unit interval.
        let key = serde_json::to_vec(&(
            AGENTIC_SNAPSHOT_SCHEMA_V1,
            self.seed,
            self.graph.graph_digest(),
            &play.play_id,
            lane_id,
            play_ordinal,
        ))?;
        let digest = blake3::hash(&key);
        let bits = u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap());
        let unit = (bits >> 11) as f64 / ((1_u64 << 53) as f64);
        let t_star_ms = cut_ms.unwrap_or(first + (0.25 + 0.5 * unit) * (last - first));
        if !t_star_ms.is_finite() || t_star_ms < first || t_star_ms > last {
            bail!("snapshot cut must lie within recorded request-start bounds [{first}, {last}]");
        }
        let play_id = format!(
            "agentic-v1:{}:{}:{lane_id}:{play_ordinal}",
            self.graph.graph_digest(),
            self.seed
        );
        let cache_id = format!("{play_id}:cache");
        let mut snapshot = AgenticPlaySnapshot {
            context: Arc::clone(self),
            source_play_index,
            slot,
            evidence: AgenticSnapshotEvidence {
                schema: AGENTIC_SNAPSHOT_SCHEMA_V1,
                graph: self.graph.identity(),
                seed: self.seed,
                lane_id,
                play_ordinal,
                source_play_id: play.play_id.clone(),
                play_id,
                cache_id,
                t_star_ms,
                recorded_start_ms: first,
                recorded_last_start_ms: last,
                requests: Vec::new(),
                primers: Vec::new(),
            },
        };
        let index_by_id = play
            .nodes
            .iter()
            .map(|&i| (self.graph.nodes[i].request_id.as_str(), i))
            .collect::<FxHashMap<_, _>>();
        let mut live_conversations = BTreeSet::new();
        let mut predecessor = BTreeMap::<&str, usize>::new();
        for &index in &play.nodes {
            let node = &self.graph.nodes[index];
            let (recorded_start, recorded_end) = self.recorded_intervals[index];
            let historical = recorded_start < t_star_ms;
            if historical
                && node.dependencies.iter().any(|edge| {
                    self.recorded_intervals[index_by_id[edge.request_id.as_str()]].0 >= t_star_ms
                })
            {
                bail!(
                    "snapshot boundary crosses a historical request's unresolved dependency: {}",
                    node.request_id
                );
            }
            if historical {
                let current = predecessor.entry(&node.session_id).or_insert(index);
                if recorded_start
                    .total_cmp(&self.recorded_intervals[*current].0)
                    .then_with(|| {
                        self.topological_rank[index].cmp(&self.topological_rank[*current])
                    })
                    .is_gt()
                {
                    *current = index;
                }
            } else {
                live_conversations.insert(node.session_id.as_str());
            }
            let mut delay = (recorded_start - t_star_ms).max(0.0);
            let mut pending = Vec::new();
            if !historical {
                for edge in &node.dependencies {
                    let (start, end) =
                        self.recorded_intervals[index_by_id[edge.request_id.as_str()]];
                    if start < t_star_ms {
                        let trigger = match edge.trigger {
                            AgenticDependencyTrigger::Dispatch => start,
                            AgenticDependencyTrigger::Completion => end.unwrap_or(start),
                        };
                        let due = trigger + edge.delay_ms;
                        if !due.is_finite() {
                            bail!(
                                "snapshot dependency deadline overflow for {}",
                                node.request_id
                            );
                        }
                        delay = delay.max(due - t_star_ms);
                    } else {
                        pending.push(snapshot.instance_request_id(&edge.request_id));
                    }
                }
            }
            snapshot.evidence.requests.push(AgenticSnapshotRequest {
                source_request_id: node.request_id.clone(),
                identity: snapshot.identity_at(index),
                recorded_start_ms: recorded_start,
                recorded_end_ms: recorded_end,
                historical,
                remaining_delay_ms: if historical { 0.0 } else { delay },
                pending_dependencies: pending,
            });
        }
        for conversation in live_conversations {
            if let Some(&index) = predecessor.get(conversation) {
                let node = &self.graph.nodes[index];
                snapshot.evidence.primers.push(AgenticPrimer {
                    source_request_id: node.request_id.clone(),
                    request_id: snapshot.instance_request_id(&node.request_id),
                    conversation_id: snapshot.instance_conversation_id(&node.session_id),
                    input_length: node.input_length,
                });
            }
        }
        Ok(snapshot)
    }
}

impl AgenticPlaySnapshot {
    pub fn evidence(&self) -> &AgenticSnapshotEvidence {
        &self.evidence
    }
    pub fn context(&self) -> &Arc<AgenticReplayContext> {
        &self.context
    }
    pub(super) fn instance_request_id(&self, source: &str) -> String {
        format!("{}:request:{source}", self.evidence.play_id)
    }
    fn instance_conversation_id(&self, source: &str) -> String {
        format!("{}:conversation:{source}", self.evidence.play_id)
    }
    pub(super) fn source_index(&self, request_id: &str) -> Result<usize> {
        self.context
            .index_by_id
            .get(request_id)
            .copied()
            .filter(|&i| self.context.node_to_play[i] == self.source_play_index)
            .with_context(|| format!("unknown source request {request_id} in snapshot play"))
    }
    pub fn identity(&self, source_request_id: &str) -> Result<AgenticRuntimeIdentity> {
        Ok(self.identity_at(self.source_index(source_request_id)?))
    }
    pub(super) fn identity_at(&self, index: usize) -> AgenticRuntimeIdentity {
        let graph = &self.context.graph;
        let play = &graph.plays[self.source_play_index];
        let node = &graph.nodes[index];
        let parent = node
            .dependencies
            .iter()
            .filter(|e| e.relation == AgenticDependencyRelation::Spawn)
            .min_by_key(|e| self.context.index_by_id.get(&e.request_id));
        AgenticRuntimeIdentity {
            request_id: self.instance_request_id(&node.request_id),
            play_id: self.evidence.play_id.clone(),
            conversation_id: self.instance_conversation_id(&node.session_id),
            lane_id: Some(format!("lane:{}", self.evidence.lane_id)),
            root_id: (play.root_nodes.len() == 1)
                .then(|| self.instance_request_id(&graph.nodes[play.root_nodes[0]].request_id)),
            parent_id: parent.map(|e| self.instance_request_id(&e.request_id)),
            cache_id: Some(self.evidence.cache_id.clone()),
        }
    }
    pub(super) fn token_ids(&self, index: usize) -> Result<Vec<u32>> {
        let base = self
            .slot
            .checked_mul(self.context.token_stride)
            .context("token range overflow")?;
        self.context.ranks[index]
            .iter()
            .map(|&rank| {
                u32::try_from(base + u64::from(rank)).context("token identity exceeds u32 capacity")
            })
            .collect()
    }
    pub(super) fn request_uuid(&self, index: usize) -> Result<Uuid> {
        let ordinal = u128::from(self.slot)
            .checked_mul(self.context.request_stride)
            .and_then(|v| v.checked_add(index as u128 + 1))
            .context("request UUID overflow")?;
        Ok(Uuid::from_u128(ordinal))
    }
    pub fn materialize_prefix(&self, source_request_id: &str, length: usize) -> Result<Vec<u32>> {
        let index = self.source_index(source_request_id)?;
        let node: &AgenticNode = &self.context.graph.nodes[index];
        if length > node.input_length {
            bail!("prefix length exceeds original input length");
        }
        let ids = self.token_ids(index)?;
        super::trace::validate_synthesizable_prompt(
            node.input_length,
            &ids,
            self.context.graph.block_size,
        )?;
        Ok(ids
            .into_iter()
            .flat_map(|id| std::iter::repeat_n(id, self.context.graph.block_size))
            .take(length)
            .collect())
    }
    pub fn replay_hashes(
        &self,
        source_request_id: &str,
        length: usize,
        engine_block_size: usize,
    ) -> Result<ReplayRequestHashes> {
        if engine_block_size == 0 {
            bail!("engine_block_size must be greater than 0");
        }
        let block_size =
            u32::try_from(engine_block_size).context("engine block size exceeds u32")?;
        let tokens = self.materialize_prefix(source_request_id, length)?;
        Ok(ReplayRequestHashes::from_tokens(&tokens, block_size))
    }
}

impl PreparedAgenticSnapshots {
    pub fn context(&self) -> &Arc<AgenticReplayContext> {
        &self.context
    }
    pub fn snapshots(&self) -> &[AgenticSnapshotEvidence] {
        &self.evidence
    }
    pub fn from_plays(plays: Vec<AgenticPlaySnapshot>) -> Result<Self> {
        let context = Arc::clone(
            &plays
                .first()
                .context("snapshot cohort must not be empty")?
                .context,
        );
        let mut lanes = BTreeSet::new();
        for play in &plays {
            if !Arc::ptr_eq(&context, &play.context) {
                bail!("snapshot views must retain the same replay context");
            }
            if !lanes.insert(play.evidence.lane_id) {
                bail!("snapshot cohort contains a duplicate lane");
            }
        }
        let evidence = plays.iter().map(|p| p.evidence.clone()).collect();
        Ok(Self {
            context,
            plays,
            evidence,
        })
    }
}
