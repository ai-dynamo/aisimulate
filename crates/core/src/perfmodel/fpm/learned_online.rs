// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native online squared-error fitting. Algorithm provenance and the differences
//! from Lin et al. (arXiv:2502.01634v1) are documented in
//! docs/perf-model/fpm_ml/online-gbdt-and-causal-selection.md. The exact local source import is recorded in
//! benchmarks/evidence/fpm_online_publication_v1/source-import.json.
//! This is an independent implementation, not a translation of the authors' code.

mod checkpoint;
mod fitting;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::estimator::{
    FpmLearnedConfig, LearnedAdaptation, LearnedDerivativeRefresh, LearnedUpdateMode,
};
use super::learned::{LearnedForwardPassModel, LearnedTarget, LearnedTree};
use super::{ForwardPassPerfModelConfig, ForwardPassRegressionWorkloadKind as Kind};
use crate::AicError;

#[cfg(test)]
use checkpoint::checkpoint_model;
pub(crate) use checkpoint::{canonical_learned_identity, checkpoint_model_for_config};

fn invalid(message: impl Into<String>) -> AicError {
    AicError::InvalidEngineConfig(format!("online learned model: {}", message.into()))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FitSettings {
    learning_rate: f64,
    l2_regularization: f64,
    min_samples_leaf: usize,
    max_depth: Option<usize>,
}

/// Immutable row-major anchors. Sharing this storage never shares mutable fits.
#[derive(Debug, Serialize, Deserialize)]
struct Anchors {
    ids: Vec<u64>,
    y: Vec<f64>,
    x: Vec<f64>,
    bins: Vec<u16>,
    width: usize,
    /// None preserves the original per-row storage and arithmetic path.
    groups: Option<AnchorGroups>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnchorGroups {
    /// One count per stored representative; `Anchors::y` contains target sums.
    counts: Vec<u64>,
    original_rows: usize,
    original_max_id: u64,
    original_ids_sha256: String,
}

impl Anchors {
    fn original_rows(&self) -> usize {
        self.groups
            .as_ref()
            .map_or(self.y.len(), |g| g.original_rows)
    }

    #[inline]
    fn count(&self, row: usize) -> u64 {
        self.groups.as_ref().map_or(1, |g| g.counts[row])
    }

    /// First-occurrence ordering keeps representatives deterministic. The key
    /// includes every feature and its missing bin, so all allowed splits route
    /// a group's members identically. Original fitting validation precedes this.
    fn group_by_bins(&self) -> Result<Self, AicError> {
        if self.groups.is_some() {
            return Err(invalid("history is already bin-grouped"));
        }
        let mut index = BTreeMap::<Vec<u16>, usize>::new();
        let mut out = Self {
            ids: Vec::new(),
            y: Vec::new(),
            x: Vec::new(),
            bins: Vec::new(),
            width: self.width,
            groups: None,
        };
        let mut counts = Vec::<u64>::new();
        let mut compensation = Vec::<f64>::new();
        for row in 0..self.y.len() {
            let range = row * self.width..(row + 1) * self.width;
            let bins = &self.bins[range.clone()];
            if let Some(&group) = index.get(bins) {
                counts[group] += 1;
                // Compensated target summation limits error for large groups.
                let corrected = self.y[row] - compensation[group];
                let next = out.y[group] + corrected;
                compensation[group] = (next - out.y[group]) - corrected;
                out.y[group] = next;
            } else {
                index.insert(bins.to_vec(), out.y.len());
                out.ids.push(self.ids[row]);
                out.y.push(self.y[row]);
                out.x.extend_from_slice(&self.x[range]);
                out.bins.extend_from_slice(bins);
                counts.push(1);
                compensation.push(0.0);
            }
        }
        if out.y.iter().any(|y| !y.is_finite()) {
            return Err(invalid("nonfinite grouped historical target sum"));
        }
        out.groups = Some(AnchorGroups {
            counts,
            original_rows: self.y.len(),
            original_max_id: self.ids.iter().copied().max().unwrap_or(0),
            original_ids_sha256: membership_hash(self.ids.iter().copied()),
        });
        Ok(out)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Row {
    id: u64,
    x: Vec<f64>,
    bins: Vec<u16>,
    y: f64,
    wall_ms: f64,
    /// Needed only in lazy mode; derivatives are authoritative, possibly stale.
    gradients: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
struct Sum {
    // Unit-weight sufficient statistics split by group. Keeping both raw groups
    // makes changing their fitting masses exact, including cached derivatives.
    g: f64,
    count: u64,
    recent_g: f64,
    recent_count: u64,
}

#[derive(Clone, Copy, Debug)]
struct GroupWeights {
    history: f64,
    recent: f64,
}

impl GroupWeights {
    fn new(config: &FpmLearnedConfig, history: usize, recent: usize) -> Self {
        let Some(beta) = config.history_weight else {
            return Self {
                history: 1.0,
                recent: 1.0,
            };
        };
        if recent == 0 || history == 0 {
            return Self {
                history: 1.0,
                recent: 1.0,
            };
        }
        let total = (history + recent) as f64;
        let support = if config.history_weight_ramp_observations == 0 {
            1.0
        } else {
            (recent as f64 / config.history_weight_ramp_observations as f64).min(1.0)
        };
        let recent_mass = (1.0 - support) * recent as f64 / total + support * (1.0 - beta);
        Self {
            history: total * (1.0 - recent_mass) / history as f64,
            recent: total * recent_mass / recent as f64,
        }
    }
    fn gradient(self, sum: Sum) -> f64 {
        if self.history == 1.0 && self.recent == 1.0 {
            return sum.g;
        }
        self.history * (sum.g - sum.recent_g) + self.recent * sum.recent_g
    }
    fn hessian(self, sum: Sum) -> f64 {
        if self.history == 1.0 && self.recent == 1.0 {
            return sum.count as f64;
        }
        self.history * (sum.count - sum.recent_count) as f64 + self.recent * sum.recent_count as f64
    }
}

fn mutable_start(config: &FpmLearnedConfig, trees: usize) -> usize {
    trees.saturating_sub(config.mutable_tree_tail.unwrap_or(trees))
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
struct NodeStats {
    total: Sum,
    /// Feature-major histograms, including a final missing bin per feature.
    histogram: Vec<Sum>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Candidate {
    feature: u16,
    bin: u16,
    missing_left: bool,
}

impl Candidate {
    fn new(feature: usize, bin: usize, missing_left: bool) -> Result<Self, AicError> {
        Ok(Self {
            feature: u16::try_from(feature)
                .map_err(|_| invalid("candidate feature exceeds supported index range"))?,
            bin: u16::try_from(bin)
                .map_err(|_| invalid("candidate bin exceeds supported index range"))?,
            missing_left,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TreeState {
    /// Includes the immutable anchors' possibly stale derivative mass. Only
    /// recent rows need individual gradients for removal; rebuilds are eager.
    stats: Vec<NodeStats>,
    /// Reserved arena slots for each region. A collapsed region retains them.
    regions: Vec<Vec<usize>>,
    candidates: Vec<Vec<Candidate>>,
    generation: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Counters {
    accepted: u64,
    evicted: u64,
    publications: u64,
    eager_updates: u64,
    lazy_updates: u64,
    lazy_fallbacks: u64,
    rebuilt_nodes: u64,
    failures: u64,
    batches_since_eager: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoreState {
    anchors: Arc<Anchors>,
    /// Derived from immutable anchors and trees before the mutable tail.
    /// Rebuilt on checkpoint load; never part of resumable training state.
    #[serde(skip)]
    frozen_anchor_prefix: Vec<f64>,
    /// Load-time measurement, not part of resumable fitting state.
    #[serde(skip)]
    history_grouping_ms: Option<f64>,
    thresholds: Vec<Vec<f64>>,
    offsets: Vec<usize>,
    settings: FitSettings,
    trees: Vec<TreeState>,
    recent: VecDeque<Row>,
    pending: Vec<Row>,
    counters: Counters,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct OnlineLearner {
    config: FpmLearnedConfig,
    /// JSON text avoids serde_json::Value's deserialize_any in binary state.
    identity_json: String,
    anchor_manifest_sha256: String,
    stores: BTreeMap<Kind, StoreState>,
    next_row_id: u64,
    version: u64,
    rejected_observations: u64,
    #[serde(skip)]
    restored_corrections: Option<String>,
}

impl OnlineLearner {
    pub(crate) fn load(
        path: &Path,
        learned: &mut LearnedForwardPassModel,
        config: &FpmLearnedConfig,
        identity: &ForwardPassPerfModelConfig,
    ) -> Result<Self, AicError> {
        checkpoint::load(path, learned, config, identity)
    }

    pub(crate) fn restored_correction_json(&self) -> Option<&str> {
        self.restored_corrections.as_deref()
    }

    /// Count public tuning inputs rejected before an updater row is constructed.
    pub(crate) fn record_rejected_observation(&mut self) {
        self.rejected_observations = self.rejected_observations.saturating_add(1);
    }

    pub(crate) fn save(
        &self,
        path: &Path,
        learned: &LearnedForwardPassModel,
        correction_json: &str,
    ) -> Result<(), AicError> {
        checkpoint::save(path, learned, self, correction_json)
    }

    pub(crate) fn observe(
        &mut self,
        kind: Kind,
        x: Vec<f64>,
        wall_ms: f64,
        learned: &mut LearnedForwardPassModel,
    ) -> Result<bool, AicError> {
        let Some(state) = self.stores.get_mut(&kind) else {
            self.rejected_observations += 1;
            return Err(invalid(format!(
                "no training checkpoint for store {kind:?}"
            )));
        };
        if x.len() != state.anchors.width
            || x.iter().any(|v| v.is_infinite())
            || !wall_ms.is_finite()
            || wall_ms <= 0.0
        {
            self.rejected_observations += 1;
            return Err(invalid("invalid observation features or latency"));
        }

        // A failed fitting batch remains pending for explicit retry. Do not let
        // ignored errors create an unbounded pending queue.
        if state.pending.len() >= self.config.update_every {
            return Err(invalid(
                "a failed batch is pending; retry flush_updates before adding rows",
            ));
        }
        let id = self.next_row_id;
        self.next_row_id = id
            .checked_add(1)
            .ok_or_else(|| invalid("row ID overflow"))?;
        let y = match learned.target() {
            LearnedTarget::LogMs => wall_ms.ln(),
            LearnedTarget::Ms => wall_ms,
        };
        let bins = bin_row(&x, &state.thresholds);
        state.pending.push(Row {
            id,
            x,
            bins,
            y,
            wall_ms,
            gradients: Vec::new(),
        });
        state.counters.accepted += 1;
        if state.pending.len() >= self.config.update_every {
            self.flush(kind, learned)
        } else {
            Ok(false)
        }
    }

    pub(crate) fn flush(
        &mut self,
        kind: Kind,
        learned: &mut LearnedForwardPassModel,
    ) -> Result<bool, AicError> {
        let Some(state) = self.stores.get_mut(&kind) else {
            return Ok(false);
        };
        let oldest_retained_id = self
            .config
            .recent_max_age_observations
            .map(|age| self.next_row_id.saturating_sub(age as u64));
        let has_expired_rows = oldest_retained_id
            .is_some_and(|oldest| state.recent.front().is_some_and(|row| row.id < oldest));
        if state.pending.is_empty() && !has_expired_rows {
            return Ok(false);
        }
        let store = learned
            .store(kind)
            .ok_or_else(|| invalid("missing predictor store"))?;
        match fitting::fit_batch(
            state,
            store,
            &self.config,
            &learned.target(),
            self.next_row_id,
        ) {
            Ok(Some((replacement, eager, fallback, rebuilt))) => {
                learned.replace_store(kind, replacement);
                state.counters.publications += 1;
                state.counters.rebuilt_nodes += rebuilt;
                if eager {
                    state.counters.eager_updates += 1;
                    state.counters.batches_since_eager = 0;
                } else {
                    state.counters.lazy_updates += 1;
                    state.counters.batches_since_eager += 1;
                }
                state.counters.lazy_fallbacks += u64::from(fallback);
                self.version += 1;
                Ok(true)
            }
            Ok(None) => Ok(false),
            Err(error) => {
                state.counters.failures += 1;
                Err(error)
            }
        }
    }

    /// Accepted recent and pending rows, excluding immutable historical anchors.
    pub(crate) fn retained_observations(&self) -> usize {
        self.stores
            .values()
            .map(|state| state.recent.len() + state.pending.len())
            .sum()
    }

    pub(crate) fn retained_observations_for(&self, kind: Kind) -> usize {
        self.stores
            .get(&kind)
            .map_or(0, |state| state.recent.len() + state.pending.len())
    }

    pub(crate) fn diagnostics(&self) -> serde_json::Value {
        let stores: BTreeMap<_, _> = self.stores.iter().map(|(kind, state)| {
            let bytes = state.estimated_bytes();
            (*kind, serde_json::json!({
                "anchor_rows": state.anchors.original_rows(), "recent_rows": state.recent.len(),
                "historical_storage_rows": state.anchors.y.len(),
                "history_grouping_ms": state.history_grouping_ms,
                "history_bin_grouping": self.config.history_bin_grouping,
                "history_compression_ratio": state.anchors.original_rows() as f64 / state.anchors.y.len() as f64,
                "historical_original_ids_sha256": state.anchors.groups.as_ref().map(|g| &g.original_ids_sha256),
                "pending_rows": state.pending.len(), "training_state_bytes": bytes,
                "retained_row_ids": state.recent.iter().map(|r| r.id).collect::<Vec<_>>(),
                "retained_row_ids_sha256": membership_hash(state.recent.iter().map(|r| r.id)),
                "recent_sampling": "fifo",
                "history_weight": self.config.history_weight,
                "history_row_weight": GroupWeights::new(&self.config, state.anchors.original_rows(), state.recent.len()).history,
                "recent_row_weight": GroupWeights::new(&self.config, state.anchors.original_rows(), state.recent.len()).recent,
                "history_weight_ramp_observations": self.config.history_weight_ramp_observations,
                "mutable_tree_tail": self.config.mutable_tree_tail,
                "update_step": self.config.update_step,
                "recent_max_age_observations": self.config.recent_max_age_observations,
                "accepted_observations": state.counters.accepted,
                "evicted_observations": state.counters.evicted,
                "publications": state.counters.publications,
                "eager_updates": state.counters.eager_updates, "lazy_updates": state.counters.lazy_updates,
                "lazy_fallbacks": state.counters.lazy_fallbacks,
                "rebuilt_nodes": state.counters.rebuilt_nodes, "update_failures": state.counters.failures,
                "batches_since_eager": state.counters.batches_since_eager,
            }))
        }).collect();
        serde_json::json!({
            "adaptation": self.config.adaptation,
            "update_mode": self.config.update_mode, "derivative_refresh": self.config.derivative_refresh,
            "model_version": self.version, "rejected_observations": self.rejected_observations,
            "training_state_bytes": self.stores.values().map(StoreState::estimated_bytes).sum::<usize>(),
            "stores": stores,
        })
    }
}

fn membership_hash(ids: impl Iterator<Item = u64>) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    for id in ids {
        hash.update(id.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

impl StoreState {
    fn estimated_bytes(&self) -> usize {
        let anchors = self.anchors.ids.capacity() * 8
            + self.anchors.y.capacity() * 8
            + self.anchors.x.capacity() * 8
            + self.anchors.bins.capacity() * 2
            + self.anchors.groups.as_ref().map_or(0, |g| {
                std::mem::size_of::<AnchorGroups>()
                    + g.counts.capacity() * 8
                    + g.original_ids_sha256.capacity()
            });
        let rows = self
            .recent
            .iter()
            .chain(self.pending.iter())
            .map(|r| {
                std::mem::size_of::<Row>()
                    + r.x.capacity() * 8
                    + r.bins.capacity() * 2
                    + r.gradients.capacity() * 8
            })
            .sum::<usize>();
        let trees = self
            .trees
            .iter()
            .map(|t| {
                t.stats
                    .iter()
                    .map(|s| {
                        std::mem::size_of::<NodeStats>()
                            + s.histogram.capacity() * std::mem::size_of::<Sum>()
                    })
                    .sum::<usize>()
                    + t.regions
                        .iter()
                        .map(|r| {
                            std::mem::size_of::<Vec<usize>>()
                                + r.capacity() * std::mem::size_of::<usize>()
                        })
                        .sum::<usize>()
                    + t.candidates
                        .iter()
                        .map(|c| {
                            std::mem::size_of::<Vec<Candidate>>()
                                + c.capacity() * std::mem::size_of::<Candidate>()
                        })
                        .sum::<usize>()
            })
            .sum::<usize>();
        anchors
            + rows
            + trees
            + self.frozen_anchor_prefix.capacity() * std::mem::size_of::<f64>()
            + self
                .thresholds
                .iter()
                .map(|v| v.capacity() * 8)
                .sum::<usize>()
    }
}

fn bin_row(x: &[f64], thresholds: &[Vec<f64>]) -> Vec<u16> {
    x.iter()
        .zip(thresholds)
        .map(|(&v, bounds)| {
            if v.is_nan() {
                (bounds.len() + 1) as u16
            } else {
                bounds.partition_point(|&b| v > b) as u16
            }
        })
        .collect()
}

/// Stable input order is part of the floating-point accumulation contract.
struct Rows<'a> {
    anchors: &'a Anchors,
    recent: Vec<&'a Row>,
}

impl<'a> Rows<'a> {
    fn len(&self) -> usize {
        self.anchors.y.len() + self.recent.len()
    }
    #[inline]
    fn x(&self, index: usize) -> &[f64] {
        let n = self.anchors.y.len();
        if index < n {
            let start = index * self.anchors.width;
            &self.anchors.x[start..start + self.anchors.width]
        } else {
            &self.recent[index - n].x
        }
    }
    #[inline]
    fn bins(&self, index: usize) -> &[u16] {
        let n = self.anchors.y.len();
        if index < n {
            let start = index * self.anchors.width;
            &self.anchors.bins[start..start + self.anchors.width]
        } else {
            &self.recent[index - n].bins
        }
    }
    #[inline]
    fn count(&self, index: usize) -> u64 {
        if index < self.anchors.y.len() {
            self.anchors.count(index)
        } else {
            1
        }
    }
    #[inline]
    fn gradient(&self, index: usize, prediction: f64) -> f64 {
        let n = self.anchors.y.len();
        if index < n {
            let count = self.anchors.count(index);
            if count == 1 {
                prediction - self.anchors.y[index]
            } else {
                prediction * count as f64 - self.anchors.y[index]
            }
        } else {
            prediction - self.recent[index - n].y
        }
    }
}

fn leaf_index(tree: &LearnedTree, x: &[f64]) -> usize {
    let mut node = 0;
    while tree.left[node] >= 0 {
        let value = x[tree.feature[node] as usize];
        let left = if value.is_nan() {
            tree.missing_left.get(node).copied().unwrap_or(true)
        } else {
            value <= tree.threshold[node]
        };
        node = if left {
            tree.left[node] as usize
        } else {
            tree.right[node] as usize
        };
    }
    node
}
