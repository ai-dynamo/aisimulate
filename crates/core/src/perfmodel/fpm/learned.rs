// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Learned forward-pass model: an offline-trained tree ensemble over
//! `ForwardPassMetrics` batch-composition features.
//!
//! Training data is the per-iteration FPM telemetry recorded by a real
//! deployment (Dynamo `DYN_FPM_TRACE`), so the model learns the deployed
//! engine's actual step time as a function of the scheduled batch. The
//! artifact is a plain JSON file written by
//! `aiconfigurator_core.sdk.fpm_learned` (Python, scikit-learn); this module
//! is the pure-Rust inference half so the simulation hot path never re-enters
//! Python.
//!
//! Artifact contract (`schema = "aic_fpm_learned_forward_perf"`, version 1):
//!
//! ```json
//! {
//!   "schema": "aic_fpm_learned_forward_perf",
//!   "schema_version": 1,
//!   "worker_type": "decode",
//!   "target": "log_ms",
//!   "features": ["num_decode_requests", "sum_decode_kv_tokens", "..."],
//!   "stores": {
//!     "pure_decode": {
//!       "baseline": 2.7,
//!       "trees": [{"left": [1, -1, -1], "right": [2, -1, -1], "feature": [0, -1, -1],
//!                  "threshold": [4.5, 0.0, 0.0], "value": [0.0, -0.1, 0.2],
//!                  "missing_left": [true, true, true]}]
//!     }
//!   },
//!   "metadata": {}
//! }
//! ```
//!
//! `features` is the ordered feature ABI: every name must be one of
//! [`feature_names`], and the trainer computes the same named quantities from
//! the same FPM fields (`fpm_learned.compute_features`). Each store is one
//! additive tree ensemble bound to a regression workload kind; a query whose
//! workload kind has no store returns `None`, mirroring an unready regression
//! store. Prediction is `baseline + sum(leaf values)`, exponentiated when
//! `target == "log_ms"`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::metrics::ForwardPassMetrics;
use super::model::{ForwardPassRegressionWorkloadKind, ForwardPassWorkerType};
use super::options::ForwardPassPerfOptions;
use crate::AicError;

pub const LEARNED_SCHEMA_NAME: &str = "aic_fpm_learned_forward_perf";
pub const LEARNED_SCHEMA_VERSION: u32 = 1;

/// Aggregate iteration features derivable from FPM v1 scheduled fields.
///
/// Sums run over all attention-DP ranks; `max_rank_*` and variances take the
/// per-rank maximum, mirroring the "heaviest rank sets the step time"
/// lockstep rule used by the correction path.
pub const AGGREGATE_FEATURE_NAMES: [&str; 21] = [
    "num_active_ranks",
    "num_prefill_requests",
    "sum_prefill_tokens",
    "sum_prefill_kv_tokens",
    "var_prefill_length",
    "num_decode_requests",
    "sum_decode_kv_tokens",
    "var_decode_kv_tokens",
    "max_rank_prefill_tokens",
    "max_rank_decode_kv_tokens",
    "max_rank_num_decode_requests",
    "mean_prefill_chunk",
    "mean_prefill_kv",
    "prefill_attention_pairs",
    "mean_decode_kv",
    "sum_decode_kv_squared",
    "log1p_sum_prefill_tokens",
    "log1p_sum_prefill_kv_tokens",
    "log1p_sum_decode_kv_tokens",
    "log1p_prefill_attention_pairs",
    "log1p_sum_decode_kv_squared",
];

/// Per-request features over the iteration's `(extend, past)` list (all ranks
/// concatenated). These are the 18 features of the SGLang simulator
/// `MLTimePredictor` (`tools/sglang-simulator/.../time_predictor/ml.py`),
/// prefixed `req_`. They are NaN when the producer did not emit
/// `extend_lengths` / `past_kv_lengths`; tree models route NaN through
/// `missing_left`.
pub const REQUEST_FEATURE_NAMES: [&str; 18] = [
    "req_batch_size",
    "req_sum_extend",
    "req_max_extend",
    "req_min_extend",
    "req_sum_past",
    "req_max_past",
    "req_min_past",
    "req_sum_extend_x_past",
    "req_sum_extend_squared",
    "req_sum_past_squared",
    "req_sum_attn_flops",
    "req_sum_extend_x_max_past",
    "req_log1p_sum_past",
    "req_log1p_sum_attn_flops",
    "req_batch_size_x_sum_extend",
    "req_max_past_minus_min_past",
    "req_is_decode",
    "req_is_prefill",
];

/// Number of HiSim-style request slots. Requests are sorted by past KV
/// descending; slot `i` exposes `(present, past, extend)` like the padded
/// `(max_bs, 2)` feature of HiSim's decode XGBoost ratio model
/// (`hisim/time_predictor/aiconfigurator.py::_build_xgb_feature_maxbs_2`),
/// extended with the request's extend length so prefill chunks are covered too.
/// Requests beyond the last slot fold into the aggregate/`req_*` features only.
pub const SLOT_COUNT: usize = 32;

/// Full ordered feature table: aggregates, per-request, then
/// `slot{i}_present`, `slot{i}_past`, `slot{i}_extend` for `i < SLOT_COUNT`.
pub fn feature_names() -> Vec<String> {
    let mut names: Vec<String> = AGGREGATE_FEATURE_NAMES
        .iter()
        .chain(REQUEST_FEATURE_NAMES.iter())
        .map(|name| name.to_string())
        .collect();
    for slot in 0..SLOT_COUNT {
        names.push(format!("slot{slot}_present"));
        names.push(format!("slot{slot}_past"));
        names.push(format!("slot{slot}_extend"));
    }
    names
}

/// Total feature count (`feature_names().len()`).
pub const FEATURE_COUNT: usize =
    AGGREGATE_FEATURE_NAMES.len() + REQUEST_FEATURE_NAMES.len() + 3 * SLOT_COUNT;

fn feature_index(name: &str) -> Option<usize> {
    if let Some(index) = AGGREGATE_FEATURE_NAMES
        .iter()
        .position(|known| *known == name)
    {
        return Some(index);
    }
    if let Some(index) = REQUEST_FEATURE_NAMES
        .iter()
        .position(|known| *known == name)
    {
        return Some(AGGREGATE_FEATURE_NAMES.len() + index);
    }
    let rest = name.strip_prefix("slot")?;
    let (slot, kind) = rest.split_once('_')?;
    let slot: usize = slot.parse().ok()?;
    if slot >= SLOT_COUNT {
        return None;
    }
    let offset = match kind {
        "present" => 0,
        "past" => 1,
        "extend" => 2,
        _ => return None,
    };
    Some(AGGREGATE_FEATURE_NAMES.len() + REQUEST_FEATURE_NAMES.len() + 3 * slot + offset)
}

const MIN_POSITIVE_PREDICTION_MS: f64 = 1e-6;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LearnedTarget {
    LogMs,
    Ms,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct LearnedTree {
    pub(crate) left: Vec<i32>,
    pub(crate) right: Vec<i32>,
    pub(crate) feature: Vec<i32>,
    pub(crate) threshold: Vec<f64>,
    pub(crate) value: Vec<f64>,
    #[serde(default)]
    pub(crate) missing_left: Vec<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct LearnedStore {
    pub(crate) baseline: f64,
    pub(crate) trees: Vec<LearnedTree>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct LearnedArtifact {
    pub(crate) schema: String,
    pub(crate) schema_version: u32,
    pub(crate) worker_type: ForwardPassWorkerType,
    pub(crate) target: LearnedTarget,
    pub(crate) features: Vec<String>,
    pub(crate) stores: BTreeMap<ForwardPassRegressionWorkloadKind, LearnedStore>,
    #[serde(default)]
    pub(crate) metadata: serde_json::Value,
}

/// Validated, query-ready learned model.
#[derive(Clone, Debug)]
pub(crate) struct LearnedForwardPassModel {
    worker_type: ForwardPassWorkerType,
    target: LearnedTarget,
    feature_indices: Vec<usize>,
    feature_names: Vec<String>,
    stores: BTreeMap<ForwardPassRegressionWorkloadKind, LearnedStore>,
    metadata: serde_json::Value,
}

fn invalid(message: impl Into<String>) -> AicError {
    AicError::InvalidEngineConfig(format!("learned forward-pass model: {}", message.into()))
}

impl LearnedForwardPassModel {
    pub(crate) fn from_path(path: &Path) -> Result<Self, AicError> {
        let text = std::fs::read_to_string(path).map_err(|source| AicError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_json(&text)
    }

    pub(crate) fn from_json(text: &str) -> Result<Self, AicError> {
        let artifact: LearnedArtifact =
            serde_json::from_str(text).map_err(|e| invalid(format!("invalid JSON: {e}")))?;
        Self::from_artifact(artifact)
    }

    pub(crate) fn from_artifact(artifact: LearnedArtifact) -> Result<Self, AicError> {
        if artifact.schema != LEARNED_SCHEMA_NAME {
            return Err(invalid(format!(
                "schema {:?} != {LEARNED_SCHEMA_NAME:?}",
                artifact.schema
            )));
        }
        if artifact.schema_version != LEARNED_SCHEMA_VERSION {
            return Err(invalid(format!(
                "schema_version {} != {LEARNED_SCHEMA_VERSION}",
                artifact.schema_version
            )));
        }
        if artifact.features.is_empty() {
            return Err(invalid("features list is empty"));
        }
        let mut feature_indices = Vec::with_capacity(artifact.features.len());
        for name in &artifact.features {
            let index =
                feature_index(name).ok_or_else(|| invalid(format!("unknown feature {name:?}")))?;
            feature_indices.push(index);
        }
        if artifact.stores.is_empty() {
            return Err(invalid("no stores"));
        }
        let allowed = allowed_workload_kinds(artifact.worker_type);
        for (kind, store) in &artifact.stores {
            if !allowed.contains(kind) {
                return Err(invalid(format!(
                    "store {kind:?} is not valid for worker_type {:?}",
                    artifact.worker_type
                )));
            }
            if !store.baseline.is_finite() {
                return Err(invalid(format!("store {kind:?} baseline is not finite")));
            }
            for (tree_index, tree) in store.trees.iter().enumerate() {
                validate_tree(tree, artifact.features.len())
                    .map_err(|e| invalid(format!("store {kind:?} tree {tree_index}: {e}")))?;
            }
        }
        Ok(Self {
            worker_type: artifact.worker_type,
            target: artifact.target,
            feature_indices,
            feature_names: artifact.features,
            stores: artifact.stores,
            metadata: artifact.metadata,
        })
    }

    pub(crate) fn worker_type(&self) -> ForwardPassWorkerType {
        self.worker_type
    }

    pub(crate) fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    pub(crate) fn metadata(&self) -> &serde_json::Value {
        &self.metadata
    }

    pub(crate) fn has_store(&self, kind: ForwardPassRegressionWorkloadKind) -> bool {
        self.stores.contains_key(&kind)
    }

    pub(crate) fn store_kinds(&self) -> Vec<ForwardPassRegressionWorkloadKind> {
        self.stores.keys().copied().collect()
    }

    /// Predict milliseconds for one iteration's named feature vector. Returns
    /// `None` when the workload kind has no trained store.
    pub(crate) fn predict_ms(
        &self,
        kind: ForwardPassRegressionWorkloadKind,
        features: &IterationFeatureVector,
    ) -> Option<f64> {
        let store = self.stores.get(&kind)?;
        let x: Vec<f64> = self
            .feature_indices
            .iter()
            .map(|&index| features.values[index])
            .collect();
        let mut raw = store.baseline;
        for tree in &store.trees {
            raw += tree_leaf_value(tree, &x);
        }
        let ms = match self.target {
            LearnedTarget::LogMs => raw.exp(),
            LearnedTarget::Ms => raw,
        };
        if !ms.is_finite() {
            return None;
        }
        Some(ms.max(MIN_POSITIVE_PREDICTION_MS))
    }
}

fn allowed_workload_kinds(
    worker_type: ForwardPassWorkerType,
) -> &'static [ForwardPassRegressionWorkloadKind] {
    use ForwardPassRegressionWorkloadKind::*;
    match worker_type {
        ForwardPassWorkerType::Prefill => &[PurePrefill],
        ForwardPassWorkerType::Decode => &[PureDecode],
        ForwardPassWorkerType::Aggregated => &[
            PureDecode,
            ContainsLocallyMixed,
            CrossRankAggregated,
            PurePrefill,
        ],
    }
}

fn validate_tree(tree: &LearnedTree, n_features: usize) -> Result<(), String> {
    let n = tree.value.len();
    if n == 0 {
        return Err("empty tree".to_string());
    }
    if tree.left.len() != n
        || tree.right.len() != n
        || tree.feature.len() != n
        || tree.threshold.len() != n
    {
        return Err("node arrays have mismatched lengths".to_string());
    }
    if !tree.missing_left.is_empty() && tree.missing_left.len() != n {
        return Err("missing_left length mismatch".to_string());
    }
    for i in 0..n {
        let (l, r) = (tree.left[i], tree.right[i]);
        let is_leaf = l < 0 && r < 0;
        if is_leaf {
            if !tree.value[i].is_finite() {
                return Err(format!("leaf {i} value is not finite"));
            }
            continue;
        }
        if l < 0 || r < 0 || l as usize >= n || r as usize >= n {
            return Err(format!("node {i} has out-of-range children ({l}, {r})"));
        }
        if l as usize <= i || r as usize <= i {
            // Children must come after their parent so a forward walk terminates.
            return Err(format!("node {i} children must have larger indices"));
        }
        let f = tree.feature[i];
        if f < 0 || f as usize >= n_features {
            return Err(format!("node {i} feature index {f} out of range"));
        }
        if !tree.threshold[i].is_finite() {
            return Err(format!("node {i} threshold is not finite"));
        }
    }
    Ok(())
}

fn tree_leaf_value(tree: &LearnedTree, x: &[f64]) -> f64 {
    let mut node = 0usize;
    loop {
        let left = tree.left[node];
        let right = tree.right[node];
        if left < 0 && right < 0 {
            return tree.value[node];
        }
        let value = x[tree.feature[node] as usize];
        let go_left = if value.is_nan() {
            tree.missing_left.get(node).copied().unwrap_or(true)
        } else {
            value <= tree.threshold[node]
        };
        node = if go_left {
            left as usize
        } else {
            right as usize
        };
    }
}

/// Named feature values for one iteration, indexed like [`feature_names`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IterationFeatureVector {
    pub(crate) values: Vec<f64>,
}

impl IterationFeatureVector {
    pub(crate) fn get(&self, name: &str) -> Option<f64> {
        feature_index(name).map(|index| self.values[index])
    }

    /// Compute the named features from one iteration's per-rank FPMs. Callers
    /// validate the metrics first; ranks with no scheduled work are skipped.
    pub(crate) fn from_metrics(metrics_by_rank: &[ForwardPassMetrics]) -> Self {
        let mut num_active_ranks = 0.0_f64;
        let mut num_prefill = 0.0_f64;
        let mut sum_ptok = 0.0_f64;
        let mut sum_pkv = 0.0_f64;
        let mut var_plen = 0.0_f64;
        let mut num_decode = 0.0_f64;
        let mut sum_dkv = 0.0_f64;
        let mut var_dkv = 0.0_f64;
        let mut max_rank_ptok = 0.0_f64;
        let mut max_rank_dkv = 0.0_f64;
        let mut max_rank_nd = 0.0_f64;
        let mut attention_pairs = 0.0_f64;
        let mut sum_dkv_sq = 0.0_f64;
        // Per-request pairs over all active ranks; `None` when any active rank
        // lacks the lists (mixed producers make per-request features unreliable).
        let mut pairs: Option<Vec<(f64, f64)>> = Some(Vec::new());

        for metrics in metrics_by_rank {
            let s = &metrics.scheduled_requests;
            let np = f64::from(s.num_prefill_requests);
            let ptok = f64::from(s.sum_prefill_tokens);
            let pkv = f64::from(s.sum_prefill_kv_tokens);
            let nd = f64::from(s.num_decode_requests);
            let dkv = f64::from(s.sum_decode_kv_tokens);
            let has_prefill = ptok > 0.0;
            let has_decode = nd > 0.0;
            if !has_prefill && !has_decode {
                continue;
            }
            num_active_ranks += 1.0;
            num_prefill += np;
            sum_ptok += ptok;
            sum_pkv += pkv;
            var_plen = var_plen.max(s.var_prefill_length.max(0.0));
            num_decode += nd;
            sum_dkv += dkv;
            var_dkv = var_dkv.max(s.var_decode_kv_tokens.max(0.0));
            max_rank_ptok = max_rank_ptok.max(ptok);
            max_rank_dkv = max_rank_dkv.max(dkv);
            max_rank_nd = max_rank_nd.max(nd);
            if has_prefill && np > 0.0 {
                // Balanced-partition attention proxy: sum over requests of
                // e * (p + (e + 1) / 2) with e = ptok / np and p = pkv / np.
                attention_pairs += pkv * ptok / np + ptok * ptok / (2.0 * np) + ptok / 2.0;
            }
            if has_decode {
                // Recover sum(kv_i^2) from the population variance:
                // n * var + n * mean^2.
                let mean = dkv / nd;
                sum_dkv_sq += nd * s.var_decode_kv_tokens.max(0.0) + nd * mean * mean;
            }
            match pairs.as_mut() {
                Some(list)
                    if !s.extend_lengths.is_empty()
                        && s.extend_lengths.len() == s.past_kv_lengths.len() =>
                {
                    list.extend(
                        s.extend_lengths
                            .iter()
                            .zip(s.past_kv_lengths.iter())
                            .map(|(&e, &p)| (f64::from(e), f64::from(p))),
                    );
                }
                _ => pairs = None,
            }
        }

        let mean_prefill_chunk = if num_prefill > 0.0 {
            sum_ptok / num_prefill
        } else {
            0.0
        };
        let mean_prefill_kv = if num_prefill > 0.0 {
            sum_pkv / num_prefill
        } else {
            0.0
        };
        let mean_decode_kv = if num_decode > 0.0 {
            sum_dkv / num_decode
        } else {
            0.0
        };

        let mut values = Vec::with_capacity(FEATURE_COUNT);
        values.extend_from_slice(&[
            num_active_ranks,
            num_prefill,
            sum_ptok,
            sum_pkv,
            var_plen,
            num_decode,
            sum_dkv,
            var_dkv,
            max_rank_ptok,
            max_rank_dkv,
            max_rank_nd,
            mean_prefill_chunk,
            mean_prefill_kv,
            attention_pairs,
            mean_decode_kv,
            sum_dkv_sq,
            sum_ptok.ln_1p(),
            sum_pkv.ln_1p(),
            sum_dkv.ln_1p(),
            attention_pairs.ln_1p(),
            sum_dkv_sq.ln_1p(),
        ]);
        match pairs {
            Some(mut list) if !list.is_empty() => {
                // Sort by past descending, then extend descending (stable slot semantics).
                list.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal))
                });
                let n = list.len() as f64;
                let sum_e: f64 = list.iter().map(|(e, _)| e).sum();
                let sum_p: f64 = list.iter().map(|(_, p)| p).sum();
                let sum_ep: f64 = list.iter().map(|(e, p)| e * p).sum();
                let sum_e2: f64 = list.iter().map(|(e, _)| e * e).sum();
                let sum_p2: f64 = list.iter().map(|(_, p)| p * p).sum();
                let sum_attn: f64 = list.iter().map(|(e, p)| e * (p + e / 2.0)).sum();
                let max_e = list.iter().map(|(e, _)| *e).fold(f64::MIN, f64::max);
                let min_e = list.iter().map(|(e, _)| *e).fold(f64::MAX, f64::min);
                let max_p = list.iter().map(|(_, p)| *p).fold(f64::MIN, f64::max);
                let min_p = list.iter().map(|(_, p)| *p).fold(f64::MAX, f64::min);
                let is_decode = if list.iter().all(|(e, _)| *e <= 1.0) {
                    1.0
                } else {
                    0.0
                };
                let is_prefill = if list.iter().any(|(e, _)| *e > 1.0) {
                    1.0
                } else {
                    0.0
                };
                values.extend_from_slice(&[
                    n,
                    sum_e,
                    max_e,
                    min_e,
                    sum_p,
                    max_p,
                    min_p,
                    sum_ep,
                    sum_e2,
                    sum_p2,
                    sum_attn,
                    sum_e * max_p,
                    sum_p.ln_1p(),
                    sum_attn.ln_1p(),
                    n * sum_e,
                    max_p - min_p,
                    is_decode,
                    is_prefill,
                ]);
                for slot in 0..SLOT_COUNT {
                    match list.get(slot) {
                        Some((e, p)) => values.extend_from_slice(&[1.0, *p, *e]),
                        None => values.extend_from_slice(&[0.0, 0.0, 0.0]),
                    }
                }
            }
            _ => {
                // Producer emitted no per-request lists: NaN for req_* features
                // (missing-value routing) and empty slots.
                values.extend(std::iter::repeat_n(f64::NAN, REQUEST_FEATURE_NAMES.len()));
                values.extend(std::iter::repeat_n(0.0, 3 * SLOT_COUNT));
            }
        }
        debug_assert_eq!(values.len(), FEATURE_COUNT);
        Self { values }
    }
}

/// Options-independent artifact loader entry used by the model constructors.
pub(crate) fn load_learned(
    source: &str,
    _options: &ForwardPassPerfOptions,
) -> Result<LearnedForwardPassModel, AicError> {
    let trimmed = source.trim_start();
    if trimmed.starts_with('{') {
        LearnedForwardPassModel::from_json(source)
    } else {
        LearnedForwardPassModel::from_path(Path::new(source))
    }
}
