// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Data-calibrated empirical estimation via SOL-utilization.
//!
//! Mirrors `aisimulate.sdk.operations.util_empirical`: each op's
//! empirical estimate is `latency = SOL(query) / util` where
//! `util = SOL / measured > 0` is read best-effort from collected samples in
//! per-axis normalised log space. `util` is an effective calibration factor,
//! not a bounded physical efficiency (it may exceed 1); it is never clamped.
//! Every grid uses the same two-neighbour inverse-distance weighting
//! (`k=2`, `p=1`) without requiring a Cartesian product; queries outside the
//! measured range are clamped per axis before neighbour selection, so
//! extrapolation freezes boundary utilization. Well-formed grids use an
//! immutable k-d tree and a bounded query cache; tiny or non-indexable grids
//! retain the linear lookup for compatibility.
//!
//! When *no* samples exist for the requested slice (no own-shape, no
//! cross-shape/sibling transfer reference), [`estimate`] returns
//! [`AicError::EmpiricalNotImplemented`] rather than a fabricated
//! `SOL / constant` — coverage gaps surface honestly, exactly like Python's
//! `EmpiricalNotImplementedError`. Genuinely table-less ops (mem / p2p /
//! element-wise) keep their analytic formulas and never call [`estimate`].
//!
//! Divergences from the Python module, by design:
//! - No provenance contextvar: provenance capture feeds the Python-side
//!   support matrix; the compiled engine only returns latencies. Reference
//!   grids still carry their provenance tag for cache keying.
//! - Python keys grids by `id(node)` because database views share mutable table
//!   objects. Rust perf tables are immutable after load, so per-op wiring
//!   caches grids in a [`UtilGridCache`] keyed by the op's slice identity.
//!   Each indexed grid also owns a bounded cache of exact normalized queries.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use quick_cache::sync::{Cache, DefaultLifecycle};
use quick_cache::{DefaultHashBuilder, Equivalent, OptionsBuilder, UnitWeighter};

use crate::common::error::AicError;
use crate::perfmodel::kd_tree::{KdTree, NeighborCollector};

/// Empirical provenance tiers, mirroring Python's `PROVENANCE_ORDER`
/// (`sdk/operations/util_empirical.py`): ordered by DECREASING confidence,
/// so the max rank fired during a run is the run's effective data source
/// (Python `worst_provenance`). `Silicon` is the default when nothing fired;
/// operators never note it.
///
/// The accumulation cell lives on `PerfDatabase` (shared across mode views);
/// operators call `PerfDatabase::note_provenance` at the same sites Python
/// calls `note_provenance` — after a successful `estimate` with the tier the
/// call site knows (Python passes it as `estimate`'s `provenance` param).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ProvenanceTier {
    /// Pure silicon table data (default; never recorded by operators).
    Silicon = 0,
    /// Own-shape util (no transfer).
    Empirical = 1,
    /// Cross-shape, same quant.
    XShape = 2,
    /// Cross-quant, same profile.
    XQuant = 3,
    /// Cross-quant, cross profile.
    XProfile = 4,
    /// Cross-op (borrowed a different op's util).
    XOp = 5,
}

impl ProvenanceTier {
    /// The Python tag string (`PROVENANCE_ORDER` spelling) for this tier.
    pub fn as_str(self) -> &'static str {
        match self {
            ProvenanceTier::Silicon => "silicon",
            ProvenanceTier::Empirical => "empirical",
            ProvenanceTier::XShape => "xshape",
            ProvenanceTier::XQuant => "xquant",
            ProvenanceTier::XProfile => "xprofile",
            ProvenanceTier::XOp => "xop",
        }
    }

    /// Inverse of [`Self::as_str`] for call sites that carry the Python tag
    /// string (e.g. a reference grid's `reference_provenance`). Unknown tags
    /// yield `None` so callers choose their own default tier.
    pub fn from_tag(tag: &str) -> Option<ProvenanceTier> {
        match tag {
            "silicon" => Some(ProvenanceTier::Silicon),
            "empirical" => Some(ProvenanceTier::Empirical),
            "xshape" => Some(ProvenanceTier::XShape),
            "xquant" => Some(ProvenanceTier::XQuant),
            "xprofile" => Some(ProvenanceTier::XProfile),
            "xop" => Some(ProvenanceTier::XOp),
            _ => None,
        }
    }

    /// Inverse of `tier as u8` for reading the accumulation cell back.
    /// Out-of-range ranks clamp to the least-confident tier.
    pub fn from_rank(rank: u8) -> ProvenanceTier {
        match rank {
            0 => ProvenanceTier::Silicon,
            1 => ProvenanceTier::Empirical,
            2 => ProvenanceTier::XShape,
            3 => ProvenanceTier::XQuant,
            4 => ProvenanceTier::XProfile,
            _ => ProvenanceTier::XOp,
        }
    }
}

/// One collected calibration point: continuous-axis coordinates plus the
/// positive effective calibration factor `util = SOL / measured`.
#[derive(Clone, Debug, PartialEq)]
pub struct UtilSample {
    pub coords: Vec<f64>,
    pub util: f64,
}

impl UtilSample {
    pub fn new(coords: Vec<f64>, util: f64) -> Self {
        Self { coords, util }
    }
}

/// Build util samples from `(coords, latency_ms)` points and an analytic SOL.
///
/// Mirrors Python `build_samples`: a point is kept only when both the
/// measured latency and its SOL are strictly positive (NaN fails both
/// comparisons and is dropped, matching Python truthiness + `> 0`).
pub fn build_samples<I, F>(points: I, sol_fn: F) -> Vec<UtilSample>
where
    I: IntoIterator<Item = (Vec<f64>, f64)>,
    F: Fn(&[f64]) -> f64,
{
    let mut samples = Vec::new();
    for (coords, latency_ms) in points {
        if latency_ms > 0.0 {
            let sol = sol_fn(&coords);
            if sol > 0.0 {
                samples.push(UtilSample::new(coords, sol / latency_ms));
            }
        }
    }
    samples
}

/// Two-neighbour util lookup in per-axis normalised log space.
///
/// The query is clamped independently on every axis, then the two nearest
/// samples are combined with inverse-distance weights (`k=2`, `p=1`). Exact
/// hits return the collected utilization unchanged. Works for ragged grids
/// without operation-specific Cartesian bracketing; callers remain
/// responsible for slicing categorical/kernel-regime axes. Grids with at
/// least three finite, equal-dimensional samples use an immutable k-d tree and
/// bounded query cache; other grids retain the linear lookup.
#[derive(Debug, Clone)]
pub struct UtilGrid {
    /// Normalised log-space coordinates, one row per sample.
    norm: Vec<Vec<f64>>,
    utils: Vec<f64>,
    mins: Vec<f64>,
    spans: Vec<f64>,
    index: Option<KdTree>,
    query_cache: Arc<OnceLock<UtilQueryCache>>,
    /// Transfer tag of the reference slice this grid was built from
    /// (`xshape` / `xquant` / ...), when borrowed from a sibling.
    pub reference_provenance: Option<&'static str>,
}

const UTIL_QUERY_CACHE_CAPACITY: usize = 32_768;
const UTIL_QUERY_CACHE_SHARDS: usize = 16;

type UtilQueryCache = Cache<UtilQueryKey, f64>;

#[derive(Debug, PartialEq, Eq)]
struct UtilQueryKey(Box<[u64]>);

impl UtilQueryKey {
    fn new(query: &[f64]) -> Self {
        Self(
            query
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
    }
}

impl Hash for UtilQueryKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_util_query(self.0.len(), self.0.iter().copied(), state);
    }
}

#[derive(Clone, Copy)]
struct UtilQueryRef<'a>(&'a [f64]);

impl Hash for UtilQueryRef<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_util_query(
            self.0.len(),
            self.0.iter().map(|value| value.to_bits()),
            state,
        );
    }
}

impl Equivalent<UtilQueryKey> for UtilQueryRef<'_> {
    fn equivalent(&self, key: &UtilQueryKey) -> bool {
        self.0.len() == key.0.len()
            && self
                .0
                .iter()
                .zip(key.0.iter())
                .all(|(value, bits)| value.to_bits() == *bits)
    }
}

fn hash_util_query<H: Hasher>(len: usize, bits: impl Iterator<Item = u64>, state: &mut H) {
    len.hash(state);
    for bits in bits {
        bits.hash(state);
    }
}

fn util_query_cache() -> UtilQueryCache {
    let options = OptionsBuilder::new()
        .estimated_items_capacity(UTIL_QUERY_CACHE_CAPACITY)
        .weight_capacity(UTIL_QUERY_CACHE_CAPACITY as u64)
        .shards(UTIL_QUERY_CACHE_SHARDS)
        .build()
        .expect("valid static util query cache options");
    Cache::with_options(
        options,
        UnitWeighter,
        DefaultHashBuilder::default(),
        DefaultLifecycle::default(),
    )
}

#[derive(Clone, Copy, Debug)]
struct Neighbor {
    sample: usize,
    distance: f64,
    distance_squared: f64,
}

impl Neighbor {
    fn cmp(self, other: Self) -> std::cmp::Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.sample.cmp(&other.sample))
    }
}

#[derive(Debug, Default)]
struct NearestTwo {
    nearest: Option<Neighbor>,
    second: Option<Neighbor>,
}

impl NearestTwo {
    fn consider(&mut self, candidate: Neighbor) {
        if self
            .nearest
            .is_none_or(|nearest| candidate.cmp(nearest).is_lt())
        {
            self.second = self.nearest;
            self.nearest = Some(candidate);
        } else if self
            .second
            .is_none_or(|second| candidate.cmp(second).is_lt())
        {
            self.second = Some(candidate);
        }
    }
}

impl NeighborCollector for NearestTwo {
    fn consider(&mut self, sample: usize, distance_squared: f64) {
        self.consider(Neighbor {
            sample,
            distance: distance_squared.sqrt(),
            distance_squared,
        });
    }

    fn cutoff_distance_squared(&self) -> Option<f64> {
        // Preserve UtilGrid's existing prediction behavior. Its ordering uses
        // rounded sqrt distances, while this bound uses squared distance; two
        // different squared distances can round to the same ordering key.
        self.second.map(|second| second.distance_squared)
    }
}

fn log_floor(value: f64) -> f64 {
    value.max(1e-9).ln()
}

impl UtilGrid {
    pub fn new(samples: Vec<UtilSample>) -> Self {
        if samples.is_empty() {
            return Self {
                norm: Vec::new(),
                utils: Vec::new(),
                mins: Vec::new(),
                spans: Vec::new(),
                index: None,
                query_cache: Arc::new(OnceLock::new()),
                reference_provenance: None,
            };
        }
        let dims = samples[0].coords.len();
        let logc: Vec<Vec<f64>> = samples
            .iter()
            .map(|s| s.coords.iter().map(|&c| log_floor(c)).collect())
            .collect();
        let mut mins = vec![f64::INFINITY; dims];
        let mut maxs = vec![f64::NEG_INFINITY; dims];
        for row in &logc {
            for (a, &v) in row.iter().enumerate() {
                mins[a] = mins[a].min(v);
                maxs[a] = maxs[a].max(v);
            }
        }
        let spans: Vec<f64> = mins
            .iter()
            .zip(&maxs)
            .map(|(&lo, &hi)| if hi - lo > 0.0 { hi - lo } else { 1.0 })
            .collect();
        let norm: Vec<Vec<f64>> = logc
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(a, &v)| (v - mins[a]) / spans[a])
                    .collect()
            })
            .collect();
        let utils = samples.iter().map(|s| s.util).collect();
        let index = KdTree::build(norm.as_slice());
        Self {
            norm,
            utils,
            mins,
            spans,
            index,
            query_cache: Arc::new(OnceLock::new()),
            reference_provenance: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.utils.is_empty()
    }

    /// Interpolated utilization at `query`, or `None` for an empty grid.
    pub fn util(&self, query: &[f64]) -> Option<f64> {
        if self.utils.is_empty() {
            return None;
        }
        // Per-axis clamp to [0, 1] freezes boundary utilization for
        // out-of-range queries (mirrors `np.clip`).
        let q: Vec<f64> = query
            .iter()
            .enumerate()
            .map(|(a, &v)| ((log_floor(v) - self.mins[a]) / self.spans[a]).clamp(0.0, 1.0))
            .collect();
        let (nearest, cache) =
            if let Some(index) = self.index.as_ref().filter(|index| index.can_query(&q)) {
                let cache = self.query_cache.get_or_init(util_query_cache);
                if let Some(value) = cache.get(&UtilQueryRef(&q)) {
                    return Some(value);
                }
                let mut nearest = NearestTwo::default();
                index.search(self.norm.as_slice(), &q, &mut nearest);
                (nearest, Some(cache))
            } else {
                // Preserve the former behavior for tiny, non-finite, ragged,
                // or dimension-mismatched grids that cannot use the index.
                let mut nearest = NearestTwo::default();
                for (sample, row) in self.norm.iter().enumerate() {
                    let distance_squared = row
                        .iter()
                        .zip(&q)
                        .map(|(&x, &y)| (x - y) * (x - y))
                        .sum::<f64>();
                    nearest.consider(Neighbor {
                        sample,
                        distance: distance_squared.sqrt(),
                        distance_squared,
                    });
                }
                (nearest, None)
            };

        let second = nearest.second;
        let nearest = nearest.nearest.expect("non-empty util grid");
        let util = if nearest.distance == 0.0 {
            self.utils[nearest.sample]
        } else {
            let mut weighted = 0.0;
            let mut weight_sum = 0.0;
            for neighbor in [Some(nearest), second].into_iter().flatten() {
                let w = 1.0 / neighbor.distance;
                weighted += self.utils[neighbor.sample] * w;
                weight_sum += w;
            }
            weighted / weight_sum
        };
        if util.is_finite() {
            if let Some(cache) = cache {
                cache.insert(UtilQueryKey::new(&q), util);
            }
        }
        Some(util)
    }
}

/// Return `(latency_ms, util)` from the util grid, or the typed coverage
/// error.
///
/// Mirrors Python `estimate`: `None`, empty grids, and non-positive utils all
/// surface as [`AicError::EmpiricalNotImplemented`] — there is no own-shape,
/// cross-shape, or sibling data to calibrate from, so the gap surfaces
/// instead of inventing a `SOL / constant` placeholder.
///
/// `util_scale` is the cross-op level-alignment hook (1.0 = no change). When
/// a CROSS-OP transfer borrows a *different* op's util grid, the caller
/// passes a manual scale `k` so `latency = SOL / (util * k)`.
pub fn estimate(
    sol_query: f64,
    query: &[f64],
    grid: Option<&UtilGrid>,
    util_scale: f64,
) -> Result<(f64, f64), AicError> {
    if let Some(util) = grid.and_then(|g| g.util(query)) {
        if util > 0.0 {
            return Ok((sol_query / (util * util_scale), util));
        }
    }
    Err(AicError::EmpiricalNotImplemented(format!(
        "No empirical utilisation data to estimate this op at query={query:?}: \
         no own-shape, cross-shape, or sibling transfer reference available."
    )))
}

/// Nearest reference index by categorical shape features in per-dim
/// normalised log space (mirrors Python `_nearest_candidate`; ties keep the
/// first candidate, matching `np.argmin`). Returns `None` for an empty list.
pub fn nearest_candidate_index(query_features: &[f64], candidates: &[Vec<f64>]) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }
    let dims = query_features.len();
    let feats: Vec<Vec<f64>> = candidates
        .iter()
        .map(|c| c.iter().map(|&v| log_floor(v)).collect())
        .collect();
    let mut mins = vec![f64::INFINITY; dims];
    let mut maxs = vec![f64::NEG_INFINITY; dims];
    for row in &feats {
        for (a, &v) in row.iter().enumerate() {
            mins[a] = mins[a].min(v);
            maxs[a] = maxs[a].max(v);
        }
    }
    let spans: Vec<f64> = mins
        .iter()
        .zip(&maxs)
        .map(|(&lo, &hi)| if hi - lo > 0.0 { hi - lo } else { 1.0 })
        .collect();
    // NOTE: the query is intentionally NOT clamped here (unlike UtilGrid) —
    // Python normalises the query into the candidates' span without clipping.
    let q: Vec<f64> = query_features
        .iter()
        .enumerate()
        .map(|(a, &v)| (log_floor(v) - mins[a]) / spans[a])
        .collect();
    let mut best = 0;
    let mut best_dist2 = f64::INFINITY;
    for (i, row) in feats.iter().enumerate() {
        let dist2: f64 = row
            .iter()
            .enumerate()
            .map(|(a, &v)| {
                let n = (v - mins[a]) / spans[a];
                (n - q[a]) * (n - q[a])
            })
            .sum();
        if dist2 < best_dist2 {
            best_dist2 = dist2;
            best = i;
        }
    }
    Some(best)
}

/// Process-lifetime memo of built util grids, keyed by the caller's slice
/// identity. Rust perf tables are immutable after load and each
/// `PerfDatabase` owns its cache, so a plain keyed map replaces Python's
/// `id(node)`-qualified module cache.
#[derive(Debug, Default)]
pub struct UtilGridCache {
    grids: Mutex<HashMap<String, Option<Arc<UtilGrid>>>>,
}

impl UtilGridCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fetch or build the grid for `key`.
    ///
    /// `builder` mirrors Python's `grid_for` contract: `Ok(None)` when the
    /// slice has no usable calibration data (a typed coverage miss — memoised,
    /// the caller then raises via [`estimate`] with `grid=None`), `Err` for
    /// programming/schema errors (propagated, NOT memoised, never converted
    /// into a fallback).
    pub fn get_or_try_build<F>(
        &self,
        key: &str,
        builder: F,
    ) -> Result<Option<Arc<UtilGrid>>, AicError>
    where
        F: FnOnce() -> Result<Option<UtilGrid>, AicError>,
    {
        let mut grids = self.grids.lock().expect("util grid cache poisoned");
        if let Some(cached) = grids.get(key) {
            return Ok(cached.clone());
        }
        let built = builder()?.map(Arc::new);
        grids.insert(key.to_string(), built.clone());
        Ok(built)
    }
}

#[derive(Clone, Copy, Debug)]
struct DeltaNeighbor {
    sample: usize,
    distance_squared: f64,
}

impl DeltaNeighbor {
    fn is_better_than(self, other: Self) -> bool {
        self.distance_squared
            .total_cmp(&other.distance_squared)
            .then_with(|| self.sample.cmp(&other.sample))
            .is_lt()
    }
}

#[derive(Debug)]
struct DeltaKdNode {
    sample: usize,
    point: [f64; 2],
    axis: usize,
    left: Option<usize>,
    right: Option<usize>,
}

/// Immutable exact nearest-one index for a finite two-dimensional delta grid.
#[derive(Debug)]
struct DeltaKdTree {
    nodes: Vec<DeltaKdNode>,
}

impl DeltaKdTree {
    fn build(norm: &[Vec<f64>]) -> Option<Self> {
        if norm.len() < 3
            || norm
                .iter()
                .any(|row| row.len() != 2 || row.iter().any(|value| !value.is_finite()))
        {
            return None;
        }

        fn build_nodes(
            norm: &[Vec<f64>],
            samples: &mut [usize],
            depth: usize,
            nodes: &mut Vec<DeltaKdNode>,
        ) -> Option<usize> {
            if samples.is_empty() {
                return None;
            }
            let axis = depth % 2;
            let middle = samples.len() / 2;
            samples.select_nth_unstable_by(middle, |&left, &right| {
                norm[left][axis]
                    .total_cmp(&norm[right][axis])
                    .then_with(|| left.cmp(&right))
            });
            let (left_samples, middle_and_right) = samples.split_at_mut(middle);
            let (sample, right_samples) = middle_and_right
                .split_first_mut()
                .expect("non-empty delta k-d tree partition");
            let sample = *sample;
            let node = nodes.len();
            nodes.push(DeltaKdNode {
                sample,
                point: [norm[sample][0], norm[sample][1]],
                axis,
                left: None,
                right: None,
            });
            let left = build_nodes(norm, left_samples, depth + 1, nodes);
            let right = build_nodes(norm, right_samples, depth + 1, nodes);
            nodes[node].left = left;
            nodes[node].right = right;
            Some(node)
        }

        let mut samples = (0..norm.len()).collect::<Vec<_>>();
        let mut nodes = Vec::with_capacity(norm.len());
        build_nodes(norm, &mut samples, 0, &mut nodes)?;
        Some(Self { nodes })
    }

    fn nearest(&self, query: &[f64; 2]) -> DeltaNeighbor {
        fn visit(
            tree: &DeltaKdTree,
            query: &[f64; 2],
            node_index: usize,
            nearest: &mut DeltaNeighbor,
        ) {
            let node = &tree.nodes[node_index];
            let delta_0 = node.point[0] - query[0];
            let delta_1 = node.point[1] - query[1];
            let candidate = DeltaNeighbor {
                sample: node.sample,
                distance_squared: delta_0 * delta_0 + delta_1 * delta_1,
            };
            if candidate.is_better_than(*nearest) {
                *nearest = candidate;
            }

            let delta = query[node.axis] - node.point[node.axis];
            let (near, far) = if delta.is_sign_negative() {
                (node.left, node.right)
            } else {
                (node.right, node.left)
            };
            if let Some(near) = near {
                visit(tree, query, near, nearest);
            }
            // Equality can hide an equally near sample with an earlier input
            // index. Squared comparison also preserves underflowed ties.
            if delta * delta <= nearest.distance_squared {
                if let Some(far) = far {
                    visit(tree, query, far, nearest);
                }
            }
        }

        let mut nearest = DeltaNeighbor {
            sample: self.nodes[0].sample,
            distance_squared: f64::INFINITY,
        };
        visit(self, query, 0, &mut nearest);
        nearest
    }
}

/// Nearest-point lookup over a non-negative latency *delta* table (the
/// `compute_scale` mechanism; Python `gemm._ZeroAwareDeltaLookup`).
///
/// `compute_scale` stores `max(dynamic_quant - static_quant, 0)`: zero is a
/// measured, meaningful delta, not a missing latency. A normal util grid
/// cannot represent it (`SOL / 0`), and dropping zeroes can make one positive
/// noise sample the reference for the whole table. Select the nearest point
/// on the complete 2-D grid first: a selected zero stays zero; a positive
/// point uses frozen utilization so extrapolation scales with the query's
/// amount of work.
#[derive(Debug)]
pub struct ZeroAwareDeltaLookup {
    coords: Vec<Vec<f64>>,
    latencies: Vec<f64>,
    mins: Vec<f64>,
    spans: Vec<f64>,
    norm: Vec<Vec<f64>>,
    index: Option<DeltaKdTree>,
}

impl ZeroAwareDeltaLookup {
    /// Keep every point with `latency >= 0` (zero INCLUDED, unlike
    /// [`build_samples`]).
    pub fn new(points: Vec<(Vec<f64>, f64)>) -> Self {
        let kept: Vec<(Vec<f64>, f64)> =
            points.into_iter().filter(|(_, lat)| *lat >= 0.0).collect();
        if kept.is_empty() {
            return Self {
                coords: Vec::new(),
                latencies: Vec::new(),
                mins: Vec::new(),
                spans: Vec::new(),
                norm: Vec::new(),
                index: None,
            };
        }
        let dims = kept[0].0.len();
        let logc: Vec<Vec<f64>> = kept
            .iter()
            .map(|(c, _)| c.iter().map(|&v| log_floor(v)).collect())
            .collect();
        let mut mins = vec![f64::INFINITY; dims];
        let mut maxs = vec![f64::NEG_INFINITY; dims];
        for row in &logc {
            for (a, &v) in row.iter().enumerate() {
                mins[a] = mins[a].min(v);
                maxs[a] = maxs[a].max(v);
            }
        }
        let spans: Vec<f64> = mins
            .iter()
            .zip(&maxs)
            .map(|(&lo, &hi)| if hi - lo > 0.0 { hi - lo } else { 1.0 })
            .collect();
        let norm: Vec<Vec<f64>> = logc
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(a, &v)| (v - mins[a]) / spans[a])
                    .collect()
            })
            .collect();
        let index = DeltaKdTree::build(&norm);
        Self {
            latencies: kept.iter().map(|(_, lat)| *lat).collect(),
            coords: kept.into_iter().map(|(c, _)| c).collect(),
            mins,
            spans,
            norm,
            index,
        }
    }

    fn nearest_linear(&self, query: &[f64]) -> usize {
        let mut best = 0;
        let mut best_dist2 = f64::INFINITY;
        for (sample, row) in self.norm.iter().enumerate() {
            let distance_squared = row
                .iter()
                .zip(query)
                .map(|(&value, &query_value)| {
                    let delta = value - query_value;
                    delta * delta
                })
                .sum();
            if distance_squared < best_dist2 {
                best_dist2 = distance_squared;
                best = sample;
            }
        }
        best
    }

    /// Nearest-point delta estimate (query is NOT clamped; the frozen-util
    /// rescale `query_sol / (ref_sol / ref_latency)` carries extrapolation).
    pub fn estimate<F>(&self, query: &[f64], sol_fn: F) -> Result<f64, AicError>
    where
        F: Fn(&[f64]) -> f64,
    {
        if self.latencies.is_empty() {
            return Err(AicError::EmpiricalNotImplemented(format!(
                "No empirical compute_scale delta data is available at query={query:?}."
            )));
        }
        let best = if query.len() == 2 && self.mins.len() == 2 {
            let normalized = [
                (log_floor(query[0]) - self.mins[0]) / self.spans[0],
                (log_floor(query[1]) - self.mins[1]) / self.spans[1],
            ];
            if normalized.iter().all(|value| value.is_finite()) {
                self.index.as_ref().map_or_else(
                    || self.nearest_linear(&normalized),
                    |index| index.nearest(&normalized).sample,
                )
            } else {
                self.nearest_linear(&normalized)
            }
        } else {
            let normalized = query
                .iter()
                .enumerate()
                .map(|(axis, &value)| (log_floor(value) - self.mins[axis]) / self.spans[axis])
                .collect::<Vec<_>>();
            self.nearest_linear(&normalized)
        };
        let reference_latency = self.latencies[best];
        if reference_latency == 0.0 {
            return Ok(0.0);
        }
        let reference_sol = sol_fn(&self.coords[best]);
        let query_sol = sol_fn(query);
        if reference_sol <= 0.0 || query_sol <= 0.0 {
            return Err(AicError::EmpiricalNotImplemented(format!(
                "No positive SOL reference is available for compute_scale at query={query:?}."
            )));
        }
        Ok(query_sol / (reference_sol / reference_latency))
    }
}

/// Memo of built [`ZeroAwareDeltaLookup`]s (same keying/lifetime rationale as
/// [`UtilGridCache`]).
#[derive(Debug, Default)]
pub struct DeltaLookupCache {
    lookups: Mutex<HashMap<String, Arc<ZeroAwareDeltaLookup>>>,
}

impl DeltaLookupCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_try_build<F>(
        &self,
        key: &str,
        builder: F,
    ) -> Result<Arc<ZeroAwareDeltaLookup>, AicError>
    where
        F: FnOnce() -> Result<ZeroAwareDeltaLookup, AicError>,
    {
        let mut lookups = self.lookups.lock().expect("delta lookup cache poisoned");
        if let Some(cached) = lookups.get(key) {
            return Ok(cached.clone());
        }
        let built = Arc::new(builder()?);
        lookups.insert(key.to_string(), built.clone());
        Ok(built)
    }
}

#[cfg(test)]
mod tests {
    //! Mirrors the math anchors of `tests/unit/sdk/test_util_empirical.py`.
    //! The Python cache/`grid_for` contract tests are id()-specific and are
    //! covered by `UtilGridCache`'s own semantics instead.

    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "expected {b}, got {a}");
    }

    #[test]
    fn exact_singleton_duplicate_and_empty_grid_contracts() {
        let exact = UtilGrid::new(vec![
            UtilSample::new(vec![16.0], 0.8),
            UtilSample::new(vec![8.0], 0.2),
            UtilSample::new(vec![9.0], 0.4),
        ]);
        let linear_duplicate = UtilGrid::new(vec![
            UtilSample::new(vec![4.0], 0.6),
            UtilSample::new(vec![4.0], 0.7),
        ]);
        let indexed_duplicate = UtilGrid::new(vec![
            UtilSample::new(vec![4.0], 0.6),
            UtilSample::new(vec![4.0], 0.7),
            UtilSample::new(vec![16.0], 0.9),
        ]);

        approx(exact.util(&[9.0]).unwrap(), 0.4);
        approx(
            UtilGrid::new(vec![UtilSample::new(vec![0.0], 0.3)])
                .util(&[100.0])
                .unwrap(),
            0.3,
        );
        // Duplicate coordinates: first sample wins (stable ordering).
        approx(linear_duplicate.util(&[4.0]).unwrap(), 0.6);
        approx(indexed_duplicate.util(&[4.0]).unwrap(), 0.6);
        assert!(UtilGrid::new(vec![]).util(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn one_dim_k2_idw_uses_nearest_samples_in_normalized_log_space() {
        let grid = UtilGrid::new(vec![
            UtilSample::new(vec![16.0], 0.8),
            UtilSample::new(vec![8.0], 0.2),
            UtilSample::new(vec![9.0], 0.4),
        ]);
        let distance_9 = 11.0_f64.ln() - 9.0_f64.ln();
        let distance_8 = 11.0_f64.ln() - 8.0_f64.ln();
        let expected =
            (0.4 / distance_9 + 0.2 / distance_8) / (1.0 / distance_9 + 1.0 / distance_8);

        approx(grid.util(&[11.0]).unwrap(), expected);
    }

    #[test]
    fn multidimensional_k2_idw_uses_nearest_samples() {
        let grid = UtilGrid::new(vec![
            UtilSample::new(vec![1.0, 1.0], 0.2),
            UtilSample::new(vec![100.0, 1.0], 0.6),
            UtilSample::new(vec![100.0, 100.0], 1.0),
        ]);

        // (10, 1) is equidistant from the first two normalized-log samples.
        approx(grid.util(&[10.0, 1.0]).unwrap(), 0.4);
    }

    #[test]
    fn indexed_queries_cache_exact_normalized_coordinates() {
        let grid = UtilGrid::new(vec![
            UtilSample::new(vec![1.0], 0.2),
            UtilSample::new(vec![2.0], 0.4),
            UtilSample::new(vec![4.0], 0.8),
        ]);

        let first = grid.util(&[2.0]).unwrap();
        let second = grid.util(&[2.0]).unwrap();
        assert_eq!(first.to_bits(), second.to_bits());
        assert_eq!(grid.query_cache.get().unwrap().len(), 1);

        let low = grid.util(&[0.5]).unwrap();
        let lower = grid.util(&[0.25]).unwrap();
        assert_eq!(low.to_bits(), lower.to_bits());
        assert_eq!(grid.query_cache.get().unwrap().len(), 2);
    }

    #[test]
    fn indexed_query_cache_supports_arbitrary_dimensions() {
        let grid = UtilGrid::new(vec![
            UtilSample::new(vec![1.0; 5], 0.2),
            UtilSample::new(vec![4.0; 5], 0.4),
            UtilSample::new(vec![16.0; 5], 0.8),
        ]);
        let query = vec![2.0; 5];

        let first = grid.util(&query).unwrap();
        let second = grid.util(&query).unwrap();
        assert_eq!(first.to_bits(), second.to_bits());
        assert_eq!(grid.query_cache.get().unwrap().len(), 1);

        let mut distinct = query;
        distinct[4] = 3.0;
        grid.util(&distinct).unwrap();
        assert_eq!(grid.query_cache.get().unwrap().len(), 2);
    }

    #[test]
    fn indexed_query_cache_does_not_store_non_finite_results() {
        let grid = UtilGrid::new(vec![
            UtilSample::new(vec![1.0], f64::NAN),
            UtilSample::new(vec![2.0], 0.4),
            UtilSample::new(vec![4.0], 0.8),
        ]);

        assert!(grid.util(&[1.0]).unwrap().is_nan());
        assert_eq!(grid.query_cache.get().unwrap().len(), 0);
    }

    #[test]
    fn second_neighbor_tie_keeps_sample_order() {
        let grid = UtilGrid::new(vec![
            UtilSample::new(vec![2.0, 2.0], 0.2),
            UtilSample::new(vec![1.0, 3.0], 0.2),
            UtilSample::new(vec![3.0, 1.0], 0.9),
        ]);

        // The last two samples tie for second place. Stable order selects
        // the first one, whose utilization matches the nearest sample.
        approx(grid.util(&[2.5, 2.5]).unwrap(), 0.2);
    }

    #[test]
    fn second_neighbor_can_cross_kd_tree_split() {
        let grid = UtilGrid::new(vec![
            UtilSample::new(vec![1.0, 1.0], 0.1),
            UtilSample::new(vec![2.0, 1.0], 0.2),
            UtilSample::new(vec![4.0, 100.0], 0.4),
            UtilSample::new(vec![8.0, 100.0], 0.6),
            UtilSample::new(vec![16.0, 1.0], 0.8),
        ]);
        let query_x = 6.0_f64.ln() / 16.0_f64.ln();
        let distance_16 = 1.0 - query_x;
        let distance_2 = query_x - 0.25;
        let expected =
            (0.8 / distance_16 + 0.2 / distance_2) / (1.0 / distance_16 + 1.0 / distance_2);

        approx(grid.util(&[6.0, 1.0]).unwrap(), expected);
    }

    #[test]
    fn far_branch_pruning_preserves_underflowed_distance_ties() {
        let tiny = f64::MIN_POSITIVE;
        let norm = vec![vec![0.0], vec![tiny], vec![3.0 * tiny]];
        let tree = KdTree::build(norm.as_slice()).unwrap();

        // All three squared distances underflow to zero. The far branch still
        // contains the first sample, which stable ordering must select.
        let mut nearest = NearestTwo::default();
        tree.search(norm.as_slice(), &[2.0 * tiny], &mut nearest);
        assert_eq!(nearest.nearest.unwrap().sample, 0);
        assert_eq!(nearest.second.unwrap().sample, 1);
    }

    #[test]
    fn one_dim_extrapolation_clamps_to_measured_bounds() {
        let grid = UtilGrid::new(vec![
            UtilSample::new(vec![8.0], 0.2),
            UtilSample::new(vec![16.0], 0.8),
        ]);

        approx(grid.util(&[1.0]).unwrap(), 0.2);
        approx(grid.util(&[128.0]).unwrap(), 0.8);
        assert!(grid.query_cache.get().is_none());
    }

    #[test]
    fn multidimensional_extrapolation_clamps_each_axis() {
        let grid = UtilGrid::new(vec![
            UtilSample::new(vec![1.0, 1.0], 0.2),
            UtilSample::new(vec![1.0, 10.0], 0.4),
            UtilSample::new(vec![10.0, 1.0], 0.8),
        ]);

        // Clamping (0.1, 100) produces the exact measured boundary (1, 10).
        approx(grid.util(&[0.1, 100.0]).unwrap(), 0.4);
    }

    #[test]
    fn build_samples_filters_non_positive_latency_and_sol() {
        let samples = build_samples(
            vec![
                (vec![2.0], 4.0),  // kept: util = sol/lat = 2/4
                (vec![3.0], 0.0),  // dropped: latency <= 0
                (vec![4.0], -1.0), // dropped: latency < 0
                (vec![0.5], 1.0),  // dropped: sol_fn returns 0 below 1.0
            ],
            |coords| if coords[0] >= 1.0 { coords[0] } else { 0.0 },
        );
        assert_eq!(samples.len(), 1);
        approx(samples[0].util, 0.5);
    }

    #[test]
    fn estimate_returns_latency_and_util_or_typed_miss() {
        let grid = UtilGrid::new(vec![UtilSample::new(vec![1.0], 0.5)]);
        let (latency, util) = estimate(1.0, &[1.0], Some(&grid), 1.0).unwrap();
        approx(latency, 2.0);
        approx(util, 0.5);

        // Cross-op level alignment: latency = SOL / (util * k).
        let (latency, _) = estimate(1.0, &[1.0], Some(&grid), 2.0).unwrap();
        approx(latency, 1.0);

        let missing = estimate(1.0, &[1.0], None, 1.0);
        assert!(matches!(missing, Err(AicError::EmpiricalNotImplemented(_))));
        let empty = UtilGrid::new(vec![]);
        let empty_res = estimate(1.0, &[1.0], Some(&empty), 1.0);
        assert!(matches!(
            empty_res,
            Err(AicError::EmpiricalNotImplemented(_))
        ));
    }

    #[test]
    fn nearest_candidate_matches_python_normalised_log_selection() {
        // Query (90,) between features (1,) and (100,): log-nearer to 100.
        let idx = nearest_candidate_index(&[90.0], &[vec![1.0], vec![100.0]]);
        assert_eq!(idx, Some(1));
        // Single candidate always selected; empty list yields None.
        assert_eq!(nearest_candidate_index(&[5.0], &[vec![1.0]]), Some(0));
        assert_eq!(nearest_candidate_index(&[5.0], &[]), None);
    }

    #[test]
    fn util_grid_cache_memoises_hits_and_misses_but_not_errors() {
        let cache = UtilGridCache::new();
        let mut builds = 0;
        let grid = cache.get_or_try_build("k", || {
            builds += 1;
            Ok(Some(UtilGrid::new(vec![UtilSample::new(vec![1.0], 0.5)])))
        });
        assert!(grid.unwrap().is_some());
        let again = cache.get_or_try_build("k", || {
            builds += 1;
            Ok(None)
        });
        assert!(again.unwrap().is_some());
        assert_eq!(builds, 1);

        // Typed coverage miss (Ok(None)) is memoised, like Python's grid_for.
        let miss = cache.get_or_try_build("missing", || Ok(None));
        assert!(miss.unwrap().is_none());
        let miss_again =
            cache.get_or_try_build("missing", || panic!("memoised miss must not rebuild"));
        assert!(miss_again.unwrap().is_none());

        // Programming/schema errors propagate and are NOT memoised.
        let err = cache.get_or_try_build("broken", || {
            Err(AicError::PerfDatabase("schema".to_string()))
        });
        assert!(err.is_err());
        let recovered = cache.get_or_try_build("broken", || Ok(None));
        assert!(recovered.unwrap().is_none());
    }

    #[test]
    fn provenance_tier_mirrors_python_provenance_order() {
        // Rank order == PROVENANCE_ORDER index; tag strings match Python.
        let tiers = [
            (ProvenanceTier::Silicon, 0u8, "silicon"),
            (ProvenanceTier::Empirical, 1, "empirical"),
            (ProvenanceTier::XShape, 2, "xshape"),
            (ProvenanceTier::XQuant, 3, "xquant"),
            (ProvenanceTier::XProfile, 4, "xprofile"),
            (ProvenanceTier::XOp, 5, "xop"),
        ];
        for (tier, rank, tag) in tiers {
            assert_eq!(tier as u8, rank);
            assert_eq!(tier.as_str(), tag);
            assert_eq!(ProvenanceTier::from_rank(rank), tier);
            assert_eq!(ProvenanceTier::from_tag(tag), Some(tier));
        }
        assert_eq!(ProvenanceTier::from_tag("unknown"), None);
        // max-rank == worst_provenance semantics; overflow clamps to XOp.
        assert!(ProvenanceTier::XOp > ProvenanceTier::Empirical);
        assert_eq!(ProvenanceTier::from_rank(200), ProvenanceTier::XOp);
    }

    #[test]
    fn zero_aware_delta_keeps_zeroes_and_freezes_utilization() {
        let lookup =
            ZeroAwareDeltaLookup::new(vec![(vec![16.0, 16.0], 0.0), (vec![1024.0, 1024.0], 2.0)]);
        let sol = |c: &[f64]| c[0] * c[1];

        // Nearest to the zero point: the measured zero delta stays zero.
        assert_eq!(lookup.estimate(&[8.0, 8.0], sol).unwrap(), 0.0);
        // Nearest to the positive point: frozen util scales with query SOL.
        let expected = (2048.0 * 2048.0) / ((1024.0 * 1024.0) / 2.0);
        assert!((lookup.estimate(&[2048.0, 2048.0], sol).unwrap() - expected).abs() < 1e-12);
        // Empty table -> typed empirical miss.
        let empty = ZeroAwareDeltaLookup::new(vec![]);
        assert!(matches!(
            empty.estimate(&[1.0, 1.0], sol),
            Err(AicError::EmpiricalNotImplemented(_))
        ));
    }

    #[test]
    fn zero_aware_delta_index_matches_linear_grid_lookup() {
        let mut points = Vec::new();
        for m in 0..37 {
            for k in 0..44 {
                points.push((
                    vec![1.25_f64.powi(m), 1.2_f64.powi(k)],
                    (m * 44 + k + 1) as f64,
                ));
            }
        }
        let lookup = ZeroAwareDeltaLookup::new(points);
        let index = lookup.index.as_ref().expect("37 by 44 grid is indexed");

        for m in 0..37 {
            for k in 0..44 {
                let query = [1.25_f64.powi(m), 1.2_f64.powi(k)];
                let normalized = [
                    (log_floor(query[0]) - lookup.mins[0]) / lookup.spans[0],
                    (log_floor(query[1]) - lookup.mins[1]) / lookup.spans[1],
                ];
                let expected = lookup.nearest_linear(&normalized);
                let expected_estimate = 1.0 / (1.0 / lookup.latencies[expected]);
                assert_eq!(index.nearest(&normalized).sample, expected, "m={m}, k={k}");
                assert_eq!(
                    lookup.estimate(&query, |_| 1.0).unwrap().to_bits(),
                    expected_estimate.to_bits(),
                    "estimate m={m}, k={k}"
                );
            }
        }
        for m in 0..36 {
            for k in 0..43 {
                let query = [
                    (1.25_f64.powi(m) * 1.25_f64.powi(m + 1)).sqrt(),
                    (1.2_f64.powi(k) * 1.2_f64.powi(k + 1)).sqrt(),
                ];
                let normalized = [
                    (log_floor(query[0]) - lookup.mins[0]) / lookup.spans[0],
                    (log_floor(query[1]) - lookup.mins[1]) / lookup.spans[1],
                ];
                let expected = lookup.nearest_linear(&normalized);
                let expected_estimate = 1.0 / (1.0 / lookup.latencies[expected]);
                assert_eq!(
                    index.nearest(&normalized).sample,
                    expected,
                    "midpoint m={m}, k={k}"
                );
                assert_eq!(
                    lookup.estimate(&query, |_| 1.0).unwrap().to_bits(),
                    expected_estimate.to_bits(),
                    "midpoint estimate m={m}, k={k}"
                );
            }
        }
    }

    #[test]
    fn zero_aware_delta_index_preserves_ties_and_fallbacks() {
        let tied = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 1.0]];
        let tied_tree = DeltaKdTree::build(&tied).unwrap();
        assert_eq!(tied_tree.nearest(&[0.5, 0.0]).sample, 0);
        assert!(tied_tree.nearest(&[f64::NAN, 0.0]).sample < tied.len());

        let tiny = f64::MIN_POSITIVE;
        let underflowed = vec![vec![0.0, 0.0], vec![tiny, 0.0], vec![3.0 * tiny, 0.0]];
        let underflowed_tree = DeltaKdTree::build(&underflowed).unwrap();
        assert_eq!(underflowed_tree.nearest(&[2.0 * tiny, 0.0]).sample, 0);

        assert!(DeltaKdTree::build(&[vec![0.0, 0.0], vec![1.0, 1.0]]).is_none());
        assert!(DeltaKdTree::build(&[vec![0.0, 0.0], vec![1.0], vec![2.0, 2.0]]).is_none());
        assert!(
            DeltaKdTree::build(&[vec![0.0, 0.0], vec![f64::NAN, 1.0], vec![2.0, 2.0]]).is_none()
        );

        let lookup = ZeroAwareDeltaLookup::new(vec![
            (vec![1.0, 1.0], 1.0),
            (vec![2.0, 2.0], 2.0),
            (vec![4.0, 4.0], 3.0),
        ]);
        let normalized = vec![(log_floor(2.0) - lookup.mins[0]) / lookup.spans[0]];
        let expected = lookup.latencies[lookup.nearest_linear(&normalized)];
        assert_eq!(lookup.estimate(&[2.0], |_| 1.0).unwrap(), expected);

        let ragged_lookup = ZeroAwareDeltaLookup::new(vec![
            (vec![1.0, 1.0], 1.0),
            (vec![2.0], 2.0),
            (vec![4.0, 4.0], 3.0),
        ]);
        assert!(ragged_lookup.index.is_none());
    }
}
