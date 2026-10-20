// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rand_chacha::ChaCha8Rng;
use rand08::{SeedableRng, seq::SliceRandom};

use super::super::learned::{LearnedStore, validate_tree};
use super::*;

fn structural(config: &FpmLearnedConfig) -> bool {
    config.update_mode == LearnedUpdateMode::Structural
}
fn lazy(config: &FpmLearnedConfig) -> bool {
    config.derivative_refresh == LearnedDerivativeRefresh::Lazy
}

fn regions(tree: &LearnedTree) -> Result<Vec<Vec<usize>>, AicError> {
    fn visit(
        tree: &LearnedTree,
        node: usize,
        out: &mut [Vec<usize>],
        seen: &mut [bool],
    ) -> Result<Vec<usize>, AicError> {
        if seen[node] {
            return Err(invalid("seed tree has shared children"));
        }
        seen[node] = true;
        let mut slots = vec![node];
        if tree.left[node] >= 0 {
            slots.extend(visit(tree, tree.left[node] as usize, out, seen)?);
            slots.extend(visit(tree, tree.right[node] as usize, out, seen)?);
        }
        slots.sort_unstable();
        out[node] = slots.clone();
        Ok(slots)
    }
    let mut out = vec![Vec::new(); tree.value.len()];
    let mut seen = vec![false; tree.value.len()];
    visit(tree, 0, &mut out, &mut seen)?;
    if seen.iter().any(|&s| !s) {
        return Err(invalid("seed tree has unreachable nodes"));
    }
    Ok(out)
}

fn sampled_candidates(
    thresholds: &[Vec<f64>],
    tree: usize,
    node: usize,
    generation: u64,
    config: &FpmLearnedConfig,
    incumbent: Option<Candidate>,
) -> Result<Vec<Candidate>, AicError> {
    let seed = config.seed
        ^ (tree as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (node as u64).rotate_left(23)
        ^ generation.rotate_left(41);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut candidates = Vec::new();
    for (feature, bounds) in thresholds.iter().enumerate() {
        let mut indices: Vec<_> = (0..bounds.len()).collect();
        if config.split_sampling_rate < 1.0 {
            indices.shuffle(&mut rng);
            indices.truncate((config.split_sampling_rate * bounds.len() as f64).ceil() as usize);
            indices.sort_unstable();
        }
        for bin in indices {
            for missing_left in [false, true] {
                candidates.push(Candidate::new(feature, bin, missing_left)?);
            }
        }
        // This boundary separates all finite values from NaN. Preserve it
        // independently of alpha, including constant-valued features with
        // no numeric thresholds. It is filtered by min_samples_leaf when
        // this node has no missing values.
        let missing_only = Candidate::new(
            feature,
            bounds
                .iter()
                .position(|&b| b == f64::MAX)
                .unwrap_or(bounds.len()),
            false,
        )?;
        if !candidates.contains(&missing_only) {
            candidates.push(missing_only);
        }
    }
    if let Some(split) = incumbent {
        if !candidates.contains(&split) {
            candidates.push(split);
        }
    }
    candidates.sort_by_key(|c| (c.feature, c.bin, c.missing_left));
    Ok(candidates)
}

fn incumbent(
    tree: &LearnedTree,
    node: usize,
    thresholds: &[Vec<f64>],
) -> Result<Option<Candidate>, AicError> {
    if tree.left[node] < 0 {
        return Ok(None);
    }
    let feature = tree.feature[node] as usize;
    let threshold = tree.threshold[node];
    let bin = thresholds[feature]
        .iter()
        .position(|&b| b == threshold)
        .or_else(|| (threshold == f64::MAX).then_some(thresholds[feature].len()))
        .ok_or_else(|| invalid("tree threshold is absent from the fitted bin mapper"))?;
    Ok(Some(Candidate::new(
        feature,
        bin,
        tree.missing_left.get(node).copied().unwrap_or(true),
    )?))
}

fn empty_stats(regions: &[Vec<usize>], histogram_size: usize) -> Vec<NodeStats> {
    // A permanent leaf cannot expand under the capacity policy. Its total is
    // sufficient for refitting; collecting candidate histograms there wastes
    // memory and one feature-wide update for every row in every tree.
    regions
        .iter()
        .map(|region| NodeStats {
            total: Sum::default(),
            histogram: vec![Sum::default(); if region.len() > 1 { histogram_size } else { 0 }],
        })
        .collect()
}

fn validate_statistics(stats: &[NodeStats]) -> Result<(), AicError> {
    if stats.iter().any(|node| {
        !node.total.g.is_finite()
            || !node.total.recent_g.is_finite()
            || node.total.recent_count > node.total.count
            || node.histogram.iter().any(|bin| {
                !bin.g.is_finite() || !bin.recent_g.is_finite() || bin.recent_count > bin.count
            })
    }) {
        return Err(invalid("nonfinite fitted aggregate"));
    }
    Ok(())
}

#[inline]
fn add(
    stats: &mut NodeStats,
    gradient: f64,
    count: u64,
    recent: bool,
    bins: &[u16],
    offsets: &[usize],
) {
    stats.total.g += gradient;
    stats.total.count += count;
    if recent {
        stats.total.recent_g += gradient;
        stats.total.recent_count += count;
    }
    if !stats.histogram.is_empty() {
        for (feature, &bin) in bins.iter().enumerate() {
            let slot = &mut stats.histogram[offsets[feature] + bin as usize];
            slot.g += gradient;
            slot.count += count;
            if recent {
                slot.recent_g += gradient;
                slot.recent_count += count;
            }
        }
    }
}

fn accumulate(
    tree: &LearnedTree,
    rows: &Rows<'_>,
    gradients: &[f64],
    offsets: &[usize],
    hist: bool,
    regions: &[Vec<usize>],
) -> Vec<NodeStats> {
    let mut stats = empty_stats(regions, if hist { *offsets.last().unwrap() } else { 0 });
    for (i, &g) in gradients.iter().enumerate() {
        let x = rows.x(i);
        let mut node = 0;
        loop {
            add(
                &mut stats[node],
                g,
                rows.count(i),
                i >= rows.anchors.y.len(),
                rows.bins(i),
                offsets,
            );
            if tree.left[node] < 0 {
                break;
            }
            let v = x[tree.feature[node] as usize];
            let left = if v.is_nan() {
                tree.missing_left.get(node).copied().unwrap_or(true)
            } else {
                v <= tree.threshold[node]
            };
            node = if left {
                tree.left[node] as usize
            } else {
                tree.right[node] as usize
            };
        }
    }
    stats
}

pub(super) fn bootstrap(
    state: &mut StoreState,
    store: &LearnedStore,
    config: &FpmLearnedConfig,
) -> Result<(), AicError> {
    let rows = Rows {
        anchors: &state.anchors,
        recent: Vec::new(),
    };
    let mut prefix = vec![store.baseline; rows.len()];
    let mut trees = Vec::with_capacity(store.trees.len());
    for (index, tree) in store.trees.iter().enumerate() {
        let gradients: Vec<_> = prefix
            .iter()
            .enumerate()
            .map(|(i, &p)| rows.gradient(i, p))
            .collect();
        if gradients.iter().any(|g| !g.is_finite()) {
            return Err(invalid("nonfinite bootstrap derivative"));
        }
        let regions = regions(tree)?;
        let candidates = (0..tree.value.len())
            .map(|node| {
                let split = incumbent(tree, node, &state.thresholds)?;
                Ok(if structural(config) && regions[node].len() > 1 {
                    sampled_candidates(&state.thresholds, index, node, 0, config, split)?
                } else {
                    Vec::new()
                })
            })
            .collect::<Result<_, AicError>>()?;
        trees.push(TreeState {
            stats: accumulate(
                tree,
                &rows,
                &gradients,
                &state.offsets,
                structural(config),
                &regions,
            ),
            regions,
            candidates,
            generation: 0,
        });
        for (i, p) in prefix.iter_mut().enumerate() {
            *p += tree.value[leaf_index(tree, rows.x(i))];
        }
    }
    state.trees = trees;
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Scored {
    candidate: Candidate,
    gain: f64,
}

fn tolerance(a: f64, b: f64) -> f64 {
    1e-12 * a.abs().max(b.abs()).max(1.0)
}
fn strictly_better(a: f64, b: f64) -> bool {
    a - b > tolerance(a, b)
}

fn score_candidates(
    stats: &NodeStats,
    candidates: &[Candidate],
    offsets: &[usize],
    settings: &FitSettings,
    weights: GroupWeights,
) -> Result<Vec<Scored>, AicError> {
    let total = stats.total;
    if total.count == 0 {
        return Ok(Vec::new());
    }
    let mut prefix = stats.histogram.clone();
    for pair in offsets.windows(2) {
        // Exclude the last (missing) bin from numerical prefix sums.
        for bin in pair[0] + 1..pair[1] - 1 {
            prefix[bin].g += prefix[bin - 1].g;
            prefix[bin].count += prefix[bin - 1].count;
            prefix[bin].recent_g += prefix[bin - 1].recent_g;
            prefix[bin].recent_count += prefix[bin - 1].recent_count;
        }
    }
    let ridge = settings.l2_regularization;
    let total_g = weights.gradient(total);
    let total_h = weights.hessian(total);
    if total_h <= 0.0 {
        return Ok(Vec::new());
    }
    let parent = total_g * total_g / (total_h + ridge);
    let mut result = Vec::with_capacity(candidates.len());
    for &candidate in candidates {
        let feature = usize::from(candidate.feature);
        let mut left = prefix[offsets[feature] + usize::from(candidate.bin)];
        if candidate.missing_left {
            let missing = stats.histogram[offsets[feature + 1] - 1];
            left.g += missing.g;
            left.count += missing.count;
            left.recent_g += missing.recent_g;
            left.recent_count += missing.recent_count;
        }
        let right_count = total
            .count
            .checked_sub(left.count)
            .ok_or_else(|| invalid("histogram count exceeds node count"))?;
        if left.count < settings.min_samples_leaf as u64
            || right_count < settings.min_samples_leaf as u64
        {
            continue;
        }
        let left_g = weights.gradient(left);
        let left_h = weights.hessian(left);
        let right_g = total_g - left_g;
        let right_h = total_h - left_h;
        if left_h <= 0.0 || right_h <= 0.0 {
            continue;
        }
        let gain = 0.5
            * (left_g * left_g / (left_h + ridge) + right_g * right_g / (right_h + ridge) - parent);
        if !gain.is_finite() {
            return Err(invalid("nonfinite split gain"));
        }
        result.push(Scored { candidate, gain });
    }
    Ok(result)
}

fn best(scores: &[Scored], current: Option<Candidate>) -> Option<Scored> {
    let mut chosen: Option<Scored> = None;
    for &score in scores {
        match chosen {
            None => chosen = Some(score),
            Some(old) if strictly_better(score.gain, old.gain) => chosen = Some(score),
            Some(old)
                if !strictly_better(old.gain, score.gain) && current == Some(score.candidate) =>
            {
                chosen = Some(score)
            }
            _ => {}
        }
    }
    chosen
}

fn requires_rebuild(
    tree: &LearnedTree,
    node: usize,
    state: &TreeState,
    thresholds: &[Vec<f64>],
    offsets: &[usize],
    settings: &FitSettings,
    config: &FpmLearnedConfig,
    weights: GroupWeights,
) -> Result<bool, AicError> {
    let current = incumbent(tree, node, thresholds)?;
    let scores = score_candidates(
        &state.stats[node],
        &state.candidates[node],
        offsets,
        settings,
        weights,
    )?;
    let current_score = scores.iter().find(|s| Some(s.candidate) == current);
    match (current, current_score) {
        (Some(_), None) => Ok(true), // Invalid child partitions bypass tolerance.
        (Some(_), Some(score)) => {
            let rank = 1 + scores
                .iter()
                .filter(|s| strictly_better(s.gain, score.gain))
                .count();
            let allowed =
                ((config.split_rank_tolerance * scores.len() as f64).ceil() as usize).max(1);
            Ok(rank > allowed)
        }
        (None, _) => Ok(best(&scores, None).is_some_and(|s| strictly_better(s.gain, 0.0))),
    }
}

fn refit_leaves(
    tree: &mut LearnedTree,
    stats: &[NodeStats],
    settings: &FitSettings,
    weights: GroupWeights,
    step: f64,
) -> Result<(), AicError> {
    for (node, stat) in stats.iter().enumerate() {
        if tree.left[node] < 0 {
            let total = stat.total;
            // Unused arena slots and empty terminal regions contribute zero.
            let hessian = weights.hessian(total);
            let value = if hessian <= 0.0 {
                0.0
            } else {
                -settings.learning_rate * weights.gradient(total)
                    / (hessian + settings.l2_regularization)
            };
            if !value.is_finite() {
                return Err(invalid("nonfinite leaf value"));
            }
            tree.value[node] = if step == 1.0 {
                value
            } else {
                tree.value[node] + step * (value - tree.value[node])
            };
        }
    }
    Ok(())
}

struct GrowthLeaf {
    node: usize,
    depth: usize,
    rows: Vec<usize>,
    split: Option<Scored>,
}

#[allow(clippy::too_many_arguments)]
fn grow_region(
    tree: &mut LearnedTree,
    state: &mut TreeState,
    root: usize,
    root_depth: usize,
    rows: &Rows<'_>,
    gradients: &[f64],
    settings: &FitSettings,
    thresholds: &[Vec<f64>],
    offsets: &[usize],
    config: &FpmLearnedConfig,
    tree_index: usize,
) -> Result<usize, AicError> {
    let weights = GroupWeights::new(config, rows.anchors.original_rows(), rows.recent.len());
    let reserved = state.regions[root].clone();
    // Route by the still-intact ancestors before clearing the region.
    let members: Vec<_> = (0..rows.len())
        .filter(|&i| {
            let mut node = 0;
            loop {
                if node == root {
                    return true;
                }
                if tree.left[node] < 0 {
                    return false;
                }
                let v = rows.x(i)[tree.feature[node] as usize];
                let left = if v.is_nan() {
                    tree.missing_left[node]
                } else {
                    v <= tree.threshold[node]
                };
                node = if left {
                    tree.left[node] as usize
                } else {
                    tree.right[node] as usize
                };
            }
        })
        .collect();
    state.generation = state
        .generation
        .checked_add(1)
        .ok_or_else(|| invalid("node generation overflow"))?;
    // eager_fit already accumulated this root over exactly `members`, in the
    // same row order. Reuse its current sums and histogram instead of scanning
    // every feature of those rows again when rebuilding the region. Descendant
    // partitions still get fresh statistics below; candidate RNG is unchanged.
    let prepared_root = std::mem::take(&mut state.stats[root]);
    for &node in &reserved {
        tree.left[node] = -1;
        tree.right[node] = -1;
        tree.feature[node] = -1;
        tree.threshold[node] = 0.0;
        tree.value[node] = 0.0;
        tree.missing_left[node] = true;
        state.stats[node] = NodeStats::default();
        state.regions[node] = vec![node];
        state.candidates[node].clear();
    }
    let make_leaf = |node: usize,
                     depth: usize,
                     members: Vec<usize>,
                     can_expand: bool,
                     prepared: Option<NodeStats>,
                     state: &mut TreeState|
     -> Result<GrowthLeaf, AicError> {
        let can_split = can_expand && !settings.max_depth.is_some_and(|d| depth >= d);
        if let Some(mut prepared) = prepared {
            if !can_split {
                prepared.histogram.clear();
            }
            state.stats[node] = prepared;
        } else {
            if can_split {
                state.stats[node].histogram = vec![Sum::default(); *offsets.last().unwrap()];
            }
            for &i in &members {
                add(
                    &mut state.stats[node],
                    gradients[i],
                    rows.count(i),
                    i >= rows.anchors.y.len(),
                    rows.bins(i),
                    offsets,
                );
            }
        }
        if can_split {
            state.candidates[node] =
                sampled_candidates(thresholds, tree_index, node, state.generation, config, None)?;
        }
        let split = if !can_split {
            None
        } else {
            best(
                &score_candidates(
                    &state.stats[node],
                    &state.candidates[node],
                    offsets,
                    settings,
                    weights,
                )?,
                None,
            )
            .filter(|s| strictly_better(s.gain, 0.0))
        };
        Ok(GrowthLeaf {
            node,
            depth,
            rows: members,
            split,
        })
    };
    let mut frontier = vec![make_leaf(
        root,
        root_depth,
        members,
        true,
        Some(prepared_root),
        state,
    )?];
    let mut next_slot = 1;
    let mut changed = 1;
    while next_slot + 1 < reserved.len() {
        let mut winner = None;
        for (index, leaf) in frontier.iter().enumerate() {
            let Some(score) = leaf.split else {
                continue;
            };
            if winner.is_none_or(|old: usize| {
                let previous = frontier[old].split.unwrap();
                strictly_better(score.gain, previous.gain)
                    || (!strictly_better(previous.gain, score.gain)
                        && leaf.node < frontier[old].node)
            }) {
                winner = Some(index);
            }
        }
        let Some(winner) = winner else {
            break;
        };
        let leaf = frontier.swap_remove(winner);
        let split = leaf.split.unwrap().candidate;
        let left = reserved[next_slot];
        let right = reserved[next_slot + 1];
        next_slot += 2;
        tree.left[leaf.node] = left as i32;
        tree.right[leaf.node] = right as i32;
        tree.feature[leaf.node] = i32::from(split.feature);
        tree.threshold[leaf.node] = thresholds[usize::from(split.feature)]
            .get(usize::from(split.bin))
            .copied()
            .unwrap_or(f64::MAX);
        tree.missing_left[leaf.node] = split.missing_left;
        let (mut left_rows, mut right_rows) = (Vec::new(), Vec::new());
        for i in leaf.rows {
            let v = rows.x(i)[usize::from(split.feature)];
            let goes_left = if v.is_nan() {
                split.missing_left
            } else {
                v <= tree.threshold[leaf.node]
            };
            if goes_left {
                left_rows.push(i);
            } else {
                right_rows.push(i);
            }
        }
        let can_expand = next_slot + 1 < reserved.len();
        frontier.push(make_leaf(
            left,
            leaf.depth + 1,
            left_rows,
            can_expand,
            None,
            state,
        )?);
        frontier.push(make_leaf(
            right,
            leaf.depth + 1,
            right_rows,
            can_expand,
            None,
            state,
        )?);
        changed += 2;
    }
    // New child regions own only their descendant slots. The original root
    // retains the entire reserved arena, including unused capacity after collapse.
    fn assign(tree: &LearnedTree, node: usize, regions: &mut [Vec<usize>]) -> Vec<usize> {
        let mut list = vec![node];
        if tree.left[node] >= 0 {
            list.extend(assign(tree, tree.left[node] as usize, regions));
            list.extend(assign(tree, tree.right[node] as usize, regions));
        }
        list.sort_unstable();
        regions[node] = list.clone();
        list
    }
    assign(tree, root, &mut state.regions);
    state.regions[root] = reserved;
    for &node in &state.regions[root] {
        if state.regions[node].len() == 1 {
            state.stats[node].histogram = Vec::new();
            state.candidates[node] = Vec::new();
        }
    }
    Ok(changed)
}

#[allow(clippy::too_many_arguments)]
fn update_structure(
    tree: &mut LearnedTree,
    state: &mut TreeState,
    rows: &Rows<'_>,
    gradients: &[f64],
    settings: &FitSettings,
    thresholds: &[Vec<f64>],
    offsets: &[usize],
    config: &FpmLearnedConfig,
    tree_index: usize,
) -> Result<u64, AicError> {
    let weights = GroupWeights::new(config, rows.anchors.original_rows(), rows.recent.len());
    let mut pending = VecDeque::from([(0, 0)]);
    let mut rebuilt = 0;
    while let Some((node, depth)) = pending.pop_front() {
        if state.regions[node].len() > 1
            && requires_rebuild(
                tree, node, state, thresholds, offsets, settings, config, weights,
            )?
        {
            rebuilt += grow_region(
                tree, state, node, depth, rows, gradients, settings, thresholds, offsets, config,
                tree_index,
            )? as u64;
            continue; // Fresh descendants were already greedily optimized.
        }
        if tree.left[node] >= 0 {
            pending.push_back((tree.left[node] as usize, depth + 1));
            pending.push_back((tree.right[node] as usize, depth + 1));
        }
    }
    Ok(rebuilt)
}

struct EagerResult {
    store: LearnedStore,
    first_mutable_tree: usize,
    /// Replacements for the mutable suffix only; the frozen prefix stays owned
    /// by the original learner throughout fitting and publication.
    states: Vec<TreeState>,
    recent_gradients: Vec<Vec<f64>>,
    rebuilt: u64,
}

fn eager_fit(
    state: &StoreState,
    store: &LearnedStore,
    config: &FpmLearnedConfig,
    rows: &Rows<'_>,
) -> Result<EagerResult, AicError> {
    let mut replacement = store.clone(); // Small prediction arrays, never the N×T learner.
    let mut prefix = vec![store.baseline; rows.len()];
    let start = mutable_start(config, store.trees.len());
    let mut states = Vec::with_capacity(store.trees.len() - start);
    let mut recent_gradients = if lazy(config) {
        vec![vec![0.0; store.trees.len()]; rows.recent.len()]
    } else {
        Vec::new()
    };
    let mut rebuilt = 0;
    let weights = GroupWeights::new(config, rows.anchors.original_rows(), rows.recent.len());
    for (index, tree) in replacement.trees.iter_mut().enumerate() {
        if index < start {
            for (i, p) in prefix.iter_mut().enumerate() {
                *p += tree.value[leaf_index(tree, rows.x(i))];
            }
            continue;
        }
        let gradients: Vec<_> = prefix
            .iter()
            .enumerate()
            .map(|(i, &p)| rows.gradient(i, p))
            .collect();
        if gradients.iter().any(|g| !g.is_finite()) {
            return Err(invalid("nonfinite eager derivative"));
        }
        let old = &state.trees[index];
        let mut next = TreeState {
            stats: accumulate(
                tree,
                rows,
                &gradients,
                &state.offsets,
                structural(config),
                &old.regions,
            ),
            regions: old.regions.clone(),
            candidates: old.candidates.clone(),
            generation: old.generation,
        };
        validate_statistics(&next.stats)?;
        if tree.missing_left.is_empty() {
            tree.missing_left = vec![true; tree.value.len()];
        }
        if structural(config) {
            rebuilt += update_structure(
                tree,
                &mut next,
                rows,
                &gradients,
                &state.settings,
                &state.thresholds,
                &state.offsets,
                config,
                index,
            )?;
            validate_statistics(&next.stats)?;
        }
        refit_leaves(
            tree,
            &next.stats,
            &state.settings,
            weights,
            config.update_step,
        )?;
        validate_tree(tree, state.anchors.width).map_err(invalid)?;
        for (i, p) in prefix.iter_mut().enumerate() {
            *p += tree.value[leaf_index(tree, rows.x(i))];
            if !p.is_finite() {
                return Err(invalid("nonfinite updated prefix prediction"));
            }
        }
        if lazy(config) {
            for (row, &g) in recent_gradients
                .iter_mut()
                .zip(&gradients[state.anchors.y.len()..])
            {
                row[index] = g;
            }
        }
        states.push(next);
    }
    Ok(EagerResult {
        store: replacement,
        first_mutable_tree: start,
        states,
        recent_gradients,
        rebuilt,
    })
}

/// A before-image, not a subtractive undo: floating-point addition is not reversible.
struct Undo {
    tree: usize,
    node: usize,
    slot: Option<usize>,
    value: Sum,
}

/// A tree is visited once per transaction. Record its first before-image for
/// each aggregate, even when many inserted/evicted rows touch the same bin.
/// The bitmap is needed only for the current tree; the journal spans all trees.
struct FirstTouches {
    offsets: Vec<usize>,
    seen: Vec<bool>,
}

impl FirstTouches {
    fn new(stats: &[NodeStats]) -> Self {
        let mut offsets = Vec::with_capacity(stats.len());
        let mut size = 0;
        for node in stats {
            offsets.push(size);
            size += 1 + node.histogram.len();
        }
        Self {
            offsets,
            seen: vec![false; size],
        }
    }

    fn record(&mut self, journal: &mut Vec<Undo>, undo: Undo) {
        let key = self.offsets[undo.node] + undo.slot.map_or(0, |slot| slot + 1);
        if !self.seen[key] {
            self.seen[key] = true;
            journal.push(undo);
        }
    }
}

fn restore(states: &mut [TreeState], journal: &mut Vec<Undo>) {
    for undo in journal.drain(..).rev() {
        let node = &mut states[undo.tree].stats[undo.node];
        if let Some(slot) = undo.slot {
            node.histogram[slot] = undo.value;
        } else {
            node.total = undo.value;
        }
    }
}

fn delta(value: &mut Sum, gradient: f64, remove: bool) -> Result<(), AicError> {
    if remove {
        value.count = value
            .count
            .checked_sub(1)
            .ok_or_else(|| invalid("removal underflow"))?;
        value.recent_count = value
            .recent_count
            .checked_sub(1)
            .ok_or_else(|| invalid("recent removal underflow"))?;
        value.g -= gradient;
        value.recent_g -= gradient;
    } else {
        value.count += 1;
        value.recent_count += 1;
        value.g += gradient;
        value.recent_g += gradient;
    }
    if !value.g.is_finite() || !value.recent_g.is_finite() {
        return Err(invalid("nonfinite incremental aggregate"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn update_row(
    tree: &LearnedTree,
    states: &mut [TreeState],
    tree_index: usize,
    row: &Row,
    gradient: f64,
    remove: bool,
    offsets: &[usize],
    journal: &mut Vec<Undo>,
    touched: &mut FirstTouches,
) -> Result<(), AicError> {
    let mut node = 0;
    loop {
        let stats = &mut states[tree_index].stats[node];
        touched.record(
            journal,
            Undo {
                tree: tree_index,
                node,
                slot: None,
                value: stats.total,
            },
        );
        delta(&mut stats.total, gradient, remove)?;
        if !stats.histogram.is_empty() {
            for (feature, &bin) in row.bins.iter().enumerate() {
                let slot = offsets[feature] + bin as usize;
                touched.record(
                    journal,
                    Undo {
                        tree: tree_index,
                        node,
                        slot: Some(slot),
                        value: stats.histogram[slot],
                    },
                );
                delta(&mut stats.histogram[slot], gradient, remove)?;
            }
        }
        if tree.left[node] < 0 {
            break;
        }
        let v = row.x[tree.feature[node] as usize];
        let left = if v.is_nan() {
            tree.missing_left.get(node).copied().unwrap_or(true)
        } else {
            v <= tree.threshold[node]
        };
        node = if left {
            tree.left[node] as usize
        } else {
            tree.right[node] as usize
        };
    }
    Ok(())
}

struct LazyResult {
    store: LearnedStore,
    new_gradients: Vec<Vec<f64>>,
}

fn lazy_fit(
    state: &mut StoreState,
    store: &LearnedStore,
    config: &FpmLearnedConfig,
    removals: &[usize],
    additions_indices: &[usize],
    target: &LearnedTarget,
) -> Result<Option<LazyResult>, AicError> {
    let mut candidate = store.clone();
    let additions: Vec<_> = additions_indices
        .iter()
        .map(|&i| &state.pending[i])
        .collect();
    let mut prefix = vec![store.baseline; additions.len()];
    let mut new_gradients = vec![vec![0.0; store.trees.len()]; additions.len()];
    let mut journal = Vec::new();
    let recent_count = state.recent.len() - removals.len() + additions.len();
    let weights = GroupWeights::new(config, state.anchors.original_rows(), recent_count);
    let start = mutable_start(config, store.trees.len());
    let result = (|| {
        for (index, tree) in candidate.trees.iter_mut().enumerate() {
            if index < start {
                for (i, row) in additions.iter().enumerate() {
                    prefix[i] += tree.value[leaf_index(tree, &row.x)];
                }
                continue;
            }
            let mut touched = FirstTouches::new(&state.trees[index].stats);
            for &index_to_remove in removals {
                let row = &state.recent[index_to_remove];
                update_row(
                    tree,
                    &mut state.trees,
                    index,
                    row,
                    row.gradients[index],
                    true,
                    &state.offsets,
                    &mut journal,
                    &mut touched,
                )?;
            }
            for (i, row) in additions.iter().enumerate() {
                let g = prefix[i] - row.y;
                if !g.is_finite() {
                    return Err(invalid("nonfinite incoming derivative"));
                }
                new_gradients[i][index] = g;
                update_row(
                    tree,
                    &mut state.trees,
                    index,
                    row,
                    g,
                    false,
                    &state.offsets,
                    &mut journal,
                    &mut touched,
                )?;
            }
            if structural(config) {
                let mut pending = vec![0];
                while let Some(node) = pending.pop() {
                    if state.trees[index].regions[node].len() > 1
                        && requires_rebuild(
                            tree,
                            node,
                            &state.trees[index],
                            &state.thresholds,
                            &state.offsets,
                            &state.settings,
                            config,
                            weights,
                        )?
                    {
                        return Ok(false);
                    }
                    if tree.left[node] >= 0 {
                        pending.push(tree.right[node] as usize);
                        pending.push(tree.left[node] as usize);
                    }
                }
            }
            refit_leaves(
                tree,
                &state.trees[index].stats,
                &state.settings,
                weights,
                config.update_step,
            )?;
            for (i, row) in additions.iter().enumerate() {
                prefix[i] += tree.value[leaf_index(tree, &row.x)];
                if !prefix[i].is_finite() {
                    return Err(invalid("nonfinite incoming prefix"));
                }
            }
        }
        validate_prediction_range(&candidate, target)?;
        Ok(true)
    })();
    match result {
        Ok(true) => Ok(Some(LazyResult {
            store: candidate,
            new_gradients,
        })),
        Ok(false) => {
            restore(&mut state.trees, &mut journal);
            Ok(None)
        }
        Err(error) => {
            restore(&mut state.trees, &mut journal);
            Err(error)
        }
    }
}

/// Returns (predictor, eager, lazy_fallback, rebuilt_nodes). Dataset membership
/// is committed only after fitting succeeds; failed batches remain pending.
pub(super) fn fit_batch(
    state: &mut StoreState,
    store: &LearnedStore,
    config: &FpmLearnedConfig,
    target: &LearnedTarget,
) -> Result<(LearnedStore, bool, bool, u64), AicError> {
    // FIFO membership is staged until the complete fitting transaction succeeds.
    let retained_ids: std::collections::BTreeSet<u64> = state
        .recent
        .iter()
        .chain(&state.pending)
        .rev()
        .take(config.recent_window)
        .filter(|row| {
            config.recent_max_age_observations.is_none_or(|age| {
                state
                    .pending
                    .last()
                    .is_none_or(|last| row.id >= last.id.saturating_sub(age as u64 - 1))
            })
        })
        .map(|row| row.id)
        .collect();
    let removals: Vec<_> = state
        .recent
        .iter()
        .enumerate()
        .filter_map(|(i, row)| (!retained_ids.contains(&row.id)).then_some(i))
        .collect();
    let additions: Vec<_> = state
        .pending
        .iter()
        .enumerate()
        .filter_map(|(i, row)| retained_ids.contains(&row.id).then_some(i))
        .collect();
    let dropped = state.recent.len() + state.pending.len() - retained_ids.len();
    let try_lazy =
        lazy(config) && state.counters.batches_since_eager + 1 < config.eager_refresh_interval;
    if try_lazy {
        if let Some(result) = lazy_fit(state, store, config, &removals, &additions, target)? {
            state.recent.retain(|row| retained_ids.contains(&row.id));
            for (mut row, gradients) in std::mem::take(&mut state.pending)
                .into_iter()
                .filter(|row| retained_ids.contains(&row.id))
                .zip(result.new_gradients)
            {
                row.gradients = gradients;
                state.recent.push_back(row);
            }
            state.counters.evicted += dropped as u64;
            return Ok((result.store, false, false, 0));
        }
    }
    let rows = Rows {
        anchors: &state.anchors,
        recent: state
            .recent
            .iter()
            .chain(&state.pending)
            .filter(|row| retained_ids.contains(&row.id))
            .collect(),
    };
    let result = eager_fit(state, store, config, &rows)?;
    validate_prediction_range(&result.store, target)?;
    // No original training state was touched while fitting. Once every
    // fallible check passes, publish only the staged mutable-tree states.
    for (current, replacement) in state.trees[result.first_mutable_tree..]
        .iter_mut()
        .zip(result.states)
    {
        *current = replacement;
    }
    state.recent.retain(|row| retained_ids.contains(&row.id));
    state.recent.extend(
        std::mem::take(&mut state.pending)
            .into_iter()
            .filter(|row| retained_ids.contains(&row.id)),
    );
    if lazy(config) {
        for (row, gradients) in state.recent.iter_mut().zip(result.recent_gradients) {
            row.gradients = gradients;
        }
    }
    state.counters.evicted += dropped as u64;
    Ok((result.store, true, try_lazy, result.rebuilt))
}

/// A conservative bound avoids publishing a model whose finite raw leaves can
/// nevertheless overflow when summed or passed through the exponential link.
fn validate_prediction_range(store: &LearnedStore, target: &LearnedTarget) -> Result<(), AicError> {
    let (mut low, mut high) = (store.baseline, store.baseline);
    for tree in &store.trees {
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        let mut nodes = vec![0];
        while let Some(node) = nodes.pop() {
            if tree.left[node] < 0 {
                minimum = minimum.min(tree.value[node]);
                maximum = maximum.max(tree.value[node]);
            } else {
                nodes.push(tree.left[node] as usize);
                nodes.push(tree.right[node] as usize);
            }
        }
        low += minimum;
        high += maximum;
    }
    if !low.is_finite()
        || !high.is_finite()
        || (*target == LearnedTarget::LogMs && high > f64::MAX.ln())
    {
        return Err(invalid("updated prediction range can overflow"));
    }
    Ok(())
}

#[cfg(test)]
mod journal_tests {
    use super::*;

    #[test]
    fn repeated_bin_updates_keep_one_exact_before_image() {
        let tree = LearnedTree {
            left: vec![1, -1, -1],
            right: vec![2, -1, -1],
            feature: vec![0, -1, -1],
            threshold: vec![0.5, 0.0, 0.0],
            value: vec![0.0; 3],
            missing_left: vec![true; 3],
        };
        let original = Sum {
            g: 0.1,
            count: 3,
            recent_g: 0.1,
            recent_count: 3,
        };
        let mut states = vec![TreeState {
            stats: vec![
                NodeStats {
                    total: original,
                    histogram: vec![original, Sum::default(), Sum::default()],
                },
                NodeStats {
                    total: original,
                    histogram: Vec::new(),
                },
                NodeStats::default(),
            ],
            regions: vec![vec![0, 1, 2], vec![1], vec![2]],
            candidates: vec![Vec::new(); 3],
            generation: 0,
        }];
        let before = bincode::serialize(&states).unwrap();
        let row = Row {
            id: 4,
            x: vec![0.25],
            bins: vec![0],
            y: 1.0,
            wall_ms: 1.0,
            gradients: Vec::new(),
        };
        let mut journal = Vec::new();
        let mut touched = FirstTouches::new(&states[0].stats);
        for _ in 0..64 {
            for remove in [false, true] {
                update_row(
                    &tree,
                    &mut states,
                    0,
                    &row,
                    1e10,
                    remove,
                    &[0, 3],
                    &mut journal,
                    &mut touched,
                )
                .unwrap();
            }
        }
        // Both totals and the shared histogram bin are each journaled once.
        assert_eq!(journal.len(), 3);
        assert_ne!(states[0].stats[0].total.g.to_bits(), original.g.to_bits());
        restore(&mut states, &mut journal);
        assert_eq!(bincode::serialize(&states).unwrap(), before);
    }
}
