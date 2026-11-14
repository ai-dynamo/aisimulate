// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Least-recently-used order shared by the G2 and G3 tiers.

use std::num::NonZeroU32;

use crate::engine::host_offload::HostBlockKey;

/// Blocks from least to most recently used.
///
/// Blocks form a circular doubly linked list over a slab, so a touch or removal
/// relinks in O(1). The caller keeps each block's node in a slot, normally a
/// field of the block's own entry; node 0 is the sentinel between the newest
/// and oldest blocks. Callers make a cohort newest in key order, the tiers'
/// deterministic tie-break, unless a G2 tier keeps release order (a free
/// queue) or ranks blocks when their store starts.
///
/// A caller that keeps making the same blocks the newest has them join a run
/// as it moves them. The run is whole while each of its blocks is linked
/// where the caller last put it, so the caller can move them again without
/// finding them. A run stops being whole when one of its blocks is touched
/// alone, removed or joined to another run.
pub(crate) struct Lru {
    nodes: Vec<LruNode>,
    /// Unlinked nodes reused before the slab grows.
    vacant: Vec<NonZeroU32>,
    /// See [`Self::touches`].
    touches: u64,
    /// Whether each run is whole, by run. [`NO_RUN`] never is.
    whole_runs: Vec<bool>,
}

#[derive(Clone, Copy)]
struct LruNode {
    key: HostBlockKey,
    older: u32,
    newer: u32,
    /// The block's run, [`NO_RUN`] until it joins one.
    run: u32,
}

const SENTINEL: u32 = 0;
const NO_RUN: u32 = 0;

impl Default for Lru {
    fn default() -> Self {
        Self {
            nodes: vec![LruNode {
                key: HostBlockKey::new(0),
                older: SENTINEL,
                newer: SENTINEL,
                run: NO_RUN,
            }],
            vacant: Vec::new(),
            touches: 0,
            whole_runs: vec![false],
        }
    }
}

impl Lru {
    /// Make `key` the newest block, linking a node into `slot` on first touch.
    pub(crate) fn touch(&mut self, slot: &mut Option<NonZeroU32>, key: HostBlockKey) {
        let node = match *slot {
            Some(node) => {
                self.unlink(node);
                self.join_run(node, NO_RUN);
                node
            }
            None => *slot.insert(self.allocate(key)),
        };
        self.push_newest(node);
        self.touches += 1;
    }

    /// Make the adjacent blocks from `oldest` through `newest` the newest,
    /// keeping their order. No run is marked, so the span's blocks must
    /// already be in the caller's run.
    pub(crate) fn touch_span(&mut self, oldest: NonZeroU32, newest: NonZeroU32) {
        let after = self.nodes[newest.get() as usize].newer;
        if after == SENTINEL {
            return;
        }
        let before = self.nodes[oldest.get() as usize].older;
        self.nodes[before as usize].newer = after;
        self.nodes[after as usize].older = before;
        let last = self.nodes[SENTINEL as usize].older;
        self.nodes[last as usize].newer = oldest.get();
        self.nodes[oldest.get() as usize].older = last;
        self.nodes[newest.get() as usize].newer = SENTINEL;
        self.nodes[SENTINEL as usize].older = newest.get();
        self.touches += 1;
    }

    /// From `oldest`, take the newer neighbours while their keys match `keys`
    /// read backwards, join them all to `run` and make the span the newest.
    /// Returns its newest block and how many keys matched.
    pub(crate) fn touch_matching_span(
        &mut self,
        run: u32,
        oldest: NonZeroU32,
        keys: &[HostBlockKey],
    ) -> (NonZeroU32, usize) {
        let mut newest = oldest;
        let mut rest = keys;
        loop {
            self.join_run(newest, run);
            let newer = self.nodes[newest.get() as usize].newer;
            match (NonZeroU32::new(newer), rest.split_last()) {
                (Some(node), Some((key, before))) if self.nodes[newer as usize].key == *key => {
                    newest = node;
                    rest = before;
                }
                _ => break,
            }
        }
        self.touch_span(oldest, newest);
        (newest, keys.len() - rest.len())
    }

    /// A new run, whole and without blocks.
    pub(crate) fn new_run(&mut self) -> u32 {
        let run = u32::try_from(self.whole_runs.len()).expect("offload LRU run overflow");
        self.whole_runs.push(true);
        run
    }

    /// Whether `run` was whole. It is whole afterwards: the caller rejoins
    /// its blocks.
    pub(crate) fn resume_run(&mut self, run: u32) -> bool {
        std::mem::replace(&mut self.whole_runs[run as usize], true)
    }

    pub(crate) fn remove(&mut self, slot: &mut Option<NonZeroU32>) {
        if let Some(node) = slot.take() {
            self.unlink(node);
            self.join_run(node, NO_RUN);
            self.vacant.push(node);
        }
    }

    /// How many times a block or a run was made the newest. While the count
    /// stands, the newest blocks are the same, in the same order, less any
    /// removed.
    pub(crate) fn touches(&self) -> u64 {
        self.touches
    }

    /// Number of linked blocks.
    pub(crate) fn len(&self) -> usize {
        self.nodes.len() - 1 - self.vacant.len()
    }

    /// The `count` oldest blocks that pass `is_eligible`.
    pub(crate) fn victims(
        &self,
        count: usize,
        mut is_eligible: impl FnMut(HostBlockKey) -> bool,
    ) -> Vec<HostBlockKey> {
        self.oldest_first()
            .filter(|key| is_eligible(*key))
            .take(count)
            .collect()
    }

    fn oldest_first(&self) -> impl Iterator<Item = HostBlockKey> + '_ {
        self.oldest_after(None).map(|(_, key)| key)
    }

    /// Linked nodes newer than `cursor`, oldest first; all of them for `None`.
    pub(crate) fn oldest_after(
        &self,
        cursor: Option<NonZeroU32>,
    ) -> impl Iterator<Item = (NonZeroU32, HostBlockKey)> + '_ {
        let start = cursor.map_or(SENTINEL, NonZeroU32::get);
        let mut node = self.nodes[start as usize].newer;
        std::iter::from_fn(move || {
            let current = NonZeroU32::new(node)?;
            let LruNode { key, newer, .. } = self.nodes[node as usize];
            node = newer;
            Some((current, key))
        })
    }

    fn allocate(&mut self, key: HostBlockKey) -> NonZeroU32 {
        if let Some(node) = self.vacant.pop() {
            self.nodes[node.get() as usize].key = key;
            return node;
        }
        let node = u32::try_from(self.nodes.len())
            .ok()
            .and_then(NonZeroU32::new)
            .expect("offload LRU node overflow");
        self.nodes.push(LruNode {
            key,
            older: SENTINEL,
            newer: SENTINEL,
            run: NO_RUN,
        });
        node
    }

    /// Move `node` into `run`. The run it leaves is no longer whole.
    fn join_run(&mut self, node: NonZeroU32, run: u32) {
        let left = std::mem::replace(&mut self.nodes[node.get() as usize].run, run);
        if left != run {
            self.whole_runs[left as usize] = false;
        }
    }

    fn unlink(&mut self, node: NonZeroU32) {
        let LruNode { older, newer, .. } = self.nodes[node.get() as usize];
        self.nodes[older as usize].newer = newer;
        self.nodes[newer as usize].older = older;
    }

    fn push_newest(&mut self, node: NonZeroU32) {
        let node = node.get();
        let newest = self.nodes[SENTINEL as usize].older;
        let linked = &mut self.nodes[node as usize];
        linked.older = newest;
        linked.newer = SENTINEL;
        self.nodes[newest as usize].newer = node;
        self.nodes[SENTINEL as usize].older = node;
    }

    /// Bounded so that a corrupted, cyclic list fails the comparison.
    #[cfg(test)]
    pub(crate) fn order(&self) -> Vec<HostBlockKey> {
        self.oldest_first().take(self.nodes.len()).collect()
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    #[test]
    fn lru_matches_move_to_back_list() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut lru = Lru::default();
        // Tiers keep each block's node in its entry.
        let mut slots = [None; 64];
        let mut expected = Vec::new();
        for _ in 0..20_000 {
            let index = rng.random_range(0..slots.len());
            let block = HostBlockKey::new(index as u64);
            match rng.random_range(0..4) {
                0 => {
                    lru.touch(&mut slots[index], block);
                    expected.retain(|key| *key != block);
                    expected.push(block);
                }
                1 => {
                    lru.remove(&mut slots[index]);
                    expected.retain(|key| *key != block);
                }
                2 => {
                    // A span of adjacent blocks, from `block` on, moves as one.
                    if let Some(start) = expected.iter().position(|key| *key == block) {
                        let end = rng.random_range(start..expected.len());
                        let node = |key: HostBlockKey| slots[key.sequence_hash() as usize].unwrap();
                        lru.touch_span(node(expected[start]), node(expected[end]));
                        let span: Vec<_> = expected.drain(start..=end).collect();
                        expected.extend(span);
                    }
                }
                _ => {
                    let count = rng.random_range(0..8);
                    let skipped = rng.random_range(0..3);
                    let is_eligible = |key: HostBlockKey| key.sequence_hash() % 3 != skipped;
                    let victims: Vec<_> = expected
                        .iter()
                        .copied()
                        .filter(|key| is_eligible(*key))
                        .take(count)
                        .collect();
                    assert_eq!(lru.victims(count, is_eligible), victims);
                }
            }
            assert_eq!(lru.order(), expected);
        }
        // Removed nodes are reused: the slab holds at most every key plus the sentinel.
        assert!(lru.nodes.len() <= slots.len() + 1);
    }
}
