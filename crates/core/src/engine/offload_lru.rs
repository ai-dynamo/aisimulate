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
/// deterministic tie-break.
pub(crate) struct Lru {
    nodes: Vec<LruNode>,
    /// Unlinked nodes reused before the slab grows.
    vacant: Vec<NonZeroU32>,
}

#[derive(Clone, Copy)]
struct LruNode {
    key: HostBlockKey,
    older: u32,
    newer: u32,
}

const SENTINEL: u32 = 0;

impl Default for Lru {
    fn default() -> Self {
        Self {
            nodes: vec![LruNode {
                key: HostBlockKey::new(0),
                older: SENTINEL,
                newer: SENTINEL,
            }],
            vacant: Vec::new(),
        }
    }
}

impl Lru {
    /// Make `key` the newest block, linking a node into `slot` on first touch.
    pub(crate) fn touch(&mut self, slot: &mut Option<NonZeroU32>, key: HostBlockKey) {
        let node = match *slot {
            Some(node) => {
                self.unlink(node);
                node
            }
            None => *slot.insert(self.allocate(key)),
        };
        self.push_newest(node);
    }

    pub(crate) fn remove(&mut self, slot: &mut Option<NonZeroU32>) {
        if let Some(node) = slot.take() {
            self.unlink(node);
            self.vacant.push(node);
        }
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
        });
        node
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
            match rng.random_range(0..3) {
                0 => {
                    lru.touch(&mut slots[index], block);
                    expected.retain(|key| *key != block);
                    expected.push(block);
                }
                1 => {
                    lru.remove(&mut slots[index]);
                    expected.retain(|key| *key != block);
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
