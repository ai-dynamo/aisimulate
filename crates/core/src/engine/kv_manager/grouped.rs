// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cold, request-owned cache groups sharing one rank-local physical byte budget.

use crate::perfmodel::{FpmCacheGroup, FpmCacheKind};

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct GroupedLease {
    /// Blocks that the request holds across every configured group. Expired
    /// window blocks are not counted. A state group counts its fixed blocks.
    blocks: usize,
    bytes: u64,
}

impl GroupedLease {
    pub(crate) fn blocks(&self) -> usize {
        self.blocks
    }
}

pub(crate) struct GroupedKvPool {
    /// Attention and convolution groups. Their blocks follow the request length.
    groups: Vec<FpmCacheGroup>,
    /// Blocks that the state groups hold for a request with a positive target.
    state_blocks: usize,
    /// Bytes of `state_blocks`, or `None` when they overflow.
    state_bytes: Option<u64>,
    capacity_bytes: u64,
    used_bytes: u64,
    active_blocks: usize,
}

impl GroupedKvPool {
    /// `groups` must have passed [`FpmCacheGroup::validate`], as
    /// `EngineConfig::validate` checks once. Projections do not repeat it.
    pub(crate) fn new(groups: Vec<FpmCacheGroup>, capacity_bytes: u64) -> Self {
        debug_assert!(groups.iter().all(|group| group.validate().is_ok()));
        let (state, groups): (Vec<_>, Vec<_>) = groups
            .into_iter()
            .partition(|group| group.kind == FpmCacheKind::State);
        let mut state_blocks = 0;
        let mut state_bytes = Some(0u64);
        for group in &state {
            // `validate` guarantees a positive count for state groups.
            let blocks = group.blocks_per_request.map_or(0, u64::from);
            state_blocks += blocks as usize;
            state_bytes = blocks
                .checked_mul(group.page_size_bytes)
                .and_then(|bytes| state_bytes?.checked_add(bytes));
        }
        Self {
            groups,
            state_blocks,
            state_bytes,
            capacity_bytes,
            used_bytes: 0,
            active_blocks: 0,
        }
    }

    fn project(&self, computed_before: usize, computed_after: usize) -> Option<GroupedLease> {
        if computed_after < computed_before {
            return None;
        }
        // A state group holds its fixed blocks for any positive target.
        let mut projected = if computed_after == 0 {
            GroupedLease::default()
        } else {
            GroupedLease {
                blocks: self.state_blocks,
                bytes: self.state_bytes?,
            }
        };
        for group in &self.groups {
            // A forward must retain the history needed by its first query,
            // together with every token in the scheduled chunk. Freeing to
            // the final query's window here would undercount prefill storage.
            let range = group.validated_block_range(computed_before as u64, computed_after as u64);
            let blocks = range.end - range.start;
            let bytes = blocks.checked_mul(group.page_size_bytes)?;
            projected.bytes = projected.bytes.checked_add(bytes)?;
            projected.blocks += blocks as usize;
        }
        Some(projected)
    }

    pub(crate) fn required_bytes(
        &self,
        computed_before: usize,
        computed_after: usize,
    ) -> Option<u64> {
        self.project(computed_before, computed_after)
            .map(|lease| lease.bytes)
    }

    /// Reserve every group atomically. Expired blocks become reusable in this
    /// transition, but failed attempts leave both the lease and pool unchanged.
    pub(crate) fn allocate(
        &mut self,
        lease: &mut Option<GroupedLease>,
        computed_before: usize,
        computed_after: usize,
    ) -> bool {
        let Some(projected) = self.project(computed_before, computed_after) else {
            return false;
        };
        let previous_bytes = lease.as_ref().map_or(0, |lease| lease.bytes);
        let other_bytes = self.used_bytes - previous_bytes;
        if projected.bytes > self.capacity_bytes - other_bytes {
            return false;
        }
        self.used_bytes = other_bytes + projected.bytes;
        self.active_blocks = self.active_blocks - lease.as_ref().map_or(0, GroupedLease::blocks)
            + projected.blocks();
        *lease = Some(projected);
        true
    }

    pub(crate) fn release(&mut self, lease: &mut Option<GroupedLease>) {
        if let Some(lease) = lease.take() {
            self.used_bytes -= lease.bytes;
            self.active_blocks -= lease.blocks();
        }
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    pub(crate) fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    pub(crate) fn active_blocks(&self) -> usize {
        self.active_blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_groups_reserve_atomically_and_reuse_expired_pages() {
        let mut pool = GroupedKvPool::new(
            vec![
                FpmCacheGroup {
                    name: "full".into(),
                    kind: FpmCacheKind::Attention,
                    num_layers: 1,
                    block_size_tokens: 4,
                    page_size_bytes: 8,
                    sliding_window: None,
                    blocks_per_request: None,
                },
                FpmCacheGroup {
                    name: "window".into(),
                    kind: FpmCacheKind::Attention,
                    num_layers: 1,
                    block_size_tokens: 2,
                    page_size_bytes: 6,
                    sliding_window: Some(3),
                    blocks_per_request: None,
                },
            ],
            42,
        );
        let mut first = None;
        let mut second = None;
        assert!(pool.allocate(&mut first, 0, 4)); // 8 + 2 * 6 = 20
        assert!(pool.allocate(&mut second, 0, 2)); // 8 + 6 = 14
        assert!(!pool.allocate(&mut first, 4, 8)); // 16 + 3 * 6 = 34, plus 14
        assert_eq!(first.as_ref().unwrap().blocks(), 3); // 0..1 and 0..2
        assert_eq!(pool.used_bytes(), 34);
        pool.release(&mut second);
        assert!(pool.allocate(&mut first, 4, 8));
        assert_eq!(first.as_ref().unwrap().blocks(), 5); // 0..2 and 1..4
        assert_eq!(pool.used_bytes(), 34);
        assert!(pool.allocate(&mut first, 8, 9));
        assert_eq!(first.as_ref().unwrap().blocks(), 5); // 0..3 and 3..5
        assert_eq!(pool.used_bytes(), 36);
        pool.release(&mut first);
        pool.release(&mut first);
        assert_eq!(pool.used_bytes(), 0);
        assert_eq!(pool.active_blocks(), 0);
    }

    #[test]
    fn state_group_reserves_constant_bytes_and_releases_them() {
        let mut pool = GroupedKvPool::new(
            vec![
                FpmCacheGroup {
                    name: "full".into(),
                    kind: FpmCacheKind::Attention,
                    num_layers: 1,
                    block_size_tokens: 4,
                    page_size_bytes: 8,
                    sliding_window: None,
                    blocks_per_request: None,
                },
                FpmCacheGroup {
                    name: "kda".into(),
                    kind: FpmCacheKind::State,
                    num_layers: 1,
                    block_size_tokens: 4,
                    page_size_bytes: 10,
                    sliding_window: None,
                    blocks_per_request: Some(2),
                },
            ],
            100,
        );
        let mut lease = None;
        assert!(pool.allocate(&mut lease, 0, 4)); // 8 + 2 * 10 = 28
        assert_eq!(lease.as_ref().unwrap().blocks(), 3); // 0..1 and the state's 0..2
        assert_eq!(pool.used_bytes(), 28);
        assert!(pool.allocate(&mut lease, 4, 8)); // 2 * 8 + 2 * 10 = 36
        assert_eq!(lease.as_ref().unwrap().blocks(), 4); // 0..2 and the state's 0..2
        assert_eq!(pool.used_bytes(), 36);
        assert_eq!(pool.active_blocks(), 4);
        pool.release(&mut lease);
        assert_eq!(pool.used_bytes(), 0);
        assert_eq!(pool.active_blocks(), 0);
    }

    #[test]
    fn state_groups_charge_their_sum_only_for_a_positive_target() {
        let state = |name: &str, page_size_bytes, blocks| FpmCacheGroup {
            name: name.into(),
            kind: FpmCacheKind::State,
            num_layers: 1,
            block_size_tokens: 4,
            page_size_bytes,
            sliding_window: None,
            blocks_per_request: Some(blocks),
        };
        let mut pool = GroupedKvPool::new(vec![state("kda", 10, 2), state("tail", 6, 1)], 100);
        assert_eq!(pool.required_bytes(0, 0), Some(0));
        assert_eq!(pool.required_bytes(0, 1), Some(26)); // 2 * 10 + 6
        assert_eq!(pool.required_bytes(7, 9), Some(26));
        let mut lease = None;
        assert!(pool.allocate(&mut lease, 0, 0));
        assert_eq!(pool.active_blocks(), 0);
        assert!(pool.allocate(&mut lease, 0, 1));
        assert_eq!(pool.active_blocks(), 3);

        // State bytes beyond u64 fail every positive target, as the sum per projection did.
        let mut pool = GroupedKvPool::new(vec![state("huge", 1 << 53, 1 << 11)], 100);
        assert_eq!(pool.required_bytes(0, 0), Some(0));
        assert_eq!(pool.required_bytes(0, 1), None);
        assert!(!pool.allocate(&mut None, 0, 1));
    }
}
