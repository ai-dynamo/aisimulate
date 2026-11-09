// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cold, request-owned cache groups sharing one rank-local physical byte budget.

use crate::perfmodel::FpmCacheGroup;

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct GroupedLease {
    /// Absolute logical block ranges, one per configured group. Full-history
    /// groups start at zero; expired window blocks are absent. A state
    /// group's range `0..n` counts the blocks that the request holds.
    ranges: Vec<std::ops::Range<usize>>,
    bytes: u64,
}

impl GroupedLease {
    pub(crate) fn blocks(&self) -> usize {
        self.ranges.iter().map(std::ops::Range::len).sum()
    }
}

pub(crate) struct GroupedKvPool {
    groups: Vec<FpmCacheGroup>,
    capacity_bytes: u64,
    used_bytes: u64,
    active_blocks: usize,
}

impl GroupedKvPool {
    pub(crate) fn new(groups: Vec<FpmCacheGroup>, capacity_bytes: u64) -> Self {
        Self {
            groups,
            capacity_bytes,
            used_bytes: 0,
            active_blocks: 0,
        }
    }

    fn project(&self, computed_before: usize, computed_after: usize) -> Option<GroupedLease> {
        if computed_after < computed_before {
            return None;
        }
        let mut projected = GroupedLease {
            ranges: Vec::with_capacity(self.groups.len()),
            bytes: 0,
        };
        for group in &self.groups {
            // A forward must retain the history needed by its first query,
            // together with every token in the scheduled chunk. Freeing to
            // the final query's window here would undercount prefill storage.
            let range = group
                .block_range(computed_before as u64, computed_after as u64)
                .ok()?;
            let range = range.start as usize..range.end as usize;
            let bytes = (range.len() as u64).checked_mul(group.page_size_bytes)?;
            projected.bytes = projected.bytes.checked_add(bytes)?;
            projected.ranges.push(range);
        }
        Some(projected)
    }

    /// Bytes that allocate() reserves for this forward, computed without building the lease.
    pub(crate) fn required_bytes(
        &self,
        computed_before: usize,
        computed_after: usize,
    ) -> Option<u64> {
        if computed_after < computed_before {
            return None;
        }
        self.groups.iter().try_fold(0u64, |total, group| {
            let range = group
                .block_range(computed_before as u64, computed_after as u64)
                .ok()?;
            let bytes = (range.end - range.start).checked_mul(group.page_size_bytes)?;
            total.checked_add(bytes)
        })
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
    use crate::perfmodel::FpmCacheKind;

    fn group(kind: FpmCacheKind, block: u32, page: u64, window: Option<u32>) -> FpmCacheGroup {
        FpmCacheGroup {
            name: "group".into(),
            kind,
            num_layers: 1,
            block_size_tokens: block,
            page_size_bytes: page,
            sliding_window: window,
        }
    }

    #[test]
    fn required_bytes_equals_the_projected_lease_bytes() {
        let full = group(FpmCacheKind::Attention, 4, 8, None);
        let window = group(FpmCacheKind::Attention, 2, 6, Some(3));
        let convolution = group(FpmCacheKind::Convolution, 2, 2, Some(2));
        let largest_page = 1 << 53; // The largest page_size_bytes that FpmCacheGroup::validate accepts.
        let pools = [
            GroupedKvPool::new(vec![full.clone(), window.clone(), convolution.clone()], 0),
            GroupedKvPool::new(vec![full], 0),
            GroupedKvPool::new(vec![window], 0),
            GroupedKvPool::new(vec![convolution], 0),
            GroupedKvPool::new(
                vec![group(FpmCacheKind::Attention, 1, largest_page, None)],
                0,
            ),
            GroupedKvPool::new(
                vec![group(FpmCacheKind::Attention, 1, largest_page, None); 2],
                0,
            ),
            // FpmCacheGroup::validate rejects a convolution group without a window.
            GroupedKvPool::new(vec![group(FpmCacheKind::Convolution, 2, 2, None)], 0),
            // No group exists to reject a shrinking target; only the guard in required_bytes does.
            GroupedKvPool::new(vec![], 0),
        ];
        let mut spans: Vec<(usize, usize)> = (0..=12)
            .flat_map(|before| (0..=12).map(move |after| (before, after)))
            .collect();
        spans.extend([(0, 1024), (0, 2048), (0, usize::MAX)]);
        for (index, pool) in pools.iter().enumerate() {
            for &(before, after) in &spans {
                assert_eq!(
                    pool.required_bytes(before, after),
                    pool.project(before, after).map(|lease| lease.bytes),
                    "pool {index}, {before}..{after}"
                );
            }
        }
        // Target 0, inside a block, at a block boundary, then window eviction. A shrinking target projects nothing.
        for (before, after, bytes) in [
            (0, 0, Some(0)),
            (0, 3, Some(24)),
            (0, 4, Some(24)),
            (4, 8, Some(40)),
            (8, 9, Some(40)),
            (9, 9, Some(38)),
            (5, 4, None),
        ] {
            assert_eq!(
                pools[0].required_bytes(before, after),
                bytes,
                "{before}..{after}"
            );
        }
        // The pool has no 2**53 limit, unlike FpmCacheGroup::bytes_for_forward.
        assert_eq!(pools[4].required_bytes(0, 2), Some(1 << 54));
        assert_eq!(pools[4].required_bytes(0, 2048), None); // checked_mul overflows u64.
        assert_eq!(pools[5].required_bytes(0, 1024), None); // checked_add overflows u64.
        assert_eq!(pools[6].required_bytes(0, 4), None);
    }

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
        assert_eq!(first.as_ref().unwrap().ranges, vec![0..1, 0..2]);
        assert_eq!(pool.used_bytes(), 34);
        pool.release(&mut second);
        assert!(pool.allocate(&mut first, 4, 8));
        assert_eq!(first.as_ref().unwrap().ranges, vec![0..2, 1..4]);
        assert_eq!(pool.used_bytes(), 34);
        assert!(pool.allocate(&mut first, 8, 9));
        assert_eq!(first.as_ref().unwrap().ranges, vec![0..3, 3..5]);
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
        assert_eq!(lease.as_ref().unwrap().ranges, vec![0..1, 0..2]);
        assert_eq!(pool.used_bytes(), 28);
        assert!(pool.allocate(&mut lease, 4, 8)); // 2 * 8 + 2 * 10 = 36
        assert_eq!(lease.as_ref().unwrap().ranges, vec![0..2, 0..2]);
        assert_eq!(pool.used_bytes(), 36);
        assert_eq!(pool.active_blocks(), 4);
        pool.release(&mut lease);
        assert_eq!(pool.used_bytes(), 0);
        assert_eq!(pool.active_blocks(), 0);
    }
}
