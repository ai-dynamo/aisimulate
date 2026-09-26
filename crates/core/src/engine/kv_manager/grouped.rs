// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cold, request-owned cache groups sharing one rank-local physical byte budget.

use crate::perfmodel::FpmCacheGroup;

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct GroupedLease {
    /// Absolute logical block ranges, one per configured group. Full-history
    /// groups start at zero; expired window blocks are absent.
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
        let mut projected = GroupedLease::default();
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
    use crate::perfmodel::FpmCacheKind;

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
                },
                FpmCacheGroup {
                    name: "window".into(),
                    kind: FpmCacheKind::Attention,
                    num_layers: 1,
                    block_size_tokens: 2,
                    page_size_bytes: 6,
                    sliding_window: Some(3),
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
}
