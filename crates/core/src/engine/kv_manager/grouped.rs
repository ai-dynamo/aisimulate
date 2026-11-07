// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-owned cache groups sharing one rank-local physical budget. Byte
//! mode replays them cold. Block mode keeps the blocks of every group in one
//! block pool, so that cached blocks stay until eviction.

use crate::engine::cache::vllm_block_pool::{
    BlockCopyId, BlockReservation, CacheKey, PrefixHit, VllmBlockPool,
};
use crate::engine::common::hashing::SequenceHash;
use crate::perfmodel::{FpmCacheGroup, FpmCacheKind};

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct GroupedLease {
    /// Absolute logical block ranges, one per configured group. Full-history
    /// groups start at zero; expired window blocks are absent. A state
    /// group's range `0..n` counts the blocks that the request holds.
    ranges: Vec<std::ops::Range<usize>>,
    bytes: u64,
    /// The copies of every group in block mode.
    pub(crate) block_lease: Option<GroupedBlockLease>,
    /// What the first pass of a destination load still takes: blocks in
    /// block mode, bytes in byte mode. 0 once that pass has allocated, and
    /// on every other lease.
    pub(crate) first_pass_need: u64,
}

impl GroupedLease {
    pub(crate) fn blocks(&self) -> usize {
        match &self.block_lease {
            Some(lease) => lease.groups.iter().map(GroupLease::live_copies).sum(),
            None => self.ranges.iter().map(std::ops::Range::len).sum(),
        }
    }
}

/// One entry per configured group, in configuration order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct GroupedBlockLease {
    pub(crate) groups: Vec<GroupLease>,
    /// Whether the request has allocated once. The copy blocks of a partial
    /// hit come with the first allocation only.
    pub(crate) allocated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum GroupLease {
    /// `copies[i]` holds logical block `i`. It is `None` for an expired
    /// window block or a null block below a sliding-window hit. Such blocks
    /// lie below `first_live`. `finalized` counts the blocks that the request
    /// had completed at its last finalization. The next finalization visits
    /// only the full-attention blocks above it, as vLLM's `num_cached_block`
    /// does, and only the sliding-window retention positions above
    /// `finalized * block_size`. The window run that it publishes before such
    /// a position can include blocks below `finalized`.
    Attention {
        copies: Vec<Option<BlockCopyId>>,
        first_live: usize,
        finalized: usize,
    },
    State(GroupStateLease),
    /// One block for the whole request. A ring never caches it.
    Ring {
        copy: Option<BlockCopyId>,
    },
}

impl GroupLease {
    fn live_copies(&self) -> usize {
        match self {
            // The `None` copies lie exactly below `first_live`, so a count per
            // step does not scan the whole context.
            Self::Attention {
                copies, first_live, ..
            } => {
                debug_assert_eq!(copies.iter().flatten().count(), copies.len() - first_live);
                copies.len() - first_live
            }
            Self::State(state) => [
                state.running,
                state.hit.map(|(id, _)| id),
                state.previous.map(|(id, _)| id),
                state.copy,
                state.checkpoint.map(|(id, _)| id),
            ]
            .into_iter()
            .flatten()
            .count(),
            Self::Ring { copy } => usize::from(copy.is_some()),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct GroupStateLease {
    /// The block that the request writes.
    pub(crate) running: Option<BlockCopyId>,
    /// A cached hit block, and the index of the block whose end it covers.
    pub(crate) hit: Option<(BlockCopyId, usize)>,
    /// The running block before a boundary crossing, and the step after
    /// which it can go.
    pub(crate) previous: Option<(BlockCopyId, u64)>,
    /// The copy block of a partial hit.
    pub(crate) copy: Option<BlockCopyId>,
    /// A vLLM 0.32.0 prefill checkpoint block, and its token position.
    pub(crate) checkpoint: Option<(BlockCopyId, usize)>,
}

/// The block pool of block mode. Every group takes blocks of one page size
/// from it, and every cached block has a `CacheKey::Grouped` key.
pub(crate) struct GroupedBlockPool {
    pool: VllmBlockPool,
    page_size_bytes: u64,
    /// See `GroupedKvPool::retry_epoch`.
    retry_epoch: u64,
    /// The most active blocks seen since the last reset, for tests that
    /// observe a step's peak after the step.
    #[cfg(test)]
    peak_active_blocks: std::cell::Cell<usize>,
}

impl GroupedBlockPool {
    pub(crate) fn capacity_blocks(&self) -> usize {
        self.pool.capacity()
    }

    #[cfg(test)]
    pub(crate) fn peak_active_blocks(&self) -> usize {
        self.peak_active_blocks.get()
    }

    #[cfg(test)]
    pub(crate) fn reset_peak_active_blocks(&self) {
        self.peak_active_blocks.set(self.pool.num_active());
    }

    fn note_peak(&self) {
        #[cfg(test)]
        self.peak_active_blocks
            .set(self.peak_active_blocks.get().max(self.pool.num_active()));
    }

    pub(crate) fn key_hit(&self, group: u16, hash: SequenceHash) -> Option<PrefixHit> {
        self.pool.key_hit(CacheKey::Grouped { group, hash })
    }

    /// Pin `hits` and reserve `fresh` blocks, or change nothing when they do
    /// not fit. Every hit must be resident. A hit block in the inactive LRU
    /// leaves it and costs capacity again. An active hit block costs nothing.
    pub(crate) fn reserve(&mut self, hits: &[CacheKey], fresh: usize) -> Option<BlockReservation> {
        self.pool
            .reserve_keys(hits.iter().copied(), hits.len() + fresh)
            .map(|outcome| outcome.reservation)
    }

    /// Turn the pinned hits into request references. The copies follow the
    /// order of `hits`.
    pub(crate) fn activate(&mut self, reservation: &mut BlockReservation) -> Vec<BlockCopyId> {
        let copies = self
            .pool
            .activate_keys(reservation)
            .map(|(_, id)| id)
            .collect();
        self.note_peak();
        copies
    }

    pub(crate) fn take_private(&mut self, reservation: &mut BlockReservation) -> BlockCopyId {
        let id = self.pool.allocate_private(reservation);
        self.note_peak();
        id
    }

    /// Publish a private block. Returns whether `key` became visible.
    pub(crate) fn cache(&mut self, id: BlockCopyId, key: CacheKey) -> bool {
        debug_assert!(
            matches!(key, CacheKey::Grouped { .. }),
            "a grouped pool caches grouped keys only"
        );
        let visible = self.pool.cache_private_key(id, key);
        self.retry_epoch += u64::from(visible);
        visible
    }

    pub(crate) fn release(&mut self, id: BlockCopyId) {
        self.pool.release(id);
        self.retry_epoch += 1;
    }

    /// Whether `id` is a private block, which no key has published yet.
    pub(crate) fn is_private(&self, id: BlockCopyId) -> bool {
        self.pool.is_private(id)
    }

    /// Release in the given order, which is the eviction order of the cached
    /// blocks among them. Pass a request's blocks tail first, as vLLM frees
    /// them.
    pub(crate) fn release_all(&mut self, ids: impl IntoIterator<Item = BlockCopyId>) {
        for id in ids {
            self.release(id);
        }
    }

    pub(crate) fn cancel(&mut self, reservation: BlockReservation) {
        self.pool.cancel(reservation);
    }

    pub(crate) fn active_blocks(&self) -> usize {
        self.pool.num_active()
    }

    pub(crate) fn inactive_cached_blocks(&self) -> usize {
        self.pool.num_inactive()
    }

    pub(crate) fn bytes(&self, blocks: usize) -> u64 {
        blocks as u64 * self.page_size_bytes
    }
}

/// The active blocks of a grouped pool, the cached blocks that no request
/// holds, and the count and the order-free fingerprint of the visible keys.
/// Byte mode caches nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GroupedPoolState {
    active_blocks: usize,
    cached_blocks: usize,
    visible_keys: (usize, u64),
}

/// Byte mode backs `required_bytes`, `fixed_request_bytes`, `allocate` and
/// `release`. Block mode does not use them. Its callers work on `block_pool`.
pub(crate) enum GroupedMode {
    Bytes,
    Blocks(GroupedBlockPool),
}

pub(crate) struct GroupedKvPool {
    groups: Vec<FpmCacheGroup>,
    capacity_bytes: u64,
    used_bytes: u64,
    active_blocks: usize,
    mode: GroupedMode,
    /// The byte-mode count of `retry_epoch`.
    retry_epoch: u64,
}

impl GroupedKvPool {
    pub(crate) fn new(groups: Vec<FpmCacheGroup>, capacity_bytes: u64) -> Self {
        Self {
            groups,
            capacity_bytes,
            used_bytes: 0,
            active_blocks: 0,
            mode: GroupedMode::Bytes,
            retry_epoch: 0,
        }
    }

    /// Block mode for prefix caching. The caller checks that every group has
    /// pages of `page_size_bytes`.
    pub(crate) fn with_prefix_cache(
        groups: Vec<FpmCacheGroup>,
        capacity_bytes: u64,
        page_size_bytes: u64,
    ) -> Self {
        debug_assert!(
            groups
                .iter()
                .all(|group| group.page_size_bytes == page_size_bytes),
            "block mode needs one page size"
        );
        let pool = VllmBlockPool::new((capacity_bytes / page_size_bytes) as usize);
        Self {
            mode: GroupedMode::Blocks(GroupedBlockPool {
                pool,
                page_size_bytes,
                retry_epoch: 0,
                #[cfg(test)]
                peak_active_blocks: std::cell::Cell::new(0),
            }),
            ..Self::new(groups, capacity_bytes)
        }
    }

    fn debug_assert_byte_mode(&self) {
        debug_assert!(
            matches!(self.mode, GroupedMode::Bytes),
            "the byte-mode API is not called in block mode"
        );
    }

    /// Pool blocks that the groups hold for a forward over
    /// `computed_before..computed_after`, in either mode.
    pub(crate) fn projected_blocks(
        &self,
        computed_before: usize,
        computed_after: usize,
    ) -> Option<usize> {
        self.project(computed_before, computed_after)
            .map(|lease| lease.ranges.iter().map(std::ops::Range::len).sum())
    }

    /// Pool blocks that the state groups hold for every running request.
    pub(crate) fn fixed_request_blocks(&self) -> usize {
        self.groups
            .iter()
            .filter(|group| group.kind == FpmCacheKind::State)
            .map(|group| group.blocks_per_request.unwrap_or(0) as usize)
            .sum()
    }

    fn project(&self, computed_before: usize, computed_after: usize) -> Option<GroupedLease> {
        if computed_after < computed_before {
            return None;
        }
        let mut projected = GroupedLease {
            ranges: Vec::with_capacity(self.groups.len()),
            bytes: 0,
            block_lease: None,
            first_pass_need: 0,
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

    pub(crate) fn required_bytes(
        &self,
        computed_before: usize,
        computed_after: usize,
    ) -> Option<u64> {
        self.debug_assert_byte_mode();
        self.project(computed_before, computed_after)
            .map(|lease| lease.bytes)
    }

    /// Bytes that the state groups hold for every running request, whatever
    /// its context length.
    pub(crate) fn fixed_request_bytes(&self) -> u64 {
        self.debug_assert_byte_mode();
        self.groups
            .iter()
            .filter(|group| group.kind == FpmCacheKind::State)
            .map(|group| {
                u64::from(group.blocks_per_request.unwrap_or(0))
                    .saturating_mul(group.page_size_bytes)
            })
            .fold(0, u64::saturating_add)
    }

    /// Reserve every group atomically. Expired blocks become reusable in this
    /// transition, but failed attempts leave both the lease and pool unchanged.
    pub(crate) fn allocate(
        &mut self,
        lease: &mut Option<GroupedLease>,
        computed_before: usize,
        computed_after: usize,
    ) -> bool {
        self.debug_assert_byte_mode();
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
        // Window blocks that leave the window free bytes.
        self.retry_epoch += u64::from(projected.bytes < previous_bytes);
        *lease = Some(projected);
        true
    }

    pub(crate) fn release(&mut self, lease: &mut Option<GroupedLease>) {
        self.debug_assert_byte_mode();
        if let Some(lease) = lease.take() {
            self.used_bytes -= lease.bytes;
            self.active_blocks -= lease.blocks();
            self.retry_epoch += 1;
        }
    }

    /// Grows whenever the pool frees capacity or publishes a key, the two
    /// changes that can let a waiting reservation succeed. A pass frees
    /// blocks without completing a request, and an activation publishes
    /// blocks that it keeps, so the active count alone misses both.
    pub(crate) fn retry_epoch(&self) -> u64 {
        match &self.mode {
            GroupedMode::Bytes => self.retry_epoch,
            GroupedMode::Blocks(blocks) => blocks.retry_epoch,
        }
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        match &self.mode {
            GroupedMode::Bytes => self.capacity_bytes,
            GroupedMode::Blocks(blocks) => blocks.bytes(blocks.capacity_blocks()),
        }
    }

    pub(crate) fn used_bytes(&self) -> u64 {
        match &self.mode {
            GroupedMode::Bytes => self.used_bytes,
            GroupedMode::Blocks(blocks) => blocks.bytes(blocks.active_blocks()),
        }
    }

    pub(crate) fn active_blocks(&self) -> usize {
        match &self.mode {
            GroupedMode::Bytes => self.active_blocks,
            GroupedMode::Blocks(blocks) => blocks.active_blocks(),
        }
    }

    pub(crate) fn state(&self) -> GroupedPoolState {
        match &self.mode {
            GroupedMode::Bytes => GroupedPoolState {
                active_blocks: self.active_blocks,
                cached_blocks: 0,
                visible_keys: (0, 0),
            },
            GroupedMode::Blocks(blocks) => GroupedPoolState {
                active_blocks: blocks.active_blocks(),
                cached_blocks: blocks.inactive_cached_blocks(),
                visible_keys: blocks.pool.grouped_keys(),
            },
        }
    }

    /// Bytes of cached blocks that no request holds. Byte mode caches nothing.
    pub(crate) fn cached_bytes(&self) -> u64 {
        match &self.mode {
            GroupedMode::Bytes => 0,
            GroupedMode::Blocks(blocks) => blocks.bytes(blocks.inactive_cached_blocks()),
        }
    }

    pub(crate) fn block_pool(&self) -> Option<&GroupedBlockPool> {
        match &self.mode {
            GroupedMode::Bytes => None,
            GroupedMode::Blocks(blocks) => Some(blocks),
        }
    }

    pub(crate) fn block_pool_mut(&mut self) -> Option<&mut GroupedBlockPool> {
        match &mut self.mode {
            GroupedMode::Bytes => None,
            GroupedMode::Blocks(blocks) => Some(blocks),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Full attention, a ring and a prefix-cacheable state, with equal pages.
    fn three_groups() -> Vec<FpmCacheGroup> {
        [
            ("full", FpmCacheKind::Attention, None, true),
            ("ring", FpmCacheKind::State, Some(1), false),
            ("kda", FpmCacheKind::State, Some(1), true),
        ]
        .into_iter()
        .map(
            |(name, kind, blocks_per_request, prefix_cacheable)| FpmCacheGroup {
                name: name.into(),
                kind,
                num_layers: 1,
                block_size_tokens: 4,
                page_size_bytes: 8,
                sliding_window: None,
                blocks_per_request,
                prefix_cacheable: Some(prefix_cacheable),
                transfer_bytes_per_block: None,
                prefill_checkpoint_blocks: None,
                prefill_checkpoint_alignment: None,
            },
        )
        .collect()
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
                    prefix_cacheable: None,
                    transfer_bytes_per_block: None,
                    prefill_checkpoint_blocks: None,
                    prefill_checkpoint_alignment: None,
                },
                FpmCacheGroup {
                    name: "window".into(),
                    kind: FpmCacheKind::Attention,
                    num_layers: 1,
                    block_size_tokens: 2,
                    page_size_bytes: 6,
                    sliding_window: Some(3),
                    blocks_per_request: None,
                    prefix_cacheable: None,
                    transfer_bytes_per_block: None,
                    prefill_checkpoint_blocks: None,
                    prefill_checkpoint_alignment: None,
                },
            ],
            42,
        );
        assert!(pool.block_pool().is_none());
        assert_eq!(pool.capacity_bytes(), 42);
        assert_eq!(pool.fixed_request_bytes(), 0);
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
        assert_eq!(pool.cached_bytes(), 0);
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
                    prefix_cacheable: None,
                    transfer_bytes_per_block: None,
                    prefill_checkpoint_blocks: None,
                    prefill_checkpoint_alignment: None,
                },
                FpmCacheGroup {
                    name: "kda".into(),
                    kind: FpmCacheKind::State,
                    num_layers: 1,
                    block_size_tokens: 4,
                    page_size_bytes: 10,
                    sliding_window: None,
                    blocks_per_request: Some(2),
                    prefix_cacheable: None,
                    transfer_bytes_per_block: None,
                    prefill_checkpoint_blocks: None,
                    prefill_checkpoint_alignment: None,
                },
            ],
            100,
        );
        assert!(pool.block_pool().is_none());
        assert_eq!(pool.capacity_bytes(), 100);
        // Only the state group counts: 2 blocks of 10 bytes.
        assert_eq!(pool.fixed_request_bytes(), 20);
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
        assert_eq!(pool.cached_bytes(), 0);
    }

    #[test]
    fn block_mode_pins_hits_and_evicts_cached_blocks_in_lru_order() {
        let (h0, h1) = (10, 11);
        let hits = [
            CacheKey::Grouped { group: 0, hash: h0 },
            CacheKey::Grouped { group: 0, hash: h1 },
        ];
        let released_hits = || {
            let mut pool = GroupedKvPool::with_prefix_cache(three_groups(), 48, 8);
            assert_eq!(pool.capacity_bytes(), 48);
            let blocks = pool.block_pool_mut().unwrap();
            assert_eq!(blocks.capacity_blocks(), 6);
            // Request A publishes two blocks and frees them tail first, as
            // vLLM frees a request's blocks.
            let mut reservation = blocks.reserve(&[], 2).unwrap();
            let a = [
                blocks.take_private(&mut reservation),
                blocks.take_private(&mut reservation),
            ];
            assert!(blocks.cache(a[0], hits[0]));
            assert!(blocks.cache(a[1], hits[1]));
            blocks.release_all(a.into_iter().rev());
            assert_eq!(blocks.inactive_cached_blocks(), 2);
            assert!(!blocks.key_hit(0, h1).unwrap().is_active);
            assert_eq!(pool.cached_bytes(), 16);
            assert_eq!(pool.used_bytes(), 0);

            // Request B hits both. Leaving the LRU, they cost capacity again.
            let blocks = pool.block_pool_mut().unwrap();
            let mut reservation = blocks.reserve(&hits, 0).unwrap();
            let b = blocks.activate(&mut reservation);
            assert_eq!(b, a);
            assert_eq!(blocks.active_blocks(), 2);
            assert_eq!(blocks.inactive_cached_blocks(), 0);
            // Another hit on the same blocks, now active, costs nothing.
            let shared = blocks.reserve(&hits, 0).unwrap();
            assert_eq!(blocks.active_blocks(), 2);
            blocks.cancel(shared);
            assert_eq!(pool.used_bytes(), 16);
            assert_eq!(pool.active_blocks(), 2);
            assert_eq!(pool.cached_bytes(), 0);
            // B frees tail first too, so h1 is the oldest cached block.
            pool.block_pool_mut()
                .unwrap()
                .release_all(b.into_iter().rev());
            pool
        };

        // Six fresh blocks take both cached ones.
        let mut pool = released_hits();
        let blocks = pool.block_pool_mut().unwrap();
        let reservation = blocks.reserve(&[], 6).unwrap();
        assert!(blocks.key_hit(0, h0).is_none());
        assert!(blocks.key_hit(0, h1).is_none());
        blocks.cancel(reservation);
        assert_eq!(blocks.active_blocks(), 0);
        assert_eq!(blocks.inactive_cached_blocks(), 0);

        // Four blocks are free, so a fifth evicts only the oldest cached
        // block, h1.
        let mut pool = released_hits();
        let blocks = pool.block_pool_mut().unwrap();
        assert!(blocks.reserve(&[], 5).is_some());
        assert!(blocks.key_hit(0, h1).is_none());
        assert!(!blocks.key_hit(0, h0).unwrap().is_active);
    }

    #[test]
    fn block_mode_keys_are_separate_per_group() {
        let hash = 7;
        let mut pool = GroupedKvPool::with_prefix_cache(three_groups(), 48, 8);
        let blocks = pool.block_pool_mut().unwrap();
        let mut reservation = blocks.reserve(&[], 2).unwrap();
        let full = blocks.take_private(&mut reservation);
        let state = blocks.take_private(&mut reservation);
        assert!(blocks.cache(full, CacheKey::Grouped { group: 0, hash }));
        // The same hash in another group is a new key, not a second copy.
        assert!(blocks.cache(state, CacheKey::Grouped { group: 2, hash }));
        assert!(blocks.key_hit(0, hash).is_some());
        assert!(blocks.key_hit(2, hash).is_some());
        assert!(blocks.key_hit(1, hash).is_none());
        blocks.release(full);
        // Four blocks are free, so a fifth evicts the one inactive block.
        assert!(blocks.reserve(&[], 5).is_some());
        assert!(blocks.key_hit(0, hash).is_none());
        assert!(blocks.key_hit(2, hash).unwrap().is_active);
    }

    #[test]
    fn block_mode_state_follows_the_set_of_visible_keys() {
        let key = |hash| CacheKey::Grouped { group: 0, hash };
        // Publish `hashes` on fresh blocks, in order, and free them.
        let publish = |pool: &mut GroupedKvPool, hashes: &[u64]| {
            let blocks = pool.block_pool_mut().unwrap();
            let mut reservation = blocks.reserve(&[], hashes.len()).unwrap();
            for &hash in hashes {
                let id = blocks.take_private(&mut reservation);
                assert!(blocks.cache(id, key(hash)));
                blocks.release(id);
            }
        };
        let mut pool = GroupedKvPool::with_prefix_cache(three_groups(), 32, 8);
        publish(&mut pool, &[10, 11]);
        let first = pool.state();
        // The order in which the keys became visible does not count.
        let mut reversed = GroupedKvPool::with_prefix_cache(three_groups(), 32, 8);
        publish(&mut reversed, &[11, 10]);
        assert_eq!(reversed.state(), first);
        // Three fresh blocks evict the older cached block, key 10. Publishing
        // key 10 again restores the same set, the same counts and the same
        // state.
        let blocks = pool.block_pool_mut().unwrap();
        let reservation = blocks.reserve(&[], 3).unwrap();
        assert!(blocks.key_hit(0, 10).is_none());
        blocks.cancel(reservation);
        assert_ne!(pool.state(), first);
        publish(&mut pool, &[10]);
        assert_eq!(pool.state(), first);
        // Another key in the place of key 11 keeps both counts but not the
        // state.
        let mut other = GroupedKvPool::with_prefix_cache(three_groups(), 32, 8);
        publish(&mut other, &[10, 12]);
        assert_ne!(other.state(), first);
    }

    #[test]
    fn block_lease_counts_the_live_copies_of_every_group() {
        let mut pool = GroupedKvPool::with_prefix_cache(three_groups(), 64, 8);
        let blocks = pool.block_pool_mut().unwrap();
        let mut reservation = blocks.reserve(&[], 8).unwrap();
        let ids: Vec<_> = (0..8)
            .map(|_| blocks.take_private(&mut reservation))
            .collect();
        let lease = GroupedLease {
            block_lease: Some(GroupedBlockLease {
                groups: vec![
                    // A window group whose block 0 has expired.
                    GroupLease::Attention {
                        copies: vec![None, Some(ids[0]), Some(ids[1])],
                        first_live: 1,
                        finalized: 0,
                    },
                    GroupLease::Ring { copy: Some(ids[2]) },
                    GroupLease::State(GroupStateLease {
                        running: Some(ids[3]),
                        hit: Some((ids[4], 1)),
                        previous: Some((ids[5], 2)),
                        copy: Some(ids[6]),
                        checkpoint: Some((ids[7], 8)),
                    }),
                ],
                allocated: true,
            }),
            ..GroupedLease::default()
        };
        assert_eq!(lease.blocks(), 8);
    }
}
