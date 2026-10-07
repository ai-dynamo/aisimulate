// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang radix-cache adapter for engine simulation.
//!
//! Tree structure, matching, locking, and eviction are delegated to the
//! `sglang-radix-tree` crate. AISimulate continues to own physical page
//! allocation, request leases, event publication, and token accounting.

use std::borrow::{Borrow, Cow};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use crate::engine::belady::BeladyOracle;
use crate::engine::common::hashing::{SequenceHash, compute_next_seq_hash};

use sglang_radix_tree::{
    CacheAction, CacheInitParams, ChildKeyType, ComponentSet, DecLockRefParams, FULL, InsertParams,
    KeyNamespaceRef, MatchPrefixParams, PageValue, UnifiedTreeCore,
};

use crate::engine::common::hashing::LocalBlockHash;
#[cfg(test)]
use crate::engine::common::hashing::compute_block_hash_for_seq;

pub use sglang_radix_tree::NodeId;

/// Physical page identifier in the simulated SGLang KV pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KvPageId(usize);

impl KvPageId {
    pub(crate) fn index(self) -> usize {
        self.0
    }

    #[cfg(test)]
    pub(crate) fn from_token_index(index: usize, page_size: usize) -> Self {
        Self(index / page_size)
    }

    #[cfg(test)]
    pub(crate) fn first_token_index(self, page_size: usize) -> usize {
        self.0 * page_size
    }
}

/// Manages free / allocated pages for the simulated SGLang KV cache.
///
/// SGLang's paged allocator owns and frees whole pages in production.
#[derive(Clone)]
pub struct PagePool {
    next_fresh: usize,
    free: Vec<KvPageId>,
    total_pages: usize,
    page_size: usize,
}

impl PagePool {
    pub fn new(total_tokens: usize, page_size: usize) -> Self {
        assert!(page_size >= 1, "page_size must be >= 1");
        Self {
            next_fresh: 0,
            free: Vec::new(),
            total_pages: total_tokens / page_size,
            page_size,
        }
    }

    pub fn allocate_pages(&mut self, count: usize) -> Option<Vec<KvPageId>> {
        if self.available_pages() < count {
            return None;
        }

        let recycled = count.min(self.free.len());
        let fresh = count - recycled;
        let mut pages = Vec::with_capacity(count);
        pages.extend(self.free.drain(self.free.len() - recycled..));
        pages.extend((self.next_fresh..self.next_fresh + fresh).map(KvPageId));
        self.next_fresh += fresh;
        Some(pages)
    }

    #[cfg(test)]
    pub fn allocate(&mut self, token_count: usize) -> Option<Vec<usize>> {
        let mut indices = Vec::new();
        self.allocate_indices_into(token_count, &mut indices)
            .then_some(indices)
    }

    /// Append flattened indices for `new_tokens`, allocating whole pages only
    /// when the request's current final page has no remaining slots.
    #[cfg(test)]
    pub fn allocate_indices_into(&mut self, new_tokens: usize, indices: &mut Vec<usize>) -> bool {
        if new_tokens == 0 {
            return true;
        }

        let available_in_last_page = indices
            .last()
            .map_or(0, |last| self.page_size - 1 - (last % self.page_size));
        let tokens_requiring_pages = new_tokens.saturating_sub(available_in_last_page);
        let required_pages = tokens_requiring_pages.div_ceil(self.page_size);
        if self.available_pages() < required_pages {
            return false;
        }

        indices.reserve(new_tokens);
        let from_existing = new_tokens.min(available_in_last_page);
        if from_existing > 0 {
            let start = indices.last().copied().expect("last page must exist") + 1;
            indices.extend(start..start + from_existing);
        }

        let remaining = new_tokens - from_existing;
        let Some(pages) = self.allocate_pages(required_pages) else {
            return false;
        };
        for (page_idx, page) in pages.into_iter().enumerate() {
            let take = remaining
                .saturating_sub(page_idx * self.page_size)
                .min(self.page_size);
            let start = page.first_token_index(self.page_size);
            indices.extend(start..start + take);
        }
        true
    }

    pub fn free_pages(&mut self, pages: &[KvPageId]) {
        self.free.extend_from_slice(pages);
    }

    /// Free every distinct page represented by a contiguous request-index list.
    #[cfg(test)]
    pub fn free_indices(&mut self, indices: &[usize]) -> Vec<KvPageId> {
        let mut pages = Vec::with_capacity(indices.len().div_ceil(self.page_size));
        for &index in indices {
            let page = KvPageId::from_token_index(index, self.page_size);
            if pages.last().copied() != Some(page) {
                pages.push(page);
            }
        }
        self.free_pages(&pages);
        pages
    }

    #[cfg(test)]
    pub fn free(&mut self, indices: &[usize]) {
        self.free_indices(indices);
    }

    pub fn available_pages(&self) -> usize {
        self.free.len() + self.total_pages - self.next_fresh
    }

    pub fn available(&self) -> usize {
        self.available_pages() * self.page_size
    }

    pub fn total(&self) -> usize {
        self.total_pages * self.page_size
    }
}

pub(crate) struct InsertPageResult {
    pub(crate) last_node: NodeId,
    pub(crate) canonical_suffix: Vec<KvPageId>,
    pub(crate) unretained_pages: Vec<KvPageId>,
}

/// Shared immutable keys keep admission checkpoints proportional to node metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct PageKey(Arc<[LocalBlockHash]>);

impl From<Vec<LocalBlockHash>> for PageKey {
    fn from(value: Vec<LocalBlockHash>) -> Self {
        Self(value.into())
    }
}
impl AsRef<[LocalBlockHash]> for PageKey {
    fn as_ref(&self) -> &[LocalBlockHash] {
        &self.0
    }
}
impl Borrow<[LocalBlockHash]> for PageKey {
    fn borrow(&self) -> &[LocalBlockHash] {
        &self.0
    }
}
impl ChildKeyType for PageKey {
    type Atom = LocalBlockHash;
    const IS_BIGRAM: bool = false;
    fn key_from(ids: Cow<'_, Vec<i64>>) -> Cow<'_, Self> {
        Cow::Owned(
            ids.iter()
                .map(|&value| LocalBlockHash(value as u64))
                .collect::<Vec<_>>()
                .into(),
        )
    }
    fn hash_words(atom: &LocalBlockHash) -> impl Iterator<Item = u32> {
        [atom.0 as u32, (atom.0 >> 32) as u32].into_iter()
    }
    fn raw_token_ids(atoms: &[LocalBlockHash]) -> Cow<'_, [i64]> {
        Cow::Owned(atoms.iter().map(|atom| atom.0 as i64).collect())
    }
}

impl From<KvPageId> for i64 {
    fn from(page: KvPageId) -> Self {
        i64::try_from(page.0).expect("page ID fits in i64")
    }
}

type SglangTree = UnifiedTreeCore<PageKey, PageValue<KvPageId>>;

#[derive(Clone)]
struct BeladyPages {
    oracle: BeladyOracle,
    page_hashes: Vec<Option<SequenceHash>>,
}

/// Thin adapter from AISimulate's page-native cache contract to SGLang's tree.
pub struct RadixCache {
    tree: SglangTree,
    belady: Option<BeladyPages>,
    pub page_pool: PagePool,
    page_size: usize,
}

impl RadixCache {
    pub(crate) fn admission_checkpoint(&self) -> Self {
        Self {
            tree: self.tree.snapshot_full_device(),
            belady: self.belady.clone(),
            page_pool: self.page_pool.clone(),
            page_size: self.page_size,
        }
    }

    pub(crate) fn set_belady_oracle(&mut self, oracle: BeladyOracle) {
        assert_eq!(
            self.tree.evictable_size() + self.tree.protected_size(),
            0,
            "attach Belady before populating the cache"
        );
        self.belady = Some(BeladyPages {
            oracle,
            page_hashes: vec![None; self.page_pool.total_pages],
        });
    }

    fn record_belady_pages(
        &mut self,
        prefix_node: NodeId,
        keys: &[LocalBlockHash],
        pages: &[KvPageId],
    ) {
        let Some(belady) = self.belady.as_mut() else {
            return;
        };
        let prefix = self
            .tree
            .get_component_device_value(prefix_node, FULL)
            .expect("live prefix");
        let mut previous = prefix
            .as_ref()
            .and_then(|values| values.as_slice().last())
            .map(|page| belady.page_hashes[page.index()].expect("cached page has prefix identity"));
        for (&key, &page) in keys.iter().zip(pages) {
            let hash = previous.map_or(key.0, |parent| compute_next_seq_hash(parent, key));
            belady.page_hashes[page.index()] = Some(hash);
            previous = Some(hash);
        }
    }

    fn evict_belady(&mut self, target_pages: usize) -> Vec<KvPageId> {
        let belady = self.belady.as_ref().expect("Belady configured");
        let priority =
            |candidate: sglang_radix_tree::FullDeviceEvictionCandidate<'_, PageValue<KvPageId>>| {
                let page = candidate.value.as_slice().last().expect("nonempty leaf");
                let hash =
                    belady.page_hashes[page.index()].expect("cached page has prefix identity");
                Reverse((
                    Reverse(belady.oracle.next_use(hash)),
                    candidate.last_access_counter,
                    candidate.node_id,
                ))
            };
        // Start from the core's current unlocked leaves. Global forecast changes
        // are observed at each eviction; they never create or unlock local KV.
        let mut candidates: BinaryHeap<_> = self
            .tree
            .full_device_eviction_candidates()
            .map(priority)
            .collect();
        let mut pages = Vec::with_capacity(target_pages.min(self.tree.evictable_size()));
        while pages.len() < target_pages {
            let Some(Reverse((_, _, node_id))) = candidates.pop() else {
                break;
            };
            let (parent, step) = self
                .tree
                .evict_full_device_suffix(node_id, 1)
                .expect("live eviction candidate");
            for chunk in step.device_frees.get(&FULL).into_iter().flatten() {
                pages.extend_from_slice(chunk.as_slice());
            }
            if let Some(parent) = parent.and_then(|id| self.tree.full_device_eviction_candidate(id))
            {
                candidates.push(priority(parent));
            }
        }
        let belady = self.belady.as_mut().expect("Belady configured");
        for page in &pages {
            belady.page_hashes[page.index()] = None;
        }
        pages
    }
    pub fn new(total_tokens: usize, page_size: usize) -> Self {
        assert!(page_size >= 1, "page_size must be >= 1");
        let tree = SglangTree::new(
            CacheInitParams {
                // A request key already contains one hash per physical KV page.
                page_size: 1,
                eviction_policy: "lru".to_string(),
                ..Default::default()
            },
            vec![FULL],
        );
        Self {
            tree,
            belady: None,
            page_pool: PagePool::new(total_tokens, page_size),
            page_size,
        }
    }

    pub fn root(&self) -> NodeId {
        self.tree.root_node_handle(None)
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    #[cfg(test)]
    pub fn num_nodes(&self) -> usize {
        self.tree.inspect_get_all_node_ids().len()
    }

    #[cfg(test)]
    pub(crate) fn node(&self, id: NodeId) -> TreeNodeSnapshot {
        TreeNodeSnapshot {
            parent: self.tree.inspect_get_parent_node_id(id).unwrap(),
            children: self
                .tree
                .inspect_get_child_node_ids(id)
                .unwrap()
                .into_iter()
                .map(|child| {
                    let first = self.tree.inspect_get_node_token_ids(child).unwrap()[0];
                    (LocalBlockHash(first as u64), child)
                })
                .collect(),
            lock_ref: self
                .tree
                .inspect_get_component_device_lock_ref(id, FULL)
                .unwrap(),
            last_access_counter: self.tree.inspect_get_node_access_counter(id).unwrap(),
            key: self
                .tree
                .inspect_get_node_token_ids(id)
                .unwrap()
                .into_iter()
                .map(|value| LocalBlockHash(value as u64))
                .collect(),
            value: self
                .tree
                .get_component_device_value(id, FULL)
                .unwrap()
                .map_or_else(Vec::new, |value| value.as_slice().to_vec()),
        }
    }

    #[cfg(test)]
    pub(crate) fn page_hashes(&self, tokens: &[u32]) -> Vec<LocalBlockHash> {
        compute_block_hash_for_seq(tokens, self.page_size)
    }

    #[cfg(test)]
    pub fn match_prefix(&mut self, key: &[u32]) -> (usize, NodeId) {
        let page_keys = self.page_hashes(key);
        self.match_prefix_hashes(&page_keys)
    }

    #[cfg(test)]
    pub(crate) fn match_prefix_hashes(&mut self, page_keys: &[LocalBlockHash]) -> (usize, NodeId) {
        let key: PageKey = page_keys.to_vec().into();
        let result = self.tree.match_prefix(&MatchPrefixParams {
            key: &key,
            namespace: KeyNamespaceRef::default(),
        });
        assert!(
            result.cache_actions.is_empty(),
            "Full-only device match unexpectedly produced cache actions"
        );
        (
            result.device_indices.as_slice().len() * self.page_size,
            result.last_device_node_id,
        )
    }

    /// Match and protect a prefix using page hashes already owned by the request.
    pub(crate) fn match_prefix_hashes_and_lock(
        &mut self,
        page_keys: &[LocalBlockHash],
    ) -> (usize, NodeId) {
        let key: PageKey = page_keys.to_vec().into();
        let result = self.tree.match_prefix(&MatchPrefixParams {
            key: &key,
            namespace: KeyNamespaceRef::default(),
        });
        assert!(
            result.cache_actions.is_empty(),
            "Full-only device match unexpectedly produced cache actions"
        );
        let last_node = result.last_device_node_id;
        let matched_tokens = result.device_indices.as_slice().len() * self.page_size;
        self.tree
            .inc_lock_ref(last_node, ComponentSet::EMPTY)
            .expect("live prefix");
        (matched_tokens, last_node)
    }

    #[cfg(test)]
    pub fn prefix_match_len(&self, key: &[u32]) -> usize {
        let page_keys = self.page_hashes(key);
        self.prefix_match_hashes_len(&page_keys)
    }

    /// Read-only prefix match using page hashes already owned by the request.
    pub(crate) fn prefix_match_hashes_len(&self, page_keys: &[LocalBlockHash]) -> usize {
        self.tree
            .full_kv_prefix_len(page_keys, KeyNamespaceRef::default())
            * self.page_size
    }

    #[cfg(test)]
    pub fn insert(&mut self, key: &[u32], value: &[usize]) -> NodeId {
        let aligned_len = key.len() / self.page_size * self.page_size;
        assert!(
            value.len() >= aligned_len,
            "not enough token indices: need {aligned_len}, got {}",
            value.len()
        );
        let page_keys = self.page_hashes(&key[..aligned_len]);
        let pages = self.page_ids(&value[..aligned_len], page_keys.len());
        self.insert_page_hashes_from_node(self.root(), 0, &page_keys, &pages, false)
            .last_node
    }

    /// Insert page identities already materialized by the request lease.
    pub(crate) fn insert_page_hashes_from_node(
        &mut self,
        prefix_node: NodeId,
        prefix_len: usize,
        page_keys: &[LocalBlockHash],
        pages: &[KvPageId],
        chunked: bool,
    ) -> InsertPageResult {
        assert_eq!(
            prefix_len % self.page_size,
            0,
            "prefix length must be page-aligned"
        );
        assert!(
            pages.len() >= page_keys.len(),
            "not enough KV pages: need {}, got {}",
            page_keys.len(),
            pages.len()
        );
        let prefix_pages = prefix_len / self.page_size;
        assert!(
            prefix_pages <= page_keys.len(),
            "prefix pages {prefix_pages} exceed hashed pages {}",
            page_keys.len()
        );

        self.record_belady_pages(
            prefix_node,
            &page_keys[prefix_pages..],
            &pages[prefix_pages..page_keys.len()],
        );
        let key: PageKey = page_keys.to_vec().into();
        let suffix_value = PageValue::from_vec(pages[prefix_pages..page_keys.len()].to_vec());
        let result = self.tree.insert_suffix_from_node(
            prefix_node,
            prefix_pages,
            &InsertParams {
                rotation_base: None,
                session_id: None,
                swa_branching_seqlen: None,
                key: &key,
                namespace: KeyNamespaceRef::default(),
                value: suffix_value,
                prev_prefix_len: prefix_pages,
                swa_evicted_seqlen: 0,
                mamba_value: None,
                chunked,
                priority: 0,
                track_adopted_ranges: false,
            },
        );
        let last_node = result
            .last_device_node_id
            .expect("non-empty Full insert must end on a device node");
        let canonical_suffix = self
            .tree
            .collect_full_device_indices(last_node, prefix_node)
            .expect("live continuation path")
            .as_slice()
            .to_vec();
        assert_eq!(
            canonical_suffix.len(),
            page_keys.len() - prefix_pages,
            "SGLang core returned an incomplete canonical suffix"
        );

        InsertPageResult {
            last_node,
            canonical_suffix,
            unretained_pages: collect_unretained_pages(result.cache_actions),
        }
    }

    pub(crate) fn collect_path_pages(&self, last_node: NodeId) -> Vec<KvPageId> {
        self.tree
            .collect_full_device_indices(last_node, self.root())
            .expect("live prefix path")
            .as_slice()
            .to_vec()
    }

    pub(crate) fn collect_path_pages_through(
        &self,
        last_node: NodeId,
        prefix_len: usize,
    ) -> Vec<KvPageId> {
        assert_eq!(
            prefix_len % self.page_size,
            0,
            "matched SGLang prefix must be page-aligned"
        );
        let expected_pages = prefix_len / self.page_size;
        let mut pages = self.collect_path_pages(last_node);
        assert!(
            pages.len() >= expected_pages,
            "SGLang radix path returned {} pages for a {expected_pages}-page prefix",
            pages.len()
        );
        pages.truncate(expected_pages);
        pages
    }

    pub fn inc_lock_ref(&mut self, node_id: NodeId) {
        self.tree
            .inc_lock_ref(node_id, ComponentSet::EMPTY)
            .expect("live lock anchor");
    }

    pub fn dec_lock_ref(&mut self, node_id: NodeId) {
        // Full-only trees have no component UUIDs; the anchor identifies the receipt.
        self.tree
            .dec_lock_ref(
                node_id,
                &DecLockRefParams {
                    node_id: Some(node_id),
                    ..Default::default()
                },
                false,
            )
            .expect("live lock anchor");
    }

    /// Evict cache pages in SGLang LRU order.
    ///
    /// SGLang evicts complete compressed leaves, so this can release more than
    /// the requested number of tokens.
    pub fn evict(&mut self, num_tokens: usize) -> (usize, Vec<KvPageId>) {
        let target_pages = num_tokens.div_ceil(self.page_size);
        if target_pages == 0 {
            return (0, Vec::new());
        }
        if self.belady.is_some() {
            let pages = self.evict_belady(target_pages);
            self.page_pool.free_pages(&pages);
            return (pages.len() * self.page_size, pages);
        }

        let mut tracker = HashMap::from([(FULL, 0)]);
        let mut evicted_pages = Vec::with_capacity(target_pages.min(self.tree.evictable_size()));
        self.tree.evict_device_start(FULL, target_pages);
        loop {
            let (candidate, step) = self.tree.evict_device_next_node(FULL, &tracker);
            absorb_eviction_step(&mut tracker, &mut evicted_pages, step);
            let Some(candidate) = candidate else {
                break;
            };

            let (backup, step) = self
                .tree
                .evict_device_leaf(candidate, false)
                .expect("live eviction candidate");
            assert!(
                backup.is_none(),
                "write-through eviction requested a backup"
            );
            absorb_eviction_step(&mut tracker, &mut evicted_pages, step);
            if tracker.get(&FULL).copied().unwrap_or(0) >= target_pages {
                break;
            }
        }
        self.tree.evict_device_end(FULL);

        self.page_pool.free_pages(&evicted_pages);
        (evicted_pages.len() * self.page_size, evicted_pages)
    }

    pub fn evictable_size(&self) -> usize {
        self.tree.evictable_size() * self.page_size
    }

    pub fn protected_size(&self) -> usize {
        self.tree.protected_size() * self.page_size
    }

    pub fn available_tokens(&self) -> usize {
        self.page_pool.available()
    }

    pub fn total_tokens(&self) -> usize {
        self.page_pool.total()
    }

    #[cfg(test)]
    fn page_ids(&self, indices: &[usize], page_count: usize) -> Vec<KvPageId> {
        assert!(
            indices.len() >= page_count * self.page_size,
            "not enough token indices for {page_count} complete pages"
        );
        indices
            .chunks_exact(self.page_size)
            .take(page_count)
            .map(|chunk| {
                let page = KvPageId::from_token_index(chunk[0], self.page_size);
                let start = page.first_token_index(self.page_size);
                assert!(
                    chunk.iter().copied().eq(start..start + self.page_size),
                    "SGLang cached pages must contain contiguous page-aligned indices"
                );
                page
            })
            .collect()
    }
}

fn collect_unretained_pages(actions: Vec<CacheAction<PageValue<KvPageId>>>) -> Vec<KvPageId> {
    let mut pages = Vec::new();
    for action in actions {
        match action {
            CacheAction::FreeDeviceKV(chunks) | CacheAction::FreeDeviceKVFullOnly(chunks) => {
                for chunk in chunks {
                    pages.extend_from_slice(chunk.as_slice());
                }
            }
            CacheAction::BackupKV(_)
            | CacheAction::ReplaceWriteThroughOnNodeSplit { .. }
            | CacheAction::MambaEvictExcessPathStates { .. }
            | CacheAction::FreeComponentDeviceSlot { .. }
            | CacheAction::FreeComponentHostSlot { .. }
            | CacheAction::RebuildFullToSwaMapping { .. }
            | CacheAction::RecoverSwaWithLockedFull { .. }
            | CacheAction::SwaRebuild { .. } => {
                panic!("Full-only simulation insert produced an unsupported cache action")
            }
        }
    }
    pages
}

fn absorb_eviction_step(
    tracker: &mut HashMap<sglang_radix_tree::ComponentType, usize>,
    evicted_pages: &mut Vec<KvPageId>,
    step: sglang_radix_tree::EvictionStepResult<PageValue<KvPageId>>,
) {
    for (component, count) in step.tracker {
        *tracker.entry(component).or_default() += count;
    }
    for (component, chunks) in step.device_frees {
        assert_eq!(component, FULL, "Full-only tree evicted an auxiliary pool");
        for chunk in chunks {
            evicted_pages.extend_from_slice(chunk.as_slice());
        }
    }
    assert!(
        step.host_frees.is_empty(),
        "device-only simulation tree unexpectedly freed host pages"
    );
}

#[cfg(test)]
pub(crate) struct TreeNodeSnapshot {
    pub(crate) parent: Option<NodeId>,
    pub(crate) children: rustc_hash::FxHashMap<LocalBlockHash, NodeId>,
    pub(crate) lock_ref: u32,
    pub(crate) last_access_counter: i64,
    pub(crate) key: Vec<LocalBlockHash>,
    pub(crate) value: Vec<KvPageId>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::common::hashing::compute_seq_hash_for_block;
    use uuid::Uuid;

    fn oracle_for_prompts(prompts: &[&[u32]]) -> BeladyOracle {
        BeladyOracle::new(
            prompts
                .iter()
                .enumerate()
                .map(|(index, tokens)| {
                    (
                        Uuid::from_u128(index as u128 + 1),
                        compute_seq_hash_for_block(&compute_block_hash_for_seq(tokens, 1)),
                    )
                })
                .collect(),
        )
        .unwrap()
    }

    #[rstest::rstest]
    fn belady_reranks_compressed_tail_before_evicting_its_prefix(
        #[values(false, true)] shared: bool,
    ) {
        let mut cache = RadixCache::new(3, 1);
        cache.set_belady_oracle(oracle_for_prompts(&[&[1], &[3], &[1, 2]]));
        let ab = cache.page_pool.allocate(2).unwrap();
        cache.insert(&[1, 2], &ab);
        let c = cache.page_pool.allocate(1).unwrap();
        cache.insert(&[3], &c);
        let snapshot = shared.then(|| cache.admission_checkpoint());

        let (tokens, pages) = cache.evict(2);
        assert_eq!((tokens, pages), (2, vec![KvPageId(1), KvPageId(2)]));
        assert_eq!(cache.prefix_match_len(&[1, 2]), 1);
        assert_eq!(cache.prefix_match_len(&[3]), 0);
        assert_eq!(cache.available_tokens(), 2);
        if let Some(mut saved) = snapshot {
            assert_eq!(saved.prefix_match_len(&[1, 2]), 2);
            assert_eq!(saved.prefix_match_len(&[3]), 1);
            assert_eq!(saved.available_tokens(), 0);
            assert_eq!(saved.evict(2).1, vec![KvPageId(1), KvPageId(2)]);
        }
    }

    #[test]
    fn belady_observes_global_retirement_before_local_victim_selection() {
        let oracle = oracle_for_prompts(&[&[1], &[2]]);
        let mut cache = RadixCache::new(2, 1);
        cache.set_belady_oracle(oracle.clone());
        let a = cache.page_pool.allocate(1).unwrap();
        cache.insert(&[1], &a);
        let b = cache.page_pool.allocate(1).unwrap();
        cache.insert(&[2], &b);

        // Another worker commits the first request; this worker must refresh its hidden key.
        oracle.retire_requests([Uuid::from_u128(1)]);
        assert_eq!(cache.evict(1).1, vec![KvPageId(0)]);
        assert_eq!(cache.prefix_match_len(&[2]), 1);
    }

    #[test]
    fn belady_preserves_locked_prefixes_and_handles_branching_and_extension() {
        let mut cache = RadixCache::new(8, 1);
        cache.set_belady_oracle(oracle_for_prompts(&[&[1, 2], &[1, 2, 4], &[9], &[1, 2, 3]]));
        let mut pages = cache.page_pool.allocate(2).unwrap();
        let parent = cache.insert(&[1, 2], &pages);
        cache.inc_lock_ref(parent);
        pages.extend(cache.page_pool.allocate(1).unwrap());
        cache.insert_page_hashes_from_node(
            parent,
            2,
            &cache.page_hashes(&[1, 2, 3]),
            &pages.iter().copied().map(KvPageId).collect::<Vec<_>>(),
            true,
        );
        cache.dec_lock_ref(parent);
        let mut branch = pages[..2].to_vec();
        branch.extend(cache.page_pool.allocate(1).unwrap());
        cache.insert(&[1, 2, 4], &branch);
        let other = cache.page_pool.allocate(1).unwrap();
        cache.insert(&[9], &other);

        assert_eq!(cache.evict(2).1, vec![KvPageId(2), KvPageId(4)]);
        assert_eq!(cache.prefix_match_len(&[1, 2, 4]), 3);
        let (_, locked) = cache.match_prefix(&[1, 2]);
        cache.inc_lock_ref(locked);
        assert_eq!(cache.evict(8).1, vec![KvPageId(3)]);
        assert_eq!(cache.protected_size(), 2);
        cache.dec_lock_ref(locked);
        assert_eq!(cache.evict(2).1, vec![KvPageId(1), KvPageId(0)]);
        assert_eq!(cache.available_tokens(), 8);
    }

    #[test]
    fn belady_recycled_page_has_the_new_prefix_identity() {
        let oracle = oracle_for_prompts(&[&[2], &[3]]);
        let mut cache = RadixCache::new(2, 1);
        cache.set_belady_oracle(oracle);
        let first = cache.page_pool.allocate(1).unwrap();
        cache.insert(&[1], &first);
        assert_eq!(cache.evict(1).1, vec![KvPageId(0)]);
        let reused = cache.page_pool.allocate(1).unwrap();
        assert_eq!(reused, first);
        cache.insert(&[2], &reused);
        let second = cache.page_pool.allocate(1).unwrap();
        cache.insert(&[3], &second);
        assert_eq!(cache.evict(1).1, vec![KvPageId(1)]);
        assert_eq!(cache.prefix_match_len(&[2]), 1);
    }

    #[test]
    fn belady_distinguishes_equal_local_pages_below_different_prefixes() {
        let mut cache = RadixCache::new(4, 1);
        cache.set_belady_oracle(oracle_for_prompts(&[&[1, 7], &[2]]));
        let first = cache.page_pool.allocate(2).unwrap();
        cache.insert(&[1, 7], &first);
        let second = cache.page_pool.allocate(2).unwrap();
        cache.insert(&[2, 7], &second);

        assert_eq!(cache.evict(2).1, vec![KvPageId(3), KvPageId(2)]);
        assert_eq!(cache.prefix_match_len(&[1, 7]), 2);
        assert_eq!(cache.prefix_match_len(&[2, 7]), 0);
    }

    #[test]
    fn page_pool_allocate_extend_and_free() {
        let mut pool = PagePool::new(12, 4);
        assert_eq!(pool.available(), 12);
        assert!(pool.allocate(usize::MAX).is_none());
        let first = pool.allocate(3).unwrap();
        assert_eq!(pool.available(), 8);
        let mut extended = first.clone();
        assert!(pool.allocate_indices_into(1, &mut extended));
        assert_eq!(extended, vec![0, 1, 2, 3]);
        let second = pool.allocate(5).unwrap();
        assert_eq!(pool.available(), 0);
        assert!(pool.allocate(1).is_none());
        pool.free(&first);
        pool.free(&second);
        assert_eq!(pool.available(), 12);
    }

    #[test]
    fn allocation_failure_is_atomic() {
        let mut pool = PagePool::new(8, 4);
        let mut destination = pool.allocate(4).unwrap();
        let _other = pool.allocate(4).unwrap();
        let available_before = pool.available();
        let destination_before = destination.clone();

        assert!(!pool.allocate_indices_into(1, &mut destination));
        assert_eq!(destination, destination_before);
        assert_eq!(pool.available(), available_before);
    }

    #[test]
    fn match_insert_and_read_only_score_use_sglang_core() {
        let mut cache = RadixCache::new(64, 4);
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &[0, 1, 2, 3, 4, 5, 6, 7]);
        cache.insert(&[1, 2, 3, 4, 9, 10, 11, 12], &[0, 1, 2, 3, 8, 9, 10, 11]);

        assert_eq!(cache.prefix_match_len(&[1, 2, 3, 4, 13, 14, 15, 16]), 4);
        let nodes_before = cache.num_nodes();
        assert_eq!(cache.prefix_match_len(&[1, 2, 3, 4, 13, 14, 15, 16]), 4);
        assert_eq!(cache.num_nodes(), nodes_before);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]).0, 8);
    }

    #[test]
    fn continuation_returns_canonical_suffix_and_duplicate_pages() {
        let mut cache = RadixCache::new(64, 4);
        let existing = cache.page_pool.allocate_pages(2).unwrap();
        let key = cache.page_hashes(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let _ = cache.insert_page_hashes_from_node(cache.root(), 0, &key, &existing, false);

        let (prefix_len, prefix_node) = cache.match_prefix_hashes_and_lock(&key[..1]);
        assert_eq!(prefix_len, 4);
        let incoming_suffix = cache.page_pool.allocate_pages(1).unwrap();
        let incoming = [existing[0], incoming_suffix[0]];
        let result =
            cache.insert_page_hashes_from_node(prefix_node, prefix_len, &key, &incoming, true);

        assert_eq!(result.canonical_suffix, vec![existing[1]]);
        assert_eq!(result.unretained_pages, incoming_suffix);
        assert_eq!(cache.collect_path_pages(result.last_node), existing);
        cache.dec_lock_ref(prefix_node);
    }

    #[test]
    fn lock_accounting_and_lru_eviction_follow_sglang() {
        let mut cache = RadixCache::new(64, 4);
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &[0, 1, 2, 3, 4, 5, 6, 7]);
        cache.insert(&[9, 10, 11, 12], &[8, 9, 10, 11]);

        let first_key = cache.page_hashes(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let (_, locked) = cache.match_prefix_hashes_and_lock(&first_key);
        assert_eq!(cache.protected_size(), 8);
        assert_eq!(cache.evictable_size(), 4);

        let (evicted, pages) = cache.evict(4);
        assert_eq!(evicted, 4);
        assert_eq!(pages, vec![KvPageId(2)]);
        assert_eq!(cache.match_prefix(&[9, 10, 11, 12]).0, 0);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]).0, 8);

        cache.dec_lock_ref(locked);
        assert_eq!(cache.protected_size(), 0);
        assert_eq!(cache.evictable_size(), 8);
    }

    #[test]
    fn query_methods_report_physical_token_capacity() {
        let cache = RadixCache::new(100, 4);
        assert_eq!(cache.available_tokens(), 100);
        assert_eq!(cache.total_tokens(), 100);
        assert_eq!(cache.root(), 0);
    }
}
