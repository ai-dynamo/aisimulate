// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM `OffloadingConnector` behavior over the framework-neutral host tier.
//!
//! Pinned vLLM 0.24 applies ordinary resident touches in exact reverse logical
//! order. Store and load completion keys, however, travel through Python sets;
//! their within-completion order is process/version dependent and is not a
//! stable framework contract. The host tier therefore gives each completion
//! cohort one recency epoch and uses logical identity only as a deterministic
//! tie-break inside that epoch.

use rustc_hash::FxHashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use uuid::Uuid;

use crate::engine::common::protocols::PrefillCost;
use crate::engine::common::sequence::RequestSequence;
use crate::engine::host_offload::{
    CompletedTransfer, HostBlockKey, HostOffloadObservation, HostOffloadObservationData,
    HostStoreBlockMapping, HostTier, HostTierConfig, LoadOutcome, Lookup, StoreOutcome, TransferId,
};
use crate::engine::kv_manager::{
    BlockRequestLease, DestinationReservation, G1Acquire, G1Manager, SourceReuseDependency,
};
use crate::engine::{HostOffloadObserver, NativeHostOffloadConfig, StoredBlocks};

struct LoadingPrefix {
    transfer_id: TransferId,
    keys: Vec<HostBlockKey>,
    reservation: DestinationReservation,
    original_g1_tokens: usize,
    loaded_prefix_blocks: usize,
    request_footprint_blocks: usize,
}

#[derive(Clone, Copy)]
struct ActivatedPrefix {
    original_g1_tokens: usize,
    loaded_prefix_blocks: usize,
    loaded_blocks: usize,
    request_footprint_blocks: usize,
}

enum LoadState {
    Loading(LoadingPrefix),
    Activated(ActivatedPrefix),
}

/// Request-local connector cursor and transient load ownership.
pub(super) struct VllmHostRequestState {
    prompt_keys: Vec<HostBlockKey>,
    next_store_block: usize,
    latest_store: Option<TransferId>,
    load: Option<LoadState>,
    first_admission_recorded: bool,
}

impl VllmHostRequestState {
    pub(super) fn new(
        sequence: &RequestSequence,
        lease: &BlockRequestLease,
        block_size: usize,
    ) -> Self {
        let prompt_blocks = sequence.num_input_tokens() / block_size;
        let prompt_keys = (0..prompt_blocks)
            .map(|index| {
                HostBlockKey::new(
                    lease
                        .sequence_hash(index)
                        .expect("complete prompt block must retain its sequence hash"),
                )
            })
            .collect();
        Self {
            prompt_keys,
            next_store_block: 0,
            latest_store: None,
            load: None,
            first_admission_recorded: false,
        }
    }

    fn activated(&self) -> Option<ActivatedPrefix> {
        match self.load {
            Some(LoadState::Activated(load)) => Some(load),
            _ => None,
        }
    }

    pub(super) fn reserved_blocks(&self) -> usize {
        match &self.load {
            Some(LoadState::Loading(load)) => load.loaded_prefix_blocks,
            Some(LoadState::Activated(load)) => load.loaded_prefix_blocks,
            None => 0,
        }
    }

    pub(super) fn unreserved_tail_blocks(&self) -> usize {
        match &self.load {
            Some(LoadState::Loading(load)) => load
                .request_footprint_blocks
                .saturating_sub(load.loaded_prefix_blocks),
            Some(LoadState::Activated(load)) => load
                .request_footprint_blocks
                .saturating_sub(load.loaded_prefix_blocks),
            None => 0,
        }
    }
}

pub(super) struct HostLookupHit {
    keys: Vec<HostBlockKey>,
    base_g1_blocks: usize,
    original_g1_tokens: usize,
    request_footprint_blocks: usize,
}

pub(super) enum HostLookup {
    Miss,
    Hit(HostLookupHit),
    Deferred,
}

pub(super) enum StartLoad {
    Queued,
    Deferred,
    Retry,
    CapacityBlocked,
}

/// Transfer completions that require request-owned G1 activation.
pub(super) struct CompletedLoad {
    pub(super) uuid: Uuid,
    pub(super) transfer_id: TransferId,
}

pub(super) struct HostTransferProgress {
    pub(super) completed_loads: Vec<CompletedLoad>,
    pub(super) completed_any: bool,
}

struct HostCacheDomainState {
    tier: HostTier,
    load_by_key: FxHashMap<HostBlockKey, TransferId>,
    transfer_owners: FxHashMap<TransferId, u32>,
    completed_by_rank: FxHashMap<u32, Vec<CompletedTransfer>>,
    completion_generation: u64,
}

/// One physical host-cache domain shared by its attention-DP ranks.
#[derive(Clone)]
pub(crate) struct VllmHostOffloadDomain {
    state: Arc<Mutex<HostCacheDomainState>>,
}

impl VllmHostOffloadDomain {
    pub(crate) fn new(
        config: &NativeHostOffloadConfig,
        block_size: usize,
        kv_bytes_per_token: usize,
    ) -> anyhow::Result<Self> {
        let block_bytes = block_size
            .checked_mul(kv_bytes_per_token)
            .ok_or_else(|| anyhow::anyhow!("native host block byte size overflow"))?;
        Ok(Self {
            state: Arc::new(Mutex::new(HostCacheDomainState {
                tier: HostTier::new(HostTierConfig {
                    capacity_blocks: config.num_host_blocks,
                    block_bytes,
                    d2h_bandwidth_gbps: config.d2h_bandwidth_gbps,
                    h2d_bandwidth_gbps: config.h2d_bandwidth_gbps,
                })?,
                load_by_key: FxHashMap::default(),
                transfer_owners: FxHashMap::default(),
                completed_by_rank: FxHashMap::default(),
                completion_generation: 0,
            })),
        })
    }

    fn lock(&self) -> MutexGuard<'_, HostCacheDomainState> {
        self.state.lock().expect("host cache domain mutex poisoned")
    }

    fn set_observer(&self, observer: Arc<dyn HostOffloadObserver>) {
        self.lock().tier.set_observer(observer);
    }

    fn has_transfer(&self, transfer_id: TransferId) -> bool {
        self.lock().tier.has_transfer(transfer_id)
    }

    fn touch(&self, key: HostBlockKey) {
        self.lock().tier.touch(key);
    }

    fn lookup(&self, key: HostBlockKey) -> Lookup {
        self.lock().tier.lookup(key)
    }

    fn is_loading(&self, key: HostBlockKey) -> bool {
        self.lock().load_by_key.contains_key(&key)
    }

    fn prepare_store(
        &self,
        dp_rank: u32,
        request_id: Uuid,
        blocks: &[HostBlockKey],
        now_ms: f64,
    ) -> StoreOutcome {
        let mut state = self.lock();
        let outcome = state.tier.prepare_store(request_id, blocks, now_ms);
        if let StoreOutcome::Prepared { transfer_id, .. } = &outcome {
            assert!(
                state
                    .transfer_owners
                    .insert(*transfer_id, dp_rank)
                    .is_none()
            );
        }
        outcome
    }

    fn submit_prepared_stores(&self, now_ms: f64) -> usize {
        self.lock().tier.submit_prepared_stores(now_ms)
    }

    fn schedule_load(
        &self,
        dp_rank: u32,
        request_id: Uuid,
        blocks: &[HostBlockKey],
        now_ms: f64,
        not_before_ms: f64,
    ) -> LoadOutcome {
        let mut state = self.lock();
        if blocks.iter().any(|key| state.load_by_key.contains_key(key)) {
            return LoadOutcome::Miss;
        }
        let outcome = state
            .tier
            .schedule_load(request_id, blocks, now_ms, not_before_ms);
        if let LoadOutcome::Queued(transfer_id) = outcome {
            assert!(state.transfer_owners.insert(transfer_id, dp_rank).is_none());
            for key in blocks {
                assert!(state.load_by_key.insert(*key, transfer_id).is_none());
            }
        }
        outcome
    }

    fn tick_for_rank(&self, dp_rank: u32, now_ms: f64) -> (Vec<CompletedTransfer>, u64) {
        let mut state = self.lock();
        let completed = state.tier.tick(now_ms);
        if !completed.is_empty() {
            state.completion_generation = state
                .completion_generation
                .checked_add(1)
                .expect("host cache-domain completion generation overflow");
        }
        for transfer in completed {
            let transfer_id = match &transfer {
                CompletedTransfer::Store { transfer_id, .. }
                | CompletedTransfer::Load { transfer_id, .. } => *transfer_id,
            };
            let owner = state
                .transfer_owners
                .remove(&transfer_id)
                .expect("completed host transfer lost its DP-rank owner");
            if let CompletedTransfer::Load { blocks, .. } = &transfer {
                for key in blocks {
                    assert_eq!(state.load_by_key.remove(key), Some(transfer_id));
                }
            }
            state
                .completed_by_rank
                .entry(owner)
                .or_default()
                .push(transfer);
        }
        let generation = state.completion_generation;
        let completed = state.completed_by_rank.remove(&dp_rank).unwrap_or_default();
        (completed, generation)
    }

    fn cancel_load(
        &self,
        dp_rank: u32,
        transfer_id: TransferId,
        keys: &[HostBlockKey],
        mutation_now_ms: f64,
        observed_at_ms: f64,
    ) -> bool {
        let mut state = self.lock();
        if !state
            .tier
            .cancel_load(transfer_id, mutation_now_ms, observed_at_ms)
        {
            return false;
        }
        assert_eq!(state.transfer_owners.remove(&transfer_id), Some(dp_rank));
        for key in keys {
            assert_eq!(state.load_by_key.remove(key), Some(transfer_id));
        }
        true
    }

    fn transfer_deadline(&self, transfer_id: TransferId) -> Option<f64> {
        self.lock().tier.transfer_deadline(transfer_id)
    }

    fn next_deadline_for_rank(&self, dp_rank: u32, seen_completion_generation: u64) -> Option<f64> {
        let state = self.lock();
        if state.completion_generation != seen_completion_generation
            || state
                .completed_by_rank
                .get(&dp_rank)
                .is_some_and(|completed| !completed.is_empty())
        {
            return Some(state.tier.current_time_ms());
        }
        state.tier.next_deadline()
    }

    fn current_time_ms(&self) -> f64 {
        self.lock().tier.current_time_ms()
    }

    fn has_pending_work(&self) -> bool {
        let state = self.lock();
        state.tier.has_pending_work()
            || state
                .completed_by_rank
                .values()
                .any(|completed| !completed.is_empty())
    }

    #[cfg(test)]
    fn resident_blocks(&self) -> usize {
        self.lock().tier.resident_blocks()
    }

    #[cfg(test)]
    fn is_resident(&self, key: HostBlockKey) -> bool {
        self.lock().tier.is_resident(key)
    }
}

pub(super) struct VllmHostOffloadAdapter {
    domain: VllmHostOffloadDomain,
    dp_rank: u32,
    seen_completion_generation: u64,
    compute_not_before_ms: f64,
    pending_stores: FxHashMap<TransferId, Vec<StoredBlocks>>,
    /// Present only for detailed artifact capture. Ordinary runs neither retain
    /// request-local mapping state nor allocate mapping payloads.
    observer: Option<Arc<dyn HostOffloadObserver>>,
}

impl VllmHostOffloadAdapter {
    pub(super) fn set_observer(&mut self, observer: Arc<dyn HostOffloadObserver>) {
        self.domain.set_observer(Arc::clone(&observer));
        self.observer = Some(observer);
    }

    pub(super) fn new(
        config: &NativeHostOffloadConfig,
        block_size: usize,
        kv_bytes_per_token: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self::in_domain(
            VllmHostOffloadDomain::new(config, block_size, kv_bytes_per_token)?,
            0,
        ))
    }

    pub(super) fn in_domain(domain: VllmHostOffloadDomain, dp_rank: u32) -> Self {
        Self {
            domain,
            dp_rank,
            seen_completion_generation: 0,
            compute_not_before_ms: 0.0,
            pending_stores: FxHashMap::default(),
            observer: None,
        }
    }

    /// Query G2 after the scheduler's one authoritative G1 prefix lookup.
    pub(super) fn lookup(
        &mut self,
        request: &mut VllmHostRequestState,
        sequence: &RequestSequence,
        g1_cost: &PrefillCost,
        block_size: usize,
    ) -> HostLookup {
        if request.activated().is_some() {
            return HostLookup::Miss;
        }
        if matches!(request.load, Some(LoadState::Loading(_))) {
            return HostLookup::Deferred;
        }
        if let Some(transfer_id) = request.latest_store {
            if self.domain.has_transfer(transfer_id) {
                return HostLookup::Deferred;
            }
            request.latest_store = None;
        }

        // Pinned vLLM 0.24 passes this ordered logical list to
        // LRUCachePolicy.touch(), which explicitly iterates it in reverse.
        for key in request.prompt_keys.iter().rev().copied() {
            self.domain.touch(key);
        }

        debug_assert_eq!(g1_cost.cached_tokens % block_size, 0);
        let base_g1_blocks = g1_cost.cached_tokens / block_size;
        if base_g1_blocks >= request.prompt_keys.len() {
            return HostLookup::Miss;
        }

        let mut matched = 0usize;
        let mut deferred = false;
        for key in request.prompt_keys[base_g1_blocks..].iter().copied() {
            match self.domain.lookup(key) {
                Lookup::Hit => {
                    matched += 1;
                    deferred |= self.domain.is_loading(key);
                }
                Lookup::Pending { .. } => {
                    matched += 1;
                    deferred = true;
                }
                Lookup::Miss => break,
            }
        }
        if matched == 0 {
            return HostLookup::Miss;
        }
        let keys = request.prompt_keys[base_g1_blocks..base_g1_blocks + matched].to_vec();
        if deferred || keys.iter().any(|key| self.domain.is_loading(*key)) {
            return HostLookup::Deferred;
        }
        HostLookup::Hit(HostLookupHit {
            keys,
            base_g1_blocks,
            original_g1_tokens: g1_cost.cached_tokens,
            request_footprint_blocks: sequence.current_known_blocks(),
        })
    }

    /// Reserve real G1 capacity before queuing the H2D.
    pub(super) fn start_load(
        &mut self,
        uuid: Uuid,
        request: &mut VllmHostRequestState,
        lease: &BlockRequestLease,
        hit: HostLookupHit,
        kv_manager: &mut G1Manager,
        now_ms: f64,
    ) -> StartLoad {
        if hit.keys.iter().any(|key| self.domain.is_loading(*key)) {
            return StartLoad::Deferred;
        }
        let hashes = hit
            .keys
            .iter()
            .map(|key| key.sequence_hash())
            .collect::<Vec<_>>();
        let reservation = match kv_manager.reserve_native_host_destination(
            uuid,
            lease,
            hit.base_g1_blocks,
            &hashes,
        ) {
            G1Acquire::Ready(reservation) => reservation,
            G1Acquire::CapacityExhausted => return StartLoad::CapacityBlocked,
        };

        let dependencies = kv_manager.native_destination_pending_dependencies(&reservation);
        let mut not_before_ms = now_ms;
        for dependency in dependencies {
            assert!(kv_manager.is_native_source_dependency_pending(dependency));
            let deadline = self
                .domain
                .transfer_deadline(transfer_id(dependency))
                .expect("pending source dependency must retain a submitted D2H");
            not_before_ms = not_before_ms.max(deadline);
        }

        let transfer_blocks = hit.keys.iter().rev().copied().collect::<Vec<_>>();
        let transfer_id = match self.domain.schedule_load(
            self.dp_rank,
            uuid,
            &transfer_blocks,
            now_ms,
            not_before_ms,
        ) {
            LoadOutcome::Queued(transfer_id) => transfer_id,
            LoadOutcome::Miss => {
                kv_manager.cancel_destination(reservation);
                return StartLoad::Retry;
            }
        };
        request.load = Some(LoadState::Loading(LoadingPrefix {
            transfer_id,
            keys: hit.keys,
            reservation,
            original_g1_tokens: hit.original_g1_tokens,
            loaded_prefix_blocks: hit.base_g1_blocks + hashes.len(),
            request_footprint_blocks: hit.request_footprint_blocks,
        }));
        StartLoad::Queued
    }

    /// Complete due transfers. Store dependencies become terminal before any
    /// same-timestamp H2D activation is handed back to the scheduler.
    pub(super) fn advance(
        &mut self,
        kv_manager: &mut G1Manager,
        now_ms: f64,
    ) -> HostTransferProgress {
        let (completed, generation) = self.domain.tick_for_rank(self.dp_rank, now_ms);
        let completed_any = generation != self.seen_completion_generation;
        self.seen_completion_generation = generation;
        for transfer in &completed {
            let CompletedTransfer::Store {
                request_id: _,
                transfer_id,
                blocks,
            } = transfer
            else {
                continue;
            };
            assert!(kv_manager.satisfy_native_source_dependency(source_dependency(*transfer_id)));
            if let Some(stores) = self.pending_stores.remove(transfer_id) {
                debug_assert_eq!(stores.len(), blocks.len());
                kv_manager.publish_native_host_stores(stores);
            } else {
                debug_assert!(!kv_manager.emits_native_kv_events());
            }
        }

        let mut loads = Vec::new();
        for transfer in completed {
            let CompletedTransfer::Load {
                request_id: uuid,
                transfer_id,
                blocks: _,
            } = transfer
            else {
                continue;
            };
            loads.push(CompletedLoad { uuid, transfer_id });
        }
        HostTransferProgress {
            completed_loads: loads,
            completed_any,
        }
    }

    /// Submit the stores prepared by the completed scheduler pass, then settle
    /// configured zero-latency transfers at the same boundary.
    pub(super) fn complete_engine_boundary(
        &mut self,
        kv_manager: &mut G1Manager,
        now_ms: f64,
    ) -> HostTransferProgress {
        let mut progress = self.advance(kv_manager, now_ms);
        self.domain.submit_prepared_stores(now_ms);
        let settled = self.advance(kv_manager, now_ms);
        progress.completed_loads.extend(settled.completed_loads);
        progress.completed_any |= settled.completed_any;
        progress
    }

    pub(super) fn activate_completed_load(
        &mut self,
        uuid: Uuid,
        transfer_id: TransferId,
        request: &mut VllmHostRequestState,
        sequence: &RequestSequence,
        lease: &mut BlockRequestLease,
        kv_manager: &mut G1Manager,
    ) {
        let Some(LoadState::Loading(load)) = request.load.take() else {
            panic!("completed H2D lost its request reservation")
        };
        assert_eq!(load.transfer_id, transfer_id);
        assert!(
            kv_manager
                .native_destination_pending_dependencies(&load.reservation)
                .is_empty(),
            "H2D completed before its destination dependencies"
        );
        kv_manager.activate_native_destination(uuid, sequence, lease, load.reservation);
        request.load = Some(LoadState::Activated(ActivatedPrefix {
            original_g1_tokens: load.original_g1_tokens,
            loaded_prefix_blocks: load.loaded_prefix_blocks,
            loaded_blocks: load.keys.len(),
            request_footprint_blocks: load.request_footprint_blocks,
        }));
    }

    /// Freeze the first post-H2D retry to the prefix owned by its reservation.
    pub(super) fn freeze_activated_cost(
        request: &VllmHostRequestState,
        mut cost: PrefillCost,
        block_size: usize,
    ) -> PrefillCost {
        let Some(load) = request.activated() else {
            return cost;
        };
        let loaded_tokens = load.loaded_prefix_blocks * block_size;
        assert!(cost.cached_tokens >= loaded_tokens);
        let newly_visible = cost.cached_tokens - loaded_tokens;
        assert_eq!(newly_visible % block_size, 0);
        cost.cached_tokens = loaded_tokens;
        cost.active_cached_tokens = loaded_tokens;
        cost.new_tokens += newly_visible;
        cost.new_blocks += newly_visible / block_size;
        cost
    }

    /// Advance the persistent store cursor and return first-admission tier attribution.
    pub(super) fn on_admitted(
        request: &mut VllmHostRequestState,
        reused_tokens: usize,
        block_size: usize,
    ) -> Option<(usize, usize)> {
        let attribution = match request.load.take() {
            Some(LoadState::Activated(load)) => {
                assert!(reused_tokens / block_size <= load.loaded_prefix_blocks);
                let g1 = load.original_g1_tokens.min(reused_tokens);
                let g2 = reused_tokens - g1;
                assert!(g2 <= load.loaded_blocks * block_size);
                request.next_store_block = reused_tokens / block_size;
                (g1, g2)
            }
            Some(LoadState::Loading(_)) => panic!("loading request cannot be admitted"),
            None => (reused_tokens, 0),
        };
        if request.first_admission_recorded {
            None
        } else {
            request.first_admission_recorded = true;
            Some(attribution)
        }
    }

    /// Prepare one prompt-only, missing-only, per-request write-through cohort.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe_completed_blocks(
        &mut self,
        uuid: Uuid,
        request: &mut VllmHostRequestState,
        lease: &BlockRequestLease,
        computed_tokens: usize,
        prompt_tokens: usize,
        block_size: usize,
        kv_manager: &mut G1Manager,
        now_ms: f64,
    ) {
        let eligible_blocks = computed_tokens.min(prompt_tokens) / block_size;
        let end = eligible_blocks.min(request.prompt_keys.len());
        if end <= request.next_store_block {
            return;
        }
        let start = request.next_store_block;
        let missing_indices = (start..end)
            .filter(|index| {
                matches!(
                    self.domain.lookup(request.prompt_keys[*index]),
                    Lookup::Miss
                )
            })
            .collect::<Vec<_>>();
        if missing_indices.is_empty() {
            request.next_store_block = end;
            return;
        }
        let Some(snapshot) =
            kv_manager.snapshot_native_store_sources(uuid, lease, &missing_indices)
        else {
            return;
        };
        assert_eq!(snapshot.len(), missing_indices.len());
        assert!(
            snapshot
                .sequence_hashes()
                .zip(missing_indices.iter().copied())
                .all(|(hash, index)| HostBlockKey::new(hash) == request.prompt_keys[index])
        );
        match self.domain.prepare_store(
            self.dp_rank,
            uuid,
            &request.prompt_keys[start..end],
            now_ms,
        ) {
            StoreOutcome::AlreadyPresent => request.next_store_block = end,
            StoreOutcome::RetryCapacity { .. } => {}
            StoreOutcome::Prepared {
                transfer_id,
                stored_blocks,
                evicted,
            } => {
                assert_eq!(stored_blocks, snapshot.len());
                if kv_manager.emits_native_kv_events() {
                    let store_events = kv_manager.native_host_store_events(lease, &missing_indices);
                    assert_eq!(store_events.len(), stored_blocks);
                    assert!(
                        self.pending_stores
                            .insert(transfer_id, store_events)
                            .is_none(),
                        "native host transfer ID was reused before completion"
                    );
                    kv_manager.publish_native_host_removed(
                        evicted
                            .into_iter()
                            .map(HostBlockKey::sequence_hash)
                            .collect(),
                    );
                }
                kv_manager.attach_native_store_source_dependency(
                    uuid,
                    lease,
                    snapshot,
                    source_dependency(transfer_id),
                );
                if let Some(observer) = &self.observer {
                    let mappings = missing_indices
                        .iter()
                        .map(|index| HostStoreBlockMapping {
                            block: request.prompt_keys[*index],
                            logical_block_index: *index,
                        })
                        .collect::<Vec<_>>();
                    observer.record(HostOffloadObservation {
                        request_id: uuid,
                        event: HostOffloadObservationData::StoreBlockMappings {
                            at_ms: now_ms,
                            transfer_id,
                            mappings: &mappings,
                        },
                    });
                }
                request.latest_store = Some(transfer_id);
                request.next_store_block = end;
                for key in request.prompt_keys.iter().rev().copied() {
                    self.domain.touch(key);
                }
            }
        }
    }

    /// Fence and authorize newly acquired G1 capacity whose prior owner is
    /// still being copied to host.
    pub(super) fn fence_allocation(
        &mut self,
        uuid: Uuid,
        lease: &mut BlockRequestLease,
        dependencies: &[SourceReuseDependency],
        kv_manager: &mut G1Manager,
    ) {
        if dependencies.is_empty() {
            return;
        }
        for dependency in dependencies {
            let deadline = self
                .domain
                .transfer_deadline(transfer_id(*dependency))
                .expect("pending dependency must retain a submitted D2H");
            self.compute_not_before_ms = self.compute_not_before_ms.max(deadline);
        }
        kv_manager.authorize_native_compute_after_dependencies(uuid, lease, dependencies);
    }

    /// vLLM flushes this request's prepared stores before releasing its G1 capacity.
    pub(super) fn preempt_request(&mut self, request: &VllmHostRequestState) {
        if let Some(transfer_id) = request.latest_store
            && let Some(deadline) = self.domain.transfer_deadline(transfer_id)
        {
            self.compute_not_before_ms = self.compute_not_before_ms.max(deadline);
        }
    }

    pub(super) fn cancel_request(
        &mut self,
        request: &mut VllmHostRequestState,
        kv_manager: &mut G1Manager,
        mutation_now_ms: f64,
        observed_at_ms: f64,
    ) -> bool {
        let Some(load) = request.load.take() else {
            return false;
        };
        let LoadState::Loading(load) = load else {
            return false;
        };
        assert!(self.domain.cancel_load(
            self.dp_rank,
            load.transfer_id,
            &load.keys,
            mutation_now_ms,
            observed_at_ms,
        ));
        kv_manager.cancel_destination(load.reservation);
        true
    }

    pub(super) fn compute_not_before_ms(&self, now_ms: f64) -> f64 {
        self.compute_not_before_ms.max(now_ms)
    }

    pub(super) fn next_deadline(&self) -> Option<f64> {
        self.domain
            .next_deadline_for_rank(self.dp_rank, self.seen_completion_generation)
    }

    pub(super) fn current_time_ms(&self) -> f64 {
        self.domain.current_time_ms()
    }

    pub(super) fn has_work(&self) -> bool {
        self.domain.has_pending_work()
    }

    #[cfg(test)]
    pub(super) fn resident_blocks(&self) -> usize {
        self.domain.resident_blocks()
    }
}

fn source_dependency(transfer_id: TransferId) -> SourceReuseDependency {
    SourceReuseDependency::from_adapter_id(transfer_id.get())
}

fn transfer_id(dependency: SourceReuseDependency) -> TransferId {
    TransferId::new(dependency.adapter_id())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::engine::common::protocols::{DirectRequest, KvEventPublishers, MockEngineArgs};
    use crate::engine::kv_manager::NativeAllocation;
    use crate::engine::scheduler::vllm::core::VllmCore;
    use crate::engine::trace::TraceCollector;
    use crate::engine::{HostOffloadObservation, HostOffloadObservationData};

    fn adapter(capacity_blocks: usize) -> VllmHostOffloadAdapter {
        VllmHostOffloadAdapter::new(
            &NativeHostOffloadConfig {
                num_host_blocks: capacity_blocks,
                d2h_bandwidth_gbps: 1.0,
                h2d_bandwidth_gbps: 1.0,
            },
            4,
            250_000,
        )
        .unwrap()
    }

    fn request(
        owner: Uuid,
        tokens: Vec<u32>,
    ) -> (RequestSequence, BlockRequestLease, VllmHostRequestState) {
        let (sequence, identities) =
            RequestSequence::new(tokens, 0, 0, 4, true, false, false, None);
        let lease = BlockRequestLease::new(owner, identities);
        let host = VllmHostRequestState::new(&sequence, &lease, 4);
        (sequence, lease, host)
    }

    fn seed_host(adapter: &mut VllmHostOffloadAdapter, key: HostBlockKey) -> f64 {
        let StoreOutcome::Prepared { transfer_id, .. } =
            adapter
                .domain
                .prepare_store(adapter.dp_rank, Uuid::nil(), &[key], 0.0)
        else {
            panic!("host seed was prechecked as absent")
        };
        assert_eq!(adapter.domain.submit_prepared_stores(0.0), 1);
        let deadline = adapter.domain.transfer_deadline(transfer_id).unwrap();
        let (completed, _) = adapter.domain.tick_for_rank(adapter.dp_rank, deadline);
        assert!(matches!(
            completed.as_slice(),
            [CompletedTransfer::Store { .. }]
        ));
        deadline
    }

    fn raw_cost(
        manager: &G1Manager,
        sequence: &RequestSequence,
        lease: &BlockRequestLease,
    ) -> PrefillCost {
        manager.get_native_prefill_cost(sequence, lease)
    }

    #[test]
    fn cache_domain_shares_capacity_and_transfer_lanes_across_ranks() {
        let config = NativeHostOffloadConfig {
            num_host_blocks: 2,
            d2h_bandwidth_gbps: 1.0,
            h2d_bandwidth_gbps: 1.0,
        };
        let domain = VllmHostOffloadDomain::new(&config, 4, 250_000).unwrap();
        let rank_0 = VllmHostOffloadAdapter::in_domain(domain.clone(), 0);
        let rank_1 = VllmHostOffloadAdapter::in_domain(domain.clone(), 1);
        let key_0 = HostBlockKey::new(100);
        let key_1 = HostBlockKey::new(200);

        let StoreOutcome::Prepared {
            transfer_id: store_0,
            ..
        } = domain.prepare_store(0, Uuid::from_u128(1), &[key_0], 0.0)
        else {
            panic!("rank 0 store must fit in the shared host domain")
        };
        let StoreOutcome::Prepared {
            transfer_id: store_1,
            ..
        } = domain.prepare_store(1, Uuid::from_u128(2), &[key_1], 0.0)
        else {
            panic!("rank 1 store must fit in the shared host domain")
        };
        assert_eq!(domain.submit_prepared_stores(0.0), 2);
        assert_eq!(domain.transfer_deadline(store_0), Some(1.0));
        assert_eq!(
            domain.transfer_deadline(store_1),
            Some(2.0),
            "both ranks must consume one host-scoped D2H lane"
        );

        let (rank_0_completions, _) = domain.tick_for_rank(0, 2.0);
        let (rank_1_completions, _) = domain.tick_for_rank(1, 2.0);
        assert_eq!(rank_0_completions.len(), 1);
        assert_eq!(rank_1_completions.len(), 1);
        assert_eq!(rank_0.resident_blocks(), 2);
        assert_eq!(rank_1.resident_blocks(), 2);

        let LoadOutcome::Queued(load_0) =
            domain.schedule_load(0, Uuid::from_u128(3), &[key_0], 2.0, 2.0)
        else {
            panic!("rank 0 load must use the shared host domain")
        };
        let LoadOutcome::Queued(load_1) =
            domain.schedule_load(1, Uuid::from_u128(4), &[key_1], 2.0, 2.0)
        else {
            panic!("rank 1 load must use the shared host domain")
        };
        assert_eq!(domain.transfer_deadline(load_0), Some(3.0));
        assert_eq!(
            domain.transfer_deadline(load_1),
            Some(4.0),
            "both ranks must consume one host-scoped H2D lane"
        );

        let isolated = VllmHostOffloadDomain::new(&config, 4, 250_000).unwrap();
        assert_eq!(isolated.lookup(key_0), Lookup::Miss);
        assert_eq!(isolated.lookup(key_1), Lookup::Miss);
    }

    #[test]
    fn completed_host_stores_and_evictions_emit_host_pinned_kv_events() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(1)
            .block_size(4)
            .max_num_seqs(Some(1))
            .max_num_batched_tokens(Some(4))
            .enable_prefix_caching(true)
            .kv_cache_bytes_per_token(Some(250_000))
            .native_host_offload(Some(
                NativeHostOffloadConfig::new(1).with_bandwidths(0.0, 0.0),
            ))
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new_with_kv_capture(args, 0);
        let mut collector = TraceCollector::default();

        let mut complete = |uuid, tokens: Vec<u32>, now_ms: f64| {
            core.receive(DirectRequest {
                tokens,
                max_output_tokens: 0,
                uuid: Some(uuid),
                arrival_timestamp_ms: Some(now_ms),
                ..Default::default()
            });
            let pass = core.execute_pass(&mut collector, now_ms);
            let mut events = pass.kv_events;
            core.complete_engine_boundary(pass.end_ms);
            assert!(core.is_empty());
            events.extend(core.drain_kv_events());
            events
        };

        let first = complete(Uuid::from_u128(1), vec![1, 2, 3, 4], 0.0);
        let first_host = first
            .iter()
            .find(|event| event.tier == crate::engine::KvEventTier::HostPinned)
            .expect("completed D2H must publish host residency");
        let crate::engine::KvEventData::Stored(first_store) = &first_host.data else {
            panic!("first host event must store the completed block")
        };
        let first_hash = first_store.blocks[0].block_hash;
        assert_ne!(first_store.blocks[0].tokens_hash, 0);

        let second = complete(Uuid::from_u128(2), vec![5, 6, 7, 8], 1.0);
        let host_events = second
            .iter()
            .filter(|event| event.tier == crate::engine::KvEventTier::HostPinned)
            .collect::<Vec<_>>();
        assert_eq!(host_events.len(), 2);
        assert!(matches!(
            &host_events[0].data,
            crate::engine::KvEventData::Removed { block_hashes }
                if block_hashes == &[first_hash]
        ));
        assert!(matches!(
            &host_events[1].data,
            crate::engine::KvEventData::Stored(store) if store.blocks.len() == 1
        ));
    }

    #[derive(Default)]
    struct EvictionCapture(Mutex<Vec<HostBlockKey>>);

    impl EvictionCapture {
        fn take(&self) -> Vec<HostBlockKey> {
            std::mem::take(&mut *self.0.lock().expect("eviction capture poisoned"))
        }
    }

    impl HostOffloadObserver for EvictionCapture {
        fn record(&self, observation: HostOffloadObservation<'_>) {
            if let HostOffloadObservationData::Evicted { block, .. } = observation.event {
                self.0
                    .lock()
                    .expect("eviction capture poisoned")
                    .push(block);
            }
        }
    }

    #[test]
    fn exhausted_pass_budget_does_not_touch_a_waiting_requests_host_prefix() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(64)
            .block_size(4)
            .max_num_seqs(Some(3))
            .max_num_batched_tokens(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .kv_cache_bytes_per_token(Some(250_000))
            .native_host_offload(Some(NativeHostOffloadConfig {
                num_host_blocks: 4,
                d2h_bandwidth_gbps: 0.0,
                h2d_bandwidth_gbps: 0.0,
            }))
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let evictions = Arc::new(EvictionCapture::default());
        core.set_host_offload_observer(evictions.clone());
        let mut collector = TraceCollector::default();
        let mut now_ms = 0.0;

        let mut complete_request =
            |core: &mut VllmCore, uuid: Uuid, tokens: Vec<u32>, now_ms: &mut f64| {
                core.receive(DirectRequest {
                    tokens,
                    max_output_tokens: 0,
                    uuid: Some(uuid),
                    arrival_timestamp_ms: Some(*now_ms),
                    ..Default::default()
                });
                for _ in 0..4 {
                    let pass = core.execute_pass(&mut collector, *now_ms);
                    *now_ms = pass.end_ms;
                    core.complete_engine_boundary(*now_ms);
                    if core.is_empty() {
                        return;
                    }
                }
                panic!("seed request did not complete");
            };

        // The two-block lineage supplies a non-prefix G2 candidate `a`: after
        // its first block is evicted, connector lookup still touches `a` before
        // discovering the prefix miss.
        let shared_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (_, _, shared_host) = request(Uuid::from_u128(20), shared_tokens.clone());
        let [x, a] = shared_host.prompt_keys.as_slice() else {
            panic!("shared request must have two prompt blocks")
        };
        let x = *x;
        let a = *a;
        complete_request(
            &mut core,
            Uuid::from_u128(20),
            shared_tokens.clone(),
            &mut now_ms,
        );

        // Fill the remainder of G2 in a known LRU order: X, A, B, D.
        complete_request(
            &mut core,
            Uuid::from_u128(21),
            vec![9, 10, 11, 12],
            &mut now_ms,
        );
        complete_request(
            &mut core,
            Uuid::from_u128(22),
            vec![13, 14, 15, 16],
            &mut now_ms,
        );
        assert!(evictions.take().is_empty());

        // The producer's first chunk stores C0 and evicts X, leaving A as LRU.
        let producer = Uuid::from_u128(23);
        core.receive(DirectRequest {
            tokens: vec![21, 22, 23, 24, 25, 26, 27, 28],
            max_output_tokens: 0,
            uuid: Some(producer),
            arrival_timestamp_ms: Some(now_ms),
            ..Default::default()
        });
        let first_chunk = core.execute_pass(&mut collector, now_ms);
        now_ms = first_chunk.end_ms;
        core.complete_engine_boundary(now_ms);
        assert_eq!(evictions.take(), vec![x]);
        assert!(!core.is_empty());

        // The producer is running and consumes the complete four-token budget.
        // vLLM must not query the waiting request afterward. If it does, the
        // lookup touches A despite X missing, promotes A to MRU, and the
        // producer's C1 store evicts B instead of A.
        core.receive(DirectRequest {
            tokens: shared_tokens,
            max_output_tokens: 1,
            uuid: Some(Uuid::from_u128(24)),
            arrival_timestamp_ms: Some(now_ms),
            ..Default::default()
        });
        core.execute_pass(&mut collector, now_ms);

        assert_eq!(evictions.take(), vec![a]);
    }

    #[test]
    fn h2d_reserves_real_g1_capacity_and_cancel_releases_it() {
        let mut adapter = adapter(2);
        let destination_id = Uuid::from_u128(1);
        let (destination_sequence, destination_lease, mut destination_host) =
            request(destination_id, vec![1, 2, 3, 4]);
        let key = destination_host.prompt_keys[0];
        let now_ms = seed_host(&mut adapter, key);

        let mut manager = G1Manager::new_with_caching(1, 4, KvEventPublishers::default(), 0, true);
        let source_id = Uuid::from_u128(2);
        let (mut source_sequence, mut source_lease, _) = request(source_id, vec![5, 6, 7, 8]);
        assert!(matches!(
            manager.allocate_native(source_id, &mut source_lease, 4, 0),
            NativeAllocation::Ready { dependencies, .. } if dependencies.is_empty()
        ));
        manager.finalize_native_computed_prefix(
            source_id,
            0,
            4,
            &mut source_sequence,
            &mut source_lease,
        );

        let HostLookup::Hit(hit) = adapter.lookup(
            &mut destination_host,
            &destination_sequence,
            &raw_cost(&manager, &destination_sequence, &destination_lease),
            4,
        ) else {
            panic!("seeded host block must hit")
        };
        assert!(matches!(
            adapter.start_load(
                destination_id,
                &mut destination_host,
                &destination_lease,
                hit,
                &mut manager,
                now_ms,
            ),
            StartLoad::CapacityBlocked
        ));

        manager.finish_native(source_id, source_lease);
        let HostLookup::Hit(hit) = adapter.lookup(
            &mut destination_host,
            &destination_sequence,
            &raw_cost(&manager, &destination_sequence, &destination_lease),
            4,
        ) else {
            panic!("host block must remain resident")
        };
        assert!(matches!(
            adapter.start_load(
                destination_id,
                &mut destination_host,
                &destination_lease,
                hit,
                &mut manager,
                now_ms,
            ),
            StartLoad::Queued
        ));
        let follower_id = Uuid::from_u128(8);
        let (follower_sequence, follower_lease, mut follower_host) =
            request(follower_id, vec![1, 2, 3, 4]);
        assert!(matches!(
            adapter.lookup(
                &mut follower_host,
                &follower_sequence,
                &raw_cost(&manager, &follower_sequence, &follower_lease),
                4,
            ),
            HostLookup::Deferred
        ));
        assert_eq!(manager.num_active_blocks(), 1);
        adapter.cancel_request(&mut destination_host, &mut manager, now_ms, now_ms);
        assert_eq!(manager.num_active_blocks(), 0);
        assert!(adapter.domain.is_resident(key));
        assert!(matches!(
            adapter.lookup(
                &mut follower_host,
                &follower_sequence,
                &raw_cost(&manager, &follower_sequence, &follower_lease),
                4,
            ),
            HostLookup::Hit(_)
        ));
    }

    #[test]
    fn completed_h2d_activates_the_reserved_request_lease_as_a_real_hit() {
        let mut adapter = adapter(2);
        let owner = Uuid::from_u128(3);
        let (sequence, mut lease, mut host) = request(owner, vec![1, 2, 3, 4]);
        let now_ms = seed_host(&mut adapter, host.prompt_keys[0]);
        let mut manager = G1Manager::new_with_caching(1, 4, KvEventPublishers::default(), 0, true);
        let HostLookup::Hit(hit) = adapter.lookup(
            &mut host,
            &sequence,
            &raw_cost(&manager, &sequence, &lease),
            4,
        ) else {
            panic!("seeded host block must hit")
        };
        assert!(matches!(
            adapter.start_load(owner, &mut host, &lease, hit, &mut manager, now_ms),
            StartLoad::Queued
        ));
        let transfer_id = match &host.load {
            Some(LoadState::Loading(load)) => load.transfer_id,
            _ => panic!("request must be loading"),
        };
        let deadline = adapter.domain.transfer_deadline(transfer_id).unwrap();
        let completed = adapter.advance(&mut manager, deadline).completed_loads;
        assert_eq!(completed.len(), 1);
        adapter.activate_completed_load(
            owner,
            transfer_id,
            &mut host,
            &sequence,
            &mut lease,
            &mut manager,
        );
        assert_eq!(raw_cost(&manager, &sequence, &lease).cached_tokens, 4);
        assert_eq!(lease.resident_block_count(), 1);
        assert_eq!(adapter.resident_blocks(), 1);
    }

    #[test]
    fn reused_store_source_fences_but_does_not_pin_g1_capacity() {
        let mut adapter = adapter(2);
        let mut manager = G1Manager::new_with_caching(1, 4, KvEventPublishers::default(), 0, true);
        let source_id = Uuid::from_u128(6);
        let (mut source_sequence, mut source_lease, mut source_host) =
            request(source_id, vec![1, 2, 3, 4]);
        assert!(matches!(
            manager.allocate_native(source_id, &mut source_lease, 4, 0),
            NativeAllocation::Ready { dependencies, .. } if dependencies.is_empty()
        ));
        manager.finalize_native_computed_prefix(
            source_id,
            0,
            4,
            &mut source_sequence,
            &mut source_lease,
        );
        adapter.observe_completed_blocks(
            source_id,
            &mut source_host,
            &source_lease,
            4,
            4,
            4,
            &mut manager,
            0.0,
        );
        let store_id = source_host.latest_store.unwrap();
        assert!(manager.is_native_source_dependency_pending(source_dependency(store_id)));
        adapter.complete_engine_boundary(&mut manager, 0.0);
        assert_eq!(adapter.domain.transfer_deadline(store_id), Some(1.0));
        assert!(matches!(
            adapter.lookup(
                &mut source_host,
                &source_sequence,
                &raw_cost(&manager, &source_sequence, &source_lease),
                4,
            ),
            HostLookup::Deferred
        ));

        manager.finish_native(source_id, source_lease);
        assert_eq!(manager.num_active_blocks(), 0);
        let destination_id = Uuid::from_u128(7);
        let (mut destination_sequence, mut destination_lease, _) =
            request(destination_id, vec![5, 6, 7, 8]);
        let allocation = manager.allocate_native(destination_id, &mut destination_lease, 4, 0);
        let NativeAllocation::Ready { dependencies, .. } = allocation else {
            panic!("released source capacity must be reusable")
        };
        assert_eq!(dependencies, vec![source_dependency(store_id)]);
        adapter.fence_allocation(
            destination_id,
            &mut destination_lease,
            &dependencies,
            &mut manager,
        );
        manager.finalize_native_computed_prefix(
            destination_id,
            0,
            4,
            &mut destination_sequence,
            &mut destination_lease,
        );
        assert_eq!(adapter.compute_not_before_ms(0.0), 1.0);
        adapter.advance(&mut manager, 1.0);
        assert!(!manager.is_native_source_dependency_pending(source_dependency(store_id)));
    }
}
