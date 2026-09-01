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

use std::sync::Arc;
use uuid::Uuid;

use crate::engine::HostOffloadObserver;
use crate::engine::KvEventPublisher;
use crate::engine::common::protocols::PrefillCost;
use crate::engine::common::sequence::RequestSequence;
use crate::engine::host_offload::{
    CompletedTransfer, HostBlockKey, HostCacheRankHandle, HostOffloadObservation,
    HostOffloadObservationData, HostStoreBlockMapping, LoadOutcome, Lookup, StoreOutcome,
    TransferId,
};
use crate::engine::kv_manager::{
    BlockRequestLease, DestinationReservation, G1Acquire, G1Manager, SourceReuseDependency,
};

mod events;

use events::HostKvEventTransactions;

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

pub(super) struct VllmHostOffloadAdapter {
    domain: HostCacheRankHandle,
    events: HostKvEventTransactions,
    compute_not_before_ms: f64,
    /// Present only for detailed artifact capture. Ordinary runs neither retain
    /// request-local mapping state nor allocate mapping payloads.
    observer: Option<Arc<dyn HostOffloadObserver>>,
}

impl VllmHostOffloadAdapter {
    pub(super) fn set_observer(&mut self, observer: Arc<dyn HostOffloadObserver>) {
        self.domain.set_observer(Arc::clone(&observer));
        self.observer = Some(observer);
    }

    pub(super) fn new(domain: HostCacheRankHandle, events: KvEventPublisher) -> Self {
        Self {
            domain,
            events: HostKvEventTransactions::new(events),
            compute_not_before_ms: 0.0,
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
        let transfer_id =
            match self
                .domain
                .schedule_load(uuid, &transfer_blocks, now_ms, not_before_ms)
            {
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
        let (completed, completed_any) = self.domain.tick(now_ms);
        for transfer in &completed {
            let CompletedTransfer::Store {
                request_id: _,
                transfer_id,
                blocks: _,
            } = transfer
            else {
                continue;
            };
            let dependency = source_dependency(*transfer_id);
            assert!(kv_manager.satisfy_native_source_dependency(dependency));
            self.events.complete_store(dependency);
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
        match self
            .domain
            .prepare_store(uuid, &request.prompt_keys[start..end], now_ms)
        {
            StoreOutcome::AlreadyPresent => request.next_store_block = end,
            StoreOutcome::RetryCapacity { .. } => {}
            StoreOutcome::Prepared {
                transfer_id,
                stored_blocks,
                evicted,
            } => {
                assert_eq!(stored_blocks, snapshot.len());
                let dependency = source_dependency(transfer_id);
                self.events.stage_store(
                    dependency,
                    lease,
                    &missing_indices,
                    evicted
                        .into_iter()
                        .map(HostBlockKey::sequence_hash)
                        .collect(),
                );
                kv_manager.attach_native_store_source_dependency(uuid, lease, snapshot, dependency);
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
                            mappings,
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
        self.domain.next_deadline()
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
mod tests;
