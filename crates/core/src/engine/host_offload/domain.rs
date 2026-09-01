// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Physical host-cache domains shared by attention-DP ranks.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use rustc_hash::FxHashMap;
use uuid::Uuid;

use super::{
    CompletedTransfer, HostBlockKey, HostOffloadObserver, HostTier, HostTierConfig, LoadOutcome,
    Lookup, StoreOutcome, TransferId,
};
use crate::engine::NativeHostOffloadConfig;

struct HostCacheDomainState {
    tier: HostTier,
    load_by_key: FxHashMap<HostBlockKey, TransferId>,
    transfer_owners: FxHashMap<TransferId, u32>,
    completed_by_rank: FxHashMap<u32, Vec<CompletedTransfer>>,
    completion_generation: u64,
}

/// One physical G2 cache, capacity budget, and pair of transfer lanes.
#[derive(Clone)]
pub(crate) struct HostCacheDomain {
    state: Rc<RefCell<HostCacheDomainState>>,
}

impl HostCacheDomain {
    pub(crate) fn new(
        config: &NativeHostOffloadConfig,
        block_size: usize,
        kv_bytes_per_token: usize,
    ) -> anyhow::Result<Self> {
        let block_bytes = block_size
            .checked_mul(kv_bytes_per_token)
            .ok_or_else(|| anyhow::anyhow!("native host block byte size overflow"))?;
        Ok(Self {
            state: Rc::new(RefCell::new(HostCacheDomainState {
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

    pub(crate) fn bind_rank(&self, dp_rank: u32) -> HostCacheRankHandle {
        HostCacheRankHandle {
            domain: self.clone(),
            dp_rank,
            seen_completion_generation: 0,
        }
    }

    fn mutate<T>(&self, mutation: impl FnOnce(&mut HostCacheDomainState) -> T) -> T {
        mutation(&mut self.state.borrow_mut())
    }

    fn tick_for_rank(&self, dp_rank: u32, now_ms: f64) -> (Vec<CompletedTransfer>, u64) {
        self.mutate(|state| {
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
        })
    }
}

/// Rank-bound view of a shared host-cache domain.
///
/// The handle owns the rank identity and its completion cursor so callers
/// cannot accidentally charge transfers to, or poll completions for, another
/// rank. A generation change wakes every rank after shared-domain progress,
/// including ranks whose deferred lookup did not own the completing transfer.
pub(crate) struct HostCacheRankHandle {
    domain: HostCacheDomain,
    dp_rank: u32,
    seen_completion_generation: u64,
}

impl HostCacheRankHandle {
    pub(crate) fn set_observer(&self, observer: Arc<dyn HostOffloadObserver>) {
        self.domain.state.borrow_mut().tier.set_observer(observer);
    }

    pub(crate) fn has_transfer(&self, transfer_id: TransferId) -> bool {
        self.domain.state.borrow().tier.has_transfer(transfer_id)
    }

    pub(crate) fn touch(&self, key: HostBlockKey) {
        self.domain.state.borrow_mut().tier.touch(key);
    }

    pub(crate) fn lookup(&self, key: HostBlockKey) -> Lookup {
        self.domain.state.borrow().tier.lookup(key)
    }

    pub(crate) fn is_loading(&self, key: HostBlockKey) -> bool {
        self.domain.state.borrow().load_by_key.contains_key(&key)
    }

    pub(crate) fn prepare_store(
        &self,
        request_id: Uuid,
        blocks: &[HostBlockKey],
        now_ms: f64,
    ) -> StoreOutcome {
        self.domain.mutate(|state| {
            let outcome = state.tier.prepare_store(request_id, blocks, now_ms);
            if let StoreOutcome::Prepared { transfer_id, .. } = &outcome {
                assert!(
                    state
                        .transfer_owners
                        .insert(*transfer_id, self.dp_rank)
                        .is_none()
                );
            }
            outcome
        })
    }

    pub(crate) fn submit_prepared_stores(&self, now_ms: f64) -> usize {
        self.domain
            .mutate(|state| state.tier.submit_prepared_stores(now_ms))
    }

    pub(crate) fn schedule_load(
        &self,
        request_id: Uuid,
        blocks: &[HostBlockKey],
        now_ms: f64,
        not_before_ms: f64,
    ) -> LoadOutcome {
        self.domain.mutate(|state| {
            if blocks.iter().any(|key| state.load_by_key.contains_key(key)) {
                return LoadOutcome::Miss;
            }
            let outcome = state
                .tier
                .schedule_load(request_id, blocks, now_ms, not_before_ms);
            if let LoadOutcome::Queued(transfer_id) = outcome {
                assert!(
                    state
                        .transfer_owners
                        .insert(transfer_id, self.dp_rank)
                        .is_none()
                );
                for key in blocks {
                    assert!(state.load_by_key.insert(*key, transfer_id).is_none());
                }
            }
            outcome
        })
    }

    pub(crate) fn tick(&mut self, now_ms: f64) -> (Vec<CompletedTransfer>, bool) {
        let (completed, generation) = self.domain.tick_for_rank(self.dp_rank, now_ms);
        let domain_advanced = generation != self.seen_completion_generation;
        self.seen_completion_generation = generation;
        (completed, domain_advanced)
    }

    pub(crate) fn cancel_load(
        &self,
        transfer_id: TransferId,
        keys: &[HostBlockKey],
        mutation_now_ms: f64,
        observed_at_ms: f64,
    ) -> bool {
        self.domain.mutate(|state| {
            if !state
                .tier
                .cancel_load(transfer_id, mutation_now_ms, observed_at_ms)
            {
                return false;
            }
            assert_eq!(
                state.transfer_owners.remove(&transfer_id),
                Some(self.dp_rank)
            );
            for key in keys {
                assert_eq!(state.load_by_key.remove(key), Some(transfer_id));
            }
            true
        })
    }

    pub(crate) fn transfer_deadline(&self, transfer_id: TransferId) -> Option<f64> {
        self.domain
            .state
            .borrow()
            .tier
            .transfer_deadline(transfer_id)
    }

    pub(crate) fn next_deadline(&self) -> Option<f64> {
        let state = self.domain.state.borrow();
        if state.completion_generation != self.seen_completion_generation
            || state
                .completed_by_rank
                .get(&self.dp_rank)
                .is_some_and(|completed| !completed.is_empty())
        {
            return Some(state.tier.current_time_ms());
        }
        state.tier.next_deadline()
    }

    pub(crate) fn current_time_ms(&self) -> f64 {
        self.domain.state.borrow().tier.current_time_ms()
    }

    pub(crate) fn has_pending_work(&self) -> bool {
        let state = self.domain.state.borrow();
        state.tier.has_pending_work()
            || state
                .completed_by_rank
                .values()
                .any(|completed| !completed.is_empty())
    }

    #[cfg(test)]
    pub(crate) fn resident_blocks(&self) -> usize {
        self.domain.state.borrow().tier.resident_blocks()
    }

    #[cfg(test)]
    pub(crate) fn is_resident(&self, key: HostBlockKey) -> bool {
        self.domain.state.borrow().tier.is_resident(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_domain_shares_capacity_and_transfer_lanes_across_ranks() {
        let config = NativeHostOffloadConfig {
            num_host_blocks: 2,
            d2h_bandwidth_gbps: 1.0,
            h2d_bandwidth_gbps: 1.0,
        };
        let domain = HostCacheDomain::new(&config, 4, 250_000).unwrap();
        let mut rank_0 = domain.bind_rank(0);
        let mut rank_1 = domain.bind_rank(1);
        let key_0 = HostBlockKey::new(100);
        let key_1 = HostBlockKey::new(200);

        let StoreOutcome::Prepared {
            transfer_id: store_0,
            ..
        } = rank_0.prepare_store(Uuid::from_u128(1), &[key_0], 0.0)
        else {
            panic!("rank 0 store must fit in the shared host domain")
        };
        let StoreOutcome::Prepared {
            transfer_id: store_1,
            ..
        } = rank_1.prepare_store(Uuid::from_u128(2), &[key_1], 0.0)
        else {
            panic!("rank 1 store must fit in the shared host domain")
        };
        assert_eq!(rank_0.submit_prepared_stores(0.0), 2);
        assert_eq!(rank_0.transfer_deadline(store_0), Some(1.0));
        assert_eq!(
            rank_1.transfer_deadline(store_1),
            Some(2.0),
            "both ranks must consume one host-scoped D2H lane"
        );

        let (rank_0_completions, _) = rank_0.tick(2.0);
        let (rank_1_completions, _) = rank_1.tick(2.0);
        assert_eq!(rank_0_completions.len(), 1);
        assert_eq!(rank_1_completions.len(), 1);
        assert_eq!(rank_0.resident_blocks(), 2);
        assert_eq!(rank_1.resident_blocks(), 2);

        let LoadOutcome::Queued(load_0) =
            rank_0.schedule_load(Uuid::from_u128(3), &[key_0], 2.0, 2.0)
        else {
            panic!("rank 0 load must use the shared host domain")
        };
        let LoadOutcome::Queued(load_1) =
            rank_1.schedule_load(Uuid::from_u128(4), &[key_1], 2.0, 2.0)
        else {
            panic!("rank 1 load must use the shared host domain")
        };
        assert_eq!(rank_0.transfer_deadline(load_0), Some(3.0));
        assert_eq!(
            rank_1.transfer_deadline(load_1),
            Some(4.0),
            "both ranks must consume one host-scoped H2D lane"
        );

        let isolated = HostCacheDomain::new(&config, 4, 250_000)
            .unwrap()
            .bind_rank(0);
        assert_eq!(isolated.lookup(key_0), Lookup::Miss);
        assert_eq!(isolated.lookup(key_1), Lookup::Miss);
    }
}
