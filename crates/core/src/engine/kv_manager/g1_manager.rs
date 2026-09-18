// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduler-facing facade over the native vLLM-style G1 manager.
//!
//! AISimulate owns a single native G1 implementation. External tiers and
//! provider selection are deliberately absent here.

use uuid::Uuid;

use crate::engine::belady::BeladyOracle;
use crate::engine::common::protocols::{KvEventPublishers, PrefillCost};
use crate::engine::common::sequence::RequestSequence;

use super::vllm_backend::{
    BlockRequestLease, DecodeBlockReservation as VllmDecodeBlockReservation,
    DestinationReservation as VllmDestinationReservation,
    StoreSourceSnapshot as VllmStoreSourceSnapshot, VllmAcquire, VllmKvManager,
};
pub(crate) use super::vllm_backend::{NativeAllocation, SourceReuseDependency};
use super::{DestinationReservationMode, G1Acquire};

fn into_g1_acquire<T>(outcome: VllmAcquire<T>) -> G1Acquire<T> {
    match outcome {
        VllmAcquire::Ready(value) => G1Acquire::Ready(value),
        VllmAcquire::CapacityExhausted => G1Acquire::CapacityExhausted,
    }
}

pub(crate) struct DecodeBlockReservation {
    inner: VllmDecodeBlockReservation,
}

pub(crate) struct DestinationReservation {
    inner: VllmDestinationReservation,
}

/// Opaque, short-lived G1 source view. Callers can read logical hashes but
/// physical copy identity remains private to the native manager.
#[must_use = "a store source snapshot must be attached or discarded synchronously"]
pub(crate) struct StoreSourceSnapshot<'a> {
    inner: VllmStoreSourceSnapshot<'a>,
}

impl StoreSourceSnapshot<'_> {
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    pub(crate) fn sequence_hashes(
        &self,
    ) -> impl ExactSizeIterator<Item = crate::engine::common::hashing::SequenceHash> + '_ {
        self.inner.sequence_hashes()
    }
}

impl DestinationReservation {
    pub(crate) fn transferable_prompt_tokens(&self, block_size: usize) -> usize {
        self.inner.transferable_prompt_tokens(block_size)
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Native GPU-block accounting shared by vLLM and TensorRT-LLM schedulers.
pub(crate) struct G1Manager {
    inner: VllmKvManager,
}

impl G1Manager {
    pub(crate) fn set_belady_oracle(&mut self, oracle: BeladyOracle) {
        self.inner.set_belady_oracle(oracle);
    }

    pub(crate) fn new_with_event_sink(
        max_capacity: usize,
        block_size: usize,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
    ) -> Self {
        Self::new_with_caching(max_capacity, block_size, kv_event_publishers, dp_rank, true)
    }

    pub(crate) fn new_with_caching(
        max_capacity: usize,
        block_size: usize,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
        enable_prefix_caching: bool,
    ) -> Self {
        Self {
            inner: VllmKvManager::new_with_event_sink(
                max_capacity,
                block_size,
                enable_prefix_caching,
                kv_event_publishers,
                dp_rank,
            ),
        }
    }

    pub(crate) fn allocate_native(
        &mut self,
        owner: Uuid,
        lease: &mut BlockRequestLease,
        cumulative_tokens: usize,
        reusable_prefix_blocks: usize,
    ) -> NativeAllocation<usize> {
        self.inner
            .allocate_lease(owner, lease, cumulative_tokens, reusable_prefix_blocks)
    }

    pub(crate) fn authorize_native_compute_after_dependencies(
        &mut self,
        owner: Uuid,
        lease: &mut BlockRequestLease,
        dependencies: &[SourceReuseDependency],
    ) {
        self.inner
            .authorize_lease_writes_after_dependencies(owner, lease, dependencies);
    }

    pub(crate) fn finalize_native_computed_prefix(
        &mut self,
        owner: Uuid,
        computed_before: usize,
        computed_after: usize,
        sequence: &mut RequestSequence,
        lease: &mut BlockRequestLease,
    ) {
        self.inner.finalize_lease_computed_prefix(
            owner,
            sequence,
            lease,
            computed_before,
            computed_after,
        );
    }

    pub(crate) fn preempt_native(&mut self, owner: Uuid, lease: &mut BlockRequestLease) {
        self.inner.preempt_lease(owner, lease);
    }

    pub(crate) fn finish_native(&mut self, owner: Uuid, lease: BlockRequestLease) {
        self.inner.finish_lease(owner, lease);
    }

    pub(crate) fn get_native_prefill_cost(
        &self,
        sequence: &RequestSequence,
        lease: &BlockRequestLease,
    ) -> PrefillCost {
        self.inner.get_lease_prefill_cost(sequence, lease)
    }

    pub(crate) fn reserve_native_destination_at(
        &mut self,
        owner: Uuid,
        sequence: &RequestSequence,
        lease: &BlockRequestLease,
        mode: DestinationReservationMode,
        eviction_now_ms: Option<f64>,
    ) -> G1Acquire<DestinationReservation> {
        into_g1_acquire(self.inner.reserve_destination_lease(
            owner,
            sequence,
            lease,
            mode,
            eviction_now_ms,
        ))
        .map(|inner| DestinationReservation { inner })
    }

    /// Reserve an exact G1 prefix plus only the suffix selected by the native
    /// host-cache lookup.
    pub(crate) fn reserve_native_host_destination(
        &mut self,
        owner: Uuid,
        lease: &BlockRequestLease,
        g1_prefix_blocks: usize,
        transferred_suffix: &[crate::engine::common::hashing::SequenceHash],
    ) -> G1Acquire<DestinationReservation> {
        into_g1_acquire(self.inner.reserve_external_prefix_lease(
            owner,
            lease,
            g1_prefix_blocks,
            transferred_suffix,
        ))
        .map(|inner| DestinationReservation { inner })
    }

    pub(crate) fn native_destination_pending_dependencies(
        &self,
        reservation: &DestinationReservation,
    ) -> Vec<SourceReuseDependency> {
        self.inner
            .destination_pending_dependencies(&reservation.inner)
    }

    pub(crate) fn snapshot_native_store_sources<'a>(
        &self,
        owner: Uuid,
        lease: &'a BlockRequestLease,
        block_indices: &'a [usize],
    ) -> Option<StoreSourceSnapshot<'a>> {
        self.inner
            .snapshot_store_sources(owner, lease, block_indices)
            .map(|inner| StoreSourceSnapshot { inner })
    }

    /// Attach the transfer identity after host capacity admission. Snapshot and
    /// attachment must remain in one non-yielding scheduler transition.
    pub(crate) fn attach_native_store_source_dependency(
        &mut self,
        owner: Uuid,
        lease: &BlockRequestLease,
        snapshot: StoreSourceSnapshot<'_>,
        dependency: SourceReuseDependency,
    ) {
        debug_assert_eq!(lease.owner(), owner, "native lease owner mismatch");
        self.inner
            .attach_store_source_dependency(owner, snapshot.inner, dependency);
    }

    pub(crate) fn satisfy_native_source_dependency(
        &mut self,
        dependency: SourceReuseDependency,
    ) -> bool {
        self.inner.satisfy_source_reuse_dependency(dependency)
    }

    pub(crate) fn is_native_source_dependency_pending(
        &self,
        dependency: SourceReuseDependency,
    ) -> bool {
        self.inner.is_source_reuse_dependency_pending(dependency)
    }

    pub(crate) fn activate_native_destination(
        &mut self,
        owner: Uuid,
        sequence: &RequestSequence,
        lease: &mut BlockRequestLease,
        reservation: DestinationReservation,
    ) {
        self.inner
            .activate_destination_lease(owner, sequence, lease, reservation.inner);
    }

    pub(crate) fn cancel_destination(&mut self, reservation: DestinationReservation) {
        self.inner.cancel_destination(reservation.inner);
    }

    pub(crate) fn reserve_decode_blocks(
        &mut self,
        count: usize,
    ) -> G1Acquire<DecodeBlockReservation> {
        into_g1_acquire(self.inner.reserve_decode_blocks(count))
            .map(|inner| DecodeBlockReservation { inner })
    }

    pub(crate) fn use_native_decode_reservation(
        &mut self,
        owner: Uuid,
        lease: &mut BlockRequestLease,
        cumulative_tokens: usize,
        reservation: &mut DecodeBlockReservation,
    ) -> NativeAllocation<usize> {
        self.inner.allocate_lease_from_decode_reservation(
            owner,
            lease,
            cumulative_tokens,
            &mut reservation.inner,
        )
    }

    pub(crate) fn release_decode_reservation(&mut self, reservation: DecodeBlockReservation) {
        self.inner.release_decode_reservation(reservation.inner);
    }

    pub(crate) fn num_active_blocks(&self) -> usize {
        self.inner.num_active_blocks()
    }

    pub(crate) fn num_inactive_blocks(&self) -> usize {
        self.inner.num_inactive_blocks()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::KvEventData;
    use crate::engine::common::protocols::KvEventPublishers;
    use crate::engine::scheduler::capture_kv_event_sink;

    #[test]
    fn native_finalization_publishes_parent_before_promoted_tail() {
        let owner = Uuid::from_u128(81_001);
        let (buffer, sink) = capture_kv_event_sink();
        let mut manager =
            G1Manager::new_with_event_sink(8, 4, KvEventPublishers::new(Some(sink)), 0);
        let (mut sequence, identities) = RequestSequence::new(
            (0..8).collect(),
            4,
            4,
            4,
            true,
            true,
            false,
            Some(vec![8, 9, 10, 11]),
        );
        let mut lease = BlockRequestLease::new(owner, identities);
        assert!(matches!(
            manager.allocate_native(owner, &mut lease, 8, 0),
            NativeAllocation::Ready {
                value: _,
                dependencies
            } if dependencies.is_empty()
        ));
        manager.finalize_native_computed_prefix(owner, 0, 8, &mut sequence, &mut lease);

        for _ in 0..4 {
            let (_, opened_partial) = sequence.generate_token();
            if opened_partial {
                lease.append_partial();
            }
        }
        assert!(matches!(
            manager.allocate_native(owner, &mut lease, 12, 0),
            NativeAllocation::Ready {
                value: _,
                dependencies
            } if dependencies.is_empty()
        ));
        manager.finalize_native_computed_prefix(owner, 8, 12, &mut sequence, &mut lease);

        let stored = buffer
            .drain()
            .into_iter()
            .filter_map(|event| match event.data {
                KvEventData::Stored(stored) => Some(stored),
                KvEventData::Removed { .. } => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].blocks.len(), 2);
        assert_eq!(stored[0].parent_hash, None);
        assert_eq!(stored[1].blocks.len(), 1);
        assert_eq!(stored[1].parent_hash, Some(stored[0].blocks[1].block_hash));
    }
}
