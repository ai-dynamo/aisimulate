// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Acceptance cases for the manual G1 token/state capacity model.

use super::*;

fn manager(
    capacity: usize,
    block_size: usize,
    state_blocks: usize,
    caching: bool,
) -> VllmKvManager {
    let mut manager = VllmKvManager::new_with_event_sink(
        capacity,
        block_size,
        caching,
        KvEventPublishers::default(),
        0,
    );
    manager.enable_state_cache(state_blocks, false);
    manager
}

fn request(
    owner: Uuid,
    tokens: usize,
    block_size: usize,
    caching: bool,
    output: usize,
) -> (RequestSequence, BlockRequestLease) {
    let (sequence, identities) = RequestSequence::new(
        (0..tokens).map(|value| value as u32).collect(),
        output,
        output,
        block_size,
        caching,
        true,
        false,
        Some(
            (tokens..tokens + output)
                .map(|value| value as u32)
                .collect(),
        ),
    );
    (sequence, BlockRequestLease::new(owner, identities))
}

fn allocated(result: NativeAllocation<usize>) -> usize {
    match result {
        NativeAllocation::Ready {
            value,
            dependencies,
        } => {
            assert!(dependencies.is_empty());
            value
        }
        NativeAllocation::CapacityExhausted => panic!("test allocation should fit"),
    }
}

fn checkpoint(manager: &VllmKvManager, lease: &BlockRequestLease, block: usize) -> bool {
    manager.state_cache.as_ref().unwrap().has_snapshot(
        &manager.pool,
        lease.entries[block - 1].identity.sequence_hash.unwrap(),
    )
}

fn seed_key(manager: &mut VllmKvManager, key: CacheKey) -> BlockCopyId {
    let mut reservation = manager.pool.reserve(&[], 1).unwrap().reservation;
    let copy = manager.pool.allocate_private(&mut reservation);
    manager.pool.cache_private_key(copy, key);
    manager.pool.release(copy);
    manager.pool.cancel(reservation);
    copy
}

fn clear_reclaimable_and_assert_empty(manager: &mut VllmKvManager) {
    assert_eq!(manager.pool.num_active(), 0);
    let all = manager.pool.reserve(&[], manager.pool.capacity()).unwrap();
    manager.pool.cancel(all.reservation);
    assert_eq!(manager.pool.num_inactive(), 0);
    assert_eq!(manager.pool.num_active(), 0);
}

#[test]
fn eight_block_budget_fits_two_requests_then_reuses_released_capacity() {
    // Manual fixture: 64 tokens * 16 bytes = 1024 bytes/block;
    // 1500 state bytes round to two blocks; 8192 bytes provide eight blocks.
    let mut manager = manager(8, 64, 2, false);
    let mut leases = Vec::new();
    for id in 1..=2 {
        let owner = Uuid::from_u128(id);
        let (mut sequence, mut lease) = request(owner, 128, 64, false, 0);
        assert_eq!(
            allocated(manager.allocate_lease(owner, &mut lease, 128, 0)),
            4
        );
        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 128);
        assert_eq!(lease.state.as_ref().unwrap().working.len(), 2);
        assert_eq!(manager.num_inactive_blocks(), 0);
        leases.push((owner, lease));
    }
    assert_eq!(manager.num_active_blocks(), 8);
    let third = Uuid::from_u128(3);
    let (sequence, mut lease) = request(third, 128, 64, false, 0);
    assert_eq!(
        manager
            .get_lease_prefill_cost(&sequence, &lease)
            .cached_tokens,
        0
    );
    assert!(matches!(
        manager.allocate_lease(third, &mut lease, 128, 0),
        NativeAllocation::CapacityExhausted
    ));
    assert!(lease.state.is_none());
    assert_eq!(lease.resident_block_count(), 0);
    assert_eq!(lease.allocated_tokens(), 0);
    assert_eq!(manager.num_active_blocks(), 8);

    let (owner, finished) = leases.pop().unwrap();
    manager.finish_lease(owner, finished);
    allocated(manager.allocate_lease(third, &mut lease, 128, 0));
    assert_eq!(manager.num_active_blocks(), 8);
    manager.finish_lease(third, lease);
    for (owner, lease) in leases {
        manager.finish_lease(owner, lease);
    }
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn failed_growth_preserves_existing_work_and_checkpoint() {
    let mut manager = manager(4, 4, 1, true);
    let owner = Uuid::from_u128(1);
    let (mut sequence, mut lease) = request(owner, 16, 4, true, 0);
    allocated(manager.allocate_lease(owner, &mut lease, 4, 0));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 4);
    manager.begin_step();
    let before_work = lease.state.as_ref().unwrap().working.clone();
    let before_token = lease.entries[0].copy;
    let before_counts = (manager.num_active_blocks(), manager.num_inactive_blocks());

    assert!(matches!(
        manager.allocate_lease(owner, &mut lease, 16, 0),
        NativeAllocation::CapacityExhausted
    ));
    assert_eq!(lease.allocated_tokens(), 4);
    assert_eq!(lease.resident_block_count(), 1);
    assert_eq!(lease.entries[0].copy, before_token);
    assert_eq!(lease.state.as_ref().unwrap().working, before_work);
    assert_eq!(lease.state.as_ref().unwrap().computed_tokens, 4);
    assert!(checkpoint(&manager, &lease, 1));
    assert_eq!(
        (manager.num_active_blocks(), manager.num_inactive_blocks()),
        before_counts
    );
    manager.finish_lease(owner, lease);
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn failed_restore_keeps_cached_sources_unpinned_and_new_lease_empty() {
    let mut manager = manager(5, 4, 2, true);
    let owner = Uuid::from_u128(1);
    let (sequence, mut lease) = request(owner, 4, 4, true, 0);
    let prefix = lease.entries[0].identity.sequence_hash.unwrap();
    seed_key(&mut manager, CacheKey::Token(prefix));
    for slot in 0..2 {
        seed_key(&mut manager, CacheKey::State { prefix, slot });
    }
    let mut reservation = manager.pool.reserve(&[], 1).unwrap().reservation;
    let blocker = manager.pool.allocate_private(&mut reservation);
    manager.pool.cancel(reservation);
    assert_eq!(
        manager
            .get_lease_prefill_cost(&sequence, &lease)
            .cached_tokens,
        4
    );
    assert!(matches!(
        manager.allocate_lease(owner, &mut lease, 4, 1),
        NativeAllocation::CapacityExhausted
    ));
    assert_eq!(lease.allocated_tokens(), 0);
    assert_eq!(lease.resident_block_count(), 0);
    assert!(lease.state.is_none());
    assert!(!manager.pool.prefix_hit(prefix).unwrap().is_active);
    for slot in 0..2 {
        assert!(
            !manager
                .pool
                .key_hit(CacheKey::State { prefix, slot })
                .unwrap()
                .is_active
        );
    }
    assert_eq!(manager.num_active_blocks(), 1);
    assert_eq!(manager.num_inactive_blocks(), 3);
    manager.pool.release(blocker);
    allocated(manager.allocate_lease(owner, &mut lease, 4, 1));
    manager.finish_lease(owner, lease);
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn joint_hit_uses_existing_checkpoint_below_token_upper_bound() {
    let mut manager = manager(20, 4, 2, true);
    let (sequence, lease) = request(Uuid::from_u128(1), 20, 4, true, 0);
    for entry in &lease.entries[..4] {
        seed_key(
            &mut manager,
            CacheKey::Token(entry.identity.sequence_hash.unwrap()),
        );
    }
    // Token cache reaches 16; the two saved states are at 8 and 20.
    for block in [2, 5] {
        let prefix = lease.entries[block - 1].identity.sequence_hash.unwrap();
        for slot in 0..2 {
            seed_key(&mut manager, CacheKey::State { prefix, slot });
        }
    }
    let cost = manager.get_lease_prefill_cost(&sequence, &lease);
    assert_eq!(cost.cached_tokens, 8);
    assert_eq!(cost.new_tokens, 12);
    assert_eq!(cost.new_blocks, 3);
    assert_eq!(cost.active_cached_tokens, 0);
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn token_hash_is_not_a_state_and_partial_state_is_not_restorable() {
    let mut manager = manager(16, 4, 2, true);
    let (sequence, lease) = request(Uuid::from_u128(1), 20, 4, true, 0);
    for entry in &lease.entries {
        seed_key(
            &mut manager,
            CacheKey::Token(entry.identity.sequence_hash.unwrap()),
        );
    }
    assert_eq!(
        manager
            .get_lease_prefill_cost(&sequence, &lease)
            .cached_tokens,
        0
    );
    let prefix = lease.entries[4].identity.sequence_hash.unwrap();
    seed_key(&mut manager, CacheKey::State { prefix, slot: 0 });
    assert!(!checkpoint(&manager, &lease, 5));
    assert_eq!(
        manager
            .get_lease_prefill_cost(&sequence, &lease)
            .cached_tokens,
        0
    );
    seed_key(&mut manager, CacheKey::State { prefix, slot: 1 });
    assert_eq!(
        manager
            .get_lease_prefill_cost(&sequence, &lease)
            .cached_tokens,
        20
    );
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn later_state_cannot_restore_an_earlier_fork_or_logits_boundary() {
    let mut manager = manager(16, 4, 1, true);
    let owner = Uuid::from_u128(1);
    let (mut original, mut original_lease) = request(owner, 12, 4, true, 0);
    allocated(manager.allocate_lease(owner, &mut original_lease, 12, 0));
    manager.finalize_lease_computed_prefix(owner, &mut original, &mut original_lease, 0, 12);
    manager.begin_step();
    assert!(checkpoint(&manager, &original_lease, 3));
    assert!(!checkpoint(&manager, &original_lease, 2));
    manager.finish_lease(owner, original_lease);
    manager.begin_step();

    let (fork, fork_lease) = request(Uuid::from_u128(2), 8, 4, true, 0);
    assert_eq!(
        manager
            .get_lease_prefill_cost(&fork, &fork_lease)
            .cached_tokens,
        0
    );
    let (needs_logits, logits_lease) = request(Uuid::from_u128(3), 12, 4, true, 1);
    assert_eq!(
        manager
            .get_lease_prefill_cost(&needs_logits, &logits_lease)
            .cached_tokens,
        0
    );
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn unaligned_completion_does_not_fabricate_state_at_previous_boundary() {
    let mut manager = manager(12, 4, 1, true);
    let owner = Uuid::from_u128(1);
    let (mut sequence, mut lease) = request(owner, 10, 4, true, 0);
    allocated(manager.allocate_lease(owner, &mut lease, 10, 0));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 10);
    assert_eq!(lease.state.as_ref().unwrap().computed_tokens, 10);
    assert!(!checkpoint(&manager, &lease, 1));
    assert!(!checkpoint(&manager, &lease, 2));
    assert_eq!(manager.num_inactive_blocks(), 0);
    manager.finish_lease(owner, lease);
    let (earlier, earlier_lease) = request(Uuid::from_u128(2), 8, 4, true, 0);
    assert_eq!(
        manager
            .get_lease_prefill_cost(&earlier, &earlier_lease)
            .cached_tokens,
        0
    );
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn restored_requests_have_distinct_work_without_overwriting_shared_checkpoint() {
    let mut manager = manager(20, 4, 2, true);
    let seed = Uuid::from_u128(1);
    let (mut sequence, mut seed_lease) = request(seed, 8, 4, true, 0);
    allocated(manager.allocate_lease(seed, &mut seed_lease, 8, 0));
    manager.finalize_lease_computed_prefix(seed, &mut sequence, &mut seed_lease, 0, 8);
    manager.finish_lease(seed, seed_lease);
    manager.begin_step();

    let mut readers = Vec::new();
    for id in [2, 3] {
        let owner = Uuid::from_u128(id);
        let (sequence, mut lease) = request(owner, 12, 4, true, 0);
        assert_eq!(
            manager
                .get_lease_prefill_cost(&sequence, &lease)
                .cached_tokens,
            8
        );
        allocated(manager.allocate_lease(owner, &mut lease, 12, 2));
        assert_eq!(lease.state.as_ref().unwrap().computed_tokens, 8);
        assert!(
            lease
                .state
                .as_ref()
                .unwrap()
                .working
                .iter()
                .all(|&id| manager.pool.is_private(id))
        );
        readers.push((owner, sequence, lease));
    }
    let first_work = readers[0].2.state.as_ref().unwrap().working.clone();
    let second_work = &readers[1].2.state.as_ref().unwrap().working;
    assert!(first_work.iter().all(|id| !second_work.contains(id)));
    assert_eq!(readers[0].2.entries[0].copy, readers[1].2.entries[0].copy);
    let (owner, sequence, lease) = &mut readers[0];
    manager.finalize_lease_computed_prefix(*owner, sequence, lease, 8, 12);
    assert!(checkpoint(&manager, &readers[1].2, 2));
    assert_eq!(readers[1].2.state.as_ref().unwrap().computed_tokens, 8);
    assert_eq!(manager.num_active_blocks(), 8); // Four token copies + two working states.
    for (owner, _, lease) in readers {
        manager.finish_lease(owner, lease);
    }
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn state_eviction_changes_joint_hit_while_referenced_token_kv_survives() {
    let mut manager = manager(5, 4, 1, true);
    let owner = Uuid::from_u128(1);
    let (mut sequence, mut lease) = request(owner, 8, 4, true, 4);
    allocated(manager.allocate_lease(owner, &mut lease, 8, 0));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 8);
    manager.begin_step();
    let (_, opened) = sequence.generate_token();
    assert!(opened);
    lease.append_partial();
    allocated(manager.allocate_lease(owner, &mut lease, 9, 0));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 8, 9);
    manager.begin_step();
    let (query, query_lease) = request(Uuid::from_u128(2), 8, 4, true, 0);
    assert_eq!(
        manager
            .get_lease_prefill_cost(&query, &query_lease)
            .cached_tokens,
        8
    );
    // Three token blocks + one working state are active. The preserved old
    // state is the single inactive block, now evicted by this reservation.
    let pressure = manager.pool.reserve(&[], 1).unwrap();
    assert!(pressure.removed.is_empty());
    assert!(!checkpoint(&manager, &lease, 2));
    for entry in &lease.entries[..2] {
        assert!(
            manager
                .pool
                .prefix_hit(entry.identity.sequence_hash.unwrap())
                .is_some()
        );
    }
    assert_eq!(
        manager
            .get_lease_prefill_cost(&query, &query_lease)
            .cached_tokens,
        0
    );
    manager.pool.cancel(pressure.reservation);
    manager.finish_lease(owner, lease);
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn long_decode_keeps_one_working_state_and_one_latest_checkpoint() {
    let mut manager = manager(64, 4, 2, true);
    let owner = Uuid::from_u128(1);
    let (mut sequence, mut lease) = request(owner, 4, 4, true, 80);
    allocated(manager.allocate_lease(owner, &mut lease, 4, 0));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 4);
    let initial_work = lease.state.as_ref().unwrap().working.clone();
    for before in 4..84 {
        manager.begin_step();
        let (_, opened) = sequence.generate_token();
        if opened {
            lease.append_partial();
        }
        allocated(manager.allocate_lease(owner, &mut lease, before + 1, 0));
        manager.finalize_lease_computed_prefix(
            owner,
            &mut sequence,
            &mut lease,
            before,
            before + 1,
        );
        assert_eq!(lease.state.as_ref().unwrap().working, initial_work);
        assert_eq!(lease.state.as_ref().unwrap().computed_tokens, before + 1);
        assert_eq!(
            manager.num_active_blocks(),
            lease.resident_block_count() + 2
        );
        assert_eq!(
            manager.num_inactive_blocks(),
            if (before + 1) % 4 == 0 { 0 } else { 2 }
        );
    }
    manager.begin_step();
    assert!(checkpoint(&manager, &lease, 21));
    for previous in 1..21 {
        assert!(!checkpoint(&manager, &lease, previous));
    }
    manager.finish_lease(owner, lease);
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn preemption_and_requeue_release_work_and_restore_saved_state() {
    let mut manager = manager(12, 4, 2, true);
    let owner = Uuid::from_u128(1);
    let (mut sequence, mut lease) = request(owner, 8, 4, true, 0);
    allocated(manager.allocate_lease(owner, &mut lease, 8, 0));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 8);
    manager.begin_step(); // The previous pass completed before preemption.
    let old_work = lease.state.as_ref().unwrap().working.clone();
    manager.preempt_lease(owner, &mut lease);
    assert_eq!(manager.num_active_blocks(), 0);
    assert_eq!(lease.allocated_tokens(), 0);
    assert_eq!(lease.resident_block_count(), 0);
    assert!(lease.state.as_ref().unwrap().working.is_empty());
    assert_eq!(lease.state.as_ref().unwrap().computed_tokens, 0);
    manager.begin_step();
    let cost = manager.get_lease_prefill_cost(&sequence, &lease);
    assert_eq!(cost.cached_tokens, 8);
    allocated(manager.allocate_lease(owner, &mut lease, 8, 2));
    assert!(
        lease
            .state
            .as_ref()
            .unwrap()
            .working
            .iter()
            .all(|id| !old_work.contains(id))
    );
    assert_eq!(lease.state.as_ref().unwrap().computed_tokens, 8);
    manager.finish_lease(owner, lease);
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn same_pass_preemption_discards_the_unexecuted_checkpoint() {
    let mut manager = manager(12, 4, 1, true);
    let owner = Uuid::from_u128(1);
    let (mut sequence, mut lease) = request(owner, 8, 4, true, 0);
    manager.begin_step();
    allocated(manager.allocate_lease(owner, &mut lease, 8, 0));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 8);
    manager.preempt_lease(owner, &mut lease);
    manager.begin_step();
    assert_eq!(
        manager
            .get_lease_prefill_cost(&sequence, &lease)
            .cached_tokens,
        0
    );
    manager.finish_lease(owner, lease);
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn full_pool_publishes_working_state_without_a_second_copy() {
    let mut manager = manager(4, 4, 2, true);
    let owner = Uuid::from_u128(1);
    let (mut sequence, mut lease) = request(owner, 8, 4, true, 0);
    allocated(manager.allocate_lease(owner, &mut lease, 8, 0));
    let working = lease.state.as_ref().unwrap().working.clone();
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 8);
    assert_eq!(lease.state.as_ref().unwrap().working, working);
    assert_eq!(manager.num_active_blocks(), 4);
    assert_eq!(manager.num_inactive_blocks(), 0);
    manager.begin_step();
    assert!(checkpoint(&manager, &lease, 2));
    let prefix = lease.entries[1].identity.sequence_hash.unwrap();
    manager.finish_lease(owner, lease);
    assert_eq!(manager.num_active_blocks(), 0);
    assert_eq!(manager.num_inactive_blocks(), 4);
    assert!(
        manager
            .state_cache
            .as_ref()
            .unwrap()
            .has_snapshot(&manager.pool, prefix)
    );
    // The cached state after completion is the original working allocation.
    for id in working {
        assert!(manager.pool.discard_inactive_state(id));
    }
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn newly_published_checkpoint_becomes_restorable_only_next_step() {
    let mut manager = manager(12, 4, 1, true);
    let owner = Uuid::from_u128(1);
    let (mut sequence, mut lease) = request(owner, 8, 4, true, 0);
    allocated(manager.allocate_lease(owner, &mut lease, 8, 0));
    manager.begin_step();
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 8);
    let (query, query_lease) = request(Uuid::from_u128(2), 8, 4, true, 0);
    assert_eq!(
        manager
            .get_lease_prefill_cost(&query, &query_lease)
            .cached_tokens,
        0
    );
    manager.begin_step();
    assert_eq!(
        manager
            .get_lease_prefill_cost(&query, &query_lease)
            .cached_tokens,
        8
    );
    manager.finish_lease(owner, lease);
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn mtp_bound_does_not_drop_an_already_earlier_sparse_checkpoint() {
    let mut manager = manager(32, 4, 1, true);
    manager.enable_state_cache(1, true);
    let (query, lease) = request(Uuid::from_u128(1), 24, 4, true, 4);
    for entry in &lease.entries[..5] {
        seed_key(
            &mut manager,
            CacheKey::Token(entry.identity.sequence_hash.unwrap()),
        );
    }
    let prefix = lease.entries[1].identity.sequence_hash.unwrap();
    seed_key(&mut manager, CacheKey::State { prefix, slot: 0 });
    // Token hit=20, MTP upper bound=16, actual snapshot=8. The entire
    // [8,24) suffix is already recomputed; rewinding the snapshot again is wrong.
    assert_eq!(
        manager.get_lease_prefill_cost(&query, &lease).cached_tokens,
        8
    );
    manager.finish_lease(lease.owner(), lease);
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn duplicate_snapshot_stays_hidden_if_the_older_copy_is_evicted_in_this_pass() {
    let mut manager = manager(16, 4, 1, true);
    let owner = Uuid::from_u128(1);
    let (mut sequence, mut lease) = request(owner, 8, 4, true, 0);
    let prefix = lease.entries[1].identity.sequence_hash.unwrap();
    let older = seed_key(&mut manager, CacheKey::State { prefix, slot: 0 });
    manager.begin_step();
    allocated(manager.allocate_lease(owner, &mut lease, 8, 0));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 8);
    assert!(manager.pool.discard_inactive_state(older));
    assert!(!checkpoint(&manager, &lease, 2));
    manager.begin_step();
    assert!(checkpoint(&manager, &lease, 2));
    manager.finish_lease(owner, lease);
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn continued_write_requires_checkpoint_capacity_before_mutating_state() {
    for capacity in [5, 7] {
        let mut manager = manager(capacity, 4, 2, true);
        let owner = Uuid::from_u128(1);
        let (mut sequence, mut lease) = request(owner, 8, 4, true, 4);
        allocated(manager.allocate_lease(owner, &mut lease, 8, 0));
        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 8);
        manager.begin_step();
        let working = lease.state.as_ref().unwrap().working.clone();
        assert!(working.iter().all(|&id| !manager.pool.is_private(id)));
        let (_, opened) = sequence.generate_token();
        assert!(opened);
        lease.append_partial();
        if capacity == 5 {
            assert!(matches!(
                manager.allocate_lease(owner, &mut lease, 9, 0),
                NativeAllocation::CapacityExhausted
            ));
            assert_eq!(lease.allocated_tokens(), 8);
            assert_eq!(lease.state.as_ref().unwrap().computed_tokens, 8);
            assert_eq!(lease.state.as_ref().unwrap().working, working);
            assert!(working.iter().all(|&id| !manager.pool.is_private(id)));
            assert!(checkpoint(&manager, &lease, 2));
            assert_eq!(manager.num_active_blocks(), 4);
            manager.finish_lease(owner, lease);
            clear_reclaimable_and_assert_empty(&mut manager);
            continue;
        }
        allocated(manager.allocate_lease(owner, &mut lease, 9, 0));
        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 8, 9);
        assert_eq!(lease.state.as_ref().unwrap().computed_tokens, 9);
        assert_eq!(lease.state.as_ref().unwrap().working, working);
        assert!(working.iter().all(|&id| manager.pool.is_private(id)));
        assert_eq!(manager.num_active_blocks(), 5);
        assert_eq!(
            manager.num_inactive_blocks(),
            if capacity == 7 { 2 } else { 0 }
        );
        // The old-prefix copy itself is new in this pass, so it is not a
        // same-pass hit even though the preserved prefix was already computed.
        assert!(!checkpoint(&manager, &lease, 2));
        manager.begin_step();
        assert_eq!(checkpoint(&manager, &lease, 2), capacity == 7);
        manager.finish_lease(owner, lease);
        clear_reclaimable_and_assert_empty(&mut manager);
    }
}

#[test]
fn aligned_progress_reserves_transition_peak_before_replacing_checkpoint() {
    let mut manager = manager(7, 4, 2, true);
    // Another request holds the two blocks needed for the state transition.
    let mut blocker = manager.pool.reserve(&[], 2).unwrap().reservation;
    let held = (0..2)
        .map(|_| manager.pool.allocate_private(&mut blocker))
        .collect::<Vec<_>>();
    manager.pool.cancel(blocker);
    let owner = Uuid::from_u128(1);
    let (mut sequence, mut lease) = request(owner, 12, 4, true, 0);
    allocated(manager.allocate_lease(owner, &mut lease, 8, 0));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 8);
    manager.begin_step();
    let working = lease.state.as_ref().unwrap().working.clone();
    assert!(matches!(
        manager.allocate_lease(owner, &mut lease, 12, 0),
        NativeAllocation::CapacityExhausted
    ));
    assert!(checkpoint(&manager, &lease, 2));
    assert_eq!(lease.allocated_tokens(), 8);
    for id in held {
        manager.pool.release(id);
    }
    allocated(manager.allocate_lease(owner, &mut lease, 12, 0));
    assert_eq!(manager.num_active_blocks(), 7);
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 8, 12);
    manager.begin_step();
    assert_eq!(lease.state.as_ref().unwrap().working, working);
    assert!(!checkpoint(&manager, &lease, 2));
    assert!(checkpoint(&manager, &lease, 3));
    assert_eq!(manager.num_active_blocks(), 5);
    assert_eq!(manager.num_inactive_blocks(), 0);
    manager.finish_lease(owner, lease);
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn active_state_source_is_not_charged_twice_during_restore_admission() {
    let mut manager = manager(7, 4, 2, true);
    let owner = Uuid::from_u128(1);
    let (mut sequence, mut lease) = request(owner, 8, 4, true, 0);
    allocated(manager.allocate_lease(owner, &mut lease, 8, 0));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 8);
    manager.begin_step();
    let (query, mut reader) = request(Uuid::from_u128(2), 12, 4, true, 0);
    let cost = manager.get_lease_prefill_cost(&query, &reader);
    assert_eq!(cost.cached_tokens, 8);
    assert_eq!(manager.state_restore_overhead(&reader, 8), 2);
    allocated(manager.allocate_lease(reader.owner(), &mut reader, 12, 2));
    assert_eq!(manager.num_active_blocks(), 7);
    manager.finish_lease(reader.owner(), reader);
    manager.finish_lease(owner, lease);
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn retracted_preservation_copy_does_not_resurrect_an_old_prefix() {
    let mut manager = manager(7, 4, 2, true);
    let owner = Uuid::from_u128(1);
    let (mut sequence, mut lease) = request(owner, 8, 4, true, 4);
    allocated(manager.allocate_lease(owner, &mut lease, 8, 0));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 8);
    manager.begin_step();
    let (_, opened) = sequence.generate_token();
    assert!(opened);
    lease.append_partial();
    allocated(manager.allocate_lease(owner, &mut lease, 9, 0));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 8, 9);
    manager.preempt_lease(owner, &mut lease);
    manager.begin_step();
    assert!(!checkpoint(&manager, &lease, 2));
    manager.finish_lease(owner, lease);
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn prepared_state_copy_is_retracted_before_compute_without_leaking_refs() {
    let mut manager = manager(7, 4, 2, true);
    let owner = Uuid::from_u128(1);
    let (mut sequence, mut lease) = request(owner, 12, 4, true, 0);
    allocated(manager.allocate_lease(owner, &mut lease, 12, 0));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 8);
    manager.begin_step();
    // Token capacity already covers this target, but state still needs a copy.
    allocated(manager.allocate_lease(owner, &mut lease, 9, 0));
    assert_eq!(manager.num_active_blocks(), 7);
    assert_eq!(lease.state.as_ref().unwrap().computed_tokens, 8);
    manager.preempt_lease(owner, &mut lease);
    manager.begin_step();
    assert!(!checkpoint(&manager, &lease, 2));
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn manager_decode_requirements_distinguish_sampled_tail_from_state_write() {
    for capacity in [4, 5] {
        let mut manager = manager(capacity, 4, 1, true);
        let owner = Uuid::from_u128(501);
        let (mut sequence, mut lease) = request(owner, 8, 4, true, 4);
        allocated(manager.allocate_lease(owner, &mut lease, 8, 0));
        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 8);
        manager.begin_step();
        // Three blocks are occupied. One sampled token needs only one token
        // block; computing a second output requires another state buffer.
        assert_eq!(
            manager.decode_requirement(&lease, 8, 1),
            AllocationRequirement::Blocks(1)
        );
        assert_eq!(
            manager.decode_requirement(&lease, 8, 2),
            if capacity == 4 {
                AllocationRequirement::Impossible
            } else {
                AllocationRequirement::Blocks(2)
            }
        );
        assert_eq!(manager.num_active_blocks(), 3);
        assert_eq!(lease.computed_state_tokens(), Some(8));
        assert!(checkpoint(&manager, &lease, 2));
        manager.finish_lease(owner, lease);
        clear_reclaimable_and_assert_empty(&mut manager);
    }
}

// Modified behavioral fixtures for vLLM v0.29.0 align/default retention.
// Copyright contributors to the vLLM project. Apache-2.0.
// https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/v1/core/single_type_kv_cache_manager.py
fn fine_request(
    owner: Uuid,
    tokens: usize,
    block: usize,
    unit: usize,
    output: usize,
) -> (RequestSequence, BlockRequestLease) {
    let (sequence, mut lease) = request(owner, tokens, block, true, output);
    lease.configure_prefix_hashes(&(0..tokens as u32).collect::<Vec<_>>(), unit);
    (sequence, lease)
}

fn fine_checkpoint(manager: &VllmKvManager, lease: &BlockRequestLease, tokens: usize) -> bool {
    lease
        .state_hash(tokens, manager.block_size)
        .is_some_and(|h| {
            manager
                .state_cache
                .as_ref()
                .unwrap()
                .has_snapshot(&manager.pool, h)
        })
}

#[test]
fn kda_physical_pages_retain_and_restore_both_real_checkpoints() {
    // A=16; each physical KV page and each full state occupies 1536*A bytes.
    let mut manager = manager(64, 1536, 1, true);
    manager.configure_prefix_match_unit(Some(128));
    let owner = Uuid::from_u128(601);
    let (mut sequence, mut lease) = fine_request(owner, 24300, 1536, 128, 400);
    let mut before = 0;
    for end in [7680, 15360, 23040, 24192, 24300] {
        allocated(manager.allocate_lease(owner, &mut lease, end, 0));
        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, before, end);
        manager.begin_step();
        let active_states = match end {
            7680 | 24300 => 1,
            15360 | 23040 | 24192 => 2,
            _ => unreachable!(),
        };
        assert_eq!(
            manager.num_active_blocks(),
            end.div_ceil(1536) + active_states
        );
        assert!(!fine_checkpoint(&manager, &lease, 7680));
        assert!(!fine_checkpoint(&manager, &lease, 15360));
        assert!(!fine_checkpoint(&manager, &lease, 1536));
        if end >= 24192 {
            assert!(fine_checkpoint(&manager, &lease, 23040));
            assert!(fine_checkpoint(&manager, &lease, 24192));
        }
        before = end;
    }
    assert_eq!(manager.num_active_blocks(), 17); // 16 KV pages + one working state
    assert_eq!(manager.num_inactive_blocks(), 2); // two distinct state snapshots
    for _ in 0..400 {
        let (_, opened) = sequence.generate_token();
        if opened {
            lease.append_partial();
        }
        let end = sequence.len();
        allocated(manager.allocate_lease(owner, &mut lease, end, 0));
        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, before, end);
        before = end;
        manager.begin_step();
        match end {
            24576 => assert_eq!(
                manager.num_active_blocks() + manager.num_inactive_blocks(),
                16 + 3
            ),
            24577 => assert_eq!(
                manager.num_active_blocks() + manager.num_inactive_blocks(),
                17 + 4
            ),
            24578 => assert_eq!(
                manager.num_active_blocks() + manager.num_inactive_blocks(),
                17 + 3
            ),
            _ => {}
        }
    }
    assert_eq!(manager.num_inactive_blocks(), 2);
    assert!(!fine_checkpoint(&manager, &lease, 24576));
    manager.finish_lease(owner, lease);
    for (length, expected) in [(23700, 23040), (24300, 24192)] {
        let owner = Uuid::from_u128(length as u128);
        let (query, mut reader) = fine_request(owner, length, 1536, 128, 0);
        let cost = manager.get_lease_prefill_cost(&query, &reader);
        assert_eq!(cost.cached_tokens, expected);
        allocated(manager.allocate_lease(owner, &mut reader, length, expected / 128));
        assert_eq!(reader.computed_state_tokens(), Some(expected));
        assert_eq!(reader.state.as_ref().unwrap().working.len(), 1);
        assert_eq!(reader.resident_block_count(), length.div_ceil(1536));
        manager.finish_lease(owner, reader);
    }
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn kda_partial_page_can_match_when_tokens_after_the_hit_diverge() {
    let mut manager = manager(16, 6, 1, true);
    manager.configure_prefix_match_unit(Some(2));
    let owner = Uuid::from_u128(602);
    let (mut sequence, mut lease) = fine_request(owner, 17, 6, 2, 0);
    let mut before = 0;
    for end in [6, 12, 16, 17] {
        allocated(manager.allocate_lease(owner, &mut lease, end, 0));
        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, before, end);
        manager.begin_step();
        before = end;
    }
    let source = lease.entries[2].copy.unwrap();
    manager.finish_lease(owner, lease);
    let mut tokens: Vec<u32> = (0..19).collect();
    tokens[16] = 999;
    let (mut query, ids) = RequestSequence::new(tokens.clone(), 0, 0, 6, true, true, false, None);
    let owner = Uuid::from_u128(603);
    let mut reader = BlockRequestLease::new(owner, ids);
    reader.configure_prefix_hashes(&tokens, 2);
    assert_eq!(
        manager
            .get_lease_prefill_cost(&query, &reader)
            .cached_tokens,
        16
    );
    allocated(manager.allocate_lease(owner, &mut reader, 18, 8));
    assert_ne!(
        reader.entries[2].copy.unwrap(),
        source,
        "partial-page extension needs its own physical page"
    );
    manager.finalize_lease_computed_prefix(owner, &mut query, &mut reader, 16, 18);
    manager.begin_step();
    manager.finish_lease(owner, reader);
    let (original, reader) = fine_request(Uuid::from_u128(604), 17, 6, 2, 0);
    assert_eq!(
        manager
            .get_lease_prefill_cost(&original, &reader)
            .cached_tokens,
        16
    );
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn kda_tail_preemption_retracts_new_state_and_keeps_full_checkpoint() {
    for preempt_at in [16, 17] {
        let mut manager = manager(16, 6, 1, true);
        manager.configure_prefix_match_unit(Some(2));
        let owner = Uuid::from_u128(605);
        let (mut sequence, mut lease) = fine_request(owner, 17, 6, 2, 0);
        let mut before = 0;
        for end in [6, 12, 16, 17] {
            allocated(manager.allocate_lease(owner, &mut lease, end, 0));
            manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, before, end);
            if end == preempt_at {
                break;
            }
            manager.begin_step();
            before = end;
        }
        manager.preempt_lease(owner, &mut lease);
        manager.begin_step();
        assert!(fine_checkpoint(&manager, &lease, 12));
        assert!(!fine_checkpoint(&manager, &lease, 16));
        assert_eq!(manager.num_active_blocks(), 0);
        let (q, r) = fine_request(Uuid::from_u128(606), 17, 6, 2, 0);
        assert_eq!(manager.get_lease_prefill_cost(&q, &r).cached_tokens, 12);
        manager.finish_lease(owner, lease);
        clear_reclaimable_and_assert_empty(&mut manager);
    }
}

#[test]
fn kda_pool_pressure_reclaims_state_and_all_partial_token_aliases() {
    let mut manager = manager(6, 6, 1, true);
    manager.configure_prefix_match_unit(Some(2));
    let owner = Uuid::from_u128(607);
    let (mut sequence, mut lease) = fine_request(owner, 17, 6, 2, 0);
    let mut before = 0;
    for end in [6, 12, 16, 17] {
        allocated(manager.allocate_lease(owner, &mut lease, end, 0));
        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, before, end);
        manager.begin_step();
        before = end;
    }
    assert_eq!(manager.num_active_blocks(), 4);
    assert_eq!(manager.num_inactive_blocks(), 2);
    let pressure = manager.pool.reserve(&[], 1).unwrap();
    assert!(!fine_checkpoint(&manager, &lease, 12));
    assert!(fine_checkpoint(&manager, &lease, 16));
    manager.pool.cancel(pressure.reservation);
    manager.finish_lease(owner, lease);
    clear_reclaimable_and_assert_empty(&mut manager);
    let (q, r) = fine_request(Uuid::from_u128(608), 17, 6, 2, 0);
    assert_eq!(manager.get_lease_prefill_cost(&q, &r).cached_tokens, 0);
}

#[test]
fn kda_exact_full_prompt_requires_an_actually_executed_replay_boundary() {
    for ends in [vec![6, 12, 18, 24], vec![12, 24]] {
        let mut manager = manager(16, 6, 1, true);
        manager.configure_prefix_match_unit(Some(2));
        let owner = Uuid::from_u128(609);
        let (mut sequence, mut lease) = fine_request(owner, 24, 6, 2, 0);
        let mut before = 0;
        for &end in &ends {
            allocated(manager.allocate_lease(owner, &mut lease, end, 0));
            manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, before, end);
            manager.begin_step();
            before = end;
        }
        let (q, r) = fine_request(Uuid::from_u128(610), 24, 6, 2, 1);
        assert_eq!(
            manager.get_lease_prefill_cost(&q, &r).cached_tokens,
            if ends.contains(&18) { 18 } else { 0 }
        );
        manager.finish_lease(owner, lease);
        clear_reclaimable_and_assert_empty(&mut manager);
    }
}

#[test]
fn kda_partial_restore_accounts_for_source_page_and_private_copy_atomically() {
    let mut manager = manager(5, 6, 1, true);
    manager.configure_prefix_match_unit(Some(2));
    let owner = Uuid::from_u128(611);
    let (mut sequence, mut lease) = fine_request(owner, 17, 6, 2, 0);
    let mut before = 0;
    for end in [6, 12, 16, 17] {
        allocated(manager.allocate_lease(owner, &mut lease, end, 0));
        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, before, end);
        manager.begin_step();
        before = end;
    }
    manager.finish_lease(owner, lease);
    let owner = Uuid::from_u128(612);
    let (query, mut reader) = fine_request(owner, 17, 6, 2, 0);
    let cost = manager.get_lease_prefill_cost(&query, &reader);
    assert_eq!(cost.cached_tokens, 16);
    assert_eq!(
        manager.admission_requirement(&reader, 17, &cost, 0),
        AllocationRequirement::Impossible
    );
    let before = (manager.num_active_blocks(), manager.num_inactive_blocks());
    assert!(matches!(
        manager.allocate_lease(owner, &mut reader, 17, 8),
        NativeAllocation::CapacityExhausted
    ));
    assert_eq!(
        (manager.num_active_blocks(), manager.num_inactive_blocks()),
        before
    );
    assert_eq!(reader.allocated_tokens(), 0);
    assert!(reader.state.is_none());
    assert_eq!(
        manager
            .get_lease_prefill_cost(&query, &reader)
            .cached_tokens,
        16
    );
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn kda_unpublished_turnover_source_survives_until_the_next_request_allocation() {
    let mut manager = manager(16, 6, 1, true);
    manager.configure_prefix_match_unit(Some(2));
    let owner = Uuid::from_u128(620);
    let (mut sequence, mut lease) = fine_request(owner, 25, 6, 2, 0);
    allocated(manager.allocate_lease(owner, &mut lease, 6, 0));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 6);
    let first = lease.state.as_ref().unwrap().working[0];
    manager.begin_step();
    allocated(manager.allocate_lease(owner, &mut lease, 12, 0));
    let second = lease.state.as_ref().unwrap().working[0];
    assert_ne!(first, second);
    assert!(manager.pool.is_private(first));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 6, 12);
    assert_eq!(manager.num_active_blocks(), 2 + 2);
    assert!(!fine_checkpoint(&manager, &lease, 6));
    assert!(!fine_checkpoint(&manager, &lease, 12));
    manager.begin_step();
    // A global pass boundary alone does not release the previous state slot.
    assert!(manager.pool.is_private(first));
    assert_eq!(manager.num_active_blocks(), 4);
    allocated(manager.allocate_lease(owner, &mut lease, 18, 0));
    assert!(!manager.pool.is_private(first));
    assert!(manager.pool.is_private(second));
    assert_ne!(second, lease.state.as_ref().unwrap().working[0]);
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 12, 18);
    assert_eq!(manager.num_active_blocks(), 3 + 2);
    manager.preempt_lease(owner, &mut lease);
    manager.begin_step();
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn kda_turnover_pressure_does_not_overwrite_current_state_or_invent_a_checkpoint() {
    let mut manager = manager(4, 6, 1, true);
    manager.configure_prefix_match_unit(Some(2));
    let owner = Uuid::from_u128(621);
    let (mut sequence, mut lease) = fine_request(owner, 18, 6, 2, 0);
    allocated(manager.allocate_lease(owner, &mut lease, 6, 0));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 6);
    manager.begin_step();
    allocated(manager.allocate_lease(owner, &mut lease, 12, 0));
    manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 6, 12);
    assert_eq!(manager.num_active_blocks(), 4);
    assert!(
        manager.pool.reserve(&[], 1).is_none(),
        "old source is still needed this pass"
    );
    manager.begin_step();
    let work = lease.state.as_ref().unwrap().working.clone();
    assert!(!manager.can_compute(&lease, 18, 18)); // KV3 + source1 + dest1 > capacity4
    assert!(matches!(
        manager.allocate_lease(owner, &mut lease, 18, 0),
        NativeAllocation::CapacityExhausted
    ));
    assert_eq!(lease.state.as_ref().unwrap().working, work);
    assert_eq!(lease.computed_state_tokens(), Some(12));
    assert_eq!(lease.allocated_tokens(), 12);
    assert!(fine_checkpoint(&manager, &lease, 12));
    assert!(!fine_checkpoint(&manager, &lease, 18));
    // Native-style cleanup may release a dead older source even if growth fails.
    assert_eq!(manager.num_active_blocks(), 3);
    manager.finish_lease(owner, lease);
    manager.begin_step();
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn kda_full_restore_source_is_owned_until_that_request_advances() {
    let mut manager = manager(16, 6, 1, true);
    manager.configure_prefix_match_unit(Some(2));
    let seed = Uuid::from_u128(622);
    let (mut sequence, mut lease) = fine_request(seed, 17, 6, 2, 0);
    allocated(manager.allocate_lease(seed, &mut lease, 12, 0));
    manager.finalize_lease_computed_prefix(seed, &mut sequence, &mut lease, 0, 12);
    manager.begin_step();
    let hash = lease.state_hash(12, 6).unwrap();
    manager.finish_lease(seed, lease);
    let owner = Uuid::from_u128(623);
    let (mut query, mut reader) = fine_request(owner, 25, 6, 2, 0);
    assert_eq!(
        manager
            .get_lease_prefill_cost(&query, &reader)
            .cached_tokens,
        12
    );
    allocated(manager.allocate_lease(owner, &mut reader, 18, 6));
    manager.finalize_lease_computed_prefix(owner, &mut query, &mut reader, 12, 18);
    assert!(
        manager
            .pool
            .key_hit(CacheKey::State {
                prefix: hash,
                slot: 0
            })
            .unwrap()
            .is_active
    );
    manager.begin_step();
    assert!(
        manager
            .pool
            .key_hit(CacheKey::State {
                prefix: hash,
                slot: 0
            })
            .unwrap()
            .is_active
    );
    allocated(manager.allocate_lease(owner, &mut reader, 24, 0));
    assert!(
        !manager
            .pool
            .key_hit(CacheKey::State {
                prefix: hash,
                slot: 0
            })
            .unwrap()
            .is_active
    );
    manager.finalize_lease_computed_prefix(owner, &mut query, &mut reader, 18, 24);
    manager.finish_lease(owner, reader);
    manager.begin_step();
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn kda_partial_restore_copy_refs_protect_both_kv_and_state_until_step_completion() {
    let mut manager = manager(16, 6, 1, true);
    manager.configure_prefix_match_unit(Some(2));
    let seed = Uuid::from_u128(624);
    let (mut sequence, mut lease) = fine_request(seed, 17, 6, 2, 0);
    let mut before = 0;
    for end in [6, 12, 16, 17] {
        allocated(manager.allocate_lease(seed, &mut lease, end, 0));
        manager.finalize_lease_computed_prefix(seed, &mut sequence, &mut lease, before, end);
        manager.begin_step();
        before = end;
    }
    let hash = lease.state_hash(16, 6).unwrap();
    manager.finish_lease(seed, lease);
    let mut readers = Vec::new();
    for owner in [Uuid::from_u128(625), Uuid::from_u128(626)] {
        let (query, mut reader) = fine_request(owner, 17, 6, 2, 0);
        assert_eq!(
            manager
                .get_lease_prefill_cost(&query, &reader)
                .cached_tokens,
            16
        );
        allocated(manager.allocate_lease(owner, &mut reader, 17, 8));
        readers.push(reader);
    }
    let mut cancelled = readers.pop().unwrap();
    manager.preempt_lease(cancelled.owner(), &mut cancelled);
    assert!(manager.pool.prefix_hit(hash).unwrap().is_active);
    assert!(
        manager
            .pool
            .key_hit(CacheKey::State {
                prefix: hash,
                slot: 0
            })
            .unwrap()
            .is_active
    );
    manager.begin_step();
    assert!(!manager.pool.prefix_hit(hash).unwrap().is_active);
    assert!(
        !manager
            .pool
            .key_hit(CacheKey::State {
                prefix: hash,
                slot: 0
            })
            .unwrap()
            .is_active
    );
    manager.finish_lease(cancelled.owner(), cancelled);
    for reader in readers {
        manager.finish_lease(reader.owner(), reader);
    }
    manager.begin_step();
    clear_reclaimable_and_assert_empty(&mut manager);
}

#[test]
fn kda_alignment_does_not_rotate_state_when_prefix_caching_is_disabled() {
    let mut manager = manager(8, 6, 1, false);
    manager.configure_prefix_match_unit(Some(2));
    let owner = Uuid::from_u128(627);
    let (mut sequence, mut lease) = request(owner, 17, 6, false, 0);
    let mut before = 0;
    let mut working = None;
    for end in [6, 12, 17] {
        allocated(manager.allocate_lease(owner, &mut lease, end, 0));
        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, before, end);
        let id = lease.state.as_ref().unwrap().working[0];
        assert_eq!(*working.get_or_insert(id), id);
        assert_eq!(manager.num_active_blocks(), end.div_ceil(6) + 1);
        assert_eq!(manager.num_inactive_blocks(), 0);
        manager.begin_step();
        before = end;
    }
    manager.finish_lease(owner, lease);
    clear_reclaimable_and_assert_empty(&mut manager);
}
