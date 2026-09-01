// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Mutex;

use super::*;
use crate::engine::common::protocols::{DirectRequest, KvEventPublishers, MockEngineArgs};
use crate::engine::host_offload::HostCacheDomain;
use crate::engine::kv_manager::NativeAllocation;
use crate::engine::scheduler::vllm::core::VllmCore;
use crate::engine::trace::TraceCollector;
use crate::engine::{HostOffloadObservation, HostOffloadObservationData, NativeHostOffloadConfig};

fn adapter(capacity_blocks: usize) -> VllmHostOffloadAdapter {
    let domain = HostCacheDomain::new(
        &NativeHostOffloadConfig {
            num_host_blocks: capacity_blocks,
            d2h_bandwidth_gbps: 1.0,
            h2d_bandwidth_gbps: 1.0,
        },
        4,
        250_000,
    )
    .unwrap();
    VllmHostOffloadAdapter::new(domain.bind_rank(0))
}

fn request(
    owner: Uuid,
    tokens: Vec<u32>,
) -> (RequestSequence, BlockRequestLease, VllmHostRequestState) {
    let (sequence, identities) = RequestSequence::new(tokens, 0, 0, 4, true, false, false, None);
    let lease = BlockRequestLease::new(owner, identities);
    let host = VllmHostRequestState::new(&sequence, &lease, 4);
    (sequence, lease, host)
}

fn seed_host(adapter: &mut VllmHostOffloadAdapter, key: HostBlockKey) -> f64 {
    let StoreOutcome::Prepared { transfer_id, .. } =
        adapter.domain.prepare_store(Uuid::nil(), &[key], 0.0)
    else {
        panic!("host seed was prechecked as absent")
    };
    assert_eq!(adapter.domain.submit_prepared_stores(0.0), 1);
    let deadline = adapter.domain.transfer_deadline(transfer_id).unwrap();
    let (completed, _) = adapter.domain.tick(deadline);
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
