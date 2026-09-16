// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;

use aisimulate_placement_abi::{
    AdmissionDecisionV1, ByteSliceV1, PlacementBatchResultV1, PlacementCacheSampleV1,
    PlacementDiagnosticV1, PlacementMutationKindV1, PlacementMutationV1, PlacementResultV1,
    PlacementV1, PluginDescriptorV1, PluginVTableV1, StatusV1, WorkerCapacityV1, WorkerTopologyV1,
    validate_descriptor_v1,
};

#[test]
fn placement_batch_preserves_lifecycle_order_and_all_effects() {
    let mutation = PlacementMutationV1::topology_settled(42.0);
    let placement = PlacementV1 {
        request_id: [7; 16],
        worker_id: 3,
        scheduler_id: 9,
        reported_overlap_tokens: 64,
        cache_sample: PlacementCacheSampleV1 {
            flags: PlacementCacheSampleV1::PRESENT,
            overlap_blocks: 2,
            best_available_overlap_blocks: 4,
            isl_blocks: 8,
        },
        placement_replica_id: 1,
        reserved: 0,
    };
    let result = PlacementResultV1 {
        request_id: [7; 16],
        decision: AdmissionDecisionV1::IMMEDIATE,
        reserved: [0; 4],
        placement,
    };
    let batch = PlacementBatchResultV1 {
        struct_size: std::mem::size_of::<PlacementBatchResultV1>() as u32,
        flags: 0,
        applied_mutations: 1,
        pending_count: 0,
        admission_results: (&result).into(),
        released: (&placement).into(),
        diagnostics: (&PlacementDiagnosticV1::EMPTY).into(),
    };

    assert_eq!(mutation.kind, PlacementMutationKindV1::TOPOLOGY_SETTLED);
    assert_eq!(batch.applied_mutations, 1);
    assert_eq!(batch.admission_results.len, 1);
    // Safety: the slice was constructed from the live `placement` above.
    let released = unsafe { &*batch.released.data };
    assert_eq!(released.worker_id, 3);
    assert_eq!(released.cache_sample.overlap_blocks, 2);
}

#[test]
fn placement_dtos_keep_topology_capacity_and_options_at_the_ffi_boundary() {
    let worker = WorkerTopologyV1 {
        worker_id: 3,
        scheduler_ids: Default::default(),
    };
    let capacity = WorkerCapacityV1 {
        worker_id: 3,
        total_kv_blocks: 100,
        available_kv_blocks: 75,
        max_running_requests: 8,
        flags: 0,
        reserved: 0,
    };

    assert_eq!(worker.worker_id, capacity.worker_id);
    assert_eq!(ByteSliceV1::EMPTY.len, 0);
    assert_eq!(std::mem::size_of::<StatusV1>(), 4);
}

#[test]
fn descriptor_validator_rejects_a_missing_lifecycle_batch_operation() {
    let table = PluginVTableV1 {
        struct_size: std::mem::size_of::<PluginVTableV1>() as u32,
        flags: 0,
        create: None,
        apply_batch: None,
        release_results: None,
        release_bytes: None,
        last_error: None,
        destroy: None,
    };
    let descriptor = PluginDescriptorV1 {
        abi_major: PluginDescriptorV1::ABI_MAJOR,
        abi_minor: 0,
        struct_size: std::mem::size_of::<PluginDescriptorV1>() as u32,
        flags: 0,
        capabilities: 0,
        provider_id: c"fixture".as_ptr(),
        vtable: &table,
    };

    assert!(unsafe { validate_descriptor_v1(&descriptor) }.is_err());
    assert!(unsafe { validate_descriptor_v1(std::ptr::null::<c_void>().cast()) }.is_err());
}
