// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;

use aisimulate_placement_abi::{
    AdmissionDecisionV1, AdmissionMetadataFormatV1, ByteSliceV1, DescriptorValidationError,
    KvEventV1, KvStorageTierV1, KvStoredBlockV1, MAX_ADMISSION_METADATA_BYTES_V1,
    PlacementAdmissionV1, PlacementBatchResultV1, PlacementCacheSampleV1, PlacementDiagnosticV1,
    PlacementMetadataV1, PlacementMutationKindV1, PlacementMutationPayloadV1,
    PlacementMutationSliceV1, PlacementMutationV1, PlacementResultV1, PlacementV1,
    PluginDescriptorV1, PluginVTableV1, PromptIdentityV1, StatusV1, TokenIdSliceV1,
    WorkerCapacityV1, WorkerTopologyV1, validate_descriptor_v1, validate_mutation_batch_v1,
};

#[test]
fn lossless_kv_observation_records_store_and_remove_identity() {
    let stored_blocks = [KvStoredBlockV1 {
        sequence_hash: 101,
        token_hash: 202,
    }];
    let store = KvEventV1::stored(
        7,
        3,
        KvStorageTierV1::DEVICE,
        11,
        Some(99),
        Some(4),
        &stored_blocks,
    );
    let remove = KvEventV1::removed(7, 3, KvStorageTierV1::DEVICE, 12, &[101]);

    assert_eq!(store.worker_id, 7);
    assert_eq!(store.dp_rank, 3);
    assert_eq!(store.event_id, 11);
    assert!(store.has_parent_hash());
    assert_eq!(store.parent_hash(), Some(99));
    assert!(store.has_start_position());
    assert_eq!(store.start_position(), Some(4));
    assert_eq!(
        store.stored_blocks().expect("stored payload")[0].token_hash,
        202
    );
    assert_eq!(remove.removed_hashes().expect("removed payload"), &[101]);
}

#[test]
fn lossless_kv_capability_requires_the_appended_callback() {
    let table = PluginVTableV1 {
        struct_size: std::mem::size_of::<PluginVTableV1>() as u32,
        flags: 0,
        create: None,
        apply_batch: None,
        release_results: None,
        release_bytes: None,
        last_error: None,
        destroy: None,
        apply_kv_events: Some(apply_kv_events_fixture),
    };

    assert!(table.supports_lossless_kv_events());
}

unsafe extern "C" fn apply_kv_events_fixture(
    _handle: aisimulate_placement_abi::PlacementHandleV1,
    _events: aisimulate_placement_abi::KvEventSliceV1,
    _now_ms: f64,
    _result: *mut PlacementBatchResultV1,
) -> StatusV1 {
    StatusV1::OK
}

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
        apply_kv_events: None,
    };
    let descriptor = PluginDescriptorV1 {
        abi_major: PluginDescriptorV1::ABI_MAJOR,
        abi_minor: PluginDescriptorV1::ABI_MINOR,
        struct_size: std::mem::size_of::<PluginDescriptorV1>() as u32,
        flags: 0,
        capabilities: 0,
        provider_id: c"fixture".as_ptr(),
        vtable: &table,
    };

    assert!(unsafe { validate_descriptor_v1(&descriptor) }.is_err());
    assert!(unsafe { validate_descriptor_v1(std::ptr::null::<c_void>().cast()) }.is_err());
}

#[test]
fn descriptor_validator_rejects_a_provider_before_the_identity_minor() {
    let table = PluginVTableV1 {
        struct_size: std::mem::size_of::<PluginVTableV1>() as u32,
        flags: 0,
        create: None,
        apply_batch: None,
        release_results: None,
        release_bytes: None,
        last_error: None,
        destroy: None,
        apply_kv_events: None,
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

    assert_eq!(
        unsafe { validate_descriptor_v1(&descriptor) },
        Err(DescriptorValidationError::IncompatibleMinor)
    );
}

#[test]
fn admission_prompt_identity_flags_distinguish_omitted_from_explicit_empty() {
    let omitted = admission_with_identity(PromptIdentityV1::OMITTED);
    assert!(valid_admission(omitted));

    let explicitly_empty = admission_with_identity(PromptIdentityV1 {
        flags: PromptIdentityV1::MATERIALIZED_TOKEN_IDS_PRESENT,
        reserved: 0,
        materialized_token_ids: TokenIdSliceV1::default(),
        local_block_hashes: Default::default(),
        sequence_block_hashes: Default::default(),
    });
    assert!(valid_admission(explicitly_empty));

    let tokens = [42_u32];
    let invalid_unflagged_tokens = admission_with_identity(PromptIdentityV1 {
        flags: 0,
        reserved: 0,
        materialized_token_ids: TokenIdSliceV1 {
            data: tokens.as_ptr(),
            len: 1,
        },
        local_block_hashes: Default::default(),
        sequence_block_hashes: Default::default(),
    });
    assert!(!valid_admission(invalid_unflagged_tokens));
}

#[test]
fn admission_metadata_is_tagged_and_bounded() {
    let json = br#"{"tenant":"demo"}"#;
    let valid = PlacementAdmissionV1 {
        metadata: PlacementMetadataV1 {
            format: AdmissionMetadataFormatV1::JSON_UTF8,
            flags: 0,
            bytes: ByteSliceV1 {
                data: json.as_ptr(),
                len: json.len() as u64,
            },
        },
        ..admission_with_identity(PromptIdentityV1::OMITTED)
    };
    assert!(valid_admission(valid));

    let malformed_json = b"not json";
    let malformed = PlacementAdmissionV1 {
        metadata: PlacementMetadataV1 {
            format: AdmissionMetadataFormatV1::JSON_UTF8,
            flags: 0,
            bytes: ByteSliceV1 {
                data: malformed_json.as_ptr(),
                len: malformed_json.len() as u64,
            },
        },
        ..admission_with_identity(PromptIdentityV1::OMITTED)
    };
    assert!(!valid_admission(malformed));

    let oversized_json = vec![b' '; MAX_ADMISSION_METADATA_BYTES_V1 as usize + 1];
    let oversized = PlacementAdmissionV1 {
        metadata: PlacementMetadataV1 {
            format: AdmissionMetadataFormatV1::JSON_UTF8,
            flags: 0,
            bytes: ByteSliceV1 {
                data: oversized_json.as_ptr(),
                len: oversized_json.len() as u64,
            },
        },
        ..admission_with_identity(PromptIdentityV1::OMITTED)
    };
    assert!(!valid_admission(oversized));

    let unknown_format = PlacementAdmissionV1 {
        metadata: PlacementMetadataV1 {
            format: AdmissionMetadataFormatV1(99),
            flags: 0,
            bytes: ByteSliceV1::EMPTY,
        },
        ..admission_with_identity(PromptIdentityV1::OMITTED)
    };
    assert!(!valid_admission(unknown_format));
}

fn admission_with_identity(prompt_identity: PromptIdentityV1) -> PlacementAdmissionV1 {
    PlacementAdmissionV1 {
        request_id: [7; 16],
        flags: 0,
        priority: 0,
        prompt_tokens: 0,
        max_output_tokens: 1,
        prompt_identity,
        metadata: PlacementMetadataV1::EMPTY,
        session_id: ByteSliceV1::EMPTY,
    }
}

fn valid_admission(admission: PlacementAdmissionV1) -> bool {
    let mutation = PlacementMutationV1 {
        struct_size: std::mem::size_of::<PlacementMutationV1>() as u32,
        kind: PlacementMutationKindV1::ADMIT,
        flags: 0,
        sequence: 0,
        now_ms: 0.0,
        payload: PlacementMutationPayloadV1 { admission },
    };
    // Safety: `mutation` stays live for the entire validation call.
    unsafe {
        validate_mutation_batch_v1(PlacementMutationSliceV1 {
            data: &mutation,
            len: 1,
        })
        .is_ok()
    }
}
