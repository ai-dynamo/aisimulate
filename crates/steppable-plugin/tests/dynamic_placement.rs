// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{LazyLock, Mutex};

use aisimulate_core::replay::loadgen::{DynPlacement, SteppableAgg, SteppableReplay};
use aisimulate_core::replay::{
    DirectRequest, DynamicPlacementMetadata, DynamicPlacementPolicy, NoEngineEvents,
    ReplayEngineConfig, ReplayEngineFactory,
};
use aisimulate_placement_abi::{
    AdmissionDecisionV1, ByteSliceV1, PlacementBatchResultV1, PlacementCacheSampleV1,
    PlacementDiagnosticSliceV1, PlacementHandleV1, PlacementLimitsV1, PlacementMutationKindV1,
    PlacementMutationSliceV1, PlacementResultSliceV1, PlacementResultV1, PlacementSliceV1,
    PlacementV1, PluginVTableV1, StatusV1,
};
use uuid::Uuid;

#[derive(Default)]
struct Fixture {
    admissions: Vec<PlacementResultV1>,
    admitted: Option<[u8; 16]>,
}

static FIXTURE: LazyLock<Mutex<Fixture>> = LazyLock::new(|| Mutex::new(Fixture::default()));
static TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

unsafe extern "C" fn apply_batch(
    _handle: PlacementHandleV1,
    batch: PlacementMutationSliceV1,
    result: *mut PlacementBatchResultV1,
) -> StatusV1 {
    if batch.len != 1 || batch.data.is_null() || result.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: one non-null mutation was checked above.
    let mutation = unsafe { *batch.data };
    let mut fixture = FIXTURE.lock().expect("fixture lock");
    fixture.admissions.clear();
    if mutation.kind == PlacementMutationKindV1::ADMIT {
        // Safety: the mutation kind selects the admission union member.
        let admission = unsafe { mutation.payload.admission };
        fixture.admitted = Some(admission.request_id);
        fixture.admissions.push(PlacementResultV1 {
            request_id: admission.request_id,
            decision: AdmissionDecisionV1::IMMEDIATE,
            reserved: [0; 4],
            placement: PlacementV1 {
                request_id: admission.request_id,
                worker_id: 0,
                scheduler_id: 0,
                reported_overlap_tokens: 0,
                cache_sample: PlacementCacheSampleV1 {
                    flags: 0,
                    overlap_blocks: 0,
                    best_available_overlap_blocks: 0,
                    isl_blocks: 0,
                },
                placement_replica_id: 0,
                reserved: 0,
            },
        });
    }
    // Safety: the checked non-null output has one complete result written.
    unsafe {
        *result = PlacementBatchResultV1 {
            struct_size: std::mem::size_of::<PlacementBatchResultV1>() as u32,
            flags: 0,
            applied_mutations: 1,
            pending_count: 0,
            admission_results: PlacementResultSliceV1 {
                data: fixture.admissions.as_ptr(),
                len: fixture.admissions.len() as u64,
            },
            released: PlacementSliceV1 {
                data: std::ptr::null(),
                len: 0,
            },
            diagnostics: PlacementDiagnosticSliceV1 {
                data: std::ptr::null(),
                len: 0,
            },
        };
    }
    StatusV1::OK
}

unsafe extern "C" fn release_results(_result: PlacementBatchResultV1) {}
unsafe extern "C" fn release_bytes(_bytes: ByteSliceV1) {}
unsafe extern "C" fn last_error(_handle: PlacementHandleV1, error: *mut ByteSliceV1) -> StatusV1 {
    if error.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: the checked output receives an empty diagnostic.
    unsafe { *error = ByteSliceV1::EMPTY };
    StatusV1::OK
}
unsafe extern "C" fn destroy(_handle: PlacementHandleV1) {}

static FIXTURE_VTABLE: PluginVTableV1 = PluginVTableV1 {
    struct_size: std::mem::size_of::<PluginVTableV1>() as u32,
    flags: 0,
    create: None,
    apply_batch: Some(apply_batch),
    release_results: Some(release_results),
    release_bytes: Some(release_bytes),
    last_error: Some(last_error),
    destroy: Some(destroy),
};

#[test]
fn aggregated_steppable_uses_the_dynamic_placement_abi_fixture() {
    let _test_lock = TEST_LOCK.lock().expect("test lock");
    *FIXTURE.lock().expect("fixture lock") = Fixture::default();
    let factory = ReplayEngineFactory::new();
    let mut replay = SteppableAgg::<
        DynPlacement<NoEngineEvents, DynamicPlacementMetadata>,
        NoEngineEvents,
        DynamicPlacementMetadata,
    >::with_placement(
        ReplayEngineConfig::default(),
        &factory,
        1,
        |_dp_size, _topology| {
            // Safety: this static ABI fixture remains live and accepts its sentinel handle.
            let policy = unsafe {
                DynamicPlacementPolicy::from_test_vtable(
                    FIXTURE_VTABLE,
                    PlacementLimitsV1 {
                        max_mutations: 1,
                        max_admission_results: 1,
                        max_released: 1,
                        max_diagnostic_bytes: 0,
                    },
                )
            };
            Ok(Box::new(policy))
        },
    )
    .expect("dynamic placement fixture builds the aggregate replay");

    let request_id = Uuid::from_u128(7);
    assert_eq!(
        replay
            .submit(DirectRequest {
                uuid: Some(request_id),
                tokens: vec![1, 2, 3],
                max_output_tokens: 1,
                ..Default::default()
            })
            .expect("fixture placement admits the request"),
        request_id
    );
    assert_eq!(
        FIXTURE.lock().expect("fixture lock").admitted,
        Some(request_id.into_bytes())
    );
}
