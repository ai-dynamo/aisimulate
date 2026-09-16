// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{LazyLock, Mutex};

use aisimulate_core::replay::loadgen::{
    DynPlacement, SteppableAgg, SteppableDisagg, SteppableReplay,
};
use aisimulate_core::replay::{
    DirectRequest, DynamicKvEventObservation, DynamicPlacementMetadata, DynamicPlacementPolicy,
    ReplayEngineConfig, ReplayEngineFactory,
};
use aisimulate_placement_abi::{
    AdmissionDecisionV1, ByteSliceV1, KvEventSliceV1, PlacementBatchResultV1,
    PlacementCacheSampleV1, PlacementDiagnosticSliceV1, PlacementHandleV1, PlacementLimitsV1,
    PlacementMutationKindV1, PlacementMutationSliceV1, PlacementResultSliceV1, PlacementResultV1,
    PlacementSliceV1, PlacementV1, PluginVTableV1, StatusV1,
};
use uuid::Uuid;

#[derive(Default)]
struct Fixture {
    admissions: Vec<PlacementResultV1>,
    admitted: Option<[u8; 16]>,
    prefill_admitted: Option<[u8; 16]>,
    decode_admitted: Option<[u8; 16]>,
    kv_event_count: usize,
}

unsafe extern "C" fn apply_kv_events(
    _handle: PlacementHandleV1,
    events: KvEventSliceV1,
    _now_ms: f64,
    result: *mut PlacementBatchResultV1,
) -> StatusV1 {
    if events.len != 1 || events.data.is_null() || result.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    FIXTURE.lock().expect("fixture lock").kv_event_count += 1;
    // Safety: a checked non-null output gets one complete result.
    unsafe {
        *result = PlacementBatchResultV1 {
            struct_size: std::mem::size_of::<PlacementBatchResultV1>() as u32,
            flags: 0,
            applied_mutations: 1,
            pending_count: 0,
            admission_results: PlacementResultSliceV1 {
                data: std::ptr::null(),
                len: 0,
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

unsafe extern "C" fn apply_prefill_batch(
    handle: PlacementHandleV1,
    batch: PlacementMutationSliceV1,
    result: *mut PlacementBatchResultV1,
) -> StatusV1 {
    let is_admission = batch.len == 1
        && !batch.data.is_null()
        // Safety: a non-null one-record slice is valid for this callback.
        && unsafe { (*batch.data).kind == PlacementMutationKindV1::ADMIT };
    let status = unsafe { apply_batch(handle, batch, result) };
    if status == StatusV1::OK && is_admission {
        let mut fixture = FIXTURE.lock().expect("fixture lock");
        fixture.prefill_admitted = fixture.admitted;
    }
    status
}

unsafe extern "C" fn apply_decode_batch(
    handle: PlacementHandleV1,
    batch: PlacementMutationSliceV1,
    result: *mut PlacementBatchResultV1,
) -> StatusV1 {
    let is_admission = batch.len == 1
        && !batch.data.is_null()
        // Safety: a non-null one-record slice is valid for this callback.
        && unsafe { (*batch.data).kind == PlacementMutationKindV1::ADMIT };
    let status = unsafe { apply_batch(handle, batch, result) };
    if status == StatusV1::OK && is_admission {
        let mut fixture = FIXTURE.lock().expect("fixture lock");
        fixture.decode_admitted = fixture.admitted;
    }
    status
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
    apply_kv_events: Some(apply_kv_events),
};

static PREFILL_FIXTURE_VTABLE: PluginVTableV1 = PluginVTableV1 {
    apply_batch: Some(apply_prefill_batch),
    ..FIXTURE_VTABLE
};

static DECODE_FIXTURE_VTABLE: PluginVTableV1 = PluginVTableV1 {
    apply_batch: Some(apply_decode_batch),
    ..FIXTURE_VTABLE
};

#[test]
fn aggregated_steppable_uses_the_dynamic_placement_abi_fixture() {
    let _test_lock = TEST_LOCK.lock().expect("test lock");
    *FIXTURE.lock().expect("fixture lock") = Fixture::default();
    let factory = ReplayEngineFactory::new();
    let mut replay = SteppableAgg::<
        DynPlacement<DynamicKvEventObservation, DynamicPlacementMetadata>,
        DynamicKvEventObservation,
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

#[test]
fn disaggregated_steppable_uses_distinct_dynamic_policies_for_prefill_and_decode() {
    let _test_lock = TEST_LOCK.lock().expect("test lock");
    *FIXTURE.lock().expect("fixture lock") = Fixture::default();
    let factory = ReplayEngineFactory::new();
    let mut engine = ReplayEngineConfig::default();
    engine.rank.block_size = 4;
    let mut replay = SteppableDisagg::<
        DynPlacement<DynamicKvEventObservation, DynamicPlacementMetadata>,
        DynamicKvEventObservation,
        DynamicPlacementMetadata,
    >::with_placements(
        engine,
        &factory,
        1,
        1,
        |_prefill_dp_size, prefill_topology, _decode_dp_size, decode_topology| {
            assert_eq!(prefill_topology.len(), 1);
            assert_eq!(decode_topology.len(), 1);
            // Safety: the static ABI fixtures remain live and accept their sentinel handles.
            let prefill = unsafe {
                DynamicPlacementPolicy::from_test_vtable(
                    PREFILL_FIXTURE_VTABLE,
                    PlacementLimitsV1 {
                        max_mutations: 1,
                        max_admission_results: 1,
                        max_released: 1,
                        max_diagnostic_bytes: 0,
                    },
                )
            };
            // Safety: the static ABI fixtures remain live and accept their sentinel handles.
            let decode = unsafe {
                DynamicPlacementPolicy::from_test_vtable(
                    DECODE_FIXTURE_VTABLE,
                    PlacementLimitsV1 {
                        max_mutations: 1,
                        max_admission_results: 1,
                        max_released: 1,
                        max_diagnostic_bytes: 0,
                    },
                )
            };
            Ok((Box::new(prefill), Box::new(decode)))
        },
    )
    .expect("dynamic placement fixtures build the disaggregated replay");

    let request_id = Uuid::from_u128(8);
    replay
        .submit(DirectRequest {
            uuid: Some(request_id),
            // A full prompt block forces a router-visible stored-KV event
            // during prefill.
            tokens: (1..=8).collect(),
            max_output_tokens: 1,
            ..Default::default()
        })
        .expect("prefill placement admits the request");
    while !replay.is_idle() {
        replay
            .step()
            .expect("dynamic placement fixtures make progress");
    }

    let fixture = FIXTURE.lock().expect("fixture lock");
    assert_eq!(fixture.prefill_admitted, Some(request_id.into_bytes()));
    assert_eq!(fixture.decode_admitted, Some(request_id.into_bytes()));
    assert!(
        fixture.kv_event_count > 0,
        "dynamic prefill placement receives each router-visible KV event"
    );
}
