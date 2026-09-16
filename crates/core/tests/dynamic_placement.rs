// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aisimulate_core::replay::DynamicPlacementPlugin;
use std::sync::{LazyLock, Mutex};

use aisimulate_core::replay::DirectRequest;
use aisimulate_core::replay::loadgen::{ReplayRequestHashes, ReplayRequestPayload};
use aisimulate_core::replay::{
    DynamicPlacementMetadata, DynamicPlacementPolicy, PlacementDecision, PlacementPolicy,
    ReplayAdmissionMetadata, ReplayRequestContext, WorkerTopology,
};
use aisimulate_placement_abi::{
    AdmissionDecisionV1, AdmissionMetadataFormatV1, ByteSliceV1, PlacementBatchResultV1,
    PlacementCacheSampleV1, PlacementDiagnosticSliceV1, PlacementHandleV1, PlacementLimitsV1,
    PlacementMutationKindV1, PlacementMutationSliceV1, PlacementResultSliceV1, PlacementResultV1,
    PlacementSliceV1, PlacementV1, PluginVTableV1, PromptIdentityV1, StatusV1,
};
use uuid::Uuid;

#[derive(Default)]
struct Fixture {
    kinds: Vec<PlacementMutationKindV1>,
    pending_count: u64,
    admissions: Vec<PlacementResultV1>,
    released: Vec<PlacementV1>,
    last_admission: Option<CapturedAdmission>,
}

#[derive(Debug, PartialEq)]
struct CapturedAdmission {
    prompt_identity_flags: u32,
    materialized_token_ids: Option<Vec<u32>>,
    local_block_hashes: Option<Vec<u64>>,
    sequence_block_hashes: Option<Vec<u64>>,
    metadata_format: AdmissionMetadataFormatV1,
    metadata: Option<Vec<u8>>,
}

static FIXTURE: LazyLock<Mutex<Fixture>> = LazyLock::new(|| Mutex::new(Fixture::default()));
static TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

unsafe extern "C" fn fixture_apply_batch(
    _handle: PlacementHandleV1,
    batch: PlacementMutationSliceV1,
    result: *mut PlacementBatchResultV1,
) -> StatusV1 {
    if batch.len != 1 || batch.data.is_null() || result.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: non-null, one-record input checked above.
    let mutation = unsafe { *batch.data };
    let mut fixture = FIXTURE.lock().expect("fixture lock");
    fixture.kinds.push(mutation.kind);
    fixture.admissions.clear();
    fixture.released.clear();
    match mutation.kind {
        PlacementMutationKindV1::ADMIT => {
            // Safety: `kind` selects the admission arm.
            let admission = unsafe { mutation.payload.admission };
            fixture.last_admission = Some(CapturedAdmission {
                prompt_identity_flags: admission.prompt_identity.flags,
                materialized_token_ids: capture_optional_slice(
                    admission.prompt_identity.flags,
                    PromptIdentityV1::MATERIALIZED_TOKEN_IDS_PRESENT,
                    admission.prompt_identity.materialized_token_ids.data,
                    admission.prompt_identity.materialized_token_ids.len,
                ),
                local_block_hashes: capture_optional_slice(
                    admission.prompt_identity.flags,
                    PromptIdentityV1::LOCAL_BLOCK_HASHES_PRESENT,
                    admission.prompt_identity.local_block_hashes.data,
                    admission.prompt_identity.local_block_hashes.len,
                ),
                sequence_block_hashes: capture_optional_slice(
                    admission.prompt_identity.flags,
                    PromptIdentityV1::SEQUENCE_BLOCK_HASHES_PRESENT,
                    admission.prompt_identity.sequence_block_hashes.data,
                    admission.prompt_identity.sequence_block_hashes.len,
                ),
                metadata_format: admission.metadata.format,
                metadata: (admission.metadata.format == AdmissionMetadataFormatV1::JSON_UTF8).then(
                    || {
                        // Safety: JSON metadata is valid for the duration of this callback.
                        unsafe {
                            std::slice::from_raw_parts(
                                admission.metadata.bytes.data,
                                admission.metadata.bytes.len as usize,
                            )
                            .to_vec()
                        }
                    },
                ),
            });
            fixture.pending_count = 1;
            fixture.admissions.push(PlacementResultV1 {
                request_id: admission.request_id,
                decision: AdmissionDecisionV1::QUEUED,
                reserved: [0; 4],
                placement: fixture_placement(admission.request_id),
            });
        }
        PlacementMutationKindV1::REQUEST_TERMINAL => {
            // Safety: `kind` selects the lifecycle arm.
            let lifecycle = unsafe { mutation.payload.request_lifecycle };
            fixture.pending_count = 0;
            fixture
                .released
                .push(fixture_placement(lifecycle.request_id));
        }
        _ => {}
    }
    // Safety: non-null result output checked above.
    unsafe {
        *result = PlacementBatchResultV1 {
            struct_size: std::mem::size_of::<PlacementBatchResultV1>() as u32,
            flags: 0,
            applied_mutations: 1,
            pending_count: fixture.pending_count,
            admission_results: PlacementResultSliceV1 {
                data: fixture.admissions.as_ptr(),
                len: fixture.admissions.len() as u64,
            },
            released: PlacementSliceV1 {
                data: fixture.released.as_ptr(),
                len: fixture.released.len() as u64,
            },
            diagnostics: PlacementDiagnosticSliceV1 {
                data: std::ptr::null(),
                len: 0,
            },
        };
    }
    StatusV1::OK
}

fn capture_optional_slice<T: Copy>(
    flags: u32,
    present_flag: u32,
    data: *const T,
    len: u64,
) -> Option<Vec<T>> {
    (flags & present_flag != 0).then(|| {
        // Safety: the ABI presence flag promises this slice is borrowed for this callback.
        unsafe { std::slice::from_raw_parts(data, len as usize).to_vec() }
    })
}

unsafe extern "C" fn fixture_release_results(_result: PlacementBatchResultV1) {}
unsafe extern "C" fn fixture_release_bytes(_bytes: ByteSliceV1) {}
unsafe extern "C" fn fixture_last_error(
    _handle: PlacementHandleV1,
    error: *mut ByteSliceV1,
) -> StatusV1 {
    if error.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: non-null output pointer checked above.
    unsafe { *error = ByteSliceV1::EMPTY };
    StatusV1::OK
}
unsafe extern "C" fn fixture_destroy(_handle: PlacementHandleV1) {}

static FIXTURE_VTABLE: PluginVTableV1 = PluginVTableV1 {
    struct_size: std::mem::size_of::<PluginVTableV1>() as u32,
    flags: 0,
    create: None,
    apply_batch: Some(fixture_apply_batch),
    release_results: Some(fixture_release_results),
    release_bytes: Some(fixture_release_bytes),
    last_error: Some(fixture_last_error),
    destroy: Some(fixture_destroy),
};

fn fixture_placement(request_id: [u8; 16]) -> PlacementV1 {
    PlacementV1 {
        request_id,
        worker_id: 4,
        scheduler_id: 9,
        reported_overlap_tokens: 17,
        cache_sample: PlacementCacheSampleV1 {
            flags: PlacementCacheSampleV1::PRESENT,
            overlap_blocks: 1,
            best_available_overlap_blocks: 2,
            isl_blocks: 3,
        },
        placement_replica_id: 0,
        reserved: 0,
    }
}

fn policy() -> DynamicPlacementPolicy {
    *FIXTURE.lock().expect("fixture lock") = Fixture::default();
    // Safety: this fixture table remains static and all callbacks accept its sentinel handle.
    unsafe {
        DynamicPlacementPolicy::from_test_vtable(
            FIXTURE_VTABLE,
            PlacementLimitsV1 {
                max_mutations: 1,
                max_admission_results: 1,
                max_released: 1,
                max_diagnostic_bytes: 0,
            },
        )
    }
}

#[test]
fn dynamic_placement_loader_reports_the_library_path_when_loading_fails() {
    let _test_lock = TEST_LOCK.lock().expect("test lock");
    let error = match DynamicPlacementPlugin::load("/definitely/not/a/placement/plugin.so") {
        Ok(_) => panic!("a missing placement plugin must fail to load"),
        Err(error) => error,
    };

    assert!(
        error
            .to_string()
            .contains("/definitely/not/a/placement/plugin.so"),
        "loader error must name the rejected path: {error:#}"
    );
}

#[test]
fn neutral_adapter_translates_admission_pending_and_lifecycle_mutations() {
    let _test_lock = TEST_LOCK.lock().expect("test lock");
    let mut policy = policy();
    let request_id = Uuid::from_u128(7);
    let request = ReplayRequestPayload::materialized(DirectRequest {
        uuid: Some(request_id),
        tokens: vec![1, 2, 3],
        max_output_tokens: 5,
        priority: 2,
        ..Default::default()
    });

    let effects = policy
        .place(&request, DynamicPlacementMetadata::default(), None, 3.0)
        .expect("admission succeeds");
    assert!(matches!(effects.decision, PlacementDecision::Queued));
    assert_eq!(policy.pending_count(), 1);
    assert!(policy.cancel_pending(request_id));
    policy
        .prefill_completed(request_id, 4.0)
        .expect("prefill succeeds");
    policy
        .worker_ready(
            WorkerTopology {
                worker_id: 4,
                scheduler_ids: vec![9],
            },
            4.5,
        )
        .expect("worker ready succeeds");
    policy
        .worker_draining(
            WorkerTopology {
                worker_id: 4,
                scheduler_ids: vec![9],
            },
            4.6,
        )
        .expect("worker draining succeeds");
    policy
        .worker_removed(
            WorkerTopology {
                worker_id: 4,
                scheduler_ids: vec![9],
            },
            4.7,
        )
        .expect("worker removed succeeds");
    let released = policy
        .request_terminal(request_id, 5.0)
        .expect("terminal succeeds");
    assert_eq!(released.len(), 1);
    assert_eq!(released[0].request_id, request_id);
    assert_eq!(released[0].scheduler_id, 9);
    assert_eq!(released[0].reported_overlap_tokens, 17);
    assert_eq!(policy.pending_count(), 0);
    policy.topology_settled(5.1).expect("topology settles");
    assert_eq!(
        FIXTURE.lock().expect("fixture lock").kinds,
        vec![
            PlacementMutationKindV1::ADMIT,
            PlacementMutationKindV1::CANCEL_PENDING,
            PlacementMutationKindV1::PREFILL_COMPLETED,
            PlacementMutationKindV1::WORKER_READY,
            PlacementMutationKindV1::WORKER_DRAINING,
            PlacementMutationKindV1::WORKER_REMOVED,
            PlacementMutationKindV1::REQUEST_TERMINAL,
            PlacementMutationKindV1::TOPOLOGY_SETTLED,
        ]
    );
}

#[test]
fn neutral_adapter_preserves_routing_identity_and_typed_replay_metadata() {
    let _test_lock = TEST_LOCK.lock().expect("test lock");
    let mut policy = policy();
    let request = ReplayRequestPayload::materialized(DirectRequest {
        uuid: Some(Uuid::from_u128(8)),
        tokens: vec![10, 20, 30],
        replay_context: Some(ReplayRequestContext {
            authored_id: "trace-request-8".into(),
            session_id: Some("session-a".into()),
            turn_index: Some(4),
            metadata: serde_json::json!({"tenant": "demo"}),
            prompt_token_source: Default::default(),
        }),
        ..Default::default()
    });
    let metadata = DynamicPlacementMetadata::from_hashes(Some(ReplayRequestHashes {
        local_block_hashes: vec![11, 12],
        sequence_hashes: vec![21, 22],
    }));

    policy
        .place(&request, metadata, None, 3.0)
        .expect("admission succeeds");

    let admission = FIXTURE
        .lock()
        .expect("fixture lock")
        .last_admission
        .take()
        .expect("fixture captured admission");
    assert_eq!(
        admission.prompt_identity_flags,
        PromptIdentityV1::MATERIALIZED_TOKEN_IDS_PRESENT
            | PromptIdentityV1::LOCAL_BLOCK_HASHES_PRESENT
            | PromptIdentityV1::SEQUENCE_BLOCK_HASHES_PRESENT
    );
    assert_eq!(admission.materialized_token_ids, Some(vec![10, 20, 30]));
    assert_eq!(admission.local_block_hashes, Some(vec![11, 12]));
    assert_eq!(admission.sequence_block_hashes, Some(vec![21, 22]));
    assert_eq!(
        admission.metadata_format,
        AdmissionMetadataFormatV1::JSON_UTF8
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&admission.metadata.expect("JSON metadata"))
            .expect("metadata is JSON"),
        serde_json::json!({
            "authored_id": "trace-request-8",
            "session_id": "session-a",
            "turn_index": 4,
            "metadata": {"tenant": "demo"},
            "prompt_token_source": "materialized"
        })
    );
}

#[test]
fn neutral_adapter_distinguishes_omitted_identity_and_metadata_from_known_empty() {
    let _test_lock = TEST_LOCK.lock().expect("test lock");
    let mut omitted_policy = policy();
    let deferred = ReplayRequestPayload::deferred(
        DirectRequest {
            uuid: Some(Uuid::from_u128(9)),
            ..Default::default()
        },
        3,
        vec![1, 2],
        2,
    );

    omitted_policy
        .place(
            &deferred,
            DynamicPlacementMetadata::from_hashes(None),
            None,
            3.0,
        )
        .expect("admission succeeds");

    let omitted = FIXTURE
        .lock()
        .expect("fixture lock")
        .last_admission
        .take()
        .expect("fixture captured admission");
    assert_eq!(omitted.prompt_identity_flags, 0);
    assert_eq!(omitted.materialized_token_ids, None);
    assert_eq!(omitted.local_block_hashes, None);
    assert_eq!(omitted.sequence_block_hashes, None);
    assert_eq!(omitted.metadata_format, AdmissionMetadataFormatV1::NONE);
    assert_eq!(omitted.metadata, None);

    let mut policy = policy();
    let request = ReplayRequestPayload::materialized(DirectRequest {
        uuid: Some(Uuid::from_u128(10)),
        tokens: Vec::new(),
        ..Default::default()
    });

    policy
        .place(
            &request,
            DynamicPlacementMetadata::from_hashes(Some(ReplayRequestHashes {
                local_block_hashes: Vec::new(),
                sequence_hashes: Vec::new(),
            })),
            None,
            3.0,
        )
        .expect("admission succeeds");

    let admission = FIXTURE
        .lock()
        .expect("fixture lock")
        .last_admission
        .take()
        .expect("fixture captured admission");
    assert_eq!(
        admission.prompt_identity_flags,
        PromptIdentityV1::MATERIALIZED_TOKEN_IDS_PRESENT
            | PromptIdentityV1::LOCAL_BLOCK_HASHES_PRESENT
            | PromptIdentityV1::SEQUENCE_BLOCK_HASHES_PRESENT
    );
    assert_eq!(admission.materialized_token_ids, Some(Vec::new()));
    assert_eq!(admission.local_block_hashes, Some(Vec::new()));
    assert_eq!(admission.sequence_block_hashes, Some(Vec::new()));
    assert_eq!(admission.metadata_format, AdmissionMetadataFormatV1::NONE);
    assert_eq!(admission.metadata, None);
}

#[test]
fn neutral_adapter_refuses_observation_dtos_it_cannot_encode() {
    let _test_lock = TEST_LOCK.lock().expect("test lock");
    let policy = policy();

    let error = policy
        .unsupported_observation("KvObservation")
        .expect_err("unrepresentable observations must not be dropped");

    assert!(error.to_string().contains("NoEngineEvents"));
    assert!(error.to_string().contains("KvObservation"));
}
