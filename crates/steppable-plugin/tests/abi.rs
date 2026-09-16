// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use aiperf_steppable_abi::{
    ByteSliceV1, CompactRequestV1, CreateRequestV1, DirectRequestSliceV1, DirectRequestV1,
    EngineEventSliceV1, EngineEventV1, HashBufferIdV1, HashBufferLeaseCallbacksV1,
    HashBufferRangeV1, PLUGIN_ABI_MAJOR_V1, REQUEST_FLAG_UUID, ReplayContextV1, ReplayHandleV1,
    ReplayStateV1, RequestIdMutSliceV1, RequestIdSliceV1, SLA_FLAG_TTFT, SlaThresholdsV1, StatusV1,
    StepRequestV1, StepResultV1, U32SliceV1, validate_descriptor_v1,
};

unsafe extern "C" fn count_hash_buffer_release(context: *mut c_void, buffer_id: HashBufferIdV1) {
    assert!(buffer_id.is_valid());
    // Safety: the test supplies a valid `AtomicUsize` context for the whole
    // replay lifetime and the callback only records its observable effect.
    unsafe { &*context.cast::<AtomicUsize>() }.fetch_add(1, Ordering::SeqCst);
}

fn compact_request() -> CompactRequestV1 {
    CompactRequestV1 {
        struct_size: std::mem::size_of::<CompactRequestV1>() as u32,
        flags: 0,
        input_token_count: 8,
        trace_block_size: 4,
        reserved: 0,
        hash_ids: U32SliceV1::EMPTY,
        request: DirectRequestV1 {
            struct_size: std::mem::size_of::<DirectRequestV1>() as u32,
            flags: REQUEST_FLAG_UUID,
            tokens: U32SliceV1::EMPTY,
            output_token_ids: U32SliceV1::EMPTY,
            max_output_tokens: 1,
            uuid: [31; 16],
            dp_rank: 0,
            preferred_dp_rank: 0,
            preferred_prefill_dp_rank: 0,
            arrival_timestamp_ms: 0.0,
            priority: 0,
            strict_priority: 0,
            policy_class: ByteSliceV1::EMPTY,
            replay_context: ReplayContextV1::EMPTY,
        },
    }
}

#[test]
fn plugin_entry_exposes_a_complete_v1_table() {
    let descriptor = aisimulate_steppable_plugin::aiperf_steppable_plugin_v1();

    assert!(!descriptor.is_null());
    let descriptor = unsafe { &*descriptor };
    assert_eq!(descriptor.abi_major, PLUGIN_ABI_MAJOR_V1);
    assert!(unsafe { validate_descriptor_v1(descriptor) }.is_ok());
}

#[test]
fn leased_compact_range_releases_once_when_cancelled() {
    let descriptor = unsafe { &*aisimulate_steppable_plugin::aiperf_steppable_plugin_v1() };
    let vtable = unsafe { &*descriptor.vtable };
    let tail = unsafe {
        aiperf_steppable_abi::PluginVTableV1::compact_buffer_leases(
            descriptor.vtable.cast(),
            descriptor.capabilities,
        )
    }
    .expect("AISimulate advertises a complete compact-buffer lease tail");
    let releases = AtomicUsize::new(0);
    let payload = br#"{}"#;
    let mut handle = ReplayHandleV1(std::ptr::null_mut());
    let mut error = ByteSliceV1::EMPTY;

    assert_eq!(
        unsafe {
            tail.create_with_hash_buffer_leases.unwrap()(
                CreateRequestV1 {
                    struct_size: std::mem::size_of::<CreateRequestV1>() as u32,
                    flags: 0,
                    provider_payload: ByteSliceV1 {
                        data: payload.as_ptr(),
                        len: payload.len() as u64,
                    },
                },
                HashBufferLeaseCallbacksV1 {
                    struct_size: std::mem::size_of::<HashBufferLeaseCallbacksV1>() as u32,
                    flags: 0,
                    context: (&raw const releases).cast_mut().cast(),
                    release_hash_buffer: Some(count_hash_buffer_release),
                },
                &raw mut handle,
                &raw mut error,
            )
        },
        StatusV1::OK
    );

    let hashes = [101_u32, 102];
    let mut buffer_id = HashBufferIdV1::INVALID;
    assert_eq!(
        unsafe {
            tail.register_hash_buffer.unwrap()(
                handle,
                U32SliceV1 {
                    data: hashes.as_ptr(),
                    len: hashes.len() as u64,
                },
                &raw mut buffer_id,
            )
        },
        StatusV1::OK
    );
    let misaligned = [0_u8; 12];
    let mut rejected_buffer_id = HashBufferIdV1::INVALID;
    assert_eq!(
        unsafe {
            tail.register_hash_buffer.unwrap()(
                handle,
                U32SliceV1 {
                    data: misaligned.as_ptr().add(1).cast(),
                    len: 1,
                },
                &raw mut rejected_buffer_id,
            )
        },
        StatusV1::INVALID_ARGUMENT
    );
    let mut request_id = [0; 16];
    assert_eq!(
        unsafe {
            tail.submit_compact_hash_buffer_range.unwrap()(
                handle,
                compact_request(),
                HashBufferRangeV1 {
                    buffer_id,
                    offset: 0,
                    len: hashes.len() as u64,
                },
                &raw mut request_id,
            )
        },
        StatusV1::OK
    );
    assert_eq!(releases.load(Ordering::SeqCst), 0);

    let mut event = EngineEventV1 {
        request_id: [0; 16],
        flags: 0,
        token_id: 0,
        terminal_status: 0,
        reserved: 0,
    };
    let mut canceled = 0;
    assert_eq!(
        unsafe {
            vtable.cancel.unwrap()(
                handle,
                &raw const request_id,
                &raw mut event,
                &raw mut canceled,
            )
        },
        StatusV1::OK
    );
    assert_eq!(canceled, 1);
    assert_eq!(releases.load(Ordering::SeqCst), 1);

    let mut reused_request = compact_request();
    reused_request.request.uuid = [32; 16];
    assert_eq!(
        unsafe {
            tail.submit_compact_hash_buffer_range.unwrap()(
                handle,
                reused_request,
                HashBufferRangeV1 {
                    buffer_id,
                    offset: 0,
                    len: hashes.len() as u64,
                },
                &raw mut request_id,
            )
        },
        StatusV1::INVALID_ARGUMENT
    );
    assert_eq!(releases.load(Ordering::SeqCst), 1);

    unsafe { vtable.destroy.unwrap()(handle) };
    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

#[test]
fn leased_compact_range_releases_once_when_terminal() {
    let descriptor = unsafe { &*aisimulate_steppable_plugin::aiperf_steppable_plugin_v1() };
    let vtable = unsafe { &*descriptor.vtable };
    let tail = unsafe {
        aiperf_steppable_abi::PluginVTableV1::compact_buffer_leases(
            descriptor.vtable.cast(),
            descriptor.capabilities,
        )
    }
    .expect("AISimulate advertises a complete compact-buffer lease tail");
    let releases = AtomicUsize::new(0);
    let payload = br#"{}"#;
    let mut handle = ReplayHandleV1(std::ptr::null_mut());
    let mut error = ByteSliceV1::EMPTY;
    assert_eq!(
        unsafe {
            tail.create_with_hash_buffer_leases.unwrap()(
                CreateRequestV1 {
                    struct_size: std::mem::size_of::<CreateRequestV1>() as u32,
                    flags: 0,
                    provider_payload: ByteSliceV1 {
                        data: payload.as_ptr(),
                        len: payload.len() as u64,
                    },
                },
                HashBufferLeaseCallbacksV1 {
                    struct_size: std::mem::size_of::<HashBufferLeaseCallbacksV1>() as u32,
                    flags: 0,
                    context: (&raw const releases).cast_mut().cast(),
                    release_hash_buffer: Some(count_hash_buffer_release),
                },
                &raw mut handle,
                &raw mut error,
            )
        },
        StatusV1::OK
    );
    let hashes = [151_u32, 152];
    let mut buffer_id = HashBufferIdV1::INVALID;
    assert_eq!(
        unsafe {
            tail.register_hash_buffer.unwrap()(
                handle,
                U32SliceV1 {
                    data: hashes.as_ptr(),
                    len: hashes.len() as u64,
                },
                &raw mut buffer_id,
            )
        },
        StatusV1::OK
    );
    let mut request_id = [0; 16];
    assert_eq!(
        unsafe {
            tail.submit_compact_hash_buffer_range.unwrap()(
                handle,
                compact_request(),
                HashBufferRangeV1 {
                    buffer_id,
                    offset: 0,
                    len: hashes.len() as u64,
                },
                &raw mut request_id,
            )
        },
        StatusV1::OK
    );
    assert_eq!(releases.load(Ordering::SeqCst), 0);

    let mut step = StepResultV1::EMPTY;
    assert_eq!(
        unsafe {
            vtable.step.unwrap()(
                handle,
                StepRequestV1 {
                    struct_size: std::mem::size_of::<StepRequestV1>() as u32,
                    flags: 0,
                    until_ms: f64::INFINITY,
                },
                &raw mut step,
            )
        },
        StatusV1::OK
    );
    assert!(step.events.len > 0);
    unsafe { vtable.release_events.unwrap()(step.events) };
    unsafe { vtable.release_request_facts.unwrap()(step.request_facts) };
    assert_eq!(releases.load(Ordering::SeqCst), 1);

    unsafe { vtable.destroy.unwrap()(handle) };
    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

#[test]
fn leased_compact_range_releases_once_when_replay_is_destroyed() {
    let descriptor = unsafe { &*aisimulate_steppable_plugin::aiperf_steppable_plugin_v1() };
    let vtable = unsafe { &*descriptor.vtable };
    let tail = unsafe {
        aiperf_steppable_abi::PluginVTableV1::compact_buffer_leases(
            descriptor.vtable.cast(),
            descriptor.capabilities,
        )
    }
    .expect("AISimulate advertises a complete compact-buffer lease tail");
    let releases = AtomicUsize::new(0);
    let payload = br#"{}"#;
    let mut handle = ReplayHandleV1(std::ptr::null_mut());
    let mut error = ByteSliceV1::EMPTY;

    assert_eq!(
        unsafe {
            tail.create_with_hash_buffer_leases.unwrap()(
                CreateRequestV1 {
                    struct_size: std::mem::size_of::<CreateRequestV1>() as u32,
                    flags: 0,
                    provider_payload: ByteSliceV1 {
                        data: payload.as_ptr(),
                        len: payload.len() as u64,
                    },
                },
                HashBufferLeaseCallbacksV1 {
                    struct_size: std::mem::size_of::<HashBufferLeaseCallbacksV1>() as u32,
                    flags: 0,
                    context: (&raw const releases).cast_mut().cast(),
                    release_hash_buffer: Some(count_hash_buffer_release),
                },
                &raw mut handle,
                &raw mut error,
            )
        },
        StatusV1::OK
    );
    let hashes = [201_u32, 202];
    let mut buffer_id = HashBufferIdV1::INVALID;
    assert_eq!(
        unsafe {
            tail.register_hash_buffer.unwrap()(
                handle,
                U32SliceV1 {
                    data: hashes.as_ptr(),
                    len: hashes.len() as u64,
                },
                &raw mut buffer_id,
            )
        },
        StatusV1::OK
    );
    let mut request_id = [0; 16];
    assert_eq!(
        unsafe {
            tail.submit_compact_hash_buffer_range.unwrap()(
                handle,
                compact_request(),
                HashBufferRangeV1 {
                    buffer_id,
                    offset: 0,
                    len: hashes.len() as u64,
                },
                &raw mut request_id,
            )
        },
        StatusV1::OK
    );

    unsafe { vtable.destroy.unwrap()(handle) };
    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

#[test]
fn copied_compact_submission_needs_no_lease_callbacks() {
    let descriptor = unsafe { &*aisimulate_steppable_plugin::aiperf_steppable_plugin_v1() };
    let vtable = unsafe { &*descriptor.vtable };
    let payload = br#"{}"#;
    let mut handle = ReplayHandleV1(std::ptr::null_mut());
    let mut error = ByteSliceV1::EMPTY;
    assert_eq!(
        unsafe {
            vtable.create.unwrap()(
                CreateRequestV1 {
                    struct_size: std::mem::size_of::<CreateRequestV1>() as u32,
                    flags: 0,
                    provider_payload: ByteSliceV1 {
                        data: payload.as_ptr(),
                        len: payload.len() as u64,
                    },
                },
                &raw mut handle,
                &raw mut error,
            )
        },
        StatusV1::OK
    );
    let hashes = [301_u32, 302];
    let mut request = compact_request();
    request.hash_ids = U32SliceV1 {
        data: hashes.as_ptr(),
        len: hashes.len() as u64,
    };
    let mut request_id = [0; 16];
    assert_eq!(
        unsafe { vtable.submit_compact.unwrap()(handle, request, &raw mut request_id) },
        StatusV1::OK
    );
    unsafe { vtable.destroy.unwrap()(handle) };
}

#[test]
fn backend_creates_and_destroys_a_default_replay() {
    let descriptor = unsafe { &*aisimulate_steppable_plugin::aiperf_steppable_plugin_v1() };
    let vtable = unsafe { &*descriptor.vtable };
    let payload = br#"{}"#;
    let mut handle = ReplayHandleV1(std::ptr::null_mut());
    let mut error = ByteSliceV1::EMPTY;
    let create = vtable.create.unwrap();

    assert_eq!(
        unsafe {
            create(
                CreateRequestV1 {
                    struct_size: std::mem::size_of::<CreateRequestV1>() as u32,
                    flags: REQUEST_FLAG_UUID,
                    provider_payload: ByteSliceV1 {
                        data: payload.as_ptr(),
                        len: payload.len() as u64,
                    },
                },
                &raw mut handle,
                &raw mut error,
            )
        },
        StatusV1::OK
    );
    assert!(!handle.0.is_null());
    assert_eq!(error.len, 0);

    let mut state = ReplayStateV1::EMPTY;
    assert_eq!(
        unsafe { vtable.state.unwrap()(handle, &raw mut state) },
        StatusV1::OK
    );
    assert_eq!(state.now_ms, 0.0);
    assert_eq!(state.in_flight, 0);
    assert_eq!(state.is_idle, 1);

    assert_eq!(
        unsafe { vtable.advance_now_ms.unwrap()(handle, 7.5) },
        StatusV1::OK
    );
    assert_eq!(
        unsafe { vtable.state.unwrap()(handle, &raw mut state) },
        StatusV1::OK
    );
    assert_eq!(state.now_ms, 7.5);

    let mut diagnostic = ByteSliceV1::EMPTY;
    assert_eq!(
        unsafe { vtable.last_error.unwrap()(handle, &raw mut diagnostic) },
        StatusV1::OK
    );
    assert_eq!(diagnostic.len, 0);

    assert_eq!(
        unsafe { vtable.set_capture_per_request.unwrap()(handle, 1) },
        StatusV1::OK
    );
    assert_eq!(
        unsafe {
            vtable.set_sla_thresholds.unwrap()(
                handle,
                SlaThresholdsV1 {
                    flags: SLA_FLAG_TTFT,
                    reserved: 0,
                    ttft_ms: 100.0,
                    itl_ms: 0.0,
                    e2e_ms: 0.0,
                },
            )
        },
        StatusV1::OK
    );

    let tokens = [1_u32, 2, 3];
    let mut request_id = [0; 16];
    assert_eq!(
        unsafe {
            vtable.submit.unwrap()(
                handle,
                DirectRequestV1 {
                    struct_size: std::mem::size_of::<DirectRequestV1>() as u32,
                    flags: REQUEST_FLAG_UUID,
                    tokens: U32SliceV1 {
                        data: tokens.as_ptr(),
                        len: tokens.len() as u64,
                    },
                    output_token_ids: U32SliceV1::EMPTY,
                    max_output_tokens: 1,
                    uuid: [9; 16],
                    dp_rank: 0,
                    preferred_dp_rank: 0,
                    preferred_prefill_dp_rank: 0,
                    arrival_timestamp_ms: 0.0,
                    priority: 0,
                    strict_priority: 0,
                    policy_class: ByteSliceV1::EMPTY,
                    replay_context: ReplayContextV1::EMPTY,
                },
                &raw mut request_id,
            )
        },
        StatusV1::OK
    );
    assert_eq!(request_id, [9; 16]);
    assert_eq!(
        unsafe { vtable.state.unwrap()(handle, &raw mut state) },
        StatusV1::OK
    );
    assert_eq!(state.in_flight, 1);

    let batch = [DirectRequestV1 {
        struct_size: std::mem::size_of::<DirectRequestV1>() as u32,
        flags: 0,
        tokens: U32SliceV1 {
            data: tokens.as_ptr(),
            len: tokens.len() as u64,
        },
        output_token_ids: U32SliceV1::EMPTY,
        max_output_tokens: 1,
        uuid: [0; 16],
        dp_rank: 0,
        preferred_dp_rank: 0,
        preferred_prefill_dp_rank: 0,
        arrival_timestamp_ms: 0.0,
        priority: 0,
        strict_priority: 0,
        policy_class: ByteSliceV1::EMPTY,
        replay_context: ReplayContextV1::EMPTY,
    }; 2];
    let mut batch_ids = [[0; 16]; 2];
    assert_eq!(
        unsafe {
            vtable.submit_batch.unwrap()(
                handle,
                DirectRequestSliceV1 {
                    data: batch.as_ptr(),
                    len: batch.len() as u64,
                },
                RequestIdMutSliceV1 {
                    data: batch_ids.as_mut_ptr(),
                    len: batch_ids.len() as u64,
                },
            )
        },
        StatusV1::OK
    );
    assert!(batch_ids.iter().all(|id| *id != [0; 16]));

    let mut cancellation = EngineEventV1 {
        request_id: [0; 16],
        flags: 0,
        token_id: 0,
        terminal_status: 0,
        reserved: 0,
    };
    let mut canceled = 0;
    assert_eq!(
        unsafe {
            vtable.cancel.unwrap()(
                handle,
                &raw const batch_ids[0],
                &raw mut cancellation,
                &raw mut canceled,
            )
        },
        StatusV1::OK
    );
    assert_eq!(canceled, 1);
    assert_eq!(cancellation.request_id, batch_ids[0]);
    assert_ne!(cancellation.flags & (1 << 1), 0);

    let mut canceled_batch = EngineEventSliceV1 {
        data: std::ptr::null(),
        len: 0,
    };
    assert_eq!(
        unsafe {
            vtable.cancel_batch.unwrap()(
                handle,
                RequestIdSliceV1 {
                    data: batch_ids[1..].as_ptr(),
                    len: 1,
                },
                &raw mut canceled_batch,
            )
        },
        StatusV1::OK
    );
    assert_eq!(canceled_batch.len, 1);
    unsafe { vtable.release_events.unwrap()(canceled_batch) };

    let mut stepped = StepResultV1::EMPTY;
    assert_eq!(
        unsafe {
            vtable.step.unwrap()(
                handle,
                StepRequestV1 {
                    struct_size: std::mem::size_of::<StepRequestV1>() as u32,
                    flags: 0,
                    until_ms: f64::INFINITY,
                },
                &raw mut stepped,
            )
        },
        StatusV1::OK
    );
    assert!(stepped.events.len > 0);
    assert!(stepped.request_facts.len > 0);
    unsafe { vtable.release_events.unwrap()(stepped.events) };
    unsafe { vtable.release_request_facts.unwrap()(stepped.request_facts) };

    for _ in 0..16 {
        assert_eq!(
            unsafe { vtable.state.unwrap()(handle, &raw mut state) },
            StatusV1::OK
        );
        if state.is_idle != 0 {
            break;
        }
        let mut more = StepResultV1::EMPTY;
        assert_eq!(
            unsafe {
                vtable.step.unwrap()(
                    handle,
                    StepRequestV1 {
                        struct_size: std::mem::size_of::<StepRequestV1>() as u32,
                        flags: 0,
                        until_ms: f64::INFINITY,
                    },
                    &raw mut more,
                )
            },
            StatusV1::OK
        );
        unsafe { vtable.release_events.unwrap()(more.events) };
        unsafe { vtable.release_request_facts.unwrap()(more.request_facts) };
    }
    assert_ne!(state.is_idle, 0);

    let mut report = ByteSliceV1::EMPTY;
    assert_eq!(
        unsafe { vtable.take_report.unwrap()(handle, 1.0, &raw mut report) },
        StatusV1::OK
    );
    let report_json = unsafe { std::slice::from_raw_parts(report.data, report.len as usize) };
    assert!(serde_json::from_slice::<serde_json::Value>(report_json).is_ok());
    unsafe { vtable.release_bytes.unwrap()(report) };

    unsafe { vtable.destroy.unwrap()(handle) };
}

#[test]
fn disaggregated_dynamic_placement_invalid_path_is_rejected_without_round_robin_fallback() {
    let descriptor = unsafe { &*aisimulate_steppable_plugin::aiperf_steppable_plugin_v1() };
    let vtable = unsafe { &*descriptor.vtable };
    let payload = br#"{
        "topology": "disaggregated",
        "dynamic_placement": {"library_path": "/not/used/libplacement.so"}
    }"#;
    let mut handle = ReplayHandleV1(std::ptr::null_mut());
    let mut error = ByteSliceV1::EMPTY;

    assert_eq!(
        unsafe {
            vtable.create.unwrap()(
                CreateRequestV1 {
                    struct_size: std::mem::size_of::<CreateRequestV1>() as u32,
                    flags: REQUEST_FLAG_UUID,
                    provider_payload: ByteSliceV1 {
                        data: payload.as_ptr(),
                        len: payload.len() as u64,
                    },
                },
                &raw mut handle,
                &raw mut error,
            )
        },
        StatusV1::REJECTED
    );
    assert!(handle.0.is_null());
    // Safety: a rejected create returns one owned diagnostic slice.
    let message = unsafe { std::slice::from_raw_parts(error.data, error.len as usize) };
    let message = std::str::from_utf8(message).expect("diagnostic is UTF-8");
    assert!(message.starts_with("loading placement plugin /not/used/libplacement.so"));
    unsafe { vtable.release_bytes.unwrap()(error) };
}
