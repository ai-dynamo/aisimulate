// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_steppable_abi::{
    ByteSliceV1, CreateRequestV1, DirectRequestSliceV1, DirectRequestV1, EngineEventSliceV1,
    EngineEventV1, PLUGIN_ABI_MAJOR_V1, REQUEST_FLAG_UUID, ReplayContextV1, ReplayHandleV1,
    ReplayStateV1, RequestIdMutSliceV1, RequestIdSliceV1, SLA_FLAG_TTFT, SlaThresholdsV1, StatusV1,
    StepRequestV1, StepResultV1, U32SliceV1, validate_descriptor_v1,
};

#[test]
fn plugin_entry_exposes_a_complete_v1_table() {
    let descriptor = aisimulate_steppable_plugin::aiperf_steppable_plugin_v1();

    assert!(!descriptor.is_null());
    let descriptor = unsafe { &*descriptor };
    assert_eq!(descriptor.abi_major, PLUGIN_ABI_MAJOR_V1);
    assert!(unsafe { validate_descriptor_v1(descriptor) }.is_ok());
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
