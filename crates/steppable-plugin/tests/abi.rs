// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_steppable_abi::{
    ByteSliceV1, CreateRequestV1, DirectRequestV1, PLUGIN_ABI_MAJOR_V1, ReplayContextV1,
    ReplayHandleV1, ReplayStateV1, StatusV1, StepRequestV1, StepResultV1, U32SliceV1,
    validate_descriptor_v1,
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

    let tokens = [1_u32, 2, 3];
    let mut request_id = [0; 16];
    assert_eq!(
        unsafe {
            vtable.submit.unwrap()(
                handle,
                DirectRequestV1 {
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
                },
                &raw mut request_id,
            )
        },
        StatusV1::OK
    );
    assert_ne!(request_id, [0; 16]);
    assert_eq!(
        unsafe { vtable.state.unwrap()(handle, &raw mut state) },
        StatusV1::OK
    );
    assert_eq!(state.in_flight, 1);

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

    unsafe { vtable.destroy.unwrap()(handle) };
}
