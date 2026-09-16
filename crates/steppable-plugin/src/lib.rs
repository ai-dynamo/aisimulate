// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamically loaded AISimulate implementation of AIPerf's Steppable ABI.
//!
//! Configuration crosses the boundary only once, when a replay is created.
//! Request, event, and measurement records use the ABI crate's fixed-layout
//! data-plane records.

use std::ffi::c_char;

use aiperf_steppable_abi::{
    ByteSliceV1, CreateRequestV1, DirectRequestSliceV1, DirectRequestV1, EngineEventSliceV1,
    EngineEventV1, PluginDescriptorV1, PluginVTableV1, ReplayHandleV1, ReplayStateV1,
    RequestFactSliceV1, RequestIdMutSliceV1, RequestIdSliceV1, RequestIdV1, SlaThresholdsV1,
    StatusV1, StepRequestV1, StepResultV1, U32SliceV1,
};
use aisimulate_core::replay::loadgen::{
    SteppableAgg, SteppableDisagg, SteppableEngine, SteppableReplay,
};
use aisimulate_core::replay::{DirectRequest, ReplayEngineConfig, ReplayEngineFactory};
use serde::Deserialize;

/// Topology built by the backend for one steppable replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendTopology {
    /// One aggregate worker using the single-worker steppable engine.
    #[default]
    Single,
    /// One or more aggregate workers using round-robin placement.
    Aggregated,
    /// Separate round-robin prefill and decode pools.
    Disaggregated,
}

/// Provider-owned configuration encoded in `CreateRequestV1::provider_payload`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendConfig {
    /// Backend topology selected before the replay is constructed.
    pub topology: BackendTopology,
    /// Scheduler configuration applied to the selected topology.
    pub engine: ReplayEngineConfig,
    /// Aggregate worker count.
    pub workers: usize,
    /// Prefill worker count for a disaggregated replay.
    pub prefill_workers: usize,
    /// Decode worker count for a disaggregated replay.
    pub decode_workers: usize,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            topology: BackendTopology::Single,
            engine: ReplayEngineConfig::default(),
            workers: 1,
            prefill_workers: 1,
            decode_workers: 1,
        }
    }
}

const PROVIDER_ID: &[u8] = b"aisimulate\0";

struct BackendReplay {
    engine: Box<dyn SteppableReplay>,
    last_error: String,
}

fn allocated_bytes(value: String) -> ByteSliceV1 {
    let bytes = value.into_bytes().into_boxed_slice();
    let len = bytes.len() as u64;
    let data = Box::into_raw(bytes).cast::<u8>();
    ByteSliceV1 { data, len }
}

unsafe fn backend_mut(handle: ReplayHandleV1) -> Result<&'static mut BackendReplay, StatusV1> {
    if handle.0.is_null() {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    // Safety: every non-null handle comes from `create`, and ownership stays
    // with the caller until `destroy` consumes it.
    Ok(unsafe { &mut *handle.0.cast::<BackendReplay>() })
}

unsafe fn borrowed_tokens(slice: U32SliceV1) -> Result<&'static [u32], StatusV1> {
    if slice.len > usize::MAX as u64 || (slice.data.is_null() && slice.len != 0) {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    if slice.len == 0 {
        return Ok(&[]);
    }
    // Safety: a non-empty ABI input slice is borrowed and valid for the call.
    Ok(unsafe { std::slice::from_raw_parts(slice.data, slice.len as usize) })
}

unsafe fn direct_request(request: DirectRequestV1) -> Result<DirectRequest, StatusV1> {
    if request.struct_size as usize != std::mem::size_of::<DirectRequestV1>()
        || request.max_output_tokens > usize::MAX as u64
    {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    // Safety: input slices are caller-owned and valid for this FFI call.
    let tokens = unsafe { borrowed_tokens(request.tokens) }?.to_vec();
    // Safety: input slices are caller-owned and valid for this FFI call.
    let output_token_ids = unsafe { borrowed_tokens(request.output_token_ids) }?.to_vec();
    Ok(DirectRequest {
        tokens,
        max_output_tokens: request.max_output_tokens as usize,
        output_token_ids: (!output_token_ids.is_empty()).then_some(output_token_ids),
        uuid: None,
        dp_rank: request.dp_rank,
        preferred_dp_rank: None,
        preferred_prefill_dp_rank: None,
        arrival_timestamp_ms: None,
        priority: request.priority,
        strict_priority: request.strict_priority,
        policy_class: None,
        replay_context: None,
    })
}

unsafe extern "C" fn create(
    request: CreateRequestV1,
    handle: *mut ReplayHandleV1,
    error: *mut ByteSliceV1,
) -> StatusV1 {
    if handle.is_null() || error.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: pointers were validated above and outputs are written once.
    unsafe {
        *handle = ReplayHandleV1(std::ptr::null_mut());
        *error = ByteSliceV1::EMPTY;
    }
    if request.provider_payload.len > usize::MAX as u64
        || (request.provider_payload.data.is_null() && request.provider_payload.len != 0)
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: the ABI guarantees borrowed input bytes remain valid for this
    // call; null is accepted only for an empty slice as checked above.
    let payload = if request.provider_payload.len == 0 {
        &[]
    } else {
        // Safety: a non-empty ABI input slice is valid for this call.
        unsafe {
            std::slice::from_raw_parts(
                request.provider_payload.data.cast::<u8>(),
                request.provider_payload.len as usize,
            )
        }
    };
    let config: BackendConfig = match serde_json::from_slice(payload) {
        Ok(config) => config,
        Err(parse_error) => {
            // Safety: validated non-null output pointer.
            unsafe { *error = allocated_bytes(parse_error.to_string()) };
            return StatusV1::REJECTED;
        }
    };
    let factory = ReplayEngineFactory::new();
    let created: anyhow::Result<Box<dyn SteppableReplay>> = match config.topology {
        BackendTopology::Single => SteppableEngine::new(config.engine, &factory)
            .map(|engine| Box::new(engine) as Box<dyn SteppableReplay>),
        BackendTopology::Aggregated => SteppableAgg::new(config.engine, &factory, config.workers)
            .map(|engine| Box::new(engine) as Box<dyn SteppableReplay>),
        BackendTopology::Disaggregated => SteppableDisagg::new(
            config.engine,
            &factory,
            config.prefill_workers,
            config.decode_workers,
        )
        .map(|engine| Box::new(engine) as Box<dyn SteppableReplay>),
    };
    match created {
        Ok(engine) => {
            let replay = Box::new(BackendReplay {
                engine,
                last_error: String::new(),
            });
            // Safety: validated non-null output pointer.
            unsafe { *handle = ReplayHandleV1(Box::into_raw(replay).cast()) };
            StatusV1::OK
        }
        Err(build_error) => {
            // Safety: validated non-null output pointer.
            unsafe { *error = allocated_bytes(build_error.to_string()) };
            StatusV1::REJECTED
        }
    }
}

unsafe extern "C" fn submit(
    handle: ReplayHandleV1,
    request: DirectRequestV1,
    request_id: *mut RequestIdV1,
) -> StatusV1 {
    if request_id.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: request slices are valid for this call as required by the ABI.
    let request = match unsafe { direct_request(request) } {
        Ok(request) => request,
        Err(status) => return status,
    };
    // Safety: the handle is only dereferenced after null validation.
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    match replay.engine.submit(request) {
        Ok(uuid) => {
            // Safety: validated non-null output pointer.
            unsafe { *request_id = *uuid.as_bytes() };
            StatusV1::OK
        }
        Err(error) => {
            replay.last_error = error.to_string();
            StatusV1::REJECTED
        }
    }
}

unsafe extern "C" fn submit_batch(
    _handle: ReplayHandleV1,
    _requests: DirectRequestSliceV1,
    _request_ids: RequestIdMutSliceV1,
) -> StatusV1 {
    StatusV1::UNSUPPORTED
}

unsafe extern "C" fn cancel(
    _handle: ReplayHandleV1,
    _request_id: *const RequestIdV1,
    _event: *mut EngineEventV1,
    _canceled: *mut u8,
) -> StatusV1 {
    StatusV1::UNSUPPORTED
}

unsafe extern "C" fn cancel_batch(
    _handle: ReplayHandleV1,
    _request_ids: RequestIdSliceV1,
    _events: *mut EngineEventSliceV1,
) -> StatusV1 {
    StatusV1::UNSUPPORTED
}

unsafe extern "C" fn step(
    _handle: ReplayHandleV1,
    _request: StepRequestV1,
    _result: *mut StepResultV1,
) -> StatusV1 {
    StatusV1::UNSUPPORTED
}

unsafe extern "C" fn take_report(
    _handle: ReplayHandleV1,
    _wall_ms: f64,
    _report: *mut ByteSliceV1,
) -> StatusV1 {
    StatusV1::UNSUPPORTED
}

unsafe extern "C" fn release_bytes(bytes: ByteSliceV1) {
    if bytes.data.is_null() {
        return;
    }
    if bytes.len > usize::MAX as u64 {
        return;
    }
    // Safety: every non-empty output byte slice is allocated by
    // `allocated_bytes` as an exact-length boxed slice.
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            bytes.data.cast_mut(),
            bytes.len as usize,
        )));
    }
}
unsafe extern "C" fn release_events(_events: EngineEventSliceV1) {}
unsafe extern "C" fn release_request_facts(_facts: RequestFactSliceV1) {}

unsafe extern "C" fn state(handle: ReplayHandleV1, state: *mut ReplayStateV1) -> StatusV1 {
    if state.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: the non-null handle was created by this plugin and remains live
    // until the caller invokes `destroy`.
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    let next_event_ms = replay.engine.next_event_ms().unwrap_or(f64::NAN);
    // Safety: validated non-null output pointer.
    unsafe {
        *state = ReplayStateV1 {
            now_ms: replay.engine.now_ms(),
            next_event_ms,
            in_flight: replay.engine.in_flight() as u64,
            is_idle: u8::from(replay.engine.is_idle()),
            reserved: [0; 7],
        };
    }
    StatusV1::OK
}

unsafe extern "C" fn advance_now_ms(handle: ReplayHandleV1, now_ms: f64) -> StatusV1 {
    if !now_ms.is_finite() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: the handle is only dereferenced after null validation in
    // `backend_mut`.
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    replay.engine.advance_now_ms(now_ms);
    StatusV1::OK
}

unsafe extern "C" fn set_capture_per_request(_handle: ReplayHandleV1, _capture: u8) -> StatusV1 {
    StatusV1::UNSUPPORTED
}

unsafe extern "C" fn set_sla_thresholds(
    _handle: ReplayHandleV1,
    _thresholds: SlaThresholdsV1,
) -> StatusV1 {
    StatusV1::UNSUPPORTED
}

unsafe extern "C" fn last_error(_handle: ReplayHandleV1, _error: *mut ByteSliceV1) -> StatusV1 {
    StatusV1::UNSUPPORTED
}

unsafe extern "C" fn destroy(handle: ReplayHandleV1) {
    if handle.0.is_null() {
        return;
    }
    // Safety: caller transfers the unique handle returned by `create`.
    unsafe { drop(Box::from_raw(handle.0.cast::<BackendReplay>())) };
}

static VTABLE: PluginVTableV1 = PluginVTableV1 {
    struct_size: std::mem::size_of::<PluginVTableV1>() as u32,
    flags: 0,
    create: Some(create),
    submit: Some(submit),
    submit_batch: Some(submit_batch),
    cancel: Some(cancel),
    cancel_batch: Some(cancel_batch),
    step: Some(step),
    take_report: Some(take_report),
    release_bytes: Some(release_bytes),
    release_events: Some(release_events),
    release_request_facts: Some(release_request_facts),
    state: Some(state),
    advance_now_ms: Some(advance_now_ms),
    set_capture_per_request: Some(set_capture_per_request),
    set_sla_thresholds: Some(set_sla_thresholds),
    last_error: Some(last_error),
    destroy: Some(destroy),
};

static DESCRIPTOR: PluginDescriptorV1 = PluginDescriptorV1 {
    abi_major: 1,
    abi_minor: 0,
    struct_size: std::mem::size_of::<PluginDescriptorV1>() as u32,
    flags: 0,
    capabilities: 0,
    provider_id: PROVIDER_ID.as_ptr().cast::<c_char>(),
    vtable: &VTABLE,
};

/// Returns the static V1 plugin descriptor for dynamic loading.
#[unsafe(no_mangle)]
pub extern "C" fn aiperf_steppable_plugin_v1() -> *const PluginDescriptorV1 {
    &DESCRIPTOR
}
