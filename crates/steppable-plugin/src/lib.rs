// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamically loaded AISimulate implementation of AIPerf's Steppable ABI.
//!
//! Configuration crosses the boundary only once, when a replay is created.
//! Request, event, and measurement records use the ABI crate's fixed-layout
//! data-plane records.
//!
//! The library exports one symbol, `aiperf_steppable_plugin_v1`, returning a
//! static descriptor whose provider ID is `aisimulate`. Cargo features select
//! which optional capabilities the descriptor advertises; see `Cargo.toml`.
//!
//! This is AISimulate's own steppable producer (first written as
//! `crates/steppable-plugin/src/lib.rs` at
//! `547ccf24f9b49a9b94d39c73609a1abdd8125c69`), brought to the current engine
//! by AIPerf's provider fixture at NVIDIA-dev/aiperf-rs
//! `539e4819f440cce0773fe6288341c7dc815425c3`,
//! `crates/engine/simulate/tests/support/aisimulate-provider/src/lib.rs`, and
//! returned here as the published artifact. NVIDIA copyright, Apache-2.0.
//! It decodes the complete request metadata of the pinned ABI and excludes the
//! Dynamo placement plugin, which this engine branch does not carry.
//! Opaque provenance that is not JSON is rejected before engine admission:
//! the core owns JSON provenance and cannot preserve arbitrary bytes.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use aiperf_steppable_abi::{
    BackendDistributionStatsV1, BackendFinalSummaryV1, BackendGoodputStatsV1,
    BackendLatencyStatsV1, BackendRequestCountsV1, BackendThroughputStatsV1, ByteMutSliceV1,
    ByteSliceV1, CAPABILITY_COMPACT_BUFFER_LEASES_V1, CAPABILITY_COMPACT_REQUEST_V1,
    CAPABILITY_FINAL_REPORT_BUNDLE_V1, CAPABILITY_REQUEST_REUSE_FACTS_V1,
    CAPABILITY_TIMESTAMPED_EVENTS_V1, CompactRequestV1, CreateRequestV1, DirectRequestSliceV1,
    DirectRequestV1, EngineEventSliceV1, EngineEventV1, FINAL_ARTIFACT_KIND_PER_REQUEST_JSONL_V1,
    FINAL_ARTIFACT_KIND_REPORT_JSON_V1, FinalArtifactChunkV1, FinalArtifactMetadataV1,
    HashBufferIdV1, HashBufferLeaseCallbacksV1, HashBufferRangeV1,
    MAX_REPLAY_CONTEXT_METADATA_BYTES_V1, PluginDescriptorV1, PluginVTableV1,
    REPLAY_CONTEXT_FLAG_METADATA, REPLAY_CONTEXT_FLAG_SESSION_ID, REPLAY_CONTEXT_FLAG_TURN_INDEX,
    REQUEST_FACT_FLAG_ADMISSION, REQUEST_FACT_FLAG_LATENCIES, REQUEST_FACT_FLAG_MAX_REUSE,
    REQUEST_FACT_FLAG_OUTPUT_LENGTH, REQUEST_FLAG_ARRIVAL_TIMESTAMP, REQUEST_FLAG_OUTPUT_TOKEN_IDS,
    REQUEST_FLAG_POLICY_CLASS, REQUEST_FLAG_PREFERRED_DP_RANK,
    REQUEST_FLAG_PREFERRED_PREFILL_DP_RANK, REQUEST_FLAG_REPLAY_CONTEXT, REQUEST_FLAG_UUID,
    ReplayContextV1, ReplayHandleV1, ReplayStateV1, RequestFactSliceV1, RequestFactV1,
    RequestIdMutSliceV1, RequestIdSliceV1, RequestIdV1, SLA_FLAG_E2E, SLA_FLAG_ITL, SLA_FLAG_TTFT,
    SlaThresholdsV1, StatusV1, StepRequestV1, StepResultV1, TimestampedEngineEventSliceV1,
    TimestampedEngineEventV1, TimestampedStepResultV1, U32SliceV1,
};
use aisimulate_core::replay::loadgen::steppable::{
    CompactDirectRequest, CompactHashIdsLease, SteppableAgg, SteppableDisagg, SteppableEngine,
    SteppableReplay,
};
use aisimulate_core::replay::{
    DirectRequest, ReplayEngineConfig, ReplayEngineFactory, ReplayPromptTokenSource,
    ReplayRequestContext, ReplayTerminalStatus, SlaThresholds,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// Topology built by the backend for one steppable replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendTopology {
    /// One aggregate worker using the single-worker steppable engine.
    #[default]
    Single,
    /// One or more aggregate workers using the selected placement policy.
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
    poisoned: bool,
    lease_callbacks: Option<HashBufferLeaseCallbacksV1>,
    hash_buffers: HashMap<HashBufferIdV1, RegisteredHashBuffer>,
    next_hash_buffer_id: u64,
    finalization: Finalization,
}

enum Finalization {
    Open,
    Finalizing,
    LegacyFinalized,
    Complete(FinalArtifacts),
    Poisoned,
}

struct FinalArtifacts {
    report_json: FinalArtifact,
    per_request_jsonl: FinalArtifact,
}

struct FinalArtifact {
    bytes: Vec<u8>,
    sha256: [u8; 32],
}

#[derive(Clone, Copy)]
struct RegisteredHashBuffer {
    data: *const u32,
    len: usize,
}

/// Safe core-facing owner for one validated host buffer range. The FFI pointer
/// never escapes this plugin: core sees only the `CompactHashIdsLease` trait.
#[derive(Debug)]
struct HostHashBufferLease {
    data: *const u32,
    len: usize,
    buffer_id: HashBufferIdV1,
    callbacks: HashBufferLeaseCallbacksV1,
    accepted: AtomicBool,
}

// Safety: this test provider is paired with AIP's synchronized release registry.
// The hash range is immutable and retained by that registry until its one release
// callback. Moving/sharing the owner cannot invalidate the range; the callback
// remains valid until the engine is destroyed, before the host registry drops.
unsafe impl Send for HostHashBufferLease {}
unsafe impl Sync for HostHashBufferLease {}

impl CompactHashIdsLease for HostHashBufferLease {
    fn hash_ids(&self) -> &[u32] {
        // Safety: range validation happened before this owner was constructed.
        // The creation callback contract keeps the immutable host buffer alive
        // until this owner is dropped after terminal, cancellation, or replay
        // destruction.
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }

    fn on_accepted(&self) {
        self.accepted.store(true, Ordering::Release);
    }
}

impl Drop for HostHashBufferLease {
    fn drop(&mut self) {
        if !self.accepted.swap(false, Ordering::AcqRel) {
            return;
        }
        let release = self
            .callbacks
            .release_hash_buffer
            .expect("lease creation validated the release callback");
        // Safety: the ABI requires this validated host callback not to panic
        // or re-enter the plugin. A lease has one final owner, so this is its
        // sole release call.
        unsafe { release(self.callbacks.context, self.buffer_id) };
    }
}

fn allocated_bytes(value: String) -> ByteSliceV1 {
    let bytes = value.into_bytes().into_boxed_slice();
    let len = bytes.len() as u64;
    let data = Box::into_raw(bytes).cast::<u8>();
    ByteSliceV1 { data, len }
}

unsafe fn backend_mut_even_if_poisoned(
    handle: ReplayHandleV1,
) -> Result<&'static mut BackendReplay, StatusV1> {
    if handle.0.is_null() {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    // Safety: every non-null handle comes from `create`, and ownership stays
    // with the caller until `destroy` consumes it.
    Ok(unsafe { &mut *handle.0.cast::<BackendReplay>() })
}

unsafe fn backend_mut(handle: ReplayHandleV1) -> Result<&'static mut BackendReplay, StatusV1> {
    let replay = unsafe { backend_mut_even_if_poisoned(handle) }?;
    if replay.poisoned {
        return Err(StatusV1::INTERNAL);
    }
    Ok(replay)
}

unsafe fn backend_mut_for_mutation(
    handle: ReplayHandleV1,
) -> Result<&'static mut BackendReplay, StatusV1> {
    let replay = unsafe { backend_mut(handle) }?;
    if !matches!(replay.finalization, Finalization::Open) {
        replay.last_error = "replay report was already finalized".to_owned();
        return Err(StatusV1::REJECTED);
    }
    Ok(replay)
}

fn panic_status(handle: ReplayHandleV1) -> StatusV1 {
    // Error recording is best-effort: containment is the required C-ABI
    // guarantee. Do not let recording the original panic trigger another.
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Ok(replay) = unsafe { backend_mut_even_if_poisoned(handle) } {
            replay.poisoned = true;
            replay.last_error = "panic contained at AISimulate steppable ABI boundary".to_owned();
        }
    }));
    StatusV1::INTERNAL
}

fn batch_panic_status(handle: ReplayHandleV1) -> StatusV1 {
    // A batch unwind can follow a stateful placement or replay mutation. The
    // outer provider therefore owns a fail-stop latch independent of whether
    // the inner replay had an opportunity to return its poisoned error type.
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Ok(replay) = unsafe { backend_mut_even_if_poisoned(handle) } {
            replay.poisoned = true;
            replay.last_error =
                "panic contained after AISimulate steppable batch mutation; replay is poisoned"
                    .to_owned();
        }
    }));
    StatusV1::INTERNAL
}

fn finalization_panic_status(handle: ReplayHandleV1) -> StatusV1 {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Ok(replay) = unsafe { backend_mut_even_if_poisoned(handle) } {
            replay.poisoned = true;
            replay.finalization = Finalization::Poisoned;
            replay.last_error =
                "panic contained while finalizing AISimulate steppable report; replay is poisoned"
                    .to_owned();
        }
    }));
    StatusV1::INTERNAL
}

fn catch_status(operation: impl FnOnce() -> StatusV1, handle: ReplayHandleV1) -> StatusV1 {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(status) => status,
        Err(_) => panic_status(handle),
    }
}

fn ffi_slice_len_is_valid<T>(len: u64) -> bool {
    let Ok(len) = usize::try_from(len) else {
        return false;
    };
    let element_size = std::mem::size_of::<T>();
    element_size == 0 || len <= isize::MAX as usize / element_size
}

unsafe fn borrowed_tokens<'a>(slice: U32SliceV1) -> Result<&'a [u32], StatusV1> {
    if !ffi_slice_len_is_valid::<u32>(slice.len)
        || (slice.data.is_null() && slice.len != 0)
        || (!slice.data.is_null() && !slice.data.is_aligned())
    {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    if slice.len == 0 {
        return Ok(&[]);
    }
    // Safety: a non-empty ABI input slice is borrowed and valid for the call.
    Ok(unsafe { std::slice::from_raw_parts(slice.data, slice.len as usize) })
}

unsafe fn borrowed_bytes<'a>(slice: ByteSliceV1) -> Result<&'a [u8], StatusV1> {
    if !ffi_slice_len_is_valid::<u8>(slice.len)
        || slice.len > MAX_REPLAY_CONTEXT_METADATA_BYTES_V1 as u64
        || (slice.data.is_null() && slice.len != 0)
    {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    if slice.len == 0 {
        return Ok(&[]);
    }
    // Safety: the caller keeps this bounded borrowed range live during the call.
    Ok(unsafe { std::slice::from_raw_parts(slice.data, slice.len as usize) })
}

unsafe fn owned_string(slice: ByteSliceV1) -> Result<String, StatusV1> {
    let bytes = unsafe { borrowed_bytes(slice) }?;
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|_| StatusV1::INVALID_ARGUMENT)
}

unsafe fn replay_context(context: ReplayContextV1) -> Result<ReplayRequestContext, StatusV1> {
    let known_flags = REPLAY_CONTEXT_FLAG_SESSION_ID
        | REPLAY_CONTEXT_FLAG_TURN_INDEX
        | REPLAY_CONTEXT_FLAG_METADATA;
    if context.struct_size as usize != std::mem::size_of::<ReplayContextV1>()
        || context.flags & !known_flags != 0
        || context.reserved != 0
    {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    let prompt_token_source = match context.prompt_token_source {
        0 => ReplayPromptTokenSource::Materialized,
        1 => ReplayPromptTokenSource::LengthOnlySynthetic,
        _ => return Err(StatusV1::UNSUPPORTED),
    };
    let metadata = if context.flags & REPLAY_CONTEXT_FLAG_METADATA != 0 {
        // The ABI permits opaque bytes; the pinned core only owns JSON values.
        // Reject unsupported encodings, never silently discard provenance.
        serde_json::from_slice(unsafe { borrowed_bytes(context.metadata) }?)
            .map_err(|_| StatusV1::UNSUPPORTED)?
    } else {
        serde_json::Value::Null
    };
    Ok(ReplayRequestContext {
        authored_id: unsafe { owned_string(context.authored_id) }?,
        session_id: if context.flags & REPLAY_CONTEXT_FLAG_SESSION_ID != 0 {
            Some(unsafe { owned_string(context.session_id) }?)
        } else {
            None
        },
        turn_index: if context.flags & REPLAY_CONTEXT_FLAG_TURN_INDEX != 0 {
            Some(usize::try_from(context.turn_index).map_err(|_| StatusV1::INVALID_ARGUMENT)?)
        } else {
            None
        },
        metadata,
        prompt_token_source,
        agentic: None,
    })
}

unsafe fn direct_request(request: DirectRequestV1) -> Result<DirectRequest, StatusV1> {
    let known_flags = REQUEST_FLAG_UUID
        | REQUEST_FLAG_OUTPUT_TOKEN_IDS
        | REQUEST_FLAG_PREFERRED_DP_RANK
        | REQUEST_FLAG_PREFERRED_PREFILL_DP_RANK
        | REQUEST_FLAG_ARRIVAL_TIMESTAMP
        | REQUEST_FLAG_POLICY_CLASS
        | REQUEST_FLAG_REPLAY_CONTEXT;
    if request.struct_size as usize != std::mem::size_of::<DirectRequestV1>()
        || request.max_output_tokens > usize::MAX as u64
        || request.flags & !known_flags != 0
        || (request.flags & REQUEST_FLAG_ARRIVAL_TIMESTAMP != 0
            && !request.arrival_timestamp_ms.is_finite())
        || (request.flags & REQUEST_FLAG_OUTPUT_TOKEN_IDS == 0 && request.output_token_ids.len != 0)
    {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    // Safety: all borrowed data is copied before engine mutation or returning.
    let tokens = unsafe { borrowed_tokens(request.tokens) }?.to_vec();
    let output_token_ids = if request.flags & REQUEST_FLAG_OUTPUT_TOKEN_IDS != 0 {
        // Presence is independent of cardinality: Some([]) means zero outputs.
        Some(unsafe { borrowed_tokens(request.output_token_ids) }?.to_vec())
    } else {
        None
    };
    let policy_class = if request.flags & REQUEST_FLAG_POLICY_CLASS != 0 {
        Some(unsafe { owned_string(request.policy_class) }?)
    } else {
        None
    };
    let replay_context = if request.flags & REQUEST_FLAG_REPLAY_CONTEXT != 0 {
        Some(unsafe { replay_context(request.replay_context) }?)
    } else {
        None
    };
    Ok(DirectRequest {
        tokens,
        max_output_tokens: request.max_output_tokens as usize,
        output_token_ids,
        uuid: (request.flags & REQUEST_FLAG_UUID != 0).then_some(Uuid::from_bytes(request.uuid)),
        dp_rank: request.dp_rank,
        preferred_dp_rank: (request.flags & REQUEST_FLAG_PREFERRED_DP_RANK != 0)
            .then_some(request.preferred_dp_rank),
        preferred_prefill_dp_rank: (request.flags & REQUEST_FLAG_PREFERRED_PREFILL_DP_RANK != 0)
            .then_some(request.preferred_prefill_dp_rank),
        arrival_timestamp_ms: (request.flags & REQUEST_FLAG_ARRIVAL_TIMESTAMP != 0)
            .then_some(request.arrival_timestamp_ms),
        priority: request.priority,
        strict_priority: request.strict_priority,
        policy_class,
        replay_context,
        kv_hints: None,
        images: Vec::new(),
    })
}

unsafe fn compact_request(request: CompactRequestV1) -> Result<CompactDirectRequest, StatusV1> {
    if request.struct_size as usize != std::mem::size_of::<CompactRequestV1>()
        || request.flags != 0
        || request.reserved != 0
        || request.input_token_count > usize::MAX as u64
        || request.trace_block_size == 0
    {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    let direct = unsafe { direct_request(request.request) }?;
    if !direct.tokens.is_empty() {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    let hash_ids = unsafe { borrowed_tokens(request.hash_ids) }?.to_vec();
    CompactDirectRequest::new(
        direct,
        request.input_token_count as usize,
        hash_ids,
        request.trace_block_size as usize,
    )
    .map_err(|_| StatusV1::INVALID_ARGUMENT)
}

unsafe fn compact_request_from_hash_buffer_range(
    request: CompactRequestV1,
    lease: Arc<dyn CompactHashIdsLease>,
) -> Result<CompactDirectRequest, StatusV1> {
    if request.struct_size as usize != std::mem::size_of::<CompactRequestV1>()
        || request.flags != 0
        || request.reserved != 0
        || request.input_token_count > usize::MAX as u64
        || request.trace_block_size == 0
        || request.hash_ids.len != 0
        || !request.hash_ids.data.is_null()
    {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    let direct = unsafe { direct_request(request.request) }?;
    if !direct.tokens.is_empty() {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    CompactDirectRequest::from_hash_ids_lease(
        direct,
        request.input_token_count as usize,
        lease,
        request.trace_block_size as usize,
    )
    .map_err(|_| StatusV1::INVALID_ARGUMENT)
}

unsafe fn create_impl(
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
    if request.struct_size as usize != std::mem::size_of::<CreateRequestV1>()
        || request.flags != 0
        || request.provider_payload.len > 16 * 1024 * 1024
        || !ffi_slice_len_is_valid::<u8>(request.provider_payload.len)
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
                poisoned: false,
                lease_callbacks: None,
                hash_buffers: HashMap::new(),
                next_hash_buffer_id: 1,
                finalization: Finalization::Open,
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

unsafe extern "C" fn create(
    request: CreateRequestV1,
    handle: *mut ReplayHandleV1,
    error: *mut ByteSliceV1,
) -> StatusV1 {
    if !handle.is_null() {
        // Safety: the non-null caller-owned output is valid for this call.
        unsafe { *handle = ReplayHandleV1(std::ptr::null_mut()) };
    }
    if !error.is_null() {
        // Safety: the non-null caller-owned output is valid for this call.
        unsafe { *error = ByteSliceV1::EMPTY };
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        create_impl(request, handle, error)
    })) {
        Ok(status) => status,
        Err(_) => StatusV1::INTERNAL,
    }
}

unsafe extern "C" fn create_with_hash_buffer_leases(
    request: CreateRequestV1,
    callbacks: HashBufferLeaseCallbacksV1,
    handle: *mut ReplayHandleV1,
    error: *mut ByteSliceV1,
) -> StatusV1 {
    if !handle.is_null() {
        // Safety: the non-null caller-owned output is valid for this call.
        unsafe { *handle = ReplayHandleV1(std::ptr::null_mut()) };
    }
    if !error.is_null() {
        // Safety: the non-null caller-owned output is valid for this call.
        unsafe { *error = ByteSliceV1::EMPTY };
    }
    if !callbacks.has_release_callback() {
        return StatusV1::INVALID_ARGUMENT;
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        let status = create_impl(request, handle, error);
        if status != StatusV1::OK {
            return status;
        }
        // Safety: `create_impl` returned a non-null replay handle on success.
        let replay = match backend_mut(*handle) {
            Ok(replay) => replay,
            Err(_) => return StatusV1::INTERNAL,
        };
        replay.lease_callbacks = Some(callbacks);
        StatusV1::OK
    })) {
        Ok(status) => status,
        Err(_) => StatusV1::INTERNAL,
    }
}

unsafe fn register_hash_buffer_impl(
    handle: ReplayHandleV1,
    hash_ids: U32SliceV1,
    buffer_id: *mut HashBufferIdV1,
) -> StatusV1 {
    if buffer_id.is_null()
        || hash_ids.len == 0
        || !ffi_slice_len_is_valid::<u32>(hash_ids.len)
        || hash_ids.data.is_null()
        || !hash_ids.data.is_aligned()
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    let replay = match unsafe { backend_mut_for_mutation(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    if replay.lease_callbacks.is_none() {
        return StatusV1::UNSUPPORTED;
    }
    let Some(next) = replay.next_hash_buffer_id.checked_add(1) else {
        replay.last_error = "compact hash-buffer identifier space exhausted".to_owned();
        return StatusV1::REJECTED;
    };
    let registered = HashBufferIdV1(replay.next_hash_buffer_id);
    replay.next_hash_buffer_id = next;
    replay.hash_buffers.insert(
        registered,
        RegisteredHashBuffer {
            data: hash_ids.data,
            len: hash_ids.len as usize,
        },
    );
    // Safety: validated non-null output pointer.
    unsafe { *buffer_id = registered };
    StatusV1::OK
}

unsafe extern "C" fn register_hash_buffer(
    handle: ReplayHandleV1,
    hash_ids: U32SliceV1,
    buffer_id: *mut HashBufferIdV1,
) -> StatusV1 {
    if !buffer_id.is_null() {
        // Safety: the non-null caller-owned output is valid for this call.
        unsafe { *buffer_id = HashBufferIdV1::INVALID };
    }
    catch_status(
        || unsafe { register_hash_buffer_impl(handle, hash_ids, buffer_id) },
        handle,
    )
}

unsafe fn submit_compact_hash_buffer_range_impl(
    handle: ReplayHandleV1,
    request: CompactRequestV1,
    range: HashBufferRangeV1,
    request_id: *mut RequestIdV1,
) -> StatusV1 {
    if request_id.is_null()
        || !range.is_valid()
        || range.offset > usize::MAX as u64
        || range.len > usize::MAX as u64
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    let replay = match unsafe { backend_mut_for_mutation(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    if replay.lease_callbacks.is_none() {
        return StatusV1::UNSUPPORTED;
    }
    let Some(buffer) = replay.hash_buffers.get(&range.buffer_id).copied() else {
        return StatusV1::INVALID_ARGUMENT;
    };
    let offset = range.offset as usize;
    let len = range.len as usize;
    let Some(end) = offset.checked_add(len) else {
        return StatusV1::INVALID_ARGUMENT;
    };
    if end > buffer.len {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: `offset..end` was proven to lie in the immutable registered
    // host range. Its raw representation stays inside this FFI adapter; core
    // receives an `Arc` lifecycle owner instead.
    let data = unsafe { buffer.data.add(offset) };
    let callbacks = replay
        .lease_callbacks
        .expect("lease-enabled replay checked callbacks above");
    let lease: Arc<dyn CompactHashIdsLease> = Arc::new(HostHashBufferLease {
        data,
        len,
        buffer_id: range.buffer_id,
        callbacks,
        accepted: AtomicBool::new(false),
    });
    let request = match unsafe { compact_request_from_hash_buffer_range(request, lease) } {
        Ok(request) => request,
        Err(status) => return status,
    };
    match replay.engine.submit_compact(request) {
        Ok(uuid) => {
            // An accepted range consumes its registration. The host may free
            // or recycle the backing storage as soon as the one release
            // callback arrives, so a stale buffer ID must never form another
            // lease for this replay lifetime.
            let removed = replay.hash_buffers.remove(&range.buffer_id);
            debug_assert!(
                removed.is_some(),
                "validated buffer registration disappeared"
            );
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

unsafe extern "C" fn submit_compact_hash_buffer_range(
    handle: ReplayHandleV1,
    request: CompactRequestV1,
    range: HashBufferRangeV1,
    request_id: *mut RequestIdV1,
) -> StatusV1 {
    if !request_id.is_null() {
        // Safety: the non-null caller-owned output is valid for this call.
        unsafe { *request_id = [0; 16] };
    }
    catch_status(
        || unsafe { submit_compact_hash_buffer_range_impl(handle, request, range, request_id) },
        handle,
    )
}

unsafe fn submit_impl(
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
    let replay = match unsafe { backend_mut_for_mutation(handle) } {
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

unsafe fn submit_compact_impl(
    handle: ReplayHandleV1,
    request: CompactRequestV1,
    request_id: *mut RequestIdV1,
) -> StatusV1 {
    if request_id.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    let request = match unsafe { compact_request(request) } {
        Ok(request) => request,
        Err(status) => return status,
    };
    let replay = match unsafe { backend_mut_for_mutation(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    match replay.engine.submit_compact(request) {
        Ok(uuid) => {
            unsafe { *request_id = *uuid.as_bytes() };
            StatusV1::OK
        }
        Err(error) => {
            replay.last_error = error.to_string();
            StatusV1::REJECTED
        }
    }
}

unsafe fn submit_batch_impl(
    handle: ReplayHandleV1,
    requests: DirectRequestSliceV1,
    request_ids: RequestIdMutSliceV1,
) -> StatusV1 {
    if !ffi_slice_len_is_valid::<DirectRequestV1>(requests.len)
        || request_ids.len != requests.len
        || !ffi_slice_len_is_valid::<RequestIdV1>(request_ids.len)
        || (requests.data.is_null() && requests.len != 0)
        || (request_ids.data.is_null() && request_ids.len != 0)
        || (!requests.data.is_null() && !requests.data.is_aligned())
        || (!request_ids.data.is_null() && !request_ids.data.is_aligned())
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    let output = if request_ids.len == 0 {
        &mut []
    } else {
        // Safety: the output slice passed all raw-slice size, null, and
        // alignment checks and is writable for this call.
        unsafe { std::slice::from_raw_parts_mut(request_ids.data, request_ids.len as usize) }
    };
    output.fill([0; 16]);
    let requests = if requests.len == 0 {
        &[]
    } else {
        // Safety: non-empty input batch is valid for this FFI call.
        unsafe { std::slice::from_raw_parts(requests.data, requests.len as usize) }
    };
    let mut converted = match requests
        .iter()
        .map(|request| {
            // Safety: each record's borrowed fields are valid for this call.
            unsafe { direct_request(*request) }
        })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(requests) => requests,
        Err(status) => return status,
    };
    for request in &mut converted {
        request.uuid.get_or_insert_with(Uuid::new_v4);
    }
    let replay = match unsafe { backend_mut_for_mutation(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    match replay.engine.submit_batch(converted) {
        Ok(uuids) => {
            if uuids.len() != output.len() {
                replay.last_error = format!(
                    "atomic batch returned {} request IDs for {} requests",
                    uuids.len(),
                    output.len()
                );
                return StatusV1::INTERNAL;
            }
            for (uuid, output) in uuids.into_iter().zip(output) {
                *output = *uuid.as_bytes();
            }
            StatusV1::OK
        }
        Err(error) => {
            let poisoned = error.is_poisoned();
            replay.last_error = error.to_string();
            if poisoned {
                StatusV1::INTERNAL
            } else {
                StatusV1::REJECTED
            }
        }
    }
}

unsafe extern "C" fn submit_batch(
    handle: ReplayHandleV1,
    requests: DirectRequestSliceV1,
    request_ids: RequestIdMutSliceV1,
) -> StatusV1 {
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        submit_batch_impl(handle, requests, request_ids)
    })) {
        Ok(status) => status,
        Err(_) => batch_panic_status(handle),
    }
}

unsafe fn cancel_impl(
    handle: ReplayHandleV1,
    request_id: *const RequestIdV1,
    event: *mut EngineEventV1,
    canceled: *mut u8,
) -> StatusV1 {
    if request_id.is_null() || event.is_null() || canceled.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: validated non-null input pointer.
    let request_id = Uuid::from_bytes(unsafe { *request_id });
    let replay = match unsafe { backend_mut_for_mutation(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    match replay.engine.cancel(request_id) {
        Ok(Some(terminal)) => {
            let terminal_status = match terminal.terminal_status {
                Some(ReplayTerminalStatus::Completed) => 1,
                Some(ReplayTerminalStatus::Rejected) => 2,
                Some(ReplayTerminalStatus::Canceled) => 3,
                Some(ReplayTerminalStatus::Failed) => 4,
                None => 0,
            };
            // Safety: validated output pointers.
            unsafe {
                *event = EngineEventV1 {
                    request_id: *terminal.uuid.as_bytes(),
                    flags: 1 << 1,
                    token_id: terminal.token_id.unwrap_or_default(),
                    terminal_status,
                    reserved: 0,
                };
                *canceled = 1;
            }
            StatusV1::OK
        }
        Ok(None) => {
            // Safety: validated output pointer.
            unsafe { *canceled = 0 };
            StatusV1::OK
        }
        Err(error) => {
            replay.last_error = error.to_string();
            StatusV1::REJECTED
        }
    }
}

unsafe fn cancel_batch_impl(
    handle: ReplayHandleV1,
    request_ids: RequestIdSliceV1,
    events: *mut EngineEventSliceV1,
) -> StatusV1 {
    if events.is_null()
        || !ffi_slice_len_is_valid::<RequestIdV1>(request_ids.len)
        || (request_ids.data.is_null() && request_ids.len != 0)
        || (!request_ids.data.is_null() && !request_ids.data.is_aligned())
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    let request_ids = if request_ids.len == 0 {
        &[]
    } else {
        // Safety: non-empty input batch is valid for this FFI call.
        unsafe { std::slice::from_raw_parts(request_ids.data, request_ids.len as usize) }
    };
    let replay = match unsafe { backend_mut_for_mutation(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    let mut terminals = Vec::new();
    for request_id in request_ids {
        match replay.engine.cancel(Uuid::from_bytes(*request_id)) {
            Ok(Some(terminal)) => {
                let terminal_status = match terminal.terminal_status {
                    Some(ReplayTerminalStatus::Completed) => 1,
                    Some(ReplayTerminalStatus::Rejected) => 2,
                    Some(ReplayTerminalStatus::Canceled) => 3,
                    Some(ReplayTerminalStatus::Failed) => 4,
                    None => 0,
                };
                terminals.push(EngineEventV1 {
                    request_id: *terminal.uuid.as_bytes(),
                    flags: 1 << 1,
                    token_id: terminal.token_id.unwrap_or_default(),
                    terminal_status,
                    reserved: 0,
                });
            }
            Ok(None) => {}
            Err(error) => {
                replay.last_error = error.to_string();
                return StatusV1::REJECTED;
            }
        }
    }
    let terminals = terminals.into_boxed_slice();
    let len = terminals.len() as u64;
    let data = Box::into_raw(terminals).cast::<EngineEventV1>();
    // Safety: validated non-null output pointer.
    unsafe { *events = EngineEventSliceV1 { data, len } };
    StatusV1::OK
}

unsafe fn step_impl(
    handle: ReplayHandleV1,
    request: StepRequestV1,
    result: *mut StepResultV1,
) -> StatusV1 {
    if result.is_null()
        || request.struct_size as usize != std::mem::size_of::<StepRequestV1>()
        || request.flags != 0
        || !request.until_ms.is_finite()
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: the handle is only dereferenced after null validation.
    let replay = match unsafe { backend_mut_for_mutation(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    let outcome = match replay.engine.step_until(request.until_ms) {
        Ok(outcome) => outcome,
        Err(error) => {
            replay.last_error = error.to_string();
            return StatusV1::REJECTED;
        }
    };
    let mut request_facts = Vec::with_capacity(outcome.events.len().saturating_mul(2));
    let mut seen = HashSet::new();
    for event in &outcome.events {
        if !seen.insert(event.uuid) {
            continue;
        }
        let mut flags = 0;
        let mut reused_input_tokens = 0;
        let mut admission_ms = 0.0;
        if let Some((at_ms, reused)) = replay.engine.request_admission(event.uuid) {
            flags |= REQUEST_FACT_FLAG_ADMISSION;
            admission_ms = at_ms;
            reused_input_tokens = reused as u64;
        }
        let mut ttft_ms = 0.0;
        let mut mean_itl_ms = 0.0;
        if let Some((ttft, mean_itl)) = replay.engine.request_latencies(event.uuid) {
            flags |= REQUEST_FACT_FLAG_LATENCIES;
            ttft_ms = ttft;
            mean_itl_ms = mean_itl;
        }
        let mut output_length = 0;
        if let Some(length) = replay.engine.actual_output_length(event.uuid) {
            flags |= REQUEST_FACT_FLAG_OUTPUT_LENGTH;
            output_length = length as u64;
        }
        if flags != 0 {
            request_facts.push(RequestFactV1 {
                request_id: *event.uuid.as_bytes(),
                flags,
                reserved: 0,
                reused_input_tokens,
                output_length,
                admission_ms,
                ttft_ms,
                mean_itl_ms,
            });
        }
        // FIRST and MAX are distinct ABI groups even when both values are zero.
        // The terminal outcome is queried before take_report drains the collector.
        if let Some(reused) = replay.engine.request_reused_input_tokens(event.uuid) {
            request_facts.push(RequestFactV1 {
                request_id: *event.uuid.as_bytes(),
                flags: REQUEST_FACT_FLAG_MAX_REUSE,
                reserved: 0,
                reused_input_tokens: reused as u64,
                output_length: 0,
                admission_ms: 0.0,
                ttft_ms: 0.0,
                mean_itl_ms: 0.0,
            });
        }
    }
    let request_facts = request_facts.into_boxed_slice();
    let request_facts_len = request_facts.len() as u64;
    let request_facts_data = Box::into_raw(request_facts).cast::<RequestFactV1>();
    let events = outcome
        .events
        .into_iter()
        .map(|event| {
            let mut flags = 0;
            if event.emitted_token {
                flags |= 1;
            }
            if event.terminal_status.is_some() {
                flags |= 1 << 1;
            }
            let terminal_status = match event.terminal_status {
                Some(ReplayTerminalStatus::Completed) => 1,
                Some(ReplayTerminalStatus::Rejected) => 2,
                Some(ReplayTerminalStatus::Canceled) => 3,
                Some(ReplayTerminalStatus::Failed) => 4,
                None => 0,
            };
            EngineEventV1 {
                request_id: *event.uuid.as_bytes(),
                flags,
                token_id: event.token_id.unwrap_or_default(),
                terminal_status,
                reserved: 0,
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let events_len = events.len() as u64;
    let events_data = Box::into_raw(events).cast::<EngineEventV1>();
    let next_event_ms = replay.engine.next_event_ms().unwrap_or(f64::NAN);
    // Safety: validated non-null output pointer. Event ownership transfers to
    // the host, which must call `release_events` exactly once.
    unsafe {
        *result = StepResultV1 {
            struct_size: std::mem::size_of::<StepResultV1>() as u32,
            flags: 0,
            end_ms: outcome.end_ms,
            next_event_ms,
            in_flight: replay.engine.in_flight() as u64,
            is_idle: u8::from(replay.engine.is_idle()),
            reserved: [0; 7],
            events: EngineEventSliceV1 {
                data: events_data,
                len: events_len,
            },
            request_facts: RequestFactSliceV1 {
                data: request_facts_data,
                len: request_facts_len,
            },
        };
    }
    StatusV1::OK
}

unsafe fn step_timestamped_impl(
    handle: ReplayHandleV1,
    request: StepRequestV1,
    result: *mut TimestampedStepResultV1,
) -> StatusV1 {
    if result.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    let mut legacy = StepResultV1::EMPTY;
    // Safety: this wrapper passes its own valid output storage to the stable
    // V1 step implementation, then transfers its request-fact allocation to
    // the timestamped result below.
    let status = unsafe { step(handle, request, &raw mut legacy) };
    if !status.is_ok() {
        return status;
    }
    let events = if legacy.events.len == 0 {
        Vec::new()
    } else {
        if legacy.events.data.is_null() || !legacy.events.data.is_aligned() {
            unsafe { release_request_facts(legacy.request_facts) };
            return StatusV1::INTERNAL;
        }
        // Safety: `step` returned this exact nonempty plugin-owned slice.
        unsafe { std::slice::from_raw_parts(legacy.events.data, legacy.events.len as usize) }
            .iter()
            .copied()
            .map(|event| TimestampedEngineEventV1 {
                event,
                // A steppable core invocation drains one current timestamp and
                // returns before another timestamp can be consumed; hence every
                // event emitted by this outcome occurred at its end time.
                at_ms: legacy.end_ms,
            })
            .collect::<Vec<_>>()
    }
    .into_boxed_slice();
    let event_len = events.len() as u64;
    let event_data = Box::into_raw(events).cast::<TimestampedEngineEventV1>();
    // Release the legacy event allocation after copying it. The request-fact
    // allocation is transferred verbatim to the timestamped result.
    unsafe { release_events(legacy.events) };
    unsafe {
        *result = TimestampedStepResultV1 {
            struct_size: std::mem::size_of::<TimestampedStepResultV1>() as u32,
            flags: 0,
            end_ms: legacy.end_ms,
            next_event_ms: legacy.next_event_ms,
            in_flight: legacy.in_flight,
            is_idle: legacy.is_idle,
            reserved: [0; 7],
            events: TimestampedEngineEventSliceV1 {
                data: event_data,
                len: event_len,
            },
            request_facts: legacy.request_facts,
        };
    }
    StatusV1::OK
}

unsafe extern "C" fn step_timestamped(
    handle: ReplayHandleV1,
    request: StepRequestV1,
    result: *mut TimestampedStepResultV1,
) -> StatusV1 {
    if !result.is_null() {
        // Safety: caller supplied writable output storage for this FFI call.
        unsafe { *result = TimestampedStepResultV1::EMPTY };
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        step_timestamped_impl(handle, request, result)
    })) {
        Ok(status) => status,
        Err(_) => batch_panic_status(handle),
    }
}

unsafe fn take_report_impl(
    handle: ReplayHandleV1,
    wall_ms: f64,
    report: *mut ByteSliceV1,
) -> StatusV1 {
    if report.is_null() || !wall_ms.is_finite() {
        return StatusV1::INVALID_ARGUMENT;
    }
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    if !matches!(replay.finalization, Finalization::Open) {
        replay.last_error = "replay report was already finalized through another API".to_owned();
        return StatusV1::REJECTED;
    }
    replay.finalization = Finalization::Finalizing;
    let report_value = match replay.engine.take_report(wall_ms) {
        Ok(report_value) => report_value,
        Err(error) => {
            replay.finalization = Finalization::Open;
            replay.last_error = error.to_string();
            return StatusV1::REJECTED;
        }
    };
    let encoded = match serde_json::to_string(&report_value) {
        Ok(encoded) => encoded,
        Err(error) => {
            replay.finalization = Finalization::Poisoned;
            replay.last_error = error.to_string();
            return StatusV1::INTERNAL;
        }
    };
    replay.finalization = Finalization::LegacyFinalized;
    // Safety: validated non-null output pointer.
    unsafe { *report = allocated_bytes(encoded) };
    StatusV1::OK
}

unsafe extern "C" fn take_report(
    handle: ReplayHandleV1,
    wall_ms: f64,
    report: *mut ByteSliceV1,
) -> StatusV1 {
    if !report.is_null() {
        // Safety: the non-null host-owned output is valid for this call.
        unsafe { *report = ByteSliceV1::EMPTY };
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        take_report_impl(handle, wall_ms, report)
    })) {
        Ok(status) => status,
        Err(_) => finalization_panic_status(handle),
    }
}

fn backend_distribution(
    source: &aisimulate_core::replay::TraceDistributionStats,
) -> BackendDistributionStatsV1 {
    BackendDistributionStatsV1 {
        mean_ms: source.mean_ms,
        min_ms: source.min_ms,
        max_ms: source.max_ms,
        median_ms: source.median_ms,
        p75_ms: source.p75_ms,
        p90_ms: source.p90_ms,
        p95_ms: source.p95_ms,
        p99_ms: source.p99_ms,
        std_ms: source.std_ms,
    }
}

fn backend_final_summary(
    report: &aisimulate_core::replay::ReplayReport,
) -> Result<BackendFinalSummaryV1, String> {
    let count = |value: usize, name: &str| {
        u64::try_from(value).map_err(|_| format!("{name} does not fit the Steppable ABI"))
    };
    let goodput = match &report.goodput {
        Some(goodput) => BackendGoodputStatsV1 {
            is_present: 1,
            reserved: [0; 7],
            completed_requests: count(goodput.completed_requests, "goodput completed requests")?,
            request_throughput_rps: goodput.request_throughput_rps,
            output_throughput_tok_s: goodput.output_throughput_tok_s,
        },
        None => BackendGoodputStatsV1 {
            is_present: 0,
            reserved: [0; 7],
            completed_requests: 0,
            request_throughput_rps: 0.0,
            output_throughput_tok_s: 0.0,
        },
    };
    Ok(BackendFinalSummaryV1 {
        struct_size: std::mem::size_of::<BackendFinalSummaryV1>() as u32,
        flags: 0,
        request_counts: BackendRequestCountsV1 {
            num_requests: count(report.request_counts.num_requests, "request count")?,
            completed_requests: count(report.request_counts.completed_requests, "completed count")?,
            total_input_tokens: count(
                report.request_counts.total_input_tokens,
                "input token count",
            )?,
            total_output_tokens: count(
                report.request_counts.total_output_tokens,
                "output token count",
            )?,
        },
        throughput: BackendThroughputStatsV1 {
            duration_ms: report.throughput.duration_ms,
            wall_time_ms: report.throughput.wall_time_ms,
            request_throughput_rps: report.throughput.request_throughput_rps,
            input_throughput_tok_s: report.throughput.input_throughput_tok_s,
            output_throughput_tok_s: report.throughput.output_throughput_tok_s,
            total_throughput_tok_s: report.throughput.total_throughput_tok_s,
            prefill_worker_seconds: report.throughput.prefill_worker_seconds,
            decode_worker_seconds: report.throughput.decode_worker_seconds,
            prefill_gpus_per_worker: count(
                report.throughput.prefill_gpus_per_worker,
                "prefill GPUs per worker",
            )?,
            decode_gpus_per_worker: count(
                report.throughput.decode_gpus_per_worker,
                "decode GPUs per worker",
            )?,
            gpu_hours: report.throughput.gpu_hours,
        },
        prefix_cache_reused_ratio: report.prefix_cache_reused_ratio,
        first_admission_prefix_cache_reused_ratio: report.first_admission_prefix_cache_reused_ratio,
        latency: BackendLatencyStatsV1 {
            ttft: backend_distribution(&report.latency.ttft),
            ttst: backend_distribution(&report.latency.ttst),
            tpot: backend_distribution(&report.latency.tpot),
            itl: backend_distribution(&report.latency.itl.distribution),
            itl_max_ms: report.latency.itl.max_ms,
            e2e: backend_distribution(&report.latency.e2e),
            output_token_throughput_per_user: backend_distribution(
                &report.latency.output_token_throughput_per_user,
            ),
        },
        goodput,
    })
}

fn final_artifacts(
    report: &aisimulate_core::replay::ReplayReport,
) -> Result<FinalArtifacts, String> {
    let report_json = canonical_report_json(report)?;
    let mut per_request_jsonl = Vec::new();
    for record in &report.per_request {
        serde_json::to_writer(&mut per_request_jsonl, record).map_err(|error| error.to_string())?;
        per_request_jsonl.push(b'\n');
    }
    Ok(FinalArtifacts {
        report_json: FinalArtifact::new(report_json),
        per_request_jsonl: FinalArtifact::new(per_request_jsonl),
    })
}

impl FinalArtifact {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            sha256: Sha256::digest(&bytes).into(),
            bytes,
        }
    }
}

fn canonical_report_json(
    report: &aisimulate_core::replay::ReplayReport,
) -> Result<Vec<u8>, String> {
    let value = serde_json::to_value(report).map_err(|error| error.to_string())?;
    let object = value
        .as_object()
        .ok_or_else(|| "AISimulate report did not serialize as a JSON object".to_owned())?;
    let sorted = object.iter().collect::<BTreeMap<_, _>>();
    let mut payload = serde_json::to_string_pretty(&sorted).map_err(|error| error.to_string())?;
    payload = python_json_number_exponents(&payload)?;
    payload.push('\n');
    Ok(payload.into_bytes())
}

fn python_json_number_exponents(payload: &str) -> Result<String, String> {
    let bytes = payload.as_bytes();
    let mut normalized = Vec::with_capacity(payload.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            normalized.push(byte);
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if byte == b'"' {
            in_string = true;
            normalized.push(byte);
            index += 1;
            continue;
        }
        if byte != b'e' && byte != b'E' {
            normalized.push(byte);
            index += 1;
            continue;
        }
        let mut exponent = index + 1;
        let sign = if exponent < bytes.len() && (bytes[exponent] == b'+' || bytes[exponent] == b'-')
        {
            let sign = bytes[exponent];
            exponent += 1;
            sign
        } else {
            b'+'
        };
        let digits_start = exponent;
        while exponent < bytes.len() && bytes[exponent].is_ascii_digit() {
            exponent += 1;
        }
        if exponent == digits_start {
            normalized.push(byte);
            index += 1;
            continue;
        }
        normalized.push(b'e');
        normalized.push(sign);
        if exponent - digits_start == 1 {
            normalized.push(b'0');
        }
        normalized.extend_from_slice(&bytes[digits_start..exponent]);
        index = exponent;
    }
    String::from_utf8(normalized).map_err(|error| error.to_string())
}

fn artifact(artifacts: &FinalArtifacts, kind: u32) -> Option<&FinalArtifact> {
    match kind {
        FINAL_ARTIFACT_KIND_REPORT_JSON_V1 => Some(&artifacts.report_json),
        FINAL_ARTIFACT_KIND_PER_REQUEST_JSONL_V1 => Some(&artifacts.per_request_jsonl),
        _ => None,
    }
}

unsafe fn take_final_summary_impl(
    handle: ReplayHandleV1,
    wall_ms: f64,
    summary: *mut BackendFinalSummaryV1,
) -> StatusV1 {
    if summary.is_null() || !wall_ms.is_finite() {
        return StatusV1::INVALID_ARGUMENT;
    }
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    if !matches!(replay.finalization, Finalization::Open) {
        replay.last_error = "replay report was already finalized through another API".to_owned();
        return StatusV1::REJECTED;
    }
    replay.finalization = Finalization::Finalizing;
    let report = match replay.engine.take_report(wall_ms) {
        Ok(report) => report,
        Err(error) => {
            replay.finalization = Finalization::Open;
            replay.last_error = error.to_string();
            return StatusV1::REJECTED;
        }
    };
    let summary_value = match backend_final_summary(&report) {
        Ok(summary_value) => summary_value,
        Err(error) => {
            replay.finalization = Finalization::Poisoned;
            replay.last_error = error;
            return StatusV1::INTERNAL;
        }
    };
    let artifacts = match final_artifacts(&report) {
        Ok(artifacts) => artifacts,
        Err(error) => {
            replay.finalization = Finalization::Poisoned;
            replay.last_error = error;
            return StatusV1::INTERNAL;
        }
    };
    replay.finalization = Finalization::Complete(artifacts);
    // Safety: pointer was checked non-null and points to one host-owned output record.
    unsafe { *summary = summary_value };
    StatusV1::OK
}

unsafe extern "C" fn take_final_summary(
    handle: ReplayHandleV1,
    wall_ms: f64,
    summary: *mut BackendFinalSummaryV1,
) -> StatusV1 {
    if !summary.is_null() {
        // Safety: the non-null host-owned output is valid for this call.
        unsafe { *summary = std::mem::zeroed() };
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        take_final_summary_impl(handle, wall_ms, summary)
    })) {
        Ok(status) => status,
        Err(_) => finalization_panic_status(handle),
    }
}

unsafe fn get_final_artifact_metadata_impl(
    handle: ReplayHandleV1,
    kind: u32,
    metadata: *mut FinalArtifactMetadataV1,
) -> StatusV1 {
    if metadata.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    let Finalization::Complete(artifacts) = &replay.finalization else {
        replay.last_error = "final artifacts require final summary first".to_owned();
        return StatusV1::REJECTED;
    };
    let Some(artifact) = artifact(artifacts, kind) else {
        return StatusV1::UNSUPPORTED;
    };
    let Ok(byte_len) = u64::try_from(artifact.bytes.len()) else {
        replay.last_error = "final artifact exceeds Steppable ABI length".to_owned();
        return StatusV1::INTERNAL;
    };
    // Safety: pointer was checked non-null and points to one host-owned output record.
    unsafe {
        *metadata = FinalArtifactMetadataV1 {
            struct_size: std::mem::size_of::<FinalArtifactMetadataV1>() as u32,
            kind,
            byte_len,
            sha256: artifact.sha256,
        };
    }
    StatusV1::OK
}

unsafe extern "C" fn get_final_artifact_metadata(
    handle: ReplayHandleV1,
    kind: u32,
    metadata: *mut FinalArtifactMetadataV1,
) -> StatusV1 {
    if !metadata.is_null() {
        // Safety: the non-null host-owned output is valid for this call.
        unsafe { *metadata = std::mem::zeroed() };
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        get_final_artifact_metadata_impl(handle, kind, metadata)
    })) {
        Ok(status) => status,
        Err(_) => finalization_panic_status(handle),
    }
}

unsafe fn read_final_artifact_chunk_impl(
    handle: ReplayHandleV1,
    kind: u32,
    offset: u64,
    destination: ByteMutSliceV1,
    result: *mut FinalArtifactChunkV1,
) -> StatusV1 {
    if result.is_null()
        || destination.data.is_null()
        || destination.len == 0
        || !ffi_slice_len_is_valid::<u8>(destination.len)
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    let Finalization::Complete(artifacts) = &replay.finalization else {
        replay.last_error = "final artifacts require final summary first".to_owned();
        return StatusV1::REJECTED;
    };
    let Some(artifact) = artifact(artifacts, kind) else {
        return StatusV1::UNSUPPORTED;
    };
    let Ok(offset) = usize::try_from(offset) else {
        return StatusV1::INVALID_ARGUMENT;
    };
    if offset > artifact.bytes.len() {
        return StatusV1::INVALID_ARGUMENT;
    }
    let len = destination.len.min((artifact.bytes.len() - offset) as u64) as usize;
    // Safety: destination was checked non-null and host promises its declared writable range.
    unsafe {
        std::ptr::copy_nonoverlapping(artifact.bytes[offset..].as_ptr(), destination.data, len)
    };
    // Safety: result was checked non-null and points to one host-owned output record.
    unsafe {
        *result = FinalArtifactChunkV1 {
            struct_size: std::mem::size_of::<FinalArtifactChunkV1>() as u32,
            flags: 0,
            written_len: len as u64,
            is_final: u8::from(offset + len == artifact.bytes.len()),
            reserved: [0; 7],
        };
    }
    StatusV1::OK
}

unsafe extern "C" fn read_final_artifact_chunk(
    handle: ReplayHandleV1,
    kind: u32,
    offset: u64,
    destination: ByteMutSliceV1,
    result: *mut FinalArtifactChunkV1,
) -> StatusV1 {
    if !result.is_null() {
        // Safety: the non-null host-owned output is valid for this call.
        unsafe { *result = std::mem::zeroed() };
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        read_final_artifact_chunk_impl(handle, kind, offset, destination, result)
    })) {
        Ok(status) => status,
        Err(_) => finalization_panic_status(handle),
    }
}

unsafe extern "C" fn release_bytes(bytes: ByteSliceV1) {
    if bytes.data.is_null() {
        return;
    }
    if !ffi_slice_len_is_valid::<u8>(bytes.len) {
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
unsafe extern "C" fn release_events(events: EngineEventSliceV1) {
    if events.data.is_null() {
        return;
    }
    if !ffi_slice_len_is_valid::<EngineEventV1>(events.len) || !events.data.is_aligned() {
        return;
    }
    // Safety: `step` allocates exact-length boxed slices and transfers one
    // release obligation to the host.
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            events.data.cast_mut(),
            events.len as usize,
        )));
    }
}

unsafe extern "C" fn release_timestamped_events(events: TimestampedEngineEventSliceV1) {
    if events.data.is_null()
        || !ffi_slice_len_is_valid::<TimestampedEngineEventV1>(events.len)
        || !events.data.is_aligned()
    {
        return;
    }
    // Safety: `step_timestamped` allocates exact-length boxed slices and
    // transfers one release obligation to the host.
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            events.data.cast_mut(),
            events.len as usize,
        )));
    }
}
unsafe extern "C" fn release_request_facts(facts: RequestFactSliceV1) {
    if facts.data.is_null()
        || !ffi_slice_len_is_valid::<RequestFactV1>(facts.len)
        || !facts.data.is_aligned()
    {
        return;
    }
    // Safety: `step` allocates exact-length boxed slices and transfers one
    // release obligation to the host.
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            facts.data.cast_mut(),
            facts.len as usize,
        )));
    }
}

unsafe fn state_impl(handle: ReplayHandleV1, state: *mut ReplayStateV1) -> StatusV1 {
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

unsafe fn advance_now_ms_impl(handle: ReplayHandleV1, now_ms: f64) -> StatusV1 {
    if !now_ms.is_finite() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: the handle is only dereferenced after null validation in
    // `backend_mut`.
    let replay = match unsafe { backend_mut_for_mutation(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    replay.engine.advance_now_ms(now_ms);
    StatusV1::OK
}

unsafe fn set_capture_per_request_impl(handle: ReplayHandleV1, capture: u8) -> StatusV1 {
    if capture > 1 {
        return StatusV1::INVALID_ARGUMENT;
    }
    let replay = match unsafe { backend_mut_for_mutation(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    replay.engine.set_capture_per_request(capture != 0);
    StatusV1::OK
}

unsafe fn set_sla_thresholds_impl(handle: ReplayHandleV1, thresholds: SlaThresholdsV1) -> StatusV1 {
    let selected = |flag, value| (thresholds.flags & flag != 0).then_some(value);
    let sla = SlaThresholds {
        ttft_ms: selected(SLA_FLAG_TTFT, thresholds.ttft_ms),
        itl_ms: selected(SLA_FLAG_ITL, thresholds.itl_ms),
        e2e_ms: selected(SLA_FLAG_E2E, thresholds.e2e_ms),
    };
    if ((sla.ttft_ms.is_some() || sla.itl_ms.is_some()) && sla.e2e_ms.is_some())
        || [sla.ttft_ms, sla.itl_ms, sla.e2e_ms]
            .into_iter()
            .flatten()
            .any(|value| !value.is_finite() || value <= 0.0)
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    let replay = match unsafe { backend_mut_for_mutation(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    replay.engine.set_sla_thresholds(sla);
    StatusV1::OK
}

unsafe fn last_error_impl(handle: ReplayHandleV1, error: *mut ByteSliceV1) -> StatusV1 {
    if error.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    let replay = match unsafe { backend_mut_even_if_poisoned(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    // Safety: validated non-null output pointer.
    unsafe {
        *error = if replay.last_error.is_empty() {
            ByteSliceV1::EMPTY
        } else {
            allocated_bytes(replay.last_error.clone())
        };
    }
    StatusV1::OK
}

unsafe fn destroy_impl(handle: ReplayHandleV1) {
    if handle.0.is_null() {
        return;
    }
    // Safety: caller transfers the unique handle returned by `create`.
    unsafe { drop(Box::from_raw(handle.0.cast::<BackendReplay>())) };
}

unsafe extern "C" fn destroy(handle: ReplayHandleV1) {
    // The callback contract forbids panicking or re-entry. This guard covers
    // internal destruction so an unexpected Rust panic cannot cross the ABI.
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe { destroy_impl(handle) }));
}

// Keep Rust unwinding inside the provider even when native admission or a
// getter fails unexpectedly. An unknown mutation poisons the replay; no retry
// is permitted after an internal fault.
macro_rules! guarded_status {
    ($name:ident => $implementation:ident($handle:ident: ReplayHandleV1 $(, $arg:ident: $ty:ty)* $(,)?)) => {
        unsafe extern "C" fn $name($handle: ReplayHandleV1, $($arg: $ty),*) -> StatusV1 {
            catch_status(|| unsafe { $implementation($handle, $($arg),*) }, $handle)
        }
    };
}

guarded_status!(submit => submit_impl(handle: ReplayHandleV1,
    request: DirectRequestV1, request_id: *mut RequestIdV1));
guarded_status!(submit_compact => submit_compact_impl(handle: ReplayHandleV1,
    request: CompactRequestV1, request_id: *mut RequestIdV1));
guarded_status!(cancel => cancel_impl(handle: ReplayHandleV1,
    request_id: *const RequestIdV1, event: *mut EngineEventV1, canceled: *mut u8));
guarded_status!(cancel_batch => cancel_batch_impl(handle: ReplayHandleV1,
    request_ids: RequestIdSliceV1, events: *mut EngineEventSliceV1));
guarded_status!(step => step_impl(handle: ReplayHandleV1,
    request: StepRequestV1, result: *mut StepResultV1));
guarded_status!(state => state_impl(handle: ReplayHandleV1, state: *mut ReplayStateV1));
guarded_status!(advance_now_ms => advance_now_ms_impl(handle: ReplayHandleV1, now_ms: f64));
guarded_status!(set_capture_per_request => set_capture_per_request_impl(handle: ReplayHandleV1,
    capture: u8));
guarded_status!(set_sla_thresholds => set_sla_thresholds_impl(handle: ReplayHandleV1,
    thresholds: SlaThresholdsV1));
guarded_status!(last_error => last_error_impl(handle: ReplayHandleV1, error: *mut ByteSliceV1));

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
    submit_compact: Some(submit_compact),
    create_with_hash_buffer_leases: Some(create_with_hash_buffer_leases),
    register_hash_buffer: Some(register_hash_buffer),
    submit_compact_hash_buffer_range: Some(submit_compact_hash_buffer_range),
    take_final_summary: Some(take_final_summary),
    get_final_artifact_metadata: Some(get_final_artifact_metadata),
    read_final_artifact_chunk: Some(read_final_artifact_chunk),
    step_timestamped: Some(step_timestamped),
    release_timestamped_events: Some(release_timestamped_events),
};

static DESCRIPTOR: PluginDescriptorV1 = PluginDescriptorV1 {
    abi_major: 1,
    abi_minor: 0,
    struct_size: std::mem::size_of::<PluginDescriptorV1>() as u32,
    flags: 0,
    capabilities: (if cfg!(feature = "compact") {
        CAPABILITY_COMPACT_REQUEST_V1
    } else {
        0
    }) | (if cfg!(feature = "leases") {
        CAPABILITY_COMPACT_BUFFER_LEASES_V1
    } else {
        0
    }) | CAPABILITY_FINAL_REPORT_BUNDLE_V1
        | CAPABILITY_REQUEST_REUSE_FACTS_V1
        | (if cfg!(feature = "without-timestamped-events") {
            0
        } else {
            CAPABILITY_TIMESTAMPED_EVENTS_V1
        }),
    provider_id: PROVIDER_ID.as_ptr().cast::<c_char>(),
    vtable: &VTABLE,
};

/// Returns the static V1 plugin descriptor for dynamic loading.
#[unsafe(no_mangle)]
pub extern "C" fn aiperf_steppable_plugin_v1() -> *const PluginDescriptorV1 {
    &DESCRIPTOR
}

/// Exported entry-point symbol, NUL terminated, as a host resolves it.
pub const ENTRY_SYMBOL: &[u8] = b"aiperf_steppable_plugin_v1\0";

/// Provider ID this descriptor reports.
pub const PROVIDER: &str = "aisimulate";

// The entry point must keep the canonical ABI's entry signature exactly.
const _: aiperf_steppable_abi::PluginEntryV1 = aiperf_steppable_plugin_v1;

#[cfg(test)]
mod tests;
