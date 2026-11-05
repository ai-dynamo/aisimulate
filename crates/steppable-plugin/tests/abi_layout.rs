// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The C layout this plugin was reviewed against, frozen.
//!
//! The types come from AIPerf's `aiperf-steppable-abi`, so the plugin and the
//! host can never compile against different definitions of one revision. What
//! can drift is the revision: moving the dependency pin to an ABI whose records
//! changed size, alignment or vtable slot order would still compile here while
//! changing what crosses the boundary. These tables make that move fail until
//! someone reviews the new layout and updates them deliberately.
//!
//! Values are for 64-bit targets, which are the only ones AIPerf loads
//! plugins on. AIPerf freezes a subset of the same facts in its own
//! `crates/engine/steppable-abi/tests/layout.rs`.

#![cfg(target_pointer_width = "64")]

use std::mem::{align_of, offset_of, size_of};

use aiperf_steppable_abi::*;

macro_rules! layout {
    ($($ty:ident),* $(,)?) => {
        [$((stringify!($ty), size_of::<$ty>(), align_of::<$ty>())),*]
    };
}

macro_rules! slots {
    ($($field:ident),* $(,)?) => {
        [$((stringify!($field), offset_of!(PluginVTableV1, $field))),*]
    };
}

/// `(type, size, align)` at aiperf-rs `b921cb82deb0941d2f00d3f98771d40377ff57f2`.
const RECORDS: &[(&str, usize, usize)] = &[
    ("PluginDescriptorV1", 40, 8),
    ("PluginVTableV1", 208, 8),
    ("PluginVTableV1Prefix", 136, 8),
    ("StatusV1", 4, 4),
    ("ByteSliceV1", 16, 8),
    ("ByteMutSliceV1", 16, 8),
    ("U32SliceV1", 16, 8),
    ("CreateRequestV1", 24, 8),
    ("DirectRequestV1", 184, 8),
    ("DirectRequestSliceV1", 16, 8),
    ("CompactRequestV1", 224, 8),
    ("ReplayContextV1", 72, 8),
    ("HashBufferIdV1", 8, 8),
    ("HashBufferRangeV1", 24, 8),
    ("HashBufferLeaseCallbacksV1", 24, 8),
    ("RequestIdSliceV1", 16, 8),
    ("RequestIdMutSliceV1", 16, 8),
    ("EngineEventV1", 32, 4),
    ("EngineEventSliceV1", 16, 8),
    ("TimestampedEngineEventV1", 40, 8),
    ("TimestampedEngineEventSliceV1", 16, 8),
    ("RequestFactV1", 64, 8),
    ("RequestFactSliceV1", 16, 8),
    ("ReplayHandleV1", 8, 8),
    ("ReplayStateV1", 32, 8),
    ("SlaThresholdsV1", 32, 8),
    ("StepRequestV1", 16, 8),
    ("StepResultV1", 72, 8),
    ("TimestampedStepResultV1", 72, 8),
    ("BackendFinalSummaryV1", 616, 8),
    ("FinalArtifactMetadataV1", 48, 8),
    ("FinalArtifactChunkV1", 24, 8),
];

/// `PluginVTableV1` slot offsets at the same revision, in table order.
const VTABLE_SLOTS: &[(&str, usize)] = &[
    ("struct_size", 0),
    ("flags", 4),
    ("create", 8),
    ("submit", 16),
    ("submit_batch", 24),
    ("cancel", 32),
    ("cancel_batch", 40),
    ("step", 48),
    ("take_report", 56),
    ("release_bytes", 64),
    ("release_events", 72),
    ("release_request_facts", 80),
    ("state", 88),
    ("advance_now_ms", 96),
    ("set_capture_per_request", 104),
    ("set_sla_thresholds", 112),
    ("last_error", 120),
    ("destroy", 128),
    ("submit_compact", 136),
    ("create_with_hash_buffer_leases", 144),
    ("register_hash_buffer", 152),
    ("submit_compact_hash_buffer_range", 160),
    ("take_final_summary", 168),
    ("get_final_artifact_metadata", 176),
    ("read_final_artifact_chunk", 184),
    ("step_timestamped", 192),
    ("release_timestamped_events", 200),
];

#[test]
fn exchanged_records_keep_the_reviewed_layout() {
    let actual = layout!(
        PluginDescriptorV1,
        PluginVTableV1,
        PluginVTableV1Prefix,
        StatusV1,
        ByteSliceV1,
        ByteMutSliceV1,
        U32SliceV1,
        CreateRequestV1,
        DirectRequestV1,
        DirectRequestSliceV1,
        CompactRequestV1,
        ReplayContextV1,
        HashBufferIdV1,
        HashBufferRangeV1,
        HashBufferLeaseCallbacksV1,
        RequestIdSliceV1,
        RequestIdMutSliceV1,
        EngineEventV1,
        EngineEventSliceV1,
        TimestampedEngineEventV1,
        TimestampedEngineEventSliceV1,
        RequestFactV1,
        RequestFactSliceV1,
        ReplayHandleV1,
        ReplayStateV1,
        SlaThresholdsV1,
        StepRequestV1,
        StepResultV1,
        TimestampedStepResultV1,
        BackendFinalSummaryV1,
        FinalArtifactMetadataV1,
        FinalArtifactChunkV1,
    );
    assert_eq!(actual.as_slice(), RECORDS);
}

#[test]
fn vtable_slots_keep_the_reviewed_order() {
    let actual = slots!(
        struct_size,
        flags,
        create,
        submit,
        submit_batch,
        cancel,
        cancel_batch,
        step,
        take_report,
        release_bytes,
        release_events,
        release_request_facts,
        state,
        advance_now_ms,
        set_capture_per_request,
        set_sla_thresholds,
        last_error,
        destroy,
        submit_compact,
        create_with_hash_buffer_leases,
        register_hash_buffer,
        submit_compact_hash_buffer_range,
        take_final_summary,
        get_final_artifact_metadata,
        read_final_artifact_chunk,
        step_timestamped,
        release_timestamped_events,
    );
    assert_eq!(actual.as_slice(), VTABLE_SLOTS);
    // The last slot ends the table: a new tail must extend this list.
    let (_, last) = VTABLE_SLOTS[VTABLE_SLOTS.len() - 1];
    assert_eq!(last + size_of::<usize>(), size_of::<PluginVTableV1>());
}
