// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Producer-side qualification of fields not observable under round-robin routing.

use super::*;

fn bytes(value: &[u8]) -> ByteSliceV1 {
    ByteSliceV1 {
        data: value.as_ptr(),
        len: value.len() as u64,
    }
}

fn request() -> DirectRequestV1 {
    DirectRequestV1 {
        struct_size: std::mem::size_of::<DirectRequestV1>() as u32,
        flags: 0,
        tokens: U32SliceV1::EMPTY,
        output_token_ids: U32SliceV1::EMPTY,
        max_output_tokens: 99,
        uuid: [0; 16],
        dp_rank: 0,
        preferred_dp_rank: 0,
        preferred_prefill_dp_rank: 0,
        arrival_timestamp_ms: 0.0,
        priority: 0,
        strict_priority: 0,
        policy_class: ByteSliceV1::EMPTY,
        replay_context: ReplayContextV1::EMPTY,
    }
}

#[test]
fn complete_current_request_metadata_is_owned_without_field_loss() {
    let mut raw = request();
    raw.flags = REQUEST_FLAG_UUID
        | REQUEST_FLAG_PREFERRED_DP_RANK
        | REQUEST_FLAG_PREFERRED_PREFILL_DP_RANK
        | REQUEST_FLAG_ARRIVAL_TIMESTAMP
        | REQUEST_FLAG_POLICY_CLASS
        | REQUEST_FLAG_REPLAY_CONTEXT
        | REQUEST_FLAG_OUTPUT_TOKEN_IDS;
    raw.uuid = [9; 16];
    raw.dp_rank = 2;
    raw.preferred_dp_rank = 3;
    raw.preferred_prefill_dp_rank = 4;
    raw.arrival_timestamp_ms = 12.5;
    raw.priority = -3;
    raw.strict_priority = 7;
    raw.policy_class = bytes(b"class");
    raw.replay_context = ReplayContextV1 {
        struct_size: std::mem::size_of::<ReplayContextV1>() as u32,
        flags: REPLAY_CONTEXT_FLAG_SESSION_ID
            | REPLAY_CONTEXT_FLAG_TURN_INDEX
            | REPLAY_CONTEXT_FLAG_METADATA,
        authored_id: bytes(b"authored"),
        session_id: bytes(b"session"),
        metadata: bytes(br#"{"a":[1,true]}"#),
        turn_index: 5,
        prompt_token_source: 1,
        reserved: 0,
    };
    let decoded = unsafe { direct_request(raw) }.expect("decode complete request");
    assert_eq!(decoded.uuid, Some(Uuid::from_bytes([9; 16])));
    assert_eq!(decoded.dp_rank, 2);
    assert_eq!(decoded.preferred_dp_rank, Some(3));
    assert_eq!(decoded.preferred_prefill_dp_rank, Some(4));
    assert_eq!(decoded.arrival_timestamp_ms, Some(12.5));
    assert_eq!((decoded.priority, decoded.strict_priority), (-3, 7));
    assert_eq!(decoded.policy_class.as_deref(), Some("class"));
    assert_eq!(decoded.output_token_ids, Some(Vec::new()));
    assert_eq!(decoded.effective_max_output_tokens(), 0);
    let context = decoded.replay_context.expect("retained context");
    assert_eq!(context.authored_id, "authored");
    assert_eq!(context.session_id.as_deref(), Some("session"));
    assert_eq!(context.turn_index, Some(5));
    assert_eq!(context.metadata, serde_json::json!({"a":[1,true]}));
    assert_eq!(
        context.prompt_token_source,
        ReplayPromptTokenSource::LengthOnlySynthetic
    );
}

#[test]
fn distribution_conversion_preserves_estimated_quantiles_and_exact_extrema() {
    // Native streaming stats keep exact extrema but estimate quantiles with a
    // sketch. A bin representative need not equal the extrema of equal samples.
    let source = aisimulate_core::replay::TraceDistributionStats {
        mean_ms: 100.0,
        min_ms: 100.0,
        max_ms: 100.0,
        median_ms: 100.05,
        p75_ms: 100.05,
        p90_ms: 100.05,
        p95_ms: 100.05,
        p99_ms: 100.05,
        std_ms: 0.0,
    };
    let converted = backend_distribution(&source);
    assert_eq!(converted.min_ms, source.min_ms);
    assert_eq!(converted.max_ms, source.max_ms);
    assert_eq!(converted.mean_ms, source.mean_ms);
    assert_eq!(converted.median_ms, source.median_ms);
    assert_eq!(converted.p75_ms, source.p75_ms);
    assert_eq!(converted.p90_ms, source.p90_ms);
    assert_eq!(converted.p95_ms, source.p95_ms);
    assert_eq!(converted.p99_ms, source.p99_ms);
    assert_eq!(converted.std_ms, source.std_ms);
}

#[test]
fn absent_and_present_empty_output_plans_are_distinct() {
    let mut raw = request();
    let absent = unsafe { direct_request(raw) }.expect("absent plan");
    assert_eq!(absent.output_token_ids, None);
    assert_eq!(absent.effective_max_output_tokens(), 99);
    raw.flags = REQUEST_FLAG_OUTPUT_TOKEN_IDS;
    let empty = unsafe { direct_request(raw) }.expect("empty plan");
    assert_eq!(empty.output_token_ids, Some(Vec::new()));
    assert_eq!(empty.effective_max_output_tokens(), 0);
}

#[test]
fn unsupported_or_malformed_metadata_is_rejected_before_admission() {
    let mut raw = request();
    raw.flags = REQUEST_FLAG_REPLAY_CONTEXT;
    raw.replay_context.flags = REPLAY_CONTEXT_FLAG_METADATA;
    raw.replay_context.metadata = bytes(b"opaque-not-json");
    assert_eq!(
        unsafe { direct_request(raw) }.unwrap_err(),
        StatusV1::UNSUPPORTED
    );
    raw.replay_context.metadata = bytes(b"{}");
    raw.replay_context.prompt_token_source = 99;
    assert_eq!(
        unsafe { direct_request(raw) }.unwrap_err(),
        StatusV1::UNSUPPORTED
    );
    raw.replay_context.prompt_token_source = 0;
    raw.flags |= 1 << 31;
    assert_eq!(
        unsafe { direct_request(raw) }.unwrap_err(),
        StatusV1::INVALID_ARGUMENT
    );
}
