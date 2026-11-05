// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
// SPDX-License-Identifier: Apache-2.0
// Modified behavioral adaptation of SGLang v0.5.22; see THIRD_PARTY_NOTICES.md.
// Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d

use super::super::AdmissionEvent;
use super::config::{SglangConfig, ceil_to_block};
use super::request::{SglangRequest, WaitingQueue};
use crate::engine::kv_manager::SglangKvManager;

/// Per-request prefill data needed for FPM snapshot construction.
pub(super) struct PrefillFpmItem {
    pub(super) prompt_len: usize,
    pub(super) tokens_computed: usize,
    /// KV context already present when this chunk runs (cached prefix or the request's own
    /// earlier chunks); drives the forward-pass model.
    pub(super) prefix_tokens: usize,
    /// Radix-cache reuse credited to this pass: the admission match on a request's first chunk,
    /// zero for continuations, which only extend their own tokens.
    pub(super) cache_reused_tokens: usize,
}

#[derive(Default)]
pub(super) struct AdmitResult {
    pub(super) can_run: Vec<SglangRequest>,
    pub(super) admissions: Vec<AdmissionEvent>,
    /// Per-request prefill info for building FPM snapshots.
    pub(super) prefill_fpm: Vec<PrefillFpmItem>,
    pub(super) batch_is_full: bool,
}

/// SGLang v0.5.22 `PrefillBudget` and `PrefillAdder` for a single full-KV pool.
/// AISim allocates each selected request immediately. Add those new pages back
/// when querying capacity so the offsets charge them exactly once; prefix locks
/// remain excluded from the live evictable capacity.
pub(super) fn get_new_batch_prefill(
    waiting: &mut WaitingQueue,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    new_token_ratio: f64,
    running: &[SglangRequest],
    mut batch_is_full: bool,
) -> AdmitResult {
    let mut total_offset: f64 = running
        .iter()
        .map(|req| {
            let remaining_output = req
                .remaining_output_tokens()
                .min(config.clip_max_new_tokens);
            remaining_output as f64 * new_token_ratio
        })
        .sum();
    let mut current_offset = 0.0;
    let mut allocated_this_pass = 0;
    let mut rem_input_tokens = config.max_prefill_tokens as f64;
    let mut rem_chunk_tokens = config.chunked_prefill_size as f64;

    let mut can_run = Vec::new();
    let mut admissions = Vec::new();
    let mut prefill_fpm = Vec::new();

    let available_running_slots = config.max_running_requests.saturating_sub(running.len());
    // An in-flight chunk owns KV and must run before newly matched requests,
    // including under LPM or a full running batch.
    if let Some(index) = waiting.iter().position(|req| req.materialized_tokens > 0) {
        let continuation = waiting.remove(index).expect("continuation is queued");
        waiting.push_front(continuation);
    }
    while let Some(mut req) = waiting.pop_front() {
        let first_admission = req.materialized_tokens == 0;
        if first_admission && (batch_is_full || can_run.len() >= available_running_slots) {
            batch_is_full = true;
            waiting.push_front(req);
            break;
        }
        // SGLang `Req.init_next_round_input(tree_cache)`: a waiting request matches its whole
        // prompt against the radix tree once, capped at `input_len - 1` so a request that must
        // produce a token always computes at least one. Chunked continuations are re-initialised
        // without the tree cache and never re-match; the request simply continues from the tokens
        // it already owns. Zero-output (prefix-only) replay requests keep the full match: they
        // have no first token to sample, so a fully cached prompt is no forward-pass work.
        let cached_prefix = if req.materialized_tokens == 0 && config.enable_prefix_caching {
            req.kv_lease
                .ensure_page_hashes(&req.sequence_tokens, config.block_size);
            let match_len = if req.max_output_tokens == 0 {
                req.current_sequence_len()
            } else {
                req.current_sequence_len().saturating_sub(1)
            };
            let pages = (match_len / config.block_size).min(req.kv_lease.page_hashes().len());
            kv_manager.admission_prefix_match(&mut req.kv_lease, pages)
        } else {
            0
        };
        let start = req.materialized_tokens.max(cached_prefix);
        // SGLang `extend_input_len`: tokens left to compute beyond the matched/owned prefix.
        let extend_input = req.current_sequence_len().saturating_sub(start);
        if extend_input == 0 && req.materialized_tokens > 0 {
            waiting.push_front(req);
            break;
        }

        let capacity = if first_admission {
            kv_manager.protected_admission_capacity(&req.kv_lease, cached_prefix)
        } else {
            kv_manager.cache().available_tokens() + kv_manager.cache().evictable_size
        } + allocated_this_pass;
        let remaining_total = capacity as f64 - total_offset;
        let whole_request_tokens = extend_input as f64
            + req
                .remaining_output_tokens()
                .min(config.clip_max_new_tokens) as f64
            + config.block_size as f64;
        if first_admission && whole_request_tokens >= remaining_total {
            batch_is_full = !can_run.is_empty() || !running.is_empty();
            waiting.push_front(req);
            break;
        }

        let chunk_tokens = if !first_admission {
            // Unlike a new chunk, a continuation uses the remaining budget
            // without page-flooring and may proceed even with no headroom.
            let limit = rem_chunk_tokens.min(remaining_total) as usize;
            let limit = if limit > 0 {
                limit
            } else {
                rem_chunk_tokens as usize
            };
            extend_input.min(limit)
        } else if ceil_to_block(extend_input, config.block_size) as f64 <= rem_chunk_tokens {
            extend_input
        } else {
            let chunk = (rem_chunk_tokens as usize / config.block_size) * config.block_size;
            if chunk == 0 {
                waiting.push_front(req);
                break;
            }
            chunk.min(extend_input)
        };

        let charged_input_tokens = ceil_to_block(chunk_tokens, config.block_size) as f64;
        let output_reserve = if chunk_tokens < extend_input {
            0
        } else {
            req.max_output_tokens.min(config.clip_max_new_tokens)
        };

        let chunk_end = start + chunk_tokens;
        let previously_allocated = req.allocated_tokens;
        let mut lease = std::mem::take(&mut req.kv_lease);
        let alloc_tokens = req.sequence_prefix(chunk_end);

        let prefix_len = if req.materialized_tokens > 0 {
            if !lease.is_active() {
                panic!(
                    "prefill: request {} has materialized_tokens={} but no active KV lease",
                    req.uuid, req.materialized_tokens
                );
            }
            // Continuation: extend the request's own pages; no re-match.
            kv_manager.extend_allocation(alloc_tokens, &mut lease)
        } else {
            // Locks the path the read-only match above saw, matching no further than that cap so
            // physical ownership and `cached_tokens` also stop at `input_len - 1`: the last token
            // of a fully cached prompt is computed and gets its own KV slot.
            kv_manager.allocate_for_request_lease_capped(alloc_tokens, &mut lease, start)
        };

        let Some(prefix_len) = prefix_len else {
            req.kv_lease = lease;
            batch_is_full = !can_run.is_empty() || !running.is_empty();
            waiting.push_front(req);
            break;
        };
        let tokens_computed = chunk_end.saturating_sub(prefix_len);
        // Cache reuse is what the radix tree supplied at admission; a continuation's `prefix_len`
        // is the request's own earlier chunks and must not be reported as a cache hit.
        let admission_reused_tokens = lease.admission_reused_tokens();
        let cache_reused_tokens = if first_admission { prefix_len } else { 0 };

        req.kv_lease = lease;
        req.materialized_tokens = chunk_end;
        req.allocated_tokens = ceil_to_block(chunk_end, config.block_size);
        req.debug_assert_invariants(config.block_size);

        // One admission per request: a continuation chunk is the same admission still running,
        // not a readmission.
        if first_admission {
            admissions.push(AdmissionEvent {
                uuid: req.uuid,
                reused_input_tokens: admission_reused_tokens,
                cache_tier_attribution: None,
            });
        }
        prefill_fpm.push(PrefillFpmItem {
            prompt_len: req.prompt_len(),
            tokens_computed,
            prefix_tokens: prefix_len,
            cache_reused_tokens,
        });

        allocated_this_pass += req.allocated_tokens
            - previously_allocated
            - if first_admission { prefix_len } else { 0 };
        let immediate = charged_input_tokens + config.block_size as f64;
        total_offset += immediate + output_reserve as f64;
        current_offset += immediate;
        rem_input_tokens -= charged_input_tokens;
        rem_chunk_tokens -= charged_input_tokens;
        can_run.push(req);

        let available = (kv_manager.cache().available_tokens()
            + kv_manager.cache().evictable_size
            + allocated_this_pass) as f64;
        if available <= total_offset || available <= current_offset {
            batch_is_full = true;
            break;
        }
        if rem_input_tokens <= 0.0 || rem_chunk_tokens <= 0.0 {
            break;
        }
    }

    AdmitResult {
        can_run,
        admissions,
        prefill_fpm,
        batch_is_full,
    }
}
