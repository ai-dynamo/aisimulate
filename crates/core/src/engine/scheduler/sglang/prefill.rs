// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use super::super::AdmissionEvent;
use super::config::{SglangConfig, ceil_to_block};
use super::request::SglangRequest;
use crate::engine::kv_manager::SglangKvManager;

/// Per-request prefill data needed for FPM snapshot construction.
pub(super) struct PrefillFpmItem {
    pub(super) prompt_len: usize,
    pub(super) tokens_computed: usize,
    pub(super) prefix_tokens: usize,
}

#[derive(Default)]
pub(super) struct AdmitResult {
    pub(super) can_run: Vec<SglangRequest>,
    pub(super) admissions: Vec<AdmissionEvent>,
    pub(super) total_isl: usize,
    pub(super) total_prefix: usize,
    pub(super) oom: bool,
    /// Per-request prefill info for building FPM snapshots.
    pub(super) prefill_fpm: Vec<PrefillFpmItem>,
}

pub(super) fn get_new_batch_prefill(
    waiting: &mut VecDeque<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    new_token_ratio: f64,
    running: &[SglangRequest],
) -> AdmitResult {
    let cache = kv_manager.cache();
    let reserved_decode_output: f64 = running
        .iter()
        .map(|req| {
            let remaining_output = req
                .remaining_output_tokens()
                .min(config.clip_max_new_tokens);
            remaining_output as f64 * new_token_ratio
        })
        .sum();
    // PagePool already removes the full physical footprint of every active
    // partial page from available capacity, so page slack must not be charged
    // a second time here.
    let mut rem_total_tokens =
        (cache.available_tokens() + cache.evictable_size) as f64 - reserved_decode_output;
    let mut rem_input_tokens = config.max_prefill_tokens as f64;
    let mut rem_chunk_tokens = config.chunked_prefill_size as f64;

    let mut can_run = Vec::new();
    let mut admissions = Vec::new();
    let mut prefill_fpm = Vec::new();
    let mut rejected = VecDeque::new();
    let mut oom = false;
    let mut total_isl = 0usize;
    let mut total_prefix = 0usize;

    let available_running_slots = config.max_running_requests.saturating_sub(running.len());
    while can_run.len() < available_running_slots
        && let Some(mut req) = waiting.pop_front()
    {
        // SGLang `Req.init_next_round_input(tree_cache)`: a waiting request matches its whole
        // prompt against the radix tree once, capped at `input_len - 1` so a request that must
        // produce a token always computes at least one. Chunked continuations are re-initialised
        // without the tree cache and never re-match; the request simply continues from the tokens
        // it already owns. Zero-output (prefix-only) replay requests keep the full match: they
        // have no first token to sample, so a fully cached prompt is no forward-pass work.
        let cached_prefix = if req.materialized_tokens == 0 && config.enable_prefix_caching {
            let match_len = if req.max_output_tokens == 0 {
                req.current_sequence_len()
            } else {
                req.current_sequence_len().saturating_sub(1)
            };
            let page_hashes = req.kv_lease.page_hashes();
            let pages = (match_len / config.block_size).min(page_hashes.len());
            kv_manager
                .cache()
                .prefix_match_hashes_len(&page_hashes[..pages])
        } else {
            0
        };
        let start = req.materialized_tokens.max(cached_prefix);
        // SGLang `extend_input_len`: tokens left to compute beyond the matched/owned prefix.
        let extend_input = req.current_sequence_len().saturating_sub(start);
        if extend_input == 0 && req.materialized_tokens > 0 {
            rejected.push_back(req);
            break;
        }

        let chunk_tokens = if extend_input <= config.chunked_prefill_size {
            extend_input
        } else {
            let chunk = (rem_chunk_tokens as usize / config.block_size) * config.block_size;
            if chunk == 0 {
                rejected.push_back(req);
                break;
            }
            chunk.min(extend_input)
        };

        // Budgets are charged for computed tokens only (`PrefillAdder._update_prefill_budget`),
        // not for the cached prefix.
        let charged_input_tokens = ceil_to_block(chunk_tokens, config.block_size) as f64;
        let output_reserve = if chunk_tokens < extend_input {
            0
        } else {
            req.remaining_output_tokens()
                .min(config.clip_max_new_tokens)
        };
        if charged_input_tokens + output_reserve as f64 >= rem_total_tokens {
            rejected.push_back(req);
            break;
        }
        if charged_input_tokens > rem_input_tokens || charged_input_tokens > rem_chunk_tokens {
            rejected.push_back(req);
            break;
        }

        let chunk_end = start + chunk_tokens;
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
            // Locks the same path the read-only match above saw (same thread, eviction inside
            // happens after locking), so the locked prefix is at least `cached_prefix`. Clamp to
            // the SGLang cap so a fully cached prompt still computes its last token.
            kv_manager
                .allocate_for_request_lease(alloc_tokens, &mut lease)
                .map(|matched| matched.min(start))
        };

        let Some(prefix_len) = prefix_len else {
            req.kv_lease = lease;
            rejected.push_back(req);
            oom = true;
            break;
        };
        let tokens_computed = chunk_end.saturating_sub(prefix_len);

        req.kv_lease = lease;
        req.materialized_tokens = chunk_end;
        req.allocated_tokens = ceil_to_block(chunk_end, config.block_size);
        req.debug_assert_invariants(config.block_size);

        admissions.push(AdmissionEvent {
            uuid: req.uuid,
            reused_input_tokens: prefix_len,
            cache_tier_attribution: None,
        });
        prefill_fpm.push(PrefillFpmItem {
            prompt_len: req.prompt_len(),
            tokens_computed,
            prefix_tokens: prefix_len,
        });

        total_isl += chunk_end;
        total_prefix += prefix_len;
        rem_total_tokens -= charged_input_tokens + output_reserve as f64;
        rem_input_tokens -= charged_input_tokens;
        rem_chunk_tokens -= charged_input_tokens;
        can_run.push(req);

        if rem_chunk_tokens <= 0.0 {
            break;
        }
    }

    while let Some(req) = rejected.pop_back() {
        waiting.push_front(req);
    }

    AdmitResult {
        can_run,
        admissions,
        total_isl,
        total_prefix,
        oom,
        prefill_fpm,
    }
}
