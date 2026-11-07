// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
// SPDX-License-Identifier: Apache-2.0
// Modified behavioral adaptation of SGLang v0.5.22; see THIRD_PARTY_NOTICES.md.
// Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d

use crate::engine::common::hashing::compute_next_seq_hash;

use crate::engine::kv_manager::SglangKvManager;
use rustc_hash::FxHashSet;

use super::config::{
    IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD, IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD,
    LPM_FALLBACK_THRESHOLD, SchedulePolicy, SglangConfig,
};
use super::request::WaitingQueue;

pub(super) fn apply_schedule_policy(
    waiting: &mut WaitingQueue,
    kv_manager: &SglangKvManager,
    config: &SglangConfig,
) {
    match config.schedule_policy {
        SchedulePolicy::Fifo => {}
        SchedulePolicy::Lpm => {
            if !config.enable_prefix_caching {
                return;
            }
            if waiting.len() > LPM_FALLBACK_THRESHOLD {
                return;
            }

            let page_size = config.block_size.max(1);
            let duplicate_prefix_len =
                IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD.div_ceil(page_size) * page_size;
            let mut waiting_prefixes = FxHashSet::default();
            let mut scored = Vec::with_capacity(waiting.len());
            let mut continuations = Vec::new();

            for mut req in waiting.drain(..) {
                if req.materialized_tokens > 0 {
                    continuations.push(req);
                    continue;
                }
                let sequence_tokens = &req.sequence_tokens;
                req.kv_lease.ensure_page_hashes(sequence_tokens, page_size);
                let pages = req.kv_lease.page_hashes().len();
                let matched = kv_manager.admission_host_match(&mut req.kv_lease, pages);
                let deprioritized = matched.device_tokens
                    <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD
                    && req.sequence_tokens().len() >= duplicate_prefix_len
                    && {
                        let duplicate_prefix =
                            &req.kv_lease.page_hashes()[..duplicate_prefix_len / page_size];
                        let mut local_hashes = duplicate_prefix.iter().copied();
                        let first = local_hashes
                            .next()
                            .expect("duplicate-prefix threshold must cover at least one page");
                        let sequence_hash = local_hashes.fold(first.0, compute_next_seq_hash);
                        !waiting_prefixes.insert(sequence_hash)
                    };

                scored.push((
                    matched.device_tokens + matched.host_tokens,
                    deprioritized,
                    req,
                ));
            }

            scored.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)));

            for req in continuations {
                waiting.push_back(req);
            }
            for (_, _, req) in scored {
                waiting.push_back(req);
            }
        }
    }
}
