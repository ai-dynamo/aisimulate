// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
// SPDX-License-Identifier: Apache-2.0
// Modified behavioral adaptation of SGLang v0.5.22; see THIRD_PARTY_NOTICES.md.
// Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d

use std::sync::Arc;

use crate::engine::FrontendConfig;
use crate::engine::common::perf_model::PerfModel;
use crate::engine::common::protocols::{KvTransferTimingMode, MockEngineArgs, WorkerType};

const DEFAULT_MAX_PREFILL_TOKENS: usize = 16384;
const DEFAULT_CHUNKED_PREFILL_SIZE: usize = 8192;
const DEFAULT_CLIP_MAX_NEW_TOKENS: usize = 4096;
const DEFAULT_VLM_CACHE_BYTES: u64 = 100 * 1024 * 1024;
const DEFAULT_INIT_NEW_TOKEN_RATIO: f64 = 0.7;
const DEFAULT_MIN_NEW_TOKEN_RATIO_FACTOR: f64 = 0.14;
const DEFAULT_NEW_TOKEN_RATIO_DECAY_STEPS: f64 = 600.0;
pub(super) const LPM_FALLBACK_THRESHOLD: usize = 128;
pub(super) const IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD: usize = 32;
pub(super) const IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD: usize = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SchedulePolicy {
    #[default]
    Fifo,
    Lpm,
}

pub(super) struct SglangConfig {
    pub(super) schedule_policy: SchedulePolicy,
    pub(super) max_prefill_tokens: usize,
    pub(super) max_running_requests: usize,
    pub(super) decode_request_slots: usize,
    pub(super) decode_metadata_slots: usize,
    pub(super) reserved_decode_tokens: usize,
    pub(super) chunked_prefill_size: usize,
    pub(super) prefill_decode_interval: usize,
    pub(super) clip_max_new_tokens: usize,
    pub(super) init_new_token_ratio: f64,
    pub(super) min_new_token_ratio: f64,
    pub(super) new_token_ratio_decay_step: f64,
    pub(super) perf_model: Arc<PerfModel>,
    pub(super) speedup_ratio: f64,
    pub(super) decode_speedup_ratio: f64,
    pub(super) pipeline_fill_passes: f64,
    pub(super) worker_type: WorkerType,
    pub(super) block_size: usize,
    pub(super) total_kv_tokens: usize,
    pub(super) max_model_len: Option<usize>,
    pub(super) enable_prefix_caching: bool,
    pub(super) kv_transfer_bytes_per_token: Option<usize>,
    pub(super) kv_transfer_bandwidth: Option<f64>,
    pub(super) kv_transfer_timing_mode: KvTransferTimingMode,
    pub(super) speculative_max_tokens: Option<usize>,
    /// Vision embedding cache capacity in bytes.
    pub(super) vlm_cache_bytes: u64,
    /// Whether a pass is one iteration of the overlap scheduler loop.
    pub(super) host_loop: bool,
    /// Frontend worker pools ahead of the scheduler inbox.
    pub(super) frontend: Option<FrontendConfig>,
}

impl SglangConfig {
    pub(super) fn max_request_len(&self) -> usize {
        self.max_model_len
            .unwrap_or(self.total_kv_tokens)
            .min(self.total_kv_tokens)
            .saturating_sub(1)
    }

    pub(super) fn max_input_len(&self) -> usize {
        self.max_request_len()
            .saturating_sub(5)
            .min(self.total_kv_tokens.saturating_sub(self.block_size))
    }

    pub(super) fn from_args(args: &MockEngineArgs) -> Self {
        let sglang = args.sglang.as_ref();
        debug_assert!(
            sglang
                .and_then(|config| config.page_size)
                .is_none_or(|page_size| page_size == args.block_size),
            "materialized SGLang page size must match the scheduler block size"
        );
        let schedule_conservativeness = sglang
            .and_then(|s| s.schedule_conservativeness)
            .unwrap_or(1.0);
        let init_new_token_ratio =
            (DEFAULT_INIT_NEW_TOKEN_RATIO * schedule_conservativeness).min(1.0);
        let min_new_token_ratio = init_new_token_ratio * DEFAULT_MIN_NEW_TOKEN_RATIO_FACTOR;
        let decay_steps = DEFAULT_NEW_TOKEN_RATIO_DECAY_STEPS;
        let decay_step = (init_new_token_ratio - min_new_token_ratio) / decay_steps;

        // SGLang's decode server runs a chunk cache unless its radix cache is enabled.
        let enable_prefix_caching = args.enable_prefix_caching
            && (args.worker_type != WorkerType::Decode
                || sglang.is_some_and(|s| s.disaggregation_decode_enable_radix_cache));

        // SGLang's decode server serves its waiting queue in arrival order.
        let policy_str = if args.worker_type == WorkerType::Decode {
            None
        } else {
            sglang.and_then(|s| s.schedule_policy.as_deref())
        };
        let schedule_policy = match policy_str {
            Some("lpm") => SchedulePolicy::Lpm,
            Some("fifo") | Some("fcfs") | None => SchedulePolicy::Fifo,
            Some(other) => {
                tracing::warn!(
                    "Unknown sglang schedule_policy '{}', falling back to FIFO",
                    other
                );
                SchedulePolicy::Fifo
            }
        };

        let total_kv_tokens = args.num_gpu_blocks * args.block_size;
        let context_len = args.max_model_len.unwrap_or(total_kv_tokens).max(1);
        let estimated_requests =
            (total_kv_tokens.saturating_mul(512) / context_len).clamp(2048, 4096);
        let max_running_requests = args
            .max_num_seqs
            .unwrap_or(estimated_requests)
            .min(total_kv_tokens / 2);
        let extra_decode_slots = sglang
            .and_then(|s| s.disaggregation_decode_extra_slots)
            .unwrap_or_else(|| {
                args.max_num_seqs
                    .filter(|&limit| limit <= 32)
                    .unwrap_or(0)
                    .saturating_mul(2)
            });

        Self {
            schedule_policy,
            prefill_decode_interval: args.prefill_decode_interval,
            max_running_requests,
            decode_request_slots: max_running_requests.saturating_add(extra_decode_slots),
            decode_metadata_slots: max_running_requests.saturating_mul(2),
            reserved_decode_tokens: sglang
                .and_then(|s| s.num_reserved_decode_tokens)
                .unwrap_or(512),
            max_prefill_tokens: sglang
                .and_then(|s| s.max_prefill_tokens)
                .unwrap_or(DEFAULT_MAX_PREFILL_TOKENS),
            chunked_prefill_size: sglang
                .and_then(|s| s.chunked_prefill_size)
                .unwrap_or(DEFAULT_CHUNKED_PREFILL_SIZE),
            clip_max_new_tokens: sglang
                .and_then(|s| s.clip_max_new_tokens)
                .unwrap_or(DEFAULT_CLIP_MAX_NEW_TOKENS),
            init_new_token_ratio,
            min_new_token_ratio,
            new_token_ratio_decay_step: decay_step,
            perf_model: args.perf_model.clone(),
            speedup_ratio: args.speedup_ratio,
            decode_speedup_ratio: args.decode_speedup_ratio,
            pipeline_fill_passes: args.pipeline_fill_passes,
            worker_type: args.worker_type,
            block_size: args.block_size,
            total_kv_tokens,
            max_model_len: args.max_model_len,
            enable_prefix_caching,
            kv_transfer_bytes_per_token: args.kv_transfer_bytes_per_token,
            kv_transfer_bandwidth: args.kv_transfer_bandwidth,
            kv_transfer_timing_mode: args.kv_transfer_timing_mode,
            speculative_max_tokens: args.aic_nextn.map(|nextn| nextn + 1),
            vlm_cache_bytes: sglang
                .and_then(|s| s.vlm_cache_bytes)
                .unwrap_or(DEFAULT_VLM_CACHE_BYTES),
            host_loop: sglang.is_some_and(|s| s.host_loop),
            frontend: sglang.and_then(|s| s.frontend.clone()),
        }
    }

    /// Storage reservation hint for output tokens that this worker can realize.
    ///
    /// Planned replay outputs keep their known length for allocation-free
    /// appends. Online requests use SGLang's existing decode-reservation clip
    /// so a large caller-declared limit does not eagerly size every queued
    /// request to the whole KV pool.
    pub(super) fn output_storage_hint(
        &self,
        prompt_len: usize,
        max_output_tokens: usize,
        has_planned_output: bool,
    ) -> usize {
        let realizable = self.total_kv_tokens.saturating_sub(prompt_len);
        let requested = if has_planned_output {
            max_output_tokens
        } else {
            max_output_tokens.min(self.clip_max_new_tokens)
        };
        requested.min(realizable)
    }
}

pub(super) fn ceil_to_block(tokens: usize, block_size: usize) -> usize {
    if tokens == 0 {
        return 0;
    }

    tokens.div_ceil(block_size) * block_size
}

pub(super) fn floor_to_block(tokens: usize, block_size: usize) -> usize {
    tokens / block_size * block_size
}
