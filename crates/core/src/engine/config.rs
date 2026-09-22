// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serializable engine configuration.

use std::sync::Arc;

use anyhow::{Result, ensure};
use serde::{Deserialize, Deserializer, Serialize};

use crate::engine::common::speculative::normalize_conditional_accept_rates;
use crate::engine::handoff::TransferTimingMode;
use crate::engine::timing::{TimingModel, TimingModelConfig, built_in_timing_model};

const DEFAULT_MAX_PREFILL_TOKENS: usize = 16_384;
const DEFAULT_CHUNKED_PREFILL_SIZE: usize = 8_192;
const DEFAULT_CLIP_MAX_NEW_TOKENS: usize = 4_096;
const DEFAULT_SCHEDULE_CONSERVATIVENESS: f64 = 1.0;
const DEFAULT_HOST_OFFLOAD_BANDWIDTH_GBPS: f64 = 32.0;

fn default_num_gpu_blocks() -> usize {
    16_384
}

fn deserialize_explicit_num_gpu_blocks<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    // Preserve the legacy rule that an explicit capacity must be an integer;
    // only an omitted field may use the default or state-cache-derived value.
    usize::deserialize(deserializer).map(Some)
}

fn default_block_size() -> usize {
    64
}

fn default_max_num_seqs() -> usize {
    256
}

fn default_max_num_batched_tokens() -> usize {
    8_192
}

fn default_prefill_schedule_interval() -> usize {
    1
}

fn default_true() -> bool {
    true
}

fn default_one() -> f64 {
    1.0
}

fn default_aic_mtp_seed() -> u64 {
    42
}

fn default_max_prefill_tokens() -> usize {
    DEFAULT_MAX_PREFILL_TOKENS
}

fn default_chunked_prefill_size() -> usize {
    DEFAULT_CHUNKED_PREFILL_SIZE
}

fn default_clip_max_new_tokens() -> usize {
    DEFAULT_CLIP_MAX_NEW_TOKENS
}

fn default_schedule_conservativeness() -> f64 {
    DEFAULT_SCHEDULE_CONSERVATIVENESS
}

fn default_host_offload_bandwidth_gbps() -> f64 {
    DEFAULT_HOST_OFFLOAD_BANDWIDTH_GBPS
}

/// Scheduler semantics selected for an AISimulate rank.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    /// vLLM-style block scheduling.
    #[default]
    Vllm,
    /// SGLang-style radix-cache scheduling.
    Sglang,
    /// TensorRT-LLM scheduling through the shared vLLM-style core.
    Trtllm,
}

impl Backend {
    /// Backend-native KV block size used when a caller does not provide one.
    pub const fn default_block_size(self) -> usize {
        match self {
            Self::Vllm => 64,
            Self::Sglang => 1,
            Self::Trtllm => 32,
        }
    }
}

/// Scheduler role.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerType {
    /// Prefill and decode execute on the same rank.
    #[default]
    Aggregated,
    /// The rank emits its first token with no separate decode latency.
    Prefill,
    /// The rank performs decode work only.
    Decode,
}

/// Decode preemption victim selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreemptionMode {
    /// Evict the most recently admitted runnable request.
    #[default]
    Lifo,
    /// Evict the oldest runnable request.
    Fifo,
}

/// SGLang waiting-queue ordering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SglangSchedulePolicy {
    /// First-in, first-out.
    #[default]
    Fifo,
    /// Longest cached-prefix first for bounded waiting queues.
    Lpm,
}

/// One frontend processing stage: a request-level black box served by a pool
/// of `workers`. Measured frontend service times are lowered into it by the
/// Python configuration layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendStage {
    #[serde(default = "one_worker")]
    pub workers: usize,
    /// Service time of one request running alone on the pool, in milliseconds.
    pub service_ms: f64,
    /// Entry `c - 1` scales the service time while `c` jobs share the pool;
    /// empty keeps the service time independent of sharing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub concurrency_scale: Vec<f64>,
}

/// Frontend stages that requests traverse, in order, before reaching the scheduler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendConfig {
    pub stages: Vec<FrontendStage>,
}

const fn one_worker() -> usize {
    1
}

impl FrontendConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.stages.is_empty(),
            "frontend requires at least one stage"
        );
        for (index, stage) in self.stages.iter().enumerate() {
            ensure!(
                stage.workers > 0,
                "frontend.stages[{index}].workers must be positive"
            );
            ensure!(
                stage.service_ms.is_finite() && stage.service_ms >= 0.0,
                "frontend.stages[{index}].service_ms must be finite and non-negative"
            );
            ensure!(
                stage.concurrency_scale.is_empty()
                    || stage.concurrency_scale.len() == stage.workers,
                "frontend.stages[{index}].concurrency_scale needs one entry per worker ({})",
                stage.workers
            );
            ensure!(
                stage
                    .concurrency_scale
                    .iter()
                    .all(|scale| scale.is_finite() && *scale > 0.0),
                "frontend.stages[{index}].concurrency_scale entries must be finite and positive"
            );
        }
        Ok(())
    }
}

/// Serializable SGLang scheduler controls.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SglangConfig {
    /// Waiting-queue policy.
    pub schedule_policy: SglangSchedulePolicy,
    /// Page-aware prefill-token budget per pass.
    #[serde(default = "default_max_prefill_tokens")]
    pub max_prefill_tokens: usize,
    /// Maximum prompt chunk considered in one pass.
    #[serde(default = "default_chunked_prefill_size")]
    pub chunked_prefill_size: usize,
    /// Output reservation cap used by SGLang admission control.
    #[serde(default = "default_clip_max_new_tokens")]
    pub clip_max_new_tokens: usize,
    /// Multiplier applied to SGLang's adaptive output-reservation ratio.
    #[serde(default = "default_schedule_conservativeness")]
    pub schedule_conservativeness: f64,
    /// Vision embedding cache capacity in bytes (`SGLANG_VLM_CACHE_SIZE_MB`).
    #[serde(default = "default_vlm_cache_bytes")]
    pub vlm_cache_bytes: u64,
    /// Model one pass as one iteration of SGLang's overlap scheduler loop:
    /// requests are received at iteration boundaries and a forward's outputs,
    /// terminals, and prefix-cache commits become visible one iteration later.
    /// Off, a pass is one forward whose outputs are visible when it ends. See
    /// `crates/core/src/engine/scheduler/sglang/host_loop.rs`.
    pub host_loop: bool,
}

const fn default_vlm_cache_bytes() -> u64 {
    100 * 1024 * 1024
}

impl Default for SglangConfig {
    fn default() -> Self {
        Self {
            schedule_policy: SglangSchedulePolicy::Fifo,
            max_prefill_tokens: default_max_prefill_tokens(),
            chunked_prefill_size: default_chunked_prefill_size(),
            clip_max_new_tokens: default_clip_max_new_tokens(),
            schedule_conservativeness: default_schedule_conservativeness(),
            vlm_cache_bytes: default_vlm_cache_bytes(),
            host_loop: false,
        }
    }
}

impl SglangConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.max_prefill_tokens > 0,
            "sglang.max_prefill_tokens must be positive"
        );
        ensure!(
            self.chunked_prefill_size > 0,
            "sglang.chunked_prefill_size must be positive"
        );
        ensure!(
            self.schedule_conservativeness.is_finite() && self.schedule_conservativeness >= 0.0,
            "sglang.schedule_conservativeness must be finite and non-negative"
        );
        Ok(())
    }
}

/// TensorRT-LLM capacity scheduler policy.
///
/// The mocker currently models the TensorRT-LLM default only. Keeping
/// the policy explicit prevents a config from silently falling back to vLLM
/// admission or preemption semantics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrtllmCapacityPolicy {
    /// Reserve each admitted request through completion and never evict it.
    #[default]
    GuaranteedNoEvict,
}

/// Serializable TensorRT-LLM scheduler controls.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrtllmConfig {
    /// Capacity scheduler policy.
    pub capacity_scheduler_policy: TrtllmCapacityPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum G3Scope {
    WorkerLocal,
    ClusterShared,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G3OffloadConfig {
    pub scope: G3Scope,
    pub num_g3_blocks: usize,
    #[serde(default = "G3OffloadConfig::default_latency")]
    pub latency_to_first_byte_ms: f64,
    #[serde(default = "G3OffloadConfig::default_worker_bandwidth")]
    pub read_bandwidth_gbps: f64,
    #[serde(default = "G3OffloadConfig::default_worker_bandwidth")]
    pub write_bandwidth_gbps: f64,
    #[serde(default = "G3OffloadConfig::default_shared_bandwidth")]
    pub shared_read_bandwidth_gbps: f64,
    #[serde(default = "G3OffloadConfig::default_shared_bandwidth")]
    pub shared_write_bandwidth_gbps: f64,
}

impl G3OffloadConfig {
    fn default_latency() -> f64 {
        0.1
    }

    fn default_worker_bandwidth() -> f64 {
        10.0
    }

    fn default_shared_bandwidth() -> f64 {
        80.0
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.num_g3_blocks > 0,
            "g3_offload.num_g3_blocks must be positive"
        );
        for value in [
            self.latency_to_first_byte_ms,
            self.read_bandwidth_gbps,
            self.write_bandwidth_gbps,
            self.shared_read_bandwidth_gbps,
            self.shared_write_bandwidth_gbps,
        ] {
            ensure!(
                value.is_finite() && value >= 0.0,
                "g3_offload timing must be finite and non-negative"
            );
        }
        Ok(())
    }
}

/// Physical controls for framework-native G1-to-host offload.
///
/// Framework policy remains selected by [`EngineConfig::backend`]. This
/// descriptor intentionally contains only shared capacity and transfer
/// parameters so additional framework profiles can reuse it without exposing
/// unsupported policy combinations. Physical bytes per block are derived from
/// [`EngineConfig::block_size`] and [`EngineConfig::kv_cache_bytes_per_token`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct NativeHostOffloadConfig {
    /// Physical host-cache capacity in KV blocks.
    pub num_host_blocks: usize,
    /// Modeled device-to-host bandwidth in decimal GB/s. Zero is instantaneous.
    #[serde(default = "default_host_offload_bandwidth_gbps")]
    pub d2h_bandwidth_gbps: f64,
    /// Modeled host-to-device bandwidth in decimal GB/s. Zero is instantaneous.
    #[serde(default = "default_host_offload_bandwidth_gbps")]
    pub h2d_bandwidth_gbps: f64,
}

impl NativeHostOffloadConfig {
    pub const fn new(num_host_blocks: usize) -> Self {
        Self {
            num_host_blocks,
            d2h_bandwidth_gbps: DEFAULT_HOST_OFFLOAD_BANDWIDTH_GBPS,
            h2d_bandwidth_gbps: DEFAULT_HOST_OFFLOAD_BANDWIDTH_GBPS,
        }
    }

    pub const fn with_bandwidths(mut self, d2h_gbps: f64, h2d_gbps: f64) -> Self {
        self.d2h_bandwidth_gbps = d2h_gbps;
        self.h2d_bandwidth_gbps = h2d_gbps;
        self
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.num_host_blocks > 0,
            "native_host_offload.num_host_blocks must be positive"
        );
        ensure!(
            self.d2h_bandwidth_gbps.is_finite() && self.d2h_bandwidth_gbps >= 0.0,
            "native_host_offload.d2h_bandwidth_gbps must be finite and non-negative"
        );
        ensure!(
            self.h2d_bandwidth_gbps.is_finite() && self.h2d_bandwidth_gbps >= 0.0,
            "native_host_offload.h2d_bandwidth_gbps must be finite and non-negative"
        );
        Ok(())
    }
}

/// Recurrent-state allocation size for one simulated rank/GPU.
/// Token block geometry and pool capacity use the existing EngineConfig fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateCacheConfig {
    /// Bytes occupied by one physical working state or snapshot.
    pub bytes_per_request: usize,
}

impl StateCacheConfig {
    /// State size in token-equivalent blocks, rounded up without overflow.
    pub fn state_blocks(&self, block_size: usize, bytes_per_token: usize) -> Result<usize> {
        ensure!(
            self.bytes_per_request > 0,
            "state_cache.bytes_per_request must be positive"
        );
        ensure!(
            block_size >= 2,
            "state_cache requires block_size at least two"
        );
        ensure!(
            bytes_per_token > 0,
            "state_cache requires positive kv_cache_bytes_per_token"
        );
        let block_bytes = block_size
            .checked_mul(bytes_per_token)
            .ok_or_else(|| anyhow::anyhow!("KV block byte size overflowed"))?;
        Ok(self.bytes_per_request.div_ceil(block_bytes))
    }
}

/// Serializable configuration for one scheduler rank.
///
/// Attention-DP size and worker identity belong to
/// [`crate::engine::generalized::GeneralizedEngineConfig`] and
/// [`crate::engine::generalized::EngineIdentity`], not this rank-local configuration.
///
/// [`Default`] constructs a vLLM configuration. Changing only [`Self::backend`]
/// afterward does not recompute backend-dependent fields such as
/// [`Self::block_size`]; start with [`Self::for_backend`] when constructing a
/// different backend in Rust. Deserialization selects the backend's block-size
/// default when `block_size` is omitted.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EngineConfig {
    /// Scheduler backend whose semantics this rank executes.
    ///
    /// Use [`Self::for_backend`] instead of changing this field on
    /// [`Self::default`] when backend-dependent defaults are desired.
    pub backend: Backend,
    /// Physical G1 capacity in blocks.
    #[serde(default = "default_num_gpu_blocks")]
    pub num_gpu_blocks: usize,
    /// KV block size in tokens.
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// Optional prompt-plus-output token limit for every backend.
    /// Prompts at or above the limit are rejected; generation stops at the limit.
    pub max_model_len: Option<usize>,
    /// Maximum concurrently runnable sequences.
    #[serde(default = "default_max_num_seqs")]
    pub max_num_seqs: usize,
    /// Per-pass token budget.
    #[serde(default = "default_max_num_batched_tokens")]
    pub max_num_batched_tokens: usize,
    /// Admit vLLM prefills only once every N attention-DP group passes.
    #[serde(default = "default_prefill_schedule_interval")]
    pub prefill_schedule_interval: usize,
    /// SGLang scheduler rounds without prefill after a globally synchronized EXTEND.
    /// Includes chunk continuation and idle rounds; zero disables the interval.
    #[serde(default)]
    pub prefill_decode_interval: usize,
    /// Whether complete blocks remain reusable after request release.
    #[serde(default = "default_true")]
    pub enable_prefix_caching: bool,
    /// Whether a prompt may be split across scheduler passes.
    #[serde(default = "default_true")]
    pub enable_chunked_prefill: bool,
    /// Divisor applied to modeled prefill and decode latency.
    #[serde(default = "default_one")]
    pub speedup_ratio: f64,
    /// Additional divisor applied to decode latency.
    #[serde(default = "default_one")]
    pub decode_speedup_ratio: f64,
    /// MTP/EAGLE draft-token count. One verification forward can emit up to
    /// `aic_nextn + 1` output tokens.
    pub aic_nextn: Option<usize>,
    /// Conditional draft acceptance rates, comma-separated.
    ///
    /// Entry `i` is the probability that draft `i` is accepted given that
    /// every preceding draft was accepted.
    pub aic_nextn_accept_rates: Option<String>,
    /// Base seed for deterministic worker-local MTP acceptance sampling.
    #[serde(default = "default_aic_mtp_seed")]
    pub aic_mtp_seed: u64,
    /// Scheduler role.
    pub worker_type: WorkerType,
    /// Decode preemption victim order.
    pub preemption_mode: PreemptionMode,
    /// Retain and expose local token-block hashes in neutral KV events.
    pub emit_kv_events: bool,
    /// Retain block token IDs alongside neutral KV events.
    pub emit_kv_token_ids: bool,
    /// Bytes transferred per prompt token for disaggregated handoff timing.
    pub kv_transfer_bytes_per_token: Option<usize>,
    /// Physical KV-cache bytes occupied by one token for host offload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache_bytes_per_token: Option<usize>,
    /// Optional framework-native host-offload simulation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_host_offload: Option<NativeHostOffloadConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub g3_offload: Option<crate::engine::G3OffloadConfig>,
    /// Optional manual vLLM G1 token/state cache configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_cache: Option<StateCacheConfig>,
    /// Modeled prefill-to-decode transfer bandwidth in decimal GB/s.
    pub kv_transfer_bandwidth: Option<f64>,
    /// Prompt footprint used to model disaggregated transfer time.
    pub kv_transfer_timing_mode: TransferTimingMode,
    /// Serializable timing-provider descriptor.
    pub timing_model: TimingModelConfig,
    /// SGLang-only scheduler controls.
    pub sglang: SglangConfig,
    /// TensorRT-LLM-only scheduler controls.
    pub trtllm: TrtllmConfig,
    /// SGLang-only frontend stages; requires `sglang.host_loop`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frontend: Option<FrontendConfig>,
    /// The rank hosts the model's vision encoder: image batches are timed
    /// through the timing provider and the encoder weights and embedding cache
    /// are deducted from the KV budget.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub vision: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EngineConfigWire {
    #[serde(default)]
    backend: Backend,
    #[serde(
        default,
        deserialize_with = "deserialize_explicit_num_gpu_blocks",
        skip_serializing_if = "Option::is_none"
    )]
    num_gpu_blocks: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    block_size: Option<usize>,
    #[serde(default)]
    max_model_len: Option<usize>,
    #[serde(default = "default_max_num_seqs")]
    max_num_seqs: usize,
    #[serde(default = "default_max_num_batched_tokens")]
    max_num_batched_tokens: usize,
    #[serde(default = "default_prefill_schedule_interval")]
    prefill_schedule_interval: usize,
    #[serde(default)]
    prefill_decode_interval: usize,
    #[serde(default = "default_true")]
    enable_prefix_caching: bool,
    #[serde(default = "default_true")]
    enable_chunked_prefill: bool,
    #[serde(default = "default_one")]
    speedup_ratio: f64,
    #[serde(default = "default_one")]
    decode_speedup_ratio: f64,
    #[serde(default)]
    aic_nextn: Option<usize>,
    #[serde(default)]
    aic_nextn_accept_rates: Option<String>,
    #[serde(default = "default_aic_mtp_seed")]
    aic_mtp_seed: u64,
    #[serde(default)]
    worker_type: WorkerType,
    #[serde(default)]
    preemption_mode: PreemptionMode,
    #[serde(default)]
    emit_kv_events: bool,
    #[serde(default)]
    emit_kv_token_ids: bool,
    #[serde(default, alias = "kv_bytes_per_token")]
    kv_transfer_bytes_per_token: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    kv_cache_bytes_per_token: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    native_host_offload: Option<NativeHostOffloadConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    state_cache: Option<StateCacheConfig>,
    #[serde(default)]
    g3_offload: Option<crate::engine::G3OffloadConfig>,
    #[serde(default)]
    kv_transfer_bandwidth: Option<f64>,
    #[serde(default)]
    kv_transfer_timing_mode: TransferTimingMode,
    #[serde(default)]
    timing_model: TimingModelConfig,
    #[serde(default)]
    sglang: SglangConfig,
    #[serde(default)]
    trtllm: TrtllmConfig,
    #[serde(default)]
    frontend: Option<FrontendConfig>,
    #[serde(default)]
    vision: bool,
}

impl<'de> Deserialize<'de> for EngineConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = EngineConfigWire::deserialize(deserializer)?;
        if wire.state_cache.is_some()
            && (wire.num_gpu_blocks.is_none()
                || wire.block_size.is_none()
                || wire.kv_cache_bytes_per_token.is_none())
        {
            return Err(serde::de::Error::custom(
                "state_cache requires explicit num_gpu_blocks, block_size and kv_cache_bytes_per_token",
            ));
        }
        let num_gpu_blocks = wire.num_gpu_blocks.unwrap_or_else(default_num_gpu_blocks);
        let block_size = wire
            .block_size
            .unwrap_or_else(|| wire.backend.default_block_size());
        let config = Self {
            backend: wire.backend,
            num_gpu_blocks,
            block_size,
            max_model_len: wire.max_model_len,
            max_num_seqs: wire.max_num_seqs,
            max_num_batched_tokens: wire.max_num_batched_tokens,
            prefill_schedule_interval: wire.prefill_schedule_interval,
            prefill_decode_interval: wire.prefill_decode_interval,
            enable_prefix_caching: wire.enable_prefix_caching,
            enable_chunked_prefill: wire.enable_chunked_prefill,
            speedup_ratio: wire.speedup_ratio,
            decode_speedup_ratio: wire.decode_speedup_ratio,
            aic_nextn: wire.aic_nextn,
            aic_nextn_accept_rates: wire.aic_nextn_accept_rates,
            aic_mtp_seed: wire.aic_mtp_seed,
            worker_type: wire.worker_type,
            preemption_mode: wire.preemption_mode,
            emit_kv_events: wire.emit_kv_events,
            emit_kv_token_ids: wire.emit_kv_token_ids,
            kv_transfer_bytes_per_token: wire.kv_transfer_bytes_per_token,
            kv_cache_bytes_per_token: wire.kv_cache_bytes_per_token,
            native_host_offload: wire.native_host_offload,
            g3_offload: wire.g3_offload,
            state_cache: wire.state_cache,
            kv_transfer_bandwidth: wire.kv_transfer_bandwidth,
            kv_transfer_timing_mode: wire.kv_transfer_timing_mode,
            timing_model: wire.timing_model,
            sglang: wire.sglang,
            trtllm: wire.trtllm,
            frontend: wire.frontend,
            vision: wire.vision,
        };
        config
            .validate_state_cache()
            .map_err(serde::de::Error::custom)?;
        Ok(config)
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Vllm,
            num_gpu_blocks: default_num_gpu_blocks(),
            block_size: default_block_size(),
            max_model_len: None,
            max_num_seqs: default_max_num_seqs(),
            max_num_batched_tokens: default_max_num_batched_tokens(),
            prefill_schedule_interval: default_prefill_schedule_interval(),
            prefill_decode_interval: 0,
            enable_prefix_caching: true,
            enable_chunked_prefill: true,
            speedup_ratio: 1.0,
            decode_speedup_ratio: 1.0,
            aic_nextn: None,
            aic_nextn_accept_rates: None,
            aic_mtp_seed: default_aic_mtp_seed(),
            worker_type: WorkerType::Aggregated,
            preemption_mode: PreemptionMode::Lifo,
            emit_kv_events: false,
            emit_kv_token_ids: false,
            kv_transfer_bytes_per_token: None,
            kv_cache_bytes_per_token: None,
            native_host_offload: None,
            g3_offload: None,
            state_cache: None,
            kv_transfer_bandwidth: None,
            kv_transfer_timing_mode: TransferTimingMode::FullPrompt,
            timing_model: TimingModelConfig::Polynomial,
            sglang: SglangConfig::default(),
            trtllm: TrtllmConfig::default(),
            frontend: None,
            vision: false,
        }
    }
}

impl EngineConfig {
    /// Construct a configuration with the selected backend's native defaults.
    ///
    /// In particular, this selects [`Backend::default_block_size`] instead of
    /// inheriting the vLLM block size from [`Self::default`].
    pub fn for_backend(backend: Backend) -> Self {
        Self {
            backend,
            block_size: backend.default_block_size(),
            ..Self::default()
        }
    }

    fn validate_state_cache(&self) -> Result<()> {
        if let Some(state_cache) = &self.state_cache {
            ensure!(
                self.backend == Backend::Vllm,
                "state_cache is supported only for backend=vllm"
            );
            ensure!(
                self.worker_type == WorkerType::Aggregated,
                "state_cache is supported only for worker_type=aggregated"
            );
            ensure!(
                self.native_host_offload.is_none(),
                "state_cache does not support native_host_offload in the G1-only implementation"
            );
            ensure!(
                self.g3_offload.is_none(),
                "state_cache does not support g3_offload in the G1-only implementation"
            );
            ensure!(
                self.kv_transfer_bytes_per_token.is_none() && self.kv_transfer_bandwidth.is_none(),
                "state_cache does not support kv_transfer_bytes_per_token or kv_transfer_bandwidth"
            );
            // Keep the default accepted, including serialized configs that emit it explicitly.
            ensure!(
                self.kv_transfer_timing_mode == TransferTimingMode::FullPrompt,
                "state_cache does not support non-default kv_transfer_timing_mode"
            );
            let bytes_per_token = self
                .kv_cache_bytes_per_token
                .ok_or_else(|| anyhow::anyhow!("state_cache requires kv_cache_bytes_per_token"))?;
            let state_blocks = state_cache.state_blocks(self.block_size, bytes_per_token)?;
            let minimum = state_blocks
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("state_cache minimum capacity overflowed"))?;
            ensure!(
                self.num_gpu_blocks >= minimum,
                "state_cache capacity must fit one token block and one working state"
            );
        }
        Ok(())
    }

    pub(crate) fn validate(&self) -> Result<()> {
        self.validate_state_cache()?;
        ensure!(self.num_gpu_blocks > 0, "num_gpu_blocks must be positive");
        ensure!(self.block_size > 0, "block_size must be positive");
        if matches!(self.backend, Backend::Vllm | Backend::Trtllm) {
            ensure!(
                self.block_size >= 2,
                "vLLM/TRT-LLM block_size must be at least two"
            );
        }
        ensure!(self.max_num_seqs > 0, "max_num_seqs must be positive");
        ensure!(
            self.max_num_batched_tokens > 0,
            "max_num_batched_tokens must be positive"
        );
        ensure!(
            self.prefill_schedule_interval > 0,
            "prefill_schedule_interval must be positive"
        );
        ensure!(
            self.backend == Backend::Vllm || self.prefill_schedule_interval == 1,
            "prefill_schedule_interval is supported only for backend=vllm; use prefill_decode_interval for backend=sglang"
        );
        ensure!(
            self.backend == Backend::Sglang || self.prefill_decode_interval == 0,
            "prefill_decode_interval is supported only for backend=sglang"
        );
        ensure!(
            self.backend == Backend::Sglang || !self.sglang.host_loop,
            "sglang.host_loop is supported only for backend=sglang"
        );
        ensure!(
            !self.sglang.host_loop || self.worker_type != WorkerType::Decode,
            "sglang.host_loop is not modeled on a decode rank; the decode rank is the text-PD decode rank"
        );
        if let Some(frontend) = &self.frontend {
            ensure!(
                self.backend == Backend::Sglang && self.sglang.host_loop,
                "frontend requires backend=sglang with sglang.host_loop enabled"
            );
            frontend.validate()?;
        }
        ensure!(
            !self.vision || self.backend == Backend::Sglang,
            "vision is supported only for backend=sglang"
        );
        ensure!(
            self.max_model_len.is_none_or(|limit| limit > 0),
            "max_model_len must be positive"
        );
        ensure!(
            self.speedup_ratio.is_finite() && self.speedup_ratio >= 0.0,
            "speedup_ratio must be finite and non-negative"
        );
        ensure!(
            self.decode_speedup_ratio.is_finite() && self.decode_speedup_ratio >= 0.0,
            "decode_speedup_ratio must be finite and non-negative"
        );
        if let Some(nextn) = self.aic_nextn {
            normalize_conditional_accept_rates(nextn, self.aic_nextn_accept_rates.as_deref())?;
            ensure!(
                self.decode_speedup_ratio == 1.0,
                "aic_nextn requires decode_speedup_ratio=1.0 because MTP output acceleration is modeled by burst sampling"
            );
        } else {
            ensure!(
                self.aic_nextn_accept_rates.is_none(),
                "aic_nextn_accept_rates requires aic_nextn"
            );
        }
        if self.backend == Backend::Sglang {
            ensure!(
                self.enable_chunked_prefill,
                "enable_chunked_prefill=false is not supported for backend=sglang"
            );
            self.sglang.validate()?;
        }
        ensure!(
            !self.emit_kv_token_ids || self.emit_kv_events,
            "emit_kv_token_ids requires emit_kv_events"
        );
        ensure!(
            self.kv_transfer_bytes_per_token
                .is_none_or(|bytes| bytes > 0),
            "kv_transfer_bytes_per_token must be positive"
        );
        ensure!(
            self.kv_cache_bytes_per_token.is_none_or(|bytes| bytes > 0),
            "kv_cache_bytes_per_token must be positive"
        );
        if let Some(g3) = &self.g3_offload {
            g3.validate()?;
            anyhow::ensure!(
                self.native_host_offload.is_some(),
                "g3_offload requires native_host_offload"
            );
        }
        if let Some(host_offload) = &self.native_host_offload {
            host_offload.validate()?;
            ensure!(
                self.backend == Backend::Vllm,
                "native_host_offload is supported only for backend=vllm"
            );
            ensure!(
                self.worker_type == WorkerType::Aggregated,
                "native_host_offload is supported only for worker_type=aggregated"
            );
            ensure!(
                self.enable_prefix_caching,
                "native_host_offload requires enable_prefix_caching=true"
            );
            ensure!(
                self.aic_nextn.is_none(),
                "native_host_offload does not support aic_nextn in the initial implementation"
            );
            let kv_bytes_per_token = self.kv_cache_bytes_per_token.ok_or_else(|| {
                anyhow::anyhow!(
                    "native_host_offload requires kv_cache_bytes_per_token to derive the physical host block size"
                )
            })?;
            let block_bytes = self
                .block_size
                .checked_mul(kv_bytes_per_token)
                .filter(|bytes| *bytes > 0)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "native_host_offload requires block_size * kv_cache_bytes_per_token to produce a positive, representable block size"
                    )
                })?;
            let capacity_bytes = host_offload
                .num_host_blocks
                .checked_mul(block_bytes)
                .ok_or_else(|| {
                    anyhow::anyhow!("native_host_offload capacity in bytes overflowed")
                })?;
            for (name, bandwidth) in [
                ("d2h_bandwidth_gbps", host_offload.d2h_bandwidth_gbps),
                ("h2d_bandwidth_gbps", host_offload.h2d_bandwidth_gbps),
            ] {
                let bytes_per_ms = bandwidth * 1_000_000.0;
                ensure!(
                    bytes_per_ms.is_finite()
                        && (bandwidth == 0.0 || (capacity_bytes as f64 / bytes_per_ms).is_finite()),
                    "native_host_offload.{name} produces an unrepresentable transfer duration"
                );
            }
        }
        ensure!(
            self.kv_transfer_bandwidth
                .is_none_or(|bandwidth| bandwidth.is_finite() && bandwidth >= 0.0),
            "kv_transfer_bandwidth must be finite and non-negative"
        );
        match &self.timing_model {
            TimingModelConfig::Polynomial => {}
            TimingModelConfig::Fixed {
                prefill_ms,
                decode_ms,
            } => {
                ensure!(
                    prefill_ms.is_finite() && *prefill_ms >= 0.0,
                    "fixed prefill latency must be finite and non-negative"
                );
                ensure!(
                    decode_ms.is_finite() && *decode_ms >= 0.0,
                    "fixed decode latency must be finite and non-negative"
                );
            }
            TimingModelConfig::External { provider, .. } => {
                ensure!(
                    !provider.trim().is_empty(),
                    "timing provider cannot be empty"
                );
            }
        }
        Ok(())
    }

    pub(crate) fn built_in_timing_model(&self) -> Result<Arc<dyn TimingModel>> {
        built_in_timing_model(&self.timing_model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type InvalidHostConfigCase = (fn(&mut EngineConfig), &'static str);

    fn native_host_offload_config() -> EngineConfig {
        EngineConfig {
            block_size: 16,
            kv_cache_bytes_per_token: Some(128 * 1024),
            native_host_offload: Some(NativeHostOffloadConfig {
                num_host_blocks: 4_096,
                d2h_bandwidth_gbps: DEFAULT_HOST_OFFLOAD_BANDWIDTH_GBPS,
                h2d_bandwidth_gbps: DEFAULT_HOST_OFFLOAD_BANDWIDTH_GBPS,
            }),
            ..EngineConfig::default()
        }
    }

    fn assert_invalid_host_config(mutate: impl FnOnce(&mut EngineConfig), expected_message: &str) {
        let mut config = native_host_offload_config();
        mutate(&mut config);
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains(expected_message),
            "validation error did not contain {expected_message:?}"
        );
    }

    fn state_cache_config_json() -> serde_json::Value {
        serde_json::json!({
            "num_gpu_blocks": 8, "block_size": 64, "kv_cache_bytes_per_token": 16,
            "state_cache": {"bytes_per_request": 1500}
        })
    }

    #[test]
    fn state_cache_uses_shared_geometry_and_round_trips() {
        let config: EngineConfig = serde_json::from_value(state_cache_config_json()).unwrap();
        assert_eq!(
            config
                .state_cache
                .unwrap()
                .state_blocks(config.block_size, config.kv_cache_bytes_per_token.unwrap())
                .unwrap(),
            2
        );
        config.validate().unwrap();
        let encoded = serde_json::to_value(&config).unwrap();
        assert_eq!(encoded["num_gpu_blocks"], 8);
        assert_eq!(encoded["block_size"], 64);
        assert_eq!(
            encoded["state_cache"],
            serde_json::json!({"bytes_per_request": 1500})
        );
        assert_eq!(
            serde_json::from_value::<EngineConfig>(encoded).unwrap(),
            config
        );
        let legacy = serde_json::to_value(EngineConfig::default()).unwrap();
        assert!(legacy.get("state_cache").is_none());
        assert!(legacy.get("kv_cache_bytes_per_token").is_none());
    }

    #[test]
    fn state_cache_requires_explicit_shared_geometry_and_positive_state_size() {
        for field in ["num_gpu_blocks", "block_size", "kv_cache_bytes_per_token"] {
            let mut input = state_cache_config_json();
            input.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<EngineConfig>(input).is_err());
            for value in [
                serde_json::json!(0),
                serde_json::json!(-1),
                serde_json::json!(1.5),
                serde_json::json!(true),
                serde_json::Value::Null,
            ] {
                let mut input = state_cache_config_json();
                input[field] = value;
                assert!(serde_json::from_value::<EngineConfig>(input).is_err());
            }
        }
        for state in [
            serde_json::json!({}),
            serde_json::json!({"bytes_per_request":0}),
            serde_json::json!({"bytes_per_request":-1}),
            serde_json::json!({"bytes_per_request":true}),
            serde_json::json!({"bytes_per_request":1500,"tokens_per_block":64}),
        ] {
            let mut input = state_cache_config_json();
            input["state_cache"] = state;
            assert!(serde_json::from_value::<EngineConfig>(input).is_err());
        }
    }

    #[test]
    fn explicit_null_capacity_does_not_fall_back_to_a_default() {
        assert!(
            serde_json::from_value::<EngineConfig>(serde_json::json!({
                "num_gpu_blocks": null
            }))
            .is_err()
        );
        let mut input = state_cache_config_json();
        input["num_gpu_blocks"] = serde_json::Value::Null;
        assert!(serde_json::from_value::<EngineConfig>(input).is_err());
    }

    #[test]
    fn state_cache_validates_overflow_and_minimum_capacity() {
        for (field, value, message) in [
            ("kv_cache_bytes_per_token", usize::MAX, "overflow"),
            ("block_size", usize::MAX, "overflow"),
            ("block_size", 1, "at least two"),
            ("num_gpu_blocks", 2, "one token block and one working state"),
        ] {
            let mut input = state_cache_config_json();
            input[field] = serde_json::json!(value);
            let error = serde_json::from_value::<EngineConfig>(input).unwrap_err();
            assert!(error.to_string().contains(message), "{error}");
        }
        let mut input = state_cache_config_json();
        input["num_gpu_blocks"] = serde_json::json!(3);
        serde_json::from_value::<EngineConfig>(input)
            .unwrap()
            .validate()
            .unwrap();
        assert_eq!(
            StateCacheConfig {
                bytes_per_request: usize::MAX
            }
            .state_blocks(2, 1)
            .unwrap(),
            usize::MAX / 2 + 1
        );
    }

    #[test]
    fn state_cache_rejects_non_vllm_disaggregated_and_host_offload_configs() {
        for (field, value, message) in [
            ("backend", serde_json::json!("sglang"), "backend=vllm"),
            ("backend", serde_json::json!("trtllm"), "backend=vllm"),
            (
                "worker_type",
                serde_json::json!("prefill"),
                "worker_type=aggregated",
            ),
            (
                "worker_type",
                serde_json::json!("decode"),
                "worker_type=aggregated",
            ),
            (
                "native_host_offload",
                serde_json::json!({"num_host_blocks": 8}),
                "native_host_offload",
            ),
            (
                "g3_offload",
                serde_json::json!({"scope": "worker_local", "num_g3_blocks": 8}),
                "g3_offload",
            ),
        ] {
            let mut input = state_cache_config_json();
            input[field] = value;
            let error = serde_json::from_value::<EngineConfig>(input).unwrap_err();
            assert!(error.to_string().contains(message), "{field}: {error}");
        }
    }

    #[test]
    fn state_cache_transfer_validation_preserves_default_roundtrip() {
        for explicit_default in [false, true] {
            let mut input = state_cache_config_json();
            if explicit_default {
                input["kv_transfer_timing_mode"] = serde_json::json!("full_prompt");
            }
            let config: EngineConfig = serde_json::from_value(input).unwrap();
            config.validate().unwrap();
            let roundtrip: EngineConfig =
                serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
            assert_eq!(config, roundtrip);
        }
        for (field, value) in [
            ("kv_transfer_bytes_per_token", serde_json::json!(16)),
            ("kv_bytes_per_token", serde_json::json!(16)),
            ("kv_transfer_bandwidth", serde_json::json!(0.0)),
            (
                "kv_transfer_timing_mode",
                serde_json::json!("destination_missing"),
            ),
        ] {
            let mut input = state_cache_config_json();
            input[field] = value;
            assert!(
                serde_json::from_value::<EngineConfig>(input.clone()).is_err(),
                "{field}"
            );
            input.as_object_mut().unwrap().remove("state_cache");
            let mut config: EngineConfig = serde_json::from_value(input).unwrap();
            config.validate().unwrap();
            config.state_cache = Some(StateCacheConfig {
                bytes_per_request: 1500,
            });
            assert!(config.validate().is_err(), "{field}");
        }
    }

    #[test]
    fn state_cache_validates_geometry_for_direct_rust_construction() {
        let config: EngineConfig = serde_json::from_value(state_cache_config_json()).unwrap();
        let mut invalid = config.clone();
        invalid.kv_cache_bytes_per_token = None;
        assert!(invalid.validate().is_err());
        invalid = config;
        invalid.num_gpu_blocks = 2;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn deserialization_uses_backend_native_block_size() {
        for (backend, expected) in [("vllm", 64), ("sglang", 1), ("trtllm", 32)] {
            let config: EngineConfig =
                serde_json::from_value(serde_json::json!({ "backend": backend })).unwrap();
            assert_eq!(config.block_size, expected, "backend={backend}");
        }
    }

    #[test]
    fn for_backend_uses_backend_native_block_size() {
        for backend in [Backend::Vllm, Backend::Sglang, Backend::Trtllm] {
            let config = EngineConfig::for_backend(backend);
            assert_eq!(config.backend, backend);
            assert_eq!(config.block_size, backend.default_block_size());
        }
    }

    #[test]
    fn deserialization_preserves_an_explicit_block_size() {
        let config: EngineConfig = serde_json::from_value(serde_json::json!({
            "backend": "sglang",
            "block_size": 17
        }))
        .unwrap();
        assert_eq!(config.block_size, 17);
    }

    #[test]
    fn legacy_kv_bytes_per_token_deserializes_to_transfer_geometry() {
        let config: EngineConfig = serde_json::from_value(serde_json::json!({
            "kv_bytes_per_token": 131_072
        }))
        .unwrap();
        assert_eq!(config.kv_transfer_bytes_per_token, Some(131_072));

        let encoded = serde_json::to_value(config).unwrap();
        assert_eq!(encoded["kv_transfer_bytes_per_token"], 131_072);
        assert!(encoded.get("kv_bytes_per_token").is_none());
    }

    #[test]
    fn transfer_geometry_rejects_duplicate_new_and_legacy_keys() {
        let error = serde_json::from_value::<EngineConfig>(serde_json::json!({
            "kv_transfer_bytes_per_token": 131_072,
            "kv_bytes_per_token": 65_536
        }))
        .unwrap_err();
        assert!(error.to_string().contains("duplicate field"));
    }

    #[test]
    fn deserialization_still_rejects_unknown_fields() {
        let error = serde_json::from_value::<EngineConfig>(serde_json::json!({
            "backend": "vllm",
            "unknown": true
        }))
        .unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn native_host_offload_deserializes_with_default_bandwidths() {
        assert_eq!(DEFAULT_HOST_OFFLOAD_BANDWIDTH_GBPS, 32.0);
        assert_eq!(
            NativeHostOffloadConfig::new(1),
            NativeHostOffloadConfig {
                num_host_blocks: 1,
                d2h_bandwidth_gbps: 32.0,
                h2d_bandwidth_gbps: 32.0,
            }
        );
        let config: EngineConfig = serde_json::from_value(serde_json::json!({
            "backend": "vllm",
            "block_size": 16,
            "kv_cache_bytes_per_token": 131_072,
            "native_host_offload": {
                "num_host_blocks": 4_096
            }
        }))
        .unwrap();

        assert_eq!(
            config.native_host_offload,
            Some(NativeHostOffloadConfig {
                num_host_blocks: 4_096,
                d2h_bandwidth_gbps: DEFAULT_HOST_OFFLOAD_BANDWIDTH_GBPS,
                h2d_bandwidth_gbps: DEFAULT_HOST_OFFLOAD_BANDWIDTH_GBPS,
            })
        );
        config.validate().unwrap();

        let decoded: EngineConfig =
            serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
        assert_eq!(decoded, config);
    }

    #[test]
    fn native_host_offload_rejects_missing_or_unknown_fields() {
        let missing_capacity = serde_json::from_value::<EngineConfig>(serde_json::json!({
            "native_host_offload": {}
        }))
        .unwrap_err();
        assert!(missing_capacity.to_string().contains("num_host_blocks"));

        let unknown = serde_json::from_value::<EngineConfig>(serde_json::json!({
            "native_host_offload": {
                "num_host_blocks": 4_096,
                "policy": "custom"
            }
        }))
        .unwrap_err();
        assert!(unknown.to_string().contains("unknown field"));
    }

    #[test]
    fn native_host_offload_validates_physical_controls() {
        let cases: &[InvalidHostConfigCase] = &[
            (
                |config| {
                    config.native_host_offload.as_mut().unwrap().num_host_blocks = 0;
                },
                "num_host_blocks",
            ),
            (
                |config| {
                    config
                        .native_host_offload
                        .as_mut()
                        .unwrap()
                        .d2h_bandwidth_gbps = f64::NAN;
                },
                "d2h_bandwidth_gbps",
            ),
            (
                |config| {
                    config
                        .native_host_offload
                        .as_mut()
                        .unwrap()
                        .h2d_bandwidth_gbps = -1.0;
                },
                "h2d_bandwidth_gbps",
            ),
            (
                |config| {
                    config.block_size = usize::MAX;
                    config.kv_cache_bytes_per_token = Some(2);
                },
                "positive, representable block size",
            ),
            (
                |config| config.kv_cache_bytes_per_token = None,
                "requires kv_cache_bytes_per_token",
            ),
            (
                |config| {
                    config.native_host_offload.as_mut().unwrap().num_host_blocks = usize::MAX;
                },
                "capacity in bytes overflowed",
            ),
            (
                |config| {
                    config
                        .native_host_offload
                        .as_mut()
                        .unwrap()
                        .d2h_bandwidth_gbps = f64::MIN_POSITIVE;
                },
                "unrepresentable transfer duration",
            ),
        ];
        for &(mutate, expected) in cases {
            assert_invalid_host_config(mutate, expected);
        }
    }

    #[test]
    fn native_host_offload_rejects_unsupported_scheduler_modes() {
        let cases: &[InvalidHostConfigCase] = &[
            (|config| config.backend = Backend::Sglang, "backend=vllm"),
            (
                |config| config.worker_type = WorkerType::Prefill,
                "worker_type=aggregated",
            ),
            (
                |config| config.enable_prefix_caching = false,
                "enable_prefix_caching=true",
            ),
            (
                |config| config.aic_nextn = Some(1),
                "does not support aic_nextn",
            ),
        ];
        for &(mutate, expected) in cases {
            assert_invalid_host_config(mutate, expected);
        }
    }

    #[test]
    fn serialization_round_trip_preserves_runtime_neutral_controls() {
        let config = EngineConfig {
            backend: Backend::Sglang,
            block_size: 8,
            num_gpu_blocks: 123,
            max_num_seqs: 7,
            max_num_batched_tokens: 456,
            prefill_decode_interval: 4,
            worker_type: WorkerType::Decode,
            preemption_mode: PreemptionMode::Fifo,
            emit_kv_events: true,
            emit_kv_token_ids: true,
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 2.5,
                decode_ms: 0.75,
            },
            ..EngineConfig::for_backend(Backend::Sglang)
        };
        let encoded = serde_json::to_value(&config).unwrap();
        let decoded: EngineConfig = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, config);
    }

    #[test]
    fn scheduler_intervals_default_and_validate_for_each_backend() {
        for backend in ["vllm", "sglang", "trtllm"] {
            let mut config: EngineConfig =
                serde_json::from_value(serde_json::json!({ "backend": backend })).unwrap();
            assert_eq!(config.prefill_schedule_interval, 1);
            assert_eq!(config.prefill_decode_interval, 0);
            config.validate().unwrap();

            config.prefill_decode_interval = 20;
            if backend == "sglang" {
                config.validate().unwrap();
                let decoded: EngineConfig =
                    serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
                assert_eq!(decoded.prefill_decode_interval, 20);
            } else {
                assert!(
                    config
                        .validate()
                        .unwrap_err()
                        .to_string()
                        .contains("prefill_decode_interval is supported only for backend=sglang")
                );
            }

            config.prefill_decode_interval = 0;
            config.prefill_schedule_interval = 4;
            if backend == "vllm" {
                config.validate().unwrap();
            } else {
                assert!(
                    config
                        .validate()
                        .unwrap_err()
                        .to_string()
                        .contains("prefill_schedule_interval is supported only for backend=vllm")
                );
            }
        }
    }

    #[test]
    fn deserialization_rejects_negative_prefill_decode_interval() {
        let error = serde_json::from_value::<EngineConfig>(serde_json::json!({
            "backend": "sglang",
            "prefill_decode_interval": -1
        }))
        .unwrap_err();
        assert!(error.to_string().contains("expected usize"));
    }

    #[test]
    fn validation_rejects_zero_or_backend_invalid_capacity_fields() {
        let config = EngineConfig {
            num_gpu_blocks: 0,
            ..EngineConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("num_gpu_blocks")
        );

        let config = EngineConfig {
            block_size: 1,
            ..EngineConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("at least two")
        );

        let config = EngineConfig {
            max_model_len: Some(0),
            ..EngineConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("max_model_len")
        );

        let config = EngineConfig {
            prefill_schedule_interval: 0,
            ..EngineConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("prefill_schedule_interval")
        );
    }

    #[test]
    fn validation_accepts_sglang_page_size_one_and_rejects_invalid_controls() {
        let mut config = EngineConfig::for_backend(Backend::Sglang);
        config.validate().unwrap();

        config.sglang.chunked_prefill_size = 0;
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("chunked_prefill_size")
        );

        let mut config = EngineConfig::for_backend(Backend::Sglang);
        config.sglang.schedule_conservativeness = f64::NAN;
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("schedule_conservativeness")
        );
    }

    #[test]
    fn sglang_supports_prefix_caching_and_token_id_controls() {
        for (enable_prefix_caching, emit_kv_token_ids) in
            [(false, false), (false, true), (true, false), (true, true)]
        {
            let config = EngineConfig {
                enable_prefix_caching,
                emit_kv_events: true,
                emit_kv_token_ids,
                ..EngineConfig::for_backend(Backend::Sglang)
            };
            config.validate().unwrap();
            crate::engine::EngineFactory::new(config).unwrap();
        }
    }

    #[test]
    fn sglang_rejects_disabled_chunked_prefill_at_validation_and_factory_boundaries() {
        let config = EngineConfig {
            enable_chunked_prefill: false,
            ..EngineConfig::for_backend(Backend::Sglang)
        };
        let field = "enable_chunked_prefill";
        assert!(config.validate().unwrap_err().to_string().contains(field));
        let error = match crate::engine::EngineFactory::new(config) {
            Ok(_) => panic!("expected EngineFactory to reject {field}"),
            Err(error) => error,
        };
        assert!(error.to_string().contains(field));
    }

    #[test]
    fn max_model_len_is_supported_for_every_backend() {
        for backend in [Backend::Vllm, Backend::Sglang, Backend::Trtllm] {
            let mut config = EngineConfig::for_backend(backend);
            config.max_model_len = Some(128);
            config.validate().unwrap();
            crate::engine::EngineFactory::new(config.clone()).unwrap();
            config.max_model_len = Some(0);
            assert!(
                config
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("max_model_len")
            );
        }
    }

    #[test]
    fn mtp_configuration_validates_rates_and_decode_scaling() {
        let mut config = EngineConfig {
            aic_nextn: Some(2),
            aic_nextn_accept_rates: Some("0.8,0.5".to_string()),
            ..EngineConfig::default()
        };
        config.validate().unwrap();

        config.aic_nextn_accept_rates = Some("1.2".to_string());
        assert!(config.validate().is_err());

        config.aic_nextn_accept_rates = Some("0.8,0.5".to_string());
        config.decode_speedup_ratio = 2.0;
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("decode_speedup_ratio=1.0")
        );
    }

    #[test]
    fn mtp_rates_require_mtp_to_be_enabled() {
        let config = EngineConfig {
            aic_nextn_accept_rates: Some("0.5".to_string()),
            ..EngineConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("requires aic_nextn")
        );
    }

    #[test]
    fn kv_token_ids_require_kv_event_emission() {
        let config = EngineConfig {
            emit_kv_token_ids: true,
            emit_kv_events: false,
            ..EngineConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("emit_kv_token_ids")
        );
    }

    #[test]
    fn timing_provider_descriptors_are_validated_without_loading_them() {
        let config = EngineConfig {
            timing_model: TimingModelConfig::External {
                provider: " ".to_string(),
                config: serde_json::Value::Null,
            },
            ..EngineConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("provider cannot be empty")
        );

        let config = EngineConfig {
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: f64::NAN,
                decode_ms: 1.0,
            },
            ..EngineConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
