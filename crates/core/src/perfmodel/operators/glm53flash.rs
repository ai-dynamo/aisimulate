// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Independently expressed GLM-5.3-Flash analytical operator contracts.
//!
//! Geometry: Z.AI GLM-5.3-Flash config.json at eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
//! (MIT, Copyright (c) 2026 Z.AI Co., Ltd). Execution boundaries: vllm-project/vllm
//! ced6857afa0ea7b2e3f0846a62e1394e90f15607, vllm/models/glm5next/nvidia/{model,attention,kda}.py,
//! and sgl-project/sglang 94602c9c2b7cbdb8efd5c52802dac6a1c180089e,
//! python/sglang/srt/models/glm5_next.py (Apache-2.0). See THIRD_PARTY_NOTICES.md.
//! These are payload/roofline bounds, not allocator or kernel-launch predictions.

use crate::common::enums::{DatabaseMode, GemmQuantMode, KvCacheQuantMode, MoeQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::{SystemSpec, quant_tc_flops};
use crate::operators::base::{PerformanceResult, SolComponents, Source};
use crate::operators::op::{Op, RuntimeContext};
use crate::perf_database::PerfDatabase;
use serde::{Deserialize, Serialize};

fn zero() -> PerformanceResult {
    PerformanceResult::sol(SolComponents::new(0.0, 0.0))
}
fn leaf(spec: &SystemSpec, flops: f64, bytes: f64, rate: f64) -> PerformanceResult {
    PerformanceResult::sol(SolComponents::new(
        flops / rate * 1e3,
        bytes / spec.gpu.mem_bw * 1e3,
    ))
}
fn scalar_rate(spec: &SystemSpec) -> Result<f64, AicError> {
    spec.gpu
        .fp32_flops
        .filter(|x| x.is_finite() && *x > 0.0)
        .ok_or_else(|| {
            AicError::MissingSystemFlops("GLM-5.3-Flash scalar kernels require fp32_flops".into())
        })
}
fn identity(backend: &str, checkpoint: &str) -> Result<(), AicError> {
    if !matches!(backend, "vllm" | "sglang") || !matches!(checkpoint, "fp8" | "nvfp4") {
        return Err(AicError::ModelConfig(
            "GLM-5.3-Flash requires vllm/sglang and fp8/nvfp4 checkpoint identity".into(),
        ));
    }
    Ok(())
}
// Missing measured coverage remains an error in both SILICON and HYBRID.
fn measured<T: Serialize>(
    db: &PerfDatabase,
    component: &str,
    op: &T,
    batch: u32,
    prefix: u32,
    x: u32,
) -> Result<PerformanceResult, AicError> {
    if db.database_mode == DatabaseMode::Empirical {
        return Err(AicError::EmpiricalNotImplemented(format!(
            "GLM-5.3-Flash {component} has no empirical anchor"
        )));
    }
    let shape = serde_json::to_value(op).map_err(|e| AicError::InvalidPerfData(e.to_string()))?;
    if (shape.get("is_context") == Some(&serde_json::Value::Bool(false))
        && db.glm53flash_graph.has_measurements()?)
        || db.glm53flash_graph.requires_serving_context(&shape)?
    {
        return Err(AicError::InvalidPerfData("native graph data requires the Op RuntimeContext query; token-only/direct legacy lookup is unsupported".into()));
    }
    if batch == 0 || x == 0 {
        return Ok(PerformanceResult::with_energy(0.0, 0.0, Source::Silicon));
    }
    match db.glm53flash.query(component, op, batch, prefix, x)? {
        Some(value) => Ok(PerformanceResult::with_energy(
            value.latency,
            value.energy,
            Source::Silicon,
        )),
        None => Err(AicError::PerfDatabase(format!(
            "GLM-5.3-Flash {component} has no exact measured data for geometry, batch={batch}, prefix={prefix}, x={x}"
        ))),
    }
}
fn weight_size(q: GemmQuantMode) -> f64 {
    q.mapping().memory
        + if q == GemmQuantMode::Fp8Block {
            4.0 / (128.0 * 128.0)
        } else {
            0.0
        }
}
fn gemm(
    spec: &SystemSpec,
    x: f64,
    n: f64,
    k: f64,
    q: GemmQuantMode,
) -> Result<PerformanceResult, AicError> {
    Ok(leaf(
        spec,
        2.0 * x * n * k,
        n * k * weight_size(q) + 2.0 * x * (n + k),
        quant_tc_flops(spec, q.mapping())?,
    ))
}

/// Prefix sum over integer query positions plus the next position's fractional
/// weight. The caller applies its native short-context score-skip predicate.
/// Selected attention is dense up to topk, then uses completed pools and tail.
fn sparse_pairs(start: f64, end: f64, pool: u32, topk: u32) -> (f64, f64) {
    let (r, k) = (pool as f64, topk as f64);
    let floor_sum = |n: f64| {
        let cycles = (n / r).floor();
        let tail = n - cycles * r;
        r * cycles * (cycles - 1.0) / 2.0 + cycles * (tail + 1.0)
    };
    let cumulative = |v: f64| {
        let n = v.max(0.0).floor();
        let frac = v.max(0.0) - n;
        let pools = ((n + 1.0) / r).floor();
        let score = floor_sum(n) + frac * pools;
        let capped = floor_sum(n.min(k)) + (n - k).max(0.0) * (k / r);
        let cycles = (n / r).floor();
        let tail = n - cycles * r;
        let selected = r * capped
            + cycles * r * (r - 1.0) / 2.0
            + tail * (tail + 1.0) / 2.0
            + frac * (r * pools.min(k / r) + (n + 1.0) % r);
        (score, selected)
    };
    let (a, b) = cumulative(start);
    let (c, d) = cumulative(end);
    (c - a, d - b)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Glm53AttentionOp {
    pub name: String,
    pub is_context: bool,
    pub layer_kind: String,
    pub backend: String,
    pub checkpoint_format: String,
    pub hidden_size: u32,
    pub tp_size: u32,
    /// Local heads after tensor parallel sharding. Indexer heads are replicated.
    pub num_heads: u32,
    pub head_dim: u32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub value_head_dim: u32,
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub index_pool: u32,
    pub conv_kernel: u32,
    pub gate_lower_bound: f64,
    pub projection_quant_mode: GemmQuantMode,
    pub kv_cache_dtype: KvCacheQuantMode,
}
impl Glm53AttentionOp {
    pub fn validate(&self) -> Result<(), AicError> {
        identity(&self.backend, &self.checkpoint_format)?;
        if !matches!(self.layer_kind.as_str(), "kda" | "sparse_mla")
            || !matches!(self.tp_size, 1 | 2 | 4)
            || self.num_heads == 0
            || self.head_dim == 0
            || self.hidden_size == 0
        {
            return Err(AicError::ModelConfig(
                "invalid GLM-5.3-Flash attention geometry/topology".into(),
            ));
        }
        if self.layer_kind == "kda"
            && (self.conv_kernel != 4
                || self.gate_lower_bound != -5.0
                || self.projection_quant_mode != GemmQuantMode::Bfloat16)
        {
            return Err(AicError::ModelConfig(
                "GLM-5.3-Flash KDA requires BF16 projections, conv4, bounded gate -5".into(),
            ));
        }
        if self.layer_kind == "sparse_mla"
            && (self.index_pool != 4
                || self.index_topk != 2048
                || self.kv_lora_rank == 0
                || self.index_head_dim != 128
                || self.kv_cache_dtype != KvCacheQuantMode::Fp8)
        {
            return Err(AicError::ModelConfig(
                "GLM-5.3-Flash sparse MLA requires FP8 latent cache and IndexPool4/topk2048".into(),
            ));
        }
        Ok(())
    }
    /// One active sequence's persistent payload; no allocator pages, prefix
    /// snapshots or speculative states. Cache precision is independent of weights.
    pub fn cache_bytes(&self, seq_len: u32) -> Result<f64, AicError> {
        self.validate()?;
        let (n, d) = (self.num_heads as f64, self.head_dim as f64);
        if self.layer_kind == "kda" {
            // FP32 D-by-D recurrence; BF16 q/k/v histories of conv_width-1.
            Ok(n * d * d * 4.0 + 3.0 * n * d * f64::from(self.conv_kernel - 1) * 2.0)
        } else {
            // Replicated FP8 latent KV, FP8 index keys with FP32 block scale,
            // and two BF16 pool-tail buffers (raw key and compression score).
            let pooled = seq_len / self.index_pool;
            Ok(f64::from(seq_len) * f64::from(self.kv_lora_rank)
                + f64::from(pooled) * f64::from(self.index_head_dim + 4)
                + f64::from(self.index_pool * self.index_head_dim) * 4.0)
        }
    }
    pub fn weight_bytes(&self) -> f64 {
        let (h, n, d) = (
            self.hidden_size as f64,
            self.num_heads as f64,
            self.head_dim as f64,
        );
        if self.layer_kind == "kda" {
            let p = n * d;
            // qkv,b,f_a,g_a, f_b,g_b and o; FP32 conv/gates; RMS norm.
            2.0 * (h * (3.0 * p + n + 2.0 * d) + 2.0 * d * p + p * h + d)
                + 4.0 * (3.0 * p * self.conv_kernel as f64 + n + p)
        } else {
            let (q, k, v, i, j) = (
                self.q_lora_rank as f64,
                self.kv_lora_rank as f64,
                self.value_head_dim as f64,
                self.index_n_heads as f64,
                self.index_head_dim as f64,
            );
            (h * (q + k) + q * n * d + n * v * h) * weight_size(self.projection_quant_mode)
                + 2.0
                    * (k * n * (d + v)
                        + q * i * j
                        + h * (2.0 * j + if self.backend == "vllm" { i } else { 0.0 })
                        + q
                        + k
                        + 2.0 * j)
                + 4.0 * (h * i + self.index_pool as f64 * j)
                + if self.backend == "sglang" {
                    4.0 * j
                } else {
                    0.0
                }
        }
    }
    /// Same f64 oracle is used by op queries and FPM interpolation. Sparse
    /// pairs preserve causal prefixes and pool publication boundaries.
    pub fn sol(
        &self,
        spec: &SystemSpec,
        batch: f64,
        s: f64,
        prefix: f64,
    ) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        if batch <= 0.0 || s <= 0.0 {
            return Ok(zero());
        }
        let (h, n, d) = (
            self.hidden_size as f64,
            self.num_heads as f64,
            self.head_dim as f64,
        );
        let x = if self.is_context { batch * s } else { batch };
        let fp32 = scalar_rate(spec)?;
        let bf16 = quant_tc_flops(spec, GemmQuantMode::Bfloat16.mapping())?;
        let mut result = zero();
        if self.layer_kind == "kda" {
            let p = n * d;
            let mut linear = |out: f64, input: f64| -> Result<(), AicError> {
                result = result
                    .clone()
                    .plus(gemm(spec, x, out, input, GemmQuantMode::Bfloat16)?);
                Ok(())
            };
            if self.backend == "vllm" {
                linear(3.0 * p + n + 2.0 * d, h)?;
            } else {
                // SGLang retains quant_config even for excluded KDA linears;
                // do_fuse_qkvbfg is false in the pinned quantized models.
                for out in [p, p, p, n, d, d] {
                    linear(out, h)?;
                }
            }
            linear(p, d)?;
            linear(p, d)?;
            result = result.plus(leaf(
                spec,
                6.0 * x * p * self.conv_kernel as f64,
                12.0 * x * p + 12.0 * p * self.conv_kernel as f64,
                fp32,
            ));
            // Delta recurrence: K*S, rank-one update and Q*S, plus decay.
            // A chunked implementation can use tensor cores during prefill;
            // this is the mathematical lower bound, excluding launch/padding.
            let rate = if self.is_context { bf16 } else { fp32 };
            let state = n * d * d * 4.0;
            result = result.plus(leaf(
                spec,
                7.0 * x * n * d * d,
                2.0 * batch * state + 12.0 * x * p,
                rate,
            ));
            // Bounded gate, Q/K normalization, beta and gated output RMSNorm.
            result = result.plus(leaf(
                spec,
                x * (28.0 * p + 5.0 * n),
                x * (16.0 * p + 4.0 * n),
                fp32,
            ));
            result = result.plus(gemm(spec, x, h, p, GemmQuantMode::Bfloat16)?);
        } else {
            let (q, k, v, i, j) = (
                self.q_lora_rank as f64,
                self.kv_lora_rank as f64,
                self.value_head_dim as f64,
                self.index_n_heads as f64,
                self.index_head_dim as f64,
            );
            let quant = self.projection_quant_mode;
            result =
                result
                    .plus(gemm(spec, x, q + k, h, quant)?)
                    .plus(gemm(spec, x, n * d, q, quant)?);
            // Absorbed MLA: Q*W_UK and latent output*W_UV, not a full KV
            // expansion in addition to the two absorbed batched matmuls.
            result = result.plus(leaf(
                spec,
                2.0 * x * n * k * (d + v),
                2.0 * n * k * (d + v) + 2.0 * x * n * (d + v + 2.0 * k),
                bf16,
            ));
            result = result.plus(gemm(spec, x, h, n * v, quant)?);
            let end = if self.is_context { prefix + s } else { s };
            let short_prefill = self.is_context && end <= self.index_topk as f64;
            let skip_query = self.backend == "sglang" && short_prefill;
            if !skip_query {
                result = result.plus(gemm(spec, x, i * j, q, GemmQuantMode::Bfloat16)?);
                result = result.plus(leaf(
                    spec,
                    2.0 * x * h * i,
                    4.0 * h * i + 4.0 * x * (h + i),
                    fp32,
                ));
            }
            let key_width = j + if self.backend == "vllm" { i } else { 0.0 };
            result = result.plus(gemm(spec, x, key_width, h, GemmQuantMode::Bfloat16)?);
            result = result.plus(gemm(spec, x, j, h, GemmQuantMode::Bfloat16)?);
            let start = if self.is_context {
                prefix
            } else {
                (s - 1.0).max(0.0)
            };
            let (pooled_pairs, selected_pairs) =
                sparse_pairs(start, end, self.index_pool, self.index_topk);
            let skip_scores = short_prefill
                || (self.backend == "vllm" && !self.is_context && end <= self.index_topk as f64);
            let (pooled_pairs, selected_pairs) = (
                if skip_scores {
                    0.0
                } else {
                    batch * pooled_pairs
                },
                batch * selected_pairs,
            );
            let fp8 = quant_tc_flops(spec, GemmQuantMode::Fp8.mapping())?;
            let query_width = if skip_query { 0.0 } else { i * j };
            result = result.plus(leaf(
                spec,
                2.0 * pooled_pairs * i * j,
                pooled_pairs * (j + 4.0) + 2.0 * x * query_width,
                fp8,
            ));
            // Top-k, pool expansion and NoPE latent attention, KV publication.
            result = result.plus(leaf(
                spec,
                pooled_pairs * 2.0,
                pooled_pairs * 4.0 + selected_pairs * 4.0,
                fp32,
            ));
            result = result.plus(leaf(
                spec,
                4.0 * selected_pairs * n * k,
                selected_pairs * k + 4.0 * x * n * k,
                bf16,
            ));
            result = result.plus(leaf(
                spec,
                x * (5.0 * (q + k) + 12.0 * j + 7.0 * query_width),
                x * (4.0 * (q + k) + 12.0 * j + 4.0 * query_width + k),
                fp32,
            ));
        }
        Ok(result)
    }
    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        if !matches!(db.database_mode, DatabaseMode::Sol | DatabaseMode::SolFull) {
            return measured(
                db,
                "attention",
                self,
                ctx.batch_size,
                if self.is_context { ctx.prefix } else { 0 },
                ctx.s,
            );
        }
        self.sol(
            &db.system_spec,
            ctx.batch_size as f64,
            ctx.s as f64,
            ctx.prefix as f64,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Glm53MhcOp {
    pub name: String,
    pub role: String,
    pub backend: String,
    pub checkpoint_format: String,
    /// Native invocation identity, even though the local SOL work is replicated.
    pub tp_size: u32,
    pub is_context: bool,
    pub hidden_size: u32,
    pub hc_mult: u32,
    pub sinkhorn_iters: u32,
}
impl Glm53MhcOp {
    pub fn validate(&self) -> Result<(), AicError> {
        identity(&self.backend, &self.checkpoint_format)?;
        if self.hidden_size == 0
            || self.hc_mult != 4
            || self.sinkhorn_iters != 20
            || !matches!(self.tp_size, 1 | 2 | 4)
        {
            return Err(AicError::ModelConfig(
                "GLM mHC requires positive hidden size, multiplier4 and20 Sinkhorn iterations"
                    .into(),
            ));
        }
        if !matches!(
            self.role.as_str(),
            "pre" | "post" | "fused_post_pre" | "expand" | "contract"
        ) || (self.backend == "sglang" && self.role == "fused_post_pre")
        {
            return Err(AicError::ModelConfig(
                "unsupported GLM mHC backend role".into(),
            ));
        }
        Ok(())
    }
    pub fn weight_bytes(&self) -> f64 {
        let (h, c) = (self.hidden_size as f64, self.hc_mult as f64);
        if matches!(self.role.as_str(), "pre" | "fused_post_pre") {
            4.0 * ((c + 2.0) * c * (c * h + 1.0) + 3.0) + 2.0 * h
        } else {
            0.0
        }
    }
    pub fn sol(&self, spec: &SystemSpec, x: f64) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        let (h, c) = (self.hidden_size as f64, self.hc_mult as f64);

        if x <= 0.0 {
            return Ok(zero());
        }
        let mixes = (c + 2.0) * c;
        let pre = 2.0 * c * h * mixes
            + (c * c + 2.0 * c) * self.sinkhorn_iters as f64
            + 2.0 * c * h
            + 5.0 * h;
        let post = 2.0 * c * c * h + 2.0 * c * h;
        let (flops, bytes) = match self.role.as_str() {
            "pre" => (pre, (c + 1.0) * h * 2.0 + mixes * 4.0),
            "post" => (post, (2.0 * c + 1.0) * h * 2.0 + mixes * 4.0),
            "fused_post_pre" => (pre + post, (2.0 * c + 2.0) * h * 2.0 + 2.0 * mixes * 4.0),
            "expand" => (0.0, (c + 1.0) * h * 2.0),
            "contract" => ((c - 1.0) * h, (c + 1.0) * h * 2.0),
            _ => return Err(AicError::ModelConfig("unknown GLM mHC role".into())),
        };
        Ok(leaf(
            spec,
            x * flops,
            self.weight_bytes() + x * bytes,
            scalar_rate(spec)?,
        ))
    }
    pub fn query(&self, db: &PerfDatabase, tokens: u32) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        if matches!(db.database_mode, DatabaseMode::Sol | DatabaseMode::SolFull) {
            self.sol(&db.system_spec, tokens as f64)
        } else {
            measured(db, "mhc", self, 1, 0, tokens)
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Glm53RouterOp {
    pub name: String,
    pub backend: String,
    pub checkpoint_format: String,
    pub hidden_size: u32,
    pub num_experts: u32,
    pub topk: u32,
}
impl Glm53RouterOp {
    pub fn validate(&self) -> Result<(), AicError> {
        identity(&self.backend, &self.checkpoint_format)?;
        if self.hidden_size == 0 || self.topk == 0 || self.topk > self.num_experts {
            return Err(AicError::ModelConfig(
                "invalid GLM router geometry/topk".into(),
            ));
        }
        Ok(())
    }
    pub fn weight_bytes(&self) -> f64 {
        4.0 * f64::from(self.num_experts) * f64::from(self.hidden_size)
    }
    pub fn sol(&self, spec: &SystemSpec, x: f64) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        if x <= 0.0 {
            return Ok(zero());
        }
        let (h, e) = (self.hidden_size as f64, self.num_experts as f64);
        // FP32 GateLinear only. Sigmoid/bias/top-k belong to the native
        // MoE boundary; they must not also be timed here.
        Ok(leaf(
            spec,
            2.0 * x * h * e,
            self.weight_bytes() + 4.0 * x * (h + e),
            scalar_rate(spec)?,
        ))
    }
    pub fn query(&self, db: &PerfDatabase, tokens: u32) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        if matches!(db.database_mode, DatabaseMode::Sol | DatabaseMode::SolFull) {
            self.sol(&db.system_spec, tokens as f64)
        } else {
            measured(db, "router", self, 1, 0, tokens)
        }
    }
}

/// Native local FFN boundary, including gate/router, routed and shared experts,
/// clamp and activation; excluding the final separately modeled collective.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Glm53FfnOp {
    pub name: String,
    pub backend: String,
    pub checkpoint_format: String,
    pub is_context: bool,
    pub is_dense: bool,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_experts: u32,
    pub topk: u32,
    pub tp_size: u32,
    pub n_shared_experts: u32,
    pub swiglu_limit: f64,
    pub scoring_func: String,
    pub routed_scaling_factor: f64,
    pub n_group: u32,
    pub topk_group: u32,
    pub norm_topk_prob: bool,
    pub gemm_quant_mode: GemmQuantMode,
    pub shared_quant_mode: GemmQuantMode,
    pub moe_quant_mode: MoeQuantMode,
    /// Analytical composition only; measured identity excludes this list and
    /// uses the explicit physical fields above. Child display names vary by layer.
    #[serde(default)]
    pub children: Vec<Op>,
}
impl Glm53FfnOp {
    /// Validate the measured physical identity without analytical children.
    /// Row admission uses this method because children are excluded from keys.
    pub fn validate_physical(&self) -> Result<(), AicError> {
        identity(&self.backend, &self.checkpoint_format)?;
        if !matches!(self.tp_size, 1 | 2 | 4)
            || self.hidden_size != 4096
            || self.swiglu_limit != 10.0
            || self.scoring_func != "sigmoid"
            || self.routed_scaling_factor != 2.5
            || self.n_group != 1
            || self.topk_group != 1
            || !self.norm_topk_prob
            || self.n_shared_experts != 1
            || self.num_experts != 288
            || self.topk != 8
            || self.intermediate_size != if self.is_dense { 12288 } else { 2048 }
        {
            return Err(AicError::ModelConfig(
                "GLM-5.3-Flash FFN requires the native sigmoid/top8/clamp10 pure-TP contract"
                    .into(),
            ));
        }
        let (gemm_quant, shared_quant, moe_quant) = if self.checkpoint_format == "fp8" {
            (
                GemmQuantMode::Fp8Block,
                GemmQuantMode::Fp8Block,
                MoeQuantMode::Fp8Block,
            )
        } else {
            (
                GemmQuantMode::Nvfp4,
                GemmQuantMode::Bfloat16,
                MoeQuantMode::Nvfp4,
            )
        };
        if self.gemm_quant_mode != gemm_quant
            || self.shared_quant_mode != shared_quant
            || self.moe_quant_mode != moe_quant
        {
            return Err(AicError::ModelConfig(
                "GLM FFN checkpoint precision partition disagrees with its geometry".into(),
            ));
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<(), AicError> {
        self.validate_physical()?;
        if self.children.len() != if self.is_dense { 3 } else { 5 }
            || self.children.iter().any(|op| {
                !matches!(
                    op,
                    Op::Gemm(_) | Op::Elementwise(_) | Op::Moe(_) | Op::Glm53Router(_)
                )
            })
        {
            return Err(AicError::ModelConfig(
                "GLM FFN requires its complete analytical composition".into(),
            ));
        }
        let width = self.intermediate_size / self.tp_size;
        let quant = if self.is_dense {
            self.gemm_quant_mode
        } else {
            self.shared_quant_mode
        };
        let gemm_matches = |op: &Op, n: u32, k: u32| {
            matches!(op,Op::Gemm(g)
            if g.n==n && g.k==k && g.quant_mode==quant && g.scale_factor==1.0
            && g.scale_num_tokens==1 && g.seq_split==1)
        };
        let valid = gemm_matches(&self.children[0], 2 * width, self.hidden_size)
            && matches!(&self.children[1],Op::Elementwise(e) if e.scale_factor==1.0
                && e.bytes_per_token==6.0*f64::from(width) && e.scale_num_tokens==1 && e.seq_split==1)
            && gemm_matches(&self.children[2], self.hidden_size, width)
            && (self.is_dense
                || (matches!(&self.children[3],Op::Glm53Router(r)
                if r.backend==self.backend && r.checkpoint_format==self.checkpoint_format
                && r.hidden_size==self.hidden_size && r.num_experts==self.num_experts && r.topk==self.topk)
                    && matches!(&self.children[4],Op::Moe(m) if m.hidden_size==self.hidden_size
                && m.inter_size==self.intermediate_size && m.topk==self.topk && m.num_experts==self.num_experts
                && m.moe_tp_size==self.tp_size && m.moe_ep_size==1 && m.attention_dp_size==1
                && m.quant_mode==self.moe_quant_mode && m.scale_factor==1.0 && m.is_gated)));
        if !valid {
            return Err(AicError::ModelConfig(
                "GLM FFN analytical children disagree with its measured geometry".into(),
            ));
        }
        Ok(())
    }
    pub fn weight_bytes(&self) -> f64 {
        self.children.iter().map(Op::weight_bytes).sum()
    }
    pub fn sol(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        // Generic tables are never formal GLM evidence. Even HYBRID fallback
        // evaluates analytical children, so a generic MoE row cannot leak in.
        let sol_db = db.sol_full_view();
        self.children.iter().try_fold(
            zero(),
            |sum, child| Ok(sum.plus(child.query(&sol_db, ctx)?)),
        )
    }
    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        if !matches!(db.database_mode, DatabaseMode::Sol | DatabaseMode::SolFull) {
            let tokens = if self.is_context {
                ctx.batch_size.checked_mul(ctx.s).ok_or_else(|| {
                    AicError::ModelConfig("GLM FFN token count exceeds u32 coordinates".into())
                })?
            } else {
                ctx.batch_size
            };
            return measured(db, "ffn", self, 1, 0, tokens);
        }
        self.sol(db, ctx)
    }
}

/// Strict native boundary for remaining text-graph operations. Analytical
/// children never supply measured evidence; their names are not physical keys.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Glm53PrimitiveOp {
    pub name: String,
    pub role: String,
    pub backend: String,
    pub checkpoint_format: String,
    pub tp_size: u32,
    pub is_context: bool,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub token_selection: String,
    pub output_dtype: String,
    pub collective: String,
    #[serde(default)]
    pub children: Vec<Op>,
}
impl Glm53PrimitiveOp {
    pub fn validate_physical(&self) -> Result<(), AicError> {
        identity(&self.backend, &self.checkpoint_format)?;
        if !matches!(self.tp_size, 1 | 2 | 4)
            || self.hidden_size != 4096
            || self.vocab_size != 154880
        {
            return Err(AicError::ModelConfig(
                "invalid GLM primitive geometry/topology".into(),
            ));
        }
        let (selection, output, collective) = match self.role.as_str() {
            "embedding" | "final_norm" => ("all_scheduled", "bfloat16", "none"),
            "allreduce" => ("all_scheduled", "bfloat16", "all_reduce"),
            "logits" => (
                "last_per_request",
                if self.backend == "sglang" {
                    "float32"
                } else {
                    "bfloat16"
                },
                "all_gather",
            ),
            _ => return Err(AicError::ModelConfig("unknown GLM primitive role".into())),
        };
        if self.token_selection != selection
            || self.output_dtype != output
            || self.collective != collective
        {
            return Err(AicError::ModelConfig(
                "GLM primitive native boundary disagrees with geometry".into(),
            ));
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<(), AicError> {
        self.validate_physical()?;
        let h = self.hidden_size;
        let v = self.vocab_size;
        let tp = self.tp_size;
        let norm = |op: &Op, bytes: f64| {
            matches!(op, Op::Elementwise(o) if o.bytes_per_token == bytes
            && o.scale_factor == 1.0 && o.scale_num_tokens == 1 && o.seq_split == 1)
        };
        let comm = |op: &Op, size: f64, operation: &str| {
            matches!(op, Op::Nccl(o)
            if o.hidden_size == size && o.num_gpus == tp && o.operation == operation
            && o.dtype == crate::common::enums::CommQuantMode::Half && o.scale_factor == 1.0 && o.seq_split == 1)
        };
        let valid = match self.role.as_str() {
            "embedding" => {
                self.children.len() == 1
                    && matches!(&self.children[0], Op::Embedding(o)
                if o.vocab_size == v / tp && o.hidden_size == h && o.quant_mode == GemmQuantMode::Bfloat16
                && o.scale_factor == 1.0 && o.seq_split == 1)
            }
            "final_norm" => self.children.len() == 1 && norm(&self.children[0], f64::from(h) * 4.0),
            "allreduce" => {
                self.children.len() == 1 && comm(&self.children[0], f64::from(h), "all_reduce")
            }
            "logits" => {
                self.children.len() == if self.backend == "sglang" { 3 } else { 2 }
                    && matches!(&self.children[0], Op::Gemm(o) if o.n == v / tp && o.k == h
                    && o.quant_mode == GemmQuantMode::Bfloat16 && o.scale_factor == 1.0
                    && o.scale_num_tokens == 1 && o.seq_split == 1)
                    && comm(&self.children[1], f64::from(v), "all_gather")
                    && (self.backend != "sglang" || norm(&self.children[2], f64::from(v) * 6.0))
            }
            _ => false,
        };
        if !valid {
            return Err(AicError::ModelConfig(
                "GLM primitive analytical children disagree with native boundary".into(),
            ));
        }
        Ok(())
    }
    pub fn weight_bytes(&self) -> f64 {
        self.children.iter().map(Op::weight_bytes).sum()
    }
    pub fn sol(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        let mut child_ctx = *ctx;
        if self.token_selection == "last_per_request" {
            child_ctx.num_tokens = ctx.batch_size;
        }
        let sol_db = db.sol_full_view();
        self.children.iter().try_fold(zero(), |sum, child| {
            Ok(sum.plus(child.query(&sol_db, &child_ctx)?))
        })
    }
    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        if !matches!(db.database_mode, DatabaseMode::Sol | DatabaseMode::SolFull) {
            let tokens = if self.token_selection == "last_per_request" || !self.is_context {
                ctx.batch_size
            } else {
                ctx.batch_size.checked_mul(ctx.s).ok_or_else(|| {
                    AicError::ModelConfig(
                        "GLM primitive token count exceeds u32 coordinates".into(),
                    )
                })?
            };
            return measured(db, "primitive", self, 1, 0, tokens);
        }
        self.sol(db, ctx)
    }
}

/// Explicit runtime bookkeeping, separate from the 277/366 physical model
/// boundaries. SOL/eager have no graph setup cost. A selected native graph
/// profile must answer this marker from measured setup nodes exactly once.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Glm53RuntimeOp {
    pub name: String,
    pub backend: String,
    pub checkpoint_format: String,
    pub tp_size: u32,
    pub is_context: bool,
}
impl Glm53RuntimeOp {
    pub fn validate(&self) -> Result<(), AicError> {
        identity(&self.backend, &self.checkpoint_format)?;
        if self.name != "native_graph_setup" || !matches!(self.tp_size, 1 | 2 | 4) {
            return Err(AicError::ModelConfig(
                "invalid GLM native runtime marker".into(),
            ));
        }
        Ok(())
    }
    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        if matches!(db.database_mode, DatabaseMode::Sol | DatabaseMode::SolFull) {
            return Ok(zero());
        }
        if db.database_mode == DatabaseMode::Empirical {
            return Err(AicError::EmpiricalNotImplemented(
                "GLM native runtime setup has no empirical anchor".into(),
            ));
        }
        if let Some(value) = db
            .glm53flash_graph
            .query(&Op::Glm53Runtime(self.clone()), ctx)?
        {
            return Ok(value);
        }
        Ok(PerformanceResult::with_energy(0.0, 0.0, Source::Silicon))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::common::enums::TransferPolicy;
    use std::path::PathBuf;

    pub(crate) fn attention(kind: &str) -> Glm53AttentionOp {
        Glm53AttentionOp {
            name: "attention_0".into(),
            is_context: true,
            layer_kind: kind.into(),
            backend: "vllm".into(),
            checkpoint_format: "nvfp4".into(),
            hidden_size: 4096,
            tp_size: 2,
            num_heads: 32,
            head_dim: if kind == "kda" { 128 } else { 256 },
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            value_head_dim: 256,
            index_n_heads: 32,
            index_head_dim: 128,
            index_topk: 2048,
            index_pool: 4,
            conv_kernel: 4,
            gate_lower_bound: -5.0,
            projection_quant_mode: GemmQuantMode::Bfloat16,
            kv_cache_dtype: KvCacheQuantMode::Fp8,
        }
    }
    fn db(mode: DatabaseMode) -> PerfDatabase {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/aisimulate/src/aisimulate_core/systems");
        PerfDatabase::load(&root, "gb300", "sglang", "0.5.14")
            .unwrap()
            .with_mode(mode, TransferPolicy::ALL)
    }
    #[test]
    fn physical_cache_payload_and_pool_publication() {
        let kda = attention("kda");
        // TP2:32 heads*128*128*4 +3 conv histories*32*128*3*2.
        assert_eq!(kda.cache_bytes(0).unwrap(), 2_170_880.0);
        assert_eq!(kda.cache_bytes(131072).unwrap(), 2_170_880.0);
        let sparse = attention("sparse_mla");
        // FP8 latent512; index132 per4 tokens;4*128*two BF16 tail buffers.
        assert_eq!(sparse.cache_bytes(0).unwrap(), 2048.0);
        assert_eq!(sparse.cache_bytes(3).unwrap(), 3584.0);
        assert_eq!(sparse.cache_bytes(4).unwrap(), 4228.0);
        assert_eq!(sparse.cache_bytes(131072).unwrap(), 71_436_288.0);
        let mut tp4 = kda.clone();
        tp4.tp_size = 4;
        tp4.num_heads = 16;
        assert_eq!(
            tp4.cache_bytes(1).unwrap(),
            kda.cache_bytes(1).unwrap() / 2.0
        );
        let mut sparse4 = sparse.clone();
        sparse4.tp_size = 4;
        sparse4.num_heads = 16;
        assert_eq!(
            sparse4.cache_bytes(4096).unwrap(),
            sparse.cache_bytes(4096).unwrap()
        );
    }
    #[test]
    fn pool_selection_is_token_topk_not_pool_topk() {
        // 2048 token topk means512 pools. Unfinished tail adds0..3 tokens.
        assert_eq!(sparse_pairs(4095.0, 4096.0, 4, 2048), (1024.0, 2048.0));
        assert_eq!(sparse_pairs(4096.0, 4097.0, 4, 2048), (1024.0, 2049.0));
        assert_eq!(sparse_pairs(4097.0, 4098.0, 4, 2048), (1024.0, 2050.0));
        assert_eq!(sparse_pairs(0.0, 4.0, 4, 2048), (1.0, 10.0));
        assert_eq!(sparse_pairs(4096.0, 4096.5, 4, 2048), (512.0, 1024.5));
    }
    #[test]
    fn fp32_gate_has_independent_hand_calculated_roofline() {
        let mut spec = db(DatabaseMode::Sol).system_spec.clone();
        spec.gpu.mem_bw = 1e6;
        spec.gpu.fp32_flops = Some(1e9);
        let op = Glm53RouterOp {
            name: "router_3".into(),
            backend: "vllm".into(),
            checkpoint_format: "fp8".into(),
            hidden_size: 8,
            num_experts: 4,
            topk: 2,
        };
        //3 tokens:2*3*8*4=192 FP32 FLOPs;128 weight+3*(8+4)*4=272 bytes.
        let result = op.sol(&spec, 3.0).unwrap();
        assert_eq!(result.sol.unwrap().math_ms, 0.000192);
        assert_eq!(result.sol.unwrap().mem_ms, 0.272);
        assert_eq!(result.latency_ms, 0.272);
    }
    #[test]
    fn logits_use_one_row_per_request_and_include_native_vocab_gather() {
        use crate::operators::{communication::NcclOp, elementwise::ElementwiseOp, gemm::GemmOp};
        let db = db(DatabaseMode::Sol);
        let mut op = Glm53PrimitiveOp {
            name: "logits".into(),
            role: "logits".into(),
            backend: "vllm".into(),
            checkpoint_format: "fp8".into(),
            tp_size: 2,
            is_context: true,
            hidden_size: 4096,
            vocab_size: 154880,
            token_selection: "last_per_request".into(),
            output_dtype: "bfloat16".into(),
            collective: "all_gather".into(),
            children: vec![
                Op::Gemm(GemmOp::new("head", 77440, 4096, GemmQuantMode::Bfloat16)),
                Op::Nccl(NcclOp::new("vocab", 1.0, 154880.0, 2, "all_gather")),
            ],
        };
        let short = RuntimeContext {
            batch_size: 2,
            num_tokens: 2,
            ..RuntimeContext::default()
        };
        let long = RuntimeContext {
            batch_size: 2,
            num_tokens: 8192,
            s: 4096,
            prefix: 65536,
            ..RuntimeContext::default()
        };
        let expected = op.sol(&db, &short).unwrap().latency_ms;
        assert_eq!(expected, op.sol(&db, &long).unwrap().latency_ms);
        // SGLang's post-gather BF16->FP32 cast reads2 and writes4 bytes/element.
        op.backend = "sglang".into();
        op.output_dtype = "float32".into();
        op.children
            .push(Op::Elementwise(ElementwiseOp::new("cast", 6.0 * 154880.0)));
        let cast_ms = 2.0 * 6.0 * 154880.0 / db.system_spec.gpu.mem_bw * 1000.0;
        assert!((op.sol(&db, &long).unwrap().latency_ms - expected - cast_ms).abs() < 1e-12);
        op.children.pop();
        assert!(op.validate().is_err());
    }
    #[test]
    fn attention_is_finite_and_silicon_does_not_borrow_kimi_kda() {
        for kind in ["kda", "sparse_mla"] {
            let op = attention(kind);
            let spec = db(DatabaseMode::Sol).system_spec.clone();
            for (s, prefix) in [(1.0, 0.0), (4096.0, 0.0), (1024.0, 130048.0)] {
                let result = op.sol(&spec, 2.0, s, prefix).unwrap();
                assert!(result.latency_ms > 0.0 && result.latency_ms.is_finite());
            }
        }
        assert!(matches!(
            measured(
                &db(DatabaseMode::Silicon),
                "attention",
                &attention("kda"),
                1,
                0,
                128
            ),
            // This fixture intentionally points GLM at historical SGLang0.5.14.
            // Native identity rejection precedes any possible foreign-table hit.
            Err(AicError::InvalidPerfData(_))
        ));
    }

    #[test]
    fn invalid_physical_geometry_fails_before_measured_lookup() {
        let db = db(DatabaseMode::Silicon);
        let mut attention = attention("kda");
        attention.conv_kernel = 3;
        assert!(matches!(
            attention.query(&db, &RuntimeContext::default()),
            Err(AicError::ModelConfig(_))
        ));
        let mhc = Glm53MhcOp {
            name: "bad_mhc".into(),
            tp_size: 2,
            is_context: true,
            role: "unknown".into(),
            backend: "vllm".into(),
            checkpoint_format: "fp8".into(),
            hidden_size: 4096,
            hc_mult: 4,
            sinkhorn_iters: 20,
        };
        assert!(matches!(mhc.query(&db, 1), Err(AicError::ModelConfig(_))));
        let router = Glm53RouterOp {
            name: "bad_router".into(),
            backend: "vllm".into(),
            checkpoint_format: "fp8".into(),
            hidden_size: 4096,
            num_experts: 288,
            topk: 289,
        };
        assert!(matches!(
            router.query(&db, 1),
            Err(AicError::ModelConfig(_))
        ));
    }

    #[test]
    fn ffn_physical_rows_need_no_children_but_queries_reject_token_overflow() {
        use crate::operators::{elementwise::ElementwiseOp, gemm::GemmOp};
        let mut op = Glm53FfnOp {
            name: "ffn_0".into(),
            backend: "vllm".into(),
            checkpoint_format: "fp8".into(),
            is_context: true,
            is_dense: true,
            hidden_size: 4096,
            intermediate_size: 12288,
            num_experts: 288,
            topk: 8,
            tp_size: 2,
            n_shared_experts: 1,
            swiglu_limit: 10.0,
            scoring_func: "sigmoid".into(),
            routed_scaling_factor: 2.5,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            gemm_quant_mode: GemmQuantMode::Fp8Block,
            shared_quant_mode: GemmQuantMode::Fp8Block,
            moe_quant_mode: MoeQuantMode::Fp8Block,
            children: vec![],
        };
        assert!(op.validate_physical().is_ok());
        assert!(op.validate().is_err());
        op.children = vec![
            Op::Gemm(GemmOp::new("up", 12288, 4096, GemmQuantMode::Fp8Block)),
            Op::Elementwise(ElementwiseOp::new("clamp", 6.0 * 6144.0)),
            Op::Gemm(GemmOp::new("down", 4096, 6144, GemmQuantMode::Fp8Block)),
        ];
        assert!(op.validate().is_ok());
        let ctx = RuntimeContext {
            batch_size: u32::MAX,
            s: 2,
            ..RuntimeContext::default()
        };
        let error = op.query(&db(DatabaseMode::Silicon), &ctx).unwrap_err();
        assert!(
            matches!(&error, AicError::ModelConfig(message) if message.contains("exceeds u32"))
        );
    }
}
