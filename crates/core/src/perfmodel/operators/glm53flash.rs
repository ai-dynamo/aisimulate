// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Independently expressed GLM-5.3-Flash analytical operator contracts.
//!
//! Geometry: Z.AI GLM-5.3-Flash config.json at eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
//! (MIT, Copyright (c) 2026 Z.AI Co., Ltd). Execution boundaries: vllm-project/vllm
//! db9527a46873454610df6dbedf79a36d6bf1a7f6 (v0.31.0), vllm/models/glm5next/common/{model,attention,kda}.py,
//! and sgl-project/sglang 94602c9c2b7cbdb8efd5c52802dac6a1c180089e,
//! python/sglang/srt/models/glm5_next.py (Apache-2.0). See THIRD_PARTY_NOTICES.md.
//! These are payload/roofline bounds, not allocator or kernel-launch predictions.

use crate::common::enums::{DatabaseMode, GemmQuantMode, KvCacheQuantMode, MoeQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::{SystemSpec, quant_tc_flops};
use crate::operators::base::{PerformanceResult, SolComponents, Source};
use crate::operators::moe::MoeOp;
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
// Without a measured composition explicit SILICON must fail; HYBRID reports
// Source::Sol rather than relabelling an analytical value.
fn analytical_only(db: &PerfDatabase, component: &str) -> Result<(), AicError> {
    match db.database_mode {
        DatabaseMode::Silicon => Err(AicError::PerfDatabase(format!(
            "GLM-5.3-Flash {component} has no measured SILICON data"
        ))),
        DatabaseMode::Empirical => Err(AicError::EmpiricalNotImplemented(format!(
            "GLM-5.3-Flash {component} has no empirical anchor"
        ))),
        _ => Ok(()),
    }
}
/// Generic families that price a GLM boundary in SILICON/HYBRID. Their SOL
/// view is never used here: SOL stays on the GLM formulas so op-level SOL and
/// the FPM roofline are unchanged by the measured composition.
fn allowed_measured(component: &str, op: &Op) -> bool {
    match component {
        "attention" => matches!(op, Op::Gemm(_) | Op::Kda(_) | Op::Elementwise(_)),
        "ffn" => matches!(op, Op::Gemm(_) | Op::Elementwise(_) | Op::Moe(_)),
        "primitive" => matches!(
            op,
            Op::Embedding(_)
                | Op::Elementwise(_)
                | Op::Gemm(_)
                | Op::CustomAllReduce(_)
                | Op::Nccl(_)
        ),
        _ => false,
    }
}

fn validate_measured(component: &str, measured: &[Op]) -> Result<(), AicError> {
    if let Some(op) = measured.iter().find(|op| !allowed_measured(component, op)) {
        return Err(AicError::ModelConfig(format!(
            "GLM-5.3-Flash {component} measured composition cannot contain {}",
            op.name()
        )));
    }
    Ok(())
}

/// True when a generic child answers from an analytical formula in every
/// database mode (no table exists for these families); such children stay
/// admissible in SILICON, exactly as for every other model.
fn analytic_family(op: &Op) -> bool {
    matches!(op, Op::Elementwise(_) | Op::Embedding(_))
}

/// SILICON/HYBRID price of a GLM boundary: the sum of its generic children
/// queried against the generic tables. SILICON fails closed when a child
/// cannot be answered from silicon data (including families such as KDA that
/// would otherwise return their SOL on a miss). HYBRID keeps each child's own
/// labelled fallback; a missing table falls back to the GLM SOL (`Source::Sol`).
fn measured_query(
    db: &PerfDatabase,
    ctx: &RuntimeContext,
    component: &str,
    measured: &[Op],
    sol: impl FnOnce() -> Result<PerformanceResult, AicError>,
) -> Result<PerformanceResult, AicError> {
    if db.database_mode == DatabaseMode::Empirical {
        analytical_only(db, component)?;
    }
    let hybrid = db.database_mode == DatabaseMode::Hybrid;
    if measured.is_empty() {
        analytical_only(db, component)?;
        return sol();
    }
    let mut total = zero();
    for child in measured {
        let result = match child.query(db, ctx) {
            Ok(result) => result,
            // A child with no table and no empirical anchor is a coverage gap
            // in HYBRID: report the whole boundary's GLM SOL (Source::Sol).
            Err(err)
                if hybrid
                    && (err.is_missing_perf_data()
                        || matches!(err, AicError::EmpiricalNotImplemented(_))) =>
            {
                return sol();
            }
            Err(err) => return Err(err),
        };
        if db.database_mode == DatabaseMode::Silicon
            && result.latency_ms > 0.0
            && !analytic_family(child)
            && result.source != Source::Silicon
        {
            return Err(AicError::PerfDatabase(format!(
                "GLM-5.3-Flash {component} child {} has no SILICON data (source {})",
                child.name(),
                result.source.as_str()
            )));
        }
        total = total.plus(result);
    }
    Ok(total)
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
    /// SILICON/HYBRID generic composition (KDA only). Sparse MLA is priced by
    /// the GLM attention module table and must leave this empty. Not part of
    /// the measured table key.
    #[serde(default)]
    pub measured: Vec<Op>,
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
        if self.layer_kind == "sparse_mla" && !self.measured.is_empty() {
            return Err(AicError::ModelConfig(
                "GLM-5.3-Flash sparse MLA is priced by its module table, not generic children"
                    .into(),
            ));
        }
        validate_measured("attention", &self.measured)
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
        let sol = || {
            self.sol(
                &db.system_spec,
                ctx.batch_size as f64,
                ctx.s as f64,
                ctx.prefix as f64,
            )
        };
        match db.database_mode {
            DatabaseMode::Sol | DatabaseMode::SolFull => sol(),
            DatabaseMode::Empirical => {
                analytical_only(db, "attention")?;
                sol()
            }
            _ if self.layer_kind == "kda" => {
                measured_query(db, ctx, "attention", &self.measured, sol)
            }
            _ => {
                // Context: per-request query length over a cached prefix.
                // Decode: absolute per-request KV length with prefix 0.
                let prefix = if self.is_context { ctx.prefix } else { 0 };
                let measured = db.glm53_attention.query(
                    self,
                    ctx.batch_size,
                    prefix,
                    ctx.s,
                    &|batch, prefix, x| Ok(self.sol(&db.system_spec, batch, x, prefix)?.latency_ms),
                )?;
                match measured {
                    Some(latency) => Ok(PerformanceResult::new(latency, Source::Silicon)),
                    None if db.database_mode == DatabaseMode::Hybrid => sol(),
                    None => Err(AicError::PerfDatabase(format!(
                        "GLM-5.3-Flash sparse attention has no {} measurement for this geometry at {}",
                        crate::perf_database::glm53flash::BASENAME,
                        db.data_root.display()
                    ))),
                }
            }
        }
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
    /// Call sites covered by one `mhc_module_perf` row for this role. GLM
    /// rows (Ops W3, vLLM 0.31.0 / SGLang 0.5.20) follow the
    /// DeepSeek-V4 convention: `pre`, `post` and `fused_post_pre` time a
    /// layer's attention and FFN sites together (`num_sites=2`); `expand`
    /// and `contract` time one call. RMSNorm is inside `pre` and
    /// `fused_post_pre` on both backends.
    pub fn row_sites(&self) -> f64 {
        if matches!(self.role.as_str(), "pre" | "post" | "fused_post_pre") {
            2.0
        } else {
            1.0
        }
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
        identity(&self.backend, &self.checkpoint_format)?;
        let (h, c) = (self.hidden_size as f64, self.hc_mult as f64);
        if c != 4.0 || self.sinkhorn_iters != 20 || !matches!(self.tp_size, 1 | 2 | 4) {
            return Err(AicError::ModelConfig(
                "GLM mHC requires TP1/2/4, multiplier4 and20 Sinkhorn iterations".into(),
            ));
        }
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
    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        let sol = || self.sol(&db.system_spec, ctx.num_tokens as f64);
        // Validates identity/geometry in every mode.
        let analytical = sol()?;
        match db.database_mode {
            DatabaseMode::Sol | DatabaseMode::SolFull => return Ok(analytical),
            DatabaseMode::Empirical => analytical_only(db, "mhc")?,
            _ => {}
        }
        if ctx.num_tokens == 0 {
            return Ok(zero());
        }
        // Pure TP (EP=DP=1): vLLM sequence-parallel MoE is off, so mHC runs on
        // all scheduled tokens of the rank. One site is 1/row_sites of a row;
        // the row's util-hold anchor is row_sites x the per-site GLM SOL.
        let sites = self.row_sites();
        let anchor = |_: &str, tokens: f64| {
            self.sol(&db.system_spec, tokens)
                .map_or(f64::NAN, |r| r.latency_ms * sites)
        };
        match db.mhc.query_module(
            &self.role,
            ctx.num_tokens,
            self.hc_mult,
            self.hidden_size,
            &anchor,
        ) {
            Ok(row) => Ok(PerformanceResult::with_energy(
                row.latency / sites,
                row.energy / sites,
                Source::Silicon,
            )),
            Err(err) if db.database_mode == DatabaseMode::Hybrid && err.is_missing_perf_data() => {
                Ok(analytical)
            }
            Err(err) => Err(err),
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
    pub fn weight_bytes(&self) -> f64 {
        4.0 * f64::from(self.num_experts) * f64::from(self.hidden_size)
    }
    pub fn sol(&self, spec: &SystemSpec, x: f64) -> Result<PerformanceResult, AicError> {
        identity(&self.backend, &self.checkpoint_format)?;
        if self.topk == 0 || self.topk > self.num_experts {
            return Err(AicError::ModelConfig("invalid GLM router topk".into()));
        }
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
        analytical_only(db, "router")?;
        self.sol(&db.system_spec, tokens as f64)
    }
}

/// Expected number of DISTINCT routed experts touched by `tokens` tokens that
/// each select `topk` distinct experts out of `num_experts` uniformly:
/// `E * (1 - (1 - k/E)^T)`. Exact under uniform routing (per token an expert
/// is missed with probability `1 - k/E`), monotone and concave in `T`, equal
/// to `k` at `T = 1` and never above `E` or `T * k`.
pub(crate) fn expected_distinct_experts(tokens: f64, topk: u32, num_experts: u32) -> f64 {
    if tokens <= 0.0 || topk == 0 || num_experts == 0 {
        return 0.0;
    }
    let (k, e) = (f64::from(topk), f64::from(num_experts.max(topk)));
    let distinct = if tokens <= 1.0 {
        k * tokens
    } else {
        e * -((-k / e).ln_1p() * tokens).exp_m1()
    };
    distinct.min(e).min(tokens * k)
}

/// GLM routed-expert roofline. Math and activation traffic scale with the
/// actual `tokens * topk` expert assignments exactly as the generic MoE SOL
/// does; weight traffic reads each DISTINCT expert once per forward
/// ([`expected_distinct_experts`]) instead of once per assignment. Experts
/// are TP-sharded, so every rank reads `1/moe_tp` of each touched expert.
pub(crate) fn routed_moe_sol(
    m: &MoeOp,
    spec: &SystemSpec,
    x: f64,
) -> Result<PerformanceResult, AicError> {
    let tokens = x.max(0.0) * f64::from(m.attention_dp_size.max(1));
    let (h, inter) = (f64::from(m.hidden_size), f64::from(m.inter_size));
    let gemms = if m.is_gated { 3.0 } else { 2.0 };
    let (ep, tp) = (
        f64::from(m.moe_ep_size.max(1)),
        f64::from(m.moe_tp_size.max(1)),
    );
    let assignments = tokens * f64::from(m.topk) / ep;
    let experts = expected_distinct_experts(tokens, m.topk, m.num_experts) / ep;
    let q = m.quant_mode.mapping();
    let flops = 2.0 * assignments * h * inter * gemms / tp;
    let bytes = q.memory
        * (assignments * h * 2.0
            + assignments * inter * gemms / tp
            + h * inter * gemms / tp * experts);
    Ok(leaf(spec, flops, bytes, quant_tc_flops(spec, q)?)
        .clamp_non_negative()
        .scaled(m.scale_factor))
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
    /// SILICON/HYBRID generic composition: dense/shared GEMMs, BF16 router
    /// GEMM, routed MoE and analytic activation. Never used for SOL.
    #[serde(default)]
    pub measured: Vec<Op>,
}
impl Glm53FfnOp {
    pub fn validate(&self) -> Result<(), AicError> {
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
            || self.children.len() != if self.is_dense { 3 } else { 5 }
            || self.children.iter().any(|op| {
                !matches!(
                    op,
                    Op::Gemm(_) | Op::Elementwise(_) | Op::Moe(_) | Op::Glm53Router(_)
                )
            })
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
        let width = self.intermediate_size / self.tp_size;
        let quant = if self.is_dense {
            gemm_quant
        } else {
            shared_quant
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
        validate_measured("ffn", &self.measured)
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
        self.children.iter().try_fold(zero(), |sum, child| {
            let cost = match child {
                Op::Moe(m) => routed_moe_sol(m, &db.system_spec, f64::from(ctx.num_tokens))?,
                _ => child.query(&sol_db, ctx)?,
            };
            Ok(sum.plus(cost))
        })
    }
    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        match db.database_mode {
            DatabaseMode::Sol | DatabaseMode::SolFull => self.sol(db, ctx),
            _ => measured_query(db, ctx, "ffn", &self.measured, || self.sol(db, ctx)),
        }
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
    /// SILICON/HYBRID generic composition (embedding, custom all-reduce,
    /// logits GEMM/gather, analytic norm/cast). Never used for SOL.
    #[serde(default)]
    pub measured: Vec<Op>,
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
        validate_measured("primitive", &self.measured)
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
        match db.database_mode {
            DatabaseMode::Sol | DatabaseMode::SolFull => self.sol(db, ctx),
            _ => {
                let mut child_ctx = *ctx;
                if self.token_selection == "last_per_request" {
                    child_ctx.num_tokens = ctx.batch_size;
                }
                measured_query(db, &child_ctx, "primitive", &self.measured, || {
                    self.sol(db, ctx)
                })
            }
        }
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
            measured: vec![],
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
            measured: vec![],
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
    fn distinct_experts_follow_uniform_routing_expectation() {
        // 288 experts, top-8: E*(1-(280/288)^T), hand-evaluated.
        let d = |t: f64| expected_distinct_experts(t, 8, 288);
        assert!((d(1.0) - 8.0).abs() < 1e-12);
        assert!((d(2.0) - 15.777_777_777_777_8).abs() < 1e-9);
        assert!((d(32.0) - 171.079_710_393_639_7).abs() < 1e-9);
        assert!((d(256.0) - 287.787_493_780_612_1).abs() < 1e-9);
        assert_eq!(d(0.0), 0.0);
        assert!(d(1e9) <= 288.0 && d(1e9) > 287.999_999);
        let mut prev = 0.0;
        for t in 1..=512 {
            let v = d(f64::from(t));
            assert!(v > prev && v <= 8.0 * f64::from(t) && v <= 288.0);
            prev = v;
        }
    }
    #[test]
    fn routed_moe_reads_each_distinct_expert_shard_once() {
        let db = db(DatabaseMode::Sol);
        let spec = &db.system_spec;
        let op = MoeOp::new(
            "moe",
            4096,
            2048,
            8,
            288,
            2,
            1,
            MoeQuantMode::Fp8Block,
            "uniform",
        );
        let generic = |t: u32| op.query(&db, t).unwrap().sol.unwrap();
        // One token touches exactly top-k experts: identical to generic SOL.
        let one = routed_moe_sol(&op, spec, 1.0).unwrap().sol.unwrap();
        assert!((one.math_ms - generic(1).math_ms).abs() <= 1e-12 * generic(1).math_ms);
        assert!((one.mem_ms - generic(1).mem_ms).abs() <= 1e-12 * generic(1).mem_ms);
        // Batch 32: compute keeps all 256 assignments; weights drop from 256
        // to E[distinct]=171.08 expert shards of 4096*2048*3/TP2 FP8 bytes.
        let b32 = routed_moe_sol(&op, spec, 32.0).unwrap().sol.unwrap();
        assert!((b32.math_ms - generic(32).math_ms).abs() <= 1e-12 * generic(32).math_ms);
        let shard = 4096.0 * 2048.0 * 3.0 / 2.0;
        let saved =
            shard * (256.0 - expected_distinct_experts(32.0, 8, 288)) / spec.gpu.mem_bw * 1e3;
        assert!((generic(32).mem_ms - b32.mem_ms - saved).abs() <= 1e-9 * saved);
        // TP4 halves each rank's shard of every distinct expert.
        let mut tp4 = op.clone();
        tp4.moe_tp_size = 4;
        let w = |m: &MoeOp, t: f64| {
            routed_moe_sol(m, spec, t).unwrap().sol.unwrap().mem_ms
                - routed_moe_sol(m, spec, 0.0).unwrap().sol.unwrap().mem_ms
        };
        let act = |m: &MoeOp| 1.0 * 8.0 * (4096.0 * 2.0 + 2048.0 * 3.0 / f64::from(m.moe_tp_size));
        let wbytes = |m: &MoeOp, t: f64| w(m, t) * spec.gpu.mem_bw / 1e3 - t * act(m);
        let ratio = wbytes(&op, 32.0) / wbytes(&tp4, 32.0);
        assert!((ratio - 2.0).abs() < 1e-9);
    }
    #[test]
    fn ffn_op_level_and_fpm_sol_share_distinct_expert_weight_term() {
        use crate::operators::{elementwise::ElementwiseOp, gemm::GemmOp};
        let db = db(DatabaseMode::Sol);
        let q = GemmQuantMode::Fp8Block;
        let moe = MoeOp::new(
            "moe",
            4096,
            2048,
            8,
            288,
            2,
            1,
            MoeQuantMode::Fp8Block,
            "uniform",
        );
        let op = Glm53FfnOp {
            name: "ffn_3".into(),
            backend: "vllm".into(),
            checkpoint_format: "fp8".into(),
            is_context: false,
            is_dense: false,
            hidden_size: 4096,
            intermediate_size: 2048,
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
            gemm_quant_mode: q,
            shared_quant_mode: q,
            moe_quant_mode: MoeQuantMode::Fp8Block,
            children: vec![
                Op::Gemm(GemmOp::new("shared_up", 2048, 4096, q)),
                Op::Elementwise(ElementwiseOp::new("shared_act", 6.0 * 1024.0)),
                Op::Gemm(GemmOp::new("shared_down", 4096, 1024, q)),
                Op::Glm53Router(Glm53RouterOp {
                    name: "router".into(),
                    backend: "vllm".into(),
                    checkpoint_format: "fp8".into(),
                    hidden_size: 4096,
                    num_experts: 288,
                    topk: 8,
                }),
                Op::Moe(moe.clone()),
            ],
            measured: vec![],
        };
        for t in [1u32, 32, 4096] {
            let ctx = RuntimeContext {
                batch_size: t,
                num_tokens: t,
                ..RuntimeContext::default()
            };
            let op_level = op.sol(&db, &ctx).unwrap().latency_ms;
            let x = f64::from(t);
            let fpm = crate::operators::fpm_sol::op_sol_latency_ms(
                &Op::Glm53Ffn(op.clone()),
                &db,
                x,
                x,
                1.0,
                0.0,
            )
            .unwrap();
            assert!((op_level - fpm).abs() <= 1e-9 * op_level, "t={t}");
            let routed = routed_moe_sol(&moe, &db.system_spec, x).unwrap().latency_ms;
            assert!(routed <= moe.query(&db, t).unwrap().latency_ms);
        }
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
            analytical_only(&db(DatabaseMode::Silicon), "attention"),
            Err(AicError::PerfDatabase(_))
        ));
    }

    fn kda_kernel(kernel: &str, phase: &str) -> Op {
        Op::Kda(
            serde_json::from_value(serde_json::json!({
                "name": format!("{phase}_kda_0_{kernel}"), "scale_factor": 1.0,
                "kernel_source": kernel, "phase": phase, "d_model": 4096, "d_conv": 4,
                "num_k_heads": 32, "head_k_dim": 128, "num_v_heads": 32, "head_v_dim": 128,
                "draft_tokens": 0,
            }))
            .unwrap(),
        )
    }

    fn mhc_site(role: &str) -> Glm53MhcOp {
        Glm53MhcOp {
            name: "mhc_pre_attn_1".into(),
            role: role.into(),
            backend: "sglang".into(),
            checkpoint_format: "fp8".into(),
            tp_size: 2,
            is_context: false,
            hidden_size: 4096,
            hc_mult: 4,
            sinkhorn_iters: 20,
        }
    }

    fn ctx(batch: u32, s: u32, prefix: u32, is_context: bool) -> RuntimeContext {
        RuntimeContext {
            batch_size: batch,
            s,
            prefix,
            num_tokens: if is_context { batch * s } else { batch },
            ..RuntimeContext::default()
        }
    }

    #[test]
    fn sol_mode_keeps_glm_formulas_regardless_of_measured_children() {
        let db = db(DatabaseMode::Sol);
        let mut kda = attention("kda");
        let c = ctx(2, 512, 0, true);
        let bare = kda.query(&db, &c).unwrap();
        kda.measured = vec![kda_kernel("chunk_kda", "context")];
        let with = kda.query(&db, &c).unwrap();
        assert_eq!(bare.latency_ms, with.latency_ms);
        assert_eq!(
            bare.latency_ms,
            kda.sol(&db.system_spec, 2.0, 512.0, 0.0)
                .unwrap()
                .latency_ms
        );
        let mhc = mhc_site("pre");
        let expected = mhc.sol(&db.system_spec, 8.0).unwrap().latency_ms;
        assert_eq!(
            mhc.query(&db, &ctx(8, 1, 0, false)).unwrap().latency_ms,
            expected
        );
    }

    #[test]
    fn silicon_prices_generic_children_and_fails_closed_on_sol_children() {
        // gb300/sglang/0.5.14 carries DeepSeek-V4-Flash mHC rows (hc4, hidden 4096)
        // but no KDA rows for GLM's 32-head TP2 shard.
        let silicon = db(DatabaseMode::Silicon);
        // One GLM pre site is half of a two-site row (RMSNorm inside the row).
        let mhc = mhc_site("pre");
        let c = ctx(8, 1, 0, false);
        let got = mhc.query(&silicon, &c).unwrap();
        let row = silicon
            .mhc
            .query_module("pre", 8, 4, 4096, &|_, _| f64::NAN)
            .unwrap();
        assert!((got.latency_ms - 0.5 * row.latency).abs() < 1e-15 && got.latency_ms > 0.0);
        assert_eq!(got.source, Source::Silicon);
        // No fused_post_pre rows in this runtime: SILICON fails, HYBRID is SOL.
        let fused = mhc_site("fused_post_pre");
        assert!(fused.query(&silicon, &c).is_err());
        let hybrid_fused = fused.query(&db(DatabaseMode::Hybrid), &c).unwrap();
        assert_eq!(hybrid_fused.source, Source::Sol);
        assert_eq!(
            hybrid_fused.latency_ms,
            fused.sol(&silicon.system_spec, 8.0).unwrap().latency_ms
        );

        let mut kda = attention("kda");
        kda.backend = "sglang".into();
        kda.measured = vec![kda_kernel(
            "fused_recurrent_kda_packed_decode",
            "generation",
        )];
        kda.is_context = false;
        let err = kda.query(&silicon, &ctx(4, 1024, 0, false)).unwrap_err();
        assert!(err.to_string().contains("no SILICON data"), "{err}");
        // HYBRID keeps the child's labelled SOL fallback instead of failing.
        let hybrid = db(DatabaseMode::Hybrid);
        let result = kda.query(&hybrid, &ctx(4, 1024, 0, false)).unwrap();
        assert_eq!(result.source, Source::Sol);
        // Without a measured composition SILICON fails and HYBRID reports SOL.
        kda.measured.clear();
        assert!(kda.query(&silicon, &ctx(4, 1024, 0, false)).is_err());
        assert_eq!(
            kda.query(&hybrid, &ctx(4, 1024, 0, false)).unwrap().source,
            Source::Sol
        );
        // A GLM analytical op can never be smuggled in as a measured child.
        let mut bad = attention("kda");
        bad.measured = vec![Op::Glm53Router(Glm53RouterOp {
            name: "router".into(),
            backend: "sglang".into(),
            checkpoint_format: "fp8".into(),
            hidden_size: 4096,
            num_experts: 288,
            topk: 8,
        })];
        assert!(bad.query(&silicon, &c).is_err());
        let mut sparse = attention("sparse_mla");
        sparse.measured = vec![kda_kernel("chunk_kda", "context")];
        assert!(sparse.validate().is_err());
    }

    fn sparse_tree(version: &str) -> tempfile::TempDir {
        use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
        let root = tempfile::tempdir().unwrap();
        let systems = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/aisimulate/src/aisimulate_core/systems");
        std::fs::copy(systems.join("gb300.yaml"), root.path().join("gb300.yaml")).unwrap();
        let dir = root
            .path()
            .join("data/gb300/glm53_attention/vllm")
            .join(version);
        std::fs::create_dir_all(&dir).unwrap();
        let geometry: &'static str =
            crate::perf_database::glm53flash::geometry(&attention("sparse_mla"))
                .unwrap()
                .leak();
        let sha: &'static str = "a".repeat(64).leak();
        let digest: &'static str = format!("sha256:{sha}").leak();
        write_parquet(
            &dir.join(crate::perf_database::glm53flash::BASENAME),
            &[
                Col::Str("component", vec!["attention"; 2]),
                Col::Str("geometry", vec![geometry; 2]),
                Col::I64("batch_size", vec![2; 2]),
                Col::I64("prefix", vec![0; 2]),
                Col::I64("x", vec![512, 4096]),
                Col::F64("latency", vec![1.5, 9.0]),
                Col::Str("kernel_source", vec!["glm53_sparse_mla"; 2]),
                Col::Str("measurement_scope", vec!["local_compute"; 2]),
                Col::Str("source_sha256", vec![sha; 2]),
                Col::Str("config_sha256", vec![sha; 2]),
                Col::Str("runtime_digest", vec![digest; 2]),
                Col::Bool("used_cuda_graph", vec![false; 2]),
                Col::I64("sample_count", vec![5; 2]),
                Col::Str("kv_seed_regime", vec!["n/a"; 2]),
                Col::Str("execution_profile", vec!["full"; 2]),
            ],
        );
        root
    }

    #[test]
    fn sparse_attention_uses_only_the_exact_runtime_glm_table() {
        const RUNTIME: &str = "0.31.0";
        let root = sparse_tree(RUNTIME);
        let op = attention("sparse_mla");
        let load = |version: &str, mode| {
            PerfDatabase::load_resolved(root.path(), "gb300", "vllm", version, true, false, false)
                .unwrap()
                .with_mode(mode, TransferPolicy::ALL)
        };
        let c = ctx(2, 512, 0, true);
        let hit = op.query(&load(RUNTIME, DatabaseMode::Silicon), &c).unwrap();
        assert_eq!((hit.latency_ms, hit.source), (1.5, Source::Silicon));
        // Another geometry: SILICON fails closed, HYBRID reports labelled SOL.
        let mut tp4 = op.clone();
        tp4.tp_size = 4;
        tp4.num_heads = 16;
        assert!(
            tp4.query(&load(RUNTIME, DatabaseMode::Silicon), &c)
                .is_err()
        );
        let fallback = tp4.query(&load(RUNTIME, DatabaseMode::Hybrid), &c).unwrap();
        assert_eq!(fallback.source, Source::Sol);
        // Another runtime never borrows this runtime's GLM table.
        const OTHER: &str = "0.30.0+glm53tail.eb4704514fdf";
        let other = root
            .path()
            .join("data/gb300/glm53_attention/vllm")
            .join(OTHER);
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("reuse.yaml"), "schema_version: 1\nreuse: []\n").unwrap();
        let err = op
            .query(&load(OTHER, DatabaseMode::Silicon), &c)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("no glm53_attention_module_perf.parquet"),
            "{err}"
        );
    }

    #[test]
    fn mhc_reads_glm_rows_with_w3_site_semantics_from_exact_runtime() {
        use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
        const RUNTIME: &str = "0.31.0";
        let root = tempfile::tempdir().unwrap();
        let systems = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/aisimulate/src/aisimulate_core/systems");
        std::fs::copy(systems.join("gb300.yaml"), root.path().join("gb300.yaml")).unwrap();
        let dir = root.path().join("data/gb300/mhc/vllm").join(RUNTIME);
        std::fs::create_dir_all(&dir).unwrap();
        let ops = ["pre", "post", "fused_post_pre", "expand", "contract"];
        let op_names: Vec<&str> = ops.iter().flat_map(|op| [*op, *op]).collect();
        write_parquet(
            &dir.join("mhc_module_perf.parquet"),
            &[
                Col::Str("op_name", op_names),
                Col::I64("num_tokens", [16, 1024].repeat(5)),
                Col::I64("hc_mult", vec![4; 10]),
                Col::I64("hidden_size", vec![4096; 10]),
                Col::F64(
                    "latency",
                    vec![0.02, 0.04, 0.01, 0.03, 0.03, 0.07, 0.01, 0.03, 0.005, 0.017],
                ),
                Col::Str("kernel_source", vec!["fixture"; 10]),
            ],
        );
        let load = |mode| {
            PerfDatabase::load_resolved(root.path(), "gb300", "vllm", RUNTIME, false, false, false)
                .unwrap()
                .with_mode(mode, TransferPolicy::ALL)
        };
        let silicon = load(DatabaseMode::Silicon);
        let c = ctx(1024, 1, 0, false);
        let site = |role: &str| {
            let mut op = mhc_site(role);
            op.backend = "vllm".into();
            op.query(&silicon, &c).unwrap()
        };
        // Two-site rows: one site = 0.5 x row; expand/contract: one call = row.
        for (role, row, sites) in [
            ("pre", 0.04, 2.0),
            ("post", 0.03, 2.0),
            ("fused_post_pre", 0.07, 2.0),
            ("expand", 0.03, 1.0),
            ("contract", 0.017, 1.0),
        ] {
            let got = site(role);
            assert!((got.latency_ms - row / sites).abs() < 1e-15, "{role}");
            assert_eq!(got.source, Source::Silicon);
        }
        // vLLM forward: pre + 89 fused + post sites + expand + contract
        // = 0.5 + 44.5 + 0.5 rows + 1 + 1.
        let forward = site("pre").latency_ms
            + 89.0 * site("fused_post_pre").latency_ms
            + site("post").latency_ms
            + site("expand").latency_ms
            + site("contract").latency_ms;
        let rows = 0.5 * 0.04 + 44.5 * 0.07 + 0.5 * 0.03 + 0.03 + 0.017;
        assert!((forward - rows).abs() < 1e-12);
        // SOL mode never reads the table.
        let op = mhc_site("fused_post_pre");
        assert_eq!(
            op.query(&load(DatabaseMode::Sol), &c).unwrap().latency_ms,
            op.sol(&silicon.system_spec, 1024.0).unwrap().latency_ms
        );
    }
}
