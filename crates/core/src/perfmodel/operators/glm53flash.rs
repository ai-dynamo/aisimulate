// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//! Independently expressed GLM-5.3-Flash analytical operator contracts.
//!
//! Geometry: Z.AI GLM-5.3-Flash config.json at eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
//! (MIT, Copyright (c) 2026 Z.AI Co., Ltd). Execution boundaries: vllm-project/vllm
//! ced6857afa0ea7b2e3f0846a62e1394e90f15607, vllm/models/glm5next/nvidia/{model,attention,kda}.py,
//! and sgl-project/sglang 94602c9c2b7cbdb8efd5c52802dac6a1c180089e,
//! python/sglang/srt/models/glm5_next.py (Apache-2.0). See THIRD_PARTY_NOTICES.md.
//! These are payload/roofline bounds, not allocator or kernel-launch predictions.

use crate::common::enums::{DatabaseMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::{SystemSpec, quant_tc_flops};
use crate::operators::base::{PerformanceResult, SolComponents};
use crate::operators::op::RuntimeContext;
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
// The Ops PR supplies the measured table. Until then explicit SILICON must
// fail; HYBRID reports Source::Sol rather than relabelling an analytical value.
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
/// weight. Index scoring is skipped for <=topk tokens; selected attention is
/// dense there. Above topk, completed pools plus the uncompressed tail are used.
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
        let score =
            floor_sum(n) - floor_sum(n.min(k)) + frac * if n + 1.0 > k { pools } else { 0.0 };
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
                + 2.0 * (k * n * (d + v) + q * i * j + h * (2.0 * j + i) + q + k + 2.0 * j)
                + 4.0 * (h * i + self.index_pool as f64 * j)
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
            result = result.plus(gemm(spec, x, i * j, q, GemmQuantMode::Bfloat16)?);
            result = result.plus(gemm(spec, x, j + i, h, GemmQuantMode::Bfloat16)?);
            result = result.plus(gemm(spec, x, j, h, GemmQuantMode::Bfloat16)?);
            result = result.plus(leaf(
                spec,
                2.0 * x * h * i,
                4.0 * h * i + 2.0 * x * (h + i),
                fp32,
            ));
            let end = if self.is_context { prefix + s } else { s };
            let start = if self.is_context {
                prefix
            } else {
                (s - 1.0).max(0.0)
            };
            let (pooled_pairs, selected_pairs) =
                sparse_pairs(start, end, self.index_pool, self.index_topk);
            let (pooled_pairs, selected_pairs) = (batch * pooled_pairs, batch * selected_pairs);
            let fp8 = quant_tc_flops(spec, GemmQuantMode::Fp8.mapping())?;
            result = result.plus(leaf(
                spec,
                2.0 * pooled_pairs * i * j,
                pooled_pairs * (j + 4.0) + 2.0 * x * i * j,
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
                x * (5.0 * (q + k) + 12.0 * j + 7.0 * i * j),
                x * (4.0 * (q + k) + 12.0 * j + 4.0 * i * j + k),
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
        analytical_only(db, "attention")?;
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
    pub hidden_size: u32,
    pub hc_mult: u32,
    pub sinkhorn_iters: u32,
}
impl Glm53MhcOp {
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
        if c != 4.0 || self.sinkhorn_iters != 20 {
            return Err(AicError::ModelConfig(
                "GLM mHC requires multiplier4 and20 Sinkhorn iterations".into(),
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
    pub fn query(&self, db: &PerfDatabase, tokens: u32) -> Result<PerformanceResult, AicError> {
        analytical_only(db, "mhc")?;
        self.sol(&db.system_spec, tokens as f64)
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
        assert_eq!(sparse_pairs(0.0, 4.0, 4, 2048), (0.0, 10.0));
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
}
