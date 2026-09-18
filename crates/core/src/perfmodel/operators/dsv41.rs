// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek-V4.1 text AR operator and phase contracts.
//!
//! Architecture source: deepseek-ai/DeepSeek-V4.1-Flash, revision
//! fb2764a5cf321eaa5070ca8f9e892818f477c16d, config.json / inference/model.py
//! and DeepSeek_V41_Tech_Report.pdf (MIT, Copyright (c) 2023 DeepSeek).
//! Serving layout/scoring source: sgl-project/sglang at
//! 1aa0e962b206102b7c439a4a0c4981cfec6e87bc,
//! python/sglang/srt/layers/attention/deepseek_v4_backend.py and
//! python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py (Apache-2.0,
//! Copyright SGLang contributors). Independently expressed analytical
//! adaptations; see THIRD_PARTY_NOTICES.md and docs/deepseek-v41-storage.md.

use serde::{Deserialize, Serialize};

use crate::common::enums::{DatabaseMode, FmhaQuantMode, GemmQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::{SystemSpec, quant_tc_flops};
use crate::operators::base::{PerformanceResult, SolComponents, Source};
use crate::operators::op::{Op, RuntimeContext};
use crate::perf_database::PerfDatabase;

fn leaf(spec: &SystemSpec, flops: f64, bytes: f64, rate: f64) -> PerformanceResult {
    PerformanceResult::sol(SolComponents::new(
        flops / rate * 1e3,
        bytes / spec.gpu.mem_bw * 1e3,
    ))
}

fn zero() -> PerformanceResult {
    PerformanceResult::sol(SolComponents::new(0.0, 0.0))
}

/// New V41 kernels have no measured lookup in the SOL release. HYBRID's
/// analytic contribution is explicitly SOL, never an invented utilization or
/// a V4 module hit. SILICON must fail until the V41 collector publishes it.
fn analytic_mode(db: &PerfDatabase, name: &str) -> Result<(), AicError> {
    match db.database_mode {
        DatabaseMode::Silicon => Err(AicError::PerfDatabase(format!(
            "DeepSeek-V4.1 {name} has no measured SILICON data"
        ))),
        DatabaseMode::Empirical => Err(AicError::EmpiricalNotImplemented(format!(
            "DeepSeek-V4.1 {name} has no empirical anchor"
        ))),
        _ => Ok(()),
    }
}

/// Sum floor(t/r) from t=1 through n; fractional tails retain continuous
/// workload weight for FPM interpolation, with integer publication boundaries.
fn floor_prefix(n: f64, ratio: f64) -> f64 {
    let n = n.max(0.0);
    let whole = n.floor();
    let q = (whole / ratio).floor();
    let rem = whole - q * ratio;
    ratio * q * (q - 1.0) / 2.0 + q * (rem + 1.0) + (n - whole) * q
}

fn limited_pairs(query: f64, prefix: f64, limit: f64) -> f64 {
    let antiderivative = |n: f64| {
        let ramp = n.min(limit).max(0.0);
        ramp * (ramp + 1.0) / 2.0 + (n - limit).max(0.0) * limit
    };
    antiderivative(prefix + query) - antiderivative(prefix)
}

fn compressed_pairs(query: f64, prefix: f64, ratio: f64, topk: f64) -> f64 {
    let antiderivative = |n: f64| {
        let saturation = ratio * topk;
        floor_prefix(n.min(saturation), ratio) + (n - saturation).max(0.0) * topk
    };
    antiderivative(prefix + query) - antiderivative(prefix)
}

/// Persistent payload layout, independent of attention arithmetic precision.
/// LogicalFp4 is the legacy theoretical layout, not a qualified runtime layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dsv41KvCacheLayout {
    #[default]
    LogicalFp4,
    SglangFp8Bf16,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv41AttentionOp {
    pub name: String,
    pub is_context: bool,
    pub role: String,
    pub compress_ratio: u32,
    pub hidden_size: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub q_lora_rank: u32,
    pub o_lora_rank: u32,
    pub o_groups: u32,
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub window_size: u32,
    pub candidate_limit: u32,
    pub is_candidate_source: bool,
    pub bounded_prefill: bool,
    pub gemm_quant_mode: GemmQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    // Appended in schema 19. The default accepts older JSON; bincode is positional.
    #[serde(default)]
    pub kv_cache_layout: Dsv41KvCacheLayout,
}

impl Dsv41AttentionOp {
    fn validate_role(&self) -> Result<(), AicError> {
        match self.role.as_str() {
            "swa" | "full" | "reindex" | "reuse" => Ok(()),
            _ => Err(AicError::ModelConfig(format!(
                "DeepSeek-V4.1 attention role must be swa, full, reindex, or reuse; got {:?}",
                self.role
            ))),
        }
    }

    pub fn weight_bytes(&self) -> f64 {
        let (h, q, o, n, d, g) = (
            self.hidden_size as f64,
            self.q_lora_rank as f64,
            self.o_lora_rank as f64,
            self.num_heads as f64,
            self.head_dim as f64,
            self.o_groups as f64,
        );
        let w8 = self.gemm_quant_mode.mapping().memory
            + if self.gemm_quant_mode == GemmQuantMode::Fp8Block {
                1.0 / 1024.0
            } else {
                0.0
            };
        let mut bytes = (h * q + q * n * d + h * d + g * o * h) * w8 + n * d * o * 2.0;
        if self.role == "full" {
            bytes += h * d * 2.0;
            if self.compress_ratio > 1 {
                bytes += h * d * 2.0;
            }
            bytes += d * self.index_head_dim as f64 * 2.0;
        }
        if self.role == "full" || self.role == "reindex" {
            bytes += q * self.index_n_heads as f64 * self.index_head_dim as f64 * w8;
            bytes += h * self.index_n_heads as f64 * 2.0;
        }
        // Attention sinks, normalization vectors and compressor positional bias.
        bytes + n * 4.0 + (q + d) * 2.0
    }

    /// Shared f64 roofline for native queries and whole-model FPM interpolation.
    /// Every serial projection/kernel contributes its own max(math,memory).
    pub fn sol(
        &self,
        spec: &SystemSpec,
        batch: f64,
        s: f64,
        prefix: f64,
    ) -> Result<PerformanceResult, AicError> {
        self.validate_role()?;
        if self.role == "full" && self.compress_ratio == 0 {
            return Err(AicError::ModelConfig(
                "DeepSeek-V4.1 full attention requires a positive compress_ratio".into(),
            ));
        }
        if batch <= 0.0 || s <= 0.0 {
            return Ok(zero());
        }
        let (h, q, o, n, d, g) = (
            self.hidden_size as f64,
            self.q_lora_rank as f64,
            self.o_lora_rank as f64,
            self.num_heads as f64,
            self.head_dim as f64,
            self.o_groups as f64,
        );
        let (window_entry_bytes, main_entry_bytes) = match self.kv_cache_layout {
            Dsv41KvCacheLayout::LogicalFp4 => (d, d * 0.5625),
            Dsv41KvCacheLayout::SglangFp8Bf16 => {
                // Only the pinned V4.1 low-ratio, 512-wide FlashMLA contract.
                if self.head_dim != 512 || self.index_head_dim != 128 || self.compress_ratio > 2 {
                    return Err(AicError::ModelConfig(
                        "SGLang V4.1 KV layout requires head_dim=512, index=128, ratio=0/1/2"
                            .into(),
                    ));
                }
                (584.0, 584.0)
            }
        };
        let tokens = if self.is_context { batch * s } else { batch };
        let bf16 = quant_tc_flops(spec, GemmQuantMode::Bfloat16.mapping())?;
        let fp8 = quant_tc_flops(spec, GemmQuantMode::Fp8.mapping())?;
        let gemm = quant_tc_flops(spec, self.gemm_quant_mode.mapping())?;
        let attn = quant_tc_flops(spec, self.fmha_quant_mode.mapping())?;
        let w8 = self.gemm_quant_mode.mapping().memory
            + if self.gemm_quant_mode == GemmQuantMode::Fp8Block {
                1.0 / 1024.0
            } else {
                0.0
            };
        let mm = |a: f64, b: f64, nt: f64, weight: f64, rate: f64| {
            leaf(
                spec,
                2.0 * nt * a * b,
                a * b * weight + nt * (a + b) * 2.0,
                rate,
            )
        };
        let mut result = mm(h, q, tokens, w8, gemm)
            .plus(mm(q, n * d, tokens, w8, gemm))
            .plus(mm(h, d, tokens, w8, gemm))
            .plus(leaf(
                spec,
                2.0 * tokens * n * d * o,
                n * d * o * 2.0 + tokens * (n * d + g * o) * 2.0,
                bf16,
            ))
            .plus(mm(g * o, h, tokens, w8, gemm));
        if self.kv_cache_layout == Dsv41KvCacheLayout::SglangFp8Bf16 {
            // Fused norm/RoPE/store reads the BF16 projection output and writes
            // one physical SWA row. Projection's own BF16 output write is above.
            result = result.plus(leaf(
                spec,
                0.0,
                tokens * (d * 2.0 + window_entry_bytes),
                bf16,
            ));
        }
        let end = if self.is_context { prefix + s } else { s };
        let ratio = self.compress_ratio as f64;
        let compressed_len = if ratio > 0.0 {
            (end / ratio).floor()
        } else {
            0.0
        };
        if self.role == "full" {
            let mult = if self.compress_ratio > 1 { 2.0 } else { 1.0 };
            result = result.plus(mm(h, d, tokens, 2.0, bf16).scaled(mult));
            let produced = if self.is_context {
                (end / ratio).floor() - (prefix / ratio).floor()
            } else {
                (end / ratio).floor() - ((end - 1.0).max(0.0) / ratio).floor()
            };
            let ihd = self.index_head_dim as f64;
            result = result.plus(mm(d, ihd, batch * produced, 2.0, bf16));
            // Main FP4 rounding is temporary: SGLang persists only the 584-byte
            // FlashMLA row, not an additional packed main cache. Index stays FP4.
            let packed = main_entry_bytes + ihd * 0.53125;
            result = result.plus(leaf(spec, 0.0, batch * produced * (d * 2.0 + packed), bf16));
        }
        if self.role == "full" || self.role == "reindex" {
            let (inh, ihd) = (self.index_n_heads as f64, self.index_head_dim as f64);
            result = result
                .plus(mm(q, inh * ihd, tokens, w8, fp8))
                .plus(mm(h, inh, tokens, 2.0, bf16));
            // Dense/paged index GEMMs score the full compressed context before
            // two_level_decode_logits / _mask_topk_scores masks candidates.
            // candidate_limit describes eligibility, not a pre-GEMM gather.
            let index_len = compressed_len;
            let fp4 = quant_tc_flops(spec, GemmQuantMode::Nvfp4.mapping())?;
            result = result.plus(leaf(
                spec,
                2.0 * tokens * inh * ihd * index_len,
                batch * index_len * ihd * 0.53125
                    + tokens * inh * ihd * 0.53125
                    + tokens * index_len * 4.0,
                fp4,
            ));
            // Materialized scores, top-k positions, and optional coarse candidate blocks.
            let candidate_bytes = if self.is_candidate_source {
                tokens * compressed_len / 8.0 * 4.0
            } else {
                0.0
            };
            result = result.plus(leaf(
                spec,
                0.0,
                tokens * (index_len * 4.0 + self.index_topk as f64 * 4.0) + candidate_bytes,
                bf16,
            ));
        }
        let wp = if self.is_context {
            batch
                * limited_pairs(
                    s,
                    if self.bounded_prefill { 0.0 } else { prefix },
                    self.window_size as f64,
                )
        } else {
            batch * end.min(self.window_size as f64)
        };
        let cp = if self.compress_ratio == 0 {
            0.0
        } else if self.is_context {
            batch * compressed_pairs(s, prefix, ratio, self.index_topk as f64)
        } else {
            batch * compressed_len.min(self.index_topk as f64)
        };
        // SOL assumes ideal reuse of overlapping window entries across
        // queries. Pair count determines arithmetic, but HBM sees each
        // unique window KV row once. Irregular compressed top-k selections
        // retain their per-query traffic; no such reuse is guaranteed there.
        let window_rows = if self.is_context {
            batch
                * (s + if self.bounded_prefill {
                    0.0
                } else {
                    prefix.min((self.window_size as f64 - 1.0).max(0.0))
                })
        } else {
            batch * end.min(self.window_size as f64)
        };
        result = result.plus(leaf(
            spec,
            4.0 * n * d * (wp + cp),
            (window_rows * window_entry_bytes + cp * main_entry_bytes) + tokens * n * d * 4.0,
            attn,
        ));
        Ok(result)
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        self.validate_role()?;
        analytic_mode(db, "CSA2 attention")?;
        self.sol(
            &db.system_spec,
            ctx.batch_size as f64,
            ctx.s as f64,
            ctx.prefix as f64,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv41MhcOp {
    pub name: String,
    pub hidden_size: u32,
    pub hc_mult: u32,
    pub sinkhorn_iters: u32,
}
impl Dsv41MhcOp {
    pub fn weight_bytes(&self) -> f64 {
        let hc = self.hc_mult as f64;
        2.0 * ((hc + 2.0) * hc * (hc * self.hidden_size as f64 + 1.0) + 3.0) * 4.0
    }
    pub fn sol(&self, spec: &SystemSpec, tokens: f64) -> Result<PerformanceResult, AicError> {
        if tokens <= 0.0 {
            return Ok(zero());
        }
        let (h, hc) = (self.hidden_size as f64, self.hc_mult as f64);
        let mixes = (hc + 2.0) * hc;
        let ops = 2.0
            * tokens
            * (2.0 * hc * h * mixes
                + (hc * hc + 2.0 * hc) * self.sinkhorn_iters as f64
                + 2.0 * hc * hc * h
                + 2.0 * hc * h);
        // Single pass over each attention/FFN residual site. Coefficients are
        // FP32; residual activation traffic remains BF16.
        let bytes =
            self.weight_bytes() + 2.0 * tokens * hc * h * 2.0 * 2.0 + 2.0 * tokens * mixes * 4.0;
        let fp32 = spec
            .gpu
            .fp32_flops
            .filter(|v| v.is_finite() && *v > 0.0)
            .ok_or_else(|| {
                AicError::MissingSystemFlops(
                    "DeepSeek-V4.1 scalar mHC requires fp32_flops in the system spec".into(),
                )
            })?;
        Ok(leaf(spec, ops, bytes, fp32))
    }
    pub fn query(&self, db: &PerfDatabase, tokens: u32) -> Result<PerformanceResult, AicError> {
        analytic_mode(db, "single-pass mHC")?;
        self.sol(&db.system_spec, tokens as f64)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv41EngramOp {
    pub name: String,
    pub num_embeddings: u64,
    pub head_dim: u32,
    pub hash_columns: u32,
    pub hidden_size: u32,
    pub hc_mult: u32,
    pub tp_size: u32,
}
impl Dsv41EngramOp {
    pub fn weight_bytes(&self) -> f64 {
        let rows = self.num_embeddings.div_ceil(self.tp_size as u64) as f64;
        let d = self.head_dim as f64;
        let projection =
            self.hash_columns as f64 * d * self.hidden_size as f64 * (self.hc_mult + 1) as f64;
        rows * (d + d / 32.0)
            + projection * (1.0 + 1.0 / 1024.0)
            + 2.0 * self.hc_mult as f64 * self.hidden_size as f64 * 2.0
    }
    pub fn sol(&self, spec: &SystemSpec, tokens: f64) -> Result<PerformanceResult, AicError> {
        if tokens <= 0.0 {
            return Ok(zero());
        }
        let input = self.hash_columns as f64 * self.head_dim as f64;
        let output = self.hidden_size as f64 * (self.hc_mult + 1) as f64;
        let fp8 = quant_tc_flops(spec, GemmQuantMode::Fp8.mapping())?;
        let lookup = leaf(
            spec,
            0.0,
            tokens * input * ((1.0 + 1.0 / 32.0) / self.tp_size as f64 + 2.0),
            fp8,
        );
        let projection = leaf(
            spec,
            2.0 * tokens * input * output,
            input * output * (1.0 + 1.0 / 1024.0) + tokens * (input + output) * 2.0,
            fp8,
        );
        let gate = leaf(
            spec,
            0.0,
            tokens * self.hidden_size as f64 * self.hc_mult as f64 * 8.0,
            fp8,
        );
        Ok(lookup.plus(projection).plus(gate))
    }
    pub fn query(&self, db: &PerfDatabase, tokens: u32) -> Result<PerformanceResult, AicError> {
        analytic_mode(db, "Engram")?;
        self.sol(&db.system_spec, tokens as f64)
    }
}

/// Checkpoint-native dense projection. V41's FP8 block is 32x32, so it
/// must not query the legacy 128x128 FP8Block GEMM table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv41LinearOp {
    pub name: String,
    pub n: u32,
    pub k: u32,
    pub quant_mode: GemmQuantMode,
}
impl Dsv41LinearOp {
    pub fn weight_bytes(&self) -> f64 {
        let elements = self.n as f64 * self.k as f64;
        elements * self.quant_mode.mapping().memory
            + if self.quant_mode == GemmQuantMode::Fp8Block {
                self.n.div_ceil(32) as f64 * self.k.div_ceil(32) as f64
            } else {
                0.0
            }
    }
    pub fn sol(&self, spec: &SystemSpec, tokens: f64) -> Result<PerformanceResult, AicError> {
        if tokens <= 0.0 {
            return Ok(zero());
        }
        let rate = quant_tc_flops(spec, self.quant_mode.mapping())?;
        Ok(leaf(
            spec,
            2.0 * tokens * self.n as f64 * self.k as f64,
            self.weight_bytes() + tokens * (self.n + self.k) as f64 * 2.0,
            rate,
        ))
    }
    pub fn query(&self, db: &PerfDatabase, tokens: u32) -> Result<PerformanceResult, AicError> {
        analytic_mode(db, "32x32 dense projection")?;
        self.sol(&db.system_spec, tokens as f64)
    }
}

/// A decoder layer's token domain. All resident weights stay present when its
/// late-layer prefill inputs are shortened. Decoder replay never shortens decode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv41StageOp {
    pub name: String,
    pub is_context: bool,
    pub decoder_replay: bool,
    pub bounded: bool,
    pub window_size: u32,
    pub children: Vec<Op>,
}
impl Dsv41StageOp {
    pub fn scope(&self, s: f64, prefix: f64) -> (f64, f64) {
        if self.is_context && self.decoder_replay && self.bounded {
            let tail = s.min(self.window_size as f64);
            (tail, prefix + s - tail)
        } else {
            (s, prefix)
        }
    }
    pub fn weight_bytes(&self) -> f64 {
        self.children.iter().map(Op::weight_bytes).sum()
    }
    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        let (s, prefix) = self.scope(ctx.s as f64, ctx.prefix as f64);
        let mut scoped = *ctx;
        scoped.s = s as u32;
        scoped.prefix = prefix as u32;
        if self.is_context {
            scoped.num_tokens = ctx.batch_size * scoped.s;
        }
        let mut total = None;
        for op in &self.children {
            let mut child = scoped;
            if self.is_context && op.is_logits_gemm() {
                child.num_tokens = ctx.batch_size;
            }
            let result = op.query(db, &child)?;
            total = Some(match total {
                None => result,
                Some(prev) => PerformanceResult::plus(prev, result),
            });
        }
        Ok(total.unwrap_or_else(|| PerformanceResult::new(0.0, Source::Sol)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::enums::TransferPolicy;
    use std::path::PathBuf;

    fn test_db(mode: DatabaseMode) -> PerfDatabase {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/aisimulate/src/aiconfigurator_core/systems");
        PerfDatabase::load(&root, "gb300", "sglang", "0.5.14")
            .unwrap()
            .with_mode(mode, TransferPolicy::ALL)
    }

    fn unit_spec() -> SystemSpec {
        let mut spec = test_db(DatabaseMode::Sol).system_spec.clone();
        spec.gpu.mem_bw = 1e6;
        spec.gpu.bfloat16_tc_flops = Some(1e9);
        spec.gpu.fp8_tc_flops = Some(1e9);
        spec.gpu.fp4_tc_flops = Some(1e9);
        spec.gpu.fp32_flops = Some(1e9);
        spec
    }

    fn attention(role: &str, ratio: u32) -> Dsv41AttentionOp {
        Dsv41AttentionOp {
            name: "attention".into(),
            is_context: true,
            role: role.into(),
            compress_ratio: ratio,
            hidden_size: 8,
            num_heads: 2,
            head_dim: 4,
            q_lora_rank: 4,
            o_lora_rank: 2,
            o_groups: 1,
            index_n_heads: 4,
            index_head_dim: 2,
            index_topk: 3,
            window_size: 128,
            candidate_limit: 0,
            is_candidate_source: false,
            bounded_prefill: false,
            gemm_quant_mode: GemmQuantMode::Bfloat16,
            fmha_quant_mode: FmhaQuantMode::Bfloat16,
            kv_cache_layout: Dsv41KvCacheLayout::LogicalFp4,
        }
    }

    fn assert_components(result: PerformanceResult, flops: f64, bytes: f64) {
        let components = result.sol.unwrap();
        assert!((components.math_ms - flops / 1e6).abs() < 1e-10);
        assert!((components.mem_ms - bytes / 1e3).abs() < 1e-10);
        assert_eq!(result.source, Source::Sol);
    }

    #[test]
    fn all_attention_roles_have_independent_numeric_rooflines_and_weights() {
        // Hand-expanded B=1,Q=4,P=0 ledger. Base projections: 1024 FLOPs,
        // 704 bytes; the SWA kernel has 10 causal pairs but four KV rows.
        // Full ratio-two owns TWO 8x4 BF16 compressor matrices, not one.
        for (role, ratio, weights, flops, bytes) in [
            ("swa", 0, 280.0, 1344.0, 848.0),
            ("reuse", 2, 280.0, 1472.0, 857.0),
            ("reindex", 2, 408.0, 2112.0, 1308.125),
            ("full", 2, 552.0, 2656.0, 1690.75),
            ("full", 1, 488.0, 2720.0, 1654.75),
        ] {
            let op = attention(role, ratio);
            assert_eq!(op.weight_bytes(), weights, "{role}, ratio={ratio}");
            assert_components(op.sol(&unit_spec(), 1.0, 4.0, 0.0).unwrap(), flops, bytes);
        }
    }

    #[test]
    fn serialized_full_attention_rejects_zero_compression_in_both_phases() {
        let spec = unit_spec();
        for is_context in [true, false] {
            let mut value = serde_json::to_value(attention("full", 2)).unwrap();
            value["compress_ratio"] = serde_json::json!(0);
            value["is_context"] = serde_json::json!(is_context);
            let malformed: Dsv41AttentionOp = serde_json::from_value(value).unwrap();
            for (batch, sequence) in [(1.0, 128.0), (0.0, 128.0), (1.0, 0.0)] {
                let error = malformed.sol(&spec, batch, sequence, 32.0).unwrap_err();
                assert!(matches!(error, AicError::ModelConfig(ref message)
                    if message.contains("positive compress_ratio")));
            }
            // Ratio zero is valid for the uncompressed sliding-window role.
            let mut sliding_window = malformed;
            sliding_window.role = "swa".into();
            let result = sliding_window.sol(&spec, 1.0, 128.0, 32.0).unwrap();
            assert!(result.latency_ms.is_finite());
            assert!(result.latency_ms > 0.0);
        }
    }

    #[test]
    fn window_flops_use_pairs_but_hbm_uses_unique_rows() {
        let spec = unit_spec();
        let mut op = attention("swa", 0);
        let short = op.sol(&spec, 1.0, 4.0, 0.0).unwrap().sol.unwrap();
        let prefix = op.sol(&spec, 1.0, 4.0, 1000.0).unwrap().sol.unwrap();
        // Four queries now attend 512 pairs, but load just 127 additional rows.
        assert!((prefix.math_ms - short.math_ms - 32.0 * (512.0 - 10.0) / 1e6).abs() < 1e-12);
        assert!((prefix.mem_ms - short.mem_ms - 127.0 * 4.0 / 1e3).abs() < 1e-12);
        op.bounded_prefill = true;
        assert_eq!(op.sol(&spec, 1.0, 4.0, 1000.0).unwrap().sol.unwrap(), short);
        // Remove heads/projections so this isolates the 4096-row SWA traffic.
        op.hidden_size = 0;
        op.q_lora_rank = 0;
        op.o_lora_rank = 0;
        op.num_heads = 0;
        op.o_groups = 0;
        op.head_dim = 512;
        // The empty hidden->KV projection writes each KV row in BF16;
        // attention then reads each row once in FP8: (2+1)*4096*512 bytes.
        assert_components(op.sol(&spec, 1.0, 4096.0, 0.0).unwrap(), 0.0, 6_291_456.0);
    }

    #[test]
    fn candidate_eligibility_never_caps_full_context_scoring() {
        let spec = unit_spec();
        let mut op = attention("reindex", 1);
        op.is_context = false;
        op.index_topk = 3; // Attention selection has saturated well before these inputs.
        op.candidate_limit = 16_384;
        for context in [16_384.0, 16_385.0, 131_072.0] {
            let limited = op.sol(&spec, 1.0, context, 0.0).unwrap();
            op.candidate_limit = 0;
            assert_eq!(limited, op.sol(&spec, 1.0, context, 0.0).unwrap());
            op.candidate_limit = 16_384;
        }
        let lo = op.sol(&spec, 1.0, 16_384.0, 0.0).unwrap().sol.unwrap();
        let hi = op.sol(&spec, 1.0, 131_072.0, 0.0).unwrap().sol.unwrap();
        // Per extra key: 2*4 heads*2 dimensions =16 FLOPs; 2*17/32=1.0625
        // packed K bytes, plus4 score-write and4 score-read bytes. All other
        // geometry is saturated and constant in this independent decode ledger.
        let extra = 131_072.0 - 16_384.0;
        assert!((hi.math_ms - lo.math_ms - 16.0 * extra / 1e6).abs() < 1e-10);
        assert!((hi.mem_ms - lo.mem_ms - 9.0625 * extra / 1e3).abs() < 1e-10);
    }

    #[test]
    fn prefixed_prefill_scores_across_candidate_boundary() {
        let spec = unit_spec();
        let mut op = attention("reindex", 1);
        op.candidate_limit = 16_384;
        // Three actual queries with a long cached prefix. Their sparse/window
        // attention is saturated; the full index matrix still grows with end.
        for end in [16_384.0, 16_385.0, 131_072.0] {
            let limited = op.sol(&spec, 1.0, 3.0, end - 3.0).unwrap();
            op.candidate_limit = 0;
            assert_eq!(limited, op.sol(&spec, 1.0, 3.0, end - 3.0).unwrap());
            op.candidate_limit = 16_384;
        }
        let lo = op.sol(&spec, 1.0, 3.0, 16_381.0).unwrap().sol.unwrap();
        let hi = op.sol(&spec, 1.0, 3.0, 16_382.0).unwrap().sol.unwrap();
        // One extra K row: 3 queries *16 FLOPs; 1.0625 K bytes plus
        // 3*(4-byte score write +4-byte score read). Prefix is not re-executed.
        assert!((hi.math_ms - lo.math_ms - 48.0 / 1e6).abs() < 1e-10);
        assert!((hi.mem_ms - lo.mem_ms - 25.0625 / 1e3).abs() < 1e-10);
    }

    #[test]
    fn sglang_physical_reads_and_publication_count_one_persistent_record() {
        let spec = unit_spec();
        let mut op = attention("full", 2);
        op.is_context = false;
        op.head_dim = 512;
        op.index_head_dim = 128;
        // Relative to the separately hand-tested logical ledger: every token
        // stores its SWA row (1024-byte source read +584-byte write), the128
        // resident window rows add72 bytes each, and the three selected main
        // entries add296 each. Even tokens publish one more main row (+296).
        for (context, published) in [(129.0, 0.0), (130.0, 1.0)] {
            let logical = op.sol(&spec, 1.0, context, 0.0).unwrap().sol.unwrap();
            op.kv_cache_layout = Dsv41KvCacheLayout::SglangFp8Bf16;
            let physical = op.sol(&spec, 1.0, context, 0.0).unwrap().sol.unwrap();
            assert_eq!(physical.math_ms, logical.math_ms);
            let added = 1024.0 + 584.0 + 128.0 * 72.0 + (3.0 + published) * 296.0;
            assert!((physical.mem_ms - logical.mem_ms - added / 1e3).abs() < 1e-10);
            op.kv_cache_layout = Dsv41KvCacheLayout::LogicalFp4;
        }
        op.role = "swa".into();
        op.compress_ratio = 0;
        op.is_context = true;
        op.hidden_size = 0;
        op.q_lora_rank = 0;
        op.o_lora_rank = 0;
        op.num_heads = 0;
        op.o_groups = 0;
        op.kv_cache_layout = Dsv41KvCacheLayout::SglangFp8Bf16;
        // Per query: BF16 projection write1024, store read1024/write584,
        // attention read584 =3216 bytes. No persistent packed FP4 record.
        assert_components(op.sol(&spec, 1.0, 4096.0, 0.0).unwrap(), 0.0, 13_172_736.0);
    }

    #[test]
    fn physical_layout_rejects_unqualified_geometry_and_json_defaults_are_theoretical() {
        let mut op = attention("full", 2);
        let mut json = serde_json::to_value(&op).unwrap();
        json.as_object_mut().unwrap().remove("kv_cache_layout");
        let legacy: Dsv41AttentionOp = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(legacy.kv_cache_layout, Dsv41KvCacheLayout::LogicalFp4);
        json["kv_cache_layout"] = serde_json::json!("unknown");
        assert!(serde_json::from_value::<Dsv41AttentionOp>(json).is_err());
        op.kv_cache_layout = Dsv41KvCacheLayout::SglangFp8Bf16;
        assert!(op.sol(&unit_spec(), 1.0, 128.0, 0.0).is_err());
        op.head_dim = 512;
        op.index_head_dim = 128;
        for ratio in [4, 128] {
            op.compress_ratio = ratio;
            assert!(op.sol(&unit_spec(), 1.0, 128.0, 0.0).is_err());
        }
    }

    #[test]
    fn mhc_prices_both_residual_sites_at_scalar_fp32_rate() {
        let op = Dsv41MhcOp {
            name: "mhc".into(),
            hidden_size: 5120,
            hc_mult: 4,
            sinkhorn_iters: 20,
        };
        assert_eq!(op.weight_bytes(), 3_932_376.0);
        assert_components(op.sol(&unit_spec(), 2.0).unwrap(), 4_753_280.0, 4_260_440.0);
        let mut spec = unit_spec();
        for rate in [
            None,
            Some(0.0),
            Some(-1.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            spec.gpu.fp32_flops = rate;
            assert!(matches!(
                op.sol(&spec, 1.0),
                Err(AicError::MissingSystemFlops(_))
            ));
        }
    }

    #[test]
    fn engram_production_weights_include_exact_tables_and_replicated_projections() {
        for (tp_size, expected) in [
            (1, 203_073_076_240.0),
            (4, 51_004_552_072.0),
            (8, 25_659_798_088.0),
        ] {
            let total: f64 = [384_006_168, 384_016_682]
                .into_iter()
                .map(|num_embeddings| {
                    Dsv41EngramOp {
                        name: "engram".into(),
                        num_embeddings,
                        head_dim: 256,
                        hash_columns: 24,
                        hidden_size: 5120,
                        hc_mult: 4,
                        tp_size,
                    }
                    .weight_bytes()
                })
                .sum();
            assert_eq!(total, expected);
        }
        let op = Dsv41EngramOp {
            name: "small".into(),
            num_embeddings: 64,
            head_dim: 32,
            hash_columns: 2,
            hidden_size: 8,
            hc_mult: 4,
            tp_size: 4,
        };
        assert_eq!(op.weight_bytes(), 3218.5);
        // Lookup 289 bytes + projection 2978.5 bytes + gate 512 bytes.
        assert_components(op.sol(&unit_spec(), 2.0).unwrap(), 10240.0, 3779.5);
    }

    #[test]
    fn block32_linear_counts_partial_scale_tiles_and_numeric_roofline() {
        let op = Dsv41LinearOp {
            name: "linear".into(),
            n: 33,
            k: 65,
            quant_mode: GemmQuantMode::Fp8Block,
        };
        assert_eq!(op.weight_bytes(), 2151.0); // 2145 weights + 2*3 scale bytes
        assert_components(op.sol(&unit_spec(), 2.0).unwrap(), 8580.0, 2543.0);
    }

    #[test]
    fn new_operators_reject_unmeasured_modes_and_empty_work_is_zero() {
        let attn = attention("full", 2);
        let mhc = Dsv41MhcOp {
            name: "mhc".into(),
            hidden_size: 8,
            hc_mult: 4,
            sinkhorn_iters: 20,
        };
        let engram = Dsv41EngramOp {
            name: "engram".into(),
            num_embeddings: 64,
            head_dim: 32,
            hash_columns: 2,
            hidden_size: 8,
            hc_mult: 4,
            tp_size: 4,
        };
        let linear = Dsv41LinearOp {
            name: "linear".into(),
            n: 32,
            k: 32,
            quant_mode: GemmQuantMode::Fp8Block,
        };
        for mode in [DatabaseMode::Silicon, DatabaseMode::Empirical] {
            let db = test_db(mode);
            let ctx = RuntimeContext {
                batch_size: 1,
                s: 4,
                prefix: 0,
                num_tokens: 4,
                ..RuntimeContext::default()
            };
            for result in [
                attn.query(&db, &ctx),
                mhc.query(&db, 4),
                engram.query(&db, 4),
                linear.query(&db, 4),
            ] {
                match mode {
                    DatabaseMode::Silicon => {
                        assert!(matches!(result, Err(AicError::PerfDatabase(_))))
                    }
                    _ => assert!(matches!(result, Err(AicError::EmpiricalNotImplemented(_)))),
                }
            }
        }
        let spec = unit_spec();
        for result in [
            attn.sol(&spec, 1.0, 0.0, 0.0),
            mhc.sol(&spec, 0.0),
            engram.sol(&spec, 0.0),
            linear.sol(&spec, 0.0),
        ] {
            assert_eq!(result.unwrap().latency_ms, 0.0);
        }
    }
    #[test]
    fn deserialized_attention_roles_are_validated_before_empty_work() {
        for role in ["ful", "", "FULL", "swa "] {
            let mut json = serde_json::to_value(attention("full", 2)).unwrap();
            json["role"] = serde_json::Value::String(role.into());
            let op: Dsv41AttentionOp = serde_json::from_value(json).unwrap();
            for (batch, seq) in [(1.0, 4.0), (0.0, 4.0), (1.0, 0.0)] {
                assert!(matches!(
                    op.sol(&unit_spec(), batch, seq, 0.0),
                    Err(AicError::ModelConfig(_))
                ));
            }
        }
        for role in ["swa", "full", "reindex", "reuse"] {
            let json = serde_json::to_value(attention(role, 2)).unwrap();
            let op: Dsv41AttentionOp = serde_json::from_value(json.clone()).unwrap();
            assert_eq!(serde_json::to_value(&op).unwrap(), json);
            assert!(op.sol(&unit_spec(), 1.0, 4.0, 0.0).is_ok());
            assert_eq!(op.sol(&unit_spec(), 0.0, 4.0, 0.0).unwrap().latency_ms, 0.0);
        }
    }

    #[test]
    fn bounded_scope_preserves_absolute_position() {
        let stage = Dsv41StageOp {
            name: "tail".into(),
            is_context: true,
            decoder_replay: true,
            bounded: true,
            window_size: 128,
            children: vec![],
        };
        assert_eq!(stage.scope(256.0, 1000.0), (128.0, 1128.0));
        assert_eq!(stage.scope(3.0, 1000.0), (3.0, 1000.0));
    }
    #[test]
    fn compression_publication_and_saturation() {
        assert_eq!(compressed_pairs(4.0, 0.0, 2.0, 512.0), 4.0);
        assert_eq!(compressed_pairs(2.0, 1024.0, 2.0, 512.0), 1024.0);
        assert_eq!(compressed_pairs(1.0, 128.0, 2.0, 512.0), 64.0);
    }
}
