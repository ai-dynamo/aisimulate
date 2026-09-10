// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek-V4.1 text AR operator and phase contracts.
//!
//! Architecture source: deepseek-ai/DeepSeek-V4.1-Flash, revision
//! fb2764a5cf321eaa5070ca8f9e892818f477c16d, config.json / inference/model.py
//! and DeepSeek_V41_Tech_Report.pdf (MIT, Copyright (c) 2023 DeepSeek).
//! These are performance-model adaptations; see THIRD_PARTY_NOTICES.md.

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
}

impl Dsv41AttentionOp {
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
        let end = if self.is_context { prefix + s } else { s };
        let ratio = self.compress_ratio as f64;
        let compressed_len = if ratio > 0.0 {
            (end / ratio).floor()
        } else {
            0.0
        };
        if self.role == "full" {
            let mult = if self.compress_ratio > 1 { 2.0 } else { 1.0 };
            result = result.plus(mm(h, d, tokens * mult, 2.0, bf16));
            let produced = if self.is_context {
                (end / ratio).floor() - (prefix / ratio).floor()
            } else {
                (end / ratio).floor() - ((end - 1.0).max(0.0) / ratio).floor()
            };
            let ihd = self.index_head_dim as f64;
            result = result.plus(mm(d, ihd, batch * produced, 2.0, bf16));
            // Compressed main KV packs E2M1 + E4M3/16; index packs E2M1 + UE8M0/32.
            let packed = d * 0.5625 + ihd * 0.53125;
            result = result.plus(leaf(spec, 0.0, batch * produced * (d * 2.0 + packed), bf16));
        }
        if self.role == "full" || self.role == "reindex" {
            let (inh, ihd) = (self.index_n_heads as f64, self.index_head_dim as f64);
            result = result
                .plus(mm(q, inh * ihd, tokens, w8, fp8))
                .plus(mm(h, inh, tokens, 2.0, bf16));
            let index_len = if self.candidate_limit > 0 {
                compressed_len.min(self.candidate_limit as f64)
            } else {
                compressed_len
            };
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
        result = result.plus(leaf(
            spec,
            4.0 * n * d * (wp + cp),
            (wp * d + cp * d * 0.5625) + tokens * n * d * 4.0,
            attn,
        ));
        Ok(result)
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
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
