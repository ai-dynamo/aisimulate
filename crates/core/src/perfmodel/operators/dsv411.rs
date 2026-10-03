// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek-V4.1 `dsv411` family: a second, independently scoped decomposition of
//! the text decoder that coexists with the `dsv41` family.
//!
//! Differences from `dsv41`: the index scoring/selection is its own component
//! (`indexer`), physical KV byte layouts and the index scoring precision are
//! explicit per-backend fields chosen by the Python model from measured runtime
//! facts (not inferred from an SM switch), measured rows are keyed by explicit
//! numeric columns and interpolated over (query, kv_len). A stage sums its children
//! sequentially; module-level rows already contain each module's own intra-module
//! overlap, cross-module overlap is the whole-forward (FPM) model's concern.
//!
//! Architecture source: deepseek-ai/DeepSeek-V4.1-Flash, revision
//! fb2764a5cf321eaa5070ca8f9e892818f477c16d (config.json). Roofline arithmetic is
//! an independent analytical expression of that architecture; serving layout
//! facts (584-byte fp8_ds_mla rows, 68/132-byte index keys, SWA window 128) were
//! measured on sglang v0.5.21 and vLLM v0.30.0 (H20, 2026-10-02).

use serde::{Deserialize, Serialize};

use crate::common::enums::{DatabaseMode, FmhaQuantMode, GemmQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::{SystemSpec, quant_tc_flops};
use crate::operators::base::{PerformanceResult, SolComponents, Source};
use crate::operators::op::{Op, RuntimeContext};
use crate::perf_database::PerfDatabase;

pub const COMPONENT_ATTENTION_CORE: &str = "attention_core";
pub const COMPONENT_INDEXER: &str = "indexer";
pub const COMPONENT_ENGRAM: &str = "engram";
pub const COMPONENT_MHC: &str = "mhc";
pub const COMPONENT_SHARED_LINEAR: &str = "shared_linear";

fn leaf(spec: &SystemSpec, flops: f64, bytes: f64, rate: f64) -> PerformanceResult {
    PerformanceResult::sol(SolComponents::new(
        flops / rate * 1e3,
        bytes / spec.gpu.mem_bw * 1e3,
    ))
}

fn zero() -> PerformanceResult {
    PerformanceResult::sol(SolComponents::new(0.0, 0.0))
}

/// Sum floor(t/r) for t in 1..=n (continuous tail) — compressed pairs visible
/// to a growing query under an integer publication boundary.
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

/// Measured coordinates of one query: `(batch, query, kv_len)`.
/// context: query = new tokens, kv_len = cached prefix; generation: query = 1,
/// kv_len = absolute KV position (prefix + generated + 1, the engine's `s`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coordinates {
    pub batch: u32,
    pub query: u32,
    pub kv_len: u32,
}

impl Coordinates {
    pub fn attention(is_context: bool, ctx: &RuntimeContext) -> Self {
        if is_context {
            Coordinates { batch: ctx.batch_size, query: ctx.s, kv_len: ctx.prefix }
        } else {
            Coordinates { batch: ctx.batch_size, query: 1, kv_len: ctx.s }
        }
    }
    pub fn tokens(tokens: u32) -> Self {
        Coordinates { batch: 1, query: tokens, kv_len: 0 }
    }
}

/// Physical bytes per cached entry; measured runtime facts, never inferred.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv411KvLayout {
    /// One sliding-window row (both backends: 584 fp8_ds_mla).
    pub window_entry_bytes: f64,
    /// One compressed main row (both backends: 584).
    pub main_entry_bytes: f64,
    /// One index-K entry (sglang fp4: 68; vLLM fp8: 132).
    pub index_entry_bytes: f64,
}

fn block_scale_overhead(mode: GemmQuantMode) -> f64 {
    // V4.1 checkpoints quantize in 32x32 blocks with one UE8M0 byte per block.
    if mode == GemmQuantMode::Fp8Block { 1.0 / 1024.0 } else { 0.0 }
}

/// Shared SILICON/HYBRID/SOL dispatch for every measured leaf of this family.
fn query_leaf(
    db: &PerfDatabase,
    component: &str,
    structure: &str,
    is_context: bool,
    tp_size: u32,
    coords: Coordinates,
    sol: &dyn Fn(f64, f64) -> Result<PerformanceResult, AicError>,
) -> Result<PerformanceResult, AicError> {
    if coords.batch == 0 || coords.query == 0 {
        return Ok(zero());
    }
    if matches!(db.database_mode, DatabaseMode::Sol | DatabaseMode::SolFull) {
        return sol(f64::from(coords.query), f64::from(coords.kv_len));
    }
    if db.database_mode == DatabaseMode::Empirical {
        return Err(AicError::EmpiricalNotImplemented(format!(
            "dsv411 {component} has no empirical calibration"
        )));
    }
    let measured = db.dsv411.query(
        component,
        structure,
        is_context,
        tp_size,
        coords,
        &|query, kv| sol(query, kv).map(|result| result.latency_ms),
    )?;
    match measured {
        Some(value) => Ok(PerformanceResult::with_energy(value.latency, value.energy, Source::Silicon)),
        None if db.database_mode == DatabaseMode::Hybrid => {
            sol(f64::from(coords.query), f64::from(coords.kv_len))
        }
        None => Err(AicError::PerfDatabase(format!(
            "dsv411 {component} has no measured SILICON data for {structure} batch={} query={} kv_len={}",
            coords.batch, coords.query, coords.kv_len
        ))),
    }
}

/// Canonical structural key shared by the operator and the table loader: every
/// exact-match column, in a fixed order, joined by `|`.
pub fn structure_key(parts: &[(&str, String)]) -> String {
    parts
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("|")
}

// ---------------------------------------------------------------------------
// attention_core: projections, compressor (kv sources), SWA + sparse core, o_proj.
// Excludes the indexer (own component) and the TP output all-reduce.
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv411AttentionCoreOp {
    pub name: String,
    pub is_context: bool,
    pub role: String,
    pub compress_ratio: u32,
    pub tp_size: u32,
    pub hidden_size: u32,
    /// Rank-local heads and output groups.
    pub num_heads: u32,
    pub head_dim: u32,
    pub q_lora_rank: u32,
    pub o_lora_rank: u32,
    pub o_groups: u32,
    pub window_size: u32,
    pub index_topk: u32,
    pub index_head_dim: u32,
    pub kv_layout: Dsv411KvLayout,
    pub gemm_quant_mode: GemmQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
}

impl Dsv411AttentionCoreOp {
    fn validate(&self) -> Result<(), AicError> {
        if !matches!(self.role.as_str(), "swa" | "full" | "reindex" | "reuse") {
            return Err(AicError::ModelConfig(format!(
                "dsv411 attention role must be swa/full/reindex/reuse; got {:?}",
                self.role
            )));
        }
        if (self.role == "swa") != (self.compress_ratio == 0) || self.compress_ratio > 2 {
            return Err(AicError::ModelConfig("dsv411 attention role/compress_ratio mismatch".into()));
        }
        if self.window_size == 0 || self.tp_size == 0 || self.num_heads == 0 {
            return Err(AicError::ModelConfig("dsv411 attention requires positive window/tp/heads".into()));
        }
        Ok(())
    }

    pub fn structure(&self) -> String {
        structure_key(&[
            ("role", self.role.clone()),
            ("compress_ratio", self.compress_ratio.to_string()),
            ("num_heads", self.num_heads.to_string()),
            ("head_dim", self.head_dim.to_string()),
            ("q_lora_rank", self.q_lora_rank.to_string()),
            ("o_lora_rank", self.o_lora_rank.to_string()),
            ("o_groups", self.o_groups.to_string()),
            ("window_size", self.window_size.to_string()),
            ("index_topk", self.index_topk.to_string()),
            ("quant_mode", self.gemm_quant_mode.name().to_string()),
        ])
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
        let w8 = self.gemm_quant_mode.mapping().memory + block_scale_overhead(self.gemm_quant_mode);
        // fused wqa/wkv (replicated), wq_b, wo_a (BF16 after dequant on sm90
        // emulation is a runtime detail; weights stay FP8 resident), wo_b.
        let mut bytes = (h * q + q * n * d + h * d + g * o * h) * w8 + n * d * o * 2.0;
        if self.role == "full" {
            // compressor projections (two for ratio 2) and the index-K projection
            bytes += h * d * 2.0 * if self.compress_ratio > 1 { 2.0 } else { 1.0 };
            bytes += d * self.index_head_dim as f64 * 2.0;
        }
        bytes + n * 4.0 + (q + d) * 2.0
    }

    /// Roofline in ms. `query` new tokens per request, `kv_len` cached prefix
    /// (context) or absolute KV length (generation).
    pub fn sol(&self, spec: &SystemSpec, batch: f64, query: f64, kv_len: f64) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        if batch <= 0.0 || query <= 0.0 {
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
        let tokens = batch * query;
        let bf16 = quant_tc_flops(spec, GemmQuantMode::Bfloat16.mapping())?;
        let gemm = quant_tc_flops(spec, self.gemm_quant_mode.mapping())?;
        let attn = quant_tc_flops(spec, self.fmha_quant_mode.mapping())?;
        let w8 = self.gemm_quant_mode.mapping().memory + block_scale_overhead(self.gemm_quant_mode);
        let mm = |a: f64, b: f64, nt: f64, weight: f64, rate: f64| {
            leaf(spec, 2.0 * nt * a * b, a * b * weight + nt * (a + b) * 2.0, rate)
        };
        // projections: wqa+wkv (fused, replicated), wq_b, wo_a (bmm), wo_b
        let mut result = mm(h, q, tokens, w8, gemm)
            .plus(mm(q, n * d, tokens, w8, gemm))
            .plus(mm(h, d, tokens, w8, gemm))
            .plus(leaf(spec, 2.0 * tokens * n * d * o, n * d * o * 2.0 + tokens * (n * d + g * o) * 2.0, bf16))
            .plus(mm(g * o, h, tokens, w8, gemm));
        // fused qnorm/rope/kv insert: reads the BF16 projection, writes one SWA row
        result = result.plus(leaf(spec, 0.0, tokens * (d * 2.0 + self.kv_layout.window_entry_bytes), bf16));
        let end = if self.is_context { kv_len + query } else { kv_len };
        let prefix = if self.is_context { kv_len } else { end.max(1.0) - 1.0 };
        let ratio = self.compress_ratio as f64;
        let compressed_len = if ratio > 0.0 { (end / ratio).floor() } else { 0.0 };
        if self.role == "full" {
            // compressor: projection(s) + published rows (main + index-K)
            let mult = if self.compress_ratio > 1 { 2.0 } else { 1.0 };
            result = result.plus(mm(h, d, tokens, 2.0, bf16).scaled(mult));
            let produced = if self.is_context {
                (end / ratio).floor() - (prefix / ratio).floor()
            } else {
                (end / ratio).floor() - (prefix / ratio).floor()
            };
            let ihd = self.index_head_dim as f64;
            result = result.plus(mm(d, ihd, batch * produced, 2.0, bf16));
            let packed = self.kv_layout.main_entry_bytes + self.kv_layout.index_entry_bytes;
            result = result.plus(leaf(spec, 0.0, batch * produced * (d * 2.0 + packed), bf16));
        }
        // sparse core: SWA pairs + selected compressed pairs
        let wp = if self.is_context {
            batch * limited_pairs(query, prefix, self.window_size as f64)
        } else {
            batch * end.min(self.window_size as f64)
        };
        let cp = if self.compress_ratio == 0 {
            0.0
        } else if self.is_context {
            batch * compressed_pairs(query, prefix, ratio, self.index_topk as f64)
        } else {
            batch * compressed_len.min(self.index_topk as f64)
        };
        let window_rows = if self.is_context {
            batch * (query + prefix.min((self.window_size as f64 - 1.0).max(0.0)))
        } else {
            batch * end.min(self.window_size as f64)
        };
        result = result.plus(leaf(
            spec,
            4.0 * n * d * (wp + cp),
            window_rows * self.kv_layout.window_entry_bytes + cp * self.kv_layout.main_entry_bytes + tokens * n * d * 4.0,
            attn,
        ));
        Ok(result)
    }

    pub fn query(&self, db: &PerfDatabase, ctx: &RuntimeContext) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        let coords = Coordinates::attention(self.is_context, ctx);
        query_leaf(db, COMPONENT_ATTENTION_CORE, &self.structure(), self.is_context, self.tp_size, coords, &|q, kv| {
            self.sol(&db.system_spec, f64::from(coords.batch), q, kv)
        })
    }
}

// ---------------------------------------------------------------------------
// indexer: index projections, scoring over the compressed context, top-k selection
// (+ candidate-block scoring on the candidate source). Present on index sources only.
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv411IndexerOp {
    pub name: String,
    pub is_context: bool,
    pub compress_ratio: u32,
    pub tp_size: u32,
    pub hidden_size: u32,
    pub q_lora_rank: u32,
    /// Replicated across TP on both runtimes (all heads per rank).
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub is_candidate_source: bool,
    pub candidate_limit: u32,
    /// Bytes per index-K entry read while scoring (sglang 68, vLLM 132).
    pub index_entry_bytes: f64,
    /// Tensor-core class the runtime scores with (sglang/H20 bf16, vLLM/H20 fp8).
    pub scoring_quant_mode: GemmQuantMode,
    /// vLLM skips scoring when every compressed position fits in top-k.
    pub skip_within_topk: bool,
    pub gemm_quant_mode: GemmQuantMode,
}

impl Dsv411IndexerOp {
    fn validate(&self) -> Result<(), AicError> {
        if self.compress_ratio == 0 || self.compress_ratio > 2 || self.index_n_heads == 0 || self.tp_size == 0 {
            return Err(AicError::ModelConfig("dsv411 indexer requires ratio 1/2, heads and tp".into()));
        }
        Ok(())
    }

    pub fn structure(&self) -> String {
        structure_key(&[
            ("compress_ratio", self.compress_ratio.to_string()),
            ("index_n_heads", self.index_n_heads.to_string()),
            ("index_head_dim", self.index_head_dim.to_string()),
            ("index_topk", self.index_topk.to_string()),
            ("is_candidate_source", self.is_candidate_source.to_string()),
            ("candidate_limit", self.candidate_limit.to_string()),
            ("q_lora_rank", self.q_lora_rank.to_string()),
            ("quant_mode", self.gemm_quant_mode.name().to_string()),
        ])
    }

    pub fn weight_bytes(&self) -> f64 {
        let w8 = self.gemm_quant_mode.mapping().memory + block_scale_overhead(self.gemm_quant_mode);
        let (q, inh, ihd, h) = (
            self.q_lora_rank as f64,
            self.index_n_heads as f64,
            self.index_head_dim as f64,
            self.hidden_size as f64,
        );
        q * inh * ihd * w8 + h * inh * 2.0
    }

    pub fn sol(&self, spec: &SystemSpec, batch: f64, query: f64, kv_len: f64) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        if batch <= 0.0 || query <= 0.0 {
            return Ok(zero());
        }
        let (q, inh, ihd, h) = (
            self.q_lora_rank as f64,
            self.index_n_heads as f64,
            self.index_head_dim as f64,
            self.hidden_size as f64,
        );
        let tokens = batch * query;
        let bf16 = quant_tc_flops(spec, GemmQuantMode::Bfloat16.mapping())?;
        let gemm = quant_tc_flops(spec, self.gemm_quant_mode.mapping())?;
        let w8 = self.gemm_quant_mode.mapping().memory + block_scale_overhead(self.gemm_quant_mode);
        let mm = |a: f64, b: f64, nt: f64, weight: f64, rate: f64| {
            leaf(spec, 2.0 * nt * a * b, a * b * weight + nt * (a + b) * 2.0, rate)
        };
        let mut result = mm(q, inh * ihd, tokens, w8, gemm).plus(mm(h, inh, tokens, 2.0, bf16));
        let end = if self.is_context { kv_len + query } else { kv_len };
        let compressed_len = (end / self.compress_ratio as f64).floor();
        if self.skip_within_topk && compressed_len <= self.index_topk as f64 {
            // every candidate is selected without scoring (vLLM attention.py:1195-1217)
            return Ok(result);
        }
        let scoring = quant_tc_flops(spec, self.scoring_quant_mode.mapping())?;
        let query_bytes = self.scoring_quant_mode.mapping().memory.max(1.0);
        result = result.plus(leaf(
            spec,
            2.0 * tokens * inh * ihd * compressed_len,
            batch * compressed_len * self.index_entry_bytes + tokens * inh * ihd * query_bytes + tokens * compressed_len * 4.0,
            scoring,
        ));
        let candidate_bytes = if self.is_candidate_source { tokens * compressed_len / 8.0 * 4.0 } else { 0.0 };
        result = result.plus(leaf(
            spec,
            0.0,
            tokens * (compressed_len * 4.0 + self.index_topk as f64 * 4.0) + candidate_bytes,
            bf16,
        ));
        Ok(result)
    }

    pub fn query(&self, db: &PerfDatabase, ctx: &RuntimeContext) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        let coords = Coordinates::attention(self.is_context, ctx);
        query_leaf(db, COMPONENT_INDEXER, &self.structure(), self.is_context, self.tp_size, coords, &|q, kv| {
            self.sol(&db.system_spec, f64::from(coords.batch), q, kv)
        })
    }
}

// ---------------------------------------------------------------------------
// engram: lookup + wkv projection + gate; the TP collective is outside.
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv411EngramOp {
    pub name: String,
    pub is_context: bool,
    pub num_embeddings: u64,
    pub head_dim: u32,
    pub hash_columns: u32,
    pub hidden_size: u32,
    pub hc_mult: u32,
    pub tp_size: u32,
    /// "row" (sglang: rows sharded, all-reduce) or "head" (vLLM: hash heads sharded, all-gather).
    pub sharding: String,
    pub gemm_quant_mode: GemmQuantMode,
}

impl Dsv411EngramOp {
    pub fn structure(&self) -> String {
        structure_key(&[
            ("num_embeddings", self.num_embeddings.to_string()),
            ("head_dim", self.head_dim.to_string()),
            ("hash_columns", self.hash_columns.to_string()),
            ("hc_mult", self.hc_mult.to_string()),
            ("sharding", self.sharding.clone()),
            ("quant_mode", self.gemm_quant_mode.name().to_string()),
        ])
    }
    pub fn weight_bytes(&self) -> f64 {
        let rows = self.num_embeddings.div_ceil(self.tp_size as u64) as f64;
        let d = self.head_dim as f64;
        let projection = self.hash_columns as f64 * d * self.hidden_size as f64 * (self.hc_mult + 1) as f64;
        rows * (d + d / 32.0)
            + projection * (self.gemm_quant_mode.mapping().memory + block_scale_overhead(self.gemm_quant_mode))
            + 2.0 * self.hc_mult as f64 * self.hidden_size as f64 * 2.0
    }
    pub fn sol(&self, spec: &SystemSpec, tokens: f64) -> Result<PerformanceResult, AicError> {
        if tokens <= 0.0 {
            return Ok(zero());
        }
        let input = self.hash_columns as f64 * self.head_dim as f64;
        let output = self.hidden_size as f64 * (self.hc_mult + 1) as f64;
        let rate = quant_tc_flops(spec, self.gemm_quant_mode.mapping())?;
        let w8 = self.gemm_quant_mode.mapping().memory + block_scale_overhead(self.gemm_quant_mode);
        // each rank gathers 1/tp of the hash columns (row or head sharding reads the same bytes)
        let lookup = leaf(spec, 0.0, tokens * input * ((1.0 + 1.0 / 32.0) / self.tp_size as f64 + 2.0), rate);
        let projection = leaf(spec, 2.0 * tokens * input * output, input * output * w8 + tokens * (input + output) * 2.0, rate);
        let gate = leaf(spec, 0.0, tokens * self.hidden_size as f64 * self.hc_mult as f64 * 8.0, rate);
        Ok(lookup.plus(projection).plus(gate))
    }
    pub fn query(&self, db: &PerfDatabase, tokens: u32) -> Result<PerformanceResult, AicError> {
        query_leaf(db, COMPONENT_ENGRAM, &self.structure(), self.is_context, self.tp_size, Coordinates::tokens(tokens), &|q, _| {
            self.sol(&db.system_spec, q)
        })
    }
}

// ---------------------------------------------------------------------------
// mhc: both post+pre-mix sites of a layer, norm fused (as both runtimes execute it).
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv411MhcOp {
    pub name: String,
    pub is_context: bool,
    pub hidden_size: u32,
    pub hc_mult: u32,
    pub sinkhorn_iters: u32,
    pub tp_size: u32,
}

impl Dsv411MhcOp {
    pub fn structure(&self) -> String {
        structure_key(&[
            ("hidden_size", self.hidden_size.to_string()),
            ("hc_mult", self.hc_mult.to_string()),
            ("sinkhorn_iters", self.sinkhorn_iters.to_string()),
        ])
    }
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
        let ops = 2.0 * tokens * (2.0 * hc * h * mixes + (hc * hc + 2.0 * hc) * self.sinkhorn_iters as f64 + 2.0 * hc * hc * h + 2.0 * hc * h);
        // residual stream read+write per site (BF16) + fused RMSNorm output + fp32 mix stats
        let bytes = self.weight_bytes() + 2.0 * tokens * hc * h * 2.0 * 2.0 + 2.0 * tokens * h * 2.0 + 2.0 * tokens * mixes * 4.0;
        let fp32 = spec
            .gpu
            .fp32_flops
            .filter(|v| v.is_finite() && *v > 0.0)
            .ok_or_else(|| AicError::MissingSystemFlops("dsv411 mHC requires fp32_flops in the system spec".into()))?;
        Ok(leaf(spec, ops, bytes, fp32))
    }
    pub fn query(&self, db: &PerfDatabase, tokens: u32) -> Result<PerformanceResult, AicError> {
        query_leaf(db, COMPONENT_MHC, &self.structure(), self.is_context, self.tp_size, Coordinates::tokens(tokens), &|q, _| {
            self.sol(&db.system_spec, q)
        })
    }
}

// ---------------------------------------------------------------------------
// shared_linear: one shared-expert projection (gate_up or down), reduce outside.
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv411SharedLinearOp {
    pub name: String,
    pub is_context: bool,
    pub n: u32,
    pub k: u32,
    pub tp_size: u32,
    pub quant_mode: GemmQuantMode,
}

impl Dsv411SharedLinearOp {
    pub fn structure(&self) -> String {
        structure_key(&[
            ("n", self.n.to_string()),
            ("k", self.k.to_string()),
            ("quant_mode", self.quant_mode.name().to_string()),
        ])
    }
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
        Ok(leaf(spec, 2.0 * tokens * self.n as f64 * self.k as f64, self.weight_bytes() + tokens * (self.n + self.k) as f64 * 2.0, rate))
    }
    pub fn query(&self, db: &PerfDatabase, tokens: u32) -> Result<PerformanceResult, AicError> {
        query_leaf(db, COMPONENT_SHARED_LINEAR, &self.structure(), self.is_context, self.tp_size, Coordinates::tokens(tokens), &|q, _| {
            self.sol(&db.system_spec, q)
        })
    }
}

// ---------------------------------------------------------------------------
// stage: one decoder layer; children summed sequentially.
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv411StageOp {
    pub name: String,
    pub is_context: bool,
    pub children: Vec<Op>,
}

impl Dsv411StageOp {
    pub fn weight_bytes(&self) -> f64 {
        self.children.iter().map(Op::weight_bytes).sum()
    }
    pub fn query(&self, db: &PerfDatabase, ctx: &RuntimeContext) -> Result<PerformanceResult, AicError> {
        let mut total: Option<PerformanceResult> = None;
        for op in &self.children {
            let mut child = *ctx;
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
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python/aisimulate/src/aisimulate_core/systems");
        PerfDatabase::load(&root, "gb300", "sglang", "0.5.14").unwrap().with_mode(mode, TransferPolicy::ALL)
    }

    fn spec() -> SystemSpec {
        let mut spec = test_db(DatabaseMode::Sol).system_spec.clone();
        spec.gpu.mem_bw = 1e6;
        spec.gpu.bfloat16_tc_flops = Some(1e9);
        spec.gpu.fp8_tc_flops = Some(2e9);
        spec.gpu.fp4_tc_flops = Some(4e9);
        spec.gpu.fp32_flops = Some(1e9);
        spec
    }

    fn layout() -> Dsv411KvLayout {
        Dsv411KvLayout { window_entry_bytes: 584.0, main_entry_bytes: 584.0, index_entry_bytes: 132.0 }
    }

    fn attention(role: &str, ratio: u32, context: bool) -> Dsv411AttentionCoreOp {
        Dsv411AttentionCoreOp {
            name: "attention_core".into(),
            is_context: context,
            role: role.into(),
            compress_ratio: ratio,
            tp_size: 2,
            hidden_size: 5120,
            num_heads: 32,
            head_dim: 512,
            q_lora_rank: 1280,
            o_lora_rank: 1024,
            o_groups: 4,
            window_size: 128,
            index_topk: 512,
            index_head_dim: 128,
            kv_layout: layout(),
            gemm_quant_mode: GemmQuantMode::Fp8Block,
            fmha_quant_mode: FmhaQuantMode::Fp8,
        }
    }

    fn indexer(ratio: u32, context: bool, skip: bool) -> Dsv411IndexerOp {
        Dsv411IndexerOp {
            name: "indexer".into(),
            is_context: context,
            compress_ratio: ratio,
            tp_size: 2,
            hidden_size: 5120,
            q_lora_rank: 1280,
            index_n_heads: 32,
            index_head_dim: 128,
            index_topk: 512,
            is_candidate_source: false,
            candidate_limit: 0,
            index_entry_bytes: 132.0,
            scoring_quant_mode: GemmQuantMode::Fp8,
            skip_within_topk: skip,
            gemm_quant_mode: GemmQuantMode::Fp8Block,
        }
    }

    #[test]
    fn roles_validate_against_compress_ratio() {
        assert!(attention("swa", 0, true).sol(&spec(), 1.0, 128.0, 0.0).is_ok());
        assert!(attention("swa", 1, true).sol(&spec(), 1.0, 128.0, 0.0).is_err());
        assert!(attention("full", 0, true).sol(&spec(), 1.0, 128.0, 0.0).is_err());
        assert!(attention("other", 1, true).sol(&spec(), 1.0, 128.0, 0.0).is_err());
    }

    #[test]
    fn attention_core_scales_with_query_and_reads_kv_layout_bytes() {
        let op = attention("reuse", 1, true);
        let short = op.sol(&spec(), 1.0, 128.0, 0.0).unwrap().sol.unwrap();
        let long = op.sol(&spec(), 1.0, 4096.0, 0.0).unwrap().sol.unwrap();
        assert!(long.math_ms > short.math_ms && long.mem_ms > short.mem_ms);
        let mut wide = attention("reuse", 1, true);
        wide.kv_layout.main_entry_bytes *= 2.0;
        let wide_mem = wide.sol(&spec(), 1.0, 4096.0, 0.0).unwrap().sol.unwrap().mem_ms;
        assert!(wide_mem > long.mem_ms);
    }

    #[test]
    fn full_role_pays_for_compression_and_published_rows() {
        let reuse = attention("reuse", 2, true).sol(&spec(), 1.0, 1024.0, 0.0).unwrap().sol.unwrap();
        let full = attention("full", 2, true).sol(&spec(), 1.0, 1024.0, 0.0).unwrap().sol.unwrap();
        assert!(full.math_ms > reuse.math_ms && full.mem_ms > reuse.mem_ms);
    }

    #[test]
    fn indexer_skip_rule_only_removes_scoring_within_topk() {
        let scored = indexer(1, true, false).sol(&spec(), 1.0, 256.0, 0.0).unwrap().sol.unwrap();
        let skipped = indexer(1, true, true).sol(&spec(), 1.0, 256.0, 0.0).unwrap().sol.unwrap();
        assert!(skipped.math_ms < scored.math_ms);
        // beyond top-k the skip rule no longer applies
        let a = indexer(1, true, false).sol(&spec(), 1.0, 4096.0, 0.0).unwrap().sol.unwrap();
        let b = indexer(1, true, true).sol(&spec(), 1.0, 4096.0, 0.0).unwrap().sol.unwrap();
        assert_eq!(a.math_ms, b.math_ms);
    }

    #[test]
    fn indexer_scoring_rate_is_a_declared_field_not_an_sm_switch() {
        let mut fp8 = indexer(1, false, false);
        fp8.scoring_quant_mode = GemmQuantMode::Fp8;
        let mut bf16 = fp8.clone();
        bf16.scoring_quant_mode = GemmQuantMode::Bfloat16;
        let s = spec();
        let f = fp8.sol(&s, 1.0, 1.0, 8192.0).unwrap().sol.unwrap();
        let b = bf16.sol(&s, 1.0, 1.0, 8192.0).unwrap().sol.unwrap();
        assert!(f.math_ms < b.math_ms);
        let mut nvfp4 = fp8.clone();
        nvfp4.scoring_quant_mode = GemmQuantMode::Nvfp4;
        let mut hopper = s.clone();
        hopper.gpu.fp4_tc_flops = None;
        assert!(matches!(nvfp4.sol(&hopper, 1.0, 1.0, 8192.0), Err(AicError::MissingSystemFlops(_))));
    }

    #[test]
    fn stage_sums_children_sequentially_in_sol_mode() {
        let child = Op::Dsv411Mhc(Dsv411MhcOp {
            name: "mhc".into(),
            is_context: true,
            hidden_size: 5120,
            hc_mult: 4,
            sinkhorn_iters: 20,
            tp_size: 2,
        });
        let db = test_db(DatabaseMode::Sol);
        let ctx = RuntimeContext { batch_size: 1, s: 128, prefix: 0, num_tokens: 128, ..RuntimeContext::default() };
        let one = Dsv411StageOp { name: "s".into(), is_context: true, children: vec![child.clone()] };
        let two = Dsv411StageOp { name: "s".into(), is_context: true, children: vec![child.clone(), child] };
        let a = one.query(&db, &ctx).unwrap().latency_ms;
        let b = two.query(&db, &ctx).unwrap().latency_ms;
        assert!(a > 0.0 && (b - 2.0 * a).abs() < 1e-9);
    }

    #[test]
    fn structure_keys_are_stable_and_exclude_names_and_phase() {
        let a = attention("full", 2, true);
        let mut b = a.clone();
        b.name = "other".into();
        b.is_context = false;
        assert_eq!(a.structure(), b.structure());
        assert!(a.structure().starts_with("role=full|compress_ratio=2|"));
    }
}
