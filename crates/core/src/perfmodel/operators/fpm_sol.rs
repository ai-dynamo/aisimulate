// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Includes changes adapted from:
// https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/aiconfigurator-core/src/operators/fpm_sol.rs

//! SOL-mode op queries for the FPM whole-model roofline.
//!
//! Python's `forward_model="fpm"` derives its interpolation roofline from the
//! model's ORIGINAL op-level list queried on a `DatabaseMode.SOL` database
//! view: the op code runs unchanged, and every `database.query_*` call
//! answers its analytic `get_sol()` branch instead of a table lookup. Rust
//! has no database-mode switch, so this module mirrors, per op family, the
//! composition "op-level shape math × SOL leaf formula" as ONE function of
//! the (possibly NON-INTEGER) inputs.
//!
//! Inputs are `f64` on purpose: the FPM sol_fn back-maps iteration totals as
//! `s = max(total/batch, 1.0)` and `prefix = total_kv/batch`, which are
//! fractional whenever `batch` does not divide the totals — and Python keeps
//! them fractional through the whole SOL chain. The floor/ceil sites below
//! exist exactly where the Python chain floor-divides (`//` on floats is a
//! float floor; `-(-x // d)` is a float ceil) — there is NO `int()` cast
//! anywhere on the Python SOL path.
//!
//! Formula provenance (file:line refers to the Python SDK):
//! - GEMM: `operations/gemm.py:436-443` (`get_sol`), `:748-749` (m mapping),
//!   `:282-287` (tc_flops selection). fp8_static's subtraction chain is
//!   floored back to the plain GEMM SOL under SOL mode (`:787-800`), so both
//!   quant classes share one formula.
//! - Attention: `operations/attention.py:319-341` (context `get_sol`,
//!   prefix-aware), `:710-733` (generation `get_sol`), `:531` (CP chunk
//!   ceil), `:535-548` (fused rope/kv-write/qk-norm extras × 1.1 on the SOL
//!   mem-op `bytes / mem_bw * 1000`, `perf_database.py:2294`).
//! - MoE: `operations/moe.py:297-325` (`get_sol` with the activated-expert
//!   clamp; `workload_distribution` deliberately unused).
//! - MoE dispatch: `operations/moe.py:1083-1342` branch structure with the
//!   collective SOLs below; SGLang DeepEP SOL raises in Python
//!   (`NotImplementedError`) and errors here.
//! - Collectives: NCCL `operations/communication.py:384-394`, custom
//!   allreduce `:129-136` (ring, hard-coded 2 B/elem, quant ignored), P2P
//!   `:556-559` (always `inter_node_bw`, NO `p2p_latency` under SOL).
//! - Embedding/ElementWise: `operations/embedding.py:49-62`,
//!   `elementwise.py:49-66` over the SOL mem-op.
//! - DSA (DeepSeek sparse attention): `operations/dsa.py`
//!   `ContextDSAModule`/`GenerationDSAModule` `get_sol`, reused verbatim via
//!   `perf_database::dsa::{dsa_context_sol_ms, dsa_generation_sol_ms}` rather
//!   than re-derived here; this module only maps FPM coordinates, blends
//!   `full_frac` (context only, mirroring `DsaModuleOp::query_context` under
//!   `DatabaseMode::Sol`), and applies `scale_factor`.
//!
//! Known, documented approximation: Python `ElementWise` floors
//! `x // scale_num_tokens` before converting tokens to bytes; the Rust
//! `ElementwiseOp` wire form folds `scale_num_tokens` into a continuous
//! `bytes_per_token` (see `engine.py::_elementwise`), losing that floor for
//! `x % scale_num_tokens != 0`. This is the SAME approximation the silicon
//! engine-step path already ships; the error is bounded by one token's bytes
//! and is negligible against the whole-model roofline.
//!
//! A second known approximation: the DSA arms round `batch`, `s`, and
//! `prefix` to `i64` (`b.round().max(1.0) as i64`, see
//! `dsa_context_module_sol` / `dsa_generation_module_sol`) because
//! `perf_database::dsa`'s SOL ports take integer coordinates (the triangular
//! `total_kv_pairs` sum, `perf_database/dsa.rs:824-832`, is integer-only);
//! the error is bounded by one token per coordinate and only arises when
//! `batch` does not divide the FPM totals.

use crate::common::enums::{BackendKind, GemmQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::operators::DsaModuleOp;
use crate::operators::op::Op;
use crate::operators::{
    ContextAttentionOp, CustomAllReduceOp, ElementwiseOp, EmbeddingOp, GemmOp,
    GenerationAttentionOp, MoEDispatchOp, MoeOp, NcclOp, P2POp,
};
use crate::perf_database::PerfDatabase;
use crate::perf_database::dsa::{
    dsa_context_sol_flops, dsa_context_sol_ms, dsa_dims, dsa_generation_sol_flops,
    dsa_generation_sol_ms,
};

/// Python float floor-division `a // b`.
fn floor_div(a: f64, b: f64) -> f64 {
    (a / b).floor()
}

/// Python `-(-a // b)`: exact ceiling that stays float.
fn ceil_div(a: f64, b: f64) -> f64 {
    (a / b).ceil()
}

/// SOL-mode latency (ms) of one granular op at the FPM sol coordinates.
///
/// `x` is Python's `x` kwarg (token count for compute ops; `batch` for
/// `logits_gemm`), `batch`/`s`/`prefix` mirror `op.query(view, x=x,
/// batch_size=batch, beam_width=1, s=s, prefix=prefix)`. Ops ignore the
/// kwargs they ignore in Python.
pub(crate) fn op_sol_latency_ms(
    op: &Op,
    db: &PerfDatabase,
    x: f64,
    batch: f64,
    s: f64,
    prefix: f64,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    match op {
        Op::TokenScale(o) => {
            // Reject malformed directly constructed Rust values as well as
            // serialized input (which validates the widths on deserialize).
            o.scale_tokens(0)?;
            let ratio = f64::from(o.numerator) / f64::from(o.denominator);
            op_sol_latency_ms(&o.op, db, x * ratio, batch * ratio, s, prefix)
        }
        Op::Gemm(o) => Ok(gemm_sol(o, spec, x)),
        Op::Embedding(o) => Ok(embedding_sol(o, spec, x)),
        Op::Elementwise(o) => Ok(elementwise_sol(o, spec, x)),
        Op::ContextAttention(o) => Ok(context_attention_sol(o, spec, batch, s, prefix)),
        Op::GenerationAttention(o) => Ok(generation_attention_sol(o, spec, batch, s)),
        Op::DsaContext(o) => dsa_context_module_sol(o, spec, batch, s, prefix),
        Op::DsaGeneration(o) => dsa_generation_module_sol(o, spec, batch, s),
        Op::Moe(o) => Ok(moe_sol(o, spec, x)),
        Op::MoeDispatch(o) => moe_dispatch_sol(o, spec, x),
        Op::CustomAllReduce(o) => Ok(custom_allreduce_op_sol(o, spec, x)),
        Op::Nccl(o) => Ok(nccl_op_sol(o, spec, x)),
        Op::P2P(o) => Ok(p2p_sol(o, spec, x)),
        // Python `OverlapOp.query` under SOL: each group summed, max of the
        // two totals.
        Op::Overlap(o) => {
            let mut total_a = 0.0;
            for inner in &o.group_a {
                total_a += op_sol_latency_ms(inner, db, x, batch, s, prefix)?;
            }
            let mut total_b = 0.0;
            for inner in &o.group_b {
                total_b += op_sol_latency_ms(inner, db, x, batch, s, prefix)?;
            }
            Ok(total_a.max(total_b))
        }
        // Python `FallbackOp.query` under SOL: the primary's analytic SOL
        // answers (no data to miss); mirror the perf-DB-miss fallback anyway.
        Op::Fallback(o) => match op_sol_latency_ms(&o.primary, db, x, batch, s, prefix) {
            Ok(v) => Ok(v),
            Err(AicError::PerfDatabase(_)) | Err(AicError::Io { .. }) => {
                let mut total = 0.0;
                for inner in &o.fallback {
                    total += op_sol_latency_ms(inner, db, x, batch, s, prefix)?;
                }
                Ok(total)
            }
            Err(other) => Err(other),
        },
        other => Err(AicError::SolNotImplemented(format!(
            "forward_model='fpm' SOL roofline has no Rust implementation for op {}",
            other.name()
        ))),
    }
}

/// Python `GEMM._get_quant_tc_flops` (gemm.py:282-287): compute factor
/// 1/2/4 -> the matching spec field when present, else
/// `bfloat16_tc_flops * compute`.
fn quant_tc_flops(spec: &SystemSpec, quant: GemmQuantMode) -> f64 {
    let compute = quant.mapping().compute;
    let direct = if compute == 1.0 {
        spec.gpu.bfloat16_tc_flops
    } else if compute == 2.0 {
        spec.gpu.fp8_tc_flops
    } else if compute == 4.0 {
        spec.gpu.fp4_tc_flops
    } else {
        None
    };
    direct.unwrap_or_else(|| spec.gpu.bfloat16_tc_flops.unwrap_or(0.0) * compute)
}

/// gemm.py:436-443 + the m mapping at :748-749. Under SOL, fp8_static's
/// subtraction chain is floored back to this same value (:787-800).
fn gemm_sol(op: &GemmOp, spec: &SystemSpec, x: f64) -> f64 {
    // Python: `x //= scale_num_tokens` (floor, fires even at 1 for fractional
    // x), then `x = -(-x // seq_split)` (ceil).
    let m = ceil_div(
        floor_div(x, op.scale_num_tokens.max(1) as f64),
        op.seq_split.max(1) as f64,
    );
    let (n, k) = (op.n as f64, op.k as f64);
    let mapping = op.quant_mode.mapping();
    let tc_flops = quant_tc_flops(spec, op.quant_mode);
    let sol_math = 2.0 * m * n * k / tc_flops * 1000.0;
    let sol_mem = mapping.memory * (m * n + m * k + n * k) / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem) * op.scale_factor
}

/// The SOL mem-op leaf (perf_database.py:2294): `bytes / mem_bw * 1000` —
/// no empirical scaling factor, no constant latency.
fn mem_op_sol_ms(spec: &SystemSpec, mem_bytes: f64) -> f64 {
    mem_bytes / spec.gpu.mem_bw * 1000.0
}

/// embedding.py:49-62: `x = -(-x // seq_split)`, `d2d_bytes = x * hidden * 2`
/// (hard-coded bf16 bytes), one SOL mem-op.
fn embedding_sol(op: &EmbeddingOp, spec: &SystemSpec, x: f64) -> f64 {
    let tokens = ceil_div(x, op.seq_split.max(1) as f64);
    mem_op_sol_ms(spec, tokens * op.hidden_size as f64 * 2.0) * op.scale_factor
}

/// elementwise.py:49-66 over the folded `bytes_per_token` wire form (see the
/// module doc for the scale_num_tokens floor approximation).
fn elementwise_sol(op: &ElementwiseOp, spec: &SystemSpec, x: f64) -> f64 {
    // Python: `x //= scale_num_tokens` (floor) THEN `-(-x // seq_split)`
    // (ceil). The wire op carries scale_num_tokens since schema v4, so the
    // floor is exact (older folded-bytes specs deserialize with divisor 1).
    let tokens = ceil_div(
        floor_div(x, op.scale_num_tokens.max(1) as f64),
        op.seq_split.max(1) as f64,
    );
    mem_op_sol_ms(spec, op.bytes_per_token * tokens) * op.scale_factor
}

/// attention.py:319-341 — the prefix-aware context SOL (the crate's
/// `context_attention_sol_ms` is the prefix=0 specialization used by the
/// silicon interp anchors, so it cannot be reused here).
fn context_sol_one(op: &ContextAttentionOp, spec: &SystemSpec, b: f64, s: f64, p: f64) -> f64 {
    let (n, n_kv, h, w) = (
        op.n as f64,
        op.n_kv as f64,
        op.head_size as f64,
        op.window_size as f64,
    );
    let full_s = s + p;
    let ops = if op.window_size > 0 && full_s > w {
        // windowed: (full_s - p) = s new tokens each attend a w-window; no
        // causal halving in this branch (Python :332-333)
        2.0 * b * s * w * n * h * 2.0
    } else {
        2.0 * b * (full_s * full_s - p * p) * n * h * 2.0 / 2.0
    };
    // Q read + O write in bf16 on the new tokens; K and V over the FULL
    // sequence at kv-cache width. The window does NOT shrink mem (Python).
    let mem = 2.0 * b * (n * s * h + n * s * h)
        + op.kv_cache_dtype.mapping().memory * b * (2.0 * n_kv * full_s * h);
    let flops = spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
    let sol_math = ops / flops * 1000.0 / op.fmha_quant_mode.mapping().compute;
    let sol_mem = mem / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem)
}

/// ContextAttention.query under SOL (attention.py:507-558): CP zigzag chunks
/// + the enabled fused rope/kv-write/(qk-norm) extras × 1.1, each a SOL mem-op.
fn context_attention_sol(
    op: &ContextAttentionOp,
    spec: &SystemSpec,
    b: f64,
    s: f64,
    p: f64,
) -> f64 {
    let fmha = if op.cp_size > 1 {
        // Python :531: `c = max(1, -(-isl // (2 * cp)))` — float ceil.
        let c = ceil_div(s, 2.0 * op.cp_size as f64).max(1.0);
        context_sol_one(op, spec, b, c, p) + context_sol_one(op, spec, b, c, p + s - c)
    } else {
        context_sol_one(op, spec, b, s, p)
    };
    let q_num = (op.n * op.head_size) as f64;
    let k_num = (op.n_kv * op.head_size) as f64;
    let mut extra = 0.0;
    if op.use_qk_norm {
        let qk_norm =
            2.0 * mem_op_sol_ms(spec, q_num * 2.0) + 2.0 * mem_op_sol_ms(spec, k_num * 2.0);
        extra += qk_norm * 2.0;
    }
    if op.apply_rope {
        extra += 2.0 * mem_op_sol_ms(spec, q_num * 2.0 + k_num * 2.0);
    }
    let fq_mem = op.fmha_quant_mode.mapping().memory;
    extra += mem_op_sol_ms(spec, k_num * fq_mem) + mem_op_sol_ms(spec, k_num * fq_mem); // kv write (k_num == v_num)
    // Decode CP on the same engine: all-gather the cached-context KV stripes
    // (per-rank K+V of `p` tokens, comm half-elements) over the DCP group.
    let gather = if op.dcp_size > 1 && p > 0.0 {
        let kv_elems =
            2.0 * (op.n_kv * op.head_size) as f64 * op.kv_cache_dtype.mapping().memory / 2.0;
        nccl_sol(spec, op.dcp_size, "all_gather", b * p * kv_elems, 2.0)
    } else {
        0.0
    };
    (fmha + extra * 1.1 + gather) * op.scale_factor
}

/// Generation-attention SOL plus the optional Q/K RMSNorm fused extra. There
/// is no 5-sample smoothing and no prefix in SOL mode.
fn generation_attention_sol(op: &GenerationAttentionOp, spec: &SystemSpec, b: f64, s: f64) -> f64 {
    // Decode CP: the kernel sees the DCP group's `n * dcp` gathered query
    // heads over this rank's `ceil(s / dcp)` KV stripe (op-level
    // `GenerationAttentionOp::dcp_size` docs); the Q/K norm extra below stays
    // on the rank-local heads because it runs before the query gather.
    let dcp = op.dcp_size.max(1) as f64;
    let (n, n_kv, h, w) = (
        op.n as f64 * dcp,
        op.n_kv as f64,
        op.head_size as f64,
        op.window_size as f64,
    );
    let s_local = if dcp > 1.0 { ceil_div(s, dcp) } else { s };
    let kv_len = if op.window_size > 0 {
        (s_local - 1.0).min(w)
    } else {
        s_local - 1.0
    };
    // fp8 KV -> fp8 compute; everything else (incl. int8 KV) -> bf16 compute.
    let compute = if op.kv_cache_dtype == crate::common::enums::KvCacheQuantMode::Fp8 {
        2.0
    } else {
        1.0
    };
    let kv_mem = op.kv_cache_dtype.mapping().memory;
    let ops = 2.0 * b * n * h * 2.0 * kv_len;
    let mem = b * (n * h * 2.0 + 2.0 * n_kv * kv_len * h * kv_mem + n * h * 2.0);
    let flops = spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
    let sol_math = ops / flops * 1000.0 / compute;
    let sol_mem = mem / spec.gpu.mem_bw * 1000.0;
    let mut latency = sol_math.max(sol_mem);
    if op.use_qk_norm {
        let q_num = op.n as f64 * h;
        let k_num = n_kv * h;
        let qk_norm =
            2.0 * mem_op_sol_ms(spec, q_num * 2.0) + 2.0 * mem_op_sol_ms(spec, k_num * 2.0);
        latency += qk_norm * 2.0 * 1.1;
    }
    latency * op.scale_factor
}

/// Whole-forward SOL leaf for the DSA context module (`Op::DsaContext`). Reuses the
/// op-level DSA roofline exactly as `operators::dsa::query_context_table` does under
/// `DatabaseMode::Sol`: `dsa_context_sol_ms` with the op's dims, top-k and quant modes,
/// blended between the full and skip-indexer variants by the configured `full_frac`
/// (SOL mode weights by `full_frac` directly, without probing the skip table), then
/// `scale_factor`. FPM hands per-request coordinates as f64 (`s = total_prefill / batch`,
/// `prefix = total_kv / batch`); the roofline takes integers, so they are rounded.
/// DSA context parallelism composes latency-only table deltas in the op-level path
/// and has no roofline, so it stays a typed `SolNotImplemented`, and a `full_frac`
/// outside `[0, 1]` a typed `InvalidEngineConfig`.
fn dsa_context_module_sol(
    op: &DsaModuleOp,
    spec: &SystemSpec,
    b: f64,
    s: f64,
    p: f64,
) -> Result<f64, AicError> {
    if op.cp_size > 1 {
        return Err(AicError::SolNotImplemented(format!(
            "forward_model='fpm' SOL roofline has no Rust implementation for DSA context \
             parallelism (cp_size={}) on op {}",
            op.cp_size, op.name
        )));
    }
    let dims = dsa_dims(&op.architecture);
    let flops = dsa_context_sol_flops(spec, op.gemm_quant_mode, op.fmha_quant_mode)?;
    let (b, s, p) = (
        b.round().max(1.0) as i64,
        s.round().max(1.0) as i64,
        p.round().max(0.0) as i64,
    );
    let sol = |skip_indexer: bool| {
        dsa_context_sol_ms(
            spec,
            dims,
            op.index_topk as i64,
            op.kv_cache_dtype,
            op.fmha_quant_mode,
            op.gemm_quant_mode,
            b,
            s,
            p,
            op.num_heads as i64,
            skip_indexer,
            flops,
        )
    };
    // `full_frac` is a fraction of layers (Python `dsa_full_layer_fraction`), copied
    // verbatim by the `py_ops` constructors and defaulted — never validated — on the
    // wire, so a bad value first becomes observable here, and every way it lands is
    // a plausible-looking wrong number rather than a failure: w < 0 extrapolates
    // BELOW both legs, w > 1 and +inf are silently swallowed by the `w >= 1.0`
    // short-circuit, and NaN / -inf are laundered into 0.0 by the `max(0.0)` floor —
    // a free attention module. Reject instead of clamping: a clamp would price a
    // misconfigured model as if it had been configured correctly. `contains` is
    // false for NaN and for both infinities, so this one gate covers them all.
    let w = op.full_frac;
    if !(0.0..=1.0).contains(&w) {
        return Err(AicError::InvalidEngineConfig(format!(
            "DSA context op {} has dsa_full_layer_fraction={w}, which is not a fraction \
             in [0, 1]; the full/skip SOL blend is undefined outside that range",
            op.name
        )));
    }
    let ms = if w >= 1.0 {
        sol(false)
    } else {
        w * sol(false) + (1.0 - w) * sol(true)
    };
    // Decode CP on the same engine: all-gather the cached latent-KV stripes plus
    // the indexer K cache weighted by the full-indexer fraction (mirrors
    // `DsaModuleOp::query_context`'s `dcp_context_gather`).
    let gather = if op.dcp_size > 1 && p > 0 {
        let kv_elems = crate::operators::dsa::dsa_cached_context_gather_elems(op.kv_cache_dtype, w);
        nccl_sol(
            spec,
            op.dcp_size,
            "all_gather",
            b as f64 * p as f64 * kv_elems,
            2.0,
        )
    } else {
        0.0
    };
    Ok((ms.max(0.0) + gather) * op.scale_factor)
}

/// Whole-forward SOL leaf for the DSA generation module (`Op::DsaGeneration`): the
/// op-level decode roofline (`dsa_generation_sol_ms`; the attention group is bf16 and
/// the skip-indexer variant never enters the decode SOL, as in
/// `operators::dsa::query_generation_table`), then `scale_factor`.
fn dsa_generation_module_sol(
    op: &DsaModuleOp,
    spec: &SystemSpec,
    b: f64,
    s: f64,
) -> Result<f64, AicError> {
    let dims = dsa_dims(&op.architecture);
    let flops = dsa_generation_sol_flops(spec, op.gemm_quant_mode)?;
    // Decode CP geometry mirrors `DsaModuleOp::query_generation`: gathered
    // heads over this rank's KV stripe (top-k kept whole; upper bound).
    let dcp = op.dcp_size.max(1) as f64;
    let s_local = if dcp > 1.0 { ceil_div(s, dcp) } else { s };
    let ms = dsa_generation_sol_ms(
        spec,
        dims,
        op.kv_cache_dtype,
        op.gemm_quant_mode,
        b.round().max(1.0) as i64,
        s_local.round().max(1.0) as i64,
        (op.num_heads as f64 * dcp) as i64,
        flops,
    );
    Ok(ms.max(0.0) * op.scale_factor)
}

/// moe.py:297-325: MoE SOL with the activated-expert clamp. The `//` sites
/// are float floors in exactly Python's association order.
fn moe_sol(op: &MoeOp, spec: &SystemSpec, x: f64) -> f64 {
    let dp = op.attention_dp_size.max(1) as f64;
    let (h, inter) = (op.hidden_size as f64, op.inter_size as f64);
    let num_gemms = if op.is_gated { 3.0 } else { 2.0 };
    let (ep, tp) = (op.moe_ep_size.max(1) as f64, op.moe_tp_size.max(1) as f64);
    let total_tokens = x * dp * op.topk as f64;
    // ops = TT*H*I*G*2 // ep // tp
    let ops = floor_div(
        floor_div(total_tokens * h * inter * num_gemms * 2.0, ep),
        tp,
    );
    // mem = m * ( TT//ep*H*2 + TT//ep*I*G//tp + H*I*G//tp * min(E//ep, TT//ep) )
    let tt_ep = floor_div(total_tokens, ep);
    let mem = op.quant_mode.mapping().memory
        * (tt_ep * h * 2.0
            + floor_div(tt_ep * inter * num_gemms, tp)
            + floor_div(h * inter * num_gemms, tp)
                * floor_div(op.num_experts as f64, ep).min(tt_ep));
    let flops = spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
    let sol_math = ops / (flops * op.quant_mode.mapping().compute) * 1000.0;
    let sol_mem = mem / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem) * op.scale_factor
}

/// communication.py:129-136: ring allreduce, hard-coded 2 B/elem, quant
/// ignored, real tp_size (no node capping), no latency constant.
fn custom_allreduce_sol(spec: &SystemSpec, tp_size: u32, size_elems: f64) -> f64 {
    if tp_size <= 1 {
        return 0.0;
    }
    let tp = tp_size as f64;
    let bw = spec.get_p2p_bandwidth(tp_size);
    2.0 * size_elems * 2.0 / tp * (tp - 1.0) / bw * 1000.0
}

/// communication.py:384-394: NCCL collective SOL. `message_size` is an
/// element count scaled by the dtype's byte width; unknown op names are 0.
fn nccl_sol(
    spec: &SystemSpec,
    num_gpus: u32,
    operation: &str,
    message_size: f64,
    bytes_per_elem: f64,
) -> f64 {
    let n = num_gpus as f64;
    let bw = spec.get_p2p_bandwidth(num_gpus);
    match operation {
        "all_gather" | "alltoall" | "reduce_scatter" => {
            bytes_per_elem * message_size * (n - 1.0) / n / bw * 1000.0
        }
        "all_reduce" => 2.0 * bytes_per_elem * message_size * (n - 1.0) / n / bw * 1000.0,
        _ => 0.0,
    }
}

/// CustomAllReduce.query under SOL (communication.py:252-268).
fn custom_allreduce_op_sol(op: &CustomAllReduceOp, spec: &SystemSpec, x: f64) -> f64 {
    if op.tp_size == 1 {
        return 0.0;
    }
    let size = ceil_div(x, op.seq_split.max(1) as f64) * op.hidden_size as f64;
    custom_allreduce_sol(spec, op.tp_size, size) * op.scale_factor
}

/// NCCL.query under SOL (communication.py:509-519).
fn nccl_op_sol(op: &NcclOp, spec: &SystemSpec, x: f64) -> f64 {
    let msg = ceil_div(x, op.seq_split.max(1) as f64) * op.hidden_size;
    nccl_sol(
        spec,
        op.num_gpus,
        &op.operation,
        msg,
        op.dtype.mapping().memory,
    ) * op.scale_factor
}

/// P2P.query under SOL (communication.py:583-598 + :556-559): always
/// `inter_node_bw`, literal 2 B/elem, NO `p2p_latency` term.
fn p2p_sol(op: &P2POp, spec: &SystemSpec, x: f64) -> f64 {
    if op.pp_size == 1 {
        return 0.0;
    }
    let p2p_bytes = ceil_div(x, op.seq_split.max(1) as f64) * op.hidden_size as f64 * 2.0;
    p2p_bytes / spec.node.inter_node_bw * 1000.0 * op.scale_factor
}

/// MoEDispatch.query under SOL (moe.py:1083-1342). Branch structure mirrors
/// the op's silicon query (`moe_dispatch.rs`), with the collective SOLs
/// substituted for the table lookups. Message sizes here are passed straight
/// to the DB-level collectives in Python (no per-op ceil), so no ceil either.
fn moe_dispatch_sol(op: &MoEDispatchOp, spec: &SystemSpec, x: f64) -> Result<f64, AicError> {
    use crate::operators::moe_dispatch::DispatchFlavor;

    let volume = x * op.hidden_size as f64; // element count, half precision
    let num_gpus = (op.moe_tp_size * op.moe_ep_size).max(1);
    let attn_dp = op.attention_dp_size.max(1);
    let attn_tp = (num_gpus / attn_dp).max(1);
    let dp = attn_dp as f64;
    let pre = op.pre_dispatch;
    let half_bytes = 2.0; // CommQuantMode::Half.memory — MoEDispatch always passes half

    let comm = match op.flavor {
        DispatchFlavor::RetiredDeepEp => {
            return Err(AicError::InvalidEngineConfig(format!(
                "MoEDispatch '{}' (moe_backend='deepep_moe') has no native SOL \
                 (retired with AIC-1601; large-EP comm is modeled by MoeAllToAll)",
                op.name
            )));
        }
        DispatchFlavor::CustomAllReduce => match op.backend {
            // vllm (moe.py:1222-1239): additive.
            BackendKind::Vllm => {
                let mut total = 0.0;
                if attn_tp > 1 {
                    total += custom_allreduce_sol(spec, num_gpus, volume);
                }
                if attn_dp > 1 {
                    let op_name = if pre { "all_gather" } else { "reduce_scatter" };
                    total += nccl_sol(spec, num_gpus, op_name, volume * dp, half_bytes);
                }
                total
            }
            // sglang non-deepep (moe.py:1261-1342).
            BackendKind::Sglang => {
                if attn_tp > 1 && attn_dp > 1 {
                    if pre {
                        nccl_sol(spec, attn_tp, "reduce_scatter", volume, half_bytes)
                            + nccl_sol(spec, num_gpus, "all_gather", volume * dp, half_bytes)
                    } else {
                        nccl_sol(spec, num_gpus, "reduce_scatter", volume * dp, half_bytes)
                            + nccl_sol(spec, attn_tp, "all_gather", volume, half_bytes)
                    }
                } else if op.attn_cp_size > 1 {
                    if op.is_context {
                        let op_name = if pre { "all_gather" } else { "reduce_scatter" };
                        nccl_sol(spec, num_gpus, op_name, volume, half_bytes)
                    } else if pre {
                        0.0
                    } else {
                        custom_allreduce_sol(spec, num_gpus, volume)
                    }
                } else if attn_tp > 1 {
                    custom_allreduce_sol(spec, num_gpus, volume)
                } else if attn_dp > 1 {
                    let op_name = if pre { "all_gather" } else { "reduce_scatter" };
                    nccl_sol(spec, num_gpus, op_name, volume * dp, half_bytes)
                } else {
                    0.0
                }
            }
            // trtllm, sm != 100 (moe.py:1194-1221): pre/post symmetric.
            BackendKind::Trtllm => {
                if attn_tp > 1 {
                    custom_allreduce_sol(spec, num_gpus, volume)
                } else if attn_dp > 1 {
                    let op_name = if pre { "all_gather" } else { "reduce_scatter" };
                    nccl_sol(spec, num_gpus, op_name, volume * dp, half_bytes)
                } else {
                    0.0
                }
            }
        },
        // trtllm SM100 (moe.py:1095-1193).
        DispatchFlavor::TrtllmAlltoall => {
            let is_nvl72 = spec.node.num_gpus_per_node >= 72;
            let enable_alltoall = op.attention_dp_size > 1 && op.moe_tp_size == 1 && is_nvl72;
            if enable_alltoall {
                // trtllm_alltoall SOL (moe.py:2068-2107): dispatch moves the
                // moe-quant-compressed activations, combine moves bf16.
                let node_num = if op.moe_ep_size < 4 {
                    1
                } else {
                    op.moe_ep_size / 4
                };
                let bw = if node_num > 1 {
                    spec.node.inter_node_bw
                } else {
                    spec.node.intra_node_bw
                };
                let remote_ranks =
                    op.topk
                        .min(op.num_experts)
                        .min(op.moe_ep_size.saturating_sub(1)) as f64;
                let bytes_per_elem = if pre {
                    op.moe_quant.mapping().memory
                } else {
                    2.0
                };
                let data_bytes = x * remote_ranks * op.hidden_size as f64 * bytes_per_elem;
                data_bytes / bw * 1000.0
            } else if op.attention_dp_size > 1 {
                // moe.py:1142-1145 / :1173-1179: the pre all_gather moves the
                // moe-quant-COMPRESSED volume (nvfp4: V/4 + V/32; fp8: V/2);
                // the post reduce_scatter is uncompressed (asymmetric).
                if pre {
                    let compressed = match op.moe_quant.mapping().name {
                        "nvfp4" => volume / 4.0 + volume / 4.0 / 8.0,
                        "fp8" | "fp8_block" => volume / 2.0,
                        _ => volume,
                    };
                    nccl_sol(spec, num_gpus, "all_gather", compressed * dp, half_bytes)
                } else {
                    nccl_sol(spec, num_gpus, "reduce_scatter", volume * dp, half_bytes)
                }
            } else if attn_tp > 1 {
                // reduce_results defaults true (mirrors the silicon port).
                if spec.node.num_gpus_per_node == 72 && num_gpus > 4 {
                    nccl_sol(spec, num_gpus, "all_reduce", volume, half_bytes)
                } else {
                    custom_allreduce_sol(spec, num_gpus, volume)
                }
            } else {
                0.0
            }
        }
    };
    Ok(comm * op.scale_factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::common::enums::{
        CommQuantMode, FmhaQuantMode, GemmQuantMode, KvCacheQuantMode, MoeQuantMode,
    };

    /// The b200_sxm spec: mem_bw 7.7e12, bf16 2.25e15, fp8 4.5e15, fp4 9e15,
    /// intra_node_bw 8.1e11, inter_node_bw 4e10 (systems/b200_sxm.yaml).
    fn spec() -> SystemSpec {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/aisimulate/src/aisimulate_core/systems/b200_sxm.yaml");
        SystemSpec::load(&root).expect("b200 spec")
    }

    fn db() -> PerfDatabase {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/aisimulate/src/aisimulate_core/systems");
        PerfDatabase::load(&root, "b200_sxm", "vllm", "0.19.0").expect("db")
    }

    fn approx(got: f64, expected: f64) {
        assert!(
            (got - expected).abs() <= 1e-9 * expected.abs().max(1e-12),
            "got {got}, expected {expected}"
        );
    }

    /// Python oracle:
    /// PYTHONPATH=aic-core/src python3 -c "
    /// from aisimulate_core.sdk import perf_database, common
    /// from aisimulate_core.sdk.operations.gemm import GEMM
    /// view = perf_database.get_database_view('b200_sxm','vllm','0.19.0',database_mode='SOL',allow_missing_data=True)
    /// op = GEMM('qkv_gemm', 1.0, 4096, 4096, common.GEMMQuantMode.nvfp4)
    /// print(repr(float(op.query(view, x=8192.0, batch_size=4, beam_width=1, s=2048.0, prefix=0.0))))"
    #[test]
    fn gemm_sol_matches_formula() {
        let op = GemmOp {
            name: "qkv_gemm".into(),
            scale_factor: 1.0,
            n: 4096,
            k: 4096,
            quant_mode: GemmQuantMode::Nvfp4,
            scale_num_tokens: 1,
            low_precision_input: false,
            seq_split: 1,
            below_grid_sol: false,
        };
        let s = spec();
        let m = 8192.0_f64;
        let (n, k) = (4096.0_f64, 4096.0_f64);
        let tc = s.gpu.fp4_tc_flops.unwrap();
        let expected = (2.0 * m * n * k / tc * 1000.0)
            .max(9.0 / 16.0 * (m * n + m * k + n * k) / s.gpu.mem_bw * 1000.0);
        approx(gemm_sol(&op, &s, 8192.0), expected);
        // fractional x floors first (even at scale_num_tokens == 1)
        approx(gemm_sol(&op, &s, 8192.7), expected);
    }

    #[test]
    fn embedding_vs_elementwise_rounding_directions() {
        let s = spec();
        let emb = EmbeddingOp {
            name: "context_embedding".into(),
            scale_factor: 1.0,
            vocab_size: 128000,
            hidden_size: 6144,
            quant_mode: GemmQuantMode::Bfloat16,
            seq_split: 1,
        };
        // Embedding CEILS fractional x: 10.5 -> 11 tokens.
        approx(
            embedding_sol(&emb, &s, 10.5),
            11.0 * 6144.0 * 2.0 / s.gpu.mem_bw * 1000.0,
        );
        let ew = ElementwiseOp {
            name: "add_norm".into(),
            scale_factor: 2.0,
            bytes_per_token: 8192.0,
            scale_num_tokens: 1,
            seq_split: 1,
        };
        // Elementwise FLOORS first (Python `x //= scale_num_tokens` fires
        // even at divisor 1): 10.5 -> 10 tokens — the OPPOSITE rounding
        // direction from Embedding. The wire op carries scale_num_tokens
        // since schema v4, so the floor is exact.
        approx(
            elementwise_sol(&ew, &s, 10.5),
            8192.0 * 10.0 / s.gpu.mem_bw * 1000.0 * 2.0,
        );
    }

    /// Python oracle:
    /// PYTHONPATH=aic-core/src python3 -c "
    /// from aisimulate_core.sdk import perf_database, common
    /// from aisimulate_core.sdk.operations.attention import ContextAttention, GenerationAttention
    /// view = perf_database.get_database_view('b200_sxm','vllm','0.19.0',database_mode='SOL',allow_missing_data=True)
    /// op = ContextAttention('context_attention', 1.0, 48, 8, 128, kvcache_quant_mode=common.KVCacheQuantMode.fp8, fmha_quant_mode=common.FMHAQuantMode.bfloat16)
    /// print(repr(float(op.query(view, x=1, batch_size=4.0, beam_width=1, s=682.6666666666666, prefix=128.5))))"
    #[test]
    fn context_attention_sol_prefix_aware() {
        let s = spec();
        let op = ContextAttentionOp {
            name: "context_attention".into(),
            scale_factor: 1.0,
            n: 48,
            n_kv: 8,
            head_size: 128,
            window_size: 0,
            kv_cache_dtype: KvCacheQuantMode::Fp8,
            fmha_quant_mode: FmhaQuantMode::Bfloat16,
            use_qk_norm: false,
            cp_size: 1,
            lane_order: crate::operators::attention::b200_vllm_context_lane_order(),
            apply_rope: true,
            dcp_size: 1,
        };
        let (b, sq, p) = (4.0, 682.6666666666666_f64, 128.5_f64);
        let (n, n_kv, h) = (48.0, 8.0, 128.0);
        let full = sq + p;
        let ops = 2.0 * b * (full * full - p * p) * n * h * 2.0 / 2.0;
        let mem = 2.0 * b * (n * sq * h + n * sq * h) + 1.0 * b * (2.0 * n_kv * full * h);
        let fmha = (ops / s.gpu.bfloat16_tc_flops.unwrap() * 1000.0 / 1.0)
            .max(mem / s.gpu.mem_bw * 1000.0);
        let q_num = n * h;
        let k_num = n_kv * h;
        let extras = 2.0 * (q_num * 2.0 + k_num * 2.0) / s.gpu.mem_bw * 1000.0
            + (k_num * 2.0) / s.gpu.mem_bw * 1000.0
            + (k_num * 2.0) / s.gpu.mem_bw * 1000.0;
        approx(
            context_attention_sol(&op, &s, b, sq, p),
            fmha + extras * 1.1,
        );

        let mut no_rope = op;
        no_rope.apply_rope = false;
        let rope = 2.0 * mem_op_sol_ms(&s, q_num * 2.0 + k_num * 2.0);
        approx(
            context_attention_sol(&no_rope, &s, b, sq, p),
            fmha + (extras - rope) * 1.1,
        );
    }

    #[test]
    fn generation_attention_sol_fp8_kv_uses_fp8_compute() {
        let s = spec();
        let op = GenerationAttentionOp {
            name: "generation_attention".into(),
            scale_factor: 1.0,
            n: 48,
            n_kv: 8,
            head_size: 128,
            window_size: 0,
            kv_cache_dtype: KvCacheQuantMode::Fp8,
            lane_order: crate::operators::attention::b200_vllm_generation_lane_order(),
            use_qk_norm: false,
            scale_num_tokens: 1,
            verify_query_tokens: 0,
            dcp_size: 1,
        };
        let (b, sq) = (256.0, 8441.75_f64);
        let kv_len = sq - 1.0;
        let ops = 2.0 * b * 48.0 * 128.0 * 2.0 * kv_len;
        let mem = b * (48.0 * 128.0 * 2.0 + 2.0 * 8.0 * kv_len * 128.0 * 1.0 + 48.0 * 128.0 * 2.0);
        let expected = (ops / s.gpu.bfloat16_tc_flops.unwrap() * 1000.0 / 2.0)
            .max(mem / s.gpu.mem_bw * 1000.0);
        approx(generation_attention_sol(&op, &s, b, sq), expected);

        let mut normalized = op;
        normalized.use_qk_norm = true;
        let q_num = 48.0 * 128.0;
        let k_num = 8.0 * 128.0;
        let expected_extra = (2.0 * mem_op_sol_ms(&s, q_num * 2.0)
            + 2.0 * mem_op_sol_ms(&s, k_num * 2.0))
            * 2.0
            * 1.1;
        approx(
            generation_attention_sol(&normalized, &s, b, sq),
            expected + expected_extra,
        );
    }

    /// Mirrors moe.py:297-325 with the float-floor association order.
    #[test]
    fn moe_sol_floor_association() {
        let s = spec();
        let op = MoeOp {
            name: "context_moe".into(),
            scale_factor: 1.0,
            hidden_size: 6144,
            inter_size: 1536,
            topk: 8,
            num_experts: 256,
            moe_tp_size: 1,
            moe_ep_size: 4,
            attention_dp_size: 1,
            quant_mode: MoeQuantMode::Nvfp4,
            workload_distribution: "uniform".into(),
            is_gated: true,
            moe_backend: None,
            enable_eplb: false,
            is_context: true,
        };
        let x = 8192.0_f64;
        let tt = x * 8.0;
        let ops = ((tt * 6144.0 * 1536.0 * 3.0 * 2.0 / 4.0).floor() / 1.0).floor();
        let tt_ep = (tt / 4.0).floor();
        let mem = 9.0 / 16.0
            * (tt_ep * 6144.0 * 2.0
                + (tt_ep * 1536.0 * 3.0 / 1.0).floor()
                + (6144.0_f64 * 1536.0 * 3.0 / 1.0).floor() * (256.0_f64 / 4.0).floor().min(tt_ep));
        let expected = (ops / (s.gpu.bfloat16_tc_flops.unwrap() * 4.0) * 1000.0)
            .max(mem / s.gpu.mem_bw * 1000.0);
        approx(moe_sol(&op, &s, x), expected);
    }

    #[test]
    fn comm_sols_match_formulas() {
        let s = spec();
        // custom allreduce: ring, hard-coded 2 B/elem
        let car = CustomAllReduceOp {
            name: "ar".into(),
            scale_factor: 1.0,
            hidden_size: 6144,
            tp_size: 4,
            quant: CommQuantMode::Half,
            seq_split: 1,
        };
        let size = 8192.0 * 6144.0;
        let bw = s.get_p2p_bandwidth(4);
        approx(
            custom_allreduce_op_sol(&car, &s, 8192.0),
            2.0 * size * 2.0 / 4.0 * 3.0 / bw * 1000.0,
        );
        // tp==1 -> 0
        let car1 = CustomAllReduceOp {
            tp_size: 1,
            ..car.clone()
        };
        assert_eq!(custom_allreduce_op_sol(&car1, &s, 8192.0), 0.0);

        // P2P: always inter_node_bw, no latency constant
        let p2p = P2POp {
            name: "p2p".into(),
            scale_factor: 1.0,
            pp_size: 2,
            hidden_size: 6144,
            seq_split: 1,
        };
        approx(
            p2p_sol(&p2p, &s, 8192.0),
            8192.0 * 6144.0 * 2.0 / s.node.inter_node_bw * 1000.0,
        );

        // NCCL all_reduce doubles the gather/scatter traffic
        let nccl = NcclOp {
            name: "nccl".into(),
            scale_factor: 1.0,
            hidden_size: 6144.0,
            num_gpus: 8,
            dtype: CommQuantMode::Half,
            operation: "all_reduce".into(),
            seq_split: 1,
        };
        let bw8 = s.get_p2p_bandwidth(8);
        approx(
            nccl_op_sol(&nccl, &s, 1024.0),
            2.0 * 2.0 * (1024.0 * 6144.0) * 7.0 / 8.0 / bw8 * 1000.0,
        );
    }

    #[test]
    fn moe_dispatch_vllm_is_additive() {
        let s = spec();
        let op = MoEDispatchOp {
            name: "context_moe_pre_dispatch".into(),
            scale_factor: 1.0,
            hidden_size: 6144,
            topk: 8,
            num_experts: 256,
            moe_tp_size: 1,
            moe_ep_size: 4,
            attention_dp_size: 1,
            pre_dispatch: true,
            backend: BackendKind::Vllm,
            flavor: crate::operators::moe_dispatch::DispatchFlavor::CustomAllReduce,
            comm_quant: CommQuantMode::Half,
            moe_quant: MoeQuantMode::Nvfp4,
            attn_cp_size: 1,
            is_context: true,
            sms: 12,
            scale_num_tokens: 1,
            attn_ar_modeled: false,
        };
        // dp=1, attn_tp = 4/1 = 4 > 1 -> allreduce only
        let volume = 8192.0 * 6144.0;
        let bw = s.get_p2p_bandwidth(4);
        approx(
            moe_dispatch_sol(&op, &s, 8192.0).unwrap(),
            2.0 * volume * 2.0 / 4.0 * 3.0 / bw * 1000.0,
        );
    }

    #[test]
    fn overlap_and_fallback_compose() {
        let d = db();
        let ew = |bpt: f64| {
            Op::Elementwise(ElementwiseOp {
                name: "e".into(),
                scale_factor: 1.0,
                bytes_per_token: bpt,
                scale_num_tokens: 1,
                seq_split: 1,
            })
        };
        let overlap = Op::Overlap(crate::operators::OverlapOp::new(
            "ov",
            vec![ew(1000.0), ew(2000.0)],
            vec![ew(5000.0)],
        ));
        let expected = super::mem_op_sol_ms(&d.system_spec, 5000.0 * 64.0);
        approx(
            op_sol_latency_ms(&overlap, &d, 64.0, 1.0, 1.0, 0.0).unwrap(),
            expected,
        );
        let fb = Op::Fallback(crate::operators::FallbackOp::new("fb", ew(3000.0), vec![]));
        approx(
            op_sol_latency_ms(&fb, &d, 64.0, 1.0, 1.0, 0.0).unwrap(),
            super::mem_op_sol_ms(&d.system_spec, 3000.0 * 64.0),
        );
    }

    #[test]
    fn unsupported_fpm_sol_op_has_typed_error() {
        let d = db();
        let op = Op::Mamba2(crate::operators::Mamba2Op {
            name: "mamba2".into(),
            scale_factor: 1.0,
            kernel_source: "causal_conv1d_fn".into(),
            phase: "context".into(),
            d_model: 4096,
            d_state: 128,
            d_conv: 4,
            nheads: 128,
            head_dim: 64,
            n_groups: 8,
            chunk_size: 256,
        });

        let err = op_sol_latency_ms(&op, &d, 64.0, 1.0, 1.0, 0.0).unwrap_err();
        assert!(matches!(&err, AicError::SolNotImplemented(_)));
        assert!(
            err.to_string()
                .contains("no Rust implementation for op mamba2")
        );
    }

    /// Decode CP in the whole-forward SOL leaf: `n * dcp` gathered heads over
    /// a `ceil(s / dcp)` KV stripe, identical to the un-striped equivalent.
    #[test]
    fn generation_attention_fpm_sol_applies_dcp_geometry() {
        let d = db();
        let mut striped =
            GenerationAttentionOp::new("generation_attention", 12, 1, 128, KvCacheQuantMode::Fp8);
        striped.dcp_size = 4;
        let gathered =
            GenerationAttentionOp::new("generation_attention", 48, 1, 128, KvCacheQuantMode::Fp8);
        let plain =
            GenerationAttentionOp::new("generation_attention", 12, 1, 128, KvCacheQuantMode::Fp8);
        let a = op_sol_latency_ms(
            &Op::GenerationAttention(striped),
            &d,
            256.0,
            256.0,
            8192.0,
            0.0,
        )
        .unwrap();
        let b = op_sol_latency_ms(
            &Op::GenerationAttention(gathered),
            &d,
            256.0,
            256.0,
            2048.0,
            0.0,
        )
        .unwrap();
        let c = op_sol_latency_ms(
            &Op::GenerationAttention(plain),
            &d,
            256.0,
            256.0,
            8192.0,
            0.0,
        )
        .unwrap();
        assert!((a - b).abs() < 1e-9, "striped {a} vs gathered {b}");
        assert!(
            a < c,
            "dcp must shrink the KV-read-bound decode roofline: {a} vs {c}"
        );
    }

    fn glm_dsa_op(name: &str) -> DsaModuleOp {
        // nvidia/GLM-5.2-NVFP4: 64 heads, fp8 KV cache, bf16 context FMHA, nvfp4 GEMMs, index_topk 2048
        DsaModuleOp::new(
            name,
            64,
            KvCacheQuantMode::Fp8,
            FmhaQuantMode::Bfloat16,
            GemmQuantMode::Nvfp4,
            "GlmMoeDsaForCausalLM",
            2048,
        )
    }

    #[test]
    fn dsa_context_fpm_sol_reuses_the_dsa_roofline() {
        let d = db();
        let spec = &d.system_spec;
        let op = glm_dsa_op("context_attention");
        // FPM prefill coords: batch 1, 8192 new tokens, 24576 cached tokens (a chunk boundary
        // that is not a collected site in the GLM cells).
        let got = op_sol_latency_ms(&Op::DsaContext(op), &d, 8192.0, 1.0, 8192.0, 24576.0).unwrap();
        let flops =
            dsa_context_sol_flops(spec, GemmQuantMode::Nvfp4, FmhaQuantMode::Bfloat16).unwrap();
        let expected = dsa_context_sol_ms(
            spec,
            dsa_dims("GlmMoeDsaForCausalLM"),
            2048,
            KvCacheQuantMode::Fp8,
            FmhaQuantMode::Bfloat16,
            GemmQuantMode::Nvfp4,
            1,
            8192,
            24576,
            64,
            false,
            flops,
        );
        assert!(got.is_finite() && got > 0.0, "{got}");
        approx(got, expected);
    }

    #[test]
    fn dsa_context_fpm_sol_blends_skip_indexer_by_full_frac_and_scales() {
        let d = db();
        let spec = &d.system_spec;
        let mut op = glm_dsa_op("context_attention");
        op.full_frac = 0.25;
        op.scale_factor = 1.5;
        let got = op_sol_latency_ms(&Op::DsaContext(op), &d, 8192.0, 2.0, 4096.0, 65536.0).unwrap();
        let flops =
            dsa_context_sol_flops(spec, GemmQuantMode::Nvfp4, FmhaQuantMode::Bfloat16).unwrap();
        let sol = |skip: bool| {
            dsa_context_sol_ms(
                spec,
                dsa_dims("GlmMoeDsaForCausalLM"),
                2048,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Nvfp4,
                2,
                4096,
                65536,
                64,
                skip,
                flops,
            )
        };
        approx(got, (0.25 * sol(false) + 0.75 * sol(true)) * 1.5);
    }

    #[test]
    fn dsa_generation_fpm_sol_reuses_the_dsa_roofline() {
        let d = db();
        let spec = &d.system_spec;
        let op = glm_dsa_op("generation_attention");
        // FPM decode coords: batch 8, 100000 KV tokens per request (x is the batch for decode).
        let got = op_sol_latency_ms(&Op::DsaGeneration(op), &d, 8.0, 8.0, 100000.0, 0.0).unwrap();
        let flops = dsa_generation_sol_flops(spec, GemmQuantMode::Nvfp4).unwrap();
        let expected = dsa_generation_sol_ms(
            spec,
            dsa_dims("GlmMoeDsaForCausalLM"),
            KvCacheQuantMode::Fp8,
            GemmQuantMode::Nvfp4,
            8,
            100000,
            64,
            flops,
        );
        assert!(got.is_finite() && got > 0.0, "{got}");
        approx(got, expected);
    }

    /// FROZEN parity record for the DSA context arm at FPM totals that are NOT
    /// divisible by the batch. `sol_total` back-maps the raw prefill totals
    /// (batch=2, total_prefill=8193, total_kv=2049) to `s = 8193/2 = 4096.5` and
    /// `prefix = 2049/2 = 1024.5`, which the arm rounds with `f64::round`'s
    /// half-AWAY-from-zero tie-break to `s = 4097`, `prefix = 1025` (half-to-even
    /// would give 4096/1024 and 1.3674139411342223 ms instead).
    ///
    /// Expected value derived by hand from the `dsa_context_sol` terms
    /// (`perf_database/dsa.rs:771-867`) at b=2, s=4097, prefix=1025, num_heads=64,
    /// index_topk=2048, `GLM_MOE_DSA_DIMS` (hidden 6144, q_lora 2048, kv_lora 512,
    /// qk_nope 192, qk_rope 64, v 256, index 32x128) on b200_sxm (mem_bw 7.7e12,
    /// fp4 9e15, fp8 4.5e15, bf16 2.25e15), nvfp4 GEMM weights (9/16 B/elem),
    /// fp8 KV (1 B/elem), bf16 context FMHA (2 B/elem):
    ///   full_s = 4097 + 1025 = 5122 > 2048 = topk, prefix 1025 < topk
    ///     -> the ramp+saturation KV-pair branch (`dsa.rs:824-832`):
    ///        ramp = 2*(2048*2049 - 1025*1026)/2 =  3_144_702
    ///        sat  = 2*(5122 - 2048)*2048        = 12_591_104
    ///        total_kv_pairs                     = 15_735_806
    ///   gemm_group_ops     = 2_857_924_558_848 / 9.0e15
    ///   indexer_logits_ops =   343_815_520_256 / 4.5e15   (= 2*8194*32*128*5122)
    ///   sparse_attn_ops    = 2_191_431_286_784 / 2.25e15  (= 2*64*(576+512)*pairs)
    ///   sol_math = (that sum) * 1000                      = 1.3679200829440001 ms
    ///   sol_mem  = (89_837_568 + 150_994_944 + 1_352_208 + 537_001_984) B
    ///            = 779_186_704 / 7.7e12 * 1000            = 0.1011930784415... ms
    ///   time_ms  = max(math, mem)                         -> math-bound
    #[test]
    fn dsa_context_fpm_sol_frozen_at_fractional_fpm_coordinates() {
        let d = db();
        let op = glm_dsa_op("context_attention");
        let got = op_sol_latency_ms(&Op::DsaContext(op), &d, 8193.0, 2.0, 4096.5, 1024.5).unwrap();
        approx(got, 1.3679200829440001);
    }

    /// FROZEN parity record for the DSA generation arm at a decode total that is
    /// NOT divisible by the batch: `sol_total` back-maps (batch=6,
    /// total_kv=393219) to `s = 393219/6 = 65536.5`, which rounds half-away-from-zero
    /// to 65537 (half-to-even would give 65536 and 0.019327268571428573 ms).
    ///
    /// Expected value derived by hand from the `dsa_generation_sol` terms
    /// (`perf_database/dsa.rs:907-973`) at b=6, s=65537, num_heads=64, same dims,
    /// system and quant modes as the context record above. Decode clamps the
    /// attention window with the DIMS top-k, not the op's:
    /// `effective_kv = min(65537, 2048) = 2048`.
    ///   gemm_group_ops     = 2_092_695_552 / 9.0e15
    ///   indexer_logits_ops = 3_221_274_624 / 4.5e15   (= 2*6*32*128*65537)
    ///   sparse_attn_ops    = 1_711_276_032 / 2.25e15  (= 2*6*64*(576+512)*2048)
    ///   sol_math = (that sum) * 1000 = 0.00170892765866... ms
    ///   sol_mem: weights  159_711_232 elems * 9/16      =  89_837_568 B
    ///            indexer  6*65537*132 (entry_bytes 128) =  51_905_304 B
    ///            kv       6*2048*576 * 1                =   7_077_888 B
    ///            total   148_820_760 / 7.7e12 * 1000    = 0.019327371428571428 ms
    ///   time_ms  = max(math, mem) -> memory-bound, so the frozen literal pins the
    ///   generation memory model and the rounded `s` through the indexer cache term.
    #[test]
    fn dsa_generation_fpm_sol_frozen_at_fractional_fpm_coordinates() {
        let d = db();
        let op = glm_dsa_op("generation_attention");
        let got = op_sol_latency_ms(&Op::DsaGeneration(op), &d, 6.0, 6.0, 65536.5, 0.0).unwrap();
        approx(got, 0.019327371428571428);
    }

    /// `full_frac` domain. The two valid boundaries select the pure legs, and every
    /// value outside [0, 1] is a typed `InvalidEngineConfig` rather than an
    /// extrapolated blend (negative / > 1) or a silent 0 (NaN, -inf) — see the gate
    /// in `dsa_context_module_sol`.
    #[test]
    fn dsa_context_fpm_sol_full_frac_boundaries_and_domain_gate() {
        let d = db();
        let spec = &d.system_spec;
        let flops =
            dsa_context_sol_flops(spec, GemmQuantMode::Nvfp4, FmhaQuantMode::Bfloat16).unwrap();
        let leg = |skip: bool| {
            dsa_context_sol_ms(
                spec,
                dsa_dims("GlmMoeDsaForCausalLM"),
                2048,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Nvfp4,
                2,
                4096,
                65536,
                64,
                skip,
                flops,
            )
        };
        let at = |w: f64| {
            let mut op = glm_dsa_op("context_attention");
            op.full_frac = w;
            op_sol_latency_ms(&Op::DsaContext(op), &d, 8192.0, 2.0, 4096.0, 65536.0)
        };

        let full = at(1.0).unwrap();
        let skip = at(0.0).unwrap();
        assert!(full.is_finite() && full > 0.0, "{full}");
        assert!(skip.is_finite() && skip > 0.0, "{skip}");
        approx(full, leg(false));
        approx(skip, leg(true));
        // The skip leg drops the per-layer indexer, so it is strictly cheaper —
        // proof the two boundaries are not the same code path.
        assert!(
            skip < full,
            "skip-only {skip} must be below full-only {full}"
        );

        for bad in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                matches!(at(bad), Err(AicError::InvalidEngineConfig(_))),
                "full_frac={bad} must be rejected, got {:?}",
                at(bad)
            );
        }
    }

    #[test]
    fn dsa_context_cp_keeps_the_typed_sol_not_implemented() {
        let d = db();
        let mut op = glm_dsa_op("context_attention");
        op.cp_size = 2;
        let err = op_sol_latency_ms(&Op::DsaContext(op), &d, 8192.0, 1.0, 8192.0, 0.0).unwrap_err();
        assert!(matches!(&err, AicError::SolNotImplemented(_)));
        assert!(err.to_string().contains("cp_size=2"));
    }
}
