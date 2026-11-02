// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified native port of heterogeneous AFD/fabric behavior; source revision and
// original paths: root THIRD_PARTY_NOTICES.md (AISimulate heterogeneous AFD).
// Source: https://github.com/LPhgh/aisimulate/tree/af56328c4ba7ffa1b8c42c8375244568c2e7bb8c

//! AFD cross-pool transfers and intra-pool collectives. Message shapes compose
//! the native P2P, NCCL and elementwise operators; data lookup stays in those ops.

use serde::{Deserialize, Serialize};

use crate::common::enums::CommQuantMode;
use crate::common::error::AicError;
use crate::operators::communication::{NcclOp, P2POp};
use crate::operators::elementwise::ElementwiseOp;
use crate::operators::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;

/// Cross-pool transfer leg. Serde values match the Python `direction`
/// strings (`"a2f"` / `"f2a"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AfdTransferDirection {
    A2f,
    F2a,
}

/// A-rank <-> F-rank mapping. Serde values match the Python `rank_mapping`
/// strings (`"one_to_one"` / `"broadcast"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AfdRankMapping {
    OneToOne,
    Broadcast,
}

/// `comb(n, k)` in exact u128 arithmetic (Python `math.comb`), iterative so
/// every intermediate is itself a binomial coefficient. Returns `None` on
/// u128 overflow (only reachable at absurd expert/topk counts; callers fall
/// back to the f64 product ratio, <1e-12 relative error).
fn comb_exact(n: u32, k: u32) -> Option<u128> {
    let k = k.min(n.checked_sub(k)?);
    let mut result: u128 = 1;
    for i in 0..k {
        result = result.checked_mul(u128::from(n - i))?;
        result /= u128::from(i + 1);
    }
    Some(result)
}

/// `comb(a, k) / comb(b, k)` as an f64 product of per-factor ratios.
/// Mathematically identical to the exact quotient; used only when the exact
/// u128 route overflows.
fn comb_ratio_f64(a: u32, b: u32, k: u32) -> f64 {
    let mut ratio = 1.0f64;
    for i in 0..k {
        ratio *= f64::from(a - i) / f64::from(b - i);
    }
    ratio
}

/// Probability that a token must be dispatched to a given F-node.
///
/// Mirrors Python `_afd_send_prob`: MoE with expert parallelism uses
/// `1 - C(E - E/Nf, k) / C(E, k)`; dense models and degenerate configs use
/// `1 / num_f_nodes`.
pub(crate) fn afd_send_prob(num_experts: u32, topk: u32, num_f_nodes: u32) -> f64 {
    if num_experts == 0 || topk == 0 || num_f_nodes <= 1 {
        return 1.0 / f64::from(num_f_nodes.max(1));
    }
    let experts_per_node = num_experts / num_f_nodes;
    if experts_per_node == 0 {
        return 1.0 / f64::from(num_f_nodes);
    }
    let n_other = num_experts - experts_per_node;
    if topk > n_other {
        return 1.0;
    }
    let ratio = match (comb_exact(n_other, topk), comb_exact(num_experts, topk)) {
        (Some(other), Some(all)) => other as f64 / all as f64,
        _ => comb_ratio_f64(n_other, num_experts, topk),
    };
    1.0 - ratio
}

/// Prepared modulo rank mapping. A rank `replica * TP + slice` sends to
/// `rank % f_node_width` on every F node. Preparation visits one periodic
/// mapping, independent of the number of A nodes. Warm queries are O(1).
#[derive(Debug)]
pub(crate) struct AfdTokenMapping {
    tp: u32,
    counts: [u64; 2],
    // Maximum number of larger slices among slots with floor/ceil rank count.
    extras: Vec<[u64; 2]>,
    replicas: u32,
    period: u32,
    slots: Vec<Vec<(u32, u32)>>,
}

impl AfdTokenMapping {
    pub(crate) fn new(replicas: u32, tp: u32, f_width: u32) -> Result<Self, AicError> {
        if tp == 0 || f_width == 0 {
            return Err(AicError::InvalidEngineConfig(
                "invalid AFD rank mapping".into(),
            ));
        }
        let ranks = u64::from(replicas) * u64::from(tp);
        let width = u64::from(f_width);
        let counts = [ranks / width, ranks.div_ceil(width)];
        let gcd = {
            let (mut a, mut b) = (tp, f_width);
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let period_replicas = f_width / gcd;
        let mut grouped_slots = vec![Vec::new(); f_width as usize];
        for replica in 0..period_replicas {
            for slice in 0..tp {
                grouped_slots
                    [((u64::from(replica) * u64::from(tp) + u64::from(slice)) % width) as usize]
                    .push((replica, slice));
            }
        }
        if f_width % tp == 0 && ranks % width == 0 {
            return Ok(Self {
                tp,
                counts,
                extras: Vec::new(),
                replicas,
                period: period_replicas,
                slots: grouped_slots,
            });
        }
        let mut extras = vec![[0; 2]; tp as usize];
        // A TP group repeats every tp ranks; F destinations repeat every
        // f_width ranks. Count the joint period instead of visiting all nodes.
        let mut x = tp;
        let mut y = f_width;
        while y != 0 {
            (x, y) = (y, x % y);
        }
        let period = u64::from(tp) * width / u64::from(x);
        let mut slots = vec![vec![0_u64; tp as usize]; f_width as usize];
        for rank in 0..period.min(ranks) {
            slots[(rank % width) as usize][(rank % u64::from(tp)) as usize] +=
                ranks / period + u64::from(rank < ranks % period);
        }
        for (slot, slices) in slots.into_iter().enumerate() {
            let count = ranks / width + u64::from((slot as u64) < ranks % width);
            let class = usize::from(count == counts[1]);
            let mut prefix = 0;
            for remainder in 1..tp as usize {
                prefix += slices[remainder - 1];
                extras[remainder][class] = extras[remainder][class].max(prefix);
            }
        }
        Ok(Self {
            tp,
            counts,
            extras,
            replicas,
            period: period_replicas,
            slots: grouped_slots,
        })
    }

    /// Exact mapping for two contiguous replica populations. Cost is bounded
    /// by TP and F-node width, independent of request/A-node count.
    pub(crate) fn busiest_groups(&self, groups: [(u32, u32); 2]) -> u64 {
        debug_assert_eq!(groups.iter().map(|(_, n)| n).sum::<u32>(), self.replicas);
        let prefix_count = |n: u32, first: u32| {
            if n > first {
                u64::from((n - 1 - first) / self.period + 1)
            } else {
                0
            }
        };
        self.slots
            .iter()
            .map(|slots| {
                slots
                    .iter()
                    .map(|&(first, slice)| {
                        let mut start = 0;
                        let mut total = 0;
                        for (tokens, n) in groups {
                            let copies =
                                prefix_count(start + n, first) - prefix_count(start, first);
                            total += copies
                                * u64::from(tokens / self.tp + u32::from(slice < tokens % self.tp));
                            start += n;
                        }
                        total
                    })
                    .sum::<u64>()
            })
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn busiest(&self, tokens: u32) -> u64 {
        if self.extras.is_empty() {
            return u64::from(tokens.div_ceil(self.tp)) * self.counts[1];
        }
        let whole = u64::from(tokens / self.tp);
        let extras = self.extras[(tokens % self.tp) as usize];
        (whole * self.counts[0] + extras[0]).max(whole * self.counts[1] + extras[1])
    }
}

pub(crate) fn mapped_rank_tokens(
    tokens: u32,
    replicas: u32,
    tp: u32,
    node_width: u32,
) -> Result<u64, AicError> {
    // Preserve the old allocation-free common case for standalone ops.
    if tp > 0 && node_width > 0 && node_width % tp == 0 {
        let ranks = u64::from(replicas) * u64::from(tp);
        if ranks % u64::from(node_width) == 0 {
            return Ok(u64::from(tokens.div_ceil(tp)) * (ranks / u64::from(node_width)));
        }
    }
    Ok(AfdTokenMapping::new(replicas, tp, node_width)?.busiest(tokens))
}

fn empirical_zero() -> PerformanceResult {
    PerformanceResult::new(0.0, Source::Empirical)
}

/// `ceil(value / 2)` as u32 for a P2P twin's element count, erroring instead
/// of truncating when the volume exceeds the twin's u32 `hidden_size`
/// (unreachable for memory-feasible AFD topologies — the per-link volume
/// would imply terabytes of activations).
fn p2p_elements(per_link_bytes: u64) -> Result<u32, AicError> {
    let elements = per_link_bytes.div_ceil(2);
    u32::try_from(elements).map_err(|_| {
        AicError::InvalidEngineConfig(format!(
            "AFD per-link transfer volume {per_link_bytes} bytes exceeds the P2P twin's u32 element range"
        ))
    })
}

/// Immutable per-endpoint link parameters, resolved at model construction.
/// Keep the endpoints separate: max(bytes/bw + latency) is not generally
/// bytes/min(bw) + max(latency), which can invent a slower third endpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AfdFabricLink {
    pub inter_node_bw: f64,
    pub p2p_latency: f64,
    pub peer: Option<(f64, f64)>,
}

impl AfdFabricLink {
    pub fn between(a: (f64, f64), f: (f64, f64)) -> Self {
        Self {
            inter_node_bw: a.0.max(1.0),
            p2p_latency: a.1,
            peer: (a != f).then_some((f.0.max(1.0), f.1)),
        }
    }
}

/// Unidirectional cross-pool P2P transfer (A->F **or** F->A).
///
/// Port of Python `AFDTransfer`. A-side operates in DP mode: each A replica
/// holds its own independent tokens and sends/receives them with the full
/// `hidden_size` per token, so the per-link payload is symmetric for both
/// directions. `direction` declares which leg this instance models.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AfdTransferOp {
    pub name: String,
    pub scale_factor: f64,
    pub direction: AfdTransferDirection,
    pub hidden_size: u32,
    pub a_replica_count: u32,
    pub f_rank_count: u32,
    pub gpus_per_node: u32,
    #[serde(default)]
    pub f_gpus_per_node: Option<u32>,
    pub num_experts: u32,
    pub topk: u32,
    pub comm_quant_mode: CommQuantMode,
    /// Applied to LATENCY only, never energy (Python multiplies
    /// `float(result)`). Measurement-side instances carry 1.0 — the
    /// pipeline evaluator applies the configured factor exactly once.
    pub comm_overhead_factor: f64,
}

impl AfdTransferOp {
    /// Token-DP senders within an A TP group. System inter-node bandwidth is
    /// per GPU/NIC: one sender shares that budget across all F destinations,
    /// and each F node shares its local NICs across every A replica. Price the
    /// larger injection/receive load with the existing P2P oracle, retaining
    /// its propagation term and database-mode behavior.
    pub fn query_token_partitioned(
        &self,
        db: &PerfDatabase,
        tokens_per_a_replica: u32,
        a_tp_size: u32,
        fabric: Option<AfdFabricLink>,
    ) -> Result<PerformanceResult, AicError> {
        let receive = mapped_rank_tokens(
            tokens_per_a_replica,
            self.a_replica_count,
            a_tp_size,
            self.f_gpus_per_node.unwrap_or(self.gpus_per_node),
        )?;
        self.query_with_receiver_load(db, tokens_per_a_replica, a_tp_size, receive, fabric)
    }

    pub(crate) fn query_with_receiver_load(
        &self,
        db: &PerfDatabase,
        tokens_per_a_replica: u32,
        a_tp_size: u32,
        receive: u64,
        fabric: Option<AfdFabricLink>,
    ) -> Result<PerformanceResult, AicError> {
        crate::perfmodel::afd::validate_afd_attention_tp(a_tp_size)?;
        let sender =
            u64::from(tokens_per_a_replica.div_ceil(a_tp_size)) * u64::from(self.num_f_nodes());
        let tokens = u32::try_from(sender.max(receive)).map_err(|_| {
            AicError::InvalidEngineConfig("AFD per-NIC token payload exceeds u32 range".into())
        })?;
        self.query_with_fabric(db, tokens, fabric)
    }

    fn clamped(&self) -> (u32, u32, u32) {
        // Python clamps in __init__ via max(..., 1) / max(..., 0).
        (
            self.a_replica_count.max(1),
            self.f_rank_count.max(1),
            self.f_gpus_per_node.unwrap_or(self.gpus_per_node).max(1),
        )
    }

    /// Physical F-node count: `ceil(f_rank_count / gpus_per_node)`, min 1.
    pub fn num_f_nodes(&self) -> u32 {
        let (_, f_rank_count, gpus_per_node) = self.clamped();
        f_rank_count.div_ceil(gpus_per_node)
    }

    /// `tokens_per_a_replica` = tokens held by a single A replica.
    pub fn query(
        &self,
        db: &PerfDatabase,
        tokens_per_a_replica: u32,
    ) -> Result<PerformanceResult, AicError> {
        self.query_with_fabric(db, tokens_per_a_replica, None)
    }

    /// [`Self::query`] with an optional cross-SKU fabric override. `None`
    /// composes the standard P2P twin against `db` (same-system behavior);
    /// `Some(link)` evaluates the twin's exact formula with the link's
    /// bandwidth/latency substituted (SOL and non-SOL branches mirrored).
    pub fn query_with_fabric(
        &self,
        db: &PerfDatabase,
        tokens_per_a_replica: u32,
        fabric: Option<AfdFabricLink>,
    ) -> Result<PerformanceResult, AicError> {
        if tokens_per_a_replica == 0 {
            return Ok(empirical_zero());
        }
        let nf = self.num_f_nodes();
        let p_send = afd_send_prob(self.num_experts, self.topk, nf);
        let bpe = self.comm_quant_mode.mapping().memory;
        // Python: int(p_send * tokens_per_a_replica * hidden_size * bpe) — left-associative
        // float product, then truncation toward zero.
        let per_link_bytes =
            (p_send * f64::from(tokens_per_a_replica) * f64::from(self.hidden_size) * bpe) as u64;
        if per_link_bytes == 0 {
            return Ok(empirical_zero());
        }
        let twin = P2POp::new("afd_p2p", 2, p2p_elements(per_link_bytes)?);
        let mut result = match fabric {
            None => twin.query(db, 1)?,
            Some(link) => twin.query_with_link(db, 1, (link.inter_node_bw, link.p2p_latency))?,
        };
        if let Some(peer) = fabric.and_then(|link| link.peer) {
            let other = twin.query_with_link(db, 1, peer)?;
            if other.latency_ms > result.latency_ms {
                result = other;
            }
        }
        let latency = result.latency_ms * self.comm_overhead_factor * self.scale_factor;
        Ok(PerformanceResult::with_energy(
            latency,
            result.energy_wms * self.scale_factor,
            result.source,
        ))
    }
}

/// Shared shape of the two F-node intra-node collectives (AllGather before
/// F compute, ReduceScatter after it). Both bill by token volume along the
/// token dimension and return 0 outside one-to-one rank mapping or when the
/// node holds a single F-GPU.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AfdFCollectiveOp {
    pub name: String,
    pub scale_factor: f64,
    pub hidden_size: u32,
    pub a_replica_count: u32,
    pub f_rank_count: u32,
    pub gpus_per_node: u32,
    pub num_experts: u32,
    pub topk: u32,
    pub comm_quant_mode: CommQuantMode,
    pub rank_mapping: AfdRankMapping,
    /// NCCL operation of the twin: `"all_gather"` or `"reduce_scatter"`.
    pub operation: String,
}

impl AfdFCollectiveOp {
    /// Equal-count collectives pad to the busiest mapped F rank. Logical
    /// expert compute still consumes only the unpadded token population.
    pub fn query_token_partitioned(
        &self,
        db: &PerfDatabase,
        tokens_per_a_replica: u32,
        a_tp_size: u32,
    ) -> Result<PerformanceResult, AicError> {
        let tokens = mapped_rank_tokens(
            tokens_per_a_replica,
            self.a_replica_count,
            a_tp_size,
            self.gpus_per_node,
        )?;
        self.query_with_receiver_load(db, tokens)
    }

    pub(crate) fn query_with_receiver_load(
        &self,
        db: &PerfDatabase,
        tokens: u64,
    ) -> Result<PerformanceResult, AicError> {
        let padded =
            (afd_send_prob(self.num_experts, self.topk, self.num_f_nodes()) * tokens as f64).ceil();
        self.query_rank_elements(db, padded * f64::from(self.hidden_size))
    }
    /// AllGather flavor (Python `AFDFAllGather`): collects token subsets
    /// across the node's F-GPUs before F compute.
    pub fn all_gather(name: impl Into<String>, scale_factor: f64) -> Self {
        Self::with_operation(name, scale_factor, "all_gather")
    }

    /// ReduceScatter flavor (Python `AFDFReduceScatter`): scatters each
    /// A-rank's tokens onto the corresponding F-GPU after F compute.
    pub fn reduce_scatter(name: impl Into<String>, scale_factor: f64) -> Self {
        Self::with_operation(name, scale_factor, "reduce_scatter")
    }

    fn with_operation(name: impl Into<String>, scale_factor: f64, operation: &str) -> Self {
        Self {
            name: name.into(),
            scale_factor,
            hidden_size: 0,
            a_replica_count: 1,
            f_rank_count: 1,
            gpus_per_node: 8,
            num_experts: 0,
            topk: 0,
            comm_quant_mode: CommQuantMode::Half,
            rank_mapping: AfdRankMapping::OneToOne,
            operation: operation.into(),
        }
    }

    fn clamped(&self) -> (u32, u32, u32) {
        (
            self.a_replica_count.max(1),
            self.f_rank_count.max(1),
            self.gpus_per_node.max(1),
        )
    }

    pub fn num_f_nodes(&self) -> u32 {
        let (_, f_rank_count, gpus_per_node) = self.clamped();
        f_rank_count.div_ceil(gpus_per_node)
    }

    /// Number of F-GPUs within a single node.
    pub fn f_gpus_in_node(&self) -> u32 {
        let (_, f_rank_count, gpus_per_node) = self.clamped();
        f_rank_count.min(gpus_per_node)
    }

    /// `tokens_per_a_replica` = tokens held by a single A replica.
    pub fn query(
        &self,
        db: &PerfDatabase,
        tokens_per_a_replica: u32,
    ) -> Result<PerformanceResult, AicError> {
        let f_local = self.f_gpus_in_node();
        if f_local <= 1 || self.rank_mapping != AfdRankMapping::OneToOne {
            return Ok(empirical_zero());
        }
        if tokens_per_a_replica == 0 {
            return Ok(empirical_zero());
        }
        let (a_replica_count, _, _) = self.clamped();
        // Python: total = tokens_per_a_replica * a_replica_count (exact int); tokens_per_f_node =
        // p_send * total; per_rank_elements = int(tokens * hidden / f_local).
        let total = u64::from(tokens_per_a_replica) * u64::from(a_replica_count);
        let nf = self.num_f_nodes();
        let p_send = afd_send_prob(self.num_experts, self.topk, nf);
        let tokens_per_f_node = p_send * total as f64;
        let per_rank_elements =
            (tokens_per_f_node * f64::from(self.hidden_size) / f64::from(f_local)) as u64;
        if per_rank_elements == 0 {
            return Ok(empirical_zero());
        }
        self.query_rank_elements(db, per_rank_elements as f64)
    }

    fn query_rank_elements(
        &self,
        db: &PerfDatabase,
        elements: f64,
    ) -> Result<PerformanceResult, AicError> {
        if elements == 0.0 || self.f_gpus_in_node() <= 1 {
            return Ok(empirical_zero());
        }
        let twin = NcclOp {
            name: format!("afd_{}", self.operation),
            scale_factor: 1.0,
            hidden_size: elements,
            num_gpus: self.f_gpus_in_node(),
            dtype: self.comm_quant_mode,
            operation: self.operation.clone(),
            seq_split: 1,
        };
        let result = twin.query(db, 1)?;
        Ok(PerformanceResult::with_energy(
            result.latency_ms * self.scale_factor,
            result.energy_wms * self.scale_factor,
            result.source,
        ))
    }
}

/// F-node intra-node AllGather along the token dimension before F compute
/// (Python `AFDFAllGather`).
pub type AfdFAllGatherOp = AfdFCollectiveOp;

/// F-node intra-node ReduceScatter after F compute (Python
/// `AFDFReduceScatter`).
pub type AfdFReduceScatterOp = AfdFCollectiveOp;

/// A-side cross-EP combine: local HBM reduce-add of partial results.
///
/// Port of Python `AFDCombine`. When the F side uses expert parallelism
/// (`f_moe_ep_size > 1`), each A replica receives `f_moe_ep_size` partial
/// results and reduces them locally. Returns 0 for dense FFN.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AfdCombineOp {
    pub name: String,
    pub scale_factor: f64,
    pub hidden_size: u32,
    pub a_tp_size: u32,
    pub f_moe_ep_size: u32,
    pub comm_quant_mode: CommQuantMode,
}

impl AfdCombineOp {
    /// `tokens_per_a_replica` = tokens held by a single A replica.
    pub fn query(
        &self,
        db: &PerfDatabase,
        tokens_per_a_replica: u32,
    ) -> Result<PerformanceResult, AicError> {
        crate::perfmodel::afd::validate_afd_attention_tp(self.a_tp_size)?;
        let f_moe_ep_size = self.f_moe_ep_size;
        if f_moe_ep_size <= 1 {
            return Ok(empirical_zero());
        }
        if tokens_per_a_replica == 0 {
            return Ok(empirical_zero());
        }
        let a_tp_size = self.a_tp_size.max(1);
        // Python: tokens_per_a_replica = ceil(tokens_per_a_replica / a_tp_size) (exact int math);
        // total_bytes = int((f_moe_ep_size + 1) * tokens * hidden * bpe).
        let tokens_per_a_replica = u64::from(tokens_per_a_replica).div_ceil(u64::from(a_tp_size));
        let bpe = self.comm_quant_mode.mapping().memory;
        let total_bytes =
            (u64::from(f_moe_ep_size) + 1) * tokens_per_a_replica * u64::from(self.hidden_size);
        let total_bytes = (total_bytes as f64 * bpe) as u64;
        if total_bytes == 0 {
            return Ok(empirical_zero());
        }
        // ElementWise twin: dim_in = ceil(total_bytes/2), dim_out = 0, so
        // bytes_per_token = 2 * ceil(total_bytes/2) — byte-exact to +1 byte.
        let twin = ElementwiseOp {
            name: "afd_combine_mem".into(),
            scale_factor: 1.0,
            bytes_per_token: 2.0 * total_bytes.div_ceil(2) as f64,
            scale_num_tokens: 1,
            seq_split: 1,
        };
        let result = twin.query(db, 1)?;
        Ok(PerformanceResult::with_energy(
            result.latency_ms * self.scale_factor,
            result.energy_wms * self.scale_factor,
            result.source,
        ))
    }
}

/// Zero-check helper for tests: every zero branch must return an
/// empirical-tagged zero (Python `PerformanceResult(0.0, 0.0,
/// source="empirical")`).
#[cfg(test)]
fn assert_empirical_zero(result: &PerformanceResult) {
    assert_eq!(result.latency_ms, 0.0);
    assert_eq!(result.energy_wms, 0.0);
    assert_eq!(result.source, Source::Empirical);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_mapping_matches_explicit_replica_rank_population() {
        for r in [1, 3, 8, 17, 64] {
            for tp in [1, 2, 4] {
                for w in [1, 3, 8] {
                    let mapping = AfdTokenMapping::new(r, tp, w).unwrap();
                    for extra in 0..=r {
                        for (hi, lo) in [(1, 0), (7, 3), (16, 12)] {
                            let mut slots = vec![0u64; w as usize];
                            for replica in 0..r {
                                let tokens = if replica < extra { hi } else { lo };
                                for slice in 0..tp {
                                    slots[((replica * tp + slice) % w) as usize] +=
                                        u64::from(tokens / tp + u32::from(slice < tokens % tp));
                                }
                            }
                            assert_eq!(
                                mapping.busiest_groups([(hi, extra), (lo, r - extra)]),
                                *slots.iter().max().unwrap()
                            );
                        }
                    }
                }
            }
        }
    }

    use crate::perf_database::energy_test_fixtures::{
        Col, write_energy_systems_root, write_parquet,
    };

    fn fixture_db() -> (tempfile::TempDir, PerfDatabase) {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let data = write_energy_systems_root(tmp.path());
        // NCCL twin table at `<data_dir>/comm/nccl/<misc.nccl_version>` —
        // the fixture YAML declares `misc.nccl_version: test`. Rows span the
        // full message-size axis so op-level interpolation never misses.
        let nccl_dir = tmp.path().join("data/comm/nccl/test");
        std::fs::create_dir_all(&nccl_dir).expect("mkdir nccl");
        let _ = data;
        write_parquet(
            &nccl_dir.join("nccl_perf.parquet"),
            &[
                Col::Str("nccl_dtype", vec!["half", "half", "half", "half"]),
                Col::Str(
                    "op_name",
                    vec![
                        "all_gather",
                        "all_gather",
                        "reduce_scatter",
                        "reduce_scatter",
                    ],
                ),
                Col::I64("num_gpus", vec![8, 8, 8, 8]),
                Col::I64("message_size", vec![1, 1_000_000_000, 1, 1_000_000_000]),
                Col::F64("latency", vec![1.0, 1000.0, 2.0, 2000.0]),
                Col::F64("power", vec![100.0, 100.0, 200.0, 200.0]),
            ],
        );
        let db = PerfDatabase::load(tmp.path(), "testsys", "vllm", "1.0").expect("db must load");
        (tmp, db)
    }

    fn transfer_op(num_experts: u32, topk: u32, f_rank_count: u32) -> AfdTransferOp {
        AfdTransferOp {
            name: "t".into(),
            scale_factor: 1.0,
            direction: AfdTransferDirection::A2f,
            hidden_size: 5120,
            a_replica_count: 4,
            f_rank_count,
            gpus_per_node: 8,
            f_gpus_per_node: None,
            num_experts,
            topk,
            comm_quant_mode: CommQuantMode::Half,
            comm_overhead_factor: 1.0,
        }
    }

    #[test]
    fn token_partitioned_transfer_caps_sender_fanout_and_f_node_ingress() {
        let (_tmp, db) = fixture_db();
        let mut op = transfer_op(4, 2, 16);
        op.hidden_size = 1024;
        op.a_replica_count = 2;
        // Two of four experts on each of two F nodes: p=1-C(2,2)/C(4,2)=5/6.
        // N=48, TP4 => 12 tokens/sender, 2 destinations => 24 tokens/NIC.
        // Receiver load is 96/8=12 tokens/NIC, so sender wins. Payload is
        // (5/6)*24*1024*2 = 40960 bytes plus one propagation term.
        let got = op.query_token_partitioned(&db, 48, 4, None).unwrap();
        let expected = (40960.0 / db.system_spec.node.inter_node_bw
            + db.system_spec.node.p2p_latency)
            * 1000.0;
        assert!((got.latency_ms - expected).abs() < 1e-12);
        // 16 replicas reverse the bottleneck: 768/8=96 tokens/NIC => 163840 B.
        op.a_replica_count = 16;
        let got = op.query_token_partitioned(&db, 48, 4, None).unwrap();
        let expected = (163840.0 / db.system_spec.node.inter_node_bw
            + db.system_spec.node.p2p_latency)
            * 1000.0;
        assert!((got.latency_ms - expected).abs() < 1e-12);
        assert_eq!(
            op.query_token_partitioned(&db, 0, 4, None)
                .unwrap()
                .latency_ms,
            0.0
        );
        // Two A nodes, two TP4 replicas/node, N=1. Only local slots 0/4
        // participate, each receiving two tokens, rather than ceil(4/8)=1.
        op.a_replica_count = 4;
        op.f_rank_count = 8;
        let got = op.query_token_partitioned(&db, 1, 4, None).unwrap();
        let expected =
            (4096.0 / db.system_spec.node.inter_node_bw + db.system_spec.node.p2p_latency) * 1000.0;
        assert!((got.latency_ms - expected).abs() < 1e-12);
    }

    #[test]
    fn uneven_rank_collectives_include_equal_count_padding() {
        let (_tmp, db) = fixture_db();
        let mut op = AfdFCollectiveOp::all_gather("g", 1.0);
        op.hidden_size = 1024;
        op.a_replica_count = 4;
        op.f_rank_count = 8;
        op.num_experts = 128;
        op.topk = 8;
        // N=7, TP4, two A nodes: F slot loads 4/4/4/2/4/4/4/2.
        // Equal-count AG uses four tokens per rank (32 padded vs 28 logical).
        let twin = NcclOp {
            name: "hand_oracle".into(),
            scale_factor: 1.,
            hidden_size: 4096.,
            num_gpus: 8,
            dtype: CommQuantMode::Half,
            operation: "all_gather".into(),
            seq_split: 1,
        };
        assert_eq!(
            op.query_token_partitioned(&db, 7, 4).unwrap(),
            twin.query(&db, 1).unwrap()
        );
        assert_eq!(mapped_rank_tokens(1, 4, 4, 8).unwrap(), 2);
        assert_eq!(mapped_rank_tokens(7, 4, 4, 8).unwrap(), 4);
        assert_eq!(mapped_rank_tokens(1, 3, 4, 8).unwrap(), 2);
    }

    #[test]
    fn send_prob_matches_python_reference_values() {
        // Oracle: exact integer combinatorics (Python `math.comb`) evaluated
        // offline for the same arguments — not the production formula.
        // dense / degenerate: 1 / num_f_nodes
        assert_eq!(afd_send_prob(0, 0, 4), 0.25);
        assert_eq!(afd_send_prob(128, 8, 1), 1.0);
        // MoE: E=8, topk=2, Nf=2 -> 1 - C(4,2)/C(8,2) = 1 - 6/28
        assert!((afd_send_prob(8, 2, 2) - (1.0 - 6.0 / 28.0)).abs() < 1e-15);
        // MoE: E=16, topk=4, Nf=4 -> 1 - C(12,4)/C(16,4) = 1 - 495/1820
        assert!((afd_send_prob(16, 4, 4) - (1.0 - 495.0 / 1820.0)).abs() < 1e-15);
        // topk > n_other -> 1.0 (E=8, topk=7, Nf=2: n_other=4 < 7)
        assert_eq!(afd_send_prob(8, 7, 2), 1.0);
        // experts_per_node == 0 (more nodes than experts) -> 1 / Nf
        assert_eq!(afd_send_prob(2, 1, 4), 0.25);
        // DeepSeek-V3 shape: E=256, topk=8, Nf=4 ->
        // 1 - C(192,8)/C(256,8), C(192,8)=39_510_598_291_560,
        // C(256,8)=409_663_695_276_000 (math.comb, exact).
        let expected = 1.0 - 39_510_598_291_560.0f64 / 409_663_695_276_000.0f64;
        assert!((afd_send_prob(256, 8, 4) - expected).abs() < 1e-15);
    }

    #[test]
    fn transfer_composes_p2p_twin_with_hand_computed_volume() {
        let (_tmp, db) = fixture_db();
        // Dense: nf = ceil(16/8) = 2, p_send = 1/2. tokens_per_a_replica = 56 tokens,
        // hidden = 5120, bf16 (2 B): per_link_bytes = int(0.5 * 56 * 5120 * 2)
        // = 286_720 -> P2P twin elements = 143_360.
        let op = transfer_op(0, 0, 16);
        let expected_twin = P2POp {
            name: "afd_p2p".into(),
            scale_factor: 1.0,
            pp_size: 2,
            hidden_size: 143_360,
            seq_split: 1,
            span_gpus: None,
            force_scale_out: false,
        };
        let twin_result = expected_twin.query(&db, 1).expect("twin query");
        assert!(
            twin_result.latency_ms > 0.0,
            "twin must answer non-vacuously"
        );
        let result = op.query(&db, 56).expect("afd query");
        assert_eq!(result.latency_ms, twin_result.latency_ms);
        assert_eq!(result.energy_wms, twin_result.energy_wms);
        assert_eq!(result.source, twin_result.source);

        // MoE: E=8, topk=2, nf=2 -> p_send = 1 - 6/28 = 22/28;
        // p_send * 56 = 44 exactly -> per_link_bytes = 44 * 5120 * 2 =
        // 450_560 -> elements 225_280.
        let moe_op = transfer_op(8, 2, 16);
        let moe_twin = P2POp {
            hidden_size: 225_280,
            ..expected_twin.clone()
        };
        let moe_result = moe_op.query(&db, 56).expect("afd moe query");
        let moe_twin_result = moe_twin.query(&db, 1).expect("moe twin query");
        assert_eq!(moe_result.latency_ms, moe_twin_result.latency_ms);
    }

    #[test]
    fn transfer_scales_latency_by_overhead_but_never_energy() {
        let (_tmp, db) = fixture_db();
        let op = AfdTransferOp {
            scale_factor: 2.0,
            comm_overhead_factor: 1.5,
            ..transfer_op(0, 0, 16)
        };
        let base = transfer_op(0, 0, 16).query(&db, 56).expect("base");
        let scaled = op.query(&db, 56).expect("scaled");
        // Python: lat = float(result) * comm_overhead_factor, then
        // PerformanceResult(lat * scale_factor, energy=result.energy *
        // scale_factor) — energy sees the scale factor only.
        assert_eq!(scaled.latency_ms, base.latency_ms * 3.0);
        assert_eq!(scaled.energy_wms, base.energy_wms * 2.0);
        assert_eq!(scaled.source, base.source);
    }

    #[test]
    fn transfer_zero_token_and_zero_volume_are_empirical_zero() {
        let (_tmp, db) = fixture_db();
        assert_empirical_zero(&transfer_op(0, 0, 16).query(&db, 0).unwrap());
        // p_send * tokens_per_a_replica underflows int() to 0 bytes: tokens_per_a_replica=1, dense nf=32 ->
        // int((1/32) * 1 * 5120 * 2) = int(320) > 0 — instead force
        // num_experts into the degenerate 1/Nf branch with tiny volume:
        // tokens_per_a_replica=1, hidden=8, nf=8 -> int(0.125 * 1 * 8 * 2) = int(2.0) = 2 > 0;
        // truly zero: hidden=1, tokens_per_a_replica=1, nf=32 -> int((1/32)*1*1*2)=int(0.0625)=0.
        let tiny = AfdTransferOp {
            hidden_size: 1,
            a_replica_count: 1,
            f_rank_count: 256,
            ..transfer_op(0, 0, 256)
        };
        assert_empirical_zero(&tiny.query(&db, 1).unwrap());
    }

    #[test]
    fn all_gather_composes_nccl_twin_with_hand_computed_elements() {
        let (_tmp, db) = fixture_db();
        // f_local = min(16, 8) = 8; total = 56 * 4 = 224; dense nf=2 ->
        // tokens_per_f_node = 112; per_rank_elements = int(112 * 5120 / 8) =
        // 71_680.
        let mut op = AfdFCollectiveOp::all_gather("g", 1.0);
        op.hidden_size = 5120;
        op.a_replica_count = 4;
        op.f_rank_count = 16;
        let expected_twin = NcclOp {
            name: "afd_all_gather".into(),
            scale_factor: 1.0,
            hidden_size: 71_680.0,
            num_gpus: 8,
            dtype: CommQuantMode::Half,
            operation: "all_gather".into(),
            seq_split: 1,
        };
        let twin_result = expected_twin.query(&db, 1).expect("twin query");
        assert!(
            twin_result.latency_ms > 0.0,
            "twin must answer non-vacuously"
        );
        let result = op.query(&db, 56).expect("ag query");
        assert_eq!(result.latency_ms, twin_result.latency_ms);
        assert_eq!(result.energy_wms, twin_result.energy_wms);
        assert_eq!(result.source, twin_result.source);
    }

    #[test]
    fn reduce_scatter_uses_reduce_scatter_operation() {
        let (_tmp, db) = fixture_db();
        let mut op = AfdFCollectiveOp::reduce_scatter("r", 1.0);
        op.hidden_size = 5120;
        op.a_replica_count = 4;
        op.f_rank_count = 16;
        let expected_twin = NcclOp {
            name: "afd_reduce_scatter".into(),
            scale_factor: 1.0,
            hidden_size: 71_680.0,
            num_gpus: 8,
            dtype: CommQuantMode::Half,
            operation: "reduce_scatter".into(),
            seq_split: 1,
        };
        let twin_result = expected_twin.query(&db, 1).expect("twin query");
        let result = op.query(&db, 56).expect("rs query");
        assert_eq!(result.latency_ms, twin_result.latency_ms);
    }

    #[test]
    fn collectives_are_zero_for_broadcast_single_gpu_or_zero_tokens() {
        let (_tmp, db) = fixture_db();
        let mut op = AfdFCollectiveOp::all_gather("g", 1.0);
        op.hidden_size = 5120;
        op.a_replica_count = 4;
        op.f_rank_count = 16;
        assert_empirical_zero(&op.query(&db, 0).unwrap());
        op.rank_mapping = AfdRankMapping::Broadcast;
        assert_empirical_zero(&op.query(&db, 56).unwrap());
        op.rank_mapping = AfdRankMapping::OneToOne;
        op.f_rank_count = 1; // f_local = 1
        assert_empirical_zero(&op.query(&db, 56).unwrap());
    }

    #[test]
    fn fabric_override_substitutes_link_and_matches_twin_when_equal() {
        let (_tmp, db) = fixture_db();
        let op = transfer_op(0, 0, 16);
        let base = op.query(&db, 56).expect("base");
        // Link identical to the fixture spec (inter_node_bw=50e9,
        // p2p_latency=2e-6): the override path must reproduce the twin
        // bit-exactly (the same-SKU degradation regression).
        let same = op
            .query_with_fabric(
                &db,
                56,
                Some(AfdFabricLink {
                    inter_node_bw: 50e9,
                    p2p_latency: 2e-6,
                    peer: None,
                }),
            )
            .expect("same-link");
        assert_eq!(base.latency_ms, same.latency_ms);
        assert_eq!(base.source, same.source);
        // Hand-computed oracle for a weaker link: per_link_bytes = 286_720
        // (dense nf=2, tokens_per_a_replica=56, hidden=5120, bf16) -> bytes 286_720 after the
        // ceil-even round-trip; (bytes / 25e9 + 4e-6) * 1000 ms.
        let weaker = op
            .query_with_fabric(
                &db,
                56,
                Some(AfdFabricLink {
                    inter_node_bw: 25e9,
                    p2p_latency: 4e-6,
                    peer: None,
                }),
            )
            .expect("weaker-link");
        let expected = (286_720.0 / 25e9 + 4e-6) * 1000.0;
        assert!(
            (weaker.latency_ms - expected).abs() < 1e-12 * expected,
            "weaker link: actual={} expected={expected}",
            weaker.latency_ms
        );
    }

    #[test]
    fn fabric_between_preserves_both_endpoints() {
        let link = AfdFabricLink::between((50e9, 2e-6), (25e9, 4e-6));
        assert_eq!(link.inter_node_bw, 50e9);
        assert_eq!(link.p2p_latency, 2e-6);
        assert_eq!(link.peer, Some((25e9, 4e-6)));
        let clamped = AfdFabricLink::between((0.0, 0.0), (0.0, 0.0));
        assert_eq!(clamped.inter_node_bw, 1.0);
    }
    #[test]
    fn combine_composes_elementwise_twin_with_hand_computed_bytes() {
        let (_tmp, db) = fixture_db();
        // ep=4, a_tp_size=1, tokens_per_a_replica=56; total_bytes =
        // int(5 * 56 * 5120 * 2.0) = 2_867_200 -> twin bytes_per_token =
        // 2 * ceil(2_867_200 / 2) = 2_867_200.
        let op = AfdCombineOp {
            name: "c".into(),
            scale_factor: 1.0,
            hidden_size: 5120,
            a_tp_size: 1,
            f_moe_ep_size: 4,
            comm_quant_mode: CommQuantMode::Half,
        };
        let expected_twin = ElementwiseOp {
            name: "afd_combine_mem".into(),
            scale_factor: 1.0,
            bytes_per_token: 2_867_200.0,
            scale_num_tokens: 1,
            seq_split: 1,
        };
        let twin_result = expected_twin.query(&db, 1).expect("twin query");
        assert!(
            twin_result.latency_ms > 0.0,
            "twin must answer non-vacuously"
        );
        let result = op.query(&db, 56).expect("combine query");
        assert_eq!(result.latency_ms, twin_result.latency_ms);
        assert_eq!(result.energy_wms, twin_result.energy_wms);
        assert_eq!(result.source, twin_result.source);
    }

    #[test]
    fn combine_is_zero_without_expert_parallelism_or_tokens() {
        let (_tmp, db) = fixture_db();
        let dense = AfdCombineOp {
            name: "c".into(),
            scale_factor: 1.0,
            hidden_size: 5120,
            a_tp_size: 1,
            f_moe_ep_size: 1,
            comm_quant_mode: CommQuantMode::Half,
        };
        assert_empirical_zero(&dense.query(&db, 56).unwrap());
        let ep = AfdCombineOp {
            f_moe_ep_size: 4,
            ..dense
        };
        assert_empirical_zero(&ep.query(&db, 0).unwrap());
    }

    #[test]
    fn combine_supports_tp_and_still_short_circuits_zero_tokens() {
        let (_tmp, db) = fixture_db();
        let op = AfdCombineOp {
            name: "unsupported_combine".into(),
            scale_factor: 1.0,
            hidden_size: 5120,
            a_tp_size: 2,
            f_moe_ep_size: 8,
            comm_quant_mode: CommQuantMode::Half,
        };
        assert_eq!(op.query(&db, 0).unwrap().latency_ms, 0.0);
    }

    #[test]
    fn transfer_json_round_trips_canonical_counts() {
        let op = transfer_op(128, 8, 16);
        let wire = serde_json::to_value(&op).unwrap();
        assert_eq!(wire["a_replica_count"], 4);
        assert_eq!(wire["f_rank_count"], 16);
        let restored: AfdTransferOp = serde_json::from_value(wire).unwrap();
        assert_eq!(restored, op);
    }

    #[test]
    fn prepared_receiver_load_matches_explicit_token_conservation() {
        // Independent oracle: place every balanced TP slice on its actual
        // modulo destination, without the production periodic-count formula.
        for replicas in 1..=9 {
            for tp in [1, 2, 3, 4, 8] {
                for width in [1, 3, 4, 5, 8, 12] {
                    let prepared = AfdTokenMapping::new(replicas, tp, width).unwrap();
                    for tokens in 0..=25 {
                        let mut received = vec![0_u64; width as usize];
                        for replica in 0..replicas {
                            for slice in 0..tp {
                                let count = tokens / tp + u32::from(slice < tokens % tp);
                                received[((replica * tp + slice) % width) as usize] +=
                                    u64::from(count);
                            }
                        }
                        assert_eq!(received.iter().sum::<u64>(), u64::from(replicas * tokens));
                        let oracle = *received.iter().max().unwrap();
                        assert_eq!(
                            prepared.busiest(tokens),
                            oracle,
                            "replicas={replicas},tp={tp},width={width},tokens={tokens}"
                        );
                        assert_eq!(
                            mapped_rank_tokens(tokens, replicas, tp, width).unwrap(),
                            oracle
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn endpoint_bottleneck_depends_on_payload_and_database_mode() {
        let (_tmp, mut db) = fixture_db();
        let op = transfer_op(128, 8, 8); // one F node, p=1
        // A: slower bandwidth, lower latency. F: faster bandwidth, higher
        // latency. 1 token => 10,240 B: F wins; 100 tokens => 1,024,000 B: A wins.
        let link = AfdFabricLink::between((25e9, 1e-6), (100e9, 10e-6));
        let small = op.query_with_fabric(&db, 1, Some(link)).unwrap();
        let large = op.query_with_fabric(&db, 100, Some(link)).unwrap();
        assert!((small.latency_ms - 0.0101024).abs() < 1e-12);
        assert!((large.latency_ms - 0.04196).abs() < 1e-12);
        db.database_mode = crate::common::enums::DatabaseMode::Sol;
        let sol = op.query_with_fabric(&db, 1, Some(link)).unwrap();
        assert!((sol.latency_ms - 0.0004096).abs() < 1e-12);
    }
}
