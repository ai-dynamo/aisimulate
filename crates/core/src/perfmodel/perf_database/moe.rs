// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Basic MoE perf table.
//!
//! Mirrors the raw SILICON-path layout of
//! `aisimulate.sdk.operations.moe.MoE._query_moe_table`:
//!
//! `moe_data[quant][distribution][topk][num_experts][hidden][inter][moe_tp][moe_expert_compute]`
//! returns a `{num_tokens -> latency_ms}` dict.
//!
//! Resolution mirrors Python v2's `_resolve_tokens`: the token curve rides
//! the shared `perf_interp` engine (1-axis Grid, RAW lerp in range; beyond
//! the collected range the boundary util is held with `k_tail=1` and the
//! caller-supplied MoE roofline SOL carries the growth — unclamped util,
//! exactly like Python which deleted the hand-rolled overflow estimator).
//! The SOL closure comes from the operator layer (`operators/moe.rs`),
//! which owns the roofline math.
//!
//! Singleton-underflow contract (Python `_require_moe_token_points`): a
//! curve with a single token point queried BELOW that point is a structured
//! miss — one large-token row cannot define the low-token launch floor.
//!
//! `workload_distribution` falls back to `"uniform"` when the requested
//! variant is absent for the given quant, matching Python's behavior.
//!
//! WideEP MLA lives in `perf_database::wideep_mla`; the TRT-LLM all-to-all
//! table in `perf_database::trtllm_alltoall`; large-EP expert compute in
//! `perf_database::moe_expert_compute`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use super::axis_curve::LeafAxisCurve;
use super::moe_index::{MoeIndex, MoeShapeKey};
use super::perf_interp::LeafValue;
use super::{SourceResolver, kernel_source_ok};
use crate::common::enums::MoeQuantMode;
use crate::common::error::AicError;
use crate::config::{PerfDbSources, PerfSource};
use crate::perf_database::parquet_loader::{PerfReader, PerfRow};

pub struct MoeTable {
    data_root: PathBuf,
    /// Ordered, priority-sorted sources for the MoE perf file (shared-layer
    /// aware; see [`PerfSource`]). Single-primary, no-filter by default
    /// (`MoeTable::new`).
    moe_sources: Vec<PerfSource>,
    moe: OnceLock<Result<LoadedMoeGrids, AicError>>,
}

/// Which kernel grid a MoE accessor addresses: the default table or the
/// TRT-LLM `moe_torch_flow_min_latency` low-latency split (Python's
/// `_moe_data` vs `_moe_low_latency_data`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeKernel {
    Standard,
    LowLatency,
}

/// One collected sibling slice of the MoE table for a fixed
/// `(quant, distribution, moe_tp, moe_expert_compute)`: the categorical shape features
/// plus its `num_tokens -> latency_ms` curve. Consumed by the operator
/// layer's cross-shape/cross-quant transfer ladder (the algorithm lives in
/// `operators/moe.rs`; this is a data accessor payload only).
#[derive(Clone, Debug)]
pub struct MoeSiblingSlice {
    pub topk: u32,
    pub num_experts: u32,
    pub hidden_size: u32,
    pub inter_size: u32,
    /// `(num_tokens, latency_ms)` in ascending token order.
    pub points: Vec<(u32, f64)>,
}

/// Parallel grids split by `kernel_source`. `default` and `low_latency`
/// contain default-eligible rows (all legacy rows when the metadata column is
/// absent), while `by_kernel_source` retains every named lane for an exact
/// user-selected lookup.
struct LoadedMoeGrids {
    default: MoeGrids,
    low_latency: MoeGrids,
    by_kernel_source: BTreeMap<String, MoeGrids>,
}

struct MoeGrids {
    index: MoeIndex<MoeShapeKey, LeafAxisCurve>,
    /// Distinct quant names in first-seen (file row) order. Python's
    /// transfer ladder iterates the table dict in INSERTION order
    /// (`for q in moe_table`), which breaks profile-distance ties by file
    /// order — live on shards whose file order differs from sorted order
    /// (e.g. b200/vllm/0.24.0 lists `fp8_block` before `fp8`).
    quants_in_load_order: Vec<String>,
}

fn insert_moe_row(
    index: &mut MoeIndex<MoeShapeKey, BTreeMap<u32, LeafValue>>,
    quants_in_load_order: &mut Vec<String>,
    quant: &str,
    distribution: &str,
    shape: MoeShapeKey,
    num_tokens: u32,
    value: LeafValue,
) {
    if !quants_in_load_order
        .iter()
        .any(|existing| existing == quant)
    {
        quants_in_load_order.push(quant.to_owned());
    }
    // First-seen value wins within both the legacy merged grid and an exact
    // lane, including across priority-ordered shared-layer sources.
    index
        .entry(quant.to_owned(), distribution.to_owned(), shape)
        .entry(num_tokens)
        .or_insert(value);
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct MoeKey {
    quant: String,
    distribution: String,
    topk: u32,
    num_experts: u32,
    hidden_size: u32,
    inter_size: u32,
    moe_tp_size: u32,
    moe_ep_size: u32,
}

impl MoeKey {
    fn from_shape(quant: &str, distribution: &str, shape: MoeShapeKey) -> Self {
        Self {
            quant: quant.to_string(),
            distribution: distribution.to_string(),
            topk: shape.topk,
            num_experts: shape.num_experts,
            hidden_size: shape.hidden_size,
            inter_size: shape.inter_size,
            moe_tp_size: shape.moe_tp_size,
            moe_ep_size: shape.moe_ep_size,
        }
    }
}

impl MoeTable {
    /// Construct an empty table for the given data directory. No I/O. The MoE
    /// perf file is sourced solely from `data_root/moe_perf.parquet` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf) -> Self {
        Self::with_sources(data_root, &SourceResolver::fixed(PerfDbSources::default()))
            .expect("fixed-map resolution is infallible")
    }

    /// Construct with shared-layer (sibling/cross-version) sources supplied by the
    /// engine's `SourceResolver` (live resolution owns the shared-layer walk;
    /// a fixed source map is the test-only path). The MoE file falls back to its
    /// primary `data_root/moe_perf.parquet` when the resolver names no override. No I/O.
    pub fn with_sources(data_root: PathBuf, resolver: &SourceResolver) -> Result<Self, AicError> {
        let moe_sources = resolver.sources_for("moe_perf.parquet", &data_root)?;
        Ok(Self {
            data_root,
            moe_sources,
            moe: OnceLock::new(),
        })
    }

    /// Raw MoE value (latency ms + power/energy) via the perf_interp v2
    /// engine contract (1-axis token curve): exact hit / RAW lerp in range;
    /// beyond the collected range the boundary util is held (`k_tail=1`,
    /// unclamped) and `sol` — the operator layer's MoE roofline — carries
    /// the growth. Mirrors Python `MoE._query_moe_table._resolve_tokens`.
    ///
    /// Falls back to the `"uniform"` distribution if the requested
    /// distribution is absent for the given quant mode. A singleton curve
    /// queried below its only point is a structured miss (Python
    /// `_require_moe_token_points`).
    #[allow(clippy::too_many_arguments)]
    pub fn query(
        &self,
        num_tokens: u32,
        hidden_size: u32,
        inter_size: u32,
        topk: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant: MoeQuantMode,
        workload_distribution: &str,
        sol: &dyn Fn(f64) -> f64,
    ) -> Result<LeafValue, AicError> {
        self.query_for_kernel_source(
            None,
            num_tokens,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            quant,
            workload_distribution,
            sol,
        )
    }

    /// Query one exact collected kernel-source lane. The requested lane is
    /// never merged with or substituted by another lane: an unavailable lane
    /// is a typed missing-data error instead of a silent fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn query_kernel_source(
        &self,
        kernel_source: &str,
        num_tokens: u32,
        hidden_size: u32,
        inter_size: u32,
        topk: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant: MoeQuantMode,
        workload_distribution: &str,
        sol: &dyn Fn(f64) -> f64,
    ) -> Result<LeafValue, AicError> {
        self.query_for_kernel_source(
            Some(kernel_source),
            num_tokens,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            quant,
            workload_distribution,
            sol,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn query_for_kernel_source(
        &self,
        kernel_source: Option<&str>,
        num_tokens: u32,
        hidden_size: u32,
        inter_size: u32,
        topk: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant: MoeQuantMode,
        workload_distribution: &str,
        sol: &dyn Fn(f64) -> f64,
    ) -> Result<LeafValue, AicError> {
        let grids = self.grids_for_kernel_source(kernel_source)?;
        let quant_name = quant.name();

        let shape = MoeShapeKey {
            topk,
            num_experts,
            hidden_size,
            inter_size,
            moe_tp_size,
            moe_ep_size,
        };
        let (dist, by_tokens) =
            grids
                .index
                .resolve_uniform(quant_name, workload_distribution, &shape);
        let by_tokens = by_tokens.ok_or_else(|| {
            let key = MoeKey::from_shape(quant_name, dist, shape);
            AicError::PerfDatabase(format!(
                "MoE data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        if by_tokens.is_empty() {
            let key = MoeKey::from_shape(quant_name, dist, shape);
            return Err(AicError::PerfDatabase(format!(
                "MoE data has no token points for {key:?} at {}",
                self.data_root.display()
            )));
        }
        if let Some(only) = by_tokens.singleton_underflow(num_tokens) {
            let key = MoeKey::from_shape(quant_name, dist, shape);
            return Err(AicError::PerfDatabase(format!(
                "MoE silicon token underflow has only one measured point; cannot infer \
                 low-token latency from a singleton. num_tokens={num_tokens}, \
                 measured_token={only}, key={key:?}"
            )));
        }
        by_tokens.query(num_tokens as f64, sol)
    }

    /// Probe the TRT-LLM low-latency NVFP4 MoE kernel table.
    ///
    /// Returns `Ok(Some(value))` when the loaded `low_latency` grid
    /// contains a matching `(quant, distribution-after-uniform-fallback,
    /// topk, num_experts, hidden, inter, moe_tp, moe_expert_compute)` entry, and
    /// `Ok(None)` when the shape is absent — the caller should then fall
    /// through to `query()` (the default grid).
    ///
    /// Mirrors Python's small-token nvfp4 gated-MoE branch in
    /// `MoE._query_moe_table`: the low-latency table is consulted with a
    /// try/except that falls back to `_moe_data` when the SHAPE is absent
    /// (`Ok(None)` here). A singleton-underflow on a present shape is an
    /// `Err` (structured miss), not a fallback — in Python the guard fires
    /// inside `_resolve_tokens`, after the ll table has been selected.
    #[allow(clippy::too_many_arguments)]
    pub fn query_low_latency(
        &self,
        num_tokens: u32,
        hidden_size: u32,
        inter_size: u32,
        topk: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant: MoeQuantMode,
        workload_distribution: &str,
        sol: &dyn Fn(f64) -> f64,
    ) -> Result<Option<LeafValue>, AicError> {
        let loaded = self.load()?;
        let grids = &loaded.low_latency;
        if grids.index.is_empty() {
            return Ok(None);
        }
        let quant_name = quant.name();
        let shape = MoeShapeKey {
            topk,
            num_experts,
            hidden_size,
            inter_size,
            moe_tp_size,
            moe_ep_size,
        };
        let (dist, by_tokens) =
            grids
                .index
                .resolve_uniform(quant_name, workload_distribution, &shape);
        let Some(by_tokens) = by_tokens else {
            return Ok(None);
        };
        if by_tokens.is_empty() {
            return Ok(None);
        }
        if let Some(only) = by_tokens.singleton_underflow(num_tokens) {
            let key = MoeKey::from_shape(quant_name, dist, shape);
            return Err(AicError::PerfDatabase(format!(
                "MoE low-latency token underflow has only one measured point; cannot infer \
                 low-token latency from a singleton. num_tokens={num_tokens}, \
                 measured_token={only}, key={key:?}"
            )));
        }
        by_tokens.query(num_tokens as f64, sol).map(Some)
    }

    /// `true` iff the loaded low-latency grid has any rows.
    ///
    /// Older perf-DB versions predate the `kernel_source` column, so the
    /// low-latency accumulator stays empty and the small-token nvfp4 gate
    /// is short-circuited at the operator layer.
    pub fn low_latency_available(&self) -> Result<bool, AicError> {
        let loaded = self.load()?;
        Ok(!loaded.low_latency.index.is_empty())
    }

    /// Own-slice `num_tokens -> latency_ms` curve for a full MoE key, after
    /// the per-quant `"uniform"` distribution fallback. A typed miss
    /// (`AicError::PerfDatabase`) means the slice is absent or empty —
    /// mirroring Python `util_empirical.require_data_slice` as used by the
    /// empirical own-shape grid (`_slice`) and the low-latency table probe
    /// (`_moe_table`) in `MoE._query_moe_table`.
    #[allow(clippy::too_many_arguments)]
    pub fn slice_points(
        &self,
        kernel: MoeKernel,
        quant_name: &str,
        workload_distribution: &str,
        topk: u32,
        num_experts: u32,
        hidden_size: u32,
        inter_size: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
    ) -> Result<Vec<(u32, f64)>, AicError> {
        let grids = self.grids_for(kernel)?;
        self.slice_points_in_grids(
            grids,
            format!("{kernel:?}"),
            quant_name,
            workload_distribution,
            topk,
            num_experts,
            hidden_size,
            inter_size,
            moe_tp_size,
            moe_ep_size,
        )
    }

    /// Own-slice token curve from an exact collected kernel-source lane.
    #[allow(clippy::too_many_arguments)]
    pub fn slice_points_for_kernel_source(
        &self,
        kernel_source: &str,
        quant_name: &str,
        workload_distribution: &str,
        topk: u32,
        num_experts: u32,
        hidden_size: u32,
        inter_size: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
    ) -> Result<Vec<(u32, f64)>, AicError> {
        let grids = self.grids_for_kernel_source(Some(kernel_source))?;
        self.slice_points_in_grids(
            grids,
            format!("kernel_source={kernel_source:?}"),
            quant_name,
            workload_distribution,
            topk,
            num_experts,
            hidden_size,
            inter_size,
            moe_tp_size,
            moe_ep_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn slice_points_in_grids(
        &self,
        grids: &MoeGrids,
        grid_label: String,
        quant_name: &str,
        workload_distribution: &str,
        topk: u32,
        num_experts: u32,
        hidden_size: u32,
        inter_size: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
    ) -> Result<Vec<(u32, f64)>, AicError> {
        let shape = MoeShapeKey {
            topk,
            num_experts,
            hidden_size,
            inter_size,
            moe_tp_size,
            moe_ep_size,
        };
        let (dist, by_tokens) =
            grids
                .index
                .resolve_uniform(quant_name, workload_distribution, &shape);
        let by_tokens = by_tokens.filter(|curve| !curve.is_empty()).ok_or_else(|| {
            let key = MoeKey::from_shape(quant_name, dist, shape);
            AicError::PerfDatabase(format!(
                "MoE data missing for {key:?} ({grid_label}) at {}",
                self.data_root.display()
            ))
        })?;
        Ok(by_tokens
            .iter()
            .map(|(t, leaf)| (t, leaf.latency))
            .collect())
    }

    /// All collected sibling slices for `(quant, distribution-after-uniform-
    /// fallback, moe_tp, moe_expert_compute)`; empty curves skipped, an empty result is
    /// data (not an error). Mirrors the enumeration in Python `_collect`
    /// (`MoE._query_moe_table`), which walks the nested
    /// `topk -> num_experts -> hidden -> inter` dicts. NOTE: Python yields
    /// dict insertion (file row) order; the `BTreeMap` yields sorted
    /// `(topk, num_experts, hidden, inter)` order instead — observable only
    /// through exact ties in nearest-candidate selection.
    pub fn sibling_slices(
        &self,
        kernel: MoeKernel,
        quant_name: &str,
        workload_distribution: &str,
        moe_tp_size: u32,
        moe_ep_size: u32,
    ) -> Result<Vec<MoeSiblingSlice>, AicError> {
        let grids = self.grids_for(kernel)?;
        Ok(Self::sibling_slices_in_grids(
            grids,
            quant_name,
            workload_distribution,
            moe_tp_size,
            moe_ep_size,
        ))
    }

    /// Collected sibling slices from an exact kernel-source lane.
    pub fn sibling_slices_for_kernel_source(
        &self,
        kernel_source: &str,
        quant_name: &str,
        workload_distribution: &str,
        moe_tp_size: u32,
        moe_ep_size: u32,
    ) -> Result<Vec<MoeSiblingSlice>, AicError> {
        let grids = self.grids_for_kernel_source(Some(kernel_source))?;
        Ok(Self::sibling_slices_in_grids(
            grids,
            quant_name,
            workload_distribution,
            moe_tp_size,
            moe_ep_size,
        ))
    }

    fn sibling_slices_in_grids(
        grids: &MoeGrids,
        quant_name: &str,
        workload_distribution: &str,
        moe_tp_size: u32,
        moe_ep_size: u32,
    ) -> Vec<MoeSiblingSlice> {
        let (_, by_shape) = grids
            .index
            .resolve_uniform_shapes(quant_name, workload_distribution);
        let mut slices = Vec::new();
        let Some(by_shape) = by_shape else {
            return slices;
        };
        for (shape, curve) in by_shape {
            if shape.moe_tp_size != moe_tp_size
                || shape.moe_ep_size != moe_ep_size
                || curve.is_empty()
            {
                continue;
            }
            slices.push(MoeSiblingSlice {
                topk: shape.topk,
                num_experts: shape.num_experts,
                hidden_size: shape.hidden_size,
                inter_size: shape.inter_size,
                points: curve.iter().map(|(t, leaf)| (t, leaf.latency)).collect(),
            });
        }
        slices
    }

    /// Distinct quant names present in the kernel grid, in first-seen
    /// (file row) order — Python iterates the table dict in insertion
    /// order (`for q in moe_table`), and the transfer ladder's stable
    /// profile-distance sort breaks ties by that order.
    pub fn available_quants(&self, kernel: MoeKernel) -> Result<Vec<String>, AicError> {
        Ok(self.grids_for(kernel)?.quants_in_load_order.clone())
    }

    /// Distinct quant names in an exact collected kernel-source lane.
    pub fn available_quants_for_kernel_source(
        &self,
        kernel_source: &str,
    ) -> Result<Vec<String>, AicError> {
        Ok(self
            .grids_for_kernel_source(Some(kernel_source))?
            .quants_in_load_order
            .clone())
    }

    fn grids_for(&self, kernel: MoeKernel) -> Result<&MoeGrids, AicError> {
        let loaded = self.load()?;
        Ok(match kernel {
            MoeKernel::Standard => &loaded.default,
            MoeKernel::LowLatency => &loaded.low_latency,
        })
    }

    fn grids_for_kernel_source(&self, kernel_source: Option<&str>) -> Result<&MoeGrids, AicError> {
        let loaded = self.load()?;
        match kernel_source {
            None => Ok(&loaded.default),
            Some(kernel_source) => loaded.by_kernel_source.get(kernel_source).ok_or_else(|| {
                AicError::PerfDatabase(format!(
                    "MoE kernel_source {kernel_source:?} is unavailable at {}",
                    self.data_root.display()
                ))
            }),
        }
    }

    fn load(&self) -> Result<&LoadedMoeGrids, AicError> {
        let cell = self.moe.get_or_init(|| load_moe_parquet(&self.moe_sources));
        cell.as_ref().map_err(clone_err)
    }
}

/// Load the MoE table from an ordered, priority-sorted source list. Sources are
/// read in order; the first source containing a `(shape, num_tokens)` tuple wins
/// (`or_insert`), mirroring Python's `_read_filtered_rows` concatenation +
/// `load_moe_data` skip-on-key-conflict. Missing files are skipped (a sibling
/// declared in the manifest need not exist for every system); an error is
/// returned only when no source yields rows.
/// Kernel-routed quant remaps (Python `load_moe_data`'s two rules) — the
/// single home shared by the query loader and the table view, so the next
/// per-GPU-generation mxfp4-style remap lands once:
/// - Blackwell trtllm-gen MXFP4xMXFP8 rows get their dedicated quant mode;
/// - Hopper flashinfer-cutlass SM90 mixed-GEMM rows likewise.
pub(crate) fn moe_kernel_quant_rewrite(raw_quant: String, kernel_source: &str) -> String {
    match (raw_quant.as_str(), kernel_source) {
        ("w4a8_mxfp4_mxfp8", "sglang_mxfp4_flashinfer_trtllm_moe") => {
            "w4a8_mxfp4_mxfp8_trtllm".to_string()
        }
        ("w4a16_mxfp4", "sglang_flashinfer_cutlass_moe") => "w4a16_mxfp4_cutlass".to_string(),
        _ => raw_quant,
    }
}

/// Optional selection metadata shared by the query loader and coverage view.
/// An absent column preserves legacy automatic selection; malformed metadata
/// is not a coverage miss and must never trigger an estimator fallback.
pub(crate) fn moe_default_eligible(
    reader: &PerfReader,
    row: &PerfRow,
    column: Option<usize>,
    kernel_source_column: Option<usize>,
) -> Result<bool, AicError> {
    let Some(column) = column else {
        return Ok(true);
    };
    let eligible = row.bool_strict(column).map_err(|_| {
        AicError::InvalidPerfData(format!(
            "default_eligible must be a non-null Boolean at {}",
            reader.path().display()
        ))
    })?;
    if !eligible
        && row
            .str_optional(kernel_source_column)?
            .is_none_or(|source| source.trim().is_empty())
    {
        return Err(AicError::InvalidPerfData(format!(
            "default_eligible=false requires a nonblank string kernel_source at {}",
            reader.path().display()
        )));
    }
    Ok(eligible)
}

fn load_moe_parquet(sources: &[PerfSource]) -> Result<LoadedMoeGrids, AicError> {
    let mut default_index: MoeIndex<MoeShapeKey, BTreeMap<u32, LeafValue>> = MoeIndex::default();
    let mut low_latency_index: MoeIndex<MoeShapeKey, BTreeMap<u32, LeafValue>> =
        MoeIndex::default();
    let mut default_quants: Vec<String> = Vec::new();
    let mut low_latency_quants: Vec<String> = Vec::new();
    let mut kernel_source_indices: BTreeMap<
        String,
        MoeIndex<MoeShapeKey, BTreeMap<u32, LeafValue>>,
    > = BTreeMap::new();
    let mut kernel_source_quants: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let moe_dtype_col = reader.col("moe_dtype")?;
        let num_tokens_col = reader.col("num_tokens")?;
        let hidden_size_col = reader.col("hidden_size")?;
        let inter_size_col = reader.col("inter_size")?;
        let topk_col = reader.col("topk")?;
        let num_experts_col = reader.col("num_experts")?;
        let moe_tp_size_col = reader.col("moe_tp_size")?;
        let moe_ep_size_col = reader.col("moe_ep_size")?;
        let distribution_col = reader.col("distribution")?;
        let latency_col = reader.col("latency")?;
        let power_col = reader.col_optional("power");
        // Optional in older perf-DB versions; when absent every row falls into
        // the `default` grid (matching the pre-split behavior). The same column
        // gates the per-source shared-layer `kernel_source` allowlist.
        let kernel_source_col = reader.col_optional("kernel_source");
        let default_eligible_col = reader.col_optional("default_eligible");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), kernel_source_col, &row)? {
                continue;
            }
            let default_eligible =
                moe_default_eligible(&reader, &row, default_eligible_col, kernel_source_col)?;
            let kernel_source = row
                .str_optional(kernel_source_col)?
                .unwrap_or("")
                .to_string();
            // Kernel-specific mxfp4 remaps (mirror Python `load_moe_data`):
            // the collector logs two distinct kernels under one `moe_dtype`;
            // route them to dedicated quant modes so DeepSeek-V4 modeling can
            // select the right one per GPU generation.
            //  - Blackwell trtllm-gen MXFP4xMXFP8:
            //    w4a8_mxfp4_mxfp8 + sglang_mxfp4_flashinfer_trtllm_moe
            //      -> w4a8_mxfp4_mxfp8_trtllm
            //  - Hopper flashinfer cutlass SM90 mixed-GEMM:
            //    w4a16_mxfp4 + sglang_flashinfer_cutlass_moe
            //      -> w4a16_mxfp4_cutlass
            let quant = moe_kernel_quant_rewrite(row.str_owned(moe_dtype_col)?, &kernel_source);
            let distribution = row.str_owned(distribution_col)?;
            let shape = MoeShapeKey {
                topk: row.u32(topk_col)?,
                num_experts: row.u32(num_experts_col)?,
                hidden_size: row.u32(hidden_size_col)?,
                inter_size: row.u32(inter_size_col)?,
                moe_tp_size: row.u32(moe_tp_size_col)?,
                moe_ep_size: row.u32(moe_ep_size_col)?,
            };
            let (target, target_quants) = if kernel_source == "moe_torch_flow_min_latency" {
                (&mut low_latency_index, &mut low_latency_quants)
            } else {
                (&mut default_index, &mut default_quants)
            };
            let latency = row.f64(latency_col)?;
            let power = row.f64_optional(power_col)?.unwrap_or(0.0);
            let value = LeafValue::with_power(latency, power);
            let num_tokens = row.u32(num_tokens_col)?;
            if default_eligible {
                insert_moe_row(
                    target,
                    target_quants,
                    &quant,
                    &distribution,
                    shape,
                    num_tokens,
                    value,
                );
            }
            if !kernel_source.is_empty() {
                let source_index = kernel_source_indices
                    .entry(kernel_source.clone())
                    .or_default();
                let source_quants = kernel_source_quants.entry(kernel_source).or_default();
                insert_moe_row(
                    source_index,
                    source_quants,
                    &quant,
                    &distribution,
                    shape,
                    num_tokens,
                    value,
                );
            }
        }
    }
    if !any_source
        || (default_index.is_empty()
            && low_latency_index.is_empty()
            && kernel_source_indices.is_empty())
    {
        return Err(AicError::PerfDatabase(format!(
            "no rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    let by_kernel_source = kernel_source_indices
        .into_iter()
        .map(|(kernel_source, index)| {
            let quants_in_load_order = kernel_source_quants
                .remove(&kernel_source)
                .expect("kernel-source quant order is inserted with its index");
            (
                kernel_source,
                MoeGrids {
                    index: index.map_values(|curve| LeafAxisCurve::from_map("num_tokens", curve)),
                    quants_in_load_order,
                },
            )
        })
        .collect();
    Ok(LoadedMoeGrids {
        default: MoeGrids {
            index: default_index.map_values(|curve| LeafAxisCurve::from_map("num_tokens", curve)),
            quants_in_load_order: default_quants,
        },
        low_latency: MoeGrids {
            index: low_latency_index
                .map_values(|curve| LeafAxisCurve::from_map("num_tokens", curve)),
            quants_in_load_order: low_latency_quants,
        },
        by_kernel_source,
    })
}

fn clone_err(err: &AicError) -> AicError {
    match err {
        AicError::InvalidPerfData(message) => AicError::InvalidPerfData(message.clone()),
        _ => AicError::PerfDatabase(err.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn b200_vllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("python/aisimulate/src/aisimulate_core/systems/data/b200_sxm/vllm/0.19.0")
    }

    #[test]
    fn moe_table_loads_b200_vllm() {
        let table = MoeTable::new(b200_vllm_data_root());
        let _ = table.load().expect("moe_perf.parquet must load");
    }

    /// Linear token proxy — fine for key-selection tests where only the
    /// resolution path (not the extrapolated value) matters.
    fn proxy_sol(t: f64) -> f64 {
        t
    }

    #[test]
    fn moe_index_resolves_requested_and_uniform_distributions() {
        let shape = MoeShapeKey {
            topk: 2,
            num_experts: 8,
            hidden_size: 4096,
            inter_size: 2048,
            moe_tp_size: 1,
            moe_ep_size: 4,
        };
        let mut index = MoeIndex::default();
        *index.entry("fp8".into(), "power_law".into(), shape) = LeafAxisCurve::from_map(
            "num_tokens",
            BTreeMap::from([(1, LeafValue::latency_only(1.0))]),
        );
        *index.entry("fp8".into(), "uniform".into(), shape) = LeafAxisCurve::from_map(
            "num_tokens",
            BTreeMap::from([(1, LeafValue::latency_only(2.0))]),
        );
        let grids = MoeGrids {
            index,
            quants_in_load_order: vec!["fp8".to_string()],
        };

        let (dist, curve) = grids.index.resolve_uniform("fp8", "power_law", &shape);
        assert_eq!(dist, "power_law");
        assert_eq!(curve.unwrap().get(1).map(|leaf| leaf.latency), Some(1.0));

        let (dist, curve) = grids.index.resolve_uniform("fp8", "missing", &shape);
        assert_eq!(dist, "uniform");
        assert_eq!(curve.unwrap().get(1).map(|leaf| leaf.latency), Some(2.0));
    }

    #[test]
    fn moe_distribution_falls_back_to_uniform() {
        // Pick any common smoke shape; non-existent distribution should
        // fall back without erroring.
        let table = MoeTable::new(b200_vllm_data_root());
        // Use a shape that's likely covered by vLLM b200 data; if not,
        // the error should be about the topology key, not about
        // missing distribution.
        let result = table.query(
            1024,
            4096,
            2048,
            2,
            128,
            1,
            8,
            MoeQuantMode::Bfloat16,
            "nonexistent_distribution",
            &proxy_sol,
        );
        // Either succeeds (uniform fallback found a match) or errors
        // with a topology mismatch — but not a distribution-specific
        // error.
        match result {
            Ok(value) => assert!(value.latency > 0.0),
            Err(AicError::PerfDatabase(msg)) => {
                assert!(
                    !msg.contains("nonexistent_distribution"),
                    "expected uniform fallback, not literal distribution name in error: {msg}"
                );
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn moe_lazy_loads_once() {
        let table = MoeTable::new(b200_vllm_data_root());
        // Load twice; cached path should produce same outcome.
        let r1 = table.load();
        let r2 = table.load();
        assert_eq!(r1.is_ok(), r2.is_ok());
    }

    #[test]
    fn moe_low_latency_grid_split_routes_by_kernel_source() {
        // Synthetic vehicle (no data-version anchoring): rows labelled
        // `moe_torch_flow_min_latency` must land in the low_latency grid; a
        // table without such rows reports low_latency unavailable.
        use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
        let with_min = tempfile::tempdir().expect("tmpdir");
        write_parquet(
            &with_min.path().join("moe_perf.parquet"),
            &[
                Col::Str("moe_dtype", vec!["bfloat16", "bfloat16"]),
                Col::I64("num_tokens", vec![1024, 1024]),
                Col::I64("hidden_size", vec![4096, 4096]),
                Col::I64("inter_size", vec![2048, 2048]),
                Col::I64("topk", vec![2, 2]),
                Col::I64("num_experts", vec![8, 8]),
                Col::I64("moe_tp_size", vec![1, 1]),
                Col::I64("moe_ep_size", vec![1, 1]),
                Col::Str("distribution", vec!["uniform", "uniform"]),
                Col::Str(
                    "kernel_source",
                    vec!["moe_torch_flow", "moe_torch_flow_min_latency"],
                ),
                Col::F64("latency", vec![1.0, 0.5]),
            ],
        );
        let table = MoeTable::new(with_min.path().to_path_buf());
        assert!(
            table
                .low_latency_available()
                .expect("moe_perf.parquet must load")
        );

        let without = tempfile::tempdir().expect("tmpdir");
        write_parquet(
            &without.path().join("moe_perf.parquet"),
            &[
                Col::Str("moe_dtype", vec!["bfloat16"]),
                Col::I64("num_tokens", vec![1024]),
                Col::I64("hidden_size", vec![4096]),
                Col::I64("inter_size", vec![2048]),
                Col::I64("topk", vec![2]),
                Col::I64("num_experts", vec![8]),
                Col::I64("moe_tp_size", vec![1]),
                Col::I64("moe_ep_size", vec![1]),
                Col::Str("distribution", vec!["uniform"]),
                Col::Str("kernel_source", vec!["moe_torch_flow"]),
                Col::F64("latency", vec![1.0]),
            ],
        );
        let plain = MoeTable::new(without.path().to_path_buf());
        assert!(
            !plain
                .low_latency_available()
                .expect("moe_perf.parquet must load")
        );
    }

    #[test]
    fn moe_exact_kernel_source_selects_the_requested_sglang_lane() {
        // Both lanes cover the same FP8 MoE shape. Unset retains the legacy
        // first-row/default result; an exact selector must use only its lane.
        use crate::perf_database::energy_test_fixtures::{Col, write_parquet};

        let tmp = tempfile::tempdir().expect("tmpdir");
        write_parquet(
            &tmp.path().join("moe_perf.parquet"),
            &[
                Col::Str("moe_dtype", vec!["fp8_block", "fp8_block"]),
                Col::I64("num_tokens", vec![1024, 1024]),
                Col::I64("hidden_size", vec![8192, 8192]),
                Col::I64("inter_size", vec![2048, 2048]),
                Col::I64("topk", vec![10, 10]),
                Col::I64("num_experts", vec![512, 512]),
                Col::I64("moe_tp_size", vec![1, 1]),
                Col::I64("moe_ep_size", vec![8, 8]),
                Col::Str("distribution", vec!["power_law_1.2", "power_law_1.2"]),
                Col::Str(
                    "kernel_source",
                    vec!["sglang_fused_moe_triton", "sglang_flashinfer_trtllm_moe"],
                ),
                Col::F64("latency", vec![1.0, 0.25]),
            ],
        );
        let table = MoeTable::new(tmp.path().to_path_buf());

        let legacy = table
            .query(
                1024,
                8192,
                2048,
                10,
                512,
                1,
                8,
                MoeQuantMode::Fp8Block,
                "power_law_1.2",
                &proxy_sol,
            )
            .expect("default lane resolves");
        let flashinfer = table
            .query_kernel_source(
                "sglang_flashinfer_trtllm_moe",
                1024,
                8192,
                2048,
                10,
                512,
                1,
                8,
                MoeQuantMode::Fp8Block,
                "power_law_1.2",
                &proxy_sol,
            )
            .expect("requested FlashInfer lane resolves");

        assert_eq!(legacy.latency, 1.0);
        assert_eq!(flashinfer.latency, 0.25);

        let error = table
            .query_kernel_source(
                "sglang_missing_moe",
                1024,
                8192,
                2048,
                10,
                512,
                1,
                8,
                MoeQuantMode::Fp8Block,
                "power_law_1.2",
                &proxy_sol,
            )
            .expect_err("an unavailable exact lane must not fall back to the default");
        assert!(error.to_string().contains("sglang_missing_moe"));
    }

    /// ENERGY oracle on a synthetic power-carrying fixture. Python twin
    /// (pandas fixture, `energy_test_fixtures` spec):
    ///
    /// ```text
    /// db.query_moe(num_tokens=1536, hidden_size=4096, inter_size=2048,
    ///              topk=2, num_experts=8, moe_tp_size=1, moe_ep_size=1,
    ///              quant_mode=MoEQuantMode.bfloat16,
    ///              workload_distribution="uniform", database_mode=SILICON)
    /// # -> latency=2.0, energy=300.0
    /// ```
    #[test]
    fn moe_energy_matches_python_oracle() {
        use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_parquet(
            &tmp.path().join("moe_perf.parquet"),
            &[
                Col::Str("moe_dtype", vec!["bfloat16", "bfloat16"]),
                Col::I64("num_tokens", vec![1024, 2048]),
                Col::I64("hidden_size", vec![4096, 4096]),
                Col::I64("inter_size", vec![2048, 2048]),
                Col::I64("topk", vec![2, 2]),
                Col::I64("num_experts", vec![8, 8]),
                Col::I64("moe_tp_size", vec![1, 1]),
                Col::I64("moe_ep_size", vec![1, 1]),
                Col::Str("distribution", vec!["uniform", "uniform"]),
                Col::Str("kernel_source", vec!["moe_torch_flow", "moe_torch_flow"]),
                Col::F64("latency", vec![1.0, 3.0]),
                Col::F64("power", vec![100.0, 200.0]),
            ],
        );
        let table = MoeTable::new(tmp.path().to_path_buf());
        let v = table
            .query(
                1536,
                4096,
                2048,
                2,
                8,
                1,
                1,
                MoeQuantMode::Bfloat16,
                "uniform",
                &proxy_sol,
            )
            .unwrap();
        assert!((v.latency - 2.0).abs() < 1e-9, "latency {}", v.latency);
        assert!(
            (v.energy - 300.0).abs() < 1e-9 * 300.0,
            "energy {}",
            v.energy
        );
    }
}
