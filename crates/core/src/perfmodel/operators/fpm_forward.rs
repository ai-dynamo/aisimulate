// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Includes changes adapted from:
// https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/aiconfigurator-core/src/operators/fpm_forward.rs

//! Whole-model forward-pass op (Python `forward_model="fpm"`, class
//! `FPMForwardOp` in `sdk/operations/fpm_forward.py`).
//!
//! With `forward_model="fpm"` the Python model builder replaces each phase op
//! list with exactly one of these; the compiled spec carries it as
//! `Op::FpmForward`. The op answers the standard `query(db, ctx)` contract
//! from the collected [`FpmForwardTable`](crate::perf_database::FpmForwardTable)
//! cells:
//!
//! - prefill coords `(B, B*s, B*prefix)` — `s` is NEW prefill tokens per
//!   request, `prefix` the past-KV per request;
//! - decode coords `(B, B*s)` — one new token per request, `s` the per-request
//!   KV length at this decode step;
//! - hard per-axis domain gate BEFORE interpolation (FPM never extrapolates);
//! - ScatteredSites resolution with `own_curve_coverage_fallback=true` and
//!   `max_site_distance=2.0`; prefill additionally admits sites within 32 raw
//!   KV tokens while retaining the normal log2 gate on batch, so small
//!   block-aligned coordinates around zero remain connected.
//!
//! The SOL roofline anchoring the interpolation is the model's ORIGINAL
//! op-level list (carried as `sol_ops` on the wire) queried in SOL mode with
//! Python's coordinate back-mapping — see [`sol_total`].
//!
//! NOT the crate's `src/fpm/` (`ForwardPassPerfModel`) module: that is the
//! online-tuning model over Dynamo ForwardPassMetrics telemetry, an unrelated
//! concept that shares the "FPM" abbreviation.

use serde::{Deserialize, Serialize};

use crate::common::error::AicError;
use crate::operators::op::{Op, RuntimeContext};
use crate::operators::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;
use crate::perf_database::fpm_forward::{FPM_DECODE_AXES, FPM_PREFILL_AXES, FpmForwardCell};
use crate::perf_database::perf_interp::{OpInterpConfig, Resolver, ValueTransform};

/// Phase of the whole-model op. Serialized lowercase, matching the Python
/// `phase` string and the parquet `workload_kind` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FpmPhase {
    Prefill,
    Decode,
}

impl FpmPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            FpmPhase::Prefill => "prefill",
            FpmPhase::Decode => "decode",
        }
    }
}

/// One whole-model forward pass for a single phase.
///
/// `match_identity` is the 19-string cell identity (legacy schema-6 requests use 15) in
/// [`FPM_CELL_MATCH_COLUMNS`](crate::perf_database::fpm_forward::FPM_CELL_MATCH_COLUMNS)
/// order, computed by the PYTHON producer via `_norm_identity` (None -> "",
/// Enum -> `.name`) so Rust compares strings verbatim with no re-normalization
/// drift. `sol_ops` is the model's original op-level list for this phase —
/// the roofline source, serialized recursively like `Overlap`/`Fallback`
/// children.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FpmForwardOp {
    pub name: String,
    pub phase: FpmPhase,
    pub model_path: String,
    pub match_identity: Vec<String>,
    /// Per-rank whole-model weight bytes (captured from the original op-level
    /// lists at rewrite time). Carried for memory estimation; the latency
    /// path does not read it.
    #[serde(default)]
    pub weight_bytes: f64,
    /// Speculative verify width (tokens verified per request per decode
    /// step). Default 1 = plain AR decode (bit-compatible with pre-field
    /// specs). When > 1, the decode query maps the WIDENED token batch onto
    /// the AR-collected surface at the equivalent-AR point: the engine hands
    /// this op `batch = requests x width` tokens, and the true step reads
    /// each request's KV once, so the coordinates are
    /// `(tokens = batch, total_kv = batch / width * s)` — total new tokens
    /// AND total KV bytes both match the real verify step exactly; only the
    /// per-token QK-PV compute (memory-shadowed in decode) is folded onto
    /// the AR curve. Prefill is unaffected (draft prefill work rides
    /// separate op-level draft ops).
    #[serde(default = "default_verify_width")]
    pub verify_width: u32,
    pub sol_ops: Vec<Op>,
    /// Present only for an explicit table selector. Preserves the model's mode
    /// before match_identity[2] was selected; it is NOT observed runtime precision.
    #[serde(default)]
    pub original_fmha_quant_mode: Option<String>,
}

fn default_verify_width() -> u32 {
    1
}

fn data_err(msg: String) -> AicError {
    AicError::PerfDatabase(msg)
}

impl FpmForwardOp {
    fn select_cell<'a>(&self, db: &'a PerfDatabase) -> Result<&'a FpmForwardCell, AicError> {
        // Exact matching remains authoritative, including the recorded FMHA label.
        // A different precision cell is a miss; the selector never rewrites it.
        let cell = db
            .fpm_forward
            .select_cell(&self.match_identity, &self.model_path)?;
        if let Some(original) = &self.original_fmha_quant_mode {
            if let Some(warning) = cell.fmha_selector_warning(original) {
                // The perfmodel has no installed logging facade. Emit a visible,
                // once-per-cell/model-mode warning rather than a silent log record.
                eprintln!("WARNING: {warning}");
            }
        }
        Ok(cell)
    }

    /// Mirror of Python `FPMForwardOp.query`: validate kwargs, map to
    /// iteration-total coordinates, resolve against the selected cell.
    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        if self.verify_width == 0 || (self.phase == FpmPhase::Prefill && self.verify_width != 1) {
            return Err(data_err(format!(
                "invalid FPM verify_width={} for {}",
                self.verify_width,
                self.phase.as_str()
            )));
        }
        if self.phase == FpmPhase::Decode && ctx.batch_size % self.verify_width != 0 {
            return Err(data_err(format!(
                "FPM decode batch_size={} must be divisible by verify_width={}",
                ctx.batch_size, self.verify_width
            )));
        }
        let batch_size = ctx.batch_size;
        let s = ctx.s;
        if batch_size < 1 || s < 1 {
            return Err(data_err(format!(
                "invalid FPM query: batch_size={batch_size}, s={s}"
            )));
        }
        if ctx.beam_width != 1 {
            return Err(data_err(format!(
                "forward_model='fpm' has no beam-search data (beam_width={}); use \
                 forward_model='op_level'.",
                ctx.beam_width
            )));
        }
        let cell = self.select_cell(db)?;
        let b = batch_size as f64;
        let coords: Vec<f64> = match self.phase {
            FpmPhase::Prefill => {
                let prefix = ctx.prefix as f64;
                vec![b, b * s as f64, b * prefix]
            }
            // `s` is the per-request KV length at this decode step. Plain AR
            // (verify_width == 1): one new token per request, the iteration
            // reads batch*s KV tokens. Verify (width w > 1): `batch` arrives
            // WIDENED (requests x w new tokens); the step still reads each
            // request's KV once, so total KV = (batch / w) * s — the
            // equivalent-AR point on the collected surface (same token count,
            // same KV bytes as the real block-verify step).
            FpmPhase::Decode => {
                let w = f64::from(self.verify_width);
                vec![b, b / w * s as f64]
            }
        };
        self.resolve(db, cell, &coords)
    }

    /// Resolve at RAW per-rank iteration totals — the table's native
    /// coordinate system (prefill `(n_req, total_prefill, total_kv)`, decode
    /// `(n_req, total_kv)`). Used by the ForwardPassMetrics rank dispatch,
    /// where the telemetry already carries totals: converting them to
    /// per-request averages and back (the op-level path's convention) would
    /// only add integer-division rounding.
    pub fn query_totals(
        &self,
        db: &PerfDatabase,
        coords: &[f64],
    ) -> Result<PerformanceResult, AicError> {
        let expected = match self.phase {
            FpmPhase::Prefill => 3,
            FpmPhase::Decode => 2,
        };
        if coords.len() != expected {
            return Err(data_err(format!(
                "FPM {} query_totals expects {expected} coords, got {:?}",
                self.phase.as_str(),
                coords
            )));
        }
        let cell = self.select_cell(db)?;
        self.resolve(db, cell, coords)
    }

    /// Highest collected decode KV-read total for this op's selected cell.
    pub fn decode_kv_ceiling(&self, db: &PerfDatabase) -> Result<Option<u32>, AicError> {
        if self.phase != FpmPhase::Decode {
            return Err(data_err(format!(
                "decode_kv_ceiling is decode-only, called on phase {:?}",
                self.phase.as_str()
            )));
        }
        let cell = self.select_cell(db)?;
        Ok(cell.decode_domain.as_ref().map(|domain| domain[1].1))
    }

    /// Decode-pass baseline at each collected batch curve's own KV floor.
    /// Exact batches use their measured floor directly; off-lattice batches
    /// interpolate the bracket rows' floor latencies along the batch axis —
    /// over exactly the rows [`Self::decode_bracket`] keeps at `total_kv`, so
    /// both sides of `query(B, KV) - query_pass_baseline(B, KV)` run on the
    /// same curve model and the shared-pass cost cancels.
    /// This is a baseline-only left-boundary hold: ordinary decode queries
    /// remain strict and never extrapolate below a curve's collected KV range.
    /// `query(B, KV) - query_pass_baseline(B, KV)` is the decode work's
    /// marginal cost when it rides an existing (mixed) pass.
    pub fn query_pass_baseline(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        total_kv: f64,
    ) -> Result<PerformanceResult, AicError> {
        if self.phase != FpmPhase::Decode {
            return Err(data_err(format!(
                "query_pass_baseline is decode-only, called on phase {:?}",
                self.phase.as_str()
            )));
        }
        if batch_size < 1 {
            return Err(data_err(format!(
                "invalid FPM baseline query: batch_size={batch_size}"
            )));
        }
        let cell = self.select_cell(db)?;
        let Some(domain) = cell.decode_domain else {
            return Err(data_err(format!(
                "FPM cell {:?} has no decode rows (model_path={:?}).",
                cell.cell_ids, cell.model_path
            )));
        };
        let Some(index) = cell.decode_index.as_ref() else {
            return Err(self.no_rows_err(cell));
        };

        // Every curve bound comes from a real row, so resolving
        // (row, curve_min) is an exact leaf and never invokes SOL. Keeping the
        // no-SOL closure explicit makes this policy independent of model SOL
        // support (important for MiniMax-M3's currently unported MSA family).
        let no_sol = |_coords: &[f64]| f64::NAN;
        let cfg = interp_config(FpmPhase::Decode, &no_sol);
        let row_floor = |row: u32| -> Result<f64, AicError> {
            let (kv_floor, _) = cell.decode_curve_bounds.get(&row).ok_or_else(|| {
                data_err(format!(
                    "FPM decode baseline row {row} has no collected KV curve."
                ))
            })?;
            index
                .resolve_value(&cfg, &[row as f64, *kv_floor as f64])
                .map(|value| value.latency)
        };

        let batch = batch_size as f64;
        let latency = if let Some((lo_row, hi_row)) = Self::decode_bracket_rows(cell, batch) {
            // Drop a bracket row from the baseline exactly when
            // `decode_bracket` drops it from the paired query. Blending a row
            // the query could not use leaves part of the shared-pass cost in
            // the marginal — enough to clamp it to 0 in the band only one
            // curve covers, and to inflate it in the mirror band. Adjacent
            // curves' collected KV ranges are ragged on every real table, so
            // both bands exist, widest just under the decode ceiling.
            // Neither row covering is unreachable here (the query on those
            // same rows has already failed), so that arm keeps the blend.
            match (
                Self::row_covers(cell, lo_row, total_kv),
                Self::row_covers(cell, hi_row, total_kv),
            ) {
                (true, false) => row_floor(lo_row)?,
                (false, true) => row_floor(hi_row)?,
                _ => {
                    let lo_value = row_floor(lo_row)?;
                    if hi_row == lo_row {
                        lo_value
                    } else {
                        let hi_value = row_floor(hi_row)?;
                        let weight = (batch - lo_row as f64) / (hi_row as f64 - lo_row as f64);
                        lo_value + (hi_value - lo_value) * weight
                    }
                }
            }
        } else if cell.decode_curve_bounds.contains_key(&batch_size) {
            row_floor(batch_size)?
        } else {
            // Legacy tables without capture-rung pairs retain their previous
            // resolver path. Modern native FPM tables always take one of the
            // exact/bracket branches above.
            let kv_floor = batch_size.max(domain[1].0);
            return self.resolve(db, cell, &[batch, kv_floor as f64]);
        };
        if !latency.is_finite() || latency <= 0.0 {
            return Err(data_err(format!(
                "FPM decode baseline interpolation produced an invalid latency ({latency}) at batch_size={batch_size}."
            )));
        }
        Ok(PerformanceResult::new(latency, Source::Silicon))
    }

    /// Domain gate + ScatteredSites resolution, mirroring `FPMForwardOp._resolve`.
    fn resolve(
        &self,
        db: &PerfDatabase,
        cell: &FpmForwardCell,
        coords: &[f64],
    ) -> Result<PerformanceResult, AicError> {
        // Data-certified prefill batch clamp (mirrors Python _resolve): the
        // regime coordinate is the token TOTAL, which stays untouched — the
        // clamped query prices the same side of the capture cliff and is a
        // bounded upper bound on attention. Decode is NEVER clamped (its
        // batch axis carries a real regime cliff — the Task B partition).
        // KV-pressure tiering (mirrors Python _PREFILL_CLAMP_MAX_KV_PRESSURE):
        // randtok LOO measured the raw clamp at median <= 2% below kv/T = 2
        // (served as-is: measured value, no SOL) and median 14% / p90 96%
        // above it — there the measured ceiling row is rescaled by the
        // whole-model SOL ratio between the true and clamped shapes (LOO:
        // median 4.5% / p90 13%), and ONLY when this model's SOL is usable:
        // an unported roofline (MSA) answers nothing rather than
        // something half-modeled (the batch coordinate then fails the
        // domain gate below).
        const MAX_KV_PRESSURE: f64 = 2.0;
        let clamped: Vec<f64>;
        let mut clamp_scale = 1.0_f64;
        let coords = match (self.phase, cell.prefill_batch_clamp_max) {
            (FpmPhase::Prefill, Some(max)) if coords[0] > max as f64 => {
                let low_pressure = coords[2] < MAX_KV_PRESSURE * coords[1];
                let mut routed = coords;
                if low_pressure {
                    clamped = std::iter::once(max as f64)
                        .chain(coords[1..].iter().copied())
                        .collect();
                    routed = clamped.as_slice();
                } else {
                    let candidate: Vec<f64> = std::iter::once(max as f64)
                        .chain(coords[1..].iter().copied())
                        .collect();
                    let true_sol = sol_total(&self.sol_ops, self.phase, db, coords);
                    let ceiling_sol = sol_total(&self.sol_ops, self.phase, db, &candidate);
                    if let (Ok(t), Ok(c)) = (true_sol, ceiling_sol) {
                        if t.is_finite() && c.is_finite() && t > 0.0 && c > 0.0 {
                            // True shape is never costlier than the clamped
                            // one (more, shorter segments); cap at 1.
                            clamp_scale = (t / c).min(1.0);
                            clamped = candidate;
                            routed = clamped.as_slice();
                        }
                    }
                }
                routed
            }
            _ => coords,
        };
        // Per-axis inclusive bounding-box gate BEFORE perf_interp: whole-model
        // latency has no principled boundary-hold semantics.
        let (axes, domain, index): (&[&str], &[(u32, u32)], _) = match self.phase {
            FpmPhase::Prefill => {
                let Some(domain) = cell.prefill_domain.as_ref() else {
                    return Err(self.no_rows_err(cell));
                };
                (
                    &FPM_PREFILL_AXES,
                    domain.as_slice(),
                    cell.prefill_index.as_ref(),
                )
            }
            FpmPhase::Decode => {
                let Some(domain) = cell.decode_domain.as_ref() else {
                    return Err(self.no_rows_err(cell));
                };
                (
                    &FPM_DECODE_AXES,
                    domain.as_slice(),
                    cell.decode_index.as_ref(),
                )
            }
        };
        for (axis_index, (axis_name, &value)) in axes.iter().zip(coords).enumerate() {
            let (low, high) = domain[axis_index];
            if !((low as f64) <= value && value <= (high as f64)) {
                return Err(data_err(format!(
                    "FPM {} query {axis_name}={value} is outside the collected domain \
                     [{low}, {high}] for model_path={:?}. FPM never extrapolates; collect a \
                     wider sweep or use forward_model='op_level'.",
                    self.phase.as_str(),
                    cell.model_path
                )));
            }
        }
        let index = index.ok_or_else(|| self.no_rows_err(cell))?;

        // SOL support is checked LAZILY, mirroring Python: exact hits and
        // in-curve lerps never invoke the roofline (Python's SOL view answers
        // every op family; `_oplevel_sol_fn` is only called on transfer/hold
        // paths). A family the Rust SOL port does not cover yet (MSA / MLA
        // modules) therefore only fails the sol-dependent resolution paths,
        // and the error names the op.
        let sol_failure: std::cell::RefCell<Option<AicError>> = std::cell::RefCell::new(None);
        let sol = |sol_coords: &[f64]| -> f64 {
            match sol_total(&self.sol_ops, self.phase, db, sol_coords) {
                Ok(v) => v,
                Err(err) => {
                    let mut slot = sol_failure.borrow_mut();
                    if slot.is_none() {
                        *slot = Some(err);
                    }
                    f64::NAN
                }
            }
        };
        let cfg = interp_config(self.phase, &sol);
        if self.phase == FpmPhase::Decode {
            if let Some(latency) = self.decode_bracket(cell, coords, index, &cfg)? {
                if !latency.is_finite() || latency <= 0.0 {
                    return Err(data_err(format!(
                        "FPM decode bracket interpolation produced an invalid latency ({latency}) at {coords:?}."
                    )));
                }
                return Ok(PerformanceResult::new(latency, Source::Silicon));
            }
        }
        let latency = index
            .resolve_value(&cfg, coords)
            .map(|value| value.latency * clamp_scale)
            .map_err(|err| match sol_failure.borrow_mut().take() {
                Some(sol_err) => data_err(format!("{err}; SOL roofline unavailable: {sol_err}")),
                None => err,
            })?;
        if !latency.is_finite() || latency <= 0.0 {
            return Err(data_err(format!(
                "FPM {} interpolation produced an invalid latency ({latency}) at {coords:?}.",
                self.phase.as_str()
            )));
        }
        // Latency-only dataset; scale_factor is fixed 1.0 in Python.
        Ok(PerformanceResult::new(latency, Source::Silicon))
    }

    /// Resolve an OFF-LATTICE decode batch by its segment bracket (mirrors
    /// Python `_decode_bracket`): the engine pads decode batches to capture
    /// rungs — every rung marked by an (x, x+1) row pair — so a query
    /// between rungs interpolates linearly between its segment's bracket
    /// rows {lower_rung + 1, upper_rung}, both running the SAME padded
    /// graph. Coverage of each bracket row's own KV curve is checked here:
    /// an uncovered row degrades to the single covered side (or a loud
    /// miss) — never to the engine's regime-blind cross-batch k-NN.
    ///
    /// `Ok(None)` = own-site hit or no rung structure (legacy path stays).
    fn decode_bracket(
        &self,
        cell: &FpmForwardCell,
        coords: &[f64],
        index: &crate::perf_database::perf_interp::SiteIndex,
        cfg: &OpInterpConfig,
    ) -> Result<Option<f64>, AicError> {
        let (batch, kv) = (coords[0], coords[1]);
        let Some((lo_row, hi_row)) = Self::decode_bracket_rows(cell, batch) else {
            return Ok(None);
        };
        let (lo_ok, hi_ok) = (
            Self::row_covers(cell, lo_row, kv),
            Self::row_covers(cell, hi_row, kv),
        );
        if !lo_ok && !hi_ok {
            return Err(data_err(format!(
                "FPM decode bracket rows {lo_row}/{hi_row} do not cover total_kv_read_tokens={kv}                  (curves span {:?} and {:?}); FPM never extrapolates.",
                cell.decode_curve_bounds[&lo_row], cell.decode_curve_bounds[&hi_row]
            )));
        }
        let row_value = |row: u32| -> Result<f64, AicError> {
            // Own-curve evaluation (coverage pre-checked): the engine's
            // bisect on that row's own curve — cross-batch transfer never
            // runs.
            index
                .resolve_value(cfg, &[row as f64, kv])
                .map(|value| value.latency)
        };
        if !(lo_ok && hi_ok) {
            let row = if lo_ok { lo_row } else { hi_row };
            return Ok(Some(row_value(row)?));
        }
        let lo_value = row_value(lo_row)?;
        if hi_row == lo_row {
            return Ok(Some(lo_value));
        }
        let hi_value = row_value(hi_row)?;
        let weight = (batch - lo_row as f64) / (hi_row as f64 - lo_row as f64);
        Ok(Some(lo_value + (hi_value - lo_value) * weight))
    }

    /// Return the padded-graph bracket rows for an off-lattice decode batch.
    /// Exact rows and legacy tables without rung pairs intentionally return
    /// `None` so their existing resolver path remains intact.
    fn decode_bracket_rows(cell: &FpmForwardCell, batch: f64) -> Option<(u32, u32)> {
        if cell.decode_rungs.is_empty()
            || cell.decode_curve_bounds.keys().any(|&b| b as f64 == batch)
        {
            return None;
        }
        let &lower_rung = cell
            .decode_rungs
            .iter()
            .rev()
            .find(|&&r| (r as f64) < batch)?;
        let lo_row = lower_rung + 1;
        let hi_row = cell
            .decode_rungs
            .iter()
            .copied()
            .find(|&r| (r as f64) >= batch)
            .unwrap_or(*cell.decode_batches.last().expect("non-empty lattice"));
        Some((lo_row, hi_row))
    }

    /// Whether `row`'s own collected KV curve covers `kv`. The ONE coverage
    /// rule, shared by [`Self::decode_bracket`] and
    /// [`Self::query_pass_baseline`] so a query and its pass baseline never
    /// resolve on different rows.
    fn row_covers(cell: &FpmForwardCell, row: u32, kv: f64) -> bool {
        cell.decode_curve_bounds
            .get(&row)
            .is_some_and(|&(low, high)| (low as f64) <= kv && kv <= (high as f64))
    }

    fn no_rows_err(&self, cell: &FpmForwardCell) -> AicError {
        data_err(format!(
            "FPM cell {:?} has no {} rows (model_path={:?}).",
            cell.cell_ids,
            self.phase.as_str(),
            cell.model_path
        ))
    }
}

/// FPM interpolation configs: prefill sites `(batch, kv)` own the new-token
/// curve; decode sites `(batch,)` own the KV curve. Prefill adds a raw-KV
/// fallback around zero to the shared `ScatteredSites` defaults.
fn interp_config<'a>(phase: FpmPhase, sol: &'a dyn Fn(&[f64]) -> f64) -> OpInterpConfig<'a> {
    let (axes, site_axes, curve_axis): (&'static [&'static str], Vec<usize>, usize) = match phase {
        FpmPhase::Prefill => (&FPM_PREFILL_AXES, vec![0, 2], 1),
        FpmPhase::Decode => (&FPM_DECODE_AXES, vec![0], 1),
    };
    OpInterpConfig {
        axes,
        resolver: Resolver::ScatteredSites {
            site_axes,
            curve_axis,
            nn_sites: 4,
            max_site_distance: Some(2.0),
            site_axis_abs_distance_fallback: match phase {
                // Prefill sites are (batch, total KV). Around zero, the
                // log2 metric turns a small token difference into many
                // octaves (0/1 versus the first block-aligned site at 16).
                // Admit raw-KV neighbours within two 16-token blocks while
                // retaining the normal log2 gate on batch.
                FpmPhase::Prefill => Some((1, 32.0)),
                FpmPhase::Decode => None,
            },
            require_curve_coverage: true,
            k_tail: 3,
            own_curve_coverage_fallback: true,
        },
        sol_fn: sol,
        value_transform: ValueTransform::Raw,
        transform_axis: None,
    }
}

// ---------------------------------------------------------------------------
// SOL roofline: the op-level model queried in SOL mode
// ---------------------------------------------------------------------------

/// Whole-model SOL at the given FPM coordinates, mirroring Python
/// `_oplevel_sol_fn` exactly:
///
/// ```text
/// prefill (batch, total_prefill, total_kv):
///     s = max(total_prefill / batch, 1.0)      # float, NOT truncated
///     prefix = total_kv / batch                # float, unclamped
///     x = batch if "logits_gemm" in op.name else total_prefill
/// decode (batch, total_kv):
///     s = max(total_kv / batch, 1.0)
///     x = batch
/// total = Σ op.query(SOL view, x=x, batch_size=batch, beam_width=1, s=s, prefix=prefix)
/// ```
pub(crate) fn sol_total(
    sol_ops: &[Op],
    phase: FpmPhase,
    db: &PerfDatabase,
    coords: &[f64],
) -> Result<f64, AicError> {
    let (batch, s, prefix, default_x) = match phase {
        FpmPhase::Prefill => {
            let (batch, total_prefill, total_kv) = (coords[0], coords[1], coords[2]);
            (
                (batch),
                (total_prefill / batch).max(1.0),
                total_kv / batch,
                total_prefill,
            )
        }
        FpmPhase::Decode => {
            let (batch, total_kv) = (coords[0], coords[1]);
            (batch, (total_kv / batch).max(1.0), 0.0, batch)
        }
    };
    let mut total = 0.0_f64;
    for op in sol_ops {
        let x = if matches!(op, Op::Gemm(_)) && op.name().contains("logits_gemm") {
            batch
        } else {
            default_x
        };
        total += crate::operators::fpm_sol::op_sol_latency_ms(op, db, x, batch, s, prefix)?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::fpm_forward::tests::{default_identity, default_rows, write_pair};

    const SYSTEMS_ROOT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../python/aisimulate/src/aiconfigurator_core/systems"
    );

    /// A PerfDatabase whose fpm_forward table points at a temp pair; the rest
    /// of the tables point at the real checked-in b200/vllm/0.24.0 fixture
    /// (never touched by these tests).
    fn db_with_pair(dir: &std::path::Path) -> PerfDatabase {
        let mut db = PerfDatabase::load(
            std::path::Path::new(SYSTEMS_ROOT),
            "b200_sxm",
            "vllm",
            "0.24.0",
        )
        .expect("fixture db");
        db.set_fpm_forward_for_test(crate::perf_database::FpmForwardTable::new(
            dir.to_path_buf(),
            "b200_sxm",
            "vllm",
            "0.25.1",
        ));
        db
    }

    fn op(phase: FpmPhase) -> FpmForwardOp {
        FpmForwardOp {
            name: format!("fpm_forward_{}", phase.as_str()),
            phase,
            model_path: "org/model-a".to_string(),
            match_identity: default_identity(4),
            weight_bytes: 0.0,
            verify_width: 1,
            original_fmha_quant_mode: None,
            // Empty sol_ops: exact hits and in-curve lerps never call SOL.
            sol_ops: vec![],
        }
    }

    fn ctx(batch_size: u32, s: u32, prefix: u32) -> RuntimeContext {
        RuntimeContext {
            batch_size,
            s,
            prefix,
            ..Default::default()
        }
    }

    #[test]
    fn certified_prefill_batch_clamp_routes_to_the_ceiling() {
        use crate::perf_database::fpm_forward::tests::RowSpec;
        let mk = |batch: u32, total: u32, lat: f64| RowSpec {
            workload_kind: "prefill",
            batch_size: batch,
            total_prefill_tokens: total,
            total_kv_read_tokens: 0,
            latency_ms: lat,
            ..RowSpec::default()
        };
        let mut rows = Vec::new();
        for (b, bump) in [(1u32, 1.0), (2, 1.02), (4, 1.04)] {
            for (total, lat) in [(1024u32, 10.0), (2048, 20.0), (4096, 40.0)] {
                rows.push(mk(b, total, lat * bump));
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &rows);
        let db = db_with_pair(tmp.path());
        // batch 16 > ceiling 4: clamp keeps the TRUE total (regime
        // coordinate) and answers the (4, 4096, 0) leaf.
        let clamped = op(FpmPhase::Prefill)
            .query_totals(&db, &[16.0, 4096.0, 0.0])
            .unwrap();
        assert!(
            (clamped.latency_ms - 41.6).abs() < 1e-9,
            "{}",
            clamped.latency_ms
        );
        // The token axis stays honestly gated.
        let err = op(FpmPhase::Prefill)
            .query_totals(&db, &[16.0, 16384.0, 0.0])
            .unwrap_err();
        assert!(
            err.to_string().contains("outside the collected domain"),
            "{err}"
        );
    }

    #[test]
    fn clamp_tiers_on_the_kv_pressure_ceiling() {
        use crate::perf_database::fpm_forward::tests::RowSpec;
        let mk = |batch: u32, total: u32, kv: u32, lat: f64| RowSpec {
            workload_kind: "prefill",
            batch_size: batch,
            total_prefill_tokens: total,
            total_kv_read_tokens: kv,
            latency_ms: lat,
            ..RowSpec::default()
        };
        let mut rows = Vec::new();
        for (b, bump) in [(1u32, 1.0), (2, 1.02), (4, 1.04)] {
            for (total, lat) in [(1024u32, 10.0), (2048, 20.0), (4096, 40.0)] {
                rows.push(mk(b, total, 0, lat * bump));
                rows.push(mk(b, total, total, lat * bump * 1.2));
                rows.push(mk(b, total, 4 * total, lat * bump * 1.8));
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &rows);
        let db = db_with_pair(tmp.path());
        // kv/T = 1 (< 2): clamps to the (4, 4096, 4096) leaf.
        let low = op(FpmPhase::Prefill)
            .query_totals(&db, &[16.0, 4096.0, 4096.0])
            .unwrap();
        assert!(
            (low.latency_ms - 40.0 * 1.04 * 1.2).abs() < 1e-9,
            "{}",
            low.latency_ms
        );
        // kv/T = 4 (>= 2): this op has NO usable SOL (empty sol_ops -> 0),
        // so the high-pressure tier refuses rather than serve a
        // half-modeled value.
        let err = op(FpmPhase::Prefill)
            .query_totals(&db, &[16.0, 4096.0, 16384.0])
            .unwrap_err();
        assert!(
            err.to_string().contains("outside the collected domain"),
            "{err}"
        );
        // With a usable roofline the same query answers, rescaled by the
        // SOL ratio (elementwise SOL depends on the total alone -> ratio 1,
        // proving the arm activates; the ratio math itself is pinned
        // cross-engine by the parity suite).
        let mut sol_op = op(FpmPhase::Prefill);
        sol_op.sol_ops = vec![Op::Elementwise(crate::operators::ElementwiseOp {
            name: "elementwise".to_string(),
            scale_factor: 1.0,
            bytes_per_token: 4096.0,
            scale_num_tokens: 1,
            seq_split: 0,
        })];
        let high = sol_op.query_totals(&db, &[16.0, 4096.0, 16384.0]).unwrap();
        assert!(
            (high.latency_ms - 40.0 * 1.04 * 1.8).abs() < 1e-9,
            "{}",
            high.latency_ms
        );
    }

    #[test]
    fn prefill_low_kv_uses_nearby_raw_kv_sites() {
        use crate::perf_database::fpm_forward::tests::RowSpec;
        let mk = |total: u32, kv: u32, lat: f64| RowSpec {
            workload_kind: "prefill",
            batch_size: 1,
            total_prefill_tokens: total,
            total_kv_read_tokens: kv,
            latency_ms: lat,
            ..RowSpec::default()
        };
        let rows = vec![
            mk(4096, 0, 40.0),
            mk(8192, 0, 80.0),
            mk(4096, 16, 40.0),
            mk(8192, 16, 80.0),
        ];
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &rows);
        let db = db_with_pair(tmp.path());
        let mut prefill = op(FpmPhase::Prefill);
        prefill.sol_ops = vec![Op::Elementwise(crate::operators::ElementwiseOp {
            name: "elementwise".to_string(),
            scale_factor: 1.0,
            bytes_per_token: 4096.0,
            scale_num_tokens: 1,
            seq_split: 0,
        })];

        // The real replay asks for (B=1, P=8066, KV=1). Its nearest raw-KV
        // sites are 0 and 16: both are numerically close even though log2(0)
        // is floored far away and log2(16/1)=4 exceeds the normal gate of 2.
        let got = prefill.query_totals(&db, &[1.0, 8066.0, 1.0]).unwrap();
        let expected = 40.0 + (80.0 - 40.0) * (8066.0 - 4096.0) / (8192.0 - 4096.0);
        assert!(
            (got.latency_ms - expected).abs() < 1e-9,
            "{}",
            got.latency_ms
        );
    }

    fn glm_dsa_sol_ops(name: &str) -> Vec<Op> {
        let dsa = crate::operators::DsaModuleOp::new(
            name,
            64,
            crate::common::enums::KvCacheQuantMode::Fp8,
            crate::common::enums::FmhaQuantMode::Bfloat16,
            crate::common::enums::GemmQuantMode::Nvfp4,
            "GlmMoeDsaForCausalLM",
            2048,
        );
        if name == "context_attention" {
            vec![Op::DsaContext(dsa)]
        } else {
            vec![Op::DsaGeneration(dsa)]
        }
    }

    /// Inverse-distance weight of one neighbour site, restating the resolver's
    /// `w = 1.0 / (d * d + 1e-12)` over the log2 site metric
    /// (`perf_database/perf_interp.rs:1054-1064` for `d`, `:944-947` for the site
    /// logs, `:1170` for `w`). The FPM tests below have a single varying site axis,
    /// so `d` is one absolute log2 difference.
    fn idw_weight(site: f64, query: f64) -> f64 {
        let d = (site.log2() - query.log2()).abs();
        1.0 / (d * d + 1e-12)
    }

    /// A chunked prefill queries KV totals of 8192*k; k=3 (24576) is not a collected site in
    /// the GLM cells (powers of two only). With the DSA arms the SOL-ratio transfer from the
    /// neighbouring sites (16384, 32768) answers instead of `no usable neighbour site`.
    #[test]
    fn prefill_off_site_kv_resolves_through_the_dsa_sol_transfer() {
        use crate::perf_database::fpm_forward::tests::RowSpec;
        let mk = |total: u32, kv: u32, lat: f64| RowSpec {
            workload_kind: "prefill",
            batch_size: 1,
            total_prefill_tokens: total,
            total_kv_read_tokens: kv,
            latency_ms: lat,
            ..RowSpec::default()
        };
        let rows = vec![
            mk(4096, 16384, 45.0),
            mk(8192, 16384, 90.0),
            mk(4096, 32768, 50.0),
            mk(8192, 32768, 100.0),
        ];
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &rows);
        let db = db_with_pair(tmp.path());
        let mut prefill = op(FpmPhase::Prefill);
        prefill.sol_ops = glm_dsa_sol_ops("context_attention");

        let got = prefill.query_totals(&db, &[1.0, 8192.0, 24576.0]).unwrap();
        assert!(
            got.latency_ms.is_finite() && got.latency_ms > 0.0,
            "{}",
            got.latency_ms
        );

        // EXACT expected value, not a band: the query's site key (batch=1,
        // total_kv=24576) is uncollected, so `SiteIndex::resolve_pair`
        // (`perf_database/perf_interp.rs:1153-1185`) transfers UTIL from the nearest
        // collected sites — `util_i = SOL(site_i) / lat_i`, inverse-distance mean,
        // then `SOL(query) / mean_util`. That is a weighted HARMONIC-style combination
        // of the neighbours, not an arithmetic mean of SOL-rescaled latencies, so a
        // raw-neighbour or plain-lerp answer cannot satisfy it.
        //
        // Prefill site axes are (batch, total_kv) with total_prefill as the curve axis
        // (`interp_config`), and batch is 1 on the query and on both sites, so only the
        // KV axis contributes to `d`. Both neighbour curves carry an EXACT point at
        // total_prefill=8192, so `lat_i` is the measured row latency (90 / 100), not an
        // interpolation. SOL is the op-level roofline under `sol_total`'s documented
        // back-mapping (`s = max(total_prefill/batch, 1)`, `prefix = total_kv/batch`,
        // `x = total_prefill`), restated here so the expected value cannot inherit a
        // mapping bug from the code under test.
        let sol = |batch: f64, total_prefill: f64, total_kv: f64| -> f64 {
            crate::operators::fpm_sol::op_sol_latency_ms(
                &prefill.sol_ops[0],
                &db,
                total_prefill,
                batch,
                (total_prefill / batch).max(1.0),
                total_kv / batch,
            )
            .unwrap()
        };
        let (w_lo, w_hi) = (idw_weight(16384.0, 24576.0), idw_weight(32768.0, 24576.0));
        let mean_util = (w_lo * (sol(1.0, 8192.0, 16384.0) / 90.0)
            + w_hi * (sol(1.0, 8192.0, 32768.0) / 100.0))
            / (w_lo + w_hi);
        let expected = sol(1.0, 8192.0, 24576.0) / mean_util;
        assert!(
            (got.latency_ms - expected).abs() <= 1e-9 * expected,
            "transferred latency {} != the SOL-util transfer {expected}",
            got.latency_ms
        );
        // Independent sanity bound on the answer (no SOL term), so a roofline that
        // degenerated identically on both sides of the check above cannot hide here.
        assert!(
            (60.0..=130.0).contains(&got.latency_ms),
            "transferred latency {} is not between the neighbouring sites",
            got.latency_ms
        );
    }

    /// A decode batch with no (C, C+1) capture-rung pair falls back to ScatteredSites, which
    /// needs the generation roofline for the same reason.
    #[test]
    fn decode_pairless_batch_resolves_through_the_dsa_sol_transfer() {
        use crate::perf_database::fpm_forward::tests::RowSpec;
        let mk = |batch: u32, kv: u32, lat: f64| RowSpec {
            workload_kind: "decode",
            batch_size: batch,
            total_prefill_tokens: 0,
            total_kv_read_tokens: kv,
            latency_ms: lat,
            ..RowSpec::default()
        };
        let rows = vec![
            mk(8, 65536, 20.0),
            mk(8, 131072, 22.0),
            mk(16, 131072, 30.0),
            mk(16, 262144, 33.0),
        ];
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &rows);
        let db = db_with_pair(tmp.path());
        let mut decode = op(FpmPhase::Decode);
        decode.sol_ops = glm_dsa_sol_ops("generation_attention");

        let got = decode.query_totals(&db, &[12.0, 131072.0]).unwrap();
        assert!(
            got.latency_ms.is_finite() && got.latency_ms > 0.0,
            "{}",
            got.latency_ms
        );

        // EXACT expected value, by the same transfer as the prefill case
        // (`perf_database/perf_interp.rs:1153-1185`). Decode sites are (batch,) with
        // total_kv as the curve axis, so `d` is the log2 batch distance; rows 8 and 16
        // both carry an exact curve point at total_kv=131072, so `lat_i` is 22 / 30.
        // The decode SOL back-mapping is `s = max(total_kv/batch, 1)`, `prefix = 0`,
        // `x = batch` (see [`sol_total`]).
        let sol = |batch: f64, total_kv: f64| -> f64 {
            crate::operators::fpm_sol::op_sol_latency_ms(
                &decode.sol_ops[0],
                &db,
                batch,
                batch,
                (total_kv / batch).max(1.0),
                0.0,
            )
            .unwrap()
        };
        let (w_lo, w_hi) = (idw_weight(8.0, 12.0), idw_weight(16.0, 12.0));
        let mean_util = (w_lo * (sol(8.0, 131072.0) / 22.0) + w_hi * (sol(16.0, 131072.0) / 30.0))
            / (w_lo + w_hi);
        let expected = sol(12.0, 131072.0) / mean_util;
        assert!(
            (got.latency_ms - expected).abs() <= 1e-9 * expected,
            "transferred latency {} != the SOL-util transfer {expected}",
            got.latency_ms
        );
        assert!(
            (15.0..=45.0).contains(&got.latency_ms),
            "transferred latency {} is not between the neighbouring sites",
            got.latency_ms
        );
    }

    #[test]
    fn prefill_raw_kv_fallback_preserves_the_batch_gate() {
        use crate::perf_database::fpm_forward::tests::RowSpec;
        let mk = |batch: u32, total: u32, kv: u32| RowSpec {
            workload_kind: "prefill",
            batch_size: batch,
            total_prefill_tokens: total,
            total_kv_read_tokens: kv,
            latency_ms: total as f64 / 100.0,
            ..RowSpec::default()
        };
        let rows = vec![
            mk(1, 4096, 1024),
            mk(1, 8192, 1024),
            mk(16, 4096, 0),
            mk(16, 8192, 0),
            mk(16, 4096, 16),
            mk(16, 8192, 16),
        ];
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &rows);
        let db = db_with_pair(tmp.path());

        // KV=0/16 are within 32 raw tokens, but only at batch 16. They stay
        // inadmissible because batch 16 is four log2 units from batch 1.
        let err = op(FpmPhase::Prefill)
            .query_totals(&db, &[1.0, 8066.0, 1.0])
            .unwrap_err();
        assert!(
            err.to_string().contains("no site within max_site_distance"),
            "{err}"
        );
    }

    #[test]
    fn decode_bracket_resolution_and_coverage_guard() {
        use crate::perf_database::fpm_forward::tests::RowSpec;
        let mk = |batch: u32, kv: u32, lat: f64| RowSpec {
            workload_kind: "decode",
            batch_size: batch,
            total_prefill_tokens: 0,
            total_kv_read_tokens: kv,
            latency_ms: lat,
            ..RowSpec::default()
        };
        let mut rows = Vec::new();
        for (b, base) in [(496u32, 9.5), (497, 10.0), (512, 10.5)] {
            for (i, kv) in [1024u32, 2048, 4096].into_iter().enumerate() {
                rows.push(mk(b, kv, base + i as f64));
            }
        }
        for (i, kv) in [1024u32, 2048, 4096].into_iter().enumerate() {
            rows.push(mk(513, kv, 31.0 + 3.0 * i as f64));
        }
        for (i, kv) in [8192u32, 16384].into_iter().enumerate() {
            rows.push(mk(1024, kv, 62.0 + 3.0 * i as f64));
        }
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &rows);
        let db = db_with_pair(tmp.path());
        let dec = op(FpmPhase::Decode);
        // b=500 in segment (496, 512]: bracket {497, 512}, linear blend.
        let got = dec.query_totals(&db, &[500.0, 2048.0]).unwrap();
        let expected = 11.0 + (11.5 - 11.0) * (500.0 - 497.0) / (512.0 - 497.0);
        assert!(
            (got.latency_ms - expected).abs() < 1e-9,
            "{}",
            got.latency_ms
        );
        // b=600 above the last rung: bracket {513, 1024}; kv=2048 covered
        // only by 513 -> single-sided (never the cross-batch k-NN).
        let got = dec.query_totals(&db, &[600.0, 2048.0]).unwrap();
        assert!((got.latency_ms - 34.0).abs() < 1e-9, "{}", got.latency_ms);
        // kv=8192 covered only by 1024 -> the other single side.
        let got = dec.query_totals(&db, &[600.0, 8192.0]).unwrap();
        assert!((got.latency_ms - 62.0).abs() < 1e-9, "{}", got.latency_ms);
        // kv=6000 covered by NEITHER bracket row: loud miss.
        let err = dec.query_totals(&db, &[600.0, 6000.0]).unwrap_err();
        assert!(err.to_string().contains("bracket rows"), "{err}");
        // Own-site hits untouched by any of this.
        let got = dec.query_totals(&db, &[512.0, 2048.0]).unwrap();
        assert!((got.latency_ms - 11.5).abs() < 1e-9);
    }

    #[test]
    fn decode_exact_hit_returns_leaf_verbatim() {
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &default_rows());
        let db = db_with_pair(tmp.path());
        // decode row (8, 4096) -> 7.0; s = 4096/8 = 512 per request
        let r = op(FpmPhase::Decode).query(&db, &ctx(8, 512, 0)).unwrap();
        assert_eq!(r.latency_ms, 7.0);
        assert_eq!(r.source, Source::Silicon);
    }

    #[test]
    fn prefill_exact_hit_with_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &default_rows());
        let db = db_with_pair(tmp.path());
        // prefill row (1, 2048, 2048) -> 24.0: B=1, s=2048 new tokens, prefix=2048
        let r = op(FpmPhase::Prefill)
            .query(&db, &ctx(1, 2048, 2048))
            .unwrap();
        assert_eq!(r.latency_ms, 24.0);
    }

    #[test]
    fn decode_in_curve_lerp_is_linear_raw() {
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &default_rows());
        let db = db_with_pair(tmp.path());
        // Between (8, 8)->6.0 and (8, 4096)->7.0 at kv=2052 (B=8, s=256.5 ->
        // not integral; use s=256 -> kv=2048): w=(2048-8)/(4096-8)
        let r = op(FpmPhase::Decode).query(&db, &ctx(8, 256, 0)).unwrap();
        let w = (2048.0 - 8.0) / (4096.0 - 8.0);
        let expected = 6.0 + (7.0 - 6.0) * w;
        assert!((r.latency_ms - expected).abs() < 1e-12, "{}", r.latency_ms);
    }

    #[test]
    fn out_of_domain_is_a_hard_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &default_rows());
        let db = db_with_pair(tmp.path());
        // decode kv domain is [8, 65536]; B=8, s=16384 -> kv=131072 > max
        let err = op(FpmPhase::Decode)
            .query(&db, &ctx(8, 16384, 0))
            .unwrap_err();
        assert!(
            err.to_string().contains("outside the collected domain"),
            "{err}"
        );
        // batch below domain min
        let err = op(FpmPhase::Decode)
            .query(&db, &ctx(1, 512, 0))
            .unwrap_err();
        assert!(
            err.to_string().contains("outside the collected domain"),
            "{err}"
        );
    }

    #[test]
    fn invalid_query_args_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &default_rows());
        let db = db_with_pair(tmp.path());
        let err = op(FpmPhase::Decode)
            .query(&db, &ctx(0, 512, 0))
            .unwrap_err();
        assert!(err.to_string().contains("invalid FPM query"), "{err}");
        let mut c = ctx(8, 512, 0);
        c.beam_width = 4;
        let err = op(FpmPhase::Decode).query(&db, &c).unwrap_err();
        assert!(err.to_string().contains("no beam-search data"), "{err}");
    }

    #[test]
    fn pass_baseline_uses_curve_floor() {
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &default_rows());
        let db = db_with_pair(tmp.path());
        // Exact batch=8 uses that curve's measured floor (8, 8) -> 6.0.
        let r = op(FpmPhase::Decode)
            .query_pass_baseline(&db, 8, 4096.0)
            .unwrap();
        assert_eq!(r.latency_ms, 6.0);
        // prefill op: decode-only API
        let err = op(FpmPhase::Prefill)
            .query_pass_baseline(&db, 8, 4096.0)
            .unwrap_err();
        assert!(err.to_string().contains("decode-only"), "{err}");
    }

    #[test]
    fn pass_baseline_holds_each_curve_floor_then_interpolates_batch() {
        use crate::perf_database::fpm_forward::tests::RowSpec;
        let mk = |batch: u32, kv: u32, lat: f64| RowSpec {
            workload_kind: "decode",
            batch_size: batch,
            total_prefill_tokens: 0,
            total_kv_read_tokens: kv,
            latency_ms: lat,
            ..RowSpec::default()
        };
        let rows = vec![
            // Keep the cell-wide KV floor below the baseline query so this
            // reproduces the real MiniMax table's old bracket-coverage miss.
            mk(1, 2, 2.0),
            mk(1, 64, 3.0),
            mk(2, 4, 2.5),
            mk(2, 64, 3.5),
            // Rung pair (8, 9): row 9's floor is above baseline KV=15.
            mk(8, 16, 4.0),
            mk(8, 64, 5.0),
            mk(9, 18, 5.0),
            mk(9, 64, 6.0),
            // Rung pair (16, 17): row 16's floor is further above KV=15.
            mk(16, 32, 9.0),
            mk(16, 64, 10.0),
            mk(17, 34, 10.0),
            mk(17, 64, 11.0),
        ];
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &rows);
        let db = db_with_pair(tmp.path());
        let dec = op(FpmPhase::Decode);

        // Ordinary decode remains strict: neither bracket row 9 nor 16
        // covers total KV=15.
        let err = dec.query_totals(&db, &[15.0, 15.0]).unwrap_err();
        assert!(err.to_string().contains("bracket rows 9/16"), "{err}");

        // Mixed baseline is a narrow exception. At a KV BOTH bracket rows
        // cover it holds row 9 at (9,18)=5 and row 16 at (16,32)=9, then
        // interpolates batch 15 between them.
        let both = 5.0 + (9.0 - 5.0) * (15.0 - 9.0) / (16.0 - 9.0);
        let got = dec.query_pass_baseline(&db, 15, 40.0).unwrap();
        assert!((got.latency_ms - both).abs() < 1e-12, "{}", got.latency_ms);

        // Exact batches use their own curve floor without cross-batch transfer.
        let exact = dec.query_pass_baseline(&db, 16, 40.0).unwrap();
        assert_eq!(exact.latency_ms, 9.0);
    }

    /// The baseline must drop a bracket row exactly when `decode_bracket`
    /// drops it from the paired query. Blending a row the query could not use
    /// leaves part of the shared-pass cost in `query - baseline`: enough to
    /// clamp the marginal to 0 in the band only one curve covers, and to
    /// inflate it in the mirror band.
    #[test]
    fn pass_baseline_drops_the_same_uncovered_row_as_the_query() {
        use crate::perf_database::fpm_forward::tests::RowSpec;
        let mk = |batch: u32, kv: u32, lat: f64| RowSpec {
            workload_kind: "decode",
            batch_size: batch,
            total_prefill_tokens: 0,
            total_kv_read_tokens: kv,
            latency_ms: lat,
            ..RowSpec::default()
        };
        // Ragged curves in BOTH directions across one bracket: row 9 spans
        // [18, 64], row 16 spans [32, 96]. Bracket (9, 16) therefore has a
        // low band [18, 32) only row 9 covers and a high band (64, 96] only
        // row 16 covers -- the shape every collected table has around its
        // decode ceiling.
        let rows = vec![
            mk(1, 2, 2.0),
            mk(1, 96, 3.0),
            mk(2, 4, 2.5),
            mk(2, 96, 3.5),
            mk(8, 16, 4.0),
            mk(8, 96, 5.0),
            mk(9, 18, 5.0),
            mk(9, 64, 6.0),
            mk(16, 32, 9.0),
            mk(16, 96, 10.0),
            mk(17, 34, 10.0),
            mk(17, 96, 11.0),
        ];
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &rows);
        let db = db_with_pair(tmp.path());
        let dec = op(FpmPhase::Decode);
        let marginal = |batch: u32, kv: f64| {
            let q = dec
                .query_totals(&db, &[batch as f64, kv])
                .unwrap()
                .latency_ms;
            let b = dec.query_pass_baseline(&db, batch, kv).unwrap().latency_ms;
            (q, b, (q - b).max(0.0))
        };

        // Low band: only row 9 covers kv=20, so both sides use row 9 and the
        // marginal is that curve's own rise above its floor.
        let (q, b, m) = marginal(15, 20.0);
        assert!(
            (b - 5.0).abs() < 1e-12,
            "baseline {b} must be row 9's floor"
        );
        assert!(
            (q - (5.0 + (20.0 - 18.0) / (64.0 - 18.0))).abs() < 1e-12,
            "{q}"
        );
        assert!((m - (q - 5.0)).abs() < 1e-12, "{m}");
        assert!(m > 0.0, "decode riders must not be free: {m}");

        // High band: only row 16 covers kv=80, so both sides use row 16.
        let (q, b, m) = marginal(15, 80.0);
        assert!(
            (b - 9.0).abs() < 1e-12,
            "baseline {b} must be row 16's floor"
        );
        assert!(
            (q - (9.0 + (80.0 - 32.0) / (96.0 - 32.0))).abs() < 1e-12,
            "{q}"
        );
        assert!((m - (q - 9.0)).abs() < 1e-12, "{m}");

        // Both rows cover kv=40: the blend is unchanged.
        let (_, b, _) = marginal(15, 40.0);
        let blend = 5.0 + (9.0 - 5.0) * (15.0 - 9.0) / (16.0 - 9.0);
        assert!((b - blend).abs() < 1e-12, "{b}");
    }

    /// SOL support is lazy (mirrors Python, whose SOL view answers every op
    /// family): an unported family (e.g. the MLA-BMM module used below)
    /// must NOT block exact hits or in-curve lerps — only sol-dependent
    /// resolution paths fail, naming the unported op.
    #[test]
    fn unsupported_sol_family_is_lazy() {
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &default_rows());
        let db = db_with_pair(tmp.path());
        let mut o = op(FpmPhase::Decode);
        o.sol_ops = vec![Op::MlaBmm(crate::operators::MlaBmmOp {
            name: "mla_bmm_pre".into(),
            scale_factor: 1.0,
            num_heads: 128,
            is_pre: true,
            quant_mode: crate::common::enums::GemmQuantMode::Bfloat16,
        })];
        // Exact hit (8, 4096) -> 7.0: never invokes the roofline.
        let r = o.query(&db, &ctx(8, 512, 0)).unwrap();
        assert_eq!(r.latency_ms, 7.0);
        // In-curve lerp on the own site: RAW linear, no roofline either.
        assert!(o.query(&db, &ctx(8, 256, 0)).is_ok());
        // Uncollected batch -> site transfer NEEDS the roofline -> structured
        // miss naming the unported op (not a panic, not a silent number).
        let err = o.query(&db, &ctx(12, 512, 0)).unwrap_err();
        assert!(
            err.to_string().contains("SOL roofline unavailable"),
            "{err}"
        );
        assert!(err.to_string().contains("no Rust implementation"), "{err}");
    }
    #[test]
    fn verify_query_rejects_invalid_width_before_table_lookup() {
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &default_rows());
        let db = db_with_pair(tmp.path());
        let ctx = RuntimeContext {
            batch_size: 8,
            s: 4096,
            ..Default::default()
        };
        for (phase, width) in [
            (FpmPhase::Decode, 0),
            (FpmPhase::Decode, 3),
            (FpmPhase::Prefill, 8),
        ] {
            let mut query = op(phase);
            query.verify_width = width;
            assert!(
                query
                    .query(&db, &ctx)
                    .unwrap_err()
                    .to_string()
                    .contains("verify_width")
            );
        }
    }
    #[test]
    fn selector_warning_preserves_values_and_rejects_wrong_recorded_precision() {
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &default_rows());
        let db = db_with_pair(tmp.path());
        let mut selected = op(FpmPhase::Decode);
        // Independent original model mode differs from the recorded table label.
        selected.original_fmha_quant_mode = Some("fp8".into());
        let cell = db
            .fpm_forward
            .select_cell(&selected.match_identity, &selected.model_path)
            .unwrap();
        let message = cell.fmha_selector_warning("fp8").unwrap();
        assert!(message.contains("original_model_mode=\"fp8\""));
        assert!(message.contains("selector=\"bfloat16\""));
        assert!(message.contains(&format!("matched_cell_ids={:?}", cell.cell_ids)));
        assert!(message.contains("does not independently verify runtime attention precision"));
        assert!(cell.fmha_selector_warning("fp8").is_none());
        assert!(cell.fmha_selector_warning("float16").is_some());
        let base = op(FpmPhase::Decode);
        let coords = [8.0, 4096.0];
        assert_eq!(
            selected.query_totals(&db, &coords).unwrap(),
            base.query_totals(&db, &coords).unwrap()
        );
        selected.match_identity[2] = "fp8".into();
        // All query routes retain exact matching; no substitution for table precision.
        assert!(
            selected
                .query(&db, &ctx(4, 1024, 0))
                .unwrap_err()
                .to_string()
                .contains("No FPM cell matches")
        );
        assert!(
            selected
                .query_totals(&db, &coords)
                .unwrap_err()
                .to_string()
                .contains("No FPM cell matches")
        );
        assert!(
            selected
                .query_pass_baseline(&db, 4, 4096.0)
                .unwrap_err()
                .to_string()
                .contains("No FPM cell matches")
        );
        assert!(
            selected
                .decode_kv_ceiling(&db)
                .unwrap_err()
                .to_string()
                .contains("No FPM cell matches")
        );
    }
}
