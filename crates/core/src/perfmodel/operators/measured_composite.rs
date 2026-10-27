// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact measured operation scopes supplied by an external execution profile.

use serde::{Deserialize, Serialize};

use crate::AicError;
use crate::common::enums::DatabaseMode;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::op::RuntimeContext;
use crate::perf_database::PerfDatabase;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredCell {
    /// (batch, new tokens per request, cached prefix per request) for prefill;
    /// (batch, past KV per request + 1, zero) for decode.
    pub shape: [u32; 3],
    pub latency_ms: f64,
}

/// A measured execution scope represented as one operation.
///
/// The scope may contain multiple kernels or operations. Its latency comes
/// from an exact measured shape lookup instead of modeling its internals.
/// The name identifies the scope; it does not imply a confidential identity.
/// Measurements belong to one system, backend, version, and phase. Missing
/// shapes are errors, with no interpolation or analytical substitution.
/// Resident weights and latency scaling are supplied explicitly.
///
/// Mixed composition requires scopes that separate shared non-attention work
/// from phase-specific attention; a whole-phase measurement cannot supply that
/// split. The execution profile declares this composition contract.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredCompositeOp {
    pub name: String,
    pub system: String,
    pub backend: String,
    pub version: String,
    pub is_context: bool,
    /// Full resident weight inventory of the replaced scope.
    pub weight_bytes: f64,
    pub scale_factor: f64,
    /// Strictly increasing lexicographic shape order. No interpolation or tails.
    #[serde(deserialize_with = "measured_cells")]
    pub cells: Vec<MeasuredCell>,
}

fn measured_cells<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<MeasuredCell>, D::Error> {
    let cells = Vec::<MeasuredCell>::deserialize(d)?;
    if cells.windows(2).any(|pair| pair[0].shape >= pair[1].shape)
        || cells
            .iter()
            .any(|cell| !cell.latency_ms.is_finite() || cell.latency_ms <= 0.0)
    {
        return Err(serde::de::Error::custom(
            "measured cells must be finite, positive, unique and sorted",
        ));
    }
    Ok(cells)
}

pub(crate) fn invalid(message: impl std::fmt::Display) -> AicError {
    AicError::InvalidEngineConfig(format!("external execution profile: {message}"))
}

impl MeasuredCompositeOp {
    pub(crate) fn validate(&self) -> Result<(), AicError> {
        if self.name.trim().is_empty()
            || self.system.trim().is_empty()
            || self.backend.trim().is_empty()
            || self.version.trim().is_empty()
            || !self.weight_bytes.is_finite()
            || self.weight_bytes < 0.0
            || !self.scale_factor.is_finite()
            || self.scale_factor <= 0.0
            || self.cells.is_empty()
        {
            return Err(invalid("invalid measured scope identity, weights or scale"));
        }
        let mut previous = None;
        for cell in &self.cells {
            let [batch, sequence, prefix] = cell.shape;
            if batch == 0
                || sequence == 0
                || sequence.checked_add(prefix).is_none()
                || (self.is_context && batch.checked_mul(sequence).is_none())
                || (!self.is_context && prefix != 0)
                || !cell.latency_ms.is_finite()
                || cell.latency_ms <= 0.0
                || !(cell.latency_ms * self.scale_factor).is_finite()
                || previous.is_some_and(|key| key >= cell.shape)
            {
                return Err(invalid(
                    "measured cells must be finite, positive, unique and sorted",
                ));
            }
            previous = Some(cell.shape);
        }
        Ok(())
    }

    pub(crate) fn validate_database(&self, db: &PerfDatabase) -> Result<(), AicError> {
        if db.database_mode != DatabaseMode::Silicon
            || (db.system.as_str(), db.backend.as_str(), db.version.as_str())
                != (
                    self.system.as_str(),
                    self.backend.as_str(),
                    self.version.as_str(),
                )
        {
            return Err(invalid("measured scope requires its exact SILICON runtime"));
        }
        Ok(())
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        self.validate_database(db)?;
        if !self.scale_factor.is_finite()
            || self.scale_factor <= 0.0
            || ctx.beam_width != 1
            || ctx.seq_imbalance_correction_scale != 1.0
            || ctx.gen_seq_imbalance_correction_scale != 1.0
            || ctx.num_image_tokens != 0
            || ctx.num_tokens
                != if self.is_context {
                    ctx.batch_size
                        .checked_mul(ctx.s)
                        .ok_or_else(|| invalid("token count overflow"))?
                } else {
                    ctx.batch_size
                }
        {
            return Err(invalid("unsupported measured scope query"));
        }
        let key = [ctx.batch_size, ctx.s, ctx.prefix];
        let index = self
            .cells
            .binary_search_by_key(&key, |cell| cell.shape)
            .map_err(|_| invalid(format!("{} has no exact cell for {key:?}", self.name)))?;
        let latency = self.cells[index].latency_ms * self.scale_factor;
        if !latency.is_finite() || latency <= 0.0 {
            return Err(invalid("invalid measured latency"));
        }
        Ok(PerformanceResult::new(latency, Source::Silicon))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op() -> MeasuredCompositeOp {
        MeasuredCompositeOp {
            name: "scope".into(),
            system: "testsys".into(),
            backend: "vllm".into(),
            version: "1.0".into(),
            is_context: true,
            weight_bytes: 0.0,
            scale_factor: 3.0,
            cells: vec![
                MeasuredCell {
                    shape: [1, 4, 0],
                    latency_ms: 2.0,
                },
                MeasuredCell {
                    shape: [1, 4, 4],
                    latency_ms: 5.0,
                },
            ],
        }
    }

    #[test]
    fn measured_scope_has_exact_independent_oracle_and_no_tail() {
        let root = tempfile::tempdir().unwrap();
        crate::perf_database::energy_test_fixtures::write_energy_systems_root(root.path());
        let db = PerfDatabase::load(root.path(), "testsys", "vllm", "1.0").unwrap();
        let op = op();
        op.validate().unwrap();
        let mut ctx = RuntimeContext {
            s: 4,
            num_tokens: 4,
            ..Default::default()
        };
        // Hand-selected measurements: three executions of 2 ms, then 5 ms.
        let measured = op.query(&db, &ctx).unwrap();
        assert_eq!(measured.latency_ms, 6.0);
        assert_eq!(measured.energy_wms, 0.0);
        assert_eq!(measured.source, Source::Silicon);
        assert_eq!(
            measured.plus(PerformanceResult::silicon(1.0)).source,
            Source::Silicon
        );
        ctx.prefix = 4;
        assert_eq!(op.query(&db, &ctx).unwrap().latency_ms, 15.0);
        ctx.s = 5;
        ctx.num_tokens = 5;
        assert!(
            op.query(&db, &ctx)
                .unwrap_err()
                .to_string()
                .contains("no exact cell")
        );
        ctx.s = 4;
        ctx.num_tokens = 4;
        ctx.beam_width = 2;
        assert!(op.query(&db, &ctx).is_err());
    }

    #[test]
    fn measured_scope_rejects_ambiguous_or_nonfinite_cells() {
        let mut op = op();
        op.cells.reverse();
        assert!(op.validate().is_err());
        let json = serde_json::to_string(&op).unwrap();
        assert!(serde_json::from_str::<MeasuredCompositeOp>(&json).is_err());
        op.cells.reverse();
        op.cells[0].latency_ms = f64::NAN;
        assert!(op.validate().is_err());
    }
}
