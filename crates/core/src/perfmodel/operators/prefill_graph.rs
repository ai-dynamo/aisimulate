// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Latency-only composite scopes, admitted exclusively through the measured profile.
use serde::{Deserialize, Serialize};

use crate::common::enums::DatabaseMode;
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::op::RuntimeContext;
use crate::perf_database::{PerfDatabase, prefill_graph};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SglangPrefillAttentionSequenceOp {
    pub name: String,
    pub profile_id: String,
    /// Original full-model DSA inventory, not the two fixture modules.
    pub weight_bytes: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SglangPrefillCommNormBoundaryOp {
    pub name: String,
    pub profile_id: String,
    pub boundary_role: String,
}

fn validate(db: &PerfDatabase, ctx: &RuntimeContext, id: &str) -> Result<(), AicError> {
    prefill_graph::validate_id(id)?;
    if db.database_mode != DatabaseMode::Silicon
        || db.system != "vr200_hecate"
        || db.backend != "sglang"
        || db.version != prefill_graph::VERSION
        || ctx.beam_width != 1
        || ctx.seq_imbalance_correction_scale != 1.0
    {
        return Err(prefill_graph::error(
            "unsupported runtime or mode for composite query",
        ));
    }
    if prefill_graph::inner_shape(ctx.batch_size, ctx.s, ctx.prefix)? != ctx.num_tokens {
        return Err(prefill_graph::error(
            "operation token count disagrees with exact context shape",
        ));
    }
    Ok(())
}
impl SglangPrefillAttentionSequenceOp {
    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        validate(db, ctx, &self.profile_id)?;
        Ok(PerformanceResult::new(
            db.prefill_graph
                .attention(ctx.batch_size, ctx.s, ctx.prefix)?,
            Source::Estimated,
        ))
    }
}
impl SglangPrefillCommNormBoundaryOp {
    pub fn count(&self) -> Result<u32, AicError> {
        match self.boundary_role.as_str() {
            "post_attention" => Ok(78),
            "following_mlp" => Ok(77),
            _ => Err(prefill_graph::error("unknown communication boundary role")),
        }
    }
    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        validate(db, ctx, &self.profile_id)?;
        let latency = db
            .prefill_graph
            .boundary(ctx.num_tokens, &self.boundary_role)?;
        Ok(PerformanceResult::new(
            latency * self.count()? as f64,
            Source::Estimated,
        ))
    }
}
