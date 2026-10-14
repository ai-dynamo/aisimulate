// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Architecture and pool-operation validation for AFD plans.
use super::AfdPoolRole;
use crate::common::error::AicError;
use crate::operators::Op;

pub fn validate_afd_model_architecture(architecture: &str) -> Result<(), AicError> {
    if architecture == "DeepseekV41ForCausalLM" {
        return Err(AicError::UnsupportedModel(
            "AFD does not support DeepSeek-V4.1: mixed attention/FFN stages and their weight, \
             scale, Engram, and cache residency are not partitioned. Use agg, disagg, or static \
             estimation."
                .into(),
        ));
    }
    Ok(())
}

/// Validate pool ownership while retaining required A-local collectives.
pub(crate) fn validate_pool_op(op: &Op, role: AfdPoolRole) -> Result<(), AicError> {
    match op {
        Op::TokenScale(o) => return validate_pool_op(&o.op, role),
        Op::Fallback(o) => {
            validate_pool_op(&o.primary, role)?;
            for child in &o.fallback {
                validate_pool_op(child, role)?;
            }
            return Ok(());
        }
        Op::Overlap(_) => {
            return Err(AicError::InvalidEngineConfig(
                "AFD selective plans must expand OverlapOp".into(),
            ));
        }
        _ => {}
    }
    let name = op.name().to_lowercase();
    raise_if_unclassifiable_layer_family(op, &name)?;
    let routed = matches!(op, Op::Moe(_) | Op::MoeExpertCompute(_));
    if routed != (role == AfdPoolRole::F) {
        return Err(AicError::InvalidEngineConfig(format!(
            "AFD pool {:?} cannot execute '{}'",
            role,
            op.name()
        )));
    }
    Ok(())
}

fn raise_if_unclassifiable_layer_family(op: &Op, name: &str) -> Result<(), AicError> {
    let variant_family = match op {
        Op::Mamba2(_) => Some("Mamba"),
        Op::Gdn(_) => Some("GDN"),
        Op::Kda(_) => Some("KDA"),
        _ => None,
    };
    let family = variant_family.or_else(|| {
        if name.contains("mamba") {
            Some("Mamba")
        } else if name.contains("gdn") {
            Some("GDN")
        } else if name.contains("kda") {
            Some("KDA")
        } else {
            None
        }
    });
    if let Some(family) = family {
        return Err(AicError::InvalidEngineConfig(format!(
            "AFD does not support {family} op '{name}'"
        )));
    }
    Ok(())
}
