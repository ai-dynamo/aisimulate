// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified native port of heterogeneous AFD/fabric behavior; source revision and
// original paths: root THIRD_PARTY_NOTICES.md (AISimulate heterogeneous AFD).
// Source: https://github.com/LPhgh/aisimulate/tree/af56328c4ba7ffa1b8c42c8375244568c2e7bb8c

//! Attention/FFN disaggregation from source-ordered pool plans.

pub mod plan;
pub mod validation;

/// Pool-specific model composition, part of canonical FPM identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AfdPoolRole {
    A,
    F,
}

use crate::common::error::AicError;

/// Resolved AFD geometry. Counts are physical ranks/GPUs or independent A
/// replicas; sizes describe parallel groups. Shared by Rust and Python callers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct AfdParallelism {
    pub a_replica_count: u32,
    pub f_rank_count: u32,
    pub a_gpu_count: u32,
    pub f_gpu_count: u32,
    pub a_tp_size: u32,
    pub f_moe_tp_size: u32,
    pub f_moe_ep_size: u32,
}

pub fn validate_afd_attention_tp(a_tp_size: u32) -> Result<(), AicError> {
    if a_tp_size == 0 {
        return Err(AicError::InvalidEngineConfig(
            "AFD a_tp_size must be positive".into(),
        ));
    }
    Ok(())
}

/// Resolve an omitted EP to pure EP. Explicit choices are never rewritten.
pub fn resolve_afd_parallelism(
    a_tp_size: u32,
    n_a_nodes: u32,
    n_f_nodes: u32,
    gpus_per_node: u32,
    f_moe_ep_size: Option<u32>,
    num_experts: Option<u32>,
    f_gpus_per_node: Option<u32>,
) -> Result<AfdParallelism, AicError> {
    let f_gpus_per_node = f_gpus_per_node.unwrap_or(gpus_per_node);
    for (name, value) in [
        ("a_tp_size", a_tp_size),
        ("n_a_nodes", n_a_nodes),
        ("n_f_nodes", n_f_nodes),
        ("gpus_per_node", gpus_per_node),
        ("f_gpus_per_node", f_gpus_per_node),
    ] {
        if value == 0 {
            return Err(AicError::InvalidEngineConfig(format!(
                "AFD {name} must be positive"
            )));
        }
    }
    if gpus_per_node % a_tp_size != 0 {
        return Err(AicError::InvalidEngineConfig(format!(
            "AFD a_tp_size={a_tp_size} must divide gpus_per_node={gpus_per_node}"
        )));
    }
    let a_gpu_count = n_a_nodes.checked_mul(gpus_per_node);
    let f_gpu_count = n_f_nodes.checked_mul(f_gpus_per_node);
    let (Some(a_gpu_count), Some(f_gpu_count)) = (a_gpu_count, f_gpu_count) else {
        return Err(AicError::InvalidEngineConfig(
            "AFD GPU geometry exceeds u32 range".into(),
        ));
    };
    if a_gpu_count.checked_add(f_gpu_count).is_none() {
        return Err(AicError::InvalidEngineConfig(
            "AFD total GPU count exceeds u32 range".into(),
        ));
    }
    let f_moe_ep_size = f_moe_ep_size.unwrap_or(f_gpu_count);
    if f_moe_ep_size == 0 || f_gpu_count % f_moe_ep_size != 0 {
        return Err(AicError::InvalidEngineConfig(format!(
            "AFD f_moe_ep_size={f_moe_ep_size} must be positive and divide f_gpu_count={f_gpu_count}"
        )));
    }
    if let Some(num_experts) = num_experts {
        if num_experts == 0 {
            return Err(AicError::NotImplemented(
                "Dense AFD is not implemented; only pure routed MoE is supported".into(),
            ));
        }
        if num_experts % f_moe_ep_size != 0 {
            return Err(AicError::InvalidEngineConfig(format!(
                "AFD f_moe_ep_size={f_moe_ep_size} must divide num_experts={num_experts}"
            )));
        }
    }
    validate_afd_attention_tp(a_tp_size)?;
    let f_moe_tp_size = f_gpu_count / f_moe_ep_size;
    if f_moe_tp_size != 1 {
        return Err(AicError::NotImplemented(format!(
            "AFD f_moe_tp_size={f_moe_tp_size}, f_moe_ep_size={f_moe_ep_size}, f_gpu_count={f_gpu_count} is not implemented; use pure EP with f_moe_ep_size={f_gpu_count}"
        )));
    }
    Ok(AfdParallelism {
        a_replica_count: a_gpu_count / a_tp_size,
        f_rank_count: f_gpu_count,
        a_gpu_count,
        f_gpu_count,
        a_tp_size,
        f_moe_tp_size,
        f_moe_ep_size,
    })
}

/// Source-derived model facts, not user switches that enable unsupported graphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AfdModelShape {
    pub num_shared_experts: u32,
    pub num_moe_layers: u32,
}

impl AfdModelShape {
    pub fn validate(self, num_layers: u32, num_experts: u32, topk: u32) -> Result<(), AicError> {
        if num_layers == 0 || self.num_moe_layers > num_layers || topk > num_experts {
            return Err(AicError::InvalidEngineConfig(
                "AFD model layer/expert geometry is invalid".into(),
            ));
        }
        if num_experts == 0 {
            return Err(AicError::NotImplemented(
                "Dense AFD is not implemented; only pure routed MoE is supported".into(),
            ));
        }
        if topk == 0 {
            return Err(AicError::InvalidEngineConfig(
                "AFD MoE topk must be positive".into(),
            ));
        }
        if self.num_moe_layers == 0 {
            return Err(AicError::NotImplemented(
                "AFD requires at least one routed MoE layer".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) const F_ROUTING_NOT_IMPLEMENTED: &str =
    "AFD routing_location='f' is not implemented; use routing_location='a'.";

/// Where MoE routing information is produced. Independent of boundary-op placement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AfdRoutingLocation {
    #[default]
    A,
    F,
}

impl AfdRoutingLocation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::A => "a",
            Self::F => "f",
        }
    }

    /// Validate before data loading or estimator selection; this is not a data miss.
    pub fn ensure_supported(self) -> Result<(), AicError> {
        match self {
            Self::A => Ok(()),
            Self::F => Err(AicError::NotImplemented(F_ROUTING_NOT_IMPLEMENTED.into())),
        }
    }
}

#[cfg(test)]
mod support_tests {
    use super::*;

    #[test]
    fn pure_dp_ep_defaults_and_counts() {
        let shape = resolve_afd_parallelism(1, 2, 3, 8, None, Some(384), None).unwrap();
        assert_eq!((shape.a_replica_count, shape.a_gpu_count), (16, 16));
        assert_eq!((shape.f_rank_count, shape.f_gpu_count), (24, 24));
        assert_eq!(
            (shape.a_tp_size, shape.f_moe_tp_size, shape.f_moe_ep_size),
            (1, 1, 24)
        );
        let wire = serde_json::to_value(shape).unwrap();
        assert_eq!(wire["a_replica_count"], 16);
        assert!(wire.get("n_a_workers").is_none());
    }

    #[test]
    fn unsupported_combinations_are_not_data_misses() {
        for (tp, ep, experts) in [(1, Some(4), Some(128)), (1, Some(8), Some(0))] {
            assert!(matches!(
                resolve_afd_parallelism(tp, 1, 1, 8, ep, experts, None),
                Err(AicError::NotImplemented(_))
            ));
        }
        for (tp, ep, experts) in [
            (0, None, None),
            (3, None, None),
            (1, Some(0), None),
            (1, Some(3), None),
            (1, Some(8), Some(10)),
        ] {
            assert!(matches!(
                resolve_afd_parallelism(tp, 1, 1, 8, ep, experts, None),
                Err(AicError::InvalidEngineConfig(_))
            ));
        }
        assert!(resolve_afd_parallelism(1, u32::MAX, 1, 8, None, None, None).is_err());
    }

    #[test]
    fn model_shape_accepts_shared_and_mixed_but_rejects_invalid_facts() {
        AfdModelShape {
            num_shared_experts: 0,
            num_moe_layers: 48,
        }
        .validate(48, 128, 8)
        .unwrap();
        for shape in [
            AfdModelShape {
                num_shared_experts: 1,
                num_moe_layers: 48,
            },
            AfdModelShape {
                num_shared_experts: 0,
                num_moe_layers: 47,
            },
        ] {
            shape
                .validate(48, 128, 8)
                .expect("supported mixed MoE geometry");
        }
        assert!(matches!(
            AfdModelShape {
                num_shared_experts: 0,
                num_moe_layers: 49
            }
            .validate(48, 128, 8),
            Err(AicError::InvalidEngineConfig(_))
        ));
    }
}
