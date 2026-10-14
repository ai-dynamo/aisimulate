// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable, source-derived layer composition. Indices refer to phase op lists.
use super::AfdPoolRole;
use crate::common::error::AicError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AfdLayerTemplate {
    pub attention_kind: String,
    pub routed: bool,
    /// Replica-wide TP computation (A) or expert computation (F).
    pub ops: Vec<usize>,
    /// Full shared weights and router, queried on rank-local token slices.
    pub rank_ops: Vec<usize>,
    /// Strip phase work scaling when counting physical weight residency.
    pub work_scale: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AfdPhasePlan {
    pub templates: Vec<AfdLayerTemplate>,
    pub prefix: Vec<usize>,
    pub suffix: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AfdExecutionPlan {
    pub role: AfdPoolRole,
    pub model: String,
    pub hidden_size: u32,
    pub num_experts: u32,
    pub topk: u32,
    pub layers: Vec<usize>,
    pub shared_inter_size: u32,
    pub context: AfdPhasePlan,
    pub generation: AfdPhasePlan,
}

impl AfdExecutionPlan {
    pub fn validate(&self, context_len: usize, generation_len: usize) -> Result<(), AicError> {
        let invalid = || {
            AicError::InvalidEngineConfig("invalid AFD execution plan: layer/op coverage".into())
        };
        if self.hidden_size == 0
            || self.num_experts == 0
            || self.topk == 0
            || self.topk > self.num_experts
        {
            return Err(invalid());
        }
        if self.layers.is_empty() || self.context.templates.len() != self.generation.templates.len()
        {
            return Err(invalid());
        }
        for (phase, len) in [
            (&self.context, context_len),
            (&self.generation, generation_len),
        ] {
            let mut seen = vec![false; len];
            let mut rank_binding = vec![None; len];
            for t in &phase.templates {
                if (self.role == AfdPoolRole::F
                    && (!t.rank_ops.is_empty() || t.routed == t.ops.is_empty()))
                    || (self.role == AfdPoolRole::A && t.ops.is_empty())
                    || (!t.routed && !t.rank_ops.is_empty())
                {
                    return Err(invalid());
                }
                let mut local = std::collections::BTreeSet::new();
                if !t.work_scale.is_finite() || t.work_scale <= 0.0 {
                    return Err(invalid());
                }
                for &i in t.ops.iter().chain(&t.rank_ops) {
                    if i >= len || !local.insert(i) {
                        return Err(invalid());
                    }
                    seen[i] = true;
                }
                for (indices, rank_local) in [(&t.ops, false), (&t.rank_ops, true)] {
                    for &i in indices {
                        if rank_binding[i].is_some_and(|old| old != rank_local) {
                            return Err(invalid());
                        }
                        rank_binding[i] = Some(rank_local);
                    }
                }
            }
            if self.role == AfdPoolRole::F && (!phase.prefix.is_empty() || !phase.suffix.is_empty())
            {
                return Err(invalid());
            }
            for &i in phase.prefix.iter().chain(&phase.suffix) {
                if i >= len || seen[i] {
                    return Err(invalid());
                }
                seen[i] = true;
            }
            if seen.contains(&false) || self.layers.iter().any(|i| *i >= phase.templates.len()) {
                return Err(invalid());
            }
            let mut used = vec![false; phase.templates.len()];
            for &i in &self.layers {
                used[i] = true;
            }
            if used.contains(&false) {
                return Err(invalid());
            }
        }
        for (a, b) in self
            .context
            .templates
            .iter()
            .zip(&self.generation.templates)
        {
            if a.routed != b.routed || a.attention_kind != b.attention_kind {
                return Err(invalid());
            }
        }
        let mut first_routed = None;
        for (layer, &template) in self.layers.iter().enumerate() {
            if self.context.templates[template].routed {
                first_routed.get_or_insert(layer);
            } else if let Some(first) = first_routed {
                return Err(AicError::NotImplemented(format!(
                    "AFD supports only a dense prefix followed by routed MoE layers: \
                     dense layer {layer} follows routed layer {first} in {}",
                    self.model
                )));
            }
        }
        if first_routed.is_none() {
            return Err(AicError::NotImplemented(
                "AFD requires at least one routed MoE layer; use agg for dense models".into(),
            ));
        }
        Ok(())
    }
}

/// Exact logical partition; at most two nonempty rank shapes need querying.
pub fn token_shards(tokens: u32, tp: u32) -> [(u32, u32); 2] {
    let remainder = tokens % tp;
    [(tokens / tp + 1, remainder), (tokens / tp, tp - remainder)]
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan() -> AfdExecutionPlan {
        let phase = AfdPhasePlan {
            templates: vec![AfdLayerTemplate {
                attention_kind: "global".into(),
                routed: true,
                ops: vec![0],
                rank_ops: vec![1],
                work_scale: 1.0,
            }],
            prefix: vec![],
            suffix: vec![],
        };
        AfdExecutionPlan {
            role: AfdPoolRole::A,
            model: "test".into(),
            hidden_size: 32,
            num_experts: 4,
            topk: 2,
            layers: vec![0, 0],
            shared_inter_size: 16,
            context: phase.clone(),
            generation: phase,
        }
    }

    #[test]
    fn source_plan_round_trips_and_rejects_missing_or_duplicate_work() {
        let p = plan();
        p.validate(2, 2).unwrap();
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<AfdExecutionPlan>(&json).unwrap(), p);
        let mut invalid = p.clone();
        invalid.layers[0] = 1;
        assert!(invalid.validate(2, 2).is_err());
        invalid = p.clone();
        invalid.context.templates[0].rank_ops = vec![0];
        assert!(invalid.validate(2, 2).is_err());
        invalid = p.clone();
        invalid.generation.templates[0].routed = false;
        assert!(invalid.validate(2, 2).is_err());
        assert!(p.validate(3, 2).is_err());
    }

    #[test]
    fn interned_templates_cover_ops_without_repeating_one_time_work() {
        let mut p = plan();
        p.context.templates.push(p.context.templates[0].clone());
        p.generation
            .templates
            .push(p.generation.templates[0].clone());
        p.layers = vec![0, 1];
        p.validate(2, 2).unwrap();
        p.layers = vec![0];
        assert!(p.validate(2, 2).is_err());
        p.layers = vec![0, 1];
        p.context.templates[1].ops = vec![1];
        p.context.templates[1].rank_ops = vec![0];
        assert!(p.validate(2, 2).is_err());
        p.context.templates[1] = p.context.templates[0].clone();
        p.context.prefix = vec![0];
        assert!(p.validate(2, 2).is_err());
    }

    #[test]
    fn only_a_dense_prefix_followed_by_routed_layers_is_supported() {
        for role in [AfdPoolRole::A, AfdPoolRole::F] {
            let mut p = plan();
            p.role = role;
            for phase in [&mut p.context, &mut p.generation] {
                phase.templates[0].rank_ops.clear();
                let mut dense = phase.templates[0].clone();
                dense.routed = false;
                if role == AfdPoolRole::F {
                    dense.ops.clear();
                }
                phase.templates.push(dense);
            }
            p.layers = vec![1, 1, 0, 0];
            p.validate(1, 1).unwrap();
            for layers in [vec![0, 1], vec![1, 0, 1], vec![0, 1, 0]] {
                p.layers = layers;
                let error = p.validate(1, 1).unwrap_err();
                assert!(matches!(error, AicError::NotImplemented(_)));
                assert!(
                    error
                        .to_string()
                        .contains("dense prefix followed by routed MoE")
                );
            }
            p.context.templates.remove(0);
            p.generation.templates.remove(0);
            p.layers = vec![0];
            let len = usize::from(role == AfdPoolRole::A);
            assert!(matches!(
                p.validate(len, len),
                Err(AicError::NotImplemented(_))
            ));
        }
    }
    #[test]
    fn token_partition_conserves_uneven_and_empty_slices() {
        assert_eq!(token_shards(10, 4), [(3, 2), (2, 2)]);
        assert_eq!(token_shards(2, 4), [(1, 2), (0, 2)]);
        for t in [1, 2, 4, 8] {
            for n in 0..19 {
                assert_eq!(
                    token_shards(n, t)
                        .iter()
                        .map(|(n, ranks)| n * ranks)
                        .sum::<u32>(),
                    n
                );
            }
        }
    }
}
