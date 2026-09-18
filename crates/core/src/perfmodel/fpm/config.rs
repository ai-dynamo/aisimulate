// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical construction contract for [`super::ForwardPassPerfModel`].

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{EstimatorConfig, ForwardPassWorkerType};
use crate::common::enums::{DatabaseMode, TransferPolicy};
use crate::{AicError, BackendKind};

const fn one() -> u32 {
    1
}

/// Estimator selection. Auto searches the fixed priority list independently
/// of the fallback policy used for an explicitly requested estimator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimationMode {
    #[default]
    Auto,
    OpLevel,
    #[serde(alias = "fpm")]
    FpmInterpolation,
    FpmRegression,
}

impl EstimationMode {
    pub(crate) fn native_name(self) -> Option<&'static str> {
        match self {
            Self::OpLevel => Some("op_level"),
            Self::FpmInterpolation => Some("fpm"),
            Self::Auto | Self::FpmRegression => None,
        }
    }
}

/// Additional fallback after an explicitly selected estimator is unavailable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForwardPassFallbackPolicy {
    #[default]
    #[serde(alias = "error")]
    Deny,
    Allow,
    /// Legacy saved requests permitted only a direct regression fallback.
    /// Retaining this value preserves their behavior when replayed.
    #[serde(rename = "regression")]
    LegacyRegression,
}

/// Target-verification cost; acceptance and scheduler progress belong to replay.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "params",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ForwardPassSpeculationConfig {
    Ngram { num_speculative_tokens: u32 },
}

impl ForwardPassSpeculationConfig {
    pub fn num_speculative_tokens(&self) -> u32 {
        match self {
            Self::Ngram {
                num_speculative_tokens,
            } => *num_speculative_tokens,
        }
    }
}

/// Immutable model identity and selection policy for a forward-pass estimator.
///
/// This is the one public construction schema shared by Rust, Python, Replay,
/// Sweeper, and Planner. Estimator-specific tuning controls live in `estimator_config`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForwardPassPerfModelConfig {
    pub model: String,
    pub system: String,
    pub backend: BackendKind,
    pub worker_type: ForwardPassWorkerType,
    #[serde(default)]
    pub backend_version: Option<String>,

    #[serde(default = "one", alias = "tp_size")]
    pub tp: u32,
    #[serde(default = "one", alias = "pp_size")]
    pub pp: u32,
    #[serde(default = "one", alias = "attention_dp_size")]
    pub attention_dp: u32,
    #[serde(default)]
    pub moe_tp_size: Option<u32>,
    #[serde(default)]
    pub moe_ep_size: Option<u32>,

    #[serde(default, alias = "gemm_dtype")]
    pub gemm_quant_mode: Option<String>,
    #[serde(default, alias = "moe_dtype")]
    pub moe_quant_mode: Option<String>,
    #[serde(default, alias = "fmha_dtype")]
    pub fmha_quant_mode: Option<String>,
    #[serde(default, alias = "kv_cache_dtype")]
    pub kvcache_quant_mode: Option<String>,
    #[serde(default, alias = "comm_dtype")]
    pub comm_quant_mode: Option<String>,

    #[serde(default)]
    pub nextn: u32,
    #[serde(default)]
    pub speculation: Option<ForwardPassSpeculationConfig>,
    #[serde(default)]
    pub kv_block_size: Option<u32>,
    /// Preserve DeepSeek V4.1 decoder replay execution identity.
    #[serde(default)]
    pub decoder_replay: bool,
    #[serde(default)]
    #[serde(alias = "forward_model")]
    pub estimation_mode: EstimationMode,
    #[serde(default)]
    pub database_mode: DatabaseMode,
    /// Explicit transfer-kind tokens. `None` means the core default (all).
    #[serde(default)]
    pub transfer_policy: Option<Vec<String>>,
    /// Ordered request-scoped systems roots. Empty uses normal package/env discovery.
    #[serde(default)]
    pub systems_paths: Vec<PathBuf>,
    #[serde(default)]
    pub fallback_policy: ForwardPassFallbackPolicy,
    #[serde(default)]
    pub estimator_config: EstimatorConfig,
    #[serde(default)]
    pub attention_backend: Option<String>,
    #[serde(default)]
    pub enable_shared_layer: Option<bool>,
    #[serde(default)]
    pub strict_provenance: bool,
}

impl ForwardPassPerfModelConfig {
    pub fn new(
        model: impl Into<String>,
        system: impl Into<String>,
        backend: BackendKind,
        worker_type: ForwardPassWorkerType,
    ) -> Self {
        Self {
            model: model.into(),
            system: system.into(),
            backend,
            worker_type,
            backend_version: None,
            tp: 1,
            pp: 1,
            attention_dp: 1,
            moe_tp_size: None,
            moe_ep_size: None,
            gemm_quant_mode: None,
            moe_quant_mode: None,
            fmha_quant_mode: None,
            kvcache_quant_mode: None,
            comm_quant_mode: None,
            nextn: 0,
            speculation: None,
            kv_block_size: None,
            decoder_replay: false,
            estimation_mode: EstimationMode::Auto,
            database_mode: DatabaseMode::default(),
            transfer_policy: None,
            systems_paths: Vec::new(),
            fallback_policy: ForwardPassFallbackPolicy::Deny,
            estimator_config: EstimatorConfig::default(),
            attention_backend: None,
            enable_shared_layer: None,
            strict_provenance: false,
        }
    }

    pub fn candidate_modes(&self) -> Vec<EstimationMode> {
        use EstimationMode::*;
        let priority = [OpLevel, FpmInterpolation, FpmRegression];
        if self.estimation_mode == Auto {
            return priority.to_vec();
        }
        let mut candidates = vec![self.estimation_mode];
        match self.fallback_policy {
            ForwardPassFallbackPolicy::Allow => candidates.extend(
                priority
                    .into_iter()
                    .filter(|mode| *mode != self.estimation_mode),
            ),
            ForwardPassFallbackPolicy::LegacyRegression
                if self.estimation_mode != FpmRegression =>
            {
                candidates.push(FpmRegression)
            }
            _ => {}
        }
        candidates
    }

    pub(crate) fn validate(&self) -> Result<(), AicError> {
        if self.model.trim().is_empty() {
            return Err(invalid_config("model cannot be empty"));
        }
        if self.system.trim().is_empty() {
            return Err(invalid_config("system cannot be empty"));
        }
        if self
            .backend_version
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(invalid_config("backend_version cannot be empty"));
        }
        if self.tp == 0 || self.pp == 0 || self.attention_dp == 0 {
            return Err(invalid_config("tp, pp, and attention_dp must be positive"));
        }
        if self.moe_tp_size.is_some() != self.moe_ep_size.is_some() {
            return Err(invalid_config(
                "moe_tp_size and moe_ep_size must be configured together",
            ));
        }
        if let (Some(moe_tp), Some(moe_ep)) = (self.moe_tp_size, self.moe_ep_size) {
            if moe_tp == 0 || moe_ep == 0 {
                return Err(invalid_config(
                    "moe_tp_size and moe_ep_size must be positive",
                ));
            }
            if u64::from(self.tp) * u64::from(self.attention_dp)
                != u64::from(moe_tp) * u64::from(moe_ep)
            {
                return Err(invalid_config(
                    "topology requires tp * attention_dp == moe_tp_size * moe_ep_size",
                ));
            }
        }
        if self.nextn > 5 {
            return Err(invalid_config("nextn must be in 0..=5"));
        }
        if let Some(speculation) = &self.speculation {
            if self.nextn != 0 {
                return Err(invalid_config(
                    "ngram speculation cannot be combined with nextn",
                ));
            }
            if self.backend != BackendKind::Vllm {
                return Err(invalid_config("ngram speculation requires backend=vllm"));
            }
            if !(1..=5).contains(&speculation.num_speculative_tokens()) {
                return Err(invalid_config(
                    "ngram num_speculative_tokens must be in 1..=5",
                ));
            }
            if !matches!(
                self.estimation_mode,
                EstimationMode::Auto | EstimationMode::OpLevel
            ) {
                return Err(invalid_config("ngram speculation requires op_level timing"));
            }
        }
        if self.estimation_mode == EstimationMode::FpmInterpolation && self.nextn != 0 {
            return Err(invalid_config(
                "estimation_mode='fpm_interpolation' does not support MTP speculative decoding",
            ));
        }
        TransferPolicy::from_wire(self.transfer_policy.as_deref()).map_err(invalid_config)?;
        for root in &self.systems_paths {
            if root.to_str().is_none() {
                return Err(invalid_config("systems_paths entries must be valid UTF-8"));
            }
            if !root.is_dir() {
                return Err(invalid_config(format!(
                    "systems_paths entry is not an existing directory: {}",
                    root.display()
                )));
            }
        }
        self.estimator_config.validate()
    }
}

fn invalid_config(message: impl Into<String>) -> AicError {
    AicError::InvalidEngineConfig(format!(
        "invalid forward pass perf model config: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(value: serde_json::Value) -> ForwardPassPerfModelConfig {
        let mut base = serde_json::json!({
            "model": "Qwen/Qwen3-32B", "system": "h200_sxm",
            "backend": "vllm", "worker_type": "aggregated"
        });
        base.as_object_mut()
            .unwrap()
            .extend(value.as_object().unwrap().clone());
        serde_json::from_value(base).unwrap()
    }

    #[test]
    fn auto_selects_all_modes_even_when_fallback_is_denied() {
        let cfg = config(serde_json::json!({}));
        assert_eq!(cfg.fallback_policy, ForwardPassFallbackPolicy::Deny);
        assert_eq!(
            cfg.candidate_modes(),
            vec![
                EstimationMode::OpLevel,
                EstimationMode::FpmInterpolation,
                EstimationMode::FpmRegression
            ]
        );
        cfg.validate().unwrap();
    }

    #[test]
    fn prompt_lookup_is_distinct_from_mtp_and_validated_before_selection() {
        let mut cfg = config(serde_json::json!({
            "speculation": {"kind": "ngram", "params": {"num_speculative_tokens": 2}}
        }));
        cfg.validate().unwrap();
        assert_eq!(cfg.nextn, 0);
        assert_eq!(
            serde_json::from_str::<ForwardPassPerfModelConfig>(
                &serde_json::to_string(&cfg).unwrap()
            )
            .unwrap(),
            cfg
        );
        for mode in [
            EstimationMode::FpmInterpolation,
            EstimationMode::FpmRegression,
        ] {
            cfg.estimation_mode = mode;
            assert!(cfg.validate().is_err());
        }
        cfg.estimation_mode = EstimationMode::Auto;
        cfg.nextn = 2;
        assert!(cfg.validate().is_err());
        cfg.nextn = 0;
        cfg.backend = BackendKind::Sglang;
        assert!(cfg.validate().is_err());
        cfg.backend = BackendKind::Vllm;
        for depth in [0, 6] {
            cfg.speculation = Some(ForwardPassSpeculationConfig::Ngram {
                num_speculative_tokens: depth,
            });
            assert!(cfg.validate().is_err());
        }
    }

    #[test]
    fn explicit_selection_and_legacy_fallback_keep_distinct_semantics() {
        let cfg = config(serde_json::json!({"estimation_mode": "fpm_interpolation"}));
        assert_eq!(
            cfg.candidate_modes(),
            vec![EstimationMode::FpmInterpolation]
        );
        let cfg = config(
            serde_json::json!({"estimation_mode": "fpm_interpolation", "fallback_policy": "allow"}),
        );
        assert_eq!(
            cfg.candidate_modes(),
            vec![
                EstimationMode::FpmInterpolation,
                EstimationMode::OpLevel,
                EstimationMode::FpmRegression
            ]
        );
        let cfg = config(
            serde_json::json!({"forward_model": "op_level", "fallback_policy": "regression"}),
        );
        assert_eq!(
            cfg.candidate_modes(),
            vec![EstimationMode::OpLevel, EstimationMode::FpmRegression]
        );
    }

    #[test]
    fn role_and_typed_estimator_controls_are_validated() {
        assert!(
            serde_json::from_value::<ForwardPassPerfModelConfig>(
                serde_json::json!({"model":"m","system":"s","backend":"vllm"})
            )
            .is_err()
        );
        let cfg = config(serde_json::json!({"moe_tp_size": 1, "moe_ep_size": 2}));
        assert!(cfg.validate().is_err());
        let cfg = config(
            serde_json::json!({"estimator_config": {"fpm_regression": {"sampling": {"bins_per_axis": [4,16], "max_observations": 128}}, "correction": {"sampling": {"max_observations": 32}}}}),
        );
        cfg.validate().unwrap();
        assert_eq!(
            cfg.estimator_config.regression_options().max_observations,
            128
        );
        assert_eq!(
            cfg.estimator_config.correction_options().max_observations,
            32
        );
        assert!(
            serde_json::from_value::<EstimatorConfig>(
                serde_json::json!({"fpm_regression": {"unknown": 1}})
            )
            .is_err()
        );
    }
}
