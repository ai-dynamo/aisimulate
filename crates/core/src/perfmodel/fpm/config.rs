// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical construction contract for [`super::ForwardPassPerfModel`].

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::common::enums::{DatabaseMode, TransferPolicy};
use crate::{AicError, BackendKind};

const fn one() -> u32 {
    1
}

/// Forward-pass modeling implementation selected for the compiled engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForwardPassModelKind {
    /// Compose the forward pass from individually modeled operations.
    #[default]
    OpLevel,
    /// Query the exact whole-forward performance database.
    Fpm,
}

impl ForwardPassModelKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::OpLevel => "op_level",
            Self::Fpm => "fpm",
        }
    }
}

/// Policy used when the native AIC estimator cannot serve the requested config.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForwardPassFallbackPolicy {
    /// Fail closed. Replay and Sweeper use this policy unless explicitly changed.
    #[default]
    Error,
    /// Use the in-memory regression model and require observations before estimates.
    Regression,
}

/// Immutable model identity and selection policy for a forward-pass estimator.
///
/// This is the one public construction schema shared by Rust, Python, Replay,
/// Sweeper, and Planner. Runtime learning/tuning controls deliberately live in
/// [`super::ForwardPassPerfOptions`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForwardPassPerfModelConfig {
    pub model: String,
    pub system: String,
    pub backend: BackendKind,
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
    pub kv_block_size: Option<u32>,
    #[serde(default)]
    pub forward_model: ForwardPassModelKind,
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
}

impl ForwardPassPerfModelConfig {
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
        if self.forward_model == ForwardPassModelKind::Fpm && self.nextn != 0 {
            return Err(invalid_config(
                "forward_model='fpm' does not support MTP speculative decoding",
            ));
        }
        TransferPolicy::from_wire(self.transfer_policy.as_deref()).map_err(invalid_config)?;
        for root in &self.systems_paths {
            if !root.is_dir() {
                return Err(invalid_config(format!(
                    "systems_paths entry is not an existing directory: {}",
                    root.display()
                )));
            }
        }
        Ok(())
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

    fn config() -> ForwardPassPerfModelConfig {
        ForwardPassPerfModelConfig {
            model: "Qwen/Qwen3-32B".into(),
            system: "h200_sxm".into(),
            backend: BackendKind::Vllm,
            backend_version: Some("0.19.0".into()),
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
            kv_block_size: None,
            forward_model: ForwardPassModelKind::OpLevel,
            database_mode: DatabaseMode::Silicon,
            transfer_policy: None,
            systems_paths: Vec::new(),
            fallback_policy: ForwardPassFallbackPolicy::Error,
        }
    }

    #[test]
    fn serde_defaults_are_fail_closed_and_typed() {
        let parsed: ForwardPassPerfModelConfig = serde_json::from_value(serde_json::json!({
            "model": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "backend": "vllm"
        }))
        .unwrap();
        assert_eq!(parsed.tp, 1);
        assert_eq!(parsed.forward_model, ForwardPassModelKind::OpLevel);
        assert_eq!(parsed.fallback_policy, ForwardPassFallbackPolicy::Error);
        parsed.validate().unwrap();
    }

    #[test]
    fn validation_rejects_invalid_policy_and_topology() {
        let mut invalid = config();
        invalid.transfer_policy = Some(vec!["mystery".into()]);
        assert!(invalid.validate().is_err());

        invalid = config();
        invalid.moe_tp_size = Some(1);
        invalid.moe_ep_size = Some(2);
        assert!(invalid.validate().is_err());

        invalid = config();
        invalid.forward_model = ForwardPassModelKind::Fpm;
        invalid.nextn = 1;
        assert!(invalid.validate().is_err());
    }
}
