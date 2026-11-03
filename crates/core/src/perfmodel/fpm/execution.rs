// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact recorded execution identity; no timing math or alternative model factory.

use serde::{Deserialize, Serialize};

use crate::AicError;

pub const FPM_SPECULATIVE_ROUND_PREFIX: &str = "speculative_round_v1:";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FpmExecutionKind {
    SpeculativeRound,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FpmSpeculativeMethod {
    Eagle3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FpmTimingScope {
    ModelStepCuda,
}

/// A complete target-plus-draft iteration, measured with the native CUDA timer.
/// Acceptance affects replay progress and is deliberately not a timing selector.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FpmExecutionProfile {
    pub kind: FpmExecutionKind,
    pub method: FpmSpeculativeMethod,
    pub num_draft_tokens: u32,
    pub draft_model: String,
    pub draft_revision: String,
    pub timing_scope: FpmTimingScope,
}

fn invalid(message: impl Into<String>) -> AicError {
    AicError::InvalidEngineConfig(format!(
        "invalid FPM execution identity: {}",
        message.into()
    ))
}

fn pinned_revision(name: &str, value: &str) -> Result<(), AicError> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(invalid(format!(
            "{name} must be a lowercase 40-hex immutable checkpoint revision"
        )));
    }
    Ok(())
}

impl FpmExecutionProfile {
    pub fn validate(&self) -> Result<(), AicError> {
        if !(1..=5).contains(&self.num_draft_tokens) {
            return Err(invalid("num_draft_tokens must be in 1..=5"));
        }
        if self.draft_model.trim().is_empty()
            || self.draft_model.trim() != self.draft_model
            || self.draft_model.chars().any(char::is_control)
        {
            return Err(invalid(
                "draft_model must be nonblank, trimmed, and contain no control characters",
            ));
        }
        pinned_revision("draft_revision", &self.draft_revision)
    }
}

/// Parent checkpoint revision and the deployment's full-round identity.
/// Field order defines the version-1 selector serialization; consumers must use
/// `selector`, not invent a JSON ordering or use only a draft/checkpoint hash.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FpmExecutionBinding {
    pub model_revision: String,
    pub execution_profile: FpmExecutionProfile,
}

impl FpmExecutionBinding {
    pub fn validate(&self) -> Result<(), AicError> {
        pinned_revision("model_revision", &self.model_revision)?;
        self.execution_profile.validate()
    }

    pub fn selector(&self) -> Result<String, AicError> {
        self.validate()?;
        let json = serde_json::to_string(self).map_err(|error| invalid(error.to_string()))?;
        Ok(format!("{FPM_SPECULATIVE_ROUND_PREFIX}{json}"))
    }

    /// Ordinary AR and other established execution labels keep their behavior.
    /// The speculative namespace is reserved: malformed/unknown versions and
    /// noncanonical encodings fail instead of becoming silently separate cells.
    pub fn parse_selector(selector: &str) -> Result<Option<Self>, AicError> {
        if !selector.starts_with("speculative_round") {
            return Ok(None);
        }
        let payload = selector
            .strip_prefix(FPM_SPECULATIVE_ROUND_PREFIX)
            .ok_or_else(|| invalid("unsupported speculative_round selector version"))?;
        let binding: Self =
            serde_json::from_str(payload).map_err(|error| invalid(error.to_string()))?;
        if binding.selector()? != selector {
            return Err(invalid(
                "speculative_round selector must use canonical serialization",
            ));
        }
        Ok(Some(binding))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding() -> FpmExecutionBinding {
        FpmExecutionBinding {
            model_revision: "1".repeat(40),
            execution_profile: FpmExecutionProfile {
                kind: FpmExecutionKind::SpeculativeRound,
                method: FpmSpeculativeMethod::Eagle3,
                num_draft_tokens: 3,
                draft_model: "test/draft".into(),
                draft_revision: "2".repeat(40),
                timing_scope: FpmTimingScope::ModelStepCuda,
            },
        }
    }

    #[test]
    fn selector_pins_every_identity_without_changing_legacy_labels() {
        let original = binding();
        let selector = original.selector().unwrap();
        assert_eq!(
            FpmExecutionBinding::parse_selector(&selector).unwrap(),
            Some(original.clone())
        );
        for label in ["full", "decoder_bounded", "prefix_swa_replay", ""] {
            assert_eq!(FpmExecutionBinding::parse_selector(label).unwrap(), None);
        }
        for (field, value) in [
            ("model_revision", "3".repeat(40)),
            ("draft_revision", "4".repeat(40)),
            ("draft_model", "test/other-draft".into()),
            ("num_draft_tokens", "4".into()),
        ] {
            let mut changed = original.clone();
            match field {
                "model_revision" => changed.model_revision = value,
                "draft_revision" => changed.execution_profile.draft_revision = value,
                "draft_model" => changed.execution_profile.draft_model = value,
                _ => changed.execution_profile.num_draft_tokens = 4,
            }
            assert_ne!(changed.selector().unwrap(), selector);
        }
    }

    #[test]
    fn malformed_or_noncanonical_speculative_identity_is_rejected() {
        let selector = binding().selector().unwrap();
        for bad in [
            selector.replacen(':', ": ", 1),
            selector.replace("_v1:", "_v2:"),
            selector.replace("\"eagle3\"", "\"ngram\""),
            selector.replace("model_step_cuda", "scheduler_wall_time"),
            selector.replace("\"num_draft_tokens\":3", "\"num_draft_tokens\":true"),
            selector.replace("\"num_draft_tokens\":3", "\"num_draft_tokens\":0"),
            selector.replace("\"num_draft_tokens\":3", "\"num_draft_tokens\":6"),
            selector.replace("\"kind\":", "\"unknown\":1,\"kind\":"),
            selector.replace(&"2".repeat(40), "main"),
        ] {
            assert!(FpmExecutionBinding::parse_selector(&bad).is_err(), "{bad}");
        }
    }
}
