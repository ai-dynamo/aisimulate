// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical construction contract for [`super::ForwardPassPerfModel`].

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{EstimatorConfig, ForwardPassWorkerType};
use crate::common::enums::{DatabaseMode, EncoderParallel, TransferPolicy};
use crate::{AicError, BackendKind};

const fn one() -> u32 {
    1
}

/// The literal a saved identity records for a fact its producer did not
/// know, such as a Dynamo planner's deployment hardware or parallelism.
pub const UNKNOWN_IDENTITY: &str = "unknown";

/// The in-memory value of an integer identity fact recorded as `"unknown"`.
/// Configuration input cannot spell it as a number, and validation admits it
/// only for estimators that read no performance data.
pub const UNKNOWN_SIZE: u32 = u32::MAX;

/// Integer identity fact whose JSON/YAML form is a number or `"unknown"`.
struct IdentitySize(u32);

impl Serialize for IdentitySize {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() && self.0 == UNKNOWN_SIZE {
            serializer.serialize_str(UNKNOWN_IDENTITY)
        } else {
            serializer.serialize_u32(self.0)
        }
    }
}

impl<'de> Deserialize<'de> for IdentitySize {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{Error, Unexpected};
        if !deserializer.is_human_readable() {
            return u32::deserialize(deserializer).map(Self);
        }
        struct Visitor;
        impl serde::de::Visitor<'_> for Visitor {
            type Value = IdentitySize;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("an integer below 4294967295 or \"unknown\"")
            }
            fn visit_u64<E: Error>(self, value: u64) -> Result<IdentitySize, E> {
                u32::try_from(value)
                    .ok()
                    .filter(|value| *value != UNKNOWN_SIZE)
                    .map(IdentitySize)
                    .ok_or_else(|| E::invalid_value(Unexpected::Unsigned(value), &self))
            }
            fn visit_i64<E: Error>(self, value: i64) -> Result<IdentitySize, E> {
                u64::try_from(value)
                    .map_err(|_| E::invalid_value(Unexpected::Signed(value), &self))
                    .and_then(|value| self.visit_u64(value))
            }
            fn visit_str<E: Error>(self, value: &str) -> Result<IdentitySize, E> {
                (value == UNKNOWN_IDENTITY)
                    .then_some(IdentitySize(UNKNOWN_SIZE))
                    .ok_or_else(|| E::invalid_value(Unexpected::Str(value), &self))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

/// Serde adapter for an integer identity fact that may be `"unknown"`.
pub(crate) mod identity_size {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(crate) fn serialize<S: Serializer>(value: &u32, serializer: S) -> Result<S::Ok, S::Error> {
        super::IdentitySize(*value).serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u32, D::Error> {
        super::IdentitySize::deserialize(deserializer).map(|size| size.0)
    }
}

/// Serde adapter for an optional integer identity fact that may be `"unknown"`.
pub(crate) mod identity_optional_size {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(crate) fn serialize<S: Serializer>(
        value: &Option<u32>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.map(super::IdentitySize).serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<u32>, D::Error> {
        Option::<super::IdentitySize>::deserialize(deserializer).map(|size| size.map(|size| size.0))
    }
}

/// One recorded deployment limit: a positive integer or `"unknown"`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DeploymentLimit {
    Known(u64),
    #[default]
    Unknown,
}

impl DeploymentLimit {
    /// The recorded value, `None` when it is unknown.
    pub fn known(self) -> Option<u64> {
        match self {
            Self::Known(value) => Some(value),
            Self::Unknown => None,
        }
    }
}

impl Serialize for DeploymentLimit {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match (self, serializer.is_human_readable()) {
            (Self::Known(value), true) => serializer.serialize_u64(*value),
            (Self::Unknown, true) => serializer.serialize_str(UNKNOWN_IDENTITY),
            (_, false) => self.known().serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for DeploymentLimit {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{Error, Unexpected};
        if !deserializer.is_human_readable() {
            return Option::<u64>::deserialize(deserializer)
                .map(|value| value.map_or(Self::Unknown, Self::Known));
        }
        struct Visitor;
        impl serde::de::Visitor<'_> for Visitor {
            type Value = DeploymentLimit;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a positive integer or \"unknown\"")
            }
            fn visit_u64<E: Error>(self, value: u64) -> Result<DeploymentLimit, E> {
                (value > 0)
                    .then_some(DeploymentLimit::Known(value))
                    .ok_or_else(|| E::invalid_value(Unexpected::Unsigned(value), &self))
            }
            fn visit_i64<E: Error>(self, value: i64) -> Result<DeploymentLimit, E> {
                u64::try_from(value)
                    .map_err(|_| E::invalid_value(Unexpected::Signed(value), &self))
                    .and_then(|value| self.visit_u64(value))
            }
            fn visit_str<E: Error>(self, value: &str) -> Result<DeploymentLimit, E> {
                (value == UNKNOWN_IDENTITY)
                    .then_some(DeploymentLimit::Unknown)
                    .ok_or_else(|| E::invalid_value(Unexpected::Str(value), &self))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

/// Limits of the deployment a model was calibrated on, recorded as identity
/// by a producer such as the Dynamo planner. Estimation never reads them and
/// snapshot restoration compares them exactly. Replay and prediction read only
/// a known `max_kv_tokens`, as the KV capacity of a regression or learned role
/// without explicit capacity. Recommendation of such a snapshot role also
/// requires its GPUs per replica to equal a known `num_gpus` and caps its
/// scheduler domains at `max_num_batched_tokens` and `max_num_seqs`. Nothing
/// reads `context_length`. An omitted field is unknown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentLimits {
    /// GPUs of one worker: tp * pp * attention_dp (* cp_size).
    #[serde(default)]
    pub num_gpus: DeploymentLimit,
    /// The scheduler's token budget per iteration.
    #[serde(default)]
    pub max_num_batched_tokens: DeploymentLimit,
    /// The scheduler's concurrent sequence limit.
    #[serde(default)]
    pub max_num_seqs: DeploymentLimit,
    /// KV-cache capacity of one attention-DP rank, in tokens.
    #[serde(default)]
    pub max_kv_tokens: DeploymentLimit,
    /// Maximum model context length.
    #[serde(default)]
    pub context_length: DeploymentLimit,
}

impl DeploymentLimits {
    /// Every limit with its field name, in field order.
    pub fn entries(&self) -> [(&'static str, DeploymentLimit); 5] {
        [
            ("num_gpus", self.num_gpus),
            ("max_num_batched_tokens", self.max_num_batched_tokens),
            ("max_num_seqs", self.max_num_seqs),
            ("max_kv_tokens", self.max_kv_tokens),
            ("context_length", self.context_length),
        ]
    }
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
    /// Explicit artifact-backed forests; never part of automatic discovery.
    FpmLearned,
}

impl EstimationMode {
    pub(crate) fn native_name(self) -> Option<&'static str> {
        match self {
            Self::OpLevel => Some("op_level"),
            Self::FpmInterpolation => Some("fpm"),
            Self::Auto | Self::FpmRegression | Self::FpmLearned => None,
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

/// Speculative cost identity shared with the SDK scheme registry.
/// Acceptance and scheduler progress belong to replay, not this configuration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForwardPassSpeculationConfig {
    pub kind: String,
    #[serde(default)]
    pub params: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_model_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_config: Option<serde_json::Map<String, serde_json::Value>>,
}

impl ForwardPassSpeculationConfig {
    /// Preserve the existing prompt-lookup contract for canonical and replay callers.
    pub(crate) fn ngram_depth(&self) -> Result<u32, AicError> {
        if self.kind != "ngram" {
            return Err(invalid_config("expected ngram speculation"));
        }
        if self.params.len() != 1 || !self.params.contains_key("num_speculative_tokens") {
            return Err(invalid_config(
                "ngram params must contain only num_speculative_tokens",
            ));
        }
        if self.draft_model_path.is_some() || self.draft_config.is_some() {
            return Err(invalid_config("ngram does not accept a draft checkpoint"));
        }
        self.params["num_speculative_tokens"]
            .as_u64()
            .filter(|depth| (1..=5).contains(depth))
            .map(|depth| depth as u32)
            .ok_or_else(|| {
                invalid_config("ngram num_speculative_tokens must be an integer in 1..=5")
            })
    }
}

// Keep one field list for the public schema and its private wire decoder.
// A remote derive on the public type would shadow Deserialize::deserialize.
macro_rules! define_config {
    ($(#[$attr:meta])* pub struct $name:ident { $($fields:tt)* }) => {
        $(#[$attr])*
        #[derive(Clone, Debug, PartialEq, Serialize)]
        pub struct $name { $($fields)* }

        #[derive(Deserialize)]
        #[serde(remote = "ForwardPassPerfModelConfig", rename = "ForwardPassPerfModelConfig", deny_unknown_fields)]
        struct ConfigFields { $($fields)* }
    };
}

define_config! {
/// Immutable model identity and selection policy for a forward-pass estimator.
///
/// This is the one public construction schema shared by Rust, Python, Replay,
/// Sweeper, and Planner. Estimator-specific tuning controls live in `estimator_config`.
pub struct ForwardPassPerfModelConfig {
    pub model: String,
    pub system: String,
    pub backend: BackendKind,
    pub worker_type: ForwardPassWorkerType,
    #[serde(default)]
    pub backend_version: Option<String>,
    /// Full independent model identity, precision and resource profile.
    #[serde(default)]
    pub fpm_profile: Option<serde_json::Map<String, serde_json::Value>>,

    /// `tp`, `pp`, `attention_dp`, `moe_tp_size` and `moe_ep_size` hold
    /// [`UNKNOWN_SIZE`] for a fact recorded as `"unknown"`.
    #[serde(default = "one", alias = "tp_size", with = "identity_size")]
    pub tp: u32,
    #[serde(default = "one", alias = "pp_size", with = "identity_size")]
    pub pp: u32,
    #[serde(default = "one", alias = "attention_dp_size", with = "identity_size")]
    pub attention_dp: u32,
    /// Recorded decode context parallelism within the TP group.
    #[serde(default)]
    pub dcp: Option<u32>,
    #[serde(default, with = "identity_optional_size")]
    pub moe_tp_size: Option<u32>,
    #[serde(default, with = "identity_optional_size")]
    pub moe_ep_size: Option<u32>,
    /// Prefill context parallelism (SGLang `--attn-cp-size`, vLLM `-pcp`): extra
    /// attention ranks that split prefill tokens, so it widens the attention side
    /// (`tp * attention_dp * cp_size == moe_tp_size * moe_ep_size`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cp_size: Option<u32>,

    #[serde(default, alias = "gemm_dtype")]
    pub gemm_quant_mode: Option<String>,
    #[serde(default, alias = "moe_dtype")]
    pub moe_quant_mode: Option<String>,
    #[serde(default, alias = "fmha_dtype")]
    pub fmha_quant_mode: Option<String>,
    /// Exact recorded FPM table label; does not override model arithmetic.
    #[serde(default, alias = "fpm_fmha_dtype")]
    pub fpm_fmha_quant_mode: Option<String>,
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
    /// Exact collected MoE compute kernel-source lane.  This is separate from
    /// `moe_backend`, which describes topology/runtime behavior.
    #[serde(default)]
    pub moe_kernel_source: Option<String>,
    #[serde(default)]
    pub moe_backend: Option<String>,
    #[serde(default)]
    pub enable_eplb: bool,
    #[serde(default)]
    pub wideep_num_slots: Option<u32>,
    #[serde(default)]
    pub enable_shared_layer: Option<bool>,
    #[serde(default)]
    pub strict_provenance: bool,
    /// Compile the model's vision tower with this layout so the estimator can
    /// price encoder calls; `None` prices the language model only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder_parallel: Option<EncoderParallel>,
    /// Recorded limits of the calibrated deployment; identity only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deployment_limits: Option<DeploymentLimits>,
}
}

impl<'de> Deserialize<'de> for ForwardPassPerfModelConfig {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if !deserializer.is_human_readable() {
            return ConfigFields::deserialize(deserializer);
        }
        // YAML's value decoder rejects duplicate keys recursively, even when
        // the input deserializer is JSON. Reject ambiguity before normalizing.
        let value = serde_yaml::Value::deserialize(deserializer)?;
        fn finite(value: &serde_yaml::Value) -> bool {
            match value {
                serde_yaml::Value::Number(n) => n.as_f64().is_some_and(f64::is_finite),
                serde_yaml::Value::Sequence(values) => values.iter().all(finite),
                serde_yaml::Value::Mapping(values) => values.values().all(finite),
                _ => true,
            }
        }
        if !finite(&value) {
            return Err(serde::de::Error::custom(
                "configuration numbers must be finite",
            ));
        }
        let mut value = serde_json::Value::deserialize(value).map_err(serde::de::Error::custom)?;
        let mode = value
            .get("estimation_mode")
            .or_else(|| value.get("forward_model"));
        if mode == Some(&serde_json::json!("fpm_learned"))
            && let Some(config) = value.as_object_mut()
        {
            let estimator = config
                .entry("estimator_config")
                .or_insert_with(|| serde_json::json!({}));
            EstimatorConfig::apply_context_defaults(estimator, EstimationMode::FpmLearned);
        }
        #[derive(Deserialize)]
        #[serde(transparent)]
        struct Fields(#[serde(with = "ConfigFields")] ForwardPassPerfModelConfig);
        serde_path_to_error::deserialize(value)
            .map(|Fields(config)| config)
            .map_err(serde::de::Error::custom)
    }
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
            fpm_profile: None,
            tp: 1,
            pp: 1,
            attention_dp: 1,
            dcp: None,
            moe_tp_size: None,
            moe_ep_size: None,
            cp_size: None,
            gemm_quant_mode: None,
            moe_quant_mode: None,
            fmha_quant_mode: None,
            fpm_fmha_quant_mode: None,
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
            moe_backend: None,
            enable_eplb: false,
            wideep_num_slots: None,
            moe_kernel_source: None,
            enable_shared_layer: None,
            strict_provenance: false,
            encoder_parallel: None,
            deployment_limits: None,
        }
    }

    /// Identity facts this configuration records as `"unknown"`, in field
    /// order. Only `fpm_regression` and `fpm_learned` accept them.
    pub fn unknown_identity_facts(&self) -> Vec<&'static str> {
        let text = |value: Option<&String>| value.is_some_and(|value| value == UNKNOWN_IDENTITY);
        let size = |value: Option<u32>| value == Some(UNKNOWN_SIZE);
        [
            ("system", text(Some(&self.system))),
            ("backend_version", text(self.backend_version.as_ref())),
            ("tp", size(Some(self.tp))),
            ("pp", size(Some(self.pp))),
            ("attention_dp", size(Some(self.attention_dp))),
            ("moe_tp_size", size(self.moe_tp_size)),
            ("moe_ep_size", size(self.moe_ep_size)),
            ("gemm_quant_mode", text(self.gemm_quant_mode.as_ref())),
            ("moe_quant_mode", text(self.moe_quant_mode.as_ref())),
            ("fmha_quant_mode", text(self.fmha_quant_mode.as_ref())),
            ("kvcache_quant_mode", text(self.kvcache_quant_mode.as_ref())),
            ("comm_quant_mode", text(self.comm_quant_mode.as_ref())),
            ("attention_backend", text(self.attention_backend.as_ref())),
        ]
        .into_iter()
        .filter_map(|(name, unknown)| unknown.then_some(name))
        .collect()
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

    /// Validate profile metadata and pin interpolation without constructing a
    /// graph or opening performance data. This also serves configuration-only
    /// consumers such as memory planning.
    pub(crate) fn resolve(self) -> Result<Self, AicError> {
        self.resolve_with_registration().map(|(config, _)| config)
    }

    /// Retain registry availability for native candidate selection without
    /// adding transient registry facts to the serialized request.
    pub(crate) fn resolve_with_registration(self) -> Result<(Self, Option<bool>), AicError> {
        // Normalization already reads snapshot members (learned artifacts and
        // checkpoints), so their I/O failures are snapshot I/O, never missing
        // performance data, exactly as during restoration.
        if self.estimator_config.state_snapshot.is_some() {
            self.normalize().map_err(super::snapshot::as_snapshot_io)
        } else {
            self.normalize()
        }
    }

    fn normalize(mut self) -> Result<(Self, Option<bool>), AicError> {
        self.estimator_config.resolve_defaults();
        // Resolve the inputs a snapshot owns against its directory before
        // anything validates or reads them.
        let manifest = match self.estimator_config.state_snapshot.as_mut() {
            Some(snapshot) => {
                snapshot.validate()?;
                let manifest = snapshot.resolve_source()?;
                super::model::relocate_snapshot_inputs(&mut self, &manifest)?;
                Some(manifest)
            }
            None => None,
        };
        if self.estimation_mode == EstimationMode::FpmLearned
            && self.estimator_config.correction.mode == super::CorrectionMode::Auto
        {
            self.estimator_config.correction.mode = super::CorrectionMode::Hybrid;
        }
        self.validate()?;
        if self.estimation_mode == EstimationMode::FpmLearned {
            self.estimator_config.fpm_learned.resolve_source()?;
            if let Some(mixed) = &mut self.estimator_config.temporary_mixed_composition {
                mixed.original_a.resolve_source()?;
            }
        }
        if let Some(speculation) = &self.speculation {
            #[cfg(feature = "python")]
            {
                self.speculation = Some(crate::py::resolve_speculation_identity(speculation)?);
            }
            #[cfg(not(feature = "python"))]
            if !matches!(speculation.kind.as_str(), "ngram" | "none") {
                return Err(invalid_config(
                    "generic speculation validation requires the python feature",
                ));
            }
        }
        if self
            .speculation
            .as_ref()
            .is_some_and(|spec| spec.kind == "none")
        {
            self.speculation = None;
        }
        let registered = if self.fpm_profile.is_some() {
            self.estimator_config
                .fpm_interpolation
                .validate_profile_options()?;
            if self.dcp.is_some() {
                return Err(invalid_config(
                    "fpm_profile does not describe recorded DCP identity",
                ));
            }
            if self.backend_version.is_none() {
                return Err(invalid_config(
                    "fpm_profile requires a literal backend_version",
                ));
            }
            if (self.nextn != 0 && self.validate_fpm_execution()?.is_none())
                || self.speculation.is_some()
            {
                return Err(invalid_config(
                    "FPM profiles support plain autoregressive decoder-only execution",
                ));
            }
            #[cfg(feature = "python")]
            {
                let facts = crate::py::validate_forward_pass_profile(&self)?;
                self.fpm_profile = Some(facts.profile);
                self.gemm_quant_mode = Some(facts.gemm_quant_mode);
                self.moe_quant_mode = Some(facts.moe_quant_mode);
                self.fmha_quant_mode = Some(facts.fmha_quant_mode);
                self.kvcache_quant_mode = Some(facts.kvcache_quant_mode);
                self.comm_quant_mode = Some(facts.comm_quant_mode);
                self.attention_backend = Some(facts.attention_backend);
                self.fpm_resources()?;
                Some(facts.registered)
            }
            #[cfg(not(feature = "python"))]
            return Err(invalid_config(
                "FPM profile validation requires the python feature",
            ));
        } else {
            None
        };
        self.estimator_config.fpm_interpolation.method = self
            .estimator_config
            .fpm_interpolation
            .method
            .resolve(registered)?;
        if self.estimator_config.fpm_interpolation.method == super::FpmInterpolationMethod::Direct
            && self.database_mode != DatabaseMode::Silicon
        {
            return Err(invalid_config(
                "direct FPM interpolation requires database_mode='SILICON'; it has no analytical SOL model",
            ));
        }
        self.validate()?;
        if let Some(manifest) = manifest {
            super::model::verify_snapshot_identity(&self, &manifest)?;
        }
        Ok((self, registered))
    }

    pub(crate) fn validate(&self) -> Result<(), AicError> {
        crate::perfmodel::execution_profile::validate_config(self)?;
        if self.estimator_config.state_snapshot.is_some()
            && (self.estimation_mode == EstimationMode::Auto
                || self.fallback_policy != ForwardPassFallbackPolicy::Deny)
        {
            // A snapshot pins one saved model. Neither automatic search nor
            // fallback may turn a failed restoration into a different model.
            return Err(invalid_config(
                "estimator_config.state_snapshot requires an explicit estimation_mode and fallback_policy='deny'",
            ));
        }
        let learned = &self.estimator_config.fpm_learned;
        if self.estimator_config.selector != super::SelectorConfig::default()
            && (self.estimation_mode != EstimationMode::FpmLearned
                || learned.adaptation != super::LearnedAdaptation::CausalSelector)
        {
            return Err(invalid_config(
                "non-default estimator_config.selector requires estimation_mode='fpm_learned' and adaptation='causal_selector'",
            ));
        }

        if let Some(mixed) = &self.estimator_config.temporary_mixed_composition {
            if self.estimator_config.state_snapshot.is_some() {
                return Err(invalid_config(
                    "temporary_mixed_composition does not support portable state_snapshot restoration; use an online checkpoint or explicit mixed state",
                ));
            }
            if mixed.quality_gate.is_some()
                && learned.adaptation != super::LearnedAdaptation::Online
            {
                return Err(invalid_config(
                    "temporary_mixed_composition.quality_gate requires standalone online learned adaptation",
                ));
            }
            if self.estimation_mode != EstimationMode::FpmLearned {
                return Err(invalid_config(
                    "temporary_mixed_composition requires explicit estimation_mode='fpm_learned'",
                ));
            }
            if self.worker_type != ForwardPassWorkerType::Aggregated {
                return Err(invalid_config(
                    "temporary_mixed_composition requires worker_type='aggregated'",
                ));
            }
            if mixed.scenario == super::TemporaryMixedCompositionScenario::A
                && learned.adaptation != super::LearnedAdaptation::Frozen
            {
                return Err(invalid_config(
                    "temporary_mixed_composition scenario='a' requires frozen learned adaptation",
                ));
            }
        }
        if self.estimation_mode == EstimationMode::FpmLearned {
            if self.fallback_policy != ForwardPassFallbackPolicy::Deny {
                return Err(invalid_config(
                    "fpm_learned requires fallback_policy='deny'",
                ));
            }
            if learned.artifact_path.is_some() == learned.checkpoint_path.is_some() {
                return Err(invalid_config(
                    "fpm_learned requires exactly one artifact_path or checkpoint_path",
                ));
            }
            if learned.adaptation == super::LearnedAdaptation::Online {
                if learned.checkpoint_path.is_none() {
                    return Err(invalid_config(
                        "online learned adaptation requires checkpoint_path",
                    ));
                }
                if self.estimator_config.correction.enabled {
                    return Err(invalid_config(
                        "online learned adaptation requires estimator_config.correction.enabled=false",
                    ));
                }
            }
            if learned.adaptation == super::LearnedAdaptation::CausalSelector {
                if learned.checkpoint_path.is_none() {
                    return Err(invalid_config(
                        "causal_selector requires a training checkpoint_path",
                    ));
                }
                if !self.estimator_config.correction.enabled
                    || !matches!(
                        self.estimator_config.correction.mode,
                        super::CorrectionMode::Auto | super::CorrectionMode::Hybrid
                    )
                {
                    return Err(invalid_config(
                        "causal_selector requires correction.enabled=true and correction.mode='hybrid' (or auto)",
                    ));
                }
            }
            if self.nextn != 0
                || self
                    .speculation
                    .as_ref()
                    .is_some_and(|speculation| speculation.kind != "none")
            {
                return Err(invalid_config(
                    "fpm_learned does not support speculative decoding",
                ));
            }
        } else if learned.artifact_path.is_some() || learned.checkpoint_path.is_some() {
            return Err(invalid_config(
                "learned sources require explicit estimation_mode='fpm_learned'",
            ));
        }
        if self.estimator_config.fpm_interpolation.collect_coverage
            && (self.estimation_mode != EstimationMode::FpmInterpolation
                || self.fallback_policy != ForwardPassFallbackPolicy::Deny
                || self.estimator_config.fpm_interpolation.method
                    != super::FpmInterpolationMethod::Direct)
        {
            return Err(invalid_config(
                "collect_coverage requires estimation_mode='fpm_interpolation', method='direct', and fallback_policy='deny'",
            ));
        }
        let unknown = self.unknown_identity_facts();
        if !unknown.is_empty() {
            // Native estimators compile and look up performance data by
            // these facts; regression and learned models only record them.
            if !matches!(
                self.estimation_mode,
                EstimationMode::FpmRegression | EstimationMode::FpmLearned
            ) {
                return Err(invalid_config(format!(
                    "{} may be \"unknown\" only with estimation_mode 'fpm_regression' or 'fpm_learned'",
                    unknown.join(", ")
                )));
            }
            if self.fallback_policy != ForwardPassFallbackPolicy::Deny {
                return Err(invalid_config(format!(
                    "unknown {} requires fallback_policy='deny': no native estimator can fall back to it",
                    unknown.join(", ")
                )));
            }
            if self.fpm_profile.is_some() {
                return Err(invalid_config(format!(
                    "an fpm_profile describes its own identity; {} cannot be \"unknown\"",
                    unknown.join(", ")
                )));
            }
        }
        if let Some((name, _)) = self.deployment_limits.as_ref().and_then(|limits| {
            limits
                .entries()
                .into_iter()
                .find(|(_, limit)| *limit == DeploymentLimit::Known(0))
        }) {
            return Err(invalid_config(format!(
                "deployment_limits.{name} must be a positive integer or \"unknown\""
            )));
        }
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
        if self
            .moe_kernel_source
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(invalid_config("moe_kernel_source cannot be empty"));
        }
        if self.tp == 0 || self.pp == 0 || self.attention_dp == 0 {
            return Err(invalid_config("tp, pp, and attention_dp must be positive"));
        }
        if self.cp_size == Some(0) {
            return Err(invalid_config("cp_size must be positive"));
        }
        // Decode CP stripes the KV inside the TP group: it must divide tp and
        // never enters the width identity below.
        if self
            .dcp
            .is_some_and(|dcp| dcp == 0 || (self.tp != UNKNOWN_SIZE && self.tp % dcp != 0))
        {
            return Err(invalid_config("dcp must be positive and divide tp"));
        }
        self.estimator_config
            .fpm_interpolation
            .validate_quant_modes(
                self.fmha_quant_mode
                    .as_deref()
                    .or(self.fpm_fmha_quant_mode.as_deref()),
                self.comm_quant_mode.as_deref(),
            )?;
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
            // Prefill CP adds attention ranks; decode CP reuses them, so only
            // `cp_size` folds into the attention width (same identity as the
            // replay-side `AicTimingConfig::validate_parallel_shape`). Profile
            // schema also admits dense TP with unused MoE dimensions equal to
            // one, and checks the exact requested deployment below.
            // An unknown size leaves the recorded widths unchecked.
            let cp = u64::from(self.cp_size.unwrap_or(1));
            if self.fpm_profile.is_none()
                && ![self.tp, self.attention_dp, moe_tp, moe_ep].contains(&UNKNOWN_SIZE)
                && u64::from(self.tp) * u64::from(self.attention_dp) * cp
                    != u64::from(moe_tp) * u64::from(moe_ep)
            {
                return Err(invalid_config(
                    "topology requires tp * attention_dp * cp_size == moe_tp_size * moe_ep_size",
                ));
            }
        }
        if self.wideep_num_slots == Some(0) {
            return Err(invalid_config("wideep_num_slots must be positive"));
        }
        if let Some(backend) = &self.moe_backend {
            if !matches!(backend.as_str(), "default" | "deepep_moe" | "megamoe") {
                return Err(invalid_config("unsupported moe_backend"));
            }
            if backend != "default" && self.backend != BackendKind::Sglang {
                return Err(invalid_config(
                    "moe_backend override requires backend=sglang",
                ));
            }
        }
        if (self.estimation_mode == EstimationMode::FpmInterpolation || self.fpm_profile.is_some())
            && (self.enable_eplb
                || self.moe_kernel_source.is_some()
                || self.wideep_num_slots.is_some()
                || self
                    .moe_backend
                    .as_deref()
                    .is_some_and(|value| value != "default"))
        {
            return Err(invalid_config(
                "fpm_interpolation does not support EPLB, slots, moe_backend or moe_kernel_source overrides",
            ));
        }
        if self.nextn > 5 {
            return Err(invalid_config("nextn must be in 0..=5"));
        }
        if let Some(speculation) = &self.speculation {
            if self.nextn != 0 {
                return Err(invalid_config("speculation cannot be combined with nextn"));
            }
            if speculation.kind.trim().is_empty() {
                return Err(invalid_config("speculation kind cannot be empty"));
            }
            if speculation.kind == "ngram" {
                speculation.ngram_depth()?;
                if self.backend != BackendKind::Vllm {
                    return Err(invalid_config("ngram speculation requires backend=vllm"));
                }
            }
            if speculation.kind == "none"
                && (!speculation.params.is_empty()
                    || speculation.draft_model_path.is_some()
                    || speculation.draft_config.is_some())
            {
                return Err(invalid_config(
                    "none speculation does not accept parameters or a draft checkpoint",
                ));
            }
        }
        let execution = self.validate_fpm_execution()?;
        if self.estimation_mode == EstimationMode::FpmInterpolation
            && self.nextn != 0
            && execution.is_none()
        {
            return Err(invalid_config(
                "estimation_mode='fpm_interpolation' does not support MTP speculative decoding",
            ));
        }
        if let Some(selector) = self.fpm_fmha_quant_mode.as_deref() {
            if self.estimation_mode != EstimationMode::FpmInterpolation {
                return Err(invalid_config(
                    "fpm_fmha_quant_mode requires estimation_mode='fpm_interpolation'",
                ));
            }
            if !matches!(selector, "bfloat16" | "fp8" | "fp8_block") {
                return Err(invalid_config(
                    "fpm_fmha_quant_mode must be bfloat16, fp8, or fp8_block",
                ));
            }
        }
        TransferPolicy::from_wire(self.transfer_policy.as_deref()).map_err(invalid_config)?;
        for root in &self.systems_paths {
            if root.to_str().is_none() {
                return Err(invalid_config("systems_paths entries must be valid UTF-8"));
            }
            // A snapshot configuration may still name the save location of
            // packed data; construction points it into the snapshot first and
            // fingerprints whatever root it reads.
            if self.estimator_config.state_snapshot.is_none() && !root.is_dir() {
                return Err(invalid_config(format!(
                    "systems_paths entry is not an existing directory: {}",
                    root.display()
                )));
            }
        }
        self.estimator_config.validate()
    }

    pub(crate) fn validate_fpm_execution(
        &self,
    ) -> Result<Option<super::FpmExecutionBinding>, AicError> {
        let Some(binding) = self.fpm_execution_binding()? else {
            return Ok(None);
        };
        if self.estimation_mode != EstimationMode::FpmInterpolation
            || self.estimator_config.fpm_interpolation.method
                != super::FpmInterpolationMethod::Direct
            || self.fallback_policy != ForwardPassFallbackPolicy::Deny
            || self.database_mode != DatabaseMode::Silicon
        {
            return Err(invalid_config(
                "speculative-round FPM requires explicit fpm_interpolation, direct method, SILICON data and fallback_policy='deny'",
            ));
        }
        if self.nextn != binding.execution_profile.num_draft_tokens || self.speculation.is_some() {
            return Err(invalid_config(
                "speculative-round FPM nextn must match num_draft_tokens and cannot use generic speculation",
            ));
        }
        if self.backend != BackendKind::Vllm
            || self.pp != 1
            || self.attention_dp != 1
            || self.cp_size.is_some_and(|cp| cp != 1)
            || self.dcp.is_some()
            || self.moe_ep_size.is_some_and(|ep| ep != 1)
            || self.moe_tp_size.is_some_and(|tp| tp != 1 && tp != self.tp)
        {
            return Err(invalid_config(
                "speculative-round FPM supports vLLM pure TP with PP1/DP1/CP1 and no recorded DCP",
            ));
        }
        Ok(Some(binding))
    }
}

pub(super) fn invalid_config(message: impl Into<String>) -> AicError {
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

    fn mixed_config(scenario: &str, mode: &str) -> ForwardPassPerfModelConfig {
        config(serde_json::json!({
            "estimation_mode": "fpm_learned",
            "estimator_config": {
                "fpm_learned": {"artifact_path": "seed.json", "adaptation": "frozen"},
                "temporary_mixed_composition": {
                    "scenario": scenario,
                    "mode": mode,
                    "original_a": {"manifest_path": "original-a.json"}
                }
            }
        }))
    }

    #[test]
    fn temporary_mixed_composition_rejects_portable_snapshot_restoration() {
        let mut cfg = mixed_config("a_c", "compose_calibrated");
        cfg.estimator_config.state_snapshot = Some(super::super::StateSnapshotConfig {
            path: "unused-snapshot".into(),
            sha256: None,
            on_tune: super::super::StateSnapshotTuning::Continue,
        });
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("does not support portable")
        );
    }

    #[test]
    fn temporary_mixed_composition_round_trip_pins_both_sources() {
        let off = serde_json::to_value(EstimatorConfig::default()).unwrap();
        assert!(off.get("temporary_mixed_composition").is_none());
        let directory = tempfile::tempdir().unwrap();
        let seed_path = directory.path().join("seed.json");
        let manifest_path = directory.path().join("original-a.json");
        // Configuration resolution binds bytes; model construction validates
        // the actual artifact and manifest schemas separately.
        std::fs::write(&seed_path, b"seed source bytes").unwrap();
        std::fs::write(&manifest_path, b"original A source bytes").unwrap();
        let mut requested = mixed_config("a", "compose");
        requested.estimator_config.fpm_learned.artifact_path = Some(seed_path);
        requested
            .estimator_config
            .temporary_mixed_composition
            .as_mut()
            .unwrap()
            .original_a
            .manifest_path = manifest_path.clone();
        let resolved = requested.resolve().unwrap();
        let learned = &resolved.estimator_config.fpm_learned;
        assert_eq!(learned.source_sha256.as_ref().unwrap().len(), 64);
        let source = &resolved
            .estimator_config
            .temporary_mixed_composition
            .as_ref()
            .unwrap()
            .original_a;
        assert!(source.manifest_path.is_absolute());
        assert_eq!(source.source_sha256.as_ref().unwrap().len(), 64);
        let reloaded: ForwardPassPerfModelConfig =
            serde_json::from_value(serde_json::to_value(&resolved).unwrap()).unwrap();
        assert_eq!(reloaded.resolve().unwrap(), resolved);

        std::fs::write(manifest_path, b"changed original A bytes").unwrap();
        assert!(
            resolved
                .resolve()
                .unwrap_err()
                .to_string()
                .contains("original_a.source_sha256")
        );
    }

    #[test]
    fn temporary_mixed_composition_requires_explicit_scenario_and_supported_scope() {
        use super::super::LearnedAdaptation;

        mixed_config("a", "compose").validate().unwrap();
        let companion = mixed_config("a_c", "compose_calibrated");
        companion.validate().unwrap();
        for adaptation in [LearnedAdaptation::Online, LearnedAdaptation::CausalSelector] {
            let mut supported = companion.clone();
            supported.estimator_config.fpm_learned.adaptation = adaptation;
            supported.estimator_config.fpm_learned.artifact_path = None;
            supported.estimator_config.fpm_learned.checkpoint_path = Some("seed-checkpoint".into());
            supported.estimator_config.correction.enabled =
                adaptation == LearnedAdaptation::CausalSelector;
            supported.validate().unwrap();

            let policy = supported
                .estimator_config
                .temporary_mixed_composition
                .as_mut()
                .unwrap();
            policy.scenario = super::super::TemporaryMixedCompositionScenario::A;
            policy.mode = super::super::TemporaryMixedCompositionMode::Compose;
            assert!(
                supported
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("frozen")
            );
        }
        for (scenario, mode) in [("a", "compose_calibrated"), ("a_c", "compose")] {
            assert!(mixed_config(scenario, mode).validate().is_err());
        }
        for estimation_mode in [
            EstimationMode::Auto,
            EstimationMode::FpmRegression,
            EstimationMode::FpmInterpolation,
        ] {
            let mut invalid = companion.clone();
            invalid.estimation_mode = estimation_mode;
            assert!(
                invalid
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("temporary_mixed_composition")
            );
        }
        for worker_type in [
            ForwardPassWorkerType::Prefill,
            ForwardPassWorkerType::Decode,
        ] {
            let mut invalid = companion.clone();
            invalid.worker_type = worker_type;
            assert!(
                invalid
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("aggregated")
            );
        }
        let mut speculative = companion;
        speculative.nextn = 1;
        assert!(speculative.validate().is_err());
    }

    #[test]
    fn temporary_mixed_composition_validates_independent_calibration_and_weights() {
        let mut cfg = mixed_config("a_c", "compose_calibrated");
        cfg.estimator_config.correction.enabled = false;
        let mixed = cfg
            .estimator_config
            .temporary_mixed_composition
            .as_mut()
            .unwrap();
        mixed.calibration.sampling.bins_per_axis = [3, 7];
        mixed.calibration.sampling.max_observations = 24;
        mixed.calibration.min_observations = 6;
        mixed.calibration.factor_bounds.min = Some(0.8);
        mixed.calibration.factor_bounds.max = Some(1.4);
        let options = mixed.calibration.options();
        assert_eq!(
            options.correction_mode,
            super::super::CorrectionMode::Hybrid
        );
        assert_eq!(options.bucket_shape, Some([3, 7]));
        assert_eq!(
            (options.max_observations, options.min_observations),
            (24, 6)
        );
        assert_eq!(options.min_faster_correction_factor, Some(0.8));
        assert_eq!(options.max_slower_correction_factor, Some(1.4));
        cfg.validate().unwrap();
        for axis in 0..3 {
            for weight in [0.0, -1.0, f64::NAN, f64::INFINITY] {
                let mut invalid = cfg.clone();
                let weights = &mut invalid.estimator_config.features;
                *match axis {
                    0 => &mut weights.attention_kv_weight,
                    1 => &mut weights.prefill_attention_pair_weight,
                    _ => &mut weights.ffn_token_weight,
                } = weight;
                assert!(
                    invalid
                        .validate()
                        .unwrap_err()
                        .to_string()
                        .contains("features.")
                );
                invalid.estimator_config.temporary_mixed_composition = None;
                // A learned-only model still ignores regression score weights.
                invalid.validate().unwrap();
            }
        }
        let mixed = cfg
            .estimator_config
            .temporary_mixed_composition
            .as_mut()
            .unwrap();
        mixed.calibration.min_observations = 25;
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("calibration")
        );
    }

    #[test]
    fn contextual_defaults_keep_nested_error_paths() {
        for (estimator, path) in [
            (
                serde_json::json!({"fpm_regression": {"fit": {"rebuild_interval": -1}}}),
                "estimator_config.fpm_regression.fit.rebuild_interval",
            ),
            (
                serde_json::json!({"fpm_learned": {"recent_window": -1}}),
                "estimator_config.fpm_learned.recent_window",
            ),
        ] {
            let value = serde_json::json!({
                "model": "model", "system": "system", "backend": "vllm",
                "worker_type": "decode", "estimator_config": estimator,
            });
            let error = serde_json::from_value::<ForwardPassPerfModelConfig>(value).unwrap_err();
            assert!(error.to_string().contains(path), "{error}");
        }
    }

    #[test]
    fn learned_defaults_preserve_overrides_and_round_trip() {
        for (adaptation, expected) in [
            (None, serde_json::json!([64, 0.1, 32, 1, false])),
            (Some("frozen"), serde_json::json!([64, 0.1, 32, 1, true])),
            (
                Some("causal_selector"),
                serde_json::json!([8, 0.25, 8, 4, true]),
            ),
        ] {
            let mut learned = serde_json::json!({"checkpoint_path": "/seed"});
            if let Some(adaptation) = adaptation {
                learned["adaptation"] = adaptation.into();
            }
            let cfg = config(serde_json::json!({
                "estimation_mode": "fpm_learned", "estimator_config": {"fpm_learned": learned}
            }));
            cfg.validate().unwrap();
            let learned = &cfg.estimator_config.fpm_learned;
            let actual = serde_json::json!([
                learned.recent_window,
                learned.history_weight,
                learned.history_weight_ramp_observations,
                learned.update_every,
                cfg.estimator_config.correction.enabled
            ]);
            assert_eq!(actual, expected);
        }
        for weight in [serde_json::json!(0.1), serde_json::Value::Null] {
            let overrides = serde_json::json!({
                "checkpoint_path": "/seed", "adaptation": "causal_selector",
                "recent_window": 64, "history_weight": weight,
                "history_weight_ramp_observations": 0, "update_every": 1
            });
            let cfg = config(serde_json::json!({
                "estimation_mode": "fpm_learned", "estimator_config": {
                    "fpm_learned": overrides,
                    "correction": {"sampling": {"max_observations": 32}},
                    "selector": {"warmup": 3, "margin": 0}
                }
            }));
            cfg.validate().unwrap();
            let saved = serde_json::to_value(&cfg).unwrap();
            for (key, value) in overrides.as_object().unwrap() {
                assert_eq!(&saved["estimator_config"]["fpm_learned"][key], value);
            }
            let estimator = &cfg.estimator_config;
            assert_eq!(estimator.correction.sampling.max_observations, 32);
            assert_eq!(estimator.selector.warmup, 3);
            assert_eq!(estimator.selector.margin, 0.0);
            let restored: ForwardPassPerfModelConfig = serde_json::from_value(saved).unwrap();
            assert_eq!(restored, cfg);
        }
        let mut typed = config(serde_json::json!({}));
        assert!(typed.estimator_config.correction.enabled);
        // Typed fields are already explicit: selecting learned mode must not
        // silently rewrite correction, even when it equals the legacy default.
        typed.estimation_mode = EstimationMode::FpmLearned;
        typed.estimator_config.fpm_learned.checkpoint_path = Some("/seed".into());
        let error = typed.validate().unwrap_err();
        assert!(error.to_string().contains("correction.enabled=false"));
        typed.estimator_config.correction.enabled = false;
        typed.validate().unwrap();
    }

    #[test]
    fn nondefault_selector_requires_causal_selection_before_loading_sources() {
        for mode in [
            "auto",
            "op_level",
            "fpm_interpolation",
            "fpm_regression",
            "fpm_learned",
        ] {
            for adaptation in ["frozen", "online", "causal_selector"] {
                if mode == "fpm_learned" && adaptation == "causal_selector" {
                    continue;
                }
                for selector in [
                    serde_json::json!({"alpha": 0.3}),
                    serde_json::json!({"warmup": 3}),
                    serde_json::json!({"margin": 0.1}),
                    serde_json::json!({"persistence": 2}),
                ] {
                    let cfg = config(serde_json::json!({
                        "estimation_mode": mode,
                        "estimator_config": {
                            "fpm_learned": {
                                "checkpoint_path": (mode == "fpm_learned")
                                    .then_some("/missing/selector-checkpoint"),
                                "adaptation": adaptation
                            },
                            "selector": selector,
                            "correction": {"enabled": adaptation != "online"}
                        }
                    }));
                    let error = crate::ForwardPassPerfModel::best_available(cfg).unwrap_err();
                    assert!(
                        error.to_string().contains(
                            "non-default estimator_config.selector requires estimation_mode='fpm_learned' and adaptation='causal_selector'"
                        ),
                        "{mode}/{adaptation}/{selector}: {error}"
                    );
                }
            }
        }
    }

    #[test]
    fn default_selector_remains_valid_for_other_estimators() {
        for mode in ["auto", "op_level", "fpm_interpolation", "fpm_regression"] {
            let cfg = config(serde_json::json!({
                "estimation_mode": mode,
                "estimator_config": {"selector": super::super::SelectorConfig::default()}
            }));
            cfg.resolve().unwrap();
        }
        for adaptation in ["frozen", "online"] {
            config(serde_json::json!({
                "estimation_mode": "fpm_learned",
                "estimator_config": {
                    "fpm_learned": {
                        "checkpoint_path": "/missing/selector-checkpoint",
                        "adaptation": adaptation
                    },
                    "selector": super::super::SelectorConfig::default(),
                    "correction": {"enabled": adaptation != "online"}
                }
            }))
            .validate()
            .unwrap();
        }
    }

    #[test]
    fn causal_selector_accepts_custom_controls_and_disabled_speculation() {
        for correction_mode in ["auto", "hybrid"] {
            let cfg = config(serde_json::json!({
                "estimation_mode": "fpm_learned",
                "speculation": {"kind": "none"},
                "estimator_config": {
                    "fpm_learned": {
                        "checkpoint_path": "/missing/selector-checkpoint",
                        "adaptation": "causal_selector"
                    },
                    "selector": {"alpha": 0.4, "warmup": 3, "margin": 0.1, "persistence": 2},
                    "correction": {"enabled": true, "mode": correction_mode}
                }
            }));
            cfg.validate().unwrap();
            let restored: ForwardPassPerfModelConfig =
                serde_json::from_value(serde_json::to_value(&cfg).unwrap()).unwrap();
            assert_eq!(restored, cfg);
            restored.validate().unwrap();
        }
    }

    #[test]
    fn learned_configuration_rejects_fallback_before_loading_source() {
        for policy in ["allow", "regression"] {
            let error = config(serde_json::json!({
                "estimation_mode": "fpm_learned",
                "fallback_policy": policy,
                "estimator_config": {"fpm_learned": {"artifact_path": "/missing/seed.json"}}
            }))
            .resolve()
            .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("fpm_learned requires fallback_policy='deny'")
            );
        }
    }

    #[test]
    fn learned_configuration_accepts_disabled_speculation_and_validates_its_parameters() {
        let mut config = config(serde_json::json!({
            "estimation_mode": "fpm_learned",
            "speculation": {"kind": "none"},
            "estimator_config": {"fpm_learned": {"artifact_path": "seed.json", "adaptation": "frozen"}}
        }));
        config.validate().unwrap();
        config
            .speculation
            .as_mut()
            .unwrap()
            .params
            .insert("unexpected".into(), true.into());
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("none speculation does not accept parameters")
        );
        config.speculation.as_mut().unwrap().params.clear();
        config.speculation.as_mut().unwrap().kind = "ngram".into();
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("fpm_learned does not support speculative decoding")
        );
    }

    #[test]
    fn config_resolution_preserves_spline_defaults_and_registration_contract() {
        use super::super::estimator::{SplineFitConfig, SplineSearchConfig};

        for (fit, expected) in [
            (
                serde_json::json!({"kind": "spline"}),
                SplineFitConfig::default(),
            ),
            (
                serde_json::json!({"kind": "spline", "spline": {
                    "search": {"kind": "periodic", "step": 17}
                }}),
                SplineFitConfig {
                    knots_per_axis: 2,
                    search: SplineSearchConfig::periodic(17),
                },
            ),
        ] {
            let cfg = config(serde_json::json!({
                "estimation_mode": "fpm_regression", "estimator_config": {"fpm_regression": {"fit": fit}}
            }));
            let normalized = cfg.clone().resolve().unwrap();
            let (with_registration, registered) = cfg.resolve_with_registration().unwrap();
            assert_eq!(normalized, with_registration);
            assert_eq!(registered, None);
            assert_eq!(
                normalized.estimator_config.fpm_regression.fit.spline,
                Some(expected)
            );
            let reloaded: ForwardPassPerfModelConfig =
                serde_json::from_str(&serde_json::to_string(&normalized).unwrap()).unwrap();
            assert_eq!(reloaded.resolve().unwrap(), normalized);
        }
        let linear = config(serde_json::json!({})).resolve().unwrap();
        assert!(linear.estimator_config.fpm_regression.fit.spline.is_none());
    }

    #[test]
    fn interpolation_controls_are_typed_and_resolved_before_candidates() {
        use super::super::FpmInterpolationMethod::{Auto, Direct, Sol};
        for (requested, registered, expected) in [
            (Auto, None, Sol),
            (Auto, Some(true), Sol),
            (Auto, Some(false), Direct),
            (Sol, None, Sol),
            (Sol, Some(true), Sol),
            (Direct, Some(true), Direct),
            (Direct, Some(false), Direct),
        ] {
            assert_eq!(requested.resolve(registered).unwrap(), expected);
        }
        assert!(Direct.resolve(None).is_err());
        assert!(Sol.resolve(Some(false)).is_err());
        let cfg = config(serde_json::json!({})).resolve().unwrap();
        assert_eq!(cfg.estimator_config.fpm_interpolation.method, Sol);
        assert_eq!(cfg.estimation_mode, EstimationMode::Auto);
        let cfg = config(serde_json::json!({
            "estimation_mode": "fpm_regression", "fallback_policy": "allow",
            "estimator_config": {"fpm_interpolation": {"method": "direct"}}
        }));
        assert!(cfg.resolve().is_err());
        assert!(
            serde_json::from_value::<EstimatorConfig>(serde_json::json!({
                "fpm_interpolation": {"method": "dierct"}
            }))
            .is_err()
        );
    }

    #[test]
    fn coverage_controls_preserve_default_serialization_and_require_strict_direct() {
        let default = serde_json::to_value(EstimatorConfig::default()).unwrap();
        assert_eq!(
            default["fpm_interpolation"],
            serde_json::json!({"method": "auto"})
        );
        let cfg = config(serde_json::json!({
            "estimation_mode": "fpm_interpolation",
            "estimator_config": {"fpm_interpolation": {"method": "direct", "collect_coverage": true}}
        }));
        cfg.validate().unwrap();
        let reloaded: ForwardPassPerfModelConfig =
            serde_json::from_value(serde_json::to_value(&cfg).unwrap()).unwrap();
        assert_eq!(cfg, reloaded);
        for mode in [
            EstimationMode::Auto,
            EstimationMode::OpLevel,
            EstimationMode::FpmRegression,
        ] {
            let mut invalid = cfg.clone();
            invalid.estimation_mode = mode;
            assert!(
                invalid
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("collect_coverage")
            );
        }
        for method in [
            super::super::FpmInterpolationMethod::Auto,
            super::super::FpmInterpolationMethod::Sol,
        ] {
            let mut invalid = cfg.clone();
            invalid.estimator_config.fpm_interpolation.method = method;
            assert!(
                invalid
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("collect_coverage")
            );
        }
        for fallback in [
            ForwardPassFallbackPolicy::Allow,
            ForwardPassFallbackPolicy::LegacyRegression,
        ] {
            let mut invalid = cfg.clone();
            invalid.fallback_policy = fallback;
            assert!(
                invalid
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("collect_coverage")
            );
        }
    }

    #[test]
    fn fpm_selector_is_validated_and_preserved_in_canonical_identity() {
        let cfg = config(serde_json::json!({
            "estimation_mode": "fpm_interpolation", "fpm_fmha_quant_mode": "fp8"
        }));
        cfg.validate().unwrap();
        let restored: ForwardPassPerfModelConfig =
            serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(restored, cfg);
        assert_eq!(restored.fpm_fmha_quant_mode.as_deref(), Some("fp8"));
        for mode in ["auto", "op_level", "fpm_regression"] {
            let invalid =
                config(serde_json::json!({"estimation_mode": mode, "fpm_fmha_quant_mode": "fp8"}));
            assert!(
                invalid
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("requires estimation_mode='fpm_interpolation'")
            );
        }
        let invalid = config(
            serde_json::json!({"estimation_mode": "fpm_interpolation", "fpm_fmha_quant_mode": "bogus"}),
        );
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn engine_controls_round_trip_with_identity_and_policy() {
        let cfg = config(serde_json::json!({
            "backend": "sglang", "enable_eplb": true, "wideep_num_slots": 256,
            "moe_backend": "deepep_moe", "attention_backend": "fa3",
            "gemm_quant_mode": "fp8", "kvcache_quant_mode": "fp8"
        }));
        cfg.validate().unwrap();
        let decoded: ForwardPassPerfModelConfig =
            serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(decoded, cfg);
        assert!(decoded.enable_eplb);
        assert_eq!(decoded.wideep_num_slots, Some(256));
        assert_eq!(decoded.moe_backend.as_deref(), Some("deepep_moe"));
        assert_eq!(decoded.fallback_policy, ForwardPassFallbackPolicy::Deny);
    }

    #[test]
    fn profile_auto_selection_does_not_drop_unsupported_engine_controls() {
        for controls in [
            serde_json::json!({"enable_eplb": true}),
            serde_json::json!({"wideep_num_slots": 256}),
            serde_json::json!({"moe_backend": "deepep_moe", "backend": "sglang"}),
        ] {
            let mut cfg = config(controls);
            cfg.fpm_profile = Some(Default::default());
            assert!(
                cfg.validate()
                    .unwrap_err()
                    .to_string()
                    .contains("does not support EPLB")
            );
        }
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
    fn moe_kernel_source_round_trips_without_pinning_the_default() {
        let default = config(serde_json::json!({}));
        assert_eq!(default.moe_kernel_source, None);

        let pinned = config(serde_json::json!({
            "moe_kernel_source": "sglang_flashinfer_trtllm_moe"
        }));
        let round_trip: ForwardPassPerfModelConfig =
            serde_json::from_str(&serde_json::to_string(&pinned).unwrap()).unwrap();
        assert_eq!(round_trip, pinned);
    }

    #[test]
    fn interpolation_rejects_exact_moe_sources_before_fallback() {
        for fallback in ["deny", "allow"] {
            for fpm_selector in [None, Some("fp8")] {
                let cfg = config(serde_json::json!({
                    "estimation_mode": "fpm_interpolation",
                    "fallback_policy": fallback,
                    "moe_kernel_source": "source_that_does_not_exist",
                    "fpm_fmha_quant_mode": fpm_selector
                }));
                let error = crate::ForwardPassPerfModel::best_available(cfg).unwrap_err();
                assert!(error.to_string().contains("moe_kernel_source overrides"));
            }
        }
        config(serde_json::json!({"estimation_mode": "fpm_interpolation"}))
            .validate()
            .unwrap();
        config(serde_json::json!({"moe_kernel_source": "sglang_flashinfer_trtllm_moe"}))
            .validate()
            .unwrap();
    }

    #[test]
    fn blank_moe_kernel_source_is_rejected_before_estimator_selection() {
        for source in ["", " \t"] {
            let error = config(serde_json::json!({"moe_kernel_source": source}))
                .validate()
                .unwrap_err();
            assert_eq!(
                error.to_string(),
                "invalid engine config: invalid forward pass perf model config: moe_kernel_source cannot be empty"
            );
        }

        for estimation_mode in [EstimationMode::Auto, EstimationMode::FpmRegression] {
            let mut cfg = config(serde_json::json!({"moe_kernel_source": " \t"}));
            cfg.estimation_mode = estimation_mode;
            let error = crate::ForwardPassPerfModel::best_available(cfg).unwrap_err();
            assert_eq!(
                error.to_string(),
                "invalid engine config: invalid forward pass perf model config: moe_kernel_source cannot be empty"
            );
        }
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
            // Estimator capability is checked during candidate selection, not
            // syntax validation, so explicit fallback can retain this identity.
            cfg.validate().unwrap();
        }
        cfg.estimation_mode = EstimationMode::Auto;
        cfg.nextn = 2;
        assert!(cfg.validate().is_err());
        cfg.nextn = 0;
        cfg.backend = BackendKind::Sglang;
        assert!(cfg.validate().is_err());
        cfg.backend = BackendKind::Vllm;
        for depth in [0, 6] {
            cfg.speculation = Some(
                serde_json::from_value(serde_json::json!({
                    "kind": "ngram", "params": {"num_speculative_tokens": depth}
                }))
                .unwrap(),
            );
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
    fn context_parallel_knobs_fold_prefill_cp_only_and_round_trip() {
        // Unset knobs stay out of the serialized identity (pre-CP configs are
        // byte-identical) and deserialize back to None.
        let plain = config(serde_json::json!({}));
        let json = serde_json::to_string(&plain).unwrap();
        assert!(!json.contains("cp_size"), "{json}");
        assert_eq!(
            serde_json::from_str::<ForwardPassPerfModelConfig>(&json).unwrap(),
            plain
        );

        // Prefill CP widens the attention side to match the MoE width ...
        let mut cfg = config(serde_json::json!({
            "moe_tp_size": 1, "moe_ep_size": 8, "cp_size": 8, "dcp": 1
        }));
        cfg.validate().unwrap();
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("\"cp_size\":8") && json.contains("\"dcp\":1"));
        assert_eq!(
            serde_json::from_str::<ForwardPassPerfModelConfig>(&json).unwrap(),
            cfg
        );
        // ... decode CP does not: without prefill CP the widths no longer match.
        cfg.cp_size = None;
        assert!(cfg.validate().is_err());
        // Zero is rejected like every other parallel size.
        cfg.cp_size = Some(8);
        cfg.dcp = Some(0);
        assert!(cfg.validate().is_err());
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
    #[test]
    fn recorded_dcp_is_validated_and_survives_round_trip() {
        let cfg =
            config(serde_json::json!({"tp": 8, "dcp": 8, "moe_tp_size": 8, "moe_ep_size": 1}));
        cfg.validate().unwrap();
        let restored: ForwardPassPerfModelConfig =
            serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(restored.dcp, Some(8));
        assert_eq!(config(serde_json::json!({})).dcp, None);
        for value in [0, 3, 16] {
            assert!(
                config(serde_json::json!({"tp": 8, "dcp": value}))
                    .validate()
                    .is_err()
            );
        }
    }

    fn unknown_identity_json() -> serde_json::Value {
        serde_json::json!({
            "model": "Qwen/Qwen3-0.6B", "system": "unknown", "backend": "vllm",
            "worker_type": "decode", "backend_version": "unknown",
            "tp": "unknown", "pp": "unknown", "attention_dp": "unknown",
            "moe_tp_size": "unknown", "moe_ep_size": "unknown",
            "gemm_quant_mode": "unknown", "moe_quant_mode": "unknown",
            "fmha_quant_mode": "unknown", "kvcache_quant_mode": "unknown",
            "comm_quant_mode": "unknown", "attention_backend": "unknown",
            "estimation_mode": "fpm_regression"
        })
    }

    #[test]
    fn unknown_identity_facts_round_trip_and_need_a_fitted_estimator() {
        let cfg: ForwardPassPerfModelConfig =
            serde_json::from_value(unknown_identity_json()).unwrap();
        assert_eq!(
            (cfg.tp, cfg.pp, cfg.attention_dp, cfg.moe_tp_size),
            (UNKNOWN_SIZE, UNKNOWN_SIZE, UNKNOWN_SIZE, Some(UNKNOWN_SIZE))
        );
        assert_eq!(
            cfg.unknown_identity_facts(),
            [
                "system",
                "backend_version",
                "tp",
                "pp",
                "attention_dp",
                "moe_tp_size",
                "moe_ep_size",
                "gemm_quant_mode",
                "moe_quant_mode",
                "fmha_quant_mode",
                "kvcache_quant_mode",
                "comm_quant_mode",
                "attention_backend"
            ]
        );
        let resolved = cfg.clone().resolve().unwrap();
        let saved = serde_json::to_value(&resolved).unwrap();
        for (field, value) in unknown_identity_json().as_object().unwrap() {
            assert_eq!(&saved[field], value, "{field}");
        }
        let reloaded: ForwardPassPerfModelConfig = serde_json::from_value(saved.clone()).unwrap();
        assert_eq!(reloaded, resolved);
        assert_eq!(
            serde_yaml::from_str::<ForwardPassPerfModelConfig>(
                &serde_yaml::to_string(&resolved).unwrap()
            )
            .unwrap(),
            resolved
        );
        // Learned models record identity like regression does.
        let mut learned = cfg.clone();
        learned.estimation_mode = EstimationMode::FpmLearned;
        learned.estimator_config.fpm_learned.artifact_path = Some("seed.json".into());
        learned.estimator_config.fpm_learned.adaptation = super::super::LearnedAdaptation::Frozen;
        learned.validate().unwrap();
        // Estimators that read performance data by these facts reject them.
        for mode in [
            EstimationMode::Auto,
            EstimationMode::OpLevel,
            EstimationMode::FpmInterpolation,
        ] {
            let mut native = cfg.clone();
            native.estimation_mode = mode;
            let error = native.validate().unwrap_err().to_string();
            assert!(
                error.contains(
                    "may be \"unknown\" only with estimation_mode 'fpm_regression' or 'fpm_learned'"
                ),
                "{error}"
            );
            let error = crate::ForwardPassPerfModel::best_available(native).unwrap_err();
            assert!(error.to_string().contains("may be \"unknown\""), "{error}");
        }
        for one in ["system", "tp", "moe_ep_size", "attention_backend"] {
            let mut value = serde_json::json!({"estimation_mode": "op_level"});
            value[one] = "unknown".into();
            if one == "moe_ep_size" {
                value["moe_tp_size"] = 1.into();
            }
            let error = config(value).validate().unwrap_err().to_string();
            assert!(error.contains(&format!("{one} may be")), "{error}");
        }
        let mut fallback = cfg.clone();
        fallback.fallback_policy = ForwardPassFallbackPolicy::Allow;
        assert!(
            fallback
                .validate()
                .unwrap_err()
                .to_string()
                .contains("requires fallback_policy='deny'")
        );
        // Unknown widths are not compared; a known DCP need not divide them.
        let mut partial = cfg;
        (partial.moe_tp_size, partial.moe_ep_size, partial.dcp) = (Some(3), Some(5), Some(2));
        partial.validate().unwrap();
    }

    #[test]
    fn unknown_sizes_have_one_spelling() {
        for (field, value) in [
            ("tp", serde_json::json!(4294967295_u64)),
            ("tp", serde_json::json!("Unknown")),
            ("tp", serde_json::json!(-1)),
            ("tp", serde_json::json!(2.0)),
            ("moe_tp_size", serde_json::json!(4294967295_u64)),
            ("dcp", serde_json::json!("unknown")),
            ("nextn", serde_json::json!("unknown")),
        ] {
            let mut value_json = unknown_identity_json();
            value_json[field] = value.clone();
            let error =
                serde_json::from_value::<ForwardPassPerfModelConfig>(value_json).unwrap_err();
            assert!(
                error.to_string().contains(field),
                "{field}={value}: {error}"
            );
        }
        // Aliases accept the literal too, and known sizes keep their numbers.
        let cfg = config(serde_json::json!({
            "estimation_mode": "fpm_regression", "tp_size": "unknown", "pp_size": 2,
            "moe_tp_size": 4294967294_u64, "moe_ep_size": 1
        }));
        assert_eq!(
            (cfg.tp, cfg.pp, cfg.moe_tp_size),
            (UNKNOWN_SIZE, 2, Some(UNKNOWN_SIZE - 1))
        );
        let saved = serde_json::to_value(&cfg).unwrap();
        assert_eq!((&saved["tp"], &saved["pp"]), (&"unknown".into(), &2.into()));
    }

    #[test]
    fn deployment_limits_are_additive_identity() {
        let plain = config(serde_json::json!({}));
        let json = serde_json::to_string(&plain).unwrap();
        assert!(!json.contains("deployment_limits"), "{json}");
        let cfg = config(serde_json::json!({
            "deployment_limits": {"num_gpus": 8, "max_kv_tokens": 524288, "max_num_seqs": "unknown"}
        }));
        let limits = cfg.deployment_limits.unwrap();
        assert_eq!(limits.num_gpus.known(), Some(8));
        assert_eq!(limits.max_kv_tokens, DeploymentLimit::Known(524288));
        // Omitted limits are unknown and serialize so.
        assert_eq!(
            (
                limits.max_num_seqs,
                limits.max_num_batched_tokens,
                limits.context_length
            ),
            (
                DeploymentLimit::Unknown,
                DeploymentLimit::Unknown,
                DeploymentLimit::Unknown
            )
        );
        let saved = serde_json::to_value(&cfg).unwrap();
        assert_eq!(
            saved["deployment_limits"],
            serde_json::json!({
                "num_gpus": 8, "max_num_batched_tokens": "unknown", "max_num_seqs": "unknown",
                "max_kv_tokens": 524288, "context_length": "unknown"
            })
        );
        let reloaded: ForwardPassPerfModelConfig = serde_json::from_value(saved).unwrap();
        assert_eq!(reloaded, cfg);
        // Identity only: every estimation mode accepts recorded limits.
        cfg.validate().unwrap();
        cfg.clone().resolve().unwrap();
        for invalid in [
            serde_json::json!({"num_gpus": 0}),
            serde_json::json!({"num_gpus": -1}),
            serde_json::json!({"num_gpus": 1.5}),
            serde_json::json!({"num_gpus": "none"}),
            serde_json::json!({"num_gpus": null}),
            serde_json::json!({"gpus": 8}),
        ] {
            let value = serde_json::json!({
                "model": "m", "system": "s", "backend": "vllm", "worker_type": "decode",
                "deployment_limits": invalid
            });
            let error = serde_json::from_value::<ForwardPassPerfModelConfig>(value).unwrap_err();
            assert!(error.to_string().contains("deployment_limits"), "{error}");
        }
        let mut zero = cfg;
        zero.deployment_limits.as_mut().unwrap().context_length = DeploymentLimit::Known(0);
        assert!(
            zero.validate()
                .unwrap_err()
                .to_string()
                .contains("deployment_limits.context_length must be a positive integer")
        );
    }

    #[test]
    fn unrecorded_profile_quantization_rejects_explicit_overrides() {
        let mut cfg = config(serde_json::json!({"estimation_mode": "fpm_interpolation",
            "estimator_config": {"fpm_interpolation": {"text_only": true, "unrecorded_quant_modes": ["fmha", "comm"]}}}));
        cfg.validate().unwrap();
        cfg.fmha_quant_mode = Some("bfloat16".into());
        assert!(cfg.validate().is_err());
    }
}
