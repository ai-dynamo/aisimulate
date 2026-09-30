// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed estimator-specific controls. Defaults preserve the existing algorithms.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::options::{ForwardPassPerfOptions, validate_options};
use crate::AicError;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EstimatorConfig {
    pub features: RegressionFeatureWeights,
    pub op_level: OpLevelConfig,
    pub fpm_interpolation: FpmInterpolationConfig,
    pub fpm_regression: FpmRegressionConfig,
    pub correction: CorrectionConfig,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpLevelConfig {
    /// Existing measured generation-MoE distribution; context is unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_workload_distribution: Option<String>,
    /// Opt-in measured graph composition; only qualified direct prefill shapes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_graph_profile: Option<String>,
    /// Resolved immutable publication identity, retained in saved configurations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_graph_profile_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FpmInterpolationConfig {
    pub method: FpmInterpolationMethod,
    /// The profile covers text prefill/decode; encoder weights remain resident.
    #[serde(skip_serializing_if = "is_false")]
    pub text_only: bool,
    /// External parquet and its same-stem metadata sidecar.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fpm_parquet_path: Option<PathBuf>,
    /// Match null profile identities only for these unspecified quant modes.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unrecorded_quant_modes: Vec<UnrecordedFpmQuantMode>,
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnrecordedFpmQuantMode {
    Fmha,
    Comm,
}

impl FpmInterpolationConfig {
    pub(crate) fn validate_profile_options(&self) -> Result<(), AicError> {
        if self.text_only || !self.unrecorded_quant_modes.is_empty() {
            return Err(super::config::invalid_config(
                "fpm_profile does not support text_only or unrecorded_quant_modes overrides",
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_quant_modes(
        &self,
        fmha: Option<&str>,
        comm: Option<&str>,
    ) -> Result<(), AicError> {
        for mode in &self.unrecorded_quant_modes {
            let explicit = match mode {
                UnrecordedFpmQuantMode::Fmha => fmha,
                UnrecordedFpmQuantMode::Comm => comm,
            };
            if explicit.is_some() {
                return Err(super::config::invalid_config(
                    "an unrecorded FPM quant mode cannot have an explicit quantization override",
                ));
            }
        }
        Ok(())
    }
}

/// Construction-time interpolation selection. Native operators receive only
/// the resolved SOL or direct method and never change it during queries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FpmInterpolationMethod {
    #[default]
    Auto,
    Sol,
    Direct,
}

impl FpmInterpolationMethod {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Sol => "sol",
            Self::Direct => "direct",
        }
    }

    pub(crate) fn resolve(self, registered: Option<bool>) -> Result<Self, AicError> {
        match (self, registered) {
            (Self::Direct, None) => Err(AicError::InvalidEngineConfig(
                "direct FPM interpolation requires an fpm_profile with identity and resource metadata".into(),
            )),
            (Self::Sol, Some(false)) => Err(AicError::InvalidEngineConfig(
                "SOL interpolation requires a registered analytical model class; use estimator_config.fpm_interpolation.method='direct' with the supplied profile".into(),
            )),
            (Self::Auto, Some(false)) | (Self::Direct, Some(_)) => Ok(Self::Direct),
            _ => Ok(Self::Sol),
        }
    }
}

/// These weights currently affect regression. Native correction keeps its
/// established workload coordinates until a replacement is accuracy-qualified.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RegressionFeatureWeights {
    #[serde(with = "super::options::regression_weight_serde")]
    pub attention_kv_weight: f64,
    #[serde(with = "super::options::regression_weight_serde")]
    pub prefill_attention_pair_weight: f64,
    #[serde(with = "super::options::regression_weight_serde")]
    pub ffn_token_weight: f64,
}

impl Default for RegressionFeatureWeights {
    fn default() -> Self {
        Self {
            attention_kv_weight: 1.0,
            prefill_attention_pair_weight: 1.0,
            ffn_token_weight: 1.0,
        }
    }
}

/// Per-store sample retention. For the legacy one-dimensional correction grid,
/// the product of the two axis counts supplies its number of bins.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SamplingConfig {
    pub bins_per_axis: [usize; 2],
    pub max_observations: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            bins_per_axis: [4, 4],
            max_observations: 64,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegressionFitKind {
    #[default]
    #[serde(alias = "linear")]
    StandardizedNnls,
    Spline,
}

/// Learned interior knots on each raw-feature axis. Coefficient fitting remains
/// nonnegative least squares; these controls only schedule knot relocation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SplineFitConfig {
    pub knots_per_axis: usize,
    pub search: SplineSearchConfig,
}

impl Default for SplineFitConfig {
    fn default() -> Self {
        Self {
            knots_per_axis: 2,
            search: SplineSearchConfig::default(),
        }
    }
}

/// Counts are per workload store and advance on accepted observations, not
/// prediction calls or insert/evict mutations. Tolerance is a fraction (0.05=5%).
/// Use [`Self::periodic`] or [`Self::adaptive`] to construct a policy. Policies
/// and their controls may grow; downstream matches must allow new variants
/// and fields.
///
/// ```compile_fail
/// use aisimulate_core::SplineSearchConfig;
/// // Construct policies through their methods, leaving room for new controls.
/// let search = SplineSearchConfig::Periodic { step: 64 };
/// ```
///
/// ```compile_fail
/// use aisimulate_core::SplineSearchConfig;
/// // A wildcard arm is required for future search policies.
/// match SplineSearchConfig::default() {
///     SplineSearchConfig::Periodic { .. } => (),
///     SplineSearchConfig::Adaptive { .. } => (),
/// }
/// ```
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SplineSearchConfig {
    #[non_exhaustive]
    Periodic {
        #[serde(default = "default_spline_step")]
        step: usize,
    },
    #[non_exhaustive]
    Adaptive {
        #[serde(default = "default_spline_window")]
        window: usize,
        #[serde(default = "default_spline_trigger")]
        trigger: usize,
        #[serde(default = "default_spline_tolerance")]
        tolerance: f64,
        #[serde(default = "default_spline_absolute_tolerance")]
        absolute_tolerance_ms: f64,
        #[serde(default = "default_spline_step")]
        cooldown: usize,
    },
}

impl SplineSearchConfig {
    /// Search at accepted-observation counts divisible by `step`, after the
    /// initial fit. The default period is 64 accepted observations per store.
    ///
    /// This only constructs configuration. `step` must be positive; the
    /// canonical [`ForwardPassPerfModel::best_available`](crate::ForwardPassPerfModel::best_available)
    /// constructor validates it before selecting an estimator.
    pub const fn periodic(step: usize) -> Self {
        Self::Periodic { step }
    }

    /// Search when `trigger` bad observations occur among the last `window`
    /// monitored observations and `cooldown` accepted observations have elapsed
    /// since the last search. Counts are per workload store. An error is bad
    /// when it exceeds both `tolerance * actual_latency` and
    /// `absolute_tolerance_ms`. `tolerance` is a fraction (0.05 means 5%); the
    /// absolute floor is in milliseconds. A search clears the monitored window.
    ///
    /// This only constructs configuration. The canonical
    /// [`ForwardPassPerfModel::best_available`](crate::ForwardPassPerfModel::best_available)
    /// constructor requires positive `window`, `trigger`, and `cooldown`,
    /// `trigger <= window`, finite positive `tolerance`, and finite nonnegative
    /// `absolute_tolerance_ms`, before selecting an estimator.
    pub const fn adaptive(
        window: usize,
        trigger: usize,
        tolerance: f64,
        absolute_tolerance_ms: f64,
        cooldown: usize,
    ) -> Self {
        Self::Adaptive {
            window,
            trigger,
            tolerance,
            absolute_tolerance_ms,
            cooldown,
        }
    }
}

const fn default_spline_step() -> usize {
    64
}
const fn default_spline_window() -> usize {
    16
}
const fn default_spline_trigger() -> usize {
    8
}
const fn default_spline_tolerance() -> f64 {
    0.05
}
const fn default_spline_absolute_tolerance() -> f64 {
    1.0
}

impl Default for SplineSearchConfig {
    fn default() -> Self {
        Self::Adaptive {
            window: default_spline_window(),
            trigger: default_spline_trigger(),
            tolerance: default_spline_tolerance(),
            absolute_tolerance_ms: default_spline_absolute_tolerance(),
            cooldown: default_spline_step(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RegressionFitConfig {
    pub kind: RegressionFitKind,
    pub singular_ridge_scale: f64,
    /// Refresh centered statistics after this many accepted insertions and
    /// actual evictions per workload store. The default `None` disables periodic
    /// refreshes; numerical recovery and batch-fit fallbacks remain enabled.
    pub rebuild_interval: Option<usize>,
    /// Present only for spline fits. The canonical constructor and normalizer
    /// resolve omitted spline controls before recording the configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spline: Option<SplineFitConfig>,
}

impl Default for RegressionFitConfig {
    fn default() -> Self {
        Self {
            kind: RegressionFitKind::StandardizedNnls,
            singular_ridge_scale: 1e-9,
            rebuild_interval: None,
            spline: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FpmRegressionConfig {
    pub sampling: SamplingConfig,
    pub min_observations: usize,
    pub fit: RegressionFitConfig,
}

impl Default for FpmRegressionConfig {
    fn default() -> Self {
        Self {
            sampling: SamplingConfig::default(),
            min_observations: 5,
            fit: RegressionFitConfig::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CorrectionFactorBounds {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl Default for CorrectionFactorBounds {
    fn default() -> Self {
        Self {
            min: Some(0.5),
            max: Some(2.0),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionFeatureSpace {
    #[default]
    LegacyWorkload,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CorrectionConfig {
    pub enabled: bool,
    pub feature_space: CorrectionFeatureSpace,
    pub sampling: SamplingConfig,
    pub min_observations: usize,
    pub factor_bounds: CorrectionFactorBounds,
    pub max_num_tokens: u32,
    pub max_batch_size: u32,
    pub max_kv_tokens: u32,
}

impl Default for CorrectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            feature_space: CorrectionFeatureSpace::LegacyWorkload,
            sampling: SamplingConfig::default(),
            min_observations: 5,
            factor_bounds: CorrectionFactorBounds::default(),
            max_num_tokens: 8192,
            max_batch_size: 512,
            max_kv_tokens: 2_000_000,
        }
    }
}

impl EstimatorConfig {
    pub(crate) fn resolve_defaults(&mut self) {
        if self.fpm_regression.fit.kind == RegressionFitKind::Spline {
            self.fpm_regression
                .fit
                .spline
                .get_or_insert_with(SplineFitConfig::default);
        }
    }

    /// Merge explicit migration inputs before applying canonical defaults.
    /// Serialized defaults from parsing one input must not become choices that
    /// conflict with a setting supplied by another input.
    #[cfg(any(feature = "python", test))]
    pub(crate) fn migrate_legacy_inputs(
        estimator_json: Option<&str>,
        options_json: Option<&str>,
        interpolation_method: Option<&str>,
        fpm_parquet_path: Option<&str>,
    ) -> Result<Self, AicError> {
        let mut explicit = serde_json::json!({});
        if let Some(json) = estimator_json {
            let supplied: serde_json::Value = serde_json::from_str(json)
                .map_err(|error| invalid_migration("estimator_config", error))?;
            let typed: Self = serde_json::from_str(json)
                .map_err(|error| invalid_migration("estimator_config", error))?;
            // Normalize scalar representations (for example 1 and 1.0) while
            // retaining only fields that were actually supplied.
            let normalized = serde_json::to_value(typed)
                .map_err(|error| invalid_migration("estimator_config", error))?;
            explicit = explicit_fields(&supplied, &normalized);
        }
        if let Some(json) = options_json {
            let supplied: serde_json::Value = serde_json::from_str(json)
                .map_err(|error| invalid_migration("legacy options", error))?;
            let options: ForwardPassPerfOptions = serde_json::from_str(json)
                .map_err(|error| invalid_migration("legacy options", error))?;
            let normalized = serde_json::to_value(&options)
                .map_err(|error| invalid_migration("legacy options", error))?;
            let mut migrated = serde_json::json!({});
            for (legacy, paths) in [
                (
                    "max_observations",
                    &[
                        "fpm_regression.sampling.max_observations",
                        "correction.sampling.max_observations",
                    ][..],
                ),
                (
                    "min_observations",
                    &[
                        "fpm_regression.min_observations",
                        "correction.min_observations",
                    ],
                ),
                (
                    "min_faster_correction_factor",
                    &["correction.factor_bounds.min"],
                ),
                (
                    "max_slower_correction_factor",
                    &["correction.factor_bounds.max"],
                ),
                ("max_num_tokens", &["correction.max_num_tokens"]),
                ("max_batch_size", &["correction.max_batch_size"]),
                ("max_kv_tokens", &["correction.max_kv_tokens"]),
                (
                    "regression_attention_kv_weight",
                    &["features.attention_kv_weight"],
                ),
                (
                    "regression_prefill_attention_pair_weight",
                    &["features.prefill_attention_pair_weight"],
                ),
                (
                    "regression_ffn_token_weight",
                    &["features.ffn_token_weight"],
                ),
                (
                    "regression_ridge_scale",
                    &["fpm_regression.fit.singular_ridge_scale"],
                ),
            ] {
                if supplied.get(legacy).is_some() {
                    for path in paths {
                        insert_explicit(&mut migrated, path, normalized[legacy].clone());
                    }
                }
            }
            if supplied.get("bucket_count").is_some() || supplied.get("bucket_shape").is_some() {
                // Bucket shape overrides count in the legacy schema. Validate
                // those controls here because count is absent from the new schema.
                validate_options(&ForwardPassPerfOptions {
                    bucket_count: options.bucket_count,
                    bucket_shape: options.bucket_shape,
                    ..ForwardPassPerfOptions::default()
                })?;
                let bins = options.bucket_shape.unwrap_or_else(|| {
                    let n = super::samples::integer_sqrt(options.bucket_count);
                    [n, n]
                });
                for path in [
                    "fpm_regression.sampling.bins_per_axis",
                    "correction.sampling.bins_per_axis",
                ] {
                    insert_explicit(&mut migrated, path, serde_json::json!(bins));
                }
            }
            merge_explicit(&mut explicit, migrated, "estimator_config")?;
        }
        let mut interpolation = serde_json::json!({});
        if let Some(method) = interpolation_method {
            insert_explicit(
                &mut interpolation,
                "fpm_interpolation.method",
                serde_json::json!(method),
            );
        }
        if let Some(path) = fpm_parquet_path {
            insert_explicit(
                &mut interpolation,
                "fpm_interpolation.fpm_parquet_path",
                serde_json::json!(path),
            );
        }
        merge_explicit(&mut explicit, interpolation, "estimator_config")?;
        let mut config: Self = serde_json::from_value(explicit)
            .map_err(|error| invalid_migration("estimator_config", error))?;
        config.resolve_defaults();
        config.validate()?;
        Ok(config)
    }

    pub(crate) fn correction_options(&self) -> ForwardPassPerfOptions {
        let config = &self.correction;
        ForwardPassPerfOptions {
            bucket_shape: Some(config.sampling.bins_per_axis),
            max_observations: config.sampling.max_observations,
            min_observations: config.min_observations,
            min_faster_correction_factor: config.factor_bounds.min,
            max_slower_correction_factor: config.factor_bounds.max,
            max_num_tokens: config.max_num_tokens,
            max_batch_size: config.max_batch_size,
            max_kv_tokens: config.max_kv_tokens,
            ..ForwardPassPerfOptions::default()
        }
    }

    pub(crate) fn regression_options(&self) -> ForwardPassPerfOptions {
        let config = &self.fpm_regression;
        ForwardPassPerfOptions {
            bucket_shape: Some(config.sampling.bins_per_axis),
            max_observations: config.sampling.max_observations,
            min_observations: config.min_observations,
            regression_attention_kv_weight: self.features.attention_kv_weight,
            regression_prefill_attention_pair_weight: self.features.prefill_attention_pair_weight,
            regression_ffn_token_weight: self.features.ffn_token_weight,
            regression_ridge_scale: config.fit.singular_ridge_scale,
            ..ForwardPassPerfOptions::default()
        }
    }

    pub(crate) fn validate(&self) -> Result<(), AicError> {
        validate_rebuild_interval(self.fpm_regression.fit.rebuild_interval)?;
        self.validate_spline()?;
        crate::config::validate_fpm_parquet_path(
            self.fpm_interpolation.fpm_parquet_path.as_deref(),
            true,
        )?;
        let ridge = self.fpm_regression.fit.singular_ridge_scale;
        if !ridge.is_finite() || ridge < 0.0 {
            return Err(AicError::InvalidEngineConfig("estimator_config.fpm_regression.fit.singular_ridge_scale must be finite and nonnegative".into()));
        }
        validate_options(&self.correction_options()).map_err(|error| {
            AicError::InvalidEngineConfig(format!("estimator_config.correction: {error}"))
        })?;
        validate_options(&self.regression_options()).map_err(|error| {
            AicError::InvalidEngineConfig(format!("estimator_config.fpm_regression: {error}"))
        })
    }

    fn validate_spline(&self) -> Result<(), AicError> {
        let regression = &self.fpm_regression;
        let invalid = |field: &str, reason: &str| {
            AicError::InvalidEngineConfig(format!(
                "estimator_config.fpm_regression.{field} {reason}"
            ))
        };
        if regression.fit.kind != RegressionFitKind::Spline {
            return if regression.fit.spline.is_some() {
                Err(invalid("fit.spline", "requires fit.kind='spline'"))
            } else {
                Ok(())
            };
        }
        if regression.sampling.max_observations < 32 {
            return Err(invalid(
                "sampling.max_observations",
                "must be at least 32 for spline fitting",
            ));
        }
        let default_spline = SplineFitConfig::default();
        let spline = regression.fit.spline.as_ref().unwrap_or(&default_spline);
        if !matches!(spline.knots_per_axis, 2 | 3) {
            return Err(invalid("fit.spline.knots_per_axis", "must be 2 or 3"));
        }
        match spline.search {
            SplineSearchConfig::Periodic { step } => {
                if step == 0 {
                    return Err(invalid("fit.spline.search.step", "must be positive"));
                }
            }
            SplineSearchConfig::Adaptive {
                window,
                trigger,
                tolerance,
                absolute_tolerance_ms,
                cooldown,
            } => {
                if window == 0 {
                    return Err(invalid("fit.spline.search.window", "must be positive"));
                }
                if trigger == 0 || trigger > window {
                    return Err(invalid(
                        "fit.spline.search.trigger",
                        "must be positive and at most window",
                    ));
                }
                if !tolerance.is_finite() || tolerance <= 0.0 {
                    return Err(invalid(
                        "fit.spline.search.tolerance",
                        "must be finite and positive",
                    ));
                }
                if !absolute_tolerance_ms.is_finite() || absolute_tolerance_ms < 0.0 {
                    return Err(invalid(
                        "fit.spline.search.absolute_tolerance_ms",
                        "must be finite and nonnegative",
                    ));
                }
                if cooldown == 0 {
                    return Err(invalid("fit.spline.search.cooldown", "must be positive"));
                }
            }
        }
        Ok(())
    }

    /// Convert legacy flat tuning options without changing their behavior.
    pub fn from_legacy(options: ForwardPassPerfOptions) -> Result<Self, AicError> {
        validate_options(&options)?;
        let bins = options.bucket_shape.unwrap_or_else(|| {
            let n = super::samples::integer_sqrt(options.bucket_count);
            [n, n]
        });
        let sampling = SamplingConfig {
            bins_per_axis: bins,
            max_observations: options.max_observations,
        };
        Ok(Self {
            features: RegressionFeatureWeights {
                attention_kv_weight: options.regression_attention_kv_weight,
                prefill_attention_pair_weight: options.regression_prefill_attention_pair_weight,
                ffn_token_weight: options.regression_ffn_token_weight,
            },
            fpm_regression: FpmRegressionConfig {
                sampling: sampling.clone(),
                min_observations: options.min_observations,
                fit: RegressionFitConfig {
                    singular_ridge_scale: options.regression_ridge_scale,
                    ..RegressionFitConfig::default()
                },
            },
            correction: CorrectionConfig {
                sampling,
                min_observations: options.min_observations,
                factor_bounds: CorrectionFactorBounds {
                    min: options.min_faster_correction_factor,
                    max: options.max_slower_correction_factor,
                },
                max_num_tokens: options.max_num_tokens,
                max_batch_size: options.max_batch_size,
                max_kv_tokens: options.max_kv_tokens,
                ..CorrectionConfig::default()
            },
            ..Self::default()
        })
    }
}

#[cfg(any(feature = "python", test))]
fn invalid_migration(field: &str, error: serde_json::Error) -> AicError {
    AicError::InvalidEngineConfig(format!("invalid {field}: {error}"))
}

#[cfg(any(feature = "python", test))]
fn explicit_fields(
    supplied: &serde_json::Value,
    normalized: &serde_json::Value,
) -> serde_json::Value {
    match supplied {
        serde_json::Value::Object(fields) => serde_json::Value::Object(
            fields
                .iter()
                .map(|(key, value)| {
                    // Compact serde output may omit an explicit default; keep it supplied.
                    (
                        key.clone(),
                        explicit_fields(value, normalized.get(key).unwrap_or(value)),
                    )
                })
                .collect(),
        ),
        _ => normalized.clone(),
    }
}

#[cfg(any(feature = "python", test))]
fn insert_explicit(root: &mut serde_json::Value, path: &str, value: serde_json::Value) {
    if let Some((field, rest)) = path.split_once('.') {
        if root.get(field).is_none() {
            root[field] = serde_json::json!({});
        }
        insert_explicit(&mut root[field], rest, value);
    } else {
        root[path] = value;
    }
}

#[cfg(any(feature = "python", test))]
fn merge_explicit(
    target: &mut serde_json::Value,
    incoming: serde_json::Value,
    path: &str,
) -> Result<(), AicError> {
    if let (Some(target_fields), Some(incoming_fields)) =
        (target.as_object_mut(), incoming.as_object())
    {
        for (key, value) in incoming_fields {
            if let Some(existing) = target_fields.get_mut(key) {
                merge_explicit(existing, value.clone(), &format!("{path}.{key}"))?;
            } else {
                target_fields.insert(key.clone(), value.clone());
            }
        }
    } else if *target != incoming {
        if path == "estimator_config.fpm_interpolation.fpm_parquet_path"
            && target
                .as_str()
                .zip(incoming.as_str())
                .is_some_and(|(saved, supplied)| {
                    std::path::Path::new(saved) == std::path::Path::new(supplied)
                })
        {
            return Ok(());
        }
        return Err(AicError::InvalidEngineConfig(format!(
            "conflicting explicit values for {path}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod migration_tests {
    use super::{EstimatorConfig, FpmInterpolationMethod, SplineFitConfig, SplineSearchConfig};

    #[test]
    fn migration_resolves_spline_controls_after_merging_explicit_inputs() {
        for (fit, expected) in [
            (
                serde_json::json!({"kind": "spline"}),
                SplineFitConfig::default(),
            ),
            (
                serde_json::json!({"kind": "spline", "spline": {
                    "knots_per_axis": 3, "search": {"kind": "periodic", "step": 17}
                }}),
                SplineFitConfig {
                    knots_per_axis: 3,
                    search: SplineSearchConfig::periodic(17),
                },
            ),
            (
                serde_json::json!({"kind": "spline", "spline": {
                    "search": {"kind": "adaptive", "window": 9, "trigger": 3,
                        "tolerance": 0.125, "absolute_tolerance_ms": 0.25, "cooldown": 11}
                }}),
                SplineFitConfig {
                    knots_per_axis: 2,
                    search: SplineSearchConfig::adaptive(9, 3, 0.125, 0.25, 11),
                },
            ),
        ] {
            let saved = serde_json::json!({"fpm_regression": {"fit": fit}}).to_string();
            let config = EstimatorConfig::migrate_legacy_inputs(
                Some(&saved),
                Some(r#"{"max_observations":96}"#),
                None,
                None,
            )
            .unwrap();
            assert_eq!(config.fpm_regression.fit.spline, Some(expected));
            assert_eq!(config.fpm_regression.sampling.max_observations, 96);
            let encoded = serde_json::to_string(&config).unwrap();
            assert_eq!(
                EstimatorConfig::migrate_legacy_inputs(Some(&encoded), None, None, None).unwrap(),
                config,
            );
        }
        let error = EstimatorConfig::migrate_legacy_inputs(
            Some(r#"{"fpm_regression":{"fit":{"kind":"spline"}}}"#),
            Some(r#"{"max_observations":16}"#),
            None,
            None,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("sampling.max_observations must be at least 32")
        );
    }

    #[test]
    fn migration_preserves_explicit_compact_fpm_defaults() {
        let saved = r#"{"fpm_interpolation": {
            "method": "direct", "text_only": false,
            "unrecorded_quant_modes": [], "fpm_parquet_path": null
        }}"#;
        let config = EstimatorConfig::migrate_legacy_inputs(Some(saved), None, None, None).unwrap();
        assert_eq!(
            config.fpm_interpolation.method,
            FpmInterpolationMethod::Direct
        );
        assert!(!config.fpm_interpolation.text_only);
        assert!(config.fpm_interpolation.unrecorded_quant_modes.is_empty());
        assert_eq!(config.fpm_interpolation.fpm_parquet_path, None);
    }

    #[test]
    fn migration_defaults_are_applied_after_explicit_inputs_merge() {
        let saved = r#"{
            "fpm_interpolation": {"method": "direct"},
            "correction": {"enabled": false, "sampling": {"max_observations": 128}},
            "fpm_regression": {"sampling": {"max_observations": 128}}
        }"#;
        let options = r#"{"min_observations": 80, "regression_ridge_scale": 0}"#;
        let config =
            EstimatorConfig::migrate_legacy_inputs(Some(saved), Some(options), None, None).unwrap();
        assert_eq!(
            config.fpm_interpolation.method,
            FpmInterpolationMethod::Direct
        );
        assert!(!config.correction.enabled);
        assert_eq!(config.correction.min_observations, 80);
        assert_eq!(config.fpm_regression.min_observations, 80);
        assert_eq!(config.fpm_regression.fit.singular_ridge_scale, 0.0);
        assert_eq!(config.features.attention_kv_weight, 1.0);
        let reloaded: EstimatorConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(reloaded, config);
    }

    #[test]
    fn migration_accepts_equal_numeric_array_and_null_controls() {
        let saved = r#"{
            "features": {"attention_kv_weight": 1},
            "correction": {"factor_bounds": {"min": null}},
            "fpm_regression": {"sampling": {"bins_per_axis": [4, 16]}}
        }"#;
        let options = r#"{
            "regression_attention_kv_weight": 1.0,
            "min_faster_correction_factor": null,
            "bucket_shape": [4, 16]
        }"#;
        let config =
            EstimatorConfig::migrate_legacy_inputs(Some(saved), Some(options), None, None).unwrap();
        assert_eq!(config.features.attention_kv_weight, 1.0);
        assert_eq!(config.correction.factor_bounds.min, None);
        assert_eq!(config.correction.sampling.bins_per_axis, [4, 16]);
        assert_eq!(config.fpm_regression.sampling.bins_per_axis, [4, 16]);
    }

    #[test]
    fn migration_compares_parquet_paths_without_filesystem_resolution() {
        let saved = r#"{"fpm_interpolation":{"fpm_parquet_path":"/tmp/fpm.parquet"}}"#;
        for legacy in [
            "/tmp/fpm.parquet",
            "/tmp/./fpm.parquet",
            "/tmp//fpm.parquet",
        ] {
            let config =
                EstimatorConfig::migrate_legacy_inputs(Some(saved), None, None, Some(legacy))
                    .unwrap();
            assert_eq!(
                config.fpm_interpolation.fpm_parquet_path.unwrap().to_str(),
                Some("/tmp/fpm.parquet")
            );
        }
        for legacy in ["/tmp/different.parquet", "/tmp/subdir/../fpm.parquet"] {
            let error =
                EstimatorConfig::migrate_legacy_inputs(Some(saved), None, None, Some(legacy))
                    .unwrap_err();
            assert!(
                error.to_string().contains(
                    "conflicting explicit values for estimator_config.fpm_interpolation.fpm_parquet_path"
                ),
                "{error}"
            );
        }
    }

    #[test]
    fn interpolation_conflicts_include_explicit_auto_but_not_omission() {
        let config = EstimatorConfig::migrate_legacy_inputs(
            Some(r#"{"fpm_interpolation": {}}"#),
            None,
            Some("direct"),
            None,
        )
        .unwrap();
        assert_eq!(
            config.fpm_interpolation.method,
            FpmInterpolationMethod::Direct
        );
        for (canonical, legacy) in [("auto", "direct"), ("direct", "auto"), ("sol", "direct")] {
            let saved = format!(r#"{{"fpm_interpolation": {{"method": "{canonical}"}}}}"#);
            let error =
                EstimatorConfig::migrate_legacy_inputs(Some(&saved), None, Some(legacy), None)
                    .unwrap_err();
            assert!(
                error.to_string().contains(
                    "conflicting explicit values for estimator_config.fpm_interpolation.method"
                ),
                "{error}"
            );
        }
    }
}

pub(super) fn validate_rebuild_interval(interval: Option<usize>) -> Result<(), AicError> {
    if interval == Some(0) {
        return Err(AicError::InvalidEngineConfig(
            "estimator_config.fpm_regression.fit.rebuild_interval must be positive or null".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_default_and_alias_keep_existing_wire_contract() {
        let expected = serde_json::json!({
            "kind": "standardized_nnls", "singular_ridge_scale": 1e-9,
            "rebuild_interval": null,
        });
        for input in [
            "{}",
            r#"{"kind":"linear"}"#,
            r#"{"kind":"standardized_nnls"}"#,
        ] {
            let fit: RegressionFitConfig = serde_json::from_str(input).unwrap();
            assert_eq!(fit, RegressionFitConfig::default());
            assert_eq!(serde_json::to_value(fit).unwrap(), expected);
        }
    }

    #[test]
    fn spline_defaults_and_policy_specific_defaults_are_owned_by_rust() {
        let mut config = EstimatorConfig::default();
        config.fpm_regression.fit.kind = RegressionFitKind::Spline;
        config.validate().unwrap();
        config.resolve_defaults();
        let expected = SplineFitConfig {
            knots_per_axis: 2,
            search: SplineSearchConfig::Adaptive {
                window: 16,
                trigger: 8,
                tolerance: 0.05,
                absolute_tolerance_ms: 1.0,
                cooldown: 64,
            },
        };
        assert_eq!(config.fpm_regression.fit.spline, Some(expected));
        let encoded = serde_json::to_value(&config).unwrap();
        assert_eq!(
            serde_json::from_value::<EstimatorConfig>(encoded).unwrap(),
            config
        );
        let periodic: SplineSearchConfig = serde_json::from_str(r#"{"kind":"periodic"}"#).unwrap();
        assert_eq!(periodic, SplineSearchConfig::Periodic { step: 64 });
        assert_eq!(
            serde_json::from_str::<SplineSearchConfig>(r#"{"kind":"adaptive"}"#).unwrap(),
            SplineSearchConfig::default(),
        );
    }

    #[test]
    fn spline_validation_rejects_incompatible_controls_and_small_capacity() {
        let mut config = EstimatorConfig::default();
        config.fpm_regression.fit.spline = Some(SplineFitConfig::default());
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("fit.spline requires")
        );
        config.fpm_regression.fit.kind = RegressionFitKind::Spline;
        config.fpm_regression.sampling.max_observations = 31;
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("sampling.max_observations")
        );
        config.fpm_regression.sampling.max_observations = 32;
        config.validate().unwrap();
        for knots in [0, 1, 4, usize::MAX] {
            config
                .fpm_regression
                .fit
                .spline
                .as_mut()
                .unwrap()
                .knots_per_axis = knots;
            assert!(
                config
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("fit.spline.knots_per_axis")
            );
        }
        for knots in [2, 3] {
            config
                .fpm_regression
                .fit
                .spline
                .as_mut()
                .unwrap()
                .knots_per_axis = knots;
            config.validate().unwrap();
        }
    }

    #[test]
    fn spline_numeric_boundaries_are_validated_for_native_rust_callers() {
        let mut config = EstimatorConfig::default();
        config.fpm_regression.fit.kind = RegressionFitKind::Spline;
        config.resolve_defaults();
        for value in [0.0, -0.1, f64::INFINITY, f64::NAN] {
            config.fpm_regression.fit.spline.as_mut().unwrap().search =
                SplineSearchConfig::Adaptive {
                    window: 16,
                    trigger: 8,
                    tolerance: value,
                    absolute_tolerance_ms: 0.0,
                    cooldown: 1,
                };
            assert!(
                config
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("search.tolerance")
            );
        }
        for value in [-1.0, f64::INFINITY, f64::NAN] {
            config.fpm_regression.fit.spline.as_mut().unwrap().search =
                SplineSearchConfig::Adaptive {
                    window: 1,
                    trigger: 1,
                    tolerance: 0.05,
                    absolute_tolerance_ms: value,
                    cooldown: 1,
                };
            assert!(
                config
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("search.absolute_tolerance_ms")
            );
        }
        config.fpm_regression.fit.spline.as_mut().unwrap().search = SplineSearchConfig::Adaptive {
            window: 1,
            trigger: 1,
            tolerance: 0.05,
            absolute_tolerance_ms: 0.0,
            cooldown: 1,
        };
        config.validate().unwrap();
    }

    #[test]
    fn spline_search_rejects_unknown_and_other_policy_fields() {
        for input in [
            r#"{"kind":"periodic","window":16}"#,
            r#"{"kind":"adaptive","step":64}"#,
            r#"{"kind":"unknown"}"#,
            r#"{"kind":"periodic","step":true}"#,
            r#"{"kind":"adaptive","window":16.5}"#,
        ] {
            assert!(
                serde_json::from_str::<SplineSearchConfig>(input).is_err(),
                "{input}"
            );
        }
    }

    #[test]
    fn rebuild_interval_defaults_and_explicit_null_survive_serde() {
        for (json, expected) in [
            ("{}", None),
            (r#"{"rebuild_interval":7}"#, Some(7)),
            (r#"{"rebuild_interval":4096}"#, Some(4096)),
            (r#"{"rebuild_interval":null}"#, None),
        ] {
            let fit: RegressionFitConfig = serde_json::from_str(json).unwrap();
            assert_eq!(fit.rebuild_interval, expected);
            let encoded = serde_json::to_value(&fit).unwrap();
            assert_eq!(encoded["rebuild_interval"], serde_json::json!(expected));
            let decoded: RegressionFitConfig = serde_json::from_value(encoded).unwrap();
            assert_eq!(decoded, fit);
        }
    }

    #[test]
    fn rebuild_interval_is_validated_before_estimator_selection() {
        let mut config = EstimatorConfig::default();
        for interval in [Some(1), Some(4096), Some(usize::MAX), None] {
            config.fpm_regression.fit.rebuild_interval = interval;
            config.validate().unwrap();
        }
        config.fpm_regression.fit.rebuild_interval = Some(0);
        assert!(config.validate().unwrap_err().to_string().contains(
            "estimator_config.fpm_regression.fit.rebuild_interval must be positive or null"
        ));
    }

    #[test]
    fn legacy_conversion_uses_the_canonical_rebuild_default() {
        let config = EstimatorConfig::from_legacy(ForwardPassPerfOptions::default()).unwrap();
        assert_eq!(
            config.fpm_regression.fit.rebuild_interval,
            RegressionFitConfig::default().rebuild_interval
        );
    }
}
