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

/// Ordered regression coordinates. The same feature catalog is available to
/// fitting and retention, but those selections are independent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionFeatureAxis {
    #[serde(rename = "attention")]
    Attention,
    #[serde(rename = "moe")]
    Moe,
    #[serde(rename = "n")]
    Count,
    #[serde(rename = "E")]
    Extend,
    #[serde(rename = "P")]
    Past,
    #[serde(rename = "maxE")]
    MaxExtend,
    #[serde(rename = "maxP")]
    MaxPast,
    #[serde(rename = "minP")]
    MinPast,
    #[serde(rename = "P2")]
    PastSquared,
    #[serde(rename = "F")]
    AttentionPairs,
    #[serde(rename = "nE")]
    CountExtend,
    #[serde(rename = "logF")]
    LogAttentionPairs,
    #[serde(rename = "meanE")]
    MeanExtend,
    #[serde(rename = "meanP")]
    MeanPast,
    #[serde(rename = "cvE2")]
    ExtendCvSquared,
    #[serde(rename = "cvP2")]
    PastCvSquared,
    #[serde(rename = "logN")]
    LogCount,
    #[serde(rename = "n2")]
    CountSquared,
    #[serde(rename = "logP")]
    LogPast,
}

fn default_regression_axes() -> Vec<RegressionFeatureAxis> {
    vec![RegressionFeatureAxis::Attention, RegressionFeatureAxis::Moe]
}

fn is_default_regression_axes(axes: &[RegressionFeatureAxis]) -> bool {
    axes == [RegressionFeatureAxis::Attention, RegressionFeatureAxis::Moe]
}

/// Per-store regression retention. Dimension is `axes.len()`; each selected
/// feature is transformed with `ln_1p` before dynamic bucket assignment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RegressionSamplingConfig {
    /// Omitted for the legacy attention/MoE grid to preserve its wire format.
    #[serde(skip_serializing_if = "is_default_regression_axes")]
    pub axes: Vec<RegressionFeatureAxis>,
    pub bins_per_axis: Vec<usize>,
    pub max_observations: usize,
}

impl Default for RegressionSamplingConfig {
    fn default() -> Self {
        Self {
            axes: default_regression_axes(),
            bins_per_axis: vec![4, 4],
            max_observations: 64,
        }
    }
}

impl RegressionSamplingConfig {
    pub(crate) fn validate(&self) -> Result<(), AicError> {
        validate_regression_axes(&self.axes, "sampling.axes")?;
        if self.axes.len() != self.bins_per_axis.len() {
            return Err(invalid_regression_config(
                "sampling.bins_per_axis",
                "must contain exactly one bin count per sampling axis",
            ));
        }
        super::samples::validate_regression_shape(&self.bins_per_axis, self.max_observations)
    }
}

/// Coefficient publication policy. Retention and statistics still receive
/// every accepted observation. Error monitoring uses the prior raw prediction.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RegressionUpdatePolicy {
    #[default]
    Always,
    ErrorThreshold {
        relative_tolerance: f64,
        absolute_tolerance_ms: f64,
        window: usize,
        trigger: usize,
        cooldown: usize,
        startup_observations: usize,
    },
}

impl<'de> Deserialize<'de> for RegressionUpdatePolicy {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Serde's internally tagged unit variants ignore additional fields,
        // even with deny_unknown_fields. An empty struct enforces the policy
        // boundary while preserving the public unit-variant API.
        #[derive(Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
        enum Policy {
            Always {},
            ErrorThreshold {
                relative_tolerance: f64,
                absolute_tolerance_ms: f64,
                window: usize,
                trigger: usize,
                cooldown: usize,
                #[serde(default = "default_linear_startup_observations")]
                startup_observations: usize,
            },
        }
        Ok(match Policy::deserialize(deserializer)? {
            Policy::Always {} => Self::Always,
            Policy::ErrorThreshold {
                relative_tolerance,
                absolute_tolerance_ms,
                window,
                trigger,
                cooldown,
                startup_observations,
            } => Self::ErrorThreshold {
                relative_tolerance,
                absolute_tolerance_ms,
                window,
                trigger,
                cooldown,
                startup_observations,
            },
        })
    }
}

const fn default_linear_startup_observations() -> usize {
    10
}

impl RegressionUpdatePolicy {
    pub fn is_always(&self) -> bool {
        matches!(self, Self::Always)
    }
}

/// Controls specific to linear fitting. Omission of `fit.linear` keeps the
/// existing two-coordinate nonnegative fit and eager coefficient updates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LinearFitConfig {
    pub feature_axes: Vec<RegressionFeatureAxis>,
    pub non_negative: bool,
    pub update_policy: RegressionUpdatePolicy,
}

impl Default for LinearFitConfig {
    fn default() -> Self {
        Self {
            feature_axes: default_regression_axes(),
            non_negative: true,
            update_policy: RegressionUpdatePolicy::Always,
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SplineSearchConfig {
    Periodic {
        #[serde(default = "default_spline_step")]
        step: usize,
    },
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
    /// Rebuild centered statistics and reference batch coefficients after this
    /// many accepted insertions and actual evictions per workload store. Every
    /// full rebuild resets the clock and overrides lazy coefficient skipping.
    /// The default `None` disables scheduled rebuilds; numerical recovery and
    /// guarded batch-fit fallbacks remain enabled.
    pub rebuild_interval: Option<usize>,
    /// Present only for linear fits; omitted controls retain the legacy fit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub linear: Option<LinearFitConfig>,
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
            linear: None,
            spline: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FpmRegressionConfig {
    pub sampling: RegressionSamplingConfig,
    pub min_observations: usize,
    pub fit: RegressionFitConfig,
}

impl Default for FpmRegressionConfig {
    fn default() -> Self {
        Self {
            sampling: RegressionSamplingConfig::default(),
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
            // Legacy options cannot represent N-D geometry. Regression stores
            // consume the typed sampling configuration directly; this shape
            // remains only for compatibility validation and two-axis callers.
            bucket_shape: Some(
                config
                    .sampling
                    .bins_per_axis
                    .as_slice()
                    .try_into()
                    .unwrap_or([4, 4]),
            ),
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
        self.fpm_regression.sampling.validate()?;
        validate_rebuild_interval(self.fpm_regression.fit.rebuild_interval)?;
        self.validate_linear()?;
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

    fn validate_linear(&self) -> Result<(), AicError> {
        let Some(linear) = &self.fpm_regression.fit.linear else {
            return Ok(());
        };
        if self.fpm_regression.fit.kind != RegressionFitKind::StandardizedNnls {
            return Err(invalid_regression_config(
                "fit.linear",
                "requires fit.kind='standardized_nnls' (alias 'linear')",
            ));
        }
        validate_regression_axes(&linear.feature_axes, "fit.linear.feature_axes")?;
        let RegressionUpdatePolicy::ErrorThreshold {
            relative_tolerance,
            absolute_tolerance_ms,
            window,
            trigger,
            cooldown,
            startup_observations,
        } = linear.update_policy
        else {
            return Ok(());
        };
        for (field, value) in [
            ("relative_tolerance", relative_tolerance),
            ("absolute_tolerance_ms", absolute_tolerance_ms),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(invalid_regression_config(
                    &format!("fit.linear.update_policy.{field}"),
                    "must be finite and nonnegative",
                ));
            }
        }
        for (field, value) in [
            ("window", window),
            ("cooldown", cooldown),
            ("startup_observations", startup_observations),
        ] {
            if value == 0 {
                return Err(invalid_regression_config(
                    &format!("fit.linear.update_policy.{field}"),
                    "must be positive",
                ));
            }
        }
        if trigger == 0 || trigger > window {
            return Err(invalid_regression_config(
                "fit.linear.update_policy.trigger",
                "must be positive and at most window",
            ));
        }
        Ok(())
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
        if !is_default_regression_axes(&regression.sampling.axes) {
            return Err(invalid(
                "sampling.axes",
                "must be ['attention', 'moe'] for spline fitting",
            ));
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
                sampling: RegressionSamplingConfig {
                    bins_per_axis: bins.to_vec(),
                    max_observations: options.max_observations,
                    ..RegressionSamplingConfig::default()
                },
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

fn invalid_regression_config(field: &str, reason: &str) -> AicError {
    AicError::InvalidEngineConfig(format!("estimator_config.fpm_regression.{field} {reason}"))
}

fn validate_regression_axes(axes: &[RegressionFeatureAxis], field: &str) -> Result<(), AicError> {
    if !(1..=6).contains(&axes.len()) {
        return Err(invalid_regression_config(field, "must contain 1 to 6 axes"));
    }
    for (index, axis) in axes.iter().enumerate() {
        if axes[..index].contains(axis) {
            return Err(invalid_regression_config(
                field,
                "must not contain repeated axes",
            ));
        }
    }
    Ok(())
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
        assert_eq!(
            serde_json::to_value(RegressionSamplingConfig::default()).unwrap(),
            serde_json::json!({"bins_per_axis": [4, 4], "max_observations": 64}),
        );
        let migrated = EstimatorConfig::from_legacy(ForwardPassPerfOptions {
            bucket_shape: Some([2, 7]),
            max_observations: 128,
            ..ForwardPassPerfOptions::default()
        })
        .unwrap();
        assert_eq!(migrated.fpm_regression.sampling.bins_per_axis, [2, 7]);
        assert_eq!(migrated.correction.sampling.bins_per_axis, [2, 7]);
        assert_eq!(migrated.fpm_regression.sampling.max_observations, 128);
        assert!(migrated.fpm_regression.fit.linear.is_none());
    }

    #[test]
    fn independent_linear_features_and_retention_axes_round_trip() {
        let config: EstimatorConfig = serde_json::from_value(serde_json::json!({
            "fpm_regression": {
                "sampling": {"axes": ["n", "attention", "P"], "bins_per_axis": [2, 3, 4]},
                "fit": {"linear": {
                    "feature_axes": ["moe"],
                    "non_negative": false,
                    "update_policy": {
                        "kind": "error_threshold", "relative_tolerance": 0.0,
                        "absolute_tolerance_ms": 0.0, "window": 8, "trigger": 2, "cooldown": 4
                    }
                }}
            }
        }))
        .unwrap();
        config.validate().unwrap();
        let linear = config.fpm_regression.fit.linear.as_ref().unwrap();
        assert_eq!(linear.feature_axes, [RegressionFeatureAxis::Moe]);
        assert!(!linear.non_negative);
        assert!(matches!(
            linear.update_policy,
            RegressionUpdatePolicy::ErrorThreshold {
                startup_observations: 10,
                ..
            }
        ));
        assert_eq!(config.correction.sampling, SamplingConfig::default());
        assert_eq!(
            serde_json::from_value::<EstimatorConfig>(serde_json::to_value(&config).unwrap())
                .unwrap(),
            config,
        );
        let explicit_defaults: LinearFitConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(explicit_defaults, LinearFitConfig::default());
        assert!(explicit_defaults.update_policy.is_always());
    }

    #[test]
    fn linear_axes_grid_and_update_policy_reject_invalid_boundaries() {
        let default = serde_json::to_value(EstimatorConfig::default()).unwrap();
        for sampling in [
            serde_json::json!({"axes": [], "bins_per_axis": []}),
            serde_json::json!({"axes": ["n", "n"], "bins_per_axis": [2, 2]}),
            serde_json::json!({"axes": ["n"], "bins_per_axis": [2, 2]}),
            serde_json::json!({"axes": ["n"], "bins_per_axis": [0]}),
            serde_json::json!({"axes": ["n", "P"], "bins_per_axis": [usize::MAX, 2]}),
            serde_json::json!({"axes": ["n"], "bins_per_axis": [2], "max_observations": 0}),
            serde_json::json!({"axes": ["n", "E", "P", "F", "maxP", "minP", "maxE"], "bins_per_axis": [1,1,1,1,1,1,1]}),
        ] {
            let mut value = default.clone();
            value["fpm_regression"]["sampling"] = sampling;
            assert!(
                serde_json::from_value::<EstimatorConfig>(value)
                    .unwrap()
                    .validate()
                    .is_err()
            );
        }
        for axes in [
            serde_json::json!([]),
            serde_json::json!(["n", "n"]),
            serde_json::json!(["n", "E", "P", "F", "maxP", "minP", "maxE"]),
        ] {
            let mut value = default.clone();
            value["fpm_regression"]["fit"]["linear"] = serde_json::json!({"feature_axes": axes});
            assert!(
                serde_json::from_value::<EstimatorConfig>(value)
                    .unwrap()
                    .validate()
                    .is_err()
            );
        }
        for input in [
            r#"{"feature_axes":["unknown"]}"#,
            r#"{"feature_axes":["n"],"non_negative":1}"#,
            r#"{"update_policy":{"kind":"always","window":1}}"#,
            r#"{"update_policy":{"kind":"error_threshold"}}"#,
        ] {
            assert!(
                serde_json::from_str::<LinearFitConfig>(input).is_err(),
                "{input}"
            );
        }
        let valid_policy = serde_json::json!({
            "kind": "error_threshold", "relative_tolerance": 0.05,
            "absolute_tolerance_ms": 0.1, "window": 8, "trigger": 2,
            "cooldown": 4, "startup_observations": 10,
        });
        for (field, bad) in [
            ("relative_tolerance", serde_json::json!(-0.1)),
            ("absolute_tolerance_ms", serde_json::json!(-0.1)),
            ("window", serde_json::json!(0)),
            ("trigger", serde_json::json!(0)),
            ("trigger", serde_json::json!(9)),
            ("cooldown", serde_json::json!(0)),
            ("startup_observations", serde_json::json!(0)),
        ] {
            let mut value = default.clone();
            let mut policy = valid_policy.clone();
            policy[field] = bad;
            value["fpm_regression"]["fit"]["linear"] = serde_json::json!({"update_policy": policy});
            assert!(
                serde_json::from_value::<EstimatorConfig>(value)
                    .unwrap()
                    .validate()
                    .is_err(),
                "{field}"
            );
        }
        let mut config = EstimatorConfig::default();
        for (relative_tolerance, absolute_tolerance_ms) in [
            (f64::NAN, 0.0),
            (f64::INFINITY, 0.0),
            (0.0, f64::NAN),
            (0.0, f64::INFINITY),
        ] {
            config.fpm_regression.fit.linear = Some(LinearFitConfig {
                update_policy: RegressionUpdatePolicy::ErrorThreshold {
                    relative_tolerance,
                    absolute_tolerance_ms,
                    window: 1,
                    trigger: 1,
                    cooldown: 1,
                    startup_observations: 1,
                },
                ..LinearFitConfig::default()
            });
            assert!(config.validate().is_err());
        }
        config.fpm_regression.fit.linear = Some(LinearFitConfig::default());
        config.fpm_regression.fit.kind = RegressionFitKind::Spline;
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("fit.linear")
        );
        config.fpm_regression.fit.linear = None;
        config.fpm_regression.sampling.axes = vec![RegressionFeatureAxis::Count];
        config.fpm_regression.sampling.bins_per_axis = vec![4];
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("sampling.axes")
        );
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
