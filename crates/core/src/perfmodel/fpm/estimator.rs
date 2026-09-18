// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed estimator-specific controls. Defaults preserve the existing algorithms.

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
#[serde(deny_unknown_fields)]
pub struct OpLevelConfig {}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FpmInterpolationConfig {
    /// The profile covers text prefill/decode; encoder weights remain resident.
    #[serde(skip_serializing_if = "is_false")]
    pub text_only: bool,
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
    StandardizedNnls,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RegressionFitConfig {
    pub kind: RegressionFitKind,
    pub singular_ridge_scale: f64,
}

impl Default for RegressionFitConfig {
    fn default() -> Self {
        Self {
            kind: RegressionFitKind::StandardizedNnls,
            singular_ridge_scale: 1e-9,
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
