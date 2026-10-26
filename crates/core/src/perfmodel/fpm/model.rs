// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Forward-pass perf model: native AIC estimate with optional online correction
//! and regression fallback, plus readiness/diagnostics.
//!
//! The `Native` variant holds an `Arc<Engine>` and the native estimate routes
//! through [`crate::perfmodel::engine::Engine::forward_pass_time_ms`]. The online
//! correction / regression / diagnostics / readiness logic is engine-agnostic.

#[path = "combined.rs"]
mod combined;
use combined::CalibratedOnlineModel;

#[path = "model_snapshot.rs"]
mod persist;
pub use persist::{ForwardPassStateSnapshotSummary, StateSnapshotOptions};

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::perfmodel::engine::spec::EncoderImageShape;
use crate::perfmodel::engine::{Engine, PerOpValue};
use crate::{AicError, ForwardPassMetrics};

use super::config::{EstimationMode, ForwardPassFallbackPolicy, ForwardPassPerfModelConfig};
use super::correction::CorrectionBuckets;
use super::coverage::{FpmCoverageState, FpmQueryCoverage};
use super::estimator::{RegressionFitConfig, RegressionSamplingConfig, StateSnapshotTuning};
use super::learned::{IterationFeatureVector, LearnedForwardPassModel, load_learned};
use super::learned_online::{
    OnlineLearner, canonical_learned_identity, checkpoint_model_for_config,
};
use super::metrics::validate_forward_pass_metrics;
use super::options::{
    CorrectionMode, ForwardPassPerfOptions, validate_options, validate_regression_options,
};
use super::regression::{ForwardPassSplineDiagnostics, RegressionStore};
use super::samples::{AxisRange, StoreStats, WithOptions};

/// Current readiness and tuning state for a `ForwardPassPerfModel`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForwardPassPerfDiagnostics {
    /// Active prediction source. Native models become `aic_with_correction`
    /// and learned models `learned_with_correction` after at least one
    /// inferred workload kind has enough correction samples.
    pub source: ForwardPassPerfSource,
    /// Whether the active model can currently produce estimates, or why it
    /// cannot. Native and learned models are immediately ready; regression is
    /// ready when any logical store has a usable linear fit, including when
    /// spline fitting is selected. Readiness does not guarantee query coverage:
    /// another store may be cold, and a query whose linear prediction is
    /// unavailable returns `None` even if the spline component can evaluate it.
    /// See `regression_store_diagnostics` for individual readiness.
    pub readiness: ForwardPassPerfReadiness,
    /// Number of retained tuning observations. This is the total across the
    /// inferred workload kinds for Native and Learned and across all logical
    /// stores for Regression. Dynamic calibration counts distinct retained
    /// observations, not duplicated local/recent slots.
    /// Hybrid sums its independent fixed and dynamic histories, counting an
    /// observation retained by both policies once in each history.
    pub retained_observations: usize,
    /// Number of populated correction regions (Native and Learned) whose
    /// workload kind has at least `min_observations` total retained samples.
    /// Excludes the recent-global fallback. Always 0 for Regression.
    /// Hybrid sums both grids, including overlapping regions; this describes
    /// retained calibration state, not the number of regions selected by queries.
    pub correction_ready_buckets: usize,
    /// Fallback reason when `best_available` had to use regression instead of native AIC.
    pub last_warning: Option<String>,
    /// Exact immutable configuration and selected systems root used at construction.
    pub provenance: Option<ForwardPassPerfProvenance>,
}

/// Resolved construction provenance pinned by Replay, Sweeper, and Planner.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForwardPassPerfProvenance {
    /// Canonical config with the exact backend version and explicit transfer policy.
    pub config: ForwardPassPerfModelConfig,
    /// Root that supplied the native engine, or `None` for regression fallback.
    pub selected_systems_root: Option<PathBuf>,
    pub requested_estimation_mode: EstimationMode,
    pub selected_estimation_mode: EstimationMode,
    pub selection_failures: Vec<String>,
}

/// Prediction backend currently used by `ForwardPassPerfModel`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForwardPassPerfSource {
    /// Strict native AIC estimator with no correction workload kind ready yet.
    Aic,
    /// Worker-type-bound regression fallback, used without native AIC support.
    FallbackRegression,
    /// Native AIC estimator with at least one learned correction workload kind.
    AicWithCorrection,
    /// Offline-trained learned model (tree ensemble over FPM features) with no
    /// correction workload kind ready yet.
    Learned,
    /// Learned model with at least one ready online-correction workload kind.
    LearnedWithCorrection,
    /// A checkpoint-backed forest with bounded online updates.
    LearnedOnline,
    /// Per-store/query causal selection between calibrated frozen and online forests.
    LearnedCausalSelector,
}

/// Readiness state reported by `ForwardPassPerfDiagnostics`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForwardPassPerfReadiness {
    /// The model has either native AIC support or enough learned data.
    Ready,
    /// Regression fallback exists, but does not yet have enough observations.
    InsufficientData,
    /// Native AIC was unavailable and `best_available` fell back to regression.
    UnsupportedConfig,
    /// Reserved for callers that surface rejected FPM input as diagnostics.
    InvalidInput,
}

/// Engine-level worker role used by the regression fallback.
///
/// Unlike the native model's inferred rank-local workload kind, this identity
/// is fixed when the regression model is constructed and applies to every DP
/// rank and iteration observed by that model.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForwardPassWorkerType {
    /// A dedicated Prefill engine.
    Prefill,
    /// A dedicated Decode engine.
    Decode,
    /// An engine that may execute Prefill, Decode, or both.
    Aggregated,
}

/// Full-iteration workload used to select an independent regression store.
/// Idle ranks do not contribute, and any locally mixed rank takes precedence
/// over separate prefill-only and decode-only ranks.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ForwardPassRegressionWorkloadKind {
    PureDecode,
    ContainsLocallyMixed,
    CrossRankAggregated,
    PurePrefill,
}

/// Retained sample count and fit readiness for one logical regression store.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ForwardPassRegressionStoreDiagnostics {
    pub workload_kind: ForwardPassRegressionWorkloadKind,
    /// Whether this store has a usable linear fit, not merely enough samples.
    /// Spline predictions also require an available linear prediction for the
    /// query; a usable spline component alone does not make the store ready.
    pub ready: bool,
    pub retained_observations: usize,
    /// Spline component state, independent of the shared linear readiness guard.
    /// Its `ready` flag can be true while the enclosing store is not ready.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spline: Option<ForwardPassSplineDiagnostics>,
}

/// Why one logical workload store cannot estimate its iterations yet.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForwardPassStoreNotReadyReason {
    /// A regression store retains fewer than `min_observations` samples.
    InsufficientData,
    /// A regression store has enough samples but no usable linear fit, for
    /// example because its features do not vary.
    NoUsableFit,
    /// The learned forest has no store for this workload kind; its iterations
    /// return `None`.
    UntrainedForestStore,
}

/// Estimation readiness of one logical workload store of the worker role.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ForwardPassStoreReadiness {
    pub workload_kind: ForwardPassRegressionWorkloadKind,
    /// Iterations classified to this workload kind receive an estimate.
    /// Native estimators are always ready.
    pub ready: bool,
    #[serde(default)]
    pub not_ready_reason: Option<ForwardPassStoreNotReadyReason>,
    /// Observations retained by this store's estimator: regression samples or
    /// online learner rows. Native and frozen forests retain none; correction
    /// samples are reported in [`ForwardPassReadinessReport::corrections`].
    pub retained_observations: usize,
}

/// Rank-local workload that selects a correction store.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForwardPassCorrectionQueryType {
    Prefill,
    Decode,
    Mixed,
}

/// Readiness of one online correction store.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ForwardPassCorrectionReadiness {
    /// Forest store owning this calibration for the causal selector; `None`
    /// for native and frozen learned stores shared by every workload kind.
    #[serde(default)]
    pub workload_kind: Option<ForwardPassRegressionWorkloadKind>,
    pub query_type: ForwardPassCorrectionQueryType,
    /// Whether estimates apply this store's factors.
    pub enabled: bool,
    /// Whether the store has enough samples to apply learned factors; an
    /// unready store applies factor 1.0 (or the dynamic fallback).
    pub ready: bool,
    pub retained_observations: usize,
    pub ready_buckets: usize,
}

/// Per-store readiness for every estimator kind.
///
/// Callers that query only some workload kinds, such as a replay callback
/// asking for pure prefill and pure decode iterations, gate on those stores
/// rather than on the model-level summary.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ForwardPassReadinessReport {
    /// Worker role, or `None` for a native model without construction
    /// provenance.
    pub worker_type: Option<ForwardPassWorkerType>,
    /// `diagnostics().readiness == Ready`.
    pub ready: bool,
    /// One entry per logical workload store of the worker role, in
    /// pure-decode, locally-mixed, cross-rank, pure-prefill order.
    pub stores: Vec<ForwardPassStoreReadiness>,
    /// Correction stores in prefill, decode, mixed order (per forest store for
    /// the causal selector). Empty for regression.
    pub corrections: Vec<ForwardPassCorrectionReadiness>,
}

/// Fixed correction-grid limits; queries beyond them use factor 1.0 (fixed)
/// or the dynamic store (hybrid).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ForwardPassCorrectionLimits {
    pub max_num_tokens: u32,
    pub max_batch_size: u32,
    pub max_kv_tokens: u32,
}

/// Deployment topology a model was built for, from its canonical identity.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ForwardPassDeploymentTopology {
    pub worker_type: ForwardPassWorkerType,
    pub tp: u32,
    pub pp: u32,
    pub attention_dp: u32,
    #[serde(default)]
    pub dcp: Option<u32>,
    #[serde(default)]
    pub cp_size: Option<u32>,
    #[serde(default)]
    pub moe_tp_size: Option<u32>,
    #[serde(default)]
    pub moe_ep_size: Option<u32>,
    pub nextn: u32,
    /// Speculation kind, when generic speculation is configured.
    #[serde(default)]
    pub speculation: Option<String>,
    #[serde(default)]
    pub kv_block_size: Option<u32>,
    /// Present when the selected estimator applies fixed-grid correction.
    #[serde(default)]
    pub correction_limits: Option<ForwardPassCorrectionLimits>,
}

impl ForwardPassDeploymentTopology {
    /// Topology recorded in a resolved canonical configuration.
    pub fn from_config(config: &ForwardPassPerfModelConfig) -> Self {
        let correction = &config.estimator_config.correction;
        let corrected = matches!(
            config.estimation_mode,
            EstimationMode::OpLevel | EstimationMode::FpmInterpolation | EstimationMode::FpmLearned
        ) && correction.enabled
            && correction.mode != CorrectionMode::DynamicLocal;
        Self {
            worker_type: config.worker_type,
            tp: config.tp,
            pp: config.pp,
            attention_dp: config.attention_dp,
            dcp: config.dcp,
            cp_size: config.cp_size,
            moe_tp_size: config.moe_tp_size,
            moe_ep_size: config.moe_ep_size,
            nextn: config.nextn,
            speculation: config.speculation.as_ref().map(|spec| spec.kind.clone()),
            kv_block_size: config.kv_block_size,
            correction_limits: corrected.then_some(ForwardPassCorrectionLimits {
                max_num_tokens: correction.max_num_tokens,
                max_batch_size: correction.max_batch_size,
                max_kv_tokens: correction.max_kv_tokens,
            }),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RegressionStores {
    stores: Vec<(ForwardPassRegressionWorkloadKind, RegressionStore)>,
}

/// Logical workload stores a worker role can produce (shared by the
/// regression stores and the learned artifact validator).
pub(crate) fn workload_kinds_for(
    worker_type: ForwardPassWorkerType,
) -> &'static [ForwardPassRegressionWorkloadKind] {
    use ForwardPassRegressionWorkloadKind::*;
    match worker_type {
        ForwardPassWorkerType::Prefill => &[PurePrefill],
        ForwardPassWorkerType::Decode => &[PureDecode],
        ForwardPassWorkerType::Aggregated => &[
            PureDecode,
            ContainsLocallyMixed,
            CrossRankAggregated,
            PurePrefill,
        ],
    }
}

impl RegressionStores {
    fn new(
        worker_type: ForwardPassWorkerType,
        options: &ForwardPassPerfOptions,
        fit: &RegressionFitConfig,
        sampling: &RegressionSamplingConfig,
    ) -> Self {
        let kinds = workload_kinds_for(worker_type);
        Self {
            stores: kinds
                .iter()
                .map(|kind| (*kind, RegressionStore::new(options, fit, sampling)))
                .collect(),
        }
    }

    fn store(&self, workload_kind: ForwardPassRegressionWorkloadKind) -> &RegressionStore {
        &self
            .stores
            .iter()
            .find(|(kind, _)| *kind == workload_kind)
            .expect("validated regression workload must match an allocated store")
            .1
    }

    fn store_mut(
        &mut self,
        workload_kind: ForwardPassRegressionWorkloadKind,
    ) -> &mut RegressionStore {
        &mut self
            .stores
            .iter_mut()
            .find(|(kind, _)| *kind == workload_kind)
            .expect("validated regression workload must match an allocated store")
            .1
    }

    fn any_ready(&self) -> bool {
        self.stores.iter().any(|(_, store)| store.is_ready())
    }

    fn observation_count(&self) -> usize {
        self.stores
            .iter()
            .map(|(_, store)| store.observation_count())
            .sum()
    }

    fn diagnostics(&self) -> Vec<ForwardPassRegressionStoreDiagnostics> {
        self.stores
            .iter()
            .map(|(kind, store)| ForwardPassRegressionStoreDiagnostics {
                workload_kind: *kind,
                ready: store.is_ready(),
                retained_observations: store.observation_count(),
                spline: store.spline_diagnostics(),
            })
            .collect()
    }
}

/// Forward-pass-level performance model with optional online tuning.
///
/// This API intentionally stays at AIC's forward-pass abstraction. It does not
/// model TTFT, ITL, SLA, engine capacity, queueing policy, or Dynamo engine
/// limits. Callers pass FPMs for one engine iteration and receive one
/// forward-pass latency estimate in milliseconds.
///
/// Native AIC preserves its existing prefill/decode/mixed workload inference
/// from each iteration's `scheduled_requests` fields:
///
/// - prefill: scheduled prefill tokens and no scheduled decode work, using
///   `[sum_prefill_tokens]`
/// - decode: scheduled decode work and no scheduled prefill tokens, using
///   `[num_decode_requests, sum_decode_kv_tokens]`
/// - mixed/agg: both scheduled prefill and decode work, using
///   `[sum_prefill_tokens, sum_decode_kv_tokens]`
/// - empty: no scheduled prefill or decode work, estimates `0.0` and is not
///   used for tuning
///
/// Regression instead binds one immutable [`ForwardPassWorkerType`] at
/// construction. Prefill and Decode each own one two-dimensional store and
/// enforce strict role compatibility. Aggregated accepts any phase composition
/// and owns four independent stores selected from all active ranks: pure decode,
/// contains locally mixed work, cross-rank aggregated, and pure prefill.
/// All stores use `[critical attention, global FFN/MoE]` feature axes.
/// Regression retention buckets use `log1p` coordinates, but fitting and
/// prediction use standardized raw feature values.
///
/// By default, native correction grids use fixed constructor-time ranges from
/// `ForwardPassPerfOptions`: `max_num_tokens` bounds `sum_prefill_tokens`,
/// `max_batch_size` bounds `num_decode_requests`, and `max_kv_tokens` bounds
/// `sum_decode_kv_tokens`. `min_faster_correction_factor` and
/// `max_slower_correction_factor` place independent absolute bounds on learned
/// native correction factors in each direction, defaulting to `0.5` and `2.0`.
/// Callers may explicitly disable either bound. Learned models default to
/// hybrid calibration with a fixed grid inside configured limits and dynamic
/// local calibration outside them, with expanding log-space bounds, balanced local
/// retention, and a recent-global fallback. `correction_mode` can explicitly
/// select any policy for Native or Learned.
///
/// Queued request fields are accepted for FPM schema parity but ignored by this
/// forward-pass-level model. `estimate_forward_pass_time_ms` treats FPM as a
/// workload descriptor: it uses scheduled workload fields and ignores
/// `wall_time`. `tune_with_fpms` treats FPM as observed telemetry: it uses the
/// same scheduled workload fields as features and uses positive `wall_time` as
/// the observation target. For attention-DP configurations, the input for one
/// iteration is one FPM per attention-DP rank; tuning merges that list into one
/// observation using the active backend's cross-rank feature rule and the
/// maximum finite positive `wall_time`.
#[derive(Clone, Debug)]
pub struct ForwardPassPerfModel {
    mode: ForwardPassPerfMode,
    options: ForwardPassPerfOptions,
    last_warning: Option<String>,
    provenance: Option<ForwardPassPerfProvenance>,
    is_correction_enabled: bool,
    pub(super) query_coverage: Option<FpmCoverageState>,
    /// Feedback policy of a model restored from a state snapshot.
    snapshot_tuning: Option<StateSnapshotTuning>,
}

#[derive(Clone, Debug)]
enum ForwardPassPerfMode {
    Native {
        /// Compiled engine. `Arc` so `ForwardPassPerfModel` stays `Clone`
        /// (the `Engine` itself is not `Clone`); cheap clones share the loaded
        /// op lists + perf-database tree.
        engine: Arc<Engine>,
        corrections: WorkloadStores<CorrectionBuckets>,
    },
    Regression {
        worker_type: ForwardPassWorkerType,
        regression: RegressionStores,
    },
    /// Offline-trained learned model (see [`super::learned`]) used as the base
    /// estimate, with the same online-correction grid as `Native` on top.
    Learned {
        learned: LearnedForwardPassModel,
        online: Option<OnlineLearner>,
        identity_verified: bool,
        corrections: WorkloadStores<CorrectionBuckets>,
    },
    CausalSelector(Box<CalibratedOnlineModel>),
}

impl ForwardPassPerfModel {
    /// Internal: build a native model directly from an already-compiled
    /// [`Engine`]. Holds the actual native-mode logic; the public
    /// [`Self::best_available`] constructor compiles the `Engine` (crossing
    /// into Python) and calls this.
    /// Used by the `#[cfg(test)]` suite to construct a native model from a
    /// hand-built fixture `Engine` without Python.
    pub(crate) fn from_engine(engine: Arc<Engine>, options: ForwardPassPerfOptions) -> Self {
        Self {
            mode: ForwardPassPerfMode::Native {
                engine,
                corrections: WorkloadStores::with_options(&options),
            },
            options,
            last_warning: None,
            provenance: None,
            is_correction_enabled: true,
            query_coverage: None,
            snapshot_tuning: None,
        }
    }

    /// API:
    /// `ForwardPassPerfModel::from_learned(source, options) -> Result<Self, AicError>`
    ///
    /// Description: create a forward-pass model from an offline-trained learned
    /// artifact (`aic_fpm_learned_forward_perf` JSON written by the Python
    /// `aisimulate_core.sdk.fpm_learned` trainer).
    ///
    /// `source` is either a filesystem path or the artifact JSON text itself
    /// (detected by a leading `{`). The artifact binds the worker type and the
    /// ordered feature ABI. Estimation is the learned prediction multiplied by
    /// a hybrid correction factor learned through
    /// `tune_with_fpms`; iterations whose regression workload kind has no
    /// trained store return `Ok(None)`.
    pub fn from_learned(source: &str, options: ForwardPassPerfOptions) -> Result<Self, AicError> {
        validate_options(&options)?;
        let learned = load_learned(source)?;
        let mut correction_options = options.clone();
        if correction_options.correction_mode == CorrectionMode::Auto {
            correction_options.correction_mode = CorrectionMode::Hybrid;
        }
        Ok(Self {
            mode: ForwardPassPerfMode::Learned {
                learned,
                online: None,
                identity_verified: false,
                corrections: WorkloadStores::with_options(&correction_options),
            },
            options,
            last_warning: None,
            provenance: None,
            is_correction_enabled: true,
            query_coverage: None,
            snapshot_tuning: None,
        })
    }

    fn from_learned_config(config: &ForwardPassPerfModelConfig) -> Result<Self, AicError> {
        let settings = &config.estimator_config.fpm_learned;
        let mut learned = if let Some(path) = &settings.checkpoint_path {
            checkpoint_model_for_config(path, config)?
        } else {
            use sha2::{Digest, Sha256};
            let path = settings.artifact_path.as_ref().ok_or_else(|| {
                AicError::InvalidEngineConfig("learned artifact_path is missing".into())
            })?;
            let bytes = std::fs::read(path).map_err(|source| AicError::Io {
                path: path.clone(),
                source,
            })?;
            let digest = format!("{:x}", Sha256::digest(&bytes));
            if settings
                .source_sha256
                .as_ref()
                .is_some_and(|expected| !expected.eq_ignore_ascii_case(&digest))
            {
                return Err(AicError::InvalidEngineConfig(
                    "learned artifact source_sha256 changed after normalization".into(),
                ));
            }
            let text = std::str::from_utf8(&bytes).map_err(|error| {
                AicError::InvalidEngineConfig(format!("learned artifact is not UTF-8: {error}"))
            })?;
            LearnedForwardPassModel::from_json(text)?
        };
        if learned.worker_type() != config.worker_type {
            return Err(AicError::InvalidEngineConfig(
                "learned artifact worker_type does not match canonical model identity".into(),
            ));
        }
        let artifact_identity = learned.metadata().get("identity");
        if let Some(identity) = artifact_identity {
            if identity != &canonical_learned_identity(config)? {
                return Err(AicError::InvalidEngineConfig(
                    "learned artifact metadata.identity does not match resolved deployment identity".into(),
                ));
            }
        }
        let identity_verified = settings.checkpoint_path.is_some() || artifact_identity.is_some();
        let stable = (settings.adaptation == super::LearnedAdaptation::CausalSelector)
            .then(|| learned.clone());
        let mut online = if settings.adaptation != super::LearnedAdaptation::Frozen {
            Some(OnlineLearner::load(
                settings
                    .checkpoint_path
                    .as_ref()
                    .expect("validated online checkpoint"),
                &mut learned,
                settings,
                config,
            )?)
        } else {
            None
        };
        let options = config.estimator_config.correction_options();
        let mode = if let Some(stable) = stable {
            ForwardPassPerfMode::CausalSelector(Box::new(CalibratedOnlineModel::new(
                stable,
                learned,
                online.take().expect("combined owns an online learner"),
                identity_verified,
                &options,
                &config.estimator_config.selector,
            )?))
        } else {
            ForwardPassPerfMode::Learned {
                learned,
                online,
                identity_verified,
                corrections: WorkloadStores::with_options(&options),
            }
        };
        Ok(Self {
            mode,
            options,
            last_warning: None,
            provenance: None,
            is_correction_enabled: config.estimator_config.correction.enabled,
            query_coverage: None,
            snapshot_tuning: None,
        })
    }

    /// Create a regression-only forward-pass model after resolving configuration.
    ///
    /// This mode is for native-AIC-unsupported models. It returns `None` from
    /// `estimate_forward_pass_time_ms` for non-empty iterations until the
    /// selected logical store has a usable fit from at least
    /// `options.min_observations` tuning samples.
    /// Correction factor getters always return `None` in this mode.
    #[cfg(test)]
    pub(crate) fn from_regression(
        worker_type: ForwardPassWorkerType,
        options: ForwardPassPerfOptions,
        fit: &RegressionFitConfig,
    ) -> Result<Self, AicError> {
        let sampling = RegressionSamplingConfig {
            bins_per_axis: options
                .bucket_shape
                .unwrap_or_else(|| {
                    let side = super::samples::integer_sqrt(options.bucket_count);
                    [side, side]
                })
                .to_vec(),
            max_observations: options.max_observations,
            ..Default::default()
        };
        Self::from_regression_config(worker_type, options, fit, &sampling)
    }

    fn from_regression_config(
        worker_type: ForwardPassWorkerType,
        options: ForwardPassPerfOptions,
        fit: &RegressionFitConfig,
        sampling: &RegressionSamplingConfig,
    ) -> Result<Self, AicError> {
        validate_regression_options(&options)?;
        super::estimator::validate_rebuild_interval(fit.rebuild_interval)?;
        Ok(Self {
            mode: ForwardPassPerfMode::Regression {
                worker_type,
                regression: RegressionStores::new(worker_type, &options, fit, sampling),
            },
            options,
            last_warning: None,
            provenance: None,
            is_correction_enabled: true,
            query_coverage: None,
            snapshot_tuning: None,
        })
    }

    /// Construct and pin one estimator. Auto searches all modes even when
    /// fallback is denied; explicit modes use the requested fallback policy.
    /// A regression model may be constructed before it has enough observations.
    ///
    /// With `estimator_config.state_snapshot`, construction restores the saved
    /// model instead: the configuration must equal the snapshot's pinned
    /// identity, the estimator is rebuilt without fallback, and the saved
    /// tuning state is validated and installed. Any mismatch is an error.
    pub fn best_available(config: ForwardPassPerfModelConfig) -> Result<Self, AicError> {
        let (mut config, registered) = config.resolve_with_registration()?;
        if let Some(path) = config
            .estimator_config
            .fpm_interpolation
            .fpm_parquet_path
            .as_mut()
        {
            *path = std::path::absolute(&*path).map_err(|error| {
                AicError::InvalidEngineConfig(format!("cannot resolve fpm_parquet_path: {error}"))
            })?;
        }
        if config.estimator_config.state_snapshot.is_some() {
            return persist::restore(config, registered, &Self::select);
        }
        Self::select(config, registered)
    }

    /// Select one estimator for a resolved configuration.
    fn select(
        config: ForwardPassPerfModelConfig,
        registered: Option<bool>,
    ) -> Result<Self, AicError> {
        let requested_estimation_mode = config.estimation_mode;
        let mut failures = Vec::new();
        let mut last_error = None;
        for mode in config.candidate_modes() {
            if mode == EstimationMode::OpLevel && registered == Some(false) {
                let error = AicError::UnsupportedModel(format!(
                    "op-level timing requires a registered architecture for {:?}",
                    config.model,
                ));
                failures.push(format!("{mode:?}: {error}"));
                last_error = Some(error);
                continue;
            }
            // Op-level DCP is priced by the sharded attention ops and their
            // collectives; the whole-forward paths need a measured vLLM cell
            // (the regression tables carry no DCP column).
            if config.dcp.is_some_and(|dcp| dcp > 1)
                && (mode == EstimationMode::FpmRegression
                    || (mode == EstimationMode::FpmInterpolation
                        && config.backend != crate::BackendKind::Vllm))
            {
                let error = AicError::UnsupportedModel(
                    "DCP timing requires measured vLLM FPM interpolation".into(),
                );
                failures.push(format!("{mode:?}: {error}"));
                last_error = Some(error);
                continue;
            }
            if let Some(speculation) = &config.speculation {
                if mode != EstimationMode::OpLevel {
                    let error = AicError::UnsupportedModel(format!(
                        "{} speculation requires op_level timing",
                        speculation.kind
                    ));
                    failures.push(format!("{mode:?}: {error}"));
                    last_error = Some(error);
                    continue;
                }
                if config.fpm_fmha_quant_mode.is_some() {
                    let error = AicError::UnsupportedModel(
                        "op_level speculation does not support fpm_fmha_quant_mode".into(),
                    );
                    failures.push(format!("{mode:?}: {error}"));
                    last_error = Some(error);
                    continue;
                }
            }
            // Only the native engine compiles a vision tower; the whole-forward
            // paths would record the layout in provenance and never price it.
            if config.encoder_parallel.is_some() && mode != EstimationMode::OpLevel {
                let error =
                    AicError::UnsupportedModel("encoder_parallel requires op_level timing".into());
                failures.push(format!("{mode:?}: {error}"));
                last_error = Some(error);
                continue;
            }
            if mode == EstimationMode::FpmLearned {
                // Invalid source bytes and identity never trigger fallback.
                let mut model = Self::from_learned_config(&config)?;
                let mut resolved = config.clone();
                resolved.estimation_mode = mode;
                resolved.fallback_policy = ForwardPassFallbackPolicy::Deny;
                model.provenance = Some(ForwardPassPerfProvenance {
                    config: resolved,
                    selected_systems_root: None,
                    requested_estimation_mode,
                    selected_estimation_mode: mode,
                    selection_failures: failures,
                });
                return Ok(model);
            }
            if mode == EstimationMode::FpmRegression {
                let options = config.estimator_config.regression_options();
                let mut model = Self::from_regression_config(
                    config.worker_type,
                    options,
                    &config.estimator_config.fpm_regression.fit,
                    &config.estimator_config.fpm_regression.sampling,
                )?;
                let mut resolved = config.clone();
                resolved.estimation_mode = mode;
                resolved.fallback_policy = ForwardPassFallbackPolicy::Deny;
                model.last_warning = (!failures.is_empty()).then(|| failures.join("; "));
                model.provenance = Some(ForwardPassPerfProvenance {
                    config: resolved,
                    selected_systems_root: None,
                    requested_estimation_mode,
                    selected_estimation_mode: mode,
                    selection_failures: failures,
                });
                return Ok(model);
            }
            let mut candidate = config.clone();
            candidate.estimation_mode = mode;
            match build_native_candidate(&candidate) {
                Ok((engine, root)) => {
                    candidate.backend_version = Some(engine.database().version.clone());
                    candidate.database_mode = engine.database().database_mode;
                    candidate.transfer_policy =
                        Some(transfer_policy_tokens(engine.database().transfer_policy));
                    candidate.systems_paths = vec![root.clone()];
                    candidate.fallback_policy = ForwardPassFallbackPolicy::Deny;
                    let mut model = Self::from_engine(
                        Arc::new(engine),
                        candidate.estimator_config.correction_options(),
                    );
                    model.is_correction_enabled = candidate.estimator_config.correction.enabled;
                    model.query_coverage = candidate
                        .estimator_config
                        .fpm_interpolation
                        .collect_coverage
                        .then(FpmCoverageState::default);
                    model.last_warning = (!failures.is_empty()).then(|| failures.join("; "));
                    model.provenance = Some(ForwardPassPerfProvenance {
                        config: candidate,
                        selected_systems_root: Some(root),
                        requested_estimation_mode,
                        selected_estimation_mode: mode,
                        selection_failures: failures,
                    });
                    return Ok(model);
                }
                Err(err) if can_fallback_to_regression(&err) => {
                    failures.push(format!("{mode:?}: {err}"));
                    last_error = Some(err);
                }
                Err(err) => return Err(err),
            }
        }
        // Every candidate mode failed. Surface all of them: the last one is
        // usually a whole-forward gate (for instance the DCP FPM rule) that
        // would otherwise mask the op-level reason the user actually hit.
        Err(match failures.len() {
            0 => AicError::UnsupportedModel("no estimator candidates".into()),
            1 => last_error.expect("one recorded failure"),
            _ => AicError::UnsupportedModel(format!(
                "no estimator mode accepted this configuration: {}",
                failures.join("; ")
            )),
        })
    }

    /// API:
    /// `model.estimate_forward_pass_time_ms(metrics_by_rank) -> Result<Option<f64>, AicError>`
    ///
    /// Description: estimate one forward-pass iteration in milliseconds.
    ///
    /// `metrics_by_rank` must contain the FPMs for a single engine iteration,
    /// one entry per attention-DP rank. Single-rank callers pass a one-element
    /// slice. Native workload inference and role-bound regression extraction
    /// use only `scheduled_requests`; queued fields and `wall_time` are ignored
    /// for estimation.
    ///
    /// Native models return an AIC estimate immediately, multiplied by the
    /// correction factor for the matching workload region. Fixed correction factors
    /// default to `1.0` for inferred workload kinds with fewer than
    /// `min_observations` total samples, empty regions, and queries outside the
    /// configured correction-grid workload ranges in
    /// `ForwardPassPerfOptions`. Dynamic calibration uses recent global feedback
    /// for empty regions and out-of-domain queries. Regression models return `Ok(None)` until
    /// the selected logical store has a ready linear fit. Spline predictions
    /// also require an available linear prediction for the query; otherwise
    /// they return `Ok(None)` even inside retained bounds. Empty scheduled work
    /// returns `Ok(Some(0.0))`.
    ///
    /// Pure Rust over the `Engine` — no Python re-entry.
    pub fn estimate_forward_pass_time_ms(
        &self,
        metrics_by_rank: &[ForwardPassMetrics],
    ) -> Result<Option<f64>, AicError> {
        match &self.mode {
            ForwardPassPerfMode::Native {
                engine,
                corrections,
            } => {
                let Some(feature) = IterationFeatures::from_metrics(metrics_by_rank)? else {
                    return Ok(Some(0.0));
                };
                let mut coverage = self
                    .query_coverage
                    .as_ref()
                    .map(FpmCoverageState::lock)
                    .transpose()?;
                let native = engine
                    .forward_pass_time_ms_with_coverage(metrics_by_rank, coverage.as_deref_mut())?;
                let corrected = native * self.correction_factor(corrections, &feature);
                Ok(Some(corrected))
            }
            ForwardPassPerfMode::Regression {
                worker_type,
                regression,
            } => {
                let Some(feature) = RegressionIterationFeatures::from_metrics(
                    metrics_by_rank,
                    *worker_type,
                    &self.options,
                )?
                else {
                    return Ok(Some(0.0));
                };
                regression
                    .store(feature.workload_kind)
                    .predict_metrics(feature.x, metrics_by_rank)
            }
            ForwardPassPerfMode::CausalSelector(model) => model.predict(metrics_by_rank),
            ForwardPassPerfMode::Learned {
                learned,
                corrections,
                ..
            } => {
                let Some(base) = learned_base_ms(learned, metrics_by_rank)? else {
                    return Ok(None);
                };
                if base == 0.0 || !self.is_correction_enabled {
                    return Ok(Some(base));
                }
                // A non-zero base implies scheduled work, so the correction
                // features exist; fall back to no correction defensively.
                let factor = IterationFeatures::from_validated_metrics(metrics_by_rank)
                    .map(|feature| {
                        corrections
                            .store(feature.workload_kind)
                            .correction_factor_for(&feature.x)
                    })
                    .unwrap_or(1.0);
                Ok(Some(base * factor))
            }
        }
    }

    /// Estimate the same iteration with executed direct-FPM support. The
    /// native rank maximum and online correction are explicit; measurement
    /// weights describe raw lookup values only. Detailed query state is returned;
    /// bounded coverage is retained only when explicitly enabled.
    pub fn estimate_forward_pass_detailed(
        &self,
        metrics_by_rank: &[ForwardPassMetrics],
    ) -> Result<crate::ForwardPassEstimate, AicError> {
        if let ForwardPassPerfMode::Native {
            engine,
            corrections,
        } = &self.mode
        {
            let Some(feature) = IterationFeatures::from_metrics(metrics_by_rank)? else {
                return Ok(crate::ForwardPassEstimate {
                    latency_ms: Some(0.0),
                    native_latency_ms: Some(0.0),
                    correction_factor: Some(1.0),
                    max_rank: None,
                    ranks: Vec::new(),
                });
            };
            let mut coverage = self
                .query_coverage
                .as_ref()
                .map(FpmCoverageState::lock)
                .transpose()?;
            let mut estimate =
                engine.forward_pass_estimate(metrics_by_rank, true, coverage.as_deref_mut())?;
            let factor = self.correction_factor(corrections, &feature);
            estimate.correction_factor = Some(factor);
            estimate.latency_ms = estimate.native_latency_ms.map(|native| native * factor);
            Ok(estimate)
        } else {
            Ok(crate::ForwardPassEstimate {
                latency_ms: self.estimate_forward_pass_time_ms(metrics_by_rank)?,
                native_latency_ms: None,
                correction_factor: None,
                max_rank: None,
                ranks: Vec::new(),
            })
        }
    }

    /// API:
    /// `model.tune_with_fpms(iterations) -> Result<(), AicError>`
    ///
    /// Description: tune the model from observed FPM iterations.
    ///
    /// The outer slice is a list of observed iterations. Each inner slice is
    /// the per-attention-DP-rank FPM list for one iteration:
    /// `[[iter0_rank0, iter0_rank1], [iter1_rank0, iter1_rank1]]`.
    /// Single-rank callers still use one FPM per inner slice.
    ///
    /// For each non-empty iteration, this method extracts features with the
    /// active backend and uses the maximum finite positive `wall_time` across
    /// ranks as the observed latency target in milliseconds. Iterations with
    /// no scheduled work or no positive `wall_time` are ignored. Native models
    /// infer the workload kind and update the matching region's
    /// median `observed_ms / native_ms` correction factor, with each ratio
    /// bounded by `min_faster_correction_factor` and
    /// `max_slower_correction_factor` when configured. Regions are used only
    /// after their inferred workload kind has `min_observations` total samples;
    /// in fixed mode, empty regions keep factor `1.0` and observations outside
    /// configured ranges are ignored. Dynamic mode admits valid nonnegative
    /// coordinates, expands log-space bounds, and blends local medians with
    /// recent global feedback. Ratios use the uncorrected base estimate.
    /// Regression models validate compatibility with their
    /// fixed worker type and update the selected logical store's
    /// configured regression fit.
    ///
    /// Pure Rust over the `Engine` — no Python re-entry.
    pub fn tune_with_fpms(
        &mut self,
        iterations: &[Vec<ForwardPassMetrics>],
    ) -> Result<(), AicError> {
        self.require_tunable()?;
        let Self {
            mode,
            options,
            is_correction_enabled,
            query_coverage,
            ..
        } = self;
        for metrics_by_rank in iterations {
            match mode {
                ForwardPassPerfMode::Native {
                    engine,
                    corrections,
                } => {
                    if !*is_correction_enabled {
                        let _ = IterationFeatures::from_metrics(metrics_by_rank)?;
                        continue;
                    }
                    let Some(observation) = IterationObservation::from_metrics(metrics_by_rank)?
                    else {
                        continue;
                    };
                    let mut coverage = query_coverage
                        .as_ref()
                        .map(FpmCoverageState::lock)
                        .transpose()?;
                    let native = engine.forward_pass_time_ms_with_coverage(
                        metrics_by_rank,
                        coverage.as_deref_mut(),
                    )?;
                    corrections
                        .store_mut(observation.feature.workload_kind)
                        .add_observation(observation.feature.x, observation.wall_time_ms, native);
                }
                ForwardPassPerfMode::Regression {
                    worker_type,
                    regression,
                } => {
                    let Some(observation) = RegressionIterationObservation::from_metrics(
                        metrics_by_rank,
                        *worker_type,
                        options,
                    )?
                    else {
                        continue;
                    };
                    regression
                        .store_mut(observation.feature.workload_kind)
                        .add_metrics(
                            observation.feature.x,
                            observation.wall_time_ms,
                            metrics_by_rank,
                        )?;
                }
                ForwardPassPerfMode::CausalSelector(model) => model.observe(metrics_by_rank)?,
                ForwardPassPerfMode::Learned {
                    learned,
                    online,
                    corrections,
                    ..
                } => {
                    if online.is_none() && !*is_correction_enabled {
                        classify_regression_workload(metrics_by_rank, learned.worker_type())?;
                        continue;
                    }
                    let prepared = prepare_learned_iteration(learned, metrics_by_rank);
                    let (kind, vector) = match prepared {
                        Ok(Some((kind, Some(vector)))) => (kind, vector),
                        Ok(None | Some((_, None))) => continue,
                        Err(error) => {
                            if let Some(online) = online {
                                online.record_rejected_observation();
                            }
                            return Err(error);
                        }
                    };
                    if !learned.has_store(kind) {
                        continue;
                    }
                    let Some(wall_ms) = max_positive_wall_time_ms(metrics_by_rank) else {
                        if let Some(online) = online {
                            online.record_rejected_observation();
                        }
                        continue;
                    };
                    if let Some(online) = online {
                        let x = learned.selected_features(&vector);
                        online.observe(kind, x, wall_ms, learned)?;
                    } else if *is_correction_enabled {
                        let Some(base) = learned.predict_ms(kind, &vector)? else {
                            continue;
                        };
                        if let Some(feature) =
                            IterationFeatures::from_validated_metrics(metrics_by_rank)
                        {
                            corrections
                                .store_mut(feature.workload_kind)
                                .add_observation(feature.x, wall_ms, base);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// API:
    /// `model.diagnostics() -> ForwardPassPerfDiagnostics`
    ///
    /// Description: return the current backend, readiness, retained sample
    /// count, and fallback warning.
    pub fn diagnostics(&self) -> ForwardPassPerfDiagnostics {
        match &self.mode {
            ForwardPassPerfMode::Native { corrections, .. } => {
                let ready_buckets = corrections.ready_bucket_count();
                ForwardPassPerfDiagnostics {
                    source: if corrections.any_ready() {
                        ForwardPassPerfSource::AicWithCorrection
                    } else {
                        ForwardPassPerfSource::Aic
                    },
                    readiness: ForwardPassPerfReadiness::Ready,
                    retained_observations: corrections.observation_count(),
                    correction_ready_buckets: ready_buckets,
                    last_warning: self.last_warning.clone(),
                    provenance: self.provenance.clone(),
                }
            }
            ForwardPassPerfMode::Regression { regression, .. } => {
                let ready = regression.any_ready();
                ForwardPassPerfDiagnostics {
                    source: ForwardPassPerfSource::FallbackRegression,
                    readiness: if ready {
                        ForwardPassPerfReadiness::Ready
                    } else if self.last_warning.is_some() {
                        ForwardPassPerfReadiness::UnsupportedConfig
                    } else {
                        ForwardPassPerfReadiness::InsufficientData
                    },
                    retained_observations: regression.observation_count(),
                    correction_ready_buckets: 0,
                    last_warning: self.last_warning.clone(),
                    provenance: self.provenance.clone(),
                }
            }
            ForwardPassPerfMode::CausalSelector(model) => ForwardPassPerfDiagnostics {
                source: ForwardPassPerfSource::LearnedCausalSelector,
                readiness: ForwardPassPerfReadiness::Ready,
                retained_observations: model.online.retained_observations()
                    + model.correction_count(),
                correction_ready_buckets: model.ready_bucket_count(),
                last_warning: self.last_warning.clone(),
                provenance: self.provenance.clone(),
            },
            ForwardPassPerfMode::Learned {
                corrections,
                online,
                ..
            } => {
                let ready_buckets = corrections.ready_bucket_count();
                ForwardPassPerfDiagnostics {
                    source: if online.is_some() {
                        ForwardPassPerfSource::LearnedOnline
                    } else if self.is_correction_enabled && corrections.any_ready() {
                        ForwardPassPerfSource::LearnedWithCorrection
                    } else {
                        ForwardPassPerfSource::Learned
                    },
                    readiness: ForwardPassPerfReadiness::Ready,
                    retained_observations: online.as_ref().map_or_else(
                        || corrections.observation_count(),
                        OnlineLearner::retained_observations,
                    ),
                    correction_ready_buckets: ready_buckets,
                    last_warning: self.last_warning.clone(),
                    provenance: self.provenance.clone(),
                }
            }
        }
    }

    fn require_tunable(&self) -> Result<(), AicError> {
        if self.snapshot_tuning == Some(StateSnapshotTuning::Reject) {
            return Err(AicError::InvalidEngineConfig(
                "this model was restored from a state snapshot with estimator_config.state_snapshot.on_tune='reject'; \
                 reload it with on_tune='continue' to continue tuning from the saved state"
                    .into(),
            ));
        }
        Ok(())
    }

    /// API:
    /// `model.readiness_report() -> ForwardPassReadinessReport`
    ///
    /// Description: estimation readiness of every logical workload store of
    /// the worker role and readiness of every correction store, for all
    /// estimator kinds. Unlike [`Self::regression_store_diagnostics`], native
    /// and learned stores are reported too, and correction samples are listed
    /// separately from the estimator's own retained observations.
    pub fn readiness_report(&self) -> ForwardPassReadinessReport {
        let ready = self.diagnostics().readiness == ForwardPassPerfReadiness::Ready;
        let correction_entries = |kind: Option<ForwardPassRegressionWorkloadKind>,
                                  stores: &WorkloadStores<CorrectionBuckets>,
                                  enabled: bool| {
            [
                (ForwardPassCorrectionQueryType::Prefill, &stores.prefill),
                (ForwardPassCorrectionQueryType::Decode, &stores.decode),
                (ForwardPassCorrectionQueryType::Mixed, &stores.mixed),
            ]
            .map(|(query_type, store)| ForwardPassCorrectionReadiness {
                workload_kind: kind,
                query_type,
                enabled,
                ready: store.is_ready(),
                retained_observations: store.observation_count(),
                ready_buckets: store.ready_bucket_count(),
            })
        };
        let forest_stores =
            |worker_type: ForwardPassWorkerType,
             has_store: &dyn Fn(ForwardPassRegressionWorkloadKind) -> bool,
             retained: &dyn Fn(ForwardPassRegressionWorkloadKind) -> usize| {
                workload_kinds_for(worker_type)
                    .iter()
                    .map(|&workload_kind| {
                        let ready = has_store(workload_kind);
                        ForwardPassStoreReadiness {
                            workload_kind,
                            ready,
                            not_ready_reason: (!ready)
                                .then_some(ForwardPassStoreNotReadyReason::UntrainedForestStore),
                            retained_observations: retained(workload_kind),
                        }
                    })
                    .collect::<Vec<_>>()
            };
        let (worker_type, stores, corrections) = match &self.mode {
            ForwardPassPerfMode::Native { corrections, .. } => {
                let worker_type = self.provenance.as_ref().map(|p| p.config.worker_type);
                let stores = worker_type.map_or_else(Vec::new, |worker_type| {
                    workload_kinds_for(worker_type)
                        .iter()
                        .map(|&workload_kind| ForwardPassStoreReadiness {
                            workload_kind,
                            ready: true,
                            not_ready_reason: None,
                            retained_observations: 0,
                        })
                        .collect()
                });
                (
                    worker_type,
                    stores,
                    correction_entries(None, corrections, self.is_correction_enabled).to_vec(),
                )
            }
            ForwardPassPerfMode::Regression {
                worker_type,
                regression,
            } => {
                let stores = regression
                    .stores
                    .iter()
                    .map(|(workload_kind, store)| {
                        let ready = store.is_ready();
                        let retained = store.observation_count();
                        ForwardPassStoreReadiness {
                            workload_kind: *workload_kind,
                            ready,
                            not_ready_reason: (!ready).then_some(
                                if retained < self.options.min_observations {
                                    ForwardPassStoreNotReadyReason::InsufficientData
                                } else {
                                    ForwardPassStoreNotReadyReason::NoUsableFit
                                },
                            ),
                            retained_observations: retained,
                        }
                    })
                    .collect();
                (Some(*worker_type), stores, Vec::new())
            }
            ForwardPassPerfMode::Learned {
                learned,
                online,
                corrections,
                ..
            } => (
                Some(learned.worker_type()),
                forest_stores(
                    learned.worker_type(),
                    &|kind| learned.has_store(kind),
                    &|kind| {
                        online
                            .as_ref()
                            .map_or(0, |online| online.retained_observations_for(kind))
                    },
                ),
                correction_entries(None, corrections, self.is_correction_enabled).to_vec(),
            ),
            ForwardPassPerfMode::CausalSelector(model) => (
                Some(model.stable.worker_type()),
                forest_stores(
                    model.stable.worker_type(),
                    &|kind| model.stable.has_store(kind) && model.learned.has_store(kind),
                    &|kind| model.online.retained_observations_for(kind),
                ),
                model
                    .correction_stores()
                    .iter()
                    .flat_map(|(kind, stores)| correction_entries(Some(*kind), stores, true))
                    .collect(),
            ),
        };
        ForwardPassReadinessReport {
            worker_type,
            ready,
            stores,
            corrections,
        }
    }

    /// API:
    /// `model.deployment_topology() -> Option<ForwardPassDeploymentTopology>`
    ///
    /// Description: the topology and correction limits of the canonical
    /// identity this model was constructed for. `None` for models without
    /// construction provenance, such as `from_learned` models.
    pub fn deployment_topology(&self) -> Option<ForwardPassDeploymentTopology> {
        self.provenance
            .as_ref()
            .map(|provenance| ForwardPassDeploymentTopology::from_config(&provenance.config))
    }

    /// Return readiness and retained counts for every allocated regression store.
    ///
    /// Dedicated roles return one entry. Aggregated returns four entries in
    /// pure-decode, locally-mixed, cross-rank, pure-prefill order, including
    /// empty stores. Native models return an empty list. Readiness means a
    /// usable linear fit exists, including when spline fitting is selected.
    /// Spline component readiness alone does not make the store ready.
    pub fn regression_store_diagnostics(&self) -> Vec<ForwardPassRegressionStoreDiagnostics> {
        match &self.mode {
            ForwardPassPerfMode::Native { .. } => Vec::new(),
            ForwardPassPerfMode::Regression { regression, .. } => regression.diagnostics(),
            ForwardPassPerfMode::CausalSelector(model) => {
                workload_kinds_for(model.stable.worker_type())
                    .iter()
                    .copied()
                    .map(|workload_kind| ForwardPassRegressionStoreDiagnostics {
                        workload_kind,
                        ready: model.stable.has_store(workload_kind)
                            && model.learned.has_store(workload_kind),
                        retained_observations: model
                            .online
                            .retained_observations_for(workload_kind)
                            + model.store_correction_count(workload_kind),
                        spline: None,
                    })
                    .collect()
            }
            ForwardPassPerfMode::Learned {
                learned, online, ..
            } => workload_kinds_for(learned.worker_type())
                .iter()
                .copied()
                .map(|workload_kind| ForwardPassRegressionStoreDiagnostics {
                    workload_kind,
                    ready: learned.has_store(workload_kind),
                    retained_observations: online
                        .as_ref()
                        .map_or(0, |online| online.retained_observations_for(workload_kind)),
                    spline: None,
                })
                .collect(),
        }
    }

    /// API:
    /// `model.learned_feature_names() -> Option<&[String]>`
    ///
    /// Description: the ordered feature ABI of a learned model; `None` for
    /// native and regression models.
    pub fn learned_feature_names(&self) -> Option<&[String]> {
        match &self.mode {
            ForwardPassPerfMode::Learned { learned, .. } => Some(learned.feature_names()),
            ForwardPassPerfMode::CausalSelector(model) => Some(model.stable.feature_names()),
            _ => None,
        }
    }

    /// API:
    /// `model.learned_metadata() -> Option<&serde_json::Value>`
    ///
    /// Description: free-form trainer metadata embedded in a learned artifact.
    pub fn learned_metadata(&self) -> Option<&serde_json::Value> {
        match &self.mode {
            ForwardPassPerfMode::Learned { learned, .. } => Some(learned.metadata()),
            ForwardPassPerfMode::CausalSelector(model) => Some(model.stable.metadata()),
            _ => None,
        }
    }

    /// Fit pending observations and expire globally aged rows in every online
    /// store, including partial batches and idle stores.
    /// Prediction and checkpoint saving never implicitly flush pending feedback.
    pub fn flush_updates(&mut self) -> Result<(), AicError> {
        self.require_tunable()?;
        if let ForwardPassPerfMode::CausalSelector(model) = &mut self.mode {
            for kind in model.learned.store_kinds() {
                model.online.flush(kind, &mut model.learned)?;
            }
            return Ok(());
        }
        let ForwardPassPerfMode::Learned {
            learned,
            online: Some(online),
            ..
        } = &mut self.mode
        else {
            return Err(AicError::UnsupportedModel(
                "flush_updates requires online fpm_learned".into(),
            ));
        };
        for kind in learned.store_kinds() {
            online.flush(kind, learned)?;
        }
        Ok(())
    }

    /// Export the current forest as a frozen inference artifact, without fitting.
    pub fn export_learned_artifact(&self) -> Result<String, AicError> {
        match &self.mode {
            ForwardPassPerfMode::CausalSelector(_) => Err(AicError::UnsupportedModel(
                "combined artifact export is unsupported: a forest alone cannot preserve calibration and selector state".into(),
            )),
            ForwardPassPerfMode::Learned {
                learned,
                identity_verified,
                ..
            } => {
                let Some(provenance) = self.provenance.as_ref().filter(|_| *identity_verified)
                else {
                    return learned.to_json();
                };
                let mut artifact = learned.artifact();
                let metadata = artifact.metadata.as_object_mut().ok_or_else(|| {
                    AicError::InvalidEngineConfig(
                        "verified learned artifact export requires object metadata".into(),
                    )
                })?;
                metadata.insert(
                    "identity".into(),
                    canonical_learned_identity(&provenance.config)?,
                );
                serde_json::to_string(&artifact).map_err(|error| {
                    AicError::InvalidEngineConfig(format!(
                        "cannot export learned artifact: {error}"
                    ))
                })
            }
            _ => Err(AicError::UnsupportedModel(
                "artifact export requires fpm_learned".into(),
            )),
        }
    }

    /// Save complete online state to a new directory without fitting pending rows.
    pub fn save_online_checkpoint(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), AicError> {
        if matches!(&self.mode, ForwardPassPerfMode::CausalSelector(_)) {
            return Err(AicError::UnsupportedModel("combined checkpoint saving is unsupported: frozen calibration and selector state must also be preserved".into()));
        }
        let ForwardPassPerfMode::Learned {
            learned,
            online: Some(online),
            ..
        } = &self.mode
        else {
            return Err(AicError::UnsupportedModel(
                "checkpoint saving requires online fpm_learned".into(),
            ));
        };
        online.save(path.as_ref(), learned, "{}")
    }

    /// Online update counters and deployment identity verification status.
    pub fn learned_online_diagnostics(&self) -> Option<serde_json::Value> {
        if let ForwardPassPerfMode::CausalSelector(model) = &self.mode {
            let mut value = model.online.diagnostics();
            value["adaptation"] = "causal_selector".into();
            value["identity_verified"] = model.identity_verified.into();
            return Some(value);
        }
        let ForwardPassPerfMode::Learned {
            online,
            identity_verified,
            ..
        } = &self.mode
        else {
            return None;
        };
        let mut value = online.as_ref().map_or_else(
            || serde_json::json!({"adaptation": "frozen", "pending_observations": 0}),
            OnlineLearner::diagnostics,
        );
        let object = value.as_object_mut().expect("learner diagnostics object");
        object.insert(
            "adaptation".into(),
            if online.is_some() { "online" } else { "frozen" }.into(),
        );
        object.insert("identity_verified".into(), (*identity_verified).into());
        Some(value)
    }

    /// Causal evidence and calibration readiness, isolated per model instance,
    /// forest store and query type. Prediction does not mutate these counters.
    pub fn learned_selector_diagnostics(&self) -> Option<serde_json::Value> {
        match &self.mode {
            ForwardPassPerfMode::CausalSelector(model) => Some(model.diagnostics()),
            _ => None,
        }
    }

    /// API:
    /// `model.min_correction_factor() -> Option<f64>`
    ///
    /// Description: return the smallest ready correction factor across all
    /// workload kinds. Native and Learned models both carry the online
    /// correction grid; the factor multiplies their base estimate.
    ///
    /// Returns `None` before any correction workload kind has enough samples.
    /// Regression-only models have no correction grid and always return `None`.
    pub fn min_correction_factor(&self) -> Option<f64> {
        self.correction_factors()
            .into_iter()
            .reduce(|a, b| a.min(b))
    }

    /// API:
    /// `model.max_correction_factor() -> Option<f64>`
    ///
    /// Description: return the largest ready correction factor across all
    /// workload kinds. Native and Learned models both carry the online
    /// correction grid; the factor multiplies their base estimate.
    ///
    /// Returns `None` before any correction workload kind has enough samples.
    /// Regression-only models have no correction grid and always return `None`.
    pub fn max_correction_factor(&self) -> Option<f64> {
        self.correction_factors()
            .into_iter()
            .reduce(|a, b| a.max(b))
    }

    /// API:
    /// `model.avg_correction_factor() -> Option<f64>`
    ///
    /// Description: return the arithmetic mean of ready correction factors
    /// across all workload kinds (Native and Learned models alike). Dynamic
    /// calibration includes each ready recent-global fallback once, in addition
    /// to its populated local bucket factors.
    /// Hybrid includes both grids (including overlapping regions) and the
    /// dynamic fallback; this is a state summary, not a query-weighted average.
    ///
    /// Returns `None` before any correction workload kind has enough samples.
    /// Regression-only models have no correction grid and always return `None`.
    pub fn avg_correction_factor(&self) -> Option<f64> {
        let factors = self.correction_factors();
        if factors.is_empty() {
            None
        } else {
            Some(factors.iter().sum::<f64>() / factors.len() as f64)
        }
    }

    /// API:
    /// `model.options() -> &ForwardPassPerfOptions`
    ///
    /// Description: return the immutable tuning options used by this model.
    pub fn options(&self) -> &ForwardPassPerfOptions {
        &self.options
    }

    /// Uncorrected static prefill latency, preserving the compiled engine's
    /// existing `(batch, full input length, cached prefix)` contract. Unlike
    /// telemetry estimation this does not apply learned online correction.
    /// Enabled coverage records the actual native lookup on this model.
    pub fn predict_prefill_latency(
        &self,
        batch_size: u32,
        isl: u32,
        prefix: u32,
    ) -> Result<f64, AicError> {
        let engine = self.require_native_engine()?;
        let mut coverage = self
            .query_coverage
            .as_ref()
            .map(FpmCoverageState::lock)
            .transpose()?;
        engine.predict_prefill_latency_with_coverage(
            batch_size,
            isl,
            prefix,
            coverage.as_deref_mut(),
        )
    }

    /// Uncorrected prefill latency using exact iteration totals: tokens the
    /// step computes and cached or earlier-chunk KV its requests read. Direct
    /// whole-model FPM queries these totals without per-request mean rounding.
    pub fn predict_prefill_latency_total(
        &self,
        batch_size: u32,
        total_prefill_tokens: u32,
        total_kv_read_tokens: u32,
    ) -> Result<f64, AicError> {
        let engine = self.require_native_engine()?;
        let mut coverage = self
            .query_coverage
            .as_ref()
            .map(FpmCoverageState::lock)
            .transpose()?;
        engine.predict_prefill_latency_total_with_coverage(
            batch_size,
            total_prefill_tokens,
            total_kv_read_tokens,
            coverage.as_deref_mut(),
        )
    }

    /// Uncorrected decode latency using exact past-KV totals, excluding the
    /// current input token per request. Shares the existing engine contract.
    pub fn predict_decode_latency_total(
        &self,
        batch_size: u32,
        total_past_kv_tokens: u32,
    ) -> Result<f64, AicError> {
        let engine = self.require_native_engine()?;
        let mut coverage = self
            .query_coverage
            .as_ref()
            .map(FpmCoverageState::lock)
            .transpose()?;
        engine.predict_decode_latency_total_with_coverage(
            batch_size,
            total_past_kv_tokens,
            coverage.as_deref_mut(),
        )
    }

    /// Largest collected decode KV total for the selected FPM cell. Reading
    /// this bound is not a timing lookup and does not add coverage evidence.
    pub fn fpm_decode_kv_ceiling(&self) -> Result<Option<u32>, AicError> {
        self.require_native_engine()?.fpm_decode_kv_ceiling()
    }

    /// Snapshot of actual native direct-FPM lookup resolutions, including
    /// failed lookups. `None` means collection was disabled. Counts do not
    /// include timings reused by an external cache, or calls on a separate
    /// low-level engine handle. Replay completion must be checked separately.
    pub fn fpm_query_coverage(&self) -> Result<Option<FpmQueryCoverage>, AicError> {
        self.query_coverage
            .as_ref()
            .map(|state| state.lock().map(|value| value.clone()))
            .transpose()
    }

    fn require_native_engine(&self) -> Result<Arc<Engine>, AicError> {
        self.native_engine().ok_or_else(|| {
            AicError::InvalidEngineConfig(
                "static timing requires a native forward-pass estimator".into(),
            )
        })
    }

    /// Static phase latency before online correction, using the native engine's
    /// existing integration. Decode returns the total for all generated tokens.
    pub fn static_phase_latency(
        &self,
        batch_size: u32,
        input_tokens: u32,
        output_tokens: u32,
        prefill: bool,
    ) -> Result<f64, AicError> {
        let engine = self.native_engine().ok_or_else(|| {
            AicError::InvalidEngineConfig("static phase latency requires a native estimator".into())
        })?;
        if prefill {
            engine.predict_prefill_latency(batch_size, input_tokens, 0)
        } else {
            engine.predict_decode_latency(batch_size, input_tokens, output_tokens)
        }
    }

    pub(crate) fn static_prefill_detailed(
        &self,
        batch_size: u32,
        input_tokens: u32,
        prefix: u32,
    ) -> Result<crate::ForwardPassEstimate, AicError> {
        let engine = self.native_engine().ok_or_else(|| {
            AicError::InvalidEngineConfig("static prefill requires a native estimator".into())
        })?;
        let mut coverage = self
            .query_coverage
            .as_ref()
            .map(FpmCoverageState::lock)
            .transpose()?;
        engine.predict_prefill_detailed(batch_size, input_tokens, prefix, coverage.as_deref_mut())
    }

    /// [`Self::static_prefill_detailed`] at exact iteration totals.
    pub(crate) fn static_prefill_detailed_total(
        &self,
        batch_size: u32,
        total_prefill_tokens: u32,
        total_kv_read_tokens: u32,
    ) -> Result<crate::ForwardPassEstimate, AicError> {
        let engine = self.native_engine().ok_or_else(|| {
            AicError::InvalidEngineConfig("static prefill requires a native estimator".into())
        })?;
        let mut coverage = self
            .query_coverage
            .as_ref()
            .map(FpmCoverageState::lock)
            .transpose()?;
        engine.predict_prefill_detailed_total(
            batch_size,
            total_prefill_tokens,
            total_kv_read_tokens,
            coverage.as_deref_mut(),
        )
    }

    /// Native operation evidence for one static prefill or decode step. Values
    /// precede learned online correction; SOL is a comparison only. Whole-model
    /// estimators cannot provide an operation decomposition and fail explicitly.
    pub fn static_phase_diagnostics(
        &self,
        batch_size: u32,
        context_length: u32,
        prefix: u32,
        prefill: bool,
    ) -> Result<Vec<crate::perfmodel::engine::diagnostics::StaticOperationDiagnostics>, AicError>
    {
        if self
            .provenance
            .as_ref()
            .is_some_and(|p| p.selected_estimation_mode != EstimationMode::OpLevel)
        {
            return Err(AicError::InvalidEngineConfig(
                "operation diagnostics require op_level estimation".into(),
            ));
        }
        self.native_engine()
            .ok_or_else(|| {
                AicError::InvalidEngineConfig(
                    "operation diagnostics require a native op-level estimator".into(),
                )
            })?
            .static_phase_diagnostics(batch_size, context_length, prefix, prefill)
    }

    /// Native operation values of one vision-encoder call over `shapes`, name
    /// folded like the other per-op walks. Requires a native estimator compiled
    /// with `encoder_parallel` for a model that has a vision tower.
    pub fn vision_operations(
        &self,
        shapes: &[EncoderImageShape],
    ) -> Result<Vec<PerOpValue>, AicError> {
        self.native_engine()
            .ok_or_else(|| {
                AicError::InvalidEngineConfig(
                    "vision encoder timing requires a native estimator".into(),
                )
            })?
            .evaluate_vision(shapes)
    }

    /// Latency of one vision-encoder call over `shapes`, in milliseconds.
    pub fn predict_vision_ms(&self, shapes: &[EncoderImageShape]) -> Result<f64, AicError> {
        Ok(self
            .vision_operations(shapes)?
            .iter()
            .map(|(_, latency_ms, _, _)| latency_ms)
            .sum())
    }

    pub(crate) fn native_engine(&self) -> Option<Arc<Engine>> {
        match &self.mode {
            ForwardPassPerfMode::Native { engine, .. } => Some(Arc::clone(engine)),
            ForwardPassPerfMode::Regression { .. }
            | ForwardPassPerfMode::Learned { .. }
            | ForwardPassPerfMode::CausalSelector(_) => None,
        }
    }

    /// Exact immutable construction identity and selected systems root.
    pub fn provenance(&self) -> Option<&ForwardPassPerfProvenance> {
        self.provenance.as_ref()
    }

    fn correction_factor(
        &self,
        corrections: &WorkloadStores<CorrectionBuckets>,
        feature: &IterationFeatures,
    ) -> f64 {
        if self.is_correction_enabled {
            corrections
                .store(feature.workload_kind)
                .correction_factor_for(&feature.x)
        } else {
            1.0
        }
    }

    fn correction_factors(&self) -> Vec<f64> {
        match &self.mode {
            ForwardPassPerfMode::Native { corrections, .. } => corrections.correction_factors(),
            ForwardPassPerfMode::Regression { .. } => Vec::new(),
            ForwardPassPerfMode::Learned { corrections, .. } => corrections.correction_factors(),
            ForwardPassPerfMode::CausalSelector(model) => model.correction_factors(),
        }
    }
}

#[cfg(feature = "python")]
fn build_native_candidate(
    config: &ForwardPassPerfModelConfig,
) -> Result<(Engine, PathBuf), AicError> {
    if config.estimation_mode == EstimationMode::FpmInterpolation
        && (config.enable_eplb
            || config.moe_kernel_source.is_some()
            || config.wideep_num_slots.is_some()
            || config
                .moe_backend
                .as_deref()
                .is_some_and(|value| value != "default"))
    {
        return Err(AicError::UnsupportedModel(
            "FPM interpolation does not support EPLB, slots, moe_backend or moe_kernel_source overrides".into(),
        ));
    }
    if config.estimation_mode == EstimationMode::FpmInterpolation
        && config.nextn != 0
        && config.validate_fpm_execution()?.is_none()
    {
        return Err(AicError::UnsupportedModel(
            "FPM interpolation does not support MTP".into(),
        ));
    }
    let mut last_error = None;
    for root in resolve_systems_roots(config)? {
        match build_engine_via_python(config, &root).and_then(|engine| {
            engine.validate_forward_pass_readiness(config.worker_type)?;
            Ok(engine)
        }) {
            Ok(engine) => return Ok((engine, root)),
            Err(err) if can_fallback_to_regression(&err) => last_error = Some(err),
            Err(err) => return Err(err),
        }
    }
    Err(last_error.unwrap_or_else(|| AicError::DataRoot("no systems roots".into())))
}

#[cfg(not(feature = "python"))]
fn build_native_candidate(
    _config: &ForwardPassPerfModelConfig,
) -> Result<(Engine, PathBuf), AicError> {
    Err(AicError::UnsupportedModel(
        "native engine construction requires the python feature".into(),
    ))
}

/// Learned-mode base estimate: classify the iteration with the regression
/// workload rule bound to the artifact's worker type, then query the matching
/// trained store. `Ok(Some(0.0))` for empty iterations, `Ok(None)` when the
/// kind has no store.
fn learned_base_ms(
    learned: &LearnedForwardPassModel,
    metrics_by_rank: &[ForwardPassMetrics],
) -> Result<Option<f64>, AicError> {
    let Some((kind, vector)) = prepare_learned_iteration(learned, metrics_by_rank)? else {
        return Ok(Some(0.0));
    };
    vector
        .map(|vector| learned.predict_ms(kind, &vector))
        .transpose()
        .map(Option::flatten)
}

fn prepare_learned_iteration(
    learned: &LearnedForwardPassModel,
    metrics_by_rank: &[ForwardPassMetrics],
) -> Result<
    Option<(
        ForwardPassRegressionWorkloadKind,
        Option<IterationFeatureVector>,
    )>,
    AicError,
> {
    // Routing only depends on the batch composition, never on the regression
    // tuning weights, so an invalid regression-only option cannot reject a
    // learned prediction.
    let Some(workload_kind) = classify_regression_workload(metrics_by_rank, learned.worker_type())?
    else {
        return Ok(None);
    };
    if !learned.has_store(workload_kind) {
        return Ok(Some((workload_kind, None)));
    }
    let vector =
        IterationFeatureVector::from_metrics_with(metrics_by_rank, learned.needs_slot_features());
    if learned.needs_request_features() && !vector.request_lists {
        // Without the lists every req_*/slot feature is NaN and the trees
        // would return one constant for every batch. Refuse loudly instead.
        return Err(AicError::InvalidForwardPassMetrics(
            "learned artifact uses per-request features but the iteration carries no aligned, \
             consistent extend_lengths/past_kv_lengths on every active rank; collect with the \
             per-request FPM hooks (aisimulate_core.fpm_hooks) or train with --features v1"
                .to_string(),
        ));
    }
    Ok(Some((workload_kind, Some(vector))))
}

#[cfg(feature = "python")]
fn build_engine_via_python(
    config: &ForwardPassPerfModelConfig,
    systems_root: &std::path::Path,
) -> Result<Engine, AicError> {
    let systems_path = systems_root.to_str().ok_or_else(|| {
        AicError::InvalidEngineConfig(format!(
            "systems_path is not valid UTF-8: {}",
            systems_root.display()
        ))
    })?;
    crate::py::compile_forward_pass_model_to_engine(config, systems_path)
}

#[cfg(feature = "python")]
fn resolve_systems_roots(config: &ForwardPassPerfModelConfig) -> Result<Vec<PathBuf>, AicError> {
    if !config.systems_paths.is_empty() {
        return Ok(config.systems_paths.clone());
    }
    crate::py::resolve_forward_pass_systems_roots()
}

fn transfer_policy_tokens(policy: crate::common::enums::TransferPolicy) -> Vec<String> {
    use crate::common::enums::TransferKind;
    [
        (TransferKind::XShape, "xshape"),
        (TransferKind::XQuant, "xquant"),
        (TransferKind::XProfile, "xprofile"),
        (TransferKind::XOp, "xop"),
    ]
    .into_iter()
    .filter(|(kind, _)| policy.contains(*kind))
    .map(|(_, token)| token.to_string())
    .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum WorkloadKind {
    Prefill,
    Decode,
    Mixed,
}

#[derive(Clone, Debug)]
pub(crate) struct IterationFeatures {
    pub(crate) workload_kind: WorkloadKind,
    pub(crate) x: Vec<f64>,
}

impl IterationFeatures {
    pub(crate) fn from_metrics(
        metrics_by_rank: &[ForwardPassMetrics],
    ) -> Result<Option<Self>, AicError> {
        if metrics_by_rank.is_empty() {
            return Err(AicError::InvalidForwardPassMetrics(
                "at least one attention-DP rank metric is required".to_string(),
            ));
        }
        for metrics in metrics_by_rank {
            validate_forward_pass_metrics(metrics)?;
        }

        Ok(Self::from_validated_metrics(metrics_by_rank))
    }

    fn from_validated_metrics(metrics_by_rank: &[ForwardPassMetrics]) -> Option<Self> {
        metrics_by_rank
            .iter()
            .filter_map(Self::from_single_rank)
            .max_by(|left, right| {
                left.load_score()
                    .partial_cmp(&right.load_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    fn from_single_rank(metrics: &ForwardPassMetrics) -> Option<Self> {
        let scheduled = &metrics.scheduled_requests;
        let has_prefill = scheduled.sum_prefill_tokens > 0;
        let has_decode = scheduled.num_decode_requests > 0 || scheduled.sum_decode_kv_tokens > 0;
        let feature = match (has_prefill, has_decode) {
            (false, false) => return None,
            (true, false) => Self {
                workload_kind: WorkloadKind::Prefill,
                x: vec![f64::from(scheduled.sum_prefill_tokens)],
            },
            (false, true) => Self {
                workload_kind: WorkloadKind::Decode,
                x: vec![
                    f64::from(scheduled.num_decode_requests),
                    f64::from(scheduled.sum_decode_kv_tokens),
                ],
            },
            (true, true) => Self {
                workload_kind: WorkloadKind::Mixed,
                x: vec![
                    f64::from(scheduled.sum_prefill_tokens),
                    f64::from(scheduled.sum_decode_kv_tokens),
                ],
            },
        };
        Some(feature)
    }

    fn load_score(&self) -> f64 {
        self.x.iter().sum()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct IterationObservation {
    pub(crate) feature: IterationFeatures,
    pub(crate) wall_time_ms: f64,
}

impl IterationObservation {
    pub(crate) fn from_metrics(
        metrics_by_rank: &[ForwardPassMetrics],
    ) -> Result<Option<Self>, AicError> {
        let Some(feature) = IterationFeatures::from_metrics(metrics_by_rank)? else {
            return Ok(None);
        };
        let Some(wall_time_ms) = max_positive_wall_time_ms(metrics_by_rank) else {
            return Ok(None);
        };
        Ok(Some(Self {
            feature,
            wall_time_ms,
        }))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RegressionIterationFeatures {
    pub(crate) x: [f64; 2],
    pub(crate) workload_kind: ForwardPassRegressionWorkloadKind,
}

/// Validate one iteration's per-rank metrics against the worker role and
/// classify its regression workload kind. Independent of any tuning weights,
/// so the learned model (which has no regression features) shares it.
///
/// Returns `Ok(None)` for an idle iteration.
pub(crate) fn classify_regression_workload(
    metrics_by_rank: &[ForwardPassMetrics],
    worker_type: ForwardPassWorkerType,
) -> Result<Option<ForwardPassRegressionWorkloadKind>, AicError> {
    if metrics_by_rank.is_empty() {
        return Err(AicError::InvalidForwardPassMetrics(
            "at least one attention-DP rank metric is required".to_string(),
        ));
    }
    let mut has_prefill = false;
    let mut has_decode = false;
    let mut has_locally_mixed = false;
    for metrics in metrics_by_rank {
        validate_forward_pass_metrics(metrics)?;
        let scheduled = &metrics.scheduled_requests;
        match worker_type {
            ForwardPassWorkerType::Prefill if scheduled.num_decode_requests > 0 => {
                return Err(AicError::InvalidForwardPassMetrics(
                    "prefill regression worker received scheduled decode work".to_string(),
                ));
            }
            ForwardPassWorkerType::Decode if scheduled.sum_prefill_tokens > 0 => {
                return Err(AicError::InvalidForwardPassMetrics(
                    "decode regression worker received scheduled prefill work".to_string(),
                ));
            }
            _ => {}
        }
        let rank_has_prefill = scheduled.sum_prefill_tokens > 0;
        let rank_has_decode = scheduled.num_decode_requests > 0;
        has_prefill |= rank_has_prefill;
        has_decode |= rank_has_decode;
        has_locally_mixed |= rank_has_prefill && rank_has_decode;
    }

    use ForwardPassRegressionWorkloadKind::*;
    Ok(Some(match (has_prefill, has_decode) {
        (false, false) => return Ok(None),
        (false, true) => PureDecode,
        (true, false) => PurePrefill,
        (true, true) if has_locally_mixed => ContainsLocallyMixed,
        (true, true) => CrossRankAggregated,
    }))
}

impl RegressionIterationFeatures {
    pub(crate) fn from_metrics(
        metrics_by_rank: &[ForwardPassMetrics],
        worker_type: ForwardPassWorkerType,
        options: &ForwardPassPerfOptions,
    ) -> Result<Option<Self>, AicError> {
        let Some(workload_kind) = classify_regression_workload(metrics_by_rank, worker_type)?
        else {
            return Ok(None);
        };

        let alpha = options.regression_attention_kv_weight;
        let beta = options.regression_prefill_attention_pair_weight;
        let gamma = options.regression_ffn_token_weight;
        let mut critical_attention = 0.0_f64;
        let mut sum_prefill_tokens = 0.0_f64;
        let mut sum_decode_requests = 0.0_f64;

        for metrics in metrics_by_rank {
            let scheduled = &metrics.scheduled_requests;
            let num_prefill_requests = f64::from(scheduled.num_prefill_requests);
            let prefill_tokens = f64::from(scheduled.sum_prefill_tokens);
            let prefill_kv_tokens = f64::from(scheduled.sum_prefill_kv_tokens);
            let decode_requests = f64::from(scheduled.num_decode_requests);
            let decode_kv_tokens = f64::from(scheduled.sum_decode_kv_tokens);

            let (active_prefill_kv_tokens, prefill_attention_pairs) = if prefill_tokens > 0.0 {
                debug_assert!(num_prefill_requests > 0.0);
                (
                    prefill_kv_tokens,
                    prefill_kv_tokens * prefill_tokens / num_prefill_requests
                        + prefill_tokens * prefill_tokens / (2.0 * num_prefill_requests)
                        + prefill_tokens / 2.0,
                )
            } else {
                (0.0, 0.0)
            };

            let rank_attention = match worker_type {
                ForwardPassWorkerType::Prefill => {
                    alpha * active_prefill_kv_tokens + beta * prefill_attention_pairs
                }
                ForwardPassWorkerType::Decode => alpha * decode_kv_tokens,
                ForwardPassWorkerType::Aggregated => {
                    alpha * (active_prefill_kv_tokens + decode_kv_tokens)
                        + beta * prefill_attention_pairs
                }
            };
            critical_attention = critical_attention.max(rank_attention);
            sum_prefill_tokens += prefill_tokens;
            sum_decode_requests += decode_requests;
        }

        let global_ffn = gamma
            * match worker_type {
                ForwardPassWorkerType::Prefill => sum_prefill_tokens,
                ForwardPassWorkerType::Decode => sum_decode_requests,
                ForwardPassWorkerType::Aggregated => sum_prefill_tokens + sum_decode_requests,
            };
        let x = [critical_attention, global_ffn];
        if !x.iter().all(|value| value.is_finite() && *value >= 0.0) {
            return Err(AicError::InvalidForwardPassMetrics(
                "derived regression features must be finite and nonnegative".to_string(),
            ));
        }
        Ok(Some(Self { x, workload_kind }))
    }
}

#[derive(Clone, Debug)]
struct RegressionIterationObservation {
    feature: RegressionIterationFeatures,
    wall_time_ms: f64,
}

impl RegressionIterationObservation {
    fn from_metrics(
        metrics_by_rank: &[ForwardPassMetrics],
        worker_type: ForwardPassWorkerType,
        options: &ForwardPassPerfOptions,
    ) -> Result<Option<Self>, AicError> {
        let Some(feature) =
            RegressionIterationFeatures::from_metrics(metrics_by_rank, worker_type, options)?
        else {
            return Ok(None);
        };
        let Some(wall_time_ms) = max_positive_wall_time_ms(metrics_by_rank) else {
            return Ok(None);
        };
        Ok(Some(Self {
            feature,
            wall_time_ms,
        }))
    }
}

fn max_positive_wall_time_ms(metrics_by_rank: &[ForwardPassMetrics]) -> Option<f64> {
    let wall_time_seconds = metrics_by_rank
        .iter()
        .map(|metrics| metrics.wall_time)
        .filter(|wall_time| wall_time.is_finite() && *wall_time > 0.0)
        .fold(0.0_f64, f64::max);
    // Select a usable source value before converting units. The conversion may
    // overflow; correction and regression ingestion reject that target after
    // Native has had the opportunity to surface any engine estimation error.
    (wall_time_seconds > 0.0).then_some(wall_time_seconds * 1000.0)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct WorkloadStores<T> {
    prefill: T,
    decode: T,
    mixed: T,
}

impl WorkloadStores<CorrectionBuckets> {
    /// Validate decoded correction stores against freshly configured ones.
    fn validate_restored(&self, fresh: &Self) -> Result<(), String> {
        for (name, saved, fresh) in [
            ("prefill", &self.prefill, &fresh.prefill),
            ("decode", &self.decode, &fresh.decode),
            ("mixed", &self.mixed, &fresh.mixed),
        ] {
            saved
                .validate_restored(fresh)
                .map_err(|error| format!("{name} correction store: {error}"))?;
        }
        Ok(())
    }
}

impl<T: WithOptions> WorkloadStores<T> {
    fn with_options(options: &ForwardPassPerfOptions) -> Self {
        Self {
            prefill: T::with_options(options, &[AxisRange::from_zero_to(options.max_num_tokens)]),
            decode: T::with_options(
                options,
                &[
                    AxisRange::from_zero_to(options.max_batch_size),
                    AxisRange::from_zero_to(options.max_kv_tokens),
                ],
            ),
            mixed: T::with_options(
                options,
                &[
                    AxisRange::from_zero_to(options.max_num_tokens),
                    AxisRange::from_zero_to(options.max_kv_tokens),
                ],
            ),
        }
    }
}

impl<T: StoreStats> WorkloadStores<T> {
    fn any_ready(&self) -> bool {
        self.prefill.is_ready() || self.decode.is_ready() || self.mixed.is_ready()
    }

    fn observation_count(&self) -> usize {
        self.prefill.observation_count()
            + self.decode.observation_count()
            + self.mixed.observation_count()
    }
}

impl WorkloadStores<CorrectionBuckets> {
    fn ready_bucket_count(&self) -> usize {
        self.prefill.ready_bucket_count()
            + self.decode.ready_bucket_count()
            + self.mixed.ready_bucket_count()
    }

    fn correction_factors(&self) -> Vec<f64> {
        let mut factors = self.prefill.correction_factors();
        factors.extend(self.decode.correction_factors());
        factors.extend(self.mixed.correction_factors());
        factors
    }
}

impl<T> WorkloadStores<T> {
    fn store(&self, workload_kind: WorkloadKind) -> &T {
        match workload_kind {
            WorkloadKind::Prefill => &self.prefill,
            WorkloadKind::Decode => &self.decode,
            WorkloadKind::Mixed => &self.mixed,
        }
    }

    fn store_mut(&mut self, workload_kind: WorkloadKind) -> &mut T {
        match workload_kind {
            WorkloadKind::Prefill => &mut self.prefill,
            WorkloadKind::Decode => &mut self.decode,
            WorkloadKind::Mixed => &mut self.mixed,
        }
    }
}

/// Decide whether `best_available` should fall back to regression instead of
/// propagating `err`. Covers the unsupported-model / data-availability errors
/// that mean "this model can't be served natively". A failed native build via
/// Python `compile_engine` surfaces as [`AicError::UnsupportedModel`] (see
/// `py::compile_engine_from_flat`), which is covered here.
///
/// [`AicError::InvalidEngineConfig`] is deliberately NOT fallback-safe: it is
/// used for hard caller/config errors (e.g. a non-UTF-8 `systems_path`, invalid
/// FPM options, a malformed spec). Those must surface rather than silently
/// degrade `best_available` to regression mode.
fn can_fallback_to_regression(err: &AicError) -> bool {
    matches!(
        err,
        AicError::UnsupportedModel(_)
            | AicError::DataRoot(_)
            | AicError::ModelConfig(_)
            | AicError::PerfDatabase(_)
            | AicError::Io { .. }
    )
}

#[cfg(test)]
mod rebuild_tests {
    use super::*;
    use ForwardPassRegressionWorkloadKind::*;

    #[test]
    fn spline_workload_stores_and_clones_have_independent_search_clocks() {
        use super::super::estimator::{RegressionFitKind, SplineFitConfig, SplineSearchConfig};
        let fit = RegressionFitConfig {
            kind: RegressionFitKind::Spline,
            spline: Some(SplineFitConfig {
                search: SplineSearchConfig::Periodic { step: 64 },
                ..SplineFitConfig::default()
            }),
            ..RegressionFitConfig::default()
        };
        let mut stores = RegressionStores::new(
            ForwardPassWorkerType::Aggregated,
            &ForwardPassPerfOptions::default(),
            &fit,
            &RegressionSamplingConfig::default(),
        );
        // Analytic one-axis plane: y=3+2*x. Other workload stores remain cold.
        for i in 1..=32 {
            assert!(
                stores
                    .store_mut(PurePrefill)
                    .add_observation([i as f64, 1.0], 3.0 + 2.0 * i as f64)
            );
        }
        let mut cloned = stores.clone();
        for i in 33..=64 {
            assert!(
                stores
                    .store_mut(PurePrefill)
                    .add_observation([i as f64, 1.0], 3.0 + 2.0 * i as f64)
            );
        }
        assert!(
            cloned
                .store_mut(PureDecode)
                .add_observation([1.0, 1.0], 5.0)
        );
        let original = stores.diagnostics();
        let copy = cloned.diagnostics();
        let find = |items: &[ForwardPassRegressionStoreDiagnostics], kind| {
            items
                .iter()
                .find(|store| store.workload_kind == kind)
                .unwrap()
                .spline
                .clone()
                .unwrap()
        };
        assert_eq!(
            find(&original, PurePrefill).last_search_observation,
            Some(64)
        );
        assert_eq!(find(&original, PurePrefill).knot_searches, 2);
        assert_eq!(find(&copy, PurePrefill).last_search_observation, Some(32));
        assert_eq!(find(&copy, PurePrefill).knot_searches, 1);
        assert_eq!(find(&original, PureDecode).accepted_observations, 0);
        assert_eq!(find(&copy, PureDecode).accepted_observations, 1);
        assert!(
            !original
                .iter()
                .find(|store| store.workload_kind == CrossRankAggregated)
                .unwrap()
                .ready
        );
    }

    #[test]
    fn workload_stores_and_clones_have_independent_rebuild_clocks() {
        let options = ForwardPassPerfOptions::default();
        let mut stores = RegressionStores::new(
            ForwardPassWorkerType::Aggregated,
            &options,
            &RegressionFitConfig {
                rebuild_interval: Some(3),
                ..RegressionFitConfig::default()
            },
            &RegressionSamplingConfig::default(),
        );
        assert!(
            stores
                .store_mut(PurePrefill)
                .add_observation([1.0, 2.0], 3.0)
        );
        for i in 1..=2 {
            assert!(
                stores
                    .store_mut(PureDecode)
                    .add_observation([i as f64, 1.0], 4.0)
            );
        }
        let mut cloned = stores.clone();
        assert!(
            stores
                .store_mut(PureDecode)
                .add_observation([3.0, 4.0], 5.0)
        );
        assert_eq!(stores.store(PureDecode).mutations_since_rebuild(), 0);
        assert_eq!(stores.store(PurePrefill).mutations_since_rebuild(), 1);
        assert_eq!(
            stores.store(ContainsLocallyMixed).mutations_since_rebuild(),
            0
        );
        assert_eq!(
            stores.store(CrossRankAggregated).mutations_since_rebuild(),
            0
        );
        assert_eq!(cloned.store(PureDecode).mutations_since_rebuild(), 2);
        assert!(
            cloned
                .store_mut(PurePrefill)
                .add_observation([2.0, 3.0], 4.0)
        );
        assert_eq!(cloned.store(PurePrefill).mutations_since_rebuild(), 2);
        assert_eq!(stores.store(PurePrefill).mutations_since_rebuild(), 1);
    }

    #[test]
    fn default_rebuild_interval_disables_periodic_refresh_for_every_workload_store() {
        let mut stores = RegressionStores::new(
            ForwardPassWorkerType::Aggregated,
            &ForwardPassPerfOptions::default(),
            &RegressionFitConfig::default(),
            &RegressionSamplingConfig::default(),
        );
        // Constant features avoid fit work without numerical damage. A change
        // back to a 4096-operation default would reset each clock at 4096.
        for (_, store) in &mut stores.stores {
            for _ in 0..2081 {
                assert!(store.add_observation([1.0, 2.0], 3.0));
            }
            assert_eq!(store.mutations_since_rebuild(), 4098);
        }
    }
}

#[cfg(test)]
mod fallback_errors {
    use super::*;

    #[test]
    fn canonical_learned_rechecks_the_exact_artifact_bytes_after_normalization() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("artifact.json");
        let mut artifact = serde_json::json!({
            "schema": "aic_fpm_learned_forward_perf", "schema_version": 1,
            "worker_type": "decode", "target": "ms", "features": ["num_decode_requests"],
            "stores": {"pure_decode": {"baseline": 10.0, "trees": []}}, "metadata": {}
        });
        std::fs::write(&path, artifact.to_string()).unwrap();
        let mut config = ForwardPassPerfModelConfig::new(
            "model",
            "system",
            crate::BackendKind::Vllm,
            ForwardPassWorkerType::Decode,
        );
        config.estimation_mode = EstimationMode::FpmLearned;
        config.estimator_config.fpm_learned.artifact_path = Some(path.clone());
        config.estimator_config.fpm_learned.adaptation = crate::LearnedAdaptation::Frozen;
        let resolved = config.resolve().unwrap();
        ForwardPassPerfModel::from_learned_config(&resolved).unwrap();
        artifact["stores"]["pure_decode"]["baseline"] = 20.0.into();
        std::fs::write(&path, artifact.to_string()).unwrap();
        let error = ForwardPassPerfModel::from_learned_config(&resolved).unwrap_err();
        assert!(error.to_string().contains("changed after normalization"));
    }

    #[cfg(feature = "python")]
    #[test]
    fn interpolation_candidate_does_not_discard_exact_moe_source() {
        let mut config = ForwardPassPerfModelConfig::new(
            "model",
            "system",
            crate::BackendKind::Sglang,
            crate::ForwardPassWorkerType::Decode,
        );
        config.moe_kernel_source = Some("source_that_does_not_exist".into());
        config.estimation_mode = EstimationMode::FpmInterpolation;
        let error = build_native_candidate(&config).err().unwrap();
        assert!(error.to_string().contains("moe_kernel_source overrides"));
    }

    #[test]
    fn corruption_is_not_a_coverage_gap() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>("broken: [").unwrap_err();
        assert!(!can_fallback_to_regression(&AicError::Yaml {
            path: "system.yaml".into(),
            source: yaml
        }));
        assert!(!can_fallback_to_regression(&AicError::Parquet {
            path: "gemm_perf.parquet".into(),
            source: parquet::errors::ParquetError::General("corrupt footer".into()),
        }));
        assert!(can_fallback_to_regression(&AicError::UnsupportedModel(
            "no coverage".into()
        )));
    }
}
