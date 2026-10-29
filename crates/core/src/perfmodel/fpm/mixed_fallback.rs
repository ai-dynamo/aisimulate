// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Temporary mixed composition with frozen original-A eligibility and independent
//! calibration. A+C supports ordinary-first, quality-gated, and selector routing.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::super::estimator::{
    TemporaryMixedCompositionConfig, TemporaryMixedCompositionMode,
    TemporaryMixedCompositionScenario,
};
use super::super::quality_gate::{QualityGate, QualityUpdate};
use super::super::selector::SelectedExpert;
use super::combined::MixedSelectorPrediction;
use super::*;

type Kind = ForwardPassRegressionWorkloadKind;
const MIXED_KINDS: [Kind; 2] = [Kind::CrossRankAggregated, Kind::ContainsLocallyMixed];
const VERSION: &str = "original_a_mixed_experts_attention_moe_v3";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MixedSnapshot {
    version: String,
    binding: serde_json::Value,
    admissions: BTreeMap<Kind, u64>,
    corrections: BTreeMap<Kind, super::super::correction::HybridCorrectionSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    quality_gate: Option<QualityGate>,
}

fn invalid(reason: impl std::fmt::Display) -> AicError {
    AicError::InvalidEngineConfig(format!("temporary mixed composition: {reason}"))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SeedReference {
    path: PathBuf,
    sha256: String,
    #[serde(default)]
    checkpoint_manifest_sha256: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalAManifest {
    schema_version: u32,
    source_kind: String,
    identity: serde_json::Value,
    min_store_rows: usize,
    accepted_iterations: Vec<Vec<ForwardPassMetrics>>,
    seed_artifacts: Vec<SeedReference>,
    provenance: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(super) struct ClassRoute {
    workload_kind: Kind,
    fallback_eligible: bool,
    accepted_a_groups: usize,
    reason: String,
}

#[derive(Clone, Debug)]
pub(super) struct MixedComposition {
    policy: TemporaryMixedCompositionConfig,
    routes: Vec<ClassRoute>,
    engine: Option<Arc<Engine>>,
    score_options: ForwardPassPerfOptions,
    bounds: [f64; 2],
    corrections: BTreeMap<Kind, CorrectionBuckets>,
    admissions: BTreeMap<Kind, u64>,
    identity: serde_json::Value,
    prior: serde_json::Value,
    attention_dp: usize,
    anchor_digests: BTreeSet<String>,
    pub(super) quality_gate: Option<QualityGate>,
}

/// Validated raw feedback, prepared before either history mutates.
pub(super) struct PreparedMixedObservation {
    kind: Kind,
    x: Vec<f64>,
    observed: f64,
    raw: f64,
}

fn checked_bytes(path: &Path, expected: &str) -> Result<Vec<u8>, AicError> {
    let bytes = std::fs::read(path).map_err(|source| AicError::Io {
        path: path.into(),
        source,
    })?;
    if !format!("{:x}", Sha256::digest(&bytes)).eq_ignore_ascii_case(expected) {
        return Err(invalid(format!(
            "source digest mismatch: {}",
            path.display()
        )));
    }
    Ok(bytes)
}

struct HashedFile {
    file: std::fs::File,
    digest: Sha256,
}

impl HashedFile {
    fn open(path: &Path) -> Result<Self, AicError> {
        Ok(Self {
            file: std::fs::File::open(path).map_err(|source| AicError::Io {
                path: path.into(),
                source,
            })?,
            digest: Sha256::new(),
        })
    }

    fn verify(self, path: &Path, expected: &str) -> Result<(), AicError> {
        if !format!("{:x}", self.digest.finalize()).eq_ignore_ascii_case(expected) {
            return Err(invalid(format!(
                "source digest mismatch: {}",
                path.display()
            )));
        }
        Ok(())
    }
}

impl Read for HashedFile {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let count = self.file.read(buffer)?;
        self.digest.update(&buffer[..count]);
        Ok(count)
    }
}

/// Hash the same bytes the parser consumes, with bounded read buffers instead
/// of retaining a full encoded copy alongside the decoded original-A groups.
fn checked_manifest(path: &Path, expected: &str) -> Result<OriginalAManifest, AicError> {
    let mut reader = BufReader::new(HashedFile::open(path)?);
    let manifest = serde_json::from_reader(&mut reader);
    // Finish hashing even when decoding fails, preserving digest validation.
    std::io::copy(&mut reader, &mut std::io::sink()).map_err(|source| AicError::Io {
        path: path.into(),
        source,
    })?;
    reader.into_inner().verify(path, expected)?;
    manifest.map_err(invalid)
}

fn check_source_file(path: &Path, expected: &str) -> Result<(), AicError> {
    let mut reader = HashedFile::open(path)?;
    std::io::copy(&mut reader, &mut std::io::sink()).map_err(|source| AicError::Io {
        path: path.into(),
        source,
    })?;
    reader.verify(path, expected)
}

impl MixedComposition {
    fn analyze(
        config: &ForwardPassPerfModelConfig,
        learned: &LearnedForwardPassModel,
    ) -> Result<Option<Self>, AicError> {
        let Some(policy) = &config.estimator_config.temporary_mixed_composition else {
            return Ok(None);
        };
        let source = &policy.original_a;
        let manifest = checked_manifest(
            &source.manifest_path,
            source
                .source_sha256
                .as_deref()
                .ok_or_else(|| invalid("unresolved original A digest"))?,
        )?;
        let identity = canonical_learned_identity(config)?;
        if manifest.schema_version != 1
            || manifest.source_kind != "self_benchmark"
            || manifest.identity != identity
        {
            return Err(invalid(
                "original A schema, source kind, or deployment identity mismatch",
            ));
        }
        if manifest.min_store_rows == 0
            || manifest.accepted_iterations.is_empty()
            || manifest.seed_artifacts.is_empty()
        {
            return Err(invalid(
                "original A requires accepted groups, a positive training threshold, and seed artifacts",
            ));
        }
        let digest = manifest
            .provenance
            .get("source_sha256")
            .and_then(serde_json::Value::as_str);
        if !digest.is_some_and(|v| v.len() == 64 && v.bytes().all(|c| c.is_ascii_hexdigit()))
            || !manifest.provenance["grouping"].is_object()
        {
            return Err(invalid(
                "original A requires source digest and grouping provenance",
            ));
        }
        let sources: Vec<SeedReference> = serde_json::from_value(
            manifest
                .provenance
                .get("sources")
                .cloned()
                .ok_or_else(|| invalid("original A source file references missing"))?,
        )
        .map_err(invalid)?;
        if sources.is_empty() {
            return Err(invalid("original A source files cannot be empty"));
        }
        for reference in sources {
            let path = if reference.path.is_absolute() {
                reference.path
            } else {
                source
                    .manifest_path
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join(reference.path)
            };
            check_source_file(&path, &reference.sha256)?;
        }
        let score_options = config.estimator_config.regression_options();
        let mut counts = BTreeMap::<Kind, usize>::new();
        let mut bounds = [0.0_f64; 2];
        for group in &manifest.accepted_iterations {
            validate_complete_group(group, config.attention_dp as usize)?;
            if group
                .iter()
                .any(|rank| !rank.wall_time.is_finite() || rank.wall_time < 0.0)
                || max_positive_wall_time_ms(group).is_none()
            {
                return Err(invalid(
                    "accepted original A groups require finite nonnegative rank times and a positive group measurement",
                ));
            }
            let f = RegressionIterationFeatures::from_metrics(
                group,
                config.worker_type,
                &score_options,
            )?
            .ok_or_else(|| invalid("accepted original A group is idle"))?;
            *counts.entry(f.workload_kind).or_default() += 1;
            for (max, value) in bounds.iter_mut().zip(f.x) {
                *max = max.max(value);
            }
        }
        let mut seed_kinds = BTreeSet::new();
        let mut anchor_digests = BTreeSet::new();
        for reference in &manifest.seed_artifacts {
            let path = if reference.path.is_absolute() {
                reference.path.clone()
            } else {
                source
                    .manifest_path
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join(&reference.path)
            };
            let bytes = checked_bytes(&path, &reference.sha256)?;
            let seed =
                LearnedForwardPassModel::from_json(std::str::from_utf8(&bytes).map_err(invalid)?)?;
            if let Some(expected) = &reference.checkpoint_manifest_sha256 {
                let checkpoint = path
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join("manifest.json");
                let bytes = checked_bytes(&checkpoint, expected)?;
                let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(invalid)?;
                if value["schema"] != "aisimulate_fpm_training_checkpoint"
                    || value["artifact_sha256"] != reference.sha256
                    || value["identity"] != identity
                {
                    return Err(invalid(
                        "original A training checkpoint does not bind the seed artifact",
                    ));
                }
                anchor_digests.insert(expected.to_ascii_lowercase());
            }
            if seed.metadata().get("identity") != Some(&identity)
                || seed.worker_type() != config.worker_type
                || seed.feature_names() != learned.feature_names()
                || seed.target() != learned.target()
            {
                return Err(invalid(
                    "original A seed identity, features, or target mismatch",
                ));
            }
            for kind in seed.store_kinds() {
                if !seed_kinds.insert(kind) {
                    return Err(invalid("duplicate original A seed store"));
                }
                let name = serde_json::to_value(kind).map_err(invalid)?;
                let trained =
                    seed.metadata()["train_rows"][name.as_str().expect("kind string")].as_u64();
                let count = counts.get(&kind).copied().unwrap_or(0);
                if count < manifest.min_store_rows || trained != Some(count as u64) {
                    return Err(invalid(format!(
                        "original A group count disagrees with trained store {kind:?}"
                    )));
                }
                if config.estimator_config.fpm_learned.adaptation
                    != super::super::LearnedAdaptation::Online
                    && learned.store(kind) != seed.store(kind)
                {
                    return Err(invalid(
                        "current frozen seed differs from verified original A stores",
                    ));
                }
            }
        }
        for (kind, count) in &counts {
            if *count >= manifest.min_store_rows && !seed_kinds.contains(kind) {
                return Err(invalid(format!(
                    "sufficient original A data but missing trained store {kind:?}"
                )));
            }
        }
        if config.estimator_config.fpm_learned.adaptation != super::super::LearnedAdaptation::Online
            && learned
                .store_kinds()
                .iter()
                .any(|kind| !MIXED_KINDS.contains(kind) && !seed_kinds.contains(kind))
        {
            return Err(invalid(
                "current frozen artifact contains a pure store outside original A",
            ));
        }
        let mut routes = Vec::new();
        for kind in MIXED_KINDS {
            let count = counts.get(&kind).copied().unwrap_or(0);
            let usable = seed_kinds.contains(&kind);
            if usable && !learned.has_store(kind) {
                return Err(invalid("current artifact lost an original A mixed store"));
            }
            routes.push(ClassRoute {
                workload_kind: kind,
                fallback_eligible: !usable,
                accepted_a_groups: count,
                reason: if usable {
                    "usable_original_a_store"
                } else if count == 0 {
                    "missing_original_a_data"
                } else {
                    "insufficient_original_a_data"
                }
                .into(),
            });
        }
        let value = Self {
            policy: policy.clone(),
            routes,
            engine: None,
            score_options,
            bounds,
            corrections: BTreeMap::new(),
            admissions: BTreeMap::new(),
            identity,
            prior: serde_json::Value::Null,
            attention_dp: config.attention_dp as usize,
            anchor_digests,
            quality_gate: policy.quality_gate.as_ref().map(QualityGate::new),
        };
        Ok(Some(value))
    }

    pub(super) fn construct(
        config: &ForwardPassPerfModelConfig,
        learned: &LearnedForwardPassModel,
    ) -> Result<Option<Self>, AicError> {
        let Some(mut value) = Self::analyze(config, learned)? else {
            return Ok(None);
        };
        let policy = &value.policy;
        let bounds = value.bounds;
        if value.routes.iter().any(|r| r.fallback_eligible) {
            // Preserve the compatible canonical interpolation method. Only
            // typed support misses are unavailable; malformed data remains an error.
            let mut prior_config = config.clone();
            prior_config.estimation_mode = EstimationMode::FpmInterpolation;
            prior_config.fallback_policy = ForwardPassFallbackPolicy::Deny;
            prior_config.estimator_config.temporary_mixed_composition = None;
            prior_config.estimator_config.fpm_learned = Default::default();
            prior_config.estimator_config.selector = Default::default();
            prior_config.estimator_config.correction.enabled = false;
            prior_config
                .estimator_config
                .fpm_interpolation
                .collect_coverage = false;
            let prior_model = ForwardPassPerfModel::best_available(prior_config)?;
            let ForwardPassPerfMode::Native { engine, .. } = prior_model.mode else {
                return Err(invalid("composition prior is not native interpolation"));
            };
            let table = &engine.database().fpm_forward;
            let _ = table.cells()?; // Load and validate now, not on the first query.
            // Engine construction resolves the exact deployment and profile;
            // table loading above pins the immutable measured source.
            let path = table.parquet_path();
            let file_digest = |path: &Path| -> Result<String, AicError> {
                let bytes = std::fs::read(path).map_err(|source| AicError::Io {
                    path: path.into(),
                    source,
                })?;
                Ok(format!("{:x}", Sha256::digest(&bytes)))
            };
            value.prior = serde_json::json!({"provenance":prior_model.provenance,"parquet_sha256":file_digest(path)?,"metadata_sha256":file_digest(&path.with_extension("metadata.json"))?});
            value.engine = Some(engine);
            if policy.mode == TemporaryMixedCompositionMode::ComposeCalibrated {
                let axes = [
                    AxisRange::from_zero_to_f64(bounds[0])?,
                    AxisRange::from_zero_to_f64(bounds[1])?,
                ];
                for route in &value.routes {
                    if route.fallback_eligible {
                        value.corrections.insert(
                            route.workload_kind,
                            CorrectionBuckets::new_mixed_hybrid(
                                &policy.calibration.options(),
                                &axes,
                            )?,
                        );
                        value.admissions.insert(route.workload_kind, 0);
                    }
                }
            }
        }
        Ok(Some(value))
    }

    pub(super) fn prefers_ordinary(&self) -> bool {
        self.policy.scenario == TemporaryMixedCompositionScenario::AC
    }

    /// One selected-expert query; the second expert runs only for a typed
    /// support miss. Paired shadow predictions are confined to admitted feedback.
    pub(super) fn predict_quality(
        &self,
        metrics: &[ForwardPassMetrics],
        learned: &LearnedForwardPassModel,
        kind: Kind,
        detailed: bool,
    ) -> Result<MixedSelectorPrediction, AicError> {
        let query = IterationFeatures::from_validated_metrics(metrics)
            .expect("mixed query")
            .workload_kind;
        let preferred = self
            .quality_gate
            .as_ref()
            .expect("enabled quality gate")
            .selected(kind, query);
        let online_ready = learned.has_store(kind);
        if preferred == SelectedExpert::Online && online_ready {
            let online = learned_base_ms(learned, metrics)?;
            return Ok(MixedSelectorPrediction {
                estimate: ForwardPassPerfModel::learned_estimate_details(online),
                selected: SelectedExpert::Online,
                preferred,
                stable_available: None,
                online_available: Some(online.is_some()),
                query,
                kind,
            });
        }
        let feature = self.owned_feature(metrics)?.expect("eligible mixed query");
        let prior = self.predict(metrics, &feature, detailed)?;
        if prior.latency_ms.is_some() || !online_ready {
            return Ok(MixedSelectorPrediction {
                stable_available: Some(prior.latency_ms.is_some()),
                online_available: (!online_ready).then_some(false),
                estimate: prior,
                selected: SelectedExpert::Stable,
                preferred,
                query,
                kind,
            });
        }
        let online = learned_base_ms(learned, metrics)?;
        Ok(MixedSelectorPrediction {
            estimate: ForwardPassPerfModel::learned_estimate_details(online),
            selected: SelectedExpert::Online,
            preferred,
            stable_available: Some(false),
            online_available: Some(online.is_some()),
            query,
            kind,
        })
    }

    pub(super) fn quality_route_metadata(
        &self,
        metrics: &[ForwardPassMetrics],
        prediction: &MixedSelectorPrediction,
        online_ready: bool,
    ) -> Result<serde_json::Value, AicError> {
        let mut route = self.route(metrics, prediction.selected == SelectedExpert::Online)?;
        let object = route.as_object_mut().expect("route object");
        object.extend(
            self.quality_gate
                .as_ref()
                .expect("quality gate")
                .query_diagnostics(prediction.kind, prediction.query)
                .as_object()
                .unwrap()
                .clone(),
        );
        object.extend(serde_json::json!({"selected_expert":prediction.selected.name(), "preferred_expert":prediction.preferred.name(),
            "stable_source":"interpolation", "online_ready":online_ready, "stable_available":prediction.stable_available,
            "online_available":prediction.online_available, "coverage_escape":prediction.preferred == SelectedExpert::Stable && prediction.selected == SelectedExpert::Online}).as_object().unwrap().clone());
        Ok(route)
    }

    pub(super) fn stage_quality(
        &self,
        observation: &PreparedMixedObservation,
        query: WorkloadKind,
        online_ms: f64,
    ) -> Result<QualityUpdate, AicError> {
        self.quality_gate
            .as_ref()
            .expect("enabled quality gate")
            .stage(
                observation.kind,
                query,
                self.staged_prediction(observation)?,
                online_ms,
                observation.observed,
            )
    }

    pub(super) fn eligible_kinds(&self) -> Vec<Kind> {
        self.routes
            .iter()
            .filter(|r| r.fallback_eligible)
            .map(|r| r.workload_kind)
            .collect()
    }

    pub(super) fn eligible_kind(
        &self,
        group: &[ForwardPassMetrics],
    ) -> Result<Option<Kind>, AicError> {
        if self.routes.iter().all(|r| !r.fallback_eligible) {
            return Ok(None);
        }
        let phases = group.iter().fold((false, false), |(p, d), rank| {
            (
                p || rank.scheduled_requests.sum_prefill_tokens > 0,
                d || rank.scheduled_requests.num_decode_requests > 0,
            )
        });
        if phases != (true, true) {
            return Ok(None);
        }
        Ok(
            classify_regression_workload(group, ForwardPassWorkerType::Aggregated)?.filter(
                |kind| {
                    self.routes
                        .iter()
                        .any(|r| r.workload_kind == *kind && r.fallback_eligible)
                },
            ),
        )
    }

    pub(super) fn owned_feature(
        &self,
        group: &[ForwardPassMetrics],
    ) -> Result<Option<RegressionIterationFeatures>, AicError> {
        let Some(kind) = self.eligible_kind(group)? else {
            return Ok(None);
        };
        validate_complete_group(group, self.attention_dp)?;
        RegressionIterationFeatures::from_validated_metrics(
            group,
            ForwardPassWorkerType::Aggregated,
            &self.score_options,
            kind,
        )
        .map(Some)
    }

    /// Calibrated prior from the staged raw query, before this observation is
    /// admitted. Reuse that query for selection and calibration.
    pub(super) fn staged_prediction(
        &self,
        observation: &PreparedMixedObservation,
    ) -> Result<f64, AicError> {
        let factor = self
            .corrections
            .get(&observation.kind)
            .map_or(1.0, |c| c.correction_factor_for(&observation.x));
        let value = observation.raw * factor;
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid(
                "calibrated interpolation prediction must be finite and positive",
            ));
        }
        Ok(value)
    }

    pub(super) fn predict(
        &self,
        group: &[ForwardPassMetrics],
        feature: &RegressionIterationFeatures,
        detailed: bool,
    ) -> Result<crate::ForwardPassEstimate, AicError> {
        let result = self
            .engine
            .as_ref()
            .expect("owned class has interpolation")
            .forward_pass_estimate(group, detailed, None);
        let mut estimate = match result {
            Ok(value) => value,
            Err(
                AicError::FpmQueryUnsupported(reason) | AicError::InterpolationCoverage(reason),
            ) => {
                return Ok(crate::ForwardPassEstimate {
                    unavailable_reason: Some(reason),
                    latency_ms: None,
                    native_latency_ms: None,
                    correction_factor: None,
                    max_rank: None,
                    ranks: Vec::new(),
                });
            }
            Err(error) => return Err(error),
        };
        let factor = self
            .corrections
            .get(&feature.workload_kind)
            .map_or(1.0, |c| c.correction_factor_for(&feature.x));
        estimate.correction_factor = Some(factor);
        estimate.latency_ms = estimate.native_latency_ms.map(|base| base * factor);
        if estimate
            .latency_ms
            .is_some_and(|v| !v.is_finite() || v <= 0.0)
        {
            return Err(AicError::InvalidPerfData(
                "mixed composition produced invalid latency".into(),
            ));
        }
        Ok(estimate)
    }

    pub(super) fn prepare_observation(
        &self,
        group: &[ForwardPassMetrics],
        feature: &RegressionIterationFeatures,
    ) -> Result<Option<PreparedMixedObservation>, AicError> {
        if !self.corrections.contains_key(&feature.workload_kind) {
            return Ok(None);
        }
        if group
            .iter()
            .any(|rank| !rank.wall_time.is_finite() || rank.wall_time < 0.0)
        {
            return Err(AicError::InvalidForwardPassMetrics(
                "mixed feedback rank times must be finite and nonnegative".into(),
            ));
        }
        let Some(observed) = max_positive_wall_time_ms(group) else {
            // Valid all-zero timings carry no label. Match ordinary tuning:
            // skip them before querying the prior or checking admission space.
            return Ok(None);
        };
        let admitted = self.admissions[&feature.workload_kind];
        if admitted >= usize::MAX as u64 {
            return Err(invalid("mixed admission counter exhausted"));
        }
        let raw = match self
            .engine
            .as_ref()
            .expect("owned class prior")
            .forward_pass_time_ms(group)
        {
            Ok(value) => value,
            Err(AicError::FpmQueryUnsupported(_) | AicError::InterpolationCoverage(_)) => {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        if !raw.is_finite() || raw <= 0.0 || !(observed / raw).is_finite() || observed / raw <= 0.0
        {
            return Err(AicError::InvalidPerfData(
                "mixed calibration requires a finite positive raw latency and ratio".into(),
            ));
        }
        Ok(Some(PreparedMixedObservation {
            kind: feature.workload_kind,
            x: feature.x.to_vec(),
            observed,
            raw,
        }))
    }

    pub(super) fn commit_observation(&mut self, observation: PreparedMixedObservation) {
        self.corrections
            .get_mut(&observation.kind)
            .expect("prepared correction")
            .add_observation(observation.x, observation.observed, observation.raw);
        *self
            .admissions
            .get_mut(&observation.kind)
            .expect("prepared counter") += 1;
    }

    #[cfg(test)]
    fn observe(
        &mut self,
        group: &[ForwardPassMetrics],
        feature: &RegressionIterationFeatures,
    ) -> Result<(), AicError> {
        if let Some(observation) = self.prepare_observation(group, feature)? {
            self.commit_observation(observation);
        }
        Ok(())
    }

    pub(super) fn route(
        &self,
        group: &[ForwardPassMetrics],
        ordinary_available: bool,
    ) -> Result<serde_json::Value, AicError> {
        let kind = classify_regression_workload(group, ForwardPassWorkerType::Aggregated)?;
        let fallback_eligible = kind.is_some_and(|kind| {
            self.routes
                .iter()
                .any(|r| r.workload_kind == kind && r.fallback_eligible)
        });
        let owned = fallback_eligible && !(self.prefers_ordinary() && ordinary_available);
        Ok(
            serde_json::json!({"workload_kind":kind,"fallback_eligible":fallback_eligible,"route":if owned {"interpolation"} else {"ordinary"},"owned":owned,"calibration_enabled":fallback_eligible && self.policy.mode == TemporaryMixedCompositionMode::ComposeCalibrated,"version":VERSION,"calibration_admissions":kind.and_then(|kind| self.admissions.get(&kind)).copied().unwrap_or(0)}),
        )
    }

    pub(super) fn diagnostics(&self) -> serde_json::Value {
        serde_json::json!({"version":VERSION,"selection_policy":if self.quality_gate.is_some() {"paired_ape_quality_gate"} else if self.prefers_ordinary() {"ordinary_when_available"} else {"original_a_owned"},"routes":self.routes,"policy":self.policy,"score_bounds":self.bounds,"feature_weights":[self.score_options.regression_attention_kv_weight,self.score_options.regression_prefill_attention_pair_weight,self.score_options.regression_ffn_token_weight],"interpolation":self.prior,"admissions":self.admissions,"stores":self.corrections.iter().map(|(kind,c)| (kind,serde_json::json!({"retained_observations":c.observation_count(),"ready_buckets":c.ready_bucket_count()}))).collect::<BTreeMap<_,_>>()})
    }

    pub(super) fn retained_observations(&self) -> usize {
        self.corrections
            .values()
            .map(StoreStats::observation_count)
            .sum()
    }

    pub(super) fn ready_bucket_count(&self) -> usize {
        self.corrections
            .values()
            .map(CorrectionBuckets::ready_bucket_count)
            .sum()
    }

    fn binding(&self) -> serde_json::Value {
        serde_json::json!({"identity":self.identity,"policy":self.policy,"routes":self.routes,"bounds":self.bounds,"prior":self.prior,"weights":[self.score_options.regression_attention_kv_weight,self.score_options.regression_prefill_attention_pair_weight,self.score_options.regression_ffn_token_weight]})
    }

    pub(super) fn verify_online_anchor(&self, digest: &str) -> Result<(), AicError> {
        if !self.anchor_digests.contains(&digest.to_ascii_lowercase()) {
            return Err(invalid(
                "online checkpoint anchor is not a verified original A training checkpoint",
            ));
        }
        Ok(())
    }

    pub(super) fn snapshot(&self) -> Result<serde_json::Value, AicError> {
        let corrections = self
            .corrections
            .iter()
            .map(|(kind, c)| Ok((*kind, c.snapshot_mixed_hybrid()?)))
            .collect::<Result<_, AicError>>()?;
        serde_json::to_value(MixedSnapshot {
            version: VERSION.into(),
            binding: self.binding(),
            admissions: self.admissions.clone(),
            corrections,
            quality_gate: self.quality_gate.clone(),
        })
        .map_err(invalid)
    }

    pub(super) fn restore(&mut self, value: serde_json::Value) -> Result<(), AicError> {
        let snapshot: MixedSnapshot = serde_json::from_value(value).map_err(invalid)?;
        if snapshot.version != VERSION
            || snapshot.binding != self.binding()
            || !snapshot.corrections.keys().eq(self.corrections.keys())
            || !snapshot.admissions.keys().eq(self.admissions.keys())
        {
            return Err(invalid(
                "saved mixed state does not match original A, prior, routes, geometry, or calibration settings",
            ));
        }
        match (&snapshot.quality_gate, &self.policy.quality_gate) {
            (Some(gate), Some(config)) => gate.validate_snapshot(config, &snapshot.admissions)?,
            (None, None) => {}
            _ => return Err(invalid("saved quality gate presence differs")),
        }
        let mut restored = self.corrections.clone();
        for (kind, correction) in snapshot.corrections {
            let store = restored.get_mut(&kind).expect("validated snapshot keys");
            store.restore_mixed_hybrid(correction)?;
            if store.mixed_admission_count()? != snapshot.admissions[&kind] {
                return Err(invalid(
                    "mixed admission progress disagrees with calibration state",
                ));
            }
        }
        self.corrections = restored;
        self.admissions = snapshot.admissions;
        self.quality_gate = snapshot.quality_gate;
        Ok(())
    }
}

fn validate_complete_group(
    group: &[ForwardPassMetrics],
    attention_dp: usize,
) -> Result<(), AicError> {
    if group.len() == attention_dp
        && group
            .iter()
            .enumerate()
            .all(|(index, rank)| rank.dp_rank as usize == index)
    {
        return Ok(());
    }
    if group.len() != attention_dp
        || group
            .iter()
            .map(|r| r.dp_rank as usize)
            .collect::<BTreeSet<_>>()
            != (0..attention_dp).collect()
    {
        return Err(AicError::InvalidForwardPassMetrics("mixed policy requires one metric for every attention-DP rank, with unique complete rank IDs".into()));
    }
    Ok(())
}

#[cfg(test)]
#[path = "mixed_fallback_tests.rs"]
mod tests;
