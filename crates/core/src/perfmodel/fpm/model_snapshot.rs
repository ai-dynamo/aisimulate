// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Save a tuned forward-pass model to a directory and restore it through
//! `best_available`.
//!
//! Saving captures every field that `tune_with_fpms` mutates, so a restored
//! model predicts bit-identically and continues tuning exactly as the saving
//! model would. Configuration-derived fields are rebuilt from the canonical
//! configuration and compared, never trusted from the file. Native engines
//! and learned forests are reconstructed through the ordinary estimator
//! construction; the snapshot pins what they were built from. The format is
//! described in [`super::super::snapshot`].

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::super::estimator::{
    CorrectionConfig, CorrectionFactorBounds, SamplingConfig, StateSnapshotConfig,
    StateSnapshotTuning,
};
use super::super::learned::LearnedForwardPassModel;
use super::super::samples::integer_sqrt;
use super::super::selector::{Evidence, KEY_COUNT};
use super::super::snapshot::{
    self, CONFIG_FILE, DiagnosticsRecord, EXECUTION_PROFILE_FILE, LEARNED_ARTIFACT_FILE,
    LearnedRecord, MANIFEST_FILE, Manifest, NativeRecord, ONLINE_CHECKPOINT_DIR, Producer,
    STABLE_ARTIFACT_FILE, STATE_FILE, STATE_SNAPSHOT_LAYOUT_VERSION, Stage, StateSnapshotKind,
    invalid, perf_db,
};
use super::*;

/// Options for [`ForwardPassPerfModel::to_file`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StateSnapshotOptions {
    /// Copy the performance data a native estimator reads into the snapshot
    /// and point the pinned configuration's `systems_paths` at the copy.
    /// Otherwise the snapshot references the original root by fingerprint.
    pub include_perf_db: bool,
}

/// Manifest summary of a saved snapshot, read without restoring the model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ForwardPassStateSnapshotSummary {
    pub manifest_sha256: String,
    pub kind: StateSnapshotKind,
    pub aisimulate_core_version: String,
    pub state_layout_version: u32,
    pub worker_type: ForwardPassWorkerType,
    pub selected_estimation_mode: EstimationMode,
    /// Pinned canonical identity, without `state_snapshot`. `None` for a
    /// `from_learned` model, which has no deployment identity.
    pub identity: Option<ForwardPassPerfModelConfig>,
    pub topology: Option<ForwardPassDeploymentTopology>,
    /// Readiness recorded when the snapshot was saved; restoration verifies
    /// that the restored model reports the same.
    pub readiness: ForwardPassReadinessReport,
}

/// Every mutable estimator field, in a fixed positional layout.
#[derive(Clone, Serialize, Deserialize)]
struct StateFile {
    layout_version: u32,
    state: SnapshotState,
}

#[derive(Clone, Serialize, Deserialize)]
enum SnapshotState {
    Native {
        corrections: WorkloadStores<CorrectionBuckets>,
    },
    Regression {
        stores: RegressionStores,
    },
    /// Frozen, online and `from_learned` forests share this variant; online
    /// learner state lives in the snapshot's online checkpoint.
    Learned {
        corrections: WorkloadStores<CorrectionBuckets>,
    },
    CausalSelector {
        corrections: Vec<(
            ForwardPassRegressionWorkloadKind,
            WorkloadStores<CorrectionBuckets>,
        )>,
        selector: [Evidence; KEY_COUNT],
    },
}

/// Configuration of a `from_learned` snapshot. Identity fields are fixed
/// placeholders: such a model was never bound to a deployment, so restoration
/// performs no identity checks and only these estimator controls must match.
fn from_learned_config(
    worker_type: ForwardPassWorkerType,
    options: &ForwardPassPerfOptions,
    artifact_path: PathBuf,
) -> ForwardPassPerfModelConfig {
    let mut config = ForwardPassPerfModelConfig::new(
        "from_learned",
        "from_learned",
        crate::BackendKind::Vllm,
        worker_type,
    );
    config.estimation_mode = EstimationMode::FpmLearned;
    let learned = &mut config.estimator_config.fpm_learned;
    learned.artifact_path = Some(artifact_path);
    learned.adaptation = super::super::LearnedAdaptation::Frozen;
    let bins = options.bucket_shape.unwrap_or_else(|| {
        let side = integer_sqrt(options.bucket_count);
        [side, side]
    });
    config.estimator_config.correction = CorrectionConfig {
        enabled: true,
        mode: match options.correction_mode {
            CorrectionMode::Auto => CorrectionMode::Hybrid,
            mode => mode,
        },
        sampling: SamplingConfig {
            bins_per_axis: bins,
            max_observations: options.max_observations,
        },
        min_observations: options.min_observations,
        factor_bounds: CorrectionFactorBounds {
            min: options.min_faster_correction_factor,
            max: options.max_slower_correction_factor,
        },
        max_num_tokens: options.max_num_tokens,
        max_batch_size: options.max_batch_size,
        max_kv_tokens: options.max_kv_tokens,
        ..CorrectionConfig::default()
    };
    config
}

fn unpinned(config: &ForwardPassPerfModelConfig) -> ForwardPassPerfModelConfig {
    let mut config = config.clone();
    config.estimator_config.state_snapshot = None;
    config
}

/// First differing JSON path, for actionable identity-mismatch errors.
fn first_difference(saved: &serde_json::Value, supplied: &serde_json::Value) -> String {
    match (saved, supplied) {
        (serde_json::Value::Object(left), serde_json::Value::Object(right)) => {
            for key in left.keys().chain(right.keys()) {
                if left.get(key) != right.get(key) {
                    return match (left.get(key), right.get(key)) {
                        (Some(left), Some(right)) => {
                            let rest = first_difference(left, right);
                            if rest.is_empty() {
                                key.clone()
                            } else {
                                format!("{key}.{rest}")
                            }
                        }
                        _ => key.clone(),
                    };
                }
            }
            String::new()
        }
        _ => String::new(),
    }
}

fn identity_mismatch(
    saved: &ForwardPassPerfModelConfig,
    supplied: &ForwardPassPerfModelConfig,
) -> AicError {
    let location = match (serde_json::to_value(saved), serde_json::to_value(supplied)) {
        (Ok(saved), Ok(supplied)) => first_difference(&saved, &supplied),
        _ => String::new(),
    };
    invalid(format!(
        "configuration differs from the snapshot's pinned identity at {}",
        if location.is_empty() {
            "an unserialized field"
        } else {
            &location
        }
    ))
}

impl ForwardPassPerfModel {
    /// API:
    /// `model.to_file(path, options) -> Result<ForwardPassPerfModelConfig, AicError>`
    ///
    /// Description: atomically write this model's complete tuned state to the
    /// new directory `path` and return the pinned canonical configuration
    /// that restores it through [`Self::best_available`]. The same
    /// configuration is written to `<path>/config.json`.
    ///
    /// The directory must not exist; it is staged next to its destination and
    /// renamed into place. Saving never fits pending online rows or changes
    /// the model. The returned configuration is the model's resolved
    /// provenance configuration with snapshot-owned files (learned artifacts,
    /// online checkpoints, execution profiles and, with `include_perf_db`,
    /// performance data) redirected into the snapshot, plus
    /// `estimator_config.state_snapshot` pinned to the manifest digest with
    /// `on_tune: reject`. Set `on_tune: continue` to resume tuning.
    ///
    /// Native estimators record the compiled engine-spec digest from
    /// construction and the fingerprint of their performance data taken when
    /// this process first loaded it; saving fails if that data changed since,
    /// and restoring recompiles and rejects any difference. Snapshot I/O
    /// failures are [`AicError::StateSnapshotIo`], never missing data. Models from
    /// [`Self::from_learned`] have no deployment identity: their snapshot
    /// configuration carries placeholder identity fields and restoration
    /// checks only estimator controls and file digests.
    pub fn to_file(
        &self,
        path: impl AsRef<Path>,
        options: &StateSnapshotOptions,
    ) -> Result<ForwardPassPerfModelConfig, AicError> {
        self.write_snapshot(path.as_ref(), options)
            .map_err(snapshot::as_snapshot_io)
    }

    fn write_snapshot(
        &self,
        path: &Path,
        options: &StateSnapshotOptions,
    ) -> Result<ForwardPassPerfModelConfig, AicError> {
        let mut stage = Stage::new(path)?;
        let destination = stage.destination().to_path_buf();
        let kind = self.snapshot_kind();
        let mut native = None;
        let mut learned_record = None;
        let mut identity = match (&self.provenance, kind) {
            (Some(provenance), _) => Some(unpinned(&provenance.config)),
            (None, StateSnapshotKind::LearnedWithoutIdentity) => None,
            (None, _) => {
                return Err(invalid(
                    "the model has no construction provenance; build it with best_available",
                ));
            }
        };
        match &self.mode {
            ForwardPassPerfMode::Native { engine, .. } => {
                let identity = identity.as_mut().expect("native provenance");
                let root = self
                    .provenance
                    .as_ref()
                    .and_then(|provenance| provenance.selected_systems_root.clone())
                    .ok_or_else(|| invalid("native provenance has no selected systems root"))?;
                let parquet = identity
                    .estimator_config
                    .fpm_interpolation
                    .fpm_parquet_path
                    .clone();
                // The fingerprint describes the data this process loaded; a
                // save after the data changed would pair tuned state with
                // data it was never tuned on.
                let loaded = match self.data_fingerprint().map(|loaded| loaded.as_ref()) {
                    Some(Ok(loaded)) => loaded.clone(),
                    Some(Err(error)) => {
                        return Err(invalid(format!(
                            "the performance data under {} could not be fingerprinted when it was loaded: {error}",
                            root.display()
                        )));
                    }
                    None => {
                        return Err(invalid(
                            "the native model has no load-time performance-data fingerprint; build it with best_available",
                        ));
                    }
                };
                let current = perf_db::fingerprint(&root, &identity.system, parquet.as_deref())?;
                if current != loaded {
                    return Err(invalid(format!(
                        "performance data under {} changed after this process loaded it; restart from the current data before saving",
                        root.display()
                    )));
                }
                if options.include_perf_db {
                    let (packed_root, packed_parquet) =
                        perf_db::pack(&stage, &root, &identity.system, parquet.as_deref())?;
                    let staged = stage.staging().join(perf_db::PACKED_SYSTEMS_DIR);
                    let staged_parquet = packed_parquet.as_ref().map(|path| {
                        stage
                            .staging()
                            .join(path.strip_prefix(&destination).expect("packed path"))
                    });
                    let copied =
                        perf_db::fingerprint(&staged, &identity.system, staged_parquet.as_deref())?;
                    if copied != loaded {
                        return Err(invalid("packed performance data differs from its source"));
                    }
                    identity.systems_paths = vec![packed_root];
                    if packed_parquet.is_some() {
                        identity.estimator_config.fpm_interpolation.fpm_parquet_path =
                            packed_parquet;
                    }
                }
                if let Some(selection) = identity
                    .estimator_config
                    .op_level
                    .execution_profile
                    .as_mut()
                {
                    let bytes = std::fs::read(&selection.path)
                        .map_err(|error| snapshot::io_error(&selection.path, error))?;
                    if snapshot::sha256_hex(&bytes) != selection.sha256 {
                        return Err(invalid("execution profile changed after construction"));
                    }
                    stage.write_recorded(EXECUTION_PROFILE_FILE, &bytes)?;
                    selection.path = destination.join(EXECUTION_PROFILE_FILE);
                }
                native = Some(NativeRecord {
                    engine_spec_sha256: engine.spec_sha256().to_owned(),
                    perf_db: loaded,
                    packed: options.include_perf_db,
                });
            }
            ForwardPassPerfMode::Regression { .. } => {}
            ForwardPassPerfMode::Learned {
                learned,
                online,
                identity_verified,
                ..
            } => {
                match identity.as_mut() {
                    None => {
                        stage
                            .write_recorded(LEARNED_ARTIFACT_FILE, learned.to_json()?.as_bytes())?;
                        learned_record = Some(LearnedRecord {
                            identity_verified: *identity_verified,
                            options: Some(self.options.clone()),
                        });
                    }
                    Some(identity) => {
                        let settings = &mut identity.estimator_config.fpm_learned;
                        if let Some(online) = online {
                            online.save_source_free(
                                &stage.staging().join(ONLINE_CHECKPOINT_DIR),
                                learned,
                            )?;
                            let record = stage.record_existing(&format!(
                                "{ONLINE_CHECKPOINT_DIR}/manifest.json"
                            ))?;
                            settings.checkpoint_path =
                                Some(destination.join(ONLINE_CHECKPOINT_DIR));
                            settings.artifact_path = None;
                            settings.source_sha256 = Some(record.sha256);
                        } else {
                            // Keep a supplied artifact's exact bytes; a forest
                            // loaded from a checkpoint is exported unchanged.
                            let bytes = match &settings.artifact_path {
                                Some(path) => {
                                    let bytes = std::fs::read(path)
                                        .map_err(|error| snapshot::io_error(path, error))?;
                                    if settings.source_sha256.as_deref().is_none_or(|digest| {
                                        !digest.eq_ignore_ascii_case(&snapshot::sha256_hex(&bytes))
                                    }) {
                                        return Err(invalid(
                                            "learned artifact changed after construction",
                                        ));
                                    }
                                    bytes
                                }
                                None => learned.to_json()?.into_bytes(),
                            };
                            stage.write_recorded(LEARNED_ARTIFACT_FILE, &bytes)?;
                            settings.artifact_path = Some(destination.join(LEARNED_ARTIFACT_FILE));
                            settings.checkpoint_path = None;
                            settings.source_sha256 = Some(snapshot::sha256_hex(&bytes));
                        }
                        learned_record = Some(LearnedRecord {
                            identity_verified: *identity_verified,
                            options: None,
                        });
                    }
                }
            }
            ForwardPassPerfMode::CausalSelector(model) => {
                let identity = identity.as_mut().expect("selector provenance");
                stage.write_recorded(STABLE_ARTIFACT_FILE, model.stable.to_json()?.as_bytes())?;
                model.online.save_source_free(
                    &stage.staging().join(ONLINE_CHECKPOINT_DIR),
                    &model.learned,
                )?;
                let record =
                    stage.record_existing(&format!("{ONLINE_CHECKPOINT_DIR}/manifest.json"))?;
                let settings = &mut identity.estimator_config.fpm_learned;
                settings.checkpoint_path = Some(destination.join(ONLINE_CHECKPOINT_DIR));
                settings.artifact_path = None;
                settings.source_sha256 = Some(record.sha256);
                learned_record = Some(LearnedRecord {
                    identity_verified: model.identity_verified,
                    options: None,
                });
            }
        }

        let state = StateFile {
            layout_version: STATE_SNAPSHOT_LAYOUT_VERSION,
            state: self.snapshot_state(),
        };
        let bytes = snapshot::encode_state(&state)?;
        // Prove the written bytes restore before publishing them.
        let mut decoded: StateFile = snapshot::decode_state(&bytes)?;
        self.check_state(&mut decoded.state)?;
        stage.write_recorded(STATE_FILE, &bytes)?;

        let mut pinned = match identity.clone() {
            Some(identity) => identity,
            None => {
                // Resolve against the staged artifact, then record the final path.
                let mut config = from_learned_config(
                    self.learned_worker_type()
                        .expect("from_learned snapshots hold a forest"),
                    &self.options,
                    stage.staging().join(LEARNED_ARTIFACT_FILE),
                )
                .resolve()?;
                config.estimator_config.fpm_learned.artifact_path =
                    Some(destination.join(LEARNED_ARTIFACT_FILE));
                config
            }
        };
        let diagnostics = self.diagnostics();
        let manifest = Manifest {
            schema: snapshot::SCHEMA.into(),
            schema_version: snapshot::SCHEMA_VERSION,
            state_layout_version: STATE_SNAPSHOT_LAYOUT_VERSION,
            producer: Producer::current(),
            kind,
            worker_type: pinned.worker_type,
            selected_estimation_mode: pinned.estimation_mode,
            identity,
            native,
            learned: learned_record,
            diagnostics: diagnostics_record(&diagnostics),
            readiness: self.readiness_report(),
            files: stage.take_files(),
        };
        let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|error| invalid(format!("cannot encode manifest: {error}")))?;
        manifest_bytes.push(b'\n');
        stage.write(MANIFEST_FILE, &manifest_bytes)?;
        pinned.estimator_config.state_snapshot = Some(StateSnapshotConfig {
            path: destination,
            sha256: Some(snapshot::sha256_hex(&manifest_bytes)),
            on_tune: StateSnapshotTuning::Reject,
        });
        let mut config_bytes = serde_json::to_vec_pretty(&pinned)
            .map_err(|error| invalid(format!("cannot encode config: {error}")))?;
        config_bytes.push(b'\n');
        stage.write(CONFIG_FILE, &config_bytes)?;
        stage.publish()?;
        Ok(pinned)
    }

    /// API:
    /// `ForwardPassPerfModel::inspect_state_snapshot(path) -> Result<ForwardPassStateSnapshotSummary, AicError>`
    ///
    /// Description: read a snapshot manifest without constructing a model,
    /// for readiness and topology gates. Requires the same format versions
    /// as restoration; it does not verify the state file or performance data.
    pub fn inspect_state_snapshot(
        path: impl AsRef<Path>,
    ) -> Result<ForwardPassStateSnapshotSummary, AicError> {
        let bytes = snapshot::read_manifest_bytes(path.as_ref())?;
        let manifest = Manifest::parse(&bytes)?;
        Ok(ForwardPassStateSnapshotSummary {
            manifest_sha256: snapshot::sha256_hex(&bytes),
            kind: manifest.kind,
            aisimulate_core_version: manifest.producer.aisimulate_core_version,
            state_layout_version: manifest.state_layout_version,
            worker_type: manifest.worker_type,
            selected_estimation_mode: manifest.selected_estimation_mode,
            topology: manifest
                .identity
                .as_ref()
                .map(ForwardPassDeploymentTopology::from_config),
            identity: manifest.identity,
            readiness: manifest.readiness,
        })
    }

    fn snapshot_kind(&self) -> StateSnapshotKind {
        match &self.mode {
            ForwardPassPerfMode::Native { .. } => StateSnapshotKind::Native,
            ForwardPassPerfMode::Regression { .. } => StateSnapshotKind::Regression,
            ForwardPassPerfMode::Learned {
                online: Some(_), ..
            } => StateSnapshotKind::LearnedOnline,
            ForwardPassPerfMode::Learned { .. } if self.provenance.is_none() => {
                StateSnapshotKind::LearnedWithoutIdentity
            }
            ForwardPassPerfMode::Learned { .. } => StateSnapshotKind::LearnedFrozen,
            ForwardPassPerfMode::CausalSelector(_) => StateSnapshotKind::CausalSelector,
        }
    }

    fn learned_worker_type(&self) -> Option<ForwardPassWorkerType> {
        match &self.mode {
            ForwardPassPerfMode::Learned { learned, .. } => Some(learned.worker_type()),
            ForwardPassPerfMode::CausalSelector(model) => Some(model.stable.worker_type()),
            _ => None,
        }
    }

    fn snapshot_state(&self) -> SnapshotState {
        match &self.mode {
            ForwardPassPerfMode::Native { corrections, .. } => SnapshotState::Native {
                corrections: corrections.clone(),
            },
            ForwardPassPerfMode::Regression { regression, .. } => SnapshotState::Regression {
                stores: regression.clone(),
            },
            ForwardPassPerfMode::Learned { corrections, .. } => SnapshotState::Learned {
                corrections: corrections.clone(),
            },
            ForwardPassPerfMode::CausalSelector(model) => SnapshotState::CausalSelector {
                corrections: model.correction_stores().to_vec(),
                selector: model.selector_states(),
            },
        }
    }

    /// Validate decoded state against this model's configured stores and
    /// adopt configuration-only fields into it. Does not change the model.
    fn check_state(&self, state: &mut SnapshotState) -> Result<(), AicError> {
        match (&self.mode, state) {
            (
                ForwardPassPerfMode::Native { corrections, .. },
                SnapshotState::Native { corrections: saved },
            )
            | (
                ForwardPassPerfMode::Learned { corrections, .. },
                SnapshotState::Learned { corrections: saved },
            ) => saved.validate_restored(corrections).map_err(invalid),
            (
                ForwardPassPerfMode::Regression { regression, .. },
                SnapshotState::Regression { stores },
            ) => {
                // Store kinds and order must match before any lookup by kind.
                if stores.stores.len() != regression.stores.len()
                    || stores
                        .stores
                        .iter()
                        .zip(&regression.stores)
                        .any(|((saved, _), (fresh, _))| saved != fresh)
                {
                    return Err(invalid(
                        "regression store kinds differ from the worker role's stores",
                    ));
                }
                for ((kind, saved), (_, fresh)) in stores.stores.iter_mut().zip(&regression.stores)
                {
                    saved
                        .restore_snapshot(fresh)
                        .map_err(|error| invalid(format!("{kind:?} regression store: {error}")))?;
                }
                Ok(())
            }
            (
                ForwardPassPerfMode::CausalSelector(model),
                SnapshotState::CausalSelector {
                    corrections,
                    selector,
                },
            ) => model
                .validate_snapshot(corrections, selector)
                .map_err(invalid),
            _ => Err(invalid(
                "saved state does not match the configured estimator",
            )),
        }
    }

    fn install_state(&mut self, mut state: SnapshotState) -> Result<(), AicError> {
        self.check_state(&mut state)?;
        match (&mut self.mode, state) {
            (
                ForwardPassPerfMode::Native { corrections, .. },
                SnapshotState::Native { corrections: saved },
            )
            | (
                ForwardPassPerfMode::Learned { corrections, .. },
                SnapshotState::Learned { corrections: saved },
            ) => *corrections = saved,
            (
                ForwardPassPerfMode::Regression { regression, .. },
                SnapshotState::Regression { stores },
            ) => {
                *regression = stores;
            }
            (
                ForwardPassPerfMode::CausalSelector(model),
                SnapshotState::CausalSelector {
                    corrections,
                    selector,
                },
            ) => model.install_snapshot(corrections, selector),
            _ => {
                return Err(invalid(
                    "saved state does not match the configured estimator",
                ));
            }
        }
        Ok(())
    }

    fn set_identity_verified(&mut self, verified: bool) {
        match &mut self.mode {
            ForwardPassPerfMode::Learned {
                identity_verified, ..
            } => *identity_verified = verified,
            ForwardPassPerfMode::CausalSelector(model) => model.set_identity_verified(verified),
            _ => {}
        }
    }
}

fn diagnostics_record(diagnostics: &ForwardPassPerfDiagnostics) -> DiagnosticsRecord {
    DiagnosticsRecord {
        source: diagnostics.source.clone(),
        ready: diagnostics.readiness == ForwardPassPerfReadiness::Ready,
        retained_observations: diagnostics.retained_observations,
        correction_ready_buckets: diagnostics.correction_ready_buckets,
    }
}

/// Rebuild the causal selector's experts from the snapshot: the frozen expert
/// from `learned/stable.json` and the adaptive expert with its learner from
/// the runtime checkpoint named by the configuration.
fn causal_selector_base(
    config: &ForwardPassPerfModelConfig,
    stable_bytes: &[u8],
) -> Result<ForwardPassPerfModel, AicError> {
    let settings = &config.estimator_config.fpm_learned;
    let checkpoint = settings
        .checkpoint_path
        .as_ref()
        .ok_or_else(|| invalid("causal selector snapshot requires checkpoint_path"))?;
    let mut learned = checkpoint_model_for_config(checkpoint, config)?;
    let text = std::str::from_utf8(stable_bytes)
        .map_err(|error| invalid(format!("stable forest is not UTF-8: {error}")))?;
    let stable = LearnedForwardPassModel::from_json(text)?;
    if learned.worker_type() != config.worker_type
        || stable.worker_type() != learned.worker_type()
        || stable.feature_names() != learned.feature_names()
        || stable.store_kinds() != learned.store_kinds()
        || stable.target() != learned.target()
    {
        return Err(invalid(
            "frozen and adaptive experts do not share one worker role, feature ABI and store set",
        ));
    }
    if let Some(identity) = stable.metadata().get("identity") {
        if identity != &canonical_learned_identity(config)? {
            return Err(invalid(
                "frozen expert metadata.identity does not match the resolved deployment identity",
            ));
        }
    }
    let online = OnlineLearner::load_combined_runtime(checkpoint, &mut learned, settings, config)?;
    let options = config.estimator_config.correction_options();
    let combined = CalibratedOnlineModel::new(
        stable,
        learned,
        online,
        true,
        &options,
        &config.estimator_config.selector,
    )?;
    Ok(ForwardPassPerfModel {
        mode: ForwardPassPerfMode::CausalSelector(Box::new(combined)),
        options,
        last_warning: None,
        provenance: None,
        is_correction_enabled: config.estimator_config.correction.enabled,
        query_coverage: None,
        snapshot_tuning: None,
    })
}

type Select<'a> =
    &'a dyn Fn(ForwardPassPerfModelConfig, Option<bool>) -> Result<ForwardPassPerfModel, AicError>;

/// Restore the model pinned by `config.estimator_config.state_snapshot`.
/// `select` builds the estimator for a resolved configuration without the
/// snapshot (`ForwardPassPerfModel::select` outside tests).
pub(super) fn restore(
    config: ForwardPassPerfModelConfig,
    registered: Option<bool>,
    select: Select<'_>,
) -> Result<ForwardPassPerfModel, AicError> {
    restore_verified(config, registered, select).map_err(snapshot::as_snapshot_io)
}

fn restore_verified(
    config: ForwardPassPerfModelConfig,
    registered: Option<bool>,
    select: Select<'_>,
) -> Result<ForwardPassPerfModel, AicError> {
    let source = config
        .estimator_config
        .state_snapshot
        .clone()
        .ok_or_else(|| invalid("configuration has no state_snapshot"))?;
    let root = source.path.clone();
    let manifest_bytes = snapshot::read_manifest_bytes(&root)?;
    if source.sha256.as_deref().is_none_or(|expected| {
        !expected.eq_ignore_ascii_case(&snapshot::sha256_hex(&manifest_bytes))
    }) {
        return Err(invalid("manifest.json changed after normalization"));
    }
    let manifest = Manifest::parse(&manifest_bytes)?;
    let base = unpinned(&config);
    if base.worker_type != manifest.worker_type
        || base.estimation_mode != manifest.selected_estimation_mode
    {
        return Err(invalid(
            "worker_type or estimation_mode differs from the saved model",
        ));
    }
    match &manifest.identity {
        Some(identity) if *identity != base => return Err(identity_mismatch(identity, &base)),
        Some(_) => {}
        None => {
            let options = manifest
                .learned
                .as_ref()
                .and_then(|learned| learned.options.as_ref())
                .ok_or_else(|| invalid("from_learned snapshot has no constructor options"))?;
            let expected = from_learned_config(
                manifest.worker_type,
                options,
                root.join(LEARNED_ARTIFACT_FILE),
            )
            .resolve()?;
            if expected != base {
                return Err(identity_mismatch(&expected, &base));
            }
        }
    }
    let expected_mode = match manifest.kind {
        StateSnapshotKind::Native => matches!(
            base.estimation_mode,
            EstimationMode::OpLevel | EstimationMode::FpmInterpolation
        ),
        StateSnapshotKind::Regression => base.estimation_mode == EstimationMode::FpmRegression,
        _ => base.estimation_mode == EstimationMode::FpmLearned,
    };
    if !expected_mode || (manifest.native.is_some() != (manifest.kind == StateSnapshotKind::Native))
    {
        return Err(invalid("snapshot kind contradicts its configuration"));
    }
    // Verify every pinned file before any of it is used.
    let mut stable_bytes = None;
    let mut state_bytes = None;
    let mut learned_bytes = None;
    for record in &manifest.files {
        let bytes = snapshot::read_recorded(&root, record)?;
        match record.path.as_str() {
            STATE_FILE => state_bytes = Some(bytes),
            STABLE_ARTIFACT_FILE => stable_bytes = Some(bytes),
            LEARNED_ARTIFACT_FILE => learned_bytes = Some(bytes),
            _ => {}
        }
    }
    let state_bytes = state_bytes.ok_or_else(|| invalid("manifest does not list state.bin"))?;
    let state: StateFile = snapshot::decode_state(&state_bytes)?;
    if state.layout_version != STATE_SNAPSHOT_LAYOUT_VERSION {
        return Err(invalid(
            "state.bin layout version differs from its manifest",
        ));
    }

    // Verify referenced performance data before any of it is loaded: a
    // changed file must report the fingerprint, not a loader failure. Tables
    // load lazily on first query, so no later hash here would cover every
    // read; `to_file` hashes again and rejects data changed after this check.
    let parquet = base
        .estimator_config
        .fpm_interpolation
        .fpm_parquet_path
        .as_deref();
    let native_root = match (&manifest.native, base.systems_paths.as_slice()) {
        (None, _) => None,
        (Some(native), [root]) => {
            if perf_db::current(root, &base.system, parquet)? != native.perf_db {
                return Err(invalid(format!(
                    "performance data under {} differs from the snapshot fingerprint",
                    root.display()
                )));
            }
            Some(root.clone())
        }
        (Some(_), _) => {
            return Err(invalid(
                "a native snapshot configuration must pin exactly one systems path",
            ));
        }
    };

    let mut model = match manifest.kind {
        StateSnapshotKind::LearnedWithoutIdentity => {
            let options = manifest
                .learned
                .as_ref()
                .and_then(|learned| learned.options.clone())
                .expect("checked above");
            // Build from the verified bytes; reopening the path would let a
            // concurrent writer substitute an unverified artifact.
            let artifact = learned_bytes
                .as_deref()
                .ok_or_else(|| invalid("manifest does not list the learned artifact"))?;
            let artifact = std::str::from_utf8(artifact)
                .map_err(|error| invalid(format!("learned artifact is not UTF-8: {error}")))?;
            if !artifact.trim_start().starts_with('{') {
                return Err(invalid("learned artifact is not a JSON object"));
            }
            ForwardPassPerfModel::from_learned(artifact, options)?
        }
        StateSnapshotKind::CausalSelector => causal_selector_base(
            &base,
            stable_bytes
                .as_deref()
                .ok_or_else(|| invalid("manifest does not list the frozen expert"))?,
        )?,
        _ => select(base.clone(), registered)?,
    };
    if model.snapshot_kind() != manifest.kind
        && !(manifest.kind == StateSnapshotKind::CausalSelector
            && matches!(model.mode, ForwardPassPerfMode::CausalSelector(_)))
    {
        return Err(invalid(format!(
            "configuration constructed a {:?} model for a {:?} snapshot",
            model.snapshot_kind(),
            manifest.kind
        )));
    }
    if let Some(native) = &manifest.native {
        let engine = model
            .native_engine()
            .ok_or_else(|| invalid("native snapshot did not construct a native engine"))?;
        if engine.spec_sha256() != native.engine_spec_sha256 {
            return Err(invalid(
                "the recompiled engine specification differs from the saved model's",
            ));
        }
        let selected = model
            .provenance
            .as_ref()
            .and_then(|provenance| provenance.selected_systems_root.clone());
        if selected != native_root {
            return Err(invalid("native model selected a different systems root"));
        }
        // This process may have loaded the root before its data changed;
        // only data it loaded unchanged can carry the saved state.
        let loaded = model
            .data_fingerprint()
            .and_then(|loaded| loaded.as_ref().as_ref().ok());
        if loaded != Some(&native.perf_db) {
            return Err(invalid(format!(
                "this process loaded performance data under {} before it changed; restore in a new process",
                selected.as_deref().unwrap_or(Path::new("?")).display()
            )));
        }
    }
    model.install_state(state.state)?;
    if let Some(learned) = &manifest.learned {
        model.set_identity_verified(learned.identity_verified);
    }
    match model.provenance.as_mut() {
        Some(provenance) => provenance.config = config.clone(),
        None if manifest.kind == StateSnapshotKind::CausalSelector => {
            model.provenance = Some(ForwardPassPerfProvenance {
                requested_estimation_mode: config.estimation_mode,
                selected_estimation_mode: config.estimation_mode,
                config: config.clone(),
                selected_systems_root: None,
                selection_failures: Vec::new(),
            });
        }
        // A from_learned model keeps its original absence of provenance.
        None => {}
    }
    if diagnostics_record(&model.diagnostics()) != manifest.diagnostics
        || model.readiness_report() != manifest.readiness
    {
        return Err(invalid(
            "restored diagnostics differ from those recorded when the snapshot was saved",
        ));
    }
    model.snapshot_tuning = Some(source.on_tune);
    Ok(model)
}

#[cfg(test)]
#[path = "snapshot/tests.rs"]
mod tests;
