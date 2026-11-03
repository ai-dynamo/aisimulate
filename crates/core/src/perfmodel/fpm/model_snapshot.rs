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
    LearnedRecord, LearnedRuntimeRecord, MANIFEST_FILE, MIXED_STATE_FILE, Manifest, MixedRecord,
    NativeRecord, ONLINE_CHECKPOINT_DIR, Producer, STABLE_ARTIFACT_FILE, STATE_FILE,
    STATE_SNAPSHOT_LAYOUT_VERSION, Stage, StateSnapshotKind, invalid, perf_db,
};
use super::*;

/// Options for [`ForwardPassPerfModel::save_state_snapshot`].
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
    /// Canonical pinned restore recipe, including state_snapshot selection.
    pub resolved_config: ForwardPassPerfModelConfig,
    /// Manifest-declared immutable inputs; full restoration verifies contents.
    pub dependencies: serde_json::Value,
}

impl ForwardPassStateSnapshotSummary {
    /// Execution requirements computed from the saved identity and store report.
    pub fn execution_readiness(&self) -> Option<ForwardPassExecutionReadiness> {
        let identity = self.identity.as_ref()?;
        Some(self.readiness.for_execution(
            identity.worker_type,
            &identity.backend,
            identity.attention_dp,
        ))
    }
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

/// Configuration paths of the inputs a snapshot owns, each with its fixed
/// location relative to the snapshot directory. Other paths, such as the
/// systems root of an unpacked native snapshot, name external inputs.
fn owned_paths(
    config: &mut ForwardPassPerfModelConfig,
    kind: StateSnapshotKind,
    packed: bool,
) -> Result<Vec<(&mut PathBuf, String)>, AicError> {
    let ForwardPassPerfModelConfig {
        systems_paths,
        estimator_config,
        ..
    } = config;
    let learned = &mut estimator_config.fpm_learned;
    let mut owned = Vec::new();
    match kind {
        StateSnapshotKind::Native => {
            if packed {
                let [root] = systems_paths.as_mut_slice() else {
                    return Err(invalid(
                        "a packed snapshot identity must name exactly one systems path",
                    ));
                };
                owned.push((root, perf_db::PACKED_SYSTEMS_DIR.to_owned()));
                if let Some(parquet) = estimator_config.fpm_interpolation.fpm_parquet_path.as_mut()
                {
                    let name = parquet
                        .file_name()
                        .and_then(|name| name.to_str())
                        .ok_or_else(|| invalid("packed fpm_parquet_path must name a UTF-8 file"))?
                        .to_owned();
                    owned.push((parquet, format!("{}/{name}", perf_db::PACKED_FPM_DIR)));
                }
            }
            if let Some(selection) = estimator_config.op_level.execution_profile.as_mut() {
                owned.push((&mut selection.path, EXECUTION_PROFILE_FILE.to_owned()));
            }
        }
        StateSnapshotKind::LearnedFrozen => {
            match (&mut learned.artifact_path, &learned.checkpoint_path) {
                (Some(artifact), None) => owned.push((artifact, LEARNED_ARTIFACT_FILE.to_owned())),
                _ => {
                    return Err(invalid(
                        "a frozen forest snapshot identity must name only its artifact",
                    ));
                }
            }
        }
        StateSnapshotKind::LearnedOnline | StateSnapshotKind::CausalSelector => {
            match (&learned.artifact_path, &mut learned.checkpoint_path) {
                (None, Some(checkpoint)) => {
                    owned.push((checkpoint, ONLINE_CHECKPOINT_DIR.to_owned()))
                }
                _ => {
                    return Err(invalid(
                        "an online forest snapshot identity must name only its checkpoint",
                    ));
                }
            }
        }
        StateSnapshotKind::Regression | StateSnapshotKind::LearnedWithoutIdentity => {}
    }
    // Learned source pins describe the snapshot's own exported seed. Relocate
    // every repeated reference along with the public artifact/checkpoint path.
    if let Some(source) = learned.source.resolved_source.as_mut() {
        let base = match kind {
            StateSnapshotKind::LearnedFrozen => Some(LEARNED_ARTIFACT_FILE),
            StateSnapshotKind::LearnedOnline | StateSnapshotKind::CausalSelector => {
                Some(ONLINE_CHECKPOINT_DIR)
            }
            _ => None,
        };
        if let Some(base) = base {
            let original = source
                .path
                .clone()
                .ok_or_else(|| invalid("saved learned source has no path"))?;
            if let Some(path) = source.path.as_mut() {
                owned.push((path, base.to_owned()));
            }
            if let Some(path) = source.seed_path.as_mut() {
                owned.push((path, base.to_owned()));
            }
            for dependency in &mut source.dependencies {
                let relative = if kind == StateSnapshotKind::LearnedFrozen {
                    base.to_owned()
                } else {
                    let suffix = dependency.path.strip_prefix(&original).map_err(|_| {
                        invalid("saved learned dependency is outside its checkpoint")
                    })?;
                    Path::new(base).join(suffix).to_string_lossy().into_owned()
                };
                owned.push((&mut dependency.path, relative));
            }
        }
    }
    Ok(owned)
}

fn rebase_learned_source(
    config: &mut ForwardPassPerfModelConfig,
    from: &Path,
    to: &Path,
) -> Result<(), AicError> {
    let source = &mut config.estimator_config.fpm_learned.source;
    if let Some(pin) = source.resolved_source.as_mut() {
        for path in pin
            .path
            .iter_mut()
            .chain(pin.seed_path.iter_mut())
            .chain(pin.dependencies.iter_mut().map(|dep| &mut dep.path))
        {
            let relative = path
                .strip_prefix(from)
                .map_err(|_| invalid("snapshot source is outside staging"))?;
            *path = to.join(relative);
        }
    }
    Ok(())
}

/// The directory that holds `relative` at `path`, if `path` ends with it.
fn strip_relative(path: &Path, relative: &str) -> Option<PathBuf> {
    let mut base = path.to_path_buf();
    for component in Path::new(relative).components().rev() {
        if base.file_name() != Some(component.as_os_str()) {
            return None;
        }
        base.pop();
    }
    base.is_absolute().then_some(base)
}

/// The manifest identity with its owned inputs resolved against the snapshot
/// directory `root`; `None` for a `from_learned` snapshot. Version 2 stores
/// them relative to the snapshot; version 1 stored them below the save
/// location, which is recovered from their fixed relative suffixes.
pub(super) fn rebased_identity(
    manifest: &Manifest,
    root: &Path,
) -> Result<Option<ForwardPassPerfModelConfig>, AicError> {
    let Some(mut identity) = manifest.identity.clone() else {
        return Ok(None);
    };
    if manifest.schema_version >= snapshot::SCHEMA_VERSION
        && matches!(
            manifest.kind,
            StateSnapshotKind::LearnedFrozen
                | StateSnapshotKind::LearnedOnline
                | StateSnapshotKind::CausalSelector
        )
    {
        snapshot::closure::absolute_owned(&mut identity, root)?;
        return Ok(Some(identity));
    }
    let packed = manifest.native.as_ref().is_some_and(|native| native.packed);
    let mut origin: Option<PathBuf> = None;
    for (path, relative) in owned_paths(&mut identity, manifest.kind, packed)? {
        if manifest.schema_version == snapshot::PATH_PINNED_SCHEMA_VERSION {
            let saved = strip_relative(path, &relative).ok_or_else(|| {
                invalid(format!(
                    "schema version 1 snapshot records {} for its {relative}, so it cannot be relocated; save the model again",
                    path.display()
                ))
            })?;
            if *origin.get_or_insert_with(|| saved.clone()) != saved {
                return Err(invalid(
                    "schema version 1 snapshot records its owned inputs below different directories",
                ));
            }
        } else if path.as_path() != Path::new(&relative) {
            return Err(invalid(format!(
                "manifest identity names {} instead of the snapshot's {relative}",
                path.display()
            )));
        }
        *path = root.join(&relative);
    }
    Ok(Some(identity))
}

/// Point the inputs a snapshot owns at the directory named by the
/// configuration's `state_snapshot`, whose manifest is `manifest_bytes`.
/// Owned inputs are read only from the snapshot, never from a path the
/// configuration supplies, so a copied or moved snapshot restores from its
/// own files; their content is pinned by the manifest. Whether an optional
/// owned input (an external parquet or execution profile) is configured
/// remains identity, compared at restoration.
pub(crate) fn relocate_snapshot_inputs(
    config: &mut ForwardPassPerfModelConfig,
    manifest_bytes: &[u8],
) -> Result<(), AicError> {
    let root = config
        .estimator_config
        .state_snapshot
        .as_ref()
        .ok_or_else(|| invalid("configuration has no state_snapshot"))?
        .path
        .clone();
    let manifest = Manifest::parse(manifest_bytes)?;
    // Report a different model before touching the configuration; automatic
    // selection is rejected by validation with its own message.
    if config.estimation_mode != EstimationMode::Auto
        && (config.worker_type != manifest.worker_type
            || config.estimation_mode != manifest.selected_estimation_mode)
    {
        return Err(saved_model_mismatch());
    }
    let Some(identity) = rebased_identity(&manifest, &root)? else {
        let learned = &mut config.estimator_config.fpm_learned;
        learned.artifact_path = Some(root.join(LEARNED_ARTIFACT_FILE));
        learned.checkpoint_path = None;
        learned.source.resolved_source = None;
        return Ok(());
    };
    let saved = identity.estimator_config;
    match manifest.kind {
        StateSnapshotKind::Native => {
            if manifest.native.as_ref().is_some_and(|native| native.packed) {
                config.systems_paths = identity.systems_paths;
                if let (Some(path), Some(packed)) = (
                    config
                        .estimator_config
                        .fpm_interpolation
                        .fpm_parquet_path
                        .as_mut(),
                    saved.fpm_interpolation.fpm_parquet_path,
                ) {
                    *path = packed;
                }
            }
            if let (Some(selection), Some(profile)) = (
                config.estimator_config.op_level.execution_profile.as_mut(),
                saved.op_level.execution_profile,
            ) {
                selection.path = profile.path;
            }
        }
        StateSnapshotKind::LearnedFrozen
        | StateSnapshotKind::LearnedOnline
        | StateSnapshotKind::CausalSelector => {
            let learned = &mut config.estimator_config.fpm_learned;
            learned.artifact_path = saved.fpm_learned.artifact_path;
            learned.checkpoint_path = saved.fpm_learned.checkpoint_path;
            learned.source_sha256 = saved.fpm_learned.source_sha256;
            learned.source = saved.fpm_learned.source;
        }
        StateSnapshotKind::Regression | StateSnapshotKind::LearnedWithoutIdentity => {}
    }
    Ok(())
}

fn saved_model_mismatch() -> AicError {
    invalid("worker_type or estimation_mode differs from the saved model")
}

/// Check a normalized snapshot configuration against the saved identity,
/// whose manifest is `manifest_bytes`, before anything is restored.
pub(crate) fn verify_snapshot_identity(
    config: &ForwardPassPerfModelConfig,
    manifest_bytes: &[u8],
) -> Result<(), AicError> {
    let root = &config
        .estimator_config
        .state_snapshot
        .as_ref()
        .ok_or_else(|| invalid("configuration has no state_snapshot"))?
        .path;
    check_identity(config, &Manifest::parse(manifest_bytes)?, root)
}

/// Compare everything but location: the snapshot's owned inputs were
/// relocated into `root`, and where an unpacked native snapshot finds its
/// external data is environment, verified by content at restoration. Whether
/// an external parquet is configured stays identity. Other estimators read no
/// performance data.
fn check_identity(
    config: &ForwardPassPerfModelConfig,
    manifest: &Manifest,
    root: &Path,
) -> Result<(), AicError> {
    let base = unpinned(config);
    if base.worker_type != manifest.worker_type
        || base.estimation_mode != manifest.selected_estimation_mode
    {
        return Err(saved_model_mismatch());
    }
    let packed = manifest.native.as_ref().is_some_and(|native| native.packed);
    let expected = match rebased_identity(manifest, root)? {
        Some(mut identity) => {
            // Versions 1 and 2 predate source/absence pins. Resolve only those
            // absent additive fields against their already fixed owned paths;
            // all historical identity and fitting controls remain exact.
            if manifest.schema_version < snapshot::SCHEMA_VERSION
                && identity.estimation_mode == EstimationMode::FpmLearned
                && identity
                    .estimator_config
                    .fpm_learned
                    .source
                    .resolved_source
                    .is_none()
            {
                super::super::source::resolve(&mut identity)?;
            }
            if !packed {
                identity.systems_paths = base.systems_paths.clone();
            }
            if manifest.native.is_some()
                && !packed
                && let (Some(saved), Some(supplied)) = (
                    identity
                        .estimator_config
                        .fpm_interpolation
                        .fpm_parquet_path
                        .as_mut(),
                    base.estimator_config
                        .fpm_interpolation
                        .fpm_parquet_path
                        .as_ref(),
                )
            {
                *saved = supplied.clone();
            }
            identity
        }
        None => {
            let options = manifest
                .learned
                .as_ref()
                .and_then(|learned| learned.options.as_ref())
                .ok_or_else(|| invalid("from_learned snapshot has no constructor options"))?;
            from_learned_config(
                manifest.worker_type,
                options,
                root.join(LEARNED_ARTIFACT_FILE),
            )
            .resolve()?
        }
    };
    if expected != base {
        return Err(identity_mismatch(&expected, &base));
    }
    if manifest.native.is_some()
        && !packed
        && let Some(missing) = base.systems_paths.iter().find(|root| !root.is_dir())
    {
        return Err(invalid(format!(
            "systems_paths entry {} is not a directory; a snapshot saved without include_perf_db reads its performance data from systems_paths",
            missing.display()
        )));
    }
    Ok(())
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
    /// `model.save_state_snapshot(path, options) -> Result<ForwardPassPerfModelConfig, AicError>`
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
    /// online checkpoints, execution profiles and complete learned prior data)
    /// redirected into the snapshot, plus
    /// `estimator_config.state_snapshot` pinned to the manifest digest with
    /// `on_tune: reject`. Set `on_tune: continue` to resume tuning. The
    /// `include_perf_db` option controls top-level native data; a learned
    /// fallback's native prior is always packed for complete continuation.
    ///
    /// Native estimators record the compiled engine-spec digest from
    /// construction and the fingerprint of their performance data taken when
    /// this process first loaded it; saving fails if that data changed since,
    /// and restoring recompiles and rejects any difference. Snapshot I/O
    /// failures are [`AicError::StateSnapshotIo`], never missing data. Models from
    /// [`Self::from_learned`] have no deployment identity: their snapshot
    /// configuration carries placeholder identity fields and restoration
    /// checks only estimator controls and file digests.
    pub fn save_state_snapshot(
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
        let mut mixed_record = None;
        let mut identity = match (&self.provenance, kind) {
            (Some(provenance), _) => Some(unpinned(
                self.pinned_learned_config
                    .as_ref()
                    .unwrap_or(&provenance.config),
            )),
            (None, StateSnapshotKind::LearnedWithoutIdentity) => None,
            (None, _) => {
                return Err(invalid(
                    "the model has no construction provenance; build it with best_available",
                ));
            }
        };
        if matches!(
            kind,
            StateSnapshotKind::LearnedFrozen
                | StateSnapshotKind::LearnedOnline
                | StateSnapshotKind::CausalSelector
        ) {
            let identity = identity.as_mut().expect("learned provenance");
            snapshot::closure::archive_source(&mut stage, identity)?;
            if let Some(mixed) = &self.mixed_composition {
                mixed_record = Some(pack_mixed(&mut stage, identity, mixed)?);
            }
        }
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
                    record_owned_tree(&mut stage, "perf_db")?;
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
                if let Some(online) = online {
                    online
                        .save_source_free(&stage.staging().join(ONLINE_CHECKPOINT_DIR), learned)?;
                    record_runtime_checkpoint(&mut stage)?;
                } else {
                    stage.write_recorded(LEARNED_ARTIFACT_FILE, learned.to_json()?.as_bytes())?;
                }
                learned_record = Some(LearnedRecord {
                    identity_verified: *identity_verified,
                    appended_mixed_sources: Default::default(),
                    options: identity.is_none().then(|| self.options.clone()),
                    runtime: identity.is_some().then(|| LearnedRuntimeRecord {
                        checkpoint: online.is_some().then(|| ONLINE_CHECKPOINT_DIR.into()),
                        artifact: online.is_none().then(|| LEARNED_ARTIFACT_FILE.into()),
                    }),
                });
            }
            ForwardPassPerfMode::CausalSelector(model) => {
                model.validate_appended_sources(model.appended_sources())?;
                stage.write_recorded(STABLE_ARTIFACT_FILE, model.stable.to_json()?.as_bytes())?;
                model.online.save_source_free(
                    &stage.staging().join(ONLINE_CHECKPOINT_DIR),
                    &model.learned,
                )?;
                record_runtime_checkpoint(&mut stage)?;
                learned_record = Some(LearnedRecord {
                    identity_verified: model.identity_verified,
                    appended_mixed_sources: model.appended_sources().clone(),
                    options: None,
                    runtime: Some(LearnedRuntimeRecord {
                        checkpoint: Some(ONLINE_CHECKPOINT_DIR.into()),
                        artifact: None,
                    }),
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

        if let Some(identity) = identity.as_mut() {
            if matches!(
                kind,
                StateSnapshotKind::LearnedFrozen
                    | StateSnapshotKind::LearnedOnline
                    | StateSnapshotKind::CausalSelector
            ) {
                let mut staged = identity.clone();
                snapshot::closure::relative_owned(&mut staged, &destination)?;
                snapshot::closure::absolute_owned(&mut staged, stage.staging())?;
                super::super::source::resolve(&mut staged)?;
            }
        }
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
                rebase_learned_source(&mut config, stage.staging(), &destination)?;
                config
            }
        };
        // The manifest names owned inputs relative to the snapshot, so the
        // directory restores wherever it is copied or moved.
        if let Some(identity) = identity.as_mut() {
            if matches!(
                kind,
                StateSnapshotKind::LearnedFrozen
                    | StateSnapshotKind::LearnedOnline
                    | StateSnapshotKind::CausalSelector
            ) {
                snapshot::closure::relative_owned(identity, &destination)?;
            } else {
                let packed = native.as_ref().is_some_and(|native| native.packed);
                for (path, relative) in owned_paths(identity, kind, packed)? {
                    if *path != destination.join(&relative) {
                        return Err(invalid(format!(
                            "{} is not the snapshot's {relative}",
                            path.display()
                        )));
                    }
                    *path = PathBuf::from(relative);
                }
            }
        }
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
            mixed: mixed_record,
            diagnostics: diagnostics_record(&diagnostics),
            readiness: self.readiness_record(),
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
    /// Dependency records are manifest-pinned declarations. Construct the
    /// returned `resolved_config` to verify their bytes and restore semantics.
    /// Legacy manifests may list only a nested checkpoint manifest; constructed
    /// model dependency metadata also includes its verified nested payloads.
    pub fn inspect_state_snapshot(
        path: impl AsRef<Path>,
    ) -> Result<ForwardPassStateSnapshotSummary, AicError> {
        let root = snapshot::normalized_absolute(path.as_ref())?;
        let bytes = snapshot::read_manifest_bytes(&root)?;
        let manifest = Manifest::parse(&bytes)?;
        let identity = rebased_identity(&manifest, &root)?;
        let resolved_config = summary_config(&manifest, &root, &snapshot::sha256_hex(&bytes))?;
        let dependencies = snapshot_dependencies(&manifest, &root, &bytes, None)?;
        Ok(ForwardPassStateSnapshotSummary {
            manifest_sha256: snapshot::sha256_hex(&bytes),
            kind: manifest.kind,
            aisimulate_core_version: manifest.producer.aisimulate_core_version,
            state_layout_version: manifest.state_layout_version,
            worker_type: manifest.worker_type,
            selected_estimation_mode: manifest.selected_estimation_mode,
            topology: identity
                .as_ref()
                .map(ForwardPassDeploymentTopology::from_config),
            identity,
            readiness: manifest.readiness.report(),
            resolved_config,
            dependencies,
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

fn record_owned_tree(stage: &mut Stage, relative: &str) -> Result<(), AicError> {
    fn paths(root: &Path, out: &mut Vec<PathBuf>) -> Result<(), AicError> {
        for entry in std::fs::read_dir(root).map_err(|e| snapshot::io_error(root, e))? {
            let entry = entry.map_err(|e| snapshot::io_error(root, e))?;
            let path = entry.path();
            let kind = entry
                .file_type()
                .map_err(|e| snapshot::io_error(&path, e))?;
            if kind.is_dir() {
                paths(&path, out)?;
            } else if kind.is_file() {
                out.push(path);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    paths(
        &stage.staging().join(snapshot::relative_path(relative)?),
        &mut files,
    )?;
    files.sort();
    for path in files {
        let relative = path
            .strip_prefix(stage.staging())
            .expect("owned tree path")
            .to_string_lossy()
            .into_owned();
        stage.record_existing(&relative)?;
    }
    Ok(())
}

fn record_runtime_checkpoint(stage: &mut Stage) -> Result<(), AicError> {
    record_owned_tree(stage, ONLINE_CHECKPOINT_DIR)
}

fn pack_mixed(
    stage: &mut Stage,
    identity: &mut ForwardPassPerfModelConfig,
    mixed: &MixedComposition,
) -> Result<MixedRecord, AicError> {
    if identity
        .estimator_config
        .effective_mixed_composition()
        .is_none()
    {
        return Err(invalid("mixed prior has no canonical construction policy"));
    }
    let native = if let Some(engine) = mixed.native_engine() {
        let dependency = mixed.dependency_provenance();
        let provenance: ForwardPassPerfProvenance =
            serde_json::from_value(dependency["provenance"].clone()).map_err(invalid)?;
        let root = provenance
            .selected_systems_root
            .as_deref()
            .ok_or_else(|| invalid("prior has no selected root"))?;
        let parquet = provenance
            .config
            .estimator_config
            .fpm_interpolation
            .fpm_parquet_path
            .as_deref();
        let loaded: perf_db::PerfDbFingerprint =
            serde_json::from_value(dependency["fingerprint"].clone()).map_err(invalid)?;
        if perf_db::fingerprint(root, &identity.system, parquet)? != loaded {
            return Err(invalid("native prior data changed after construction"));
        }
        let (packed_root, packed_parquet) =
            perf_db::pack_at(stage, root, &identity.system, parquet, "mixed")?;
        let staged_root = stage
            .staging()
            .join(packed_root.strip_prefix(stage.destination()).unwrap());
        let staged_parquet = packed_parquet.as_ref().map(|path| {
            stage
                .staging()
                .join(path.strip_prefix(stage.destination()).unwrap())
        });
        if perf_db::fingerprint(&staged_root, &identity.system, staged_parquet.as_deref())?
            != loaded
        {
            return Err(invalid("packed native prior differs from loaded data"));
        }
        let pin = identity
            .estimator_config
            .fpm_learned
            .source
            .resolved_fallback
            .as_mut()
            .and_then(|fallback| fallback.native.as_mut())
            .ok_or_else(|| invalid("native prior has no resolved pin"))?;
        snapshot::closure::map_native_pin(pin, &mut |path| {
            if path == root {
                return Ok(packed_root.clone());
            }
            if Some(path) == parquet {
                return packed_parquet
                    .clone()
                    .ok_or_else(|| invalid("missing packed prior parquet"));
            }
            Err(invalid(format!(
                "unowned native prior input {}",
                path.display()
            )))
        })?;
        record_owned_tree(stage, "mixed/perf_db")?;
        Some(NativeRecord {
            engine_spec_sha256: engine.spec_sha256().into(),
            perf_db: loaded,
            packed: true,
        })
    } else {
        None
    };
    let mut state = mixed.snapshot()?;
    state["binding"]["policy"] = serde_json::to_value(
        identity
            .estimator_config
            .effective_mixed_composition()
            .ok_or_else(|| invalid("missing mixed policy"))?,
    )
    .map_err(invalid)?;
    if native.is_some() {
        state["binding"]["prior"]["provenance"] = identity
            .estimator_config
            .fpm_learned
            .source
            .resolved_fallback
            .as_ref()
            .and_then(|fallback| fallback.native.as_ref())
            .unwrap()["provenance"]
            .clone();
    }
    snapshot::closure::map_mixed_binding(&mut state, &mut |path| {
        let relative = path
            .strip_prefix(stage.destination())
            .map_err(|_| invalid("mixed binding contains an external path"))?;
        Ok(snapshot::relative_path(
            relative
                .to_str()
                .ok_or_else(|| invalid("mixed path is not UTF-8"))?,
        )?
        .to_owned())
    })?;
    stage.write_recorded(
        MIXED_STATE_FILE,
        &serde_json::to_vec(&state).map_err(invalid)?,
    )?;
    Ok(MixedRecord {
        state: MIXED_STATE_FILE.into(),
        native,
    })
}

fn summary_config(
    manifest: &Manifest,
    root: &Path,
    manifest_sha256: &str,
) -> Result<ForwardPassPerfModelConfig, AicError> {
    let mut config = match rebased_identity(manifest, root)? {
        Some(identity) => identity,
        None => {
            let options = manifest
                .learned
                .as_ref()
                .and_then(|record| record.options.as_ref())
                .ok_or_else(|| invalid("from_learned snapshot has no options"))?;
            let mut config = from_learned_config(
                manifest.worker_type,
                options,
                root.join(LEARNED_ARTIFACT_FILE),
            );
            config.estimator_config.fpm_learned.source_sha256 = Some(
                manifest
                    .file(LEARNED_ARTIFACT_FILE)
                    .ok_or_else(|| invalid("missing from_learned artifact record"))?
                    .sha256
                    .clone(),
            );
            config
        }
    };
    config.estimator_config.state_snapshot = Some(StateSnapshotConfig {
        path: root.into(),
        sha256: Some(manifest_sha256.into()),
        on_tune: StateSnapshotTuning::Reject,
    });
    Ok(config)
}

fn snapshot_dependencies(
    manifest: &Manifest,
    root: &Path,
    bytes: &[u8],
    resolved: Option<&ForwardPassPerfModelConfig>,
) -> Result<serde_json::Value, AicError> {
    let identity = match resolved {
        Some(config) => Some(unpinned(config)),
        None => rebased_identity(manifest, root)?,
    };
    let mut owned = Vec::<PathBuf>::new();
    let mut external = Vec::<PathBuf>::new();
    if let Some(native) = &manifest.native {
        if let Some(identity) = &identity {
            let roots = if native.packed {
                &mut owned
            } else {
                &mut external
            };
            roots.extend(identity.systems_paths.iter().cloned());
        }
    }
    let fallback = identity
        .as_ref()
        .and_then(|identity| {
            identity
                .estimator_config
                .fpm_learned
                .source
                .resolved_fallback
                .as_ref()
        })
        .and_then(|fallback| fallback.native.clone());
    if let Some(path) = fallback
        .as_ref()
        .and_then(|pin| pin["provenance"]["selected_systems_root"].as_str())
    {
        if manifest
            .mixed
            .as_ref()
            .and_then(|mixed| mixed.native.as_ref())
            .is_some()
        {
            owned.push(path.into());
        } else {
            external.push(path.into());
        }
    }
    owned.sort();
    owned.dedup();
    external.sort();
    external.dedup();
    let mut systems_paths = owned.clone();
    systems_paths.extend(external.clone());
    let mut files = manifest.files.iter().map(|file| serde_json::json!({
        "path":root.join(&file.path), "sha256":file.sha256, "byte_length":file.size_bytes, "purpose":"snapshot_owned_input"
    })).collect::<Vec<_>>();
    // Old snapshot manifests list the runtime checkpoint manifest only.
    // Construction has verified its nested model/state/rows, so flatten those
    // resolved pins for consumers without adding inspection-time file reads.
    if let Some(source) = resolved.and_then(|config| {
        config
            .estimator_config
            .fpm_learned
            .source
            .resolved_source
            .as_ref()
    }) {
        for dependency in &source.dependencies {
            let path = serde_json::to_value(&dependency.path).map_err(invalid)?;
            if let Some(existing) = files.iter().find(|file| file["path"] == path) {
                if existing["sha256"] != dependency.sha256
                    || existing["byte_length"] != dependency.byte_length
                {
                    return Err(invalid(
                        "resolved dependency differs from snapshot file pin",
                    ));
                }
            } else {
                files.push(serde_json::to_value(dependency).map_err(invalid)?);
            }
        }
    }
    files.push(serde_json::json!({"path":root.join(MANIFEST_FILE), "sha256":snapshot::sha256_hex(bytes), "byte_length":bytes.len(), "purpose":"snapshot_manifest"}));
    Ok(
        serde_json::json!({"systems_paths":systems_paths, "owned_systems_paths":owned,
        "external_systems_paths":external, "files":files, "fallback":fallback}),
    )
}

/// Select current runtime payload only after the ordinary resolver verified
/// the archived original source and the snapshot verified every payload byte.
fn learned_runtime_base(
    canonical: &ForwardPassPerfModelConfig,
    manifest: &Manifest,
    root: &Path,
    stable_bytes: Option<&[u8]>,
) -> Result<
    (
        ForwardPassPerfModel,
        super::super::source::VerifiedLearnedSource,
    ),
    AicError,
> {
    let runtime = manifest
        .learned
        .as_ref()
        .and_then(|record| record.runtime.as_ref())
        .ok_or_else(|| invalid("schema 3 learned snapshot lacks current runtime payload"))?;
    let mut config = canonical.clone();
    let settings = &mut config.estimator_config.fpm_learned;
    let fallback = settings.source.resolved_fallback.clone();
    settings.source = Default::default();
    settings.source.resolved_fallback = fallback;
    settings.checkpoint_path = runtime.checkpoint.as_ref().map(|path| root.join(path));
    settings.artifact_path = runtime.artifact.as_ref().map(|path| root.join(path));
    let source_record = runtime
        .checkpoint
        .as_ref()
        .map(|path| format!("{path}/manifest.json"))
        .or_else(|| runtime.artifact.clone())
        .ok_or_else(|| invalid("missing current learned payload"))?;
    settings.source_sha256 = Some(
        manifest
            .file(&source_record)
            .ok_or_else(|| invalid("runtime payload is not pinned"))?
            .sha256
            .clone(),
    );
    if manifest.kind == StateSnapshotKind::CausalSelector && stable_bytes.is_none() {
        return Err(invalid("manifest does not list the frozen expert"));
    }
    let expected_identity = canonical_learned_identity(&config)?;
    let stable = stable_bytes
        .map(|bytes| {
            let text = std::str::from_utf8(bytes).map_err(invalid)?;
            let model = LearnedForwardPassModel::from_json(text)?;
            if model.worker_type() != config.worker_type
                || model
                    .metadata()
                    .get("identity")
                    .is_some_and(|identity| identity != &expected_identity)
            {
                return Err(invalid("stable expert identity differs from snapshot"));
            }
            Ok(model)
        })
        .transpose()?;
    let (mut model, receipt) =
        ForwardPassPerfModel::from_learned_state_with_receipt(&config, stable, true)?;
    if let ForwardPassPerfMode::CausalSelector(selector) = &mut model.mode {
        selector.restore_appended_sources(
            manifest
                .learned
                .as_ref()
                .expect("learned record")
                .appended_mixed_sources
                .clone(),
        )?;
    }
    model.provenance = Some(ForwardPassPerfProvenance {
        config: canonical.clone(),
        selected_systems_root: None,
        requested_estimation_mode: canonical.estimation_mode,
        selected_estimation_mode: canonical.estimation_mode,
        selection_failures: Vec::new(),
    });
    Ok((model, receipt))
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
    let settings = config.estimator_config.fpm_learned.runtime_source_config();
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
    let online = OnlineLearner::load_combined_runtime(checkpoint, &mut learned, &settings, config)?;
    let options = config.estimator_config.correction_options();
    let combined = CalibratedOnlineModel::new(
        stable,
        learned,
        online,
        true,
        &options,
        &config.estimator_config.selector,
    )?;
    // Legacy selector snapshots use this internal constructor so that the
    // frozen expert is restored independently. Capture their verified nested
    // checkpoint payloads just as the ordinary public constructor does.
    let mut pinned = config.clone();
    if pinned
        .estimator_config
        .fpm_learned
        .source
        .resolved_source
        .is_none()
    {
        super::super::source::pin_legacy_source(&mut pinned)?;
    }
    Ok(ForwardPassPerfModel {
        mode: ForwardPassPerfMode::CausalSelector(Box::new(combined)),
        options,
        last_warning: None,
        provenance: None,
        is_correction_enabled: config.estimator_config.correction.enabled,
        query_coverage: None,
        snapshot_tuning: None,
        snapshot_dependencies: None,
        pinned_learned_config: Some(pinned),
        mixed_composition: None,
    })
}

type Select<'a> =
    &'a dyn Fn(ForwardPassPerfModelConfig, Option<bool>) -> Result<ForwardPassPerfModel, AicError>;

/// Restore the model pinned by `config.estimator_config.state_snapshot`.
/// `select` builds the estimator for a resolved configuration without the
/// snapshot (`ForwardPassPerfModel::select` outside tests).
#[cfg(test)]
pub(super) fn restore(
    config: ForwardPassPerfModelConfig,
    registered: Option<bool>,
    select: Select<'_>,
) -> Result<ForwardPassPerfModel, AicError> {
    restore_with_receipt(config, registered, select, None)
}

pub(super) fn restore_with_receipt(
    config: ForwardPassPerfModelConfig,
    registered: Option<bool>,
    select: Select<'_>,
    source_receipt: Option<super::super::source::VerifiedLearnedSource>,
) -> Result<ForwardPassPerfModel, AicError> {
    restore_verified(config, registered, select, source_receipt).map_err(snapshot::as_snapshot_io)
}

fn restore_verified(
    config: ForwardPassPerfModelConfig,
    registered: Option<bool>,
    select: Select<'_>,
    source_receipt: Option<super::super::source::VerifiedLearnedSource>,
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
    check_identity(&config, &manifest, &root)?;
    let base = unpinned(&config);
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
    // Verify directly decoded members from the same bytes. Schema 3 runtime
    // members are authenticated by the checkpoint/artifact decoder instead;
    // compare its operation-local receipt to these outer pins before accepting
    // the model, rather than reading and hashing the same payload twice.
    let has_separate_runtime = manifest.schema_version >= snapshot::SCHEMA_VERSION
        && matches!(
            manifest.kind,
            StateSnapshotKind::LearnedFrozen
                | StateSnapshotKind::LearnedOnline
                | StateSnapshotKind::CausalSelector
        );
    let runtime = manifest
        .learned
        .as_ref()
        .and_then(|record| record.runtime.as_ref());
    // The archived seed has already been checked by this construction's source
    // resolver. Mixed priors reopen original-A members later, so keep their
    // independent outer verification until those readers return receipts too.
    let source_receipt = source_receipt.filter(|_| {
        manifest.schema_version >= snapshot::SCHEMA_VERSION
            && base
                .estimator_config
                .effective_mixed_composition()
                .is_none()
    });
    let mut runtime_records = Vec::new();
    let mut stable_bytes = None;
    let mut state_bytes = None;
    let mut learned_bytes = None;
    let mut mixed_bytes = None;
    for record in &manifest.files {
        let decoded_runtime_member = has_separate_runtime
            && runtime.is_some_and(|runtime| {
                runtime.artifact.as_deref() == Some(record.path.as_str())
                    || (runtime.checkpoint.is_some()
                        && matches!(
                            record.path.as_str(),
                            "learned/online/manifest.json"
                                | "learned/online/model.json"
                                | "learned/online/state.bin"
                        ))
            });
        if decoded_runtime_member {
            runtime_records.push(record);
            continue;
        }
        if record.path.starts_with("source/")
            && let Some(receipt) = &source_receipt
            && receipt
                .source
                .dependencies
                .iter()
                .any(|dependency| dependency.path == root.join(&record.path))
        {
            snapshot::verify_consumed(&root, record, receipt)?;
            continue;
        }
        match record.path.as_str() {
            STATE_FILE => state_bytes = Some(snapshot::read_recorded(&root, record)?),
            STABLE_ARTIFACT_FILE => stable_bytes = Some(snapshot::read_recorded(&root, record)?),
            LEARNED_ARTIFACT_FILE => learned_bytes = Some(snapshot::read_recorded(&root, record)?),
            MIXED_STATE_FILE => mixed_bytes = Some(snapshot::read_recorded(&root, record)?),
            _ => snapshot::verify_recorded(&root, record)?,
        }
    }
    let state_bytes = state_bytes.ok_or_else(|| invalid("manifest does not list state.bin"))?;
    let state: StateFile = snapshot::decode_state(&state_bytes)?;
    if state.layout_version != manifest.state_layout_version {
        return Err(invalid(
            "state.bin layout version differs from its manifest",
        ));
    }

    // Verify referenced performance data before any of it is loaded: a
    // changed file must report the fingerprint, not a loader failure. Tables
    // load lazily on first query, so no later hash here would cover every
    // read; `save_state_snapshot` hashes again and rejects data changed after this check.
    let parquet = base
        .estimator_config
        .fpm_interpolation
        .fpm_parquet_path
        .as_deref();
    let (native_root, data_stamp) = match (&manifest.native, base.systems_paths.as_slice()) {
        (None, _) => (None, None),
        (Some(native), [root]) => {
            let (current, stamp) = perf_db::current(root, &base.system, parquet)?;
            if current != native.perf_db {
                return Err(invalid(format!(
                    "performance data under {} differs from the snapshot fingerprint",
                    root.display()
                )));
            }
            (Some(root.clone()), Some(stamp))
        }
        (Some(_), _) => {
            return Err(invalid(
                "a native snapshot configuration must pin exactly one systems path",
            ));
        }
    };

    let mut model = if has_separate_runtime {
        let (model, receipt) =
            learned_runtime_base(&base, &manifest, &root, stable_bytes.as_deref())?;
        for record in runtime_records {
            snapshot::verify_consumed(&root, record, &receipt)?;
        }
        model
    } else {
        match manifest.kind {
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
        }
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
        // The data matched before this model loaded and only verified tables
        // stayed shared, so later restores of untouched data may share these.
        if let (Some(root), Some(stamp)) = (&native_root, data_stamp) {
            perf_db::verified(root, &base.system, parquet, engine.database(), stamp);
        }
    }
    if let Some(mixed_record) = &manifest.mixed {
        let mut state: serde_json::Value = serde_json::from_slice(
            mixed_bytes
                .as_deref()
                .ok_or_else(|| invalid("missing mixed state payload"))?,
        )
        .map_err(invalid)?;
        snapshot::closure::map_mixed_binding(&mut state, &mut |path| {
            Ok(root.join(snapshot::relative_path(
                path.to_str()
                    .ok_or_else(|| invalid("mixed path is not UTF-8"))?,
            )?))
        })?;
        let mixed = model
            .mixed_composition
            .as_mut()
            .ok_or_else(|| invalid("mixed snapshot did not construct its prior"))?;
        if let Some(native) = &mixed_record.native {
            let engine = mixed
                .native_engine()
                .ok_or_else(|| invalid("mixed snapshot lost its native engine"))?;
            if engine.spec_sha256() != native.engine_spec_sha256
                || mixed.dependency_provenance()["fingerprint"]
                    != serde_json::to_value(&native.perf_db).map_err(invalid)?
            {
                return Err(invalid("mixed native prior differs from snapshot"));
            }
        } else if mixed.native_engine().is_some() {
            return Err(invalid("mixed snapshot unexpectedly gained a native prior"));
        }
        mixed.restore(state)?;
    } else if model.mixed_composition.is_some() {
        return Err(invalid("snapshot omitted mixed state"));
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
    let mut readiness = model.readiness_record();
    if manifest.readiness.execution.is_none() {
        // Older snapshots did not record the additive execution decision. Keep
        // checking every state field they did record, including counts and
        // readiness.
        readiness.execution = None;
    }
    if diagnostics_record(&model.diagnostics()) != manifest.diagnostics
        || readiness != manifest.readiness
    {
        return Err(invalid(
            "restored diagnostics differ from those recorded when the snapshot was saved",
        ));
    }
    model.snapshot_tuning = Some(source.on_tune);
    model.snapshot_dependencies = Some(snapshot_dependencies(
        &manifest,
        &root,
        &manifest_bytes,
        Some(model.pinned_learned_config.as_ref().unwrap_or(&config)),
    )?);
    Ok(model)
}

#[cfg(test)]
#[path = "snapshot/tests.rs"]
mod tests;
