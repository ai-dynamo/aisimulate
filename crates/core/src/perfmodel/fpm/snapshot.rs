// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! On-disk format of forward-pass model state snapshots.
//!
//! [`ForwardPassPerfModel::save_state_snapshot`](crate::ForwardPassPerfModel::save_state_snapshot)
//! atomically writes a new directory:
//!
//! ```text
//! <snapshot>/
//!   manifest.json          versions, pinned identity, readiness, file digests
//!   config.json            canonical configuration accepted by best_available
//!   state.bin              tunable state, bincode 1 fixint (see STATE_LAYOUT_VERSION)
//!   learned/artifact.json  frozen forest (fpm_learned frozen, from_learned)
//!   learned/stable.json    frozen expert forest (causal selector)
//!   learned/online/        online learner checkpoint, runtime format v5
//!   execution_profile.json pinned op-level execution profile, when configured
//!   perf_db/               packed performance data (include_perf_db only)
//! ```
//!
//! `config.json` pins `estimator_config.state_snapshot.sha256` to the
//! manifest bytes, and the manifest pins every other file except the packed
//! performance-data tree, which is pinned by its tree digest. This module owns
//! the encoding, version checks, digests and atomic publication; the model
//! owns state extraction and restoration.

pub(crate) mod codec;
pub(crate) mod perf_db;

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::model::{ForwardPassPerfSource, ForwardPassReadinessReport, ForwardPassWorkerType};
use super::{EstimationMode, ForwardPassPerfModelConfig, ForwardPassPerfOptions};
use crate::AicError;

pub(crate) const SCHEMA: &str = "aisimulate_fpm_state_snapshot";
pub(crate) const SCHEMA_VERSION: u32 = 1;
/// Positional layout of `state.bin`. Bump it with any change to an encoded
/// state type (fields, order, types or enum variants); the layout golden test
/// fails until it is updated.
pub const STATE_SNAPSHOT_LAYOUT_VERSION: u32 = 1;
pub(crate) const STATE_ENCODING: &str = "bincode1_fixint";
pub(crate) const MANIFEST_FILE: &str = "manifest.json";
pub(crate) const CONFIG_FILE: &str = "config.json";
pub(crate) const STATE_FILE: &str = "state.bin";
pub(crate) const LEARNED_ARTIFACT_FILE: &str = "learned/artifact.json";
pub(crate) const STABLE_ARTIFACT_FILE: &str = "learned/stable.json";
pub(crate) const ONLINE_CHECKPOINT_DIR: &str = "learned/online";
pub(crate) const EXECUTION_PROFILE_FILE: &str = "execution_profile.json";
/// Upper bound on manifest and state reads. Online learner state is stored in
/// its own checkpoint, so `state.bin` holds only regression, correction and
/// selector state.
pub(crate) const MAX_STATE_BYTES: u64 = 1 << 32;
pub(crate) const MAX_MANIFEST_BYTES: u64 = 64 << 20;

pub(crate) fn invalid(message: impl std::fmt::Display) -> AicError {
    AicError::InvalidEngineConfig(format!("state snapshot: {message}"))
}

/// Which model structure a snapshot restores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateSnapshotKind {
    /// Op-level or FPM-interpolation engine plus correction stores.
    Native,
    /// Worker-bound regression stores.
    Regression,
    /// Frozen learned forest plus correction stores.
    LearnedFrozen,
    /// Online learned forest and learner state.
    LearnedOnline,
    /// Calibrated frozen expert, online expert and selector evidence.
    CausalSelector,
    /// A forest built by `from_learned`, which has no deployment identity.
    LearnedWithoutIdentity,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Producer {
    /// Crate and wheel releases share this version; a snapshot is accepted
    /// only by the same version.
    pub aisimulate_core_version: String,
    pub engine_spec_schema_version: u32,
    pub state_encoding: String,
}

impl Producer {
    pub(crate) fn current() -> Self {
        Self {
            aisimulate_core_version: env!("CARGO_PKG_VERSION").into(),
            engine_spec_schema_version: crate::ENGINE_SPEC_SCHEMA_VERSION,
            state_encoding: STATE_ENCODING.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileRecord {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeRecord {
    /// SHA-256 of the compiled engine specification recorded when the saving
    /// model was constructed, excluding its storage root.
    pub engine_spec_sha256: String,
    /// Fingerprint the saving model recorded when it loaded its performance
    /// data, confirmed unchanged at save time.
    pub perf_db: perf_db::PerfDbFingerprint,
    /// Whether the fingerprinted data was copied into the snapshot.
    pub packed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LearnedRecord {
    pub identity_verified: bool,
    /// Constructor options of a `from_learned` model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<ForwardPassPerfOptions>,
}

/// State-derived summary compared after restoration. Not-ready subtypes and
/// `last_warning` describe how the saving model was selected, so only
/// readiness itself is compared.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DiagnosticsRecord {
    pub source: ForwardPassPerfSource,
    pub ready: bool,
    pub retained_observations: usize,
    pub correction_ready_buckets: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub schema: String,
    pub schema_version: u32,
    pub state_layout_version: u32,
    pub producer: Producer,
    pub kind: StateSnapshotKind,
    pub worker_type: ForwardPassWorkerType,
    pub selected_estimation_mode: EstimationMode,
    /// Pinned canonical configuration without `state_snapshot`. `None` only
    /// for [`StateSnapshotKind::LearnedWithoutIdentity`].
    pub identity: Option<ForwardPassPerfModelConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native: Option<NativeRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learned: Option<LearnedRecord>,
    pub diagnostics: DiagnosticsRecord,
    pub readiness: ForwardPassReadinessReport,
    pub files: Vec<FileRecord>,
}

/// Version fields read leniently before the strict parse, so a manifest from
/// another release reports its version instead of an unknown field.
#[derive(Deserialize)]
struct ManifestHeader {
    schema: Option<serde_json::Value>,
    schema_version: Option<serde_json::Value>,
    state_layout_version: Option<serde_json::Value>,
    producer: Option<serde_json::Value>,
}

impl ManifestHeader {
    fn check(bytes: &[u8]) -> Result<(), AicError> {
        let header: Self = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("invalid manifest.json: {error}")))?;
        let show = |value: &serde_json::Value| match value {
            serde_json::Value::String(text) => text.clone(),
            serde_json::Value::Null => "missing".to_owned(),
            other => other.to_string(),
        };
        let text = |value: &Option<serde_json::Value>| {
            value.as_ref().map_or_else(|| "missing".to_owned(), show)
        };
        if header.schema.as_ref().and_then(serde_json::Value::as_str) != Some(SCHEMA)
            || header
                .schema_version
                .as_ref()
                .and_then(serde_json::Value::as_u64)
                != Some(u64::from(SCHEMA_VERSION))
        {
            return Err(invalid(format!(
                "unsupported manifest schema {} version {}",
                text(&header.schema),
                text(&header.schema_version)
            )));
        }
        if header
            .state_layout_version
            .as_ref()
            .and_then(serde_json::Value::as_u64)
            != Some(u64::from(STATE_SNAPSHOT_LAYOUT_VERSION))
        {
            return Err(invalid(format!(
                "state layout version {} is not supported by this binary (expected {STATE_SNAPSHOT_LAYOUT_VERSION})",
                text(&header.state_layout_version)
            )));
        }
        let current = Producer::current();
        let producer = header.producer.unwrap_or_default();
        let field = |name: &str| producer.get(name).cloned().unwrap_or_default();
        if field("aisimulate_core_version") != current.aisimulate_core_version.as_str()
            || field("engine_spec_schema_version") != current.engine_spec_schema_version
            || field("state_encoding") != current.state_encoding.as_str()
        {
            return Err(invalid(format!(
                "snapshot was written by aisimulate-core {} (engine spec schema {}, encoding {}); this binary is {} ({}, {})",
                show(&field("aisimulate_core_version")),
                show(&field("engine_spec_schema_version")),
                show(&field("state_encoding")),
                current.aisimulate_core_version,
                current.engine_spec_schema_version,
                current.state_encoding,
            )));
        }
        Ok(())
    }
}

impl Manifest {
    /// Parse pinned manifest bytes and require this binary's format versions.
    /// Versions are checked before unknown fields, which a newer producer may add.
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, AicError> {
        ManifestHeader::check(bytes)?;
        let manifest: Self = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("invalid manifest.json: {error}")))?;
        if manifest.producer != Producer::current() {
            return Err(invalid("manifest producer fields are malformed"));
        }
        if (manifest.kind == StateSnapshotKind::LearnedWithoutIdentity)
            != manifest.identity.is_none()
        {
            return Err(invalid(
                "only from_learned snapshots may omit their identity",
            ));
        }
        let mut paths = std::collections::BTreeSet::new();
        for file in &manifest.files {
            relative_path(&file.path)?;
            if !is_digest(&file.sha256) || !paths.insert(file.path.as_str()) {
                return Err(invalid("manifest file records are malformed"));
            }
        }
        Ok(manifest)
    }

    pub(crate) fn file(&self, path: &str) -> Option<&FileRecord> {
        self.files.iter().find(|file| file.path == path)
    }
}

pub(crate) fn is_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Report I/O failures from a snapshot save or restore, including those
/// raised by shared loaders, as snapshot I/O rather than missing data.
pub(crate) fn as_snapshot_io(error: AicError) -> AicError {
    match error {
        AicError::Io { path, source } => AicError::StateSnapshotIo { path, source },
        other => other,
    }
}

pub(crate) fn io_error(path: &Path, source: std::io::Error) -> AicError {
    AicError::StateSnapshotIo {
        path: path.to_path_buf(),
        source,
    }
}

/// Absolute, lexically normalized path: `.` is removed and `..` removes the
/// preceding component, like Python's `os.path.abspath`. Symlinks are kept,
/// so a stable symlinked mount records the same path on every host.
pub(crate) fn normalized_absolute(path: &Path) -> Result<PathBuf, AicError> {
    let absolute = std::path::absolute(path).map_err(|error| io_error(path, error))?;
    let mut out = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if out.parent().is_some() {
                    out.pop();
                }
            }
            other => out.push(other),
        }
    }
    Ok(out)
}

/// Snapshot-relative paths contain only normal components.
pub(crate) fn relative_path(path: &str) -> Result<&Path, AicError> {
    let relative = Path::new(path);
    if path.is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(invalid(format!("unsafe snapshot path {path:?}")));
    }
    Ok(relative)
}

/// Read a bounded file once; the caller hashes and parses the same bytes.
pub(crate) fn read_limited(path: &Path, limit: u64) -> Result<Vec<u8>, AicError> {
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let mut bytes = Vec::new();
    file.take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| io_error(path, error))?;
    if bytes.len() as u64 > limit {
        return Err(invalid(format!("{} exceeds {limit} bytes", path.display())));
    }
    Ok(bytes)
}

pub(crate) fn read_manifest_bytes(root: &Path) -> Result<Vec<u8>, AicError> {
    read_limited(&root.join(MANIFEST_FILE), MAX_MANIFEST_BYTES)
}

/// Read a manifest-listed file and verify its recorded size and digest.
pub(crate) fn read_recorded(root: &Path, record: &FileRecord) -> Result<Vec<u8>, AicError> {
    let path = root.join(relative_path(&record.path)?);
    if record.size_bytes > MAX_STATE_BYTES {
        return Err(invalid(format!(
            "{} is recorded as {} bytes, above the {MAX_STATE_BYTES}-byte limit",
            record.path, record.size_bytes
        )));
    }
    // Never read more than the record claims plus one byte, which is enough
    // to detect a longer file.
    let mut bytes = Vec::new();
    File::open(&path)
        .and_then(|file| {
            file.take(record.size_bytes.saturating_add(1))
                .read_to_end(&mut bytes)
        })
        .map_err(|error| io_error(&path, error))?;
    if bytes.len() as u64 != record.size_bytes || sha256_hex(&bytes) != record.sha256 {
        return Err(invalid(format!(
            "{} does not match its manifest size or SHA-256",
            record.path
        )));
    }
    Ok(bytes)
}

pub(crate) fn state_options() -> impl bincode::Options {
    use bincode::Options;
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_STATE_BYTES)
        .reject_trailing_bytes()
}

pub(crate) fn encode_state<T: Serialize>(state: &T) -> Result<Vec<u8>, AicError> {
    use bincode::Options;
    state_options()
        .serialize(state)
        .map_err(|error| invalid(format!("cannot encode state: {error}")))
}

pub(crate) fn decode_state<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, AicError> {
    use bincode::Options;
    state_options()
        .deserialize(bytes)
        .map_err(|error| invalid(format!("corrupt or truncated state.bin: {error}")))
}

/// Staging directory for one snapshot. Every staged file and directory is
/// synced before the rename that publishes it, and the parent afterwards;
/// dropping an unpublished stage removes it.
pub(crate) struct Stage {
    dir: tempfile::TempDir,
    destination: PathBuf,
    files: Vec<FileRecord>,
}

impl Stage {
    /// The destination must not exist. It is recorded as a normalized
    /// absolute path (see [`normalized_absolute`]); its parent is created.
    pub(crate) fn new(destination: &Path) -> Result<Self, AicError> {
        let absolute = normalized_absolute(destination)?;
        let name = absolute
            .file_name()
            .filter(|name| !name.is_empty() && *name != "." && *name != "..")
            .ok_or_else(|| invalid(format!("{} is not a directory name", destination.display())))?
            .to_owned();
        let parent = absolute
            .parent()
            .ok_or_else(|| invalid(format!("{} has no parent directory", destination.display())))?;
        std::fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
        let parent = parent.to_path_buf();
        let destination = parent.join(name);
        if destination.to_str().is_none() {
            return Err(invalid("snapshot path must be valid UTF-8"));
        }
        if std::fs::symlink_metadata(&destination).is_ok() {
            return Err(invalid(format!(
                "destination {} already exists; snapshots are written to a new directory",
                destination.display()
            )));
        }
        let dir = tempfile::Builder::new()
            .prefix(".fpm-state-snapshot-")
            .tempdir_in(&parent)
            .map_err(|error| io_error(&parent, error))?;
        Ok(Self {
            dir,
            destination,
            files: Vec::new(),
        })
    }

    /// Final absolute location recorded in configuration.
    pub(crate) fn destination(&self) -> &Path {
        &self.destination
    }

    pub(crate) fn staging(&self) -> &Path {
        self.dir.path()
    }

    /// Write a file that the manifest pins by digest.
    pub(crate) fn write_recorded(&mut self, relative: &str, bytes: &[u8]) -> Result<(), AicError> {
        self.write(relative, bytes)?;
        self.files.push(FileRecord {
            path: relative.into(),
            sha256: sha256_hex(bytes),
            size_bytes: bytes.len() as u64,
        });
        Ok(())
    }

    /// Record a file written by delegated machinery, such as a checkpoint
    /// manifest that pins its own payload.
    pub(crate) fn record_existing(&mut self, relative: &str) -> Result<FileRecord, AicError> {
        let path = self.dir.path().join(relative_path(relative)?);
        let bytes = std::fs::read(&path).map_err(|error| io_error(&path, error))?;
        let record = FileRecord {
            path: relative.into(),
            sha256: sha256_hex(&bytes),
            size_bytes: bytes.len() as u64,
        };
        self.files.push(record.clone());
        Ok(record)
    }

    pub(crate) fn write(&self, relative: &str, bytes: &[u8]) -> Result<(), AicError> {
        let path = self.dir.path().join(relative_path(relative)?);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
        }
        let mut file = File::create(&path).map_err(|error| io_error(&path, error))?;
        file.write_all(bytes)
            .map_err(|error| io_error(&path, error))?;
        file.sync_all().map_err(|error| io_error(&path, error))
    }

    pub(crate) fn take_files(&mut self) -> Vec<FileRecord> {
        let mut files = std::mem::take(&mut self.files);
        files.sort_by(|a, b| a.path.cmp(&b.path));
        files
    }

    /// Reserve the destination, then move the complete, synced stage into it.
    pub(crate) fn publish(self) -> Result<(), AicError> {
        sync_tree(self.dir.path())?;
        let staging = self.dir.keep();
        let destination = &self.destination;
        if let Err(error) = std::fs::create_dir(destination) {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(io_error(destination, error));
        }
        if let Err(error) = std::fs::rename(&staging, destination) {
            // Remove only our still-empty reservation; never remove other data.
            let _ = std::fs::remove_dir(destination);
            let _ = std::fs::remove_dir_all(&staging);
            return Err(io_error(destination, error));
        }
        if let Some(parent) = destination.parent() {
            if let Ok(directory) = File::open(parent) {
                let _ = directory.sync_all();
            }
        }
        Ok(())
    }
}

/// Flush every file and directory below `root`, then `root` itself, so
/// copied and delegated files are durable before the publishing rename.
fn sync_tree(root: &Path) -> Result<(), AicError> {
    for entry in std::fs::read_dir(root).map_err(|error| io_error(root, error))? {
        let path = entry.map_err(|error| io_error(root, error))?.path();
        let metadata = std::fs::symlink_metadata(&path).map_err(|error| io_error(&path, error))?;
        if metadata.is_dir() {
            sync_tree(&path)?;
        } else if metadata.is_file() {
            File::open(&path)
                .and_then(|file| file.sync_all())
                .map_err(|error| io_error(&path, error))?;
        }
    }
    File::open(root)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| io_error(root, error))
}
