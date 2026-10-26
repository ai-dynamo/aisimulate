// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Content fingerprint of the performance data a native estimator can read.
//!
//! Table loading is lazy and source resolution depends on directory
//! listings: family directories, sibling versions, declared donors and
//! cross-backend fill all decide which file supplies a table. A list of
//! files opened before saving would therefore miss both later lazy reads and
//! files whose presence changes resolution. The fingerprint instead covers
//! the complete inputs of one system's database (method
//! `systems_tree_sha256_v1`):
//!
//! - every regular file directly under the systems root, including the
//!   hardware YAML and shared manifests such as
//!   `perf_data_reuse_manifest.yaml`; and
//! - every regular file below the hardware YAML's `data_dir`.
//!
//! Each file contributes `"<sha256> <size> <relative path>\n"`; lines are
//! sorted by path and hashed again. Symlinks are followed. Other systems'
//! data directories and auxiliary subtrees are excluded. An external
//! `fpm_parquet_path` and its metadata sidecar are fingerprinted separately.
//! Packing copies exactly the fingerprinted files, preserving relative paths,
//! so a packed root resolves tables exactly like the original root.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{Stage, invalid, io_error, sha256_hex};
use crate::AicError;

pub(crate) const METHOD: &str = "systems_tree_sha256_v1";
pub(crate) const PACKED_SYSTEMS_DIR: &str = "perf_db/systems";
pub(crate) const PACKED_FPM_DIR: &str = "perf_db/fpm";
const MAX_DEPTH: usize = 32;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PerfDbFingerprint {
    pub method: String,
    /// The pinned configuration reads this snapshot's packed copy.
    pub packed: bool,
    pub system: String,
    pub data_dir: String,
    pub sha256: String,
    pub file_count: u64,
    pub size_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fpm_parquet: Option<ExternalFpmData>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExternalFpmData {
    pub sha256: String,
    pub sidecar_sha256: Option<String>,
}

struct Entry {
    relative: String,
    absolute: PathBuf,
    size: u64,
    sha256: String,
}

fn hash_file(path: &Path) -> Result<(u64, String), AicError> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|error| io_error(path, error))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1 << 16];
    let mut size = 0u64;
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| io_error(path, error))?;
        if count == 0 {
            break;
        }
        size += count as u64;
        digest.update(&buffer[..count]);
    }
    Ok((size, format!("{:x}", digest.finalize())))
}

fn data_dir(root: &Path, system: &str) -> Result<String, AicError> {
    let yaml = root.join(format!("{system}.yaml"));
    let text = std::fs::read_to_string(&yaml).map_err(|error| io_error(&yaml, error))?;
    let value: serde_yaml::Value = serde_yaml::from_str(&text)
        .map_err(|error| invalid(format!("cannot parse {}: {error}", yaml.display())))?;
    let data_dir = value
        .get("data_dir")
        .and_then(serde_yaml::Value::as_str)
        .ok_or_else(|| invalid(format!("{} has no data_dir", yaml.display())))?;
    super::relative_path(data_dir).map_err(|_| {
        invalid(format!(
            "{} data_dir must stay inside the systems root",
            yaml.display()
        ))
    })?;
    Ok(data_dir.trim_end_matches('/').to_owned())
}

fn utf8_name(path: &Path) -> Result<&str, AicError> {
    path.file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            invalid(format!(
                "non-UTF-8 performance data path {}",
                path.display()
            ))
        })
}

fn walk(dir: &Path, relative: &str, depth: usize, out: &mut Vec<Entry>) -> Result<(), AicError> {
    if depth > MAX_DEPTH {
        return Err(invalid(format!(
            "performance data deeper than {MAX_DEPTH} levels at {}",
            dir.display()
        )));
    }
    for entry in std::fs::read_dir(dir).map_err(|error| io_error(dir, error))? {
        let path = entry.map_err(|error| io_error(dir, error))?.path();
        let name = utf8_name(&path)?;
        let child = format!("{relative}/{name}");
        let metadata = std::fs::metadata(&path).map_err(|error| io_error(&path, error))?;
        if metadata.is_dir() {
            walk(&path, &child, depth + 1, out)?;
        } else if metadata.is_file() {
            let (size, sha256) = hash_file(&path)?;
            out.push(Entry {
                relative: child,
                absolute: path,
                size,
                sha256,
            });
        }
    }
    Ok(())
}

fn inventory(root: &Path, system: &str) -> Result<(String, Vec<Entry>), AicError> {
    let data_dir = data_dir(root, system)?;
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(root).map_err(|error| io_error(root, error))? {
        let path = entry.map_err(|error| io_error(root, error))?.path();
        let metadata = std::fs::metadata(&path).map_err(|error| io_error(&path, error))?;
        if metadata.is_file() {
            let (size, sha256) = hash_file(&path)?;
            entries.push(Entry {
                relative: utf8_name(&path)?.to_owned(),
                absolute: path,
                size,
                sha256,
            });
        }
    }
    let data_root = root.join(&data_dir);
    if !data_root.is_dir() {
        return Err(invalid(format!(
            "performance data directory {} does not exist",
            data_root.display()
        )));
    }
    walk(&data_root, &data_dir, 0, &mut entries)?;
    entries.sort_by(|a, b| a.relative.cmp(&b.relative));
    Ok((data_dir, entries))
}

fn external_fpm(path: Option<&Path>) -> Result<Option<ExternalFpmData>, AicError> {
    let Some(path) = path else {
        return Ok(None);
    };
    let sidecar = path.with_extension("metadata.json");
    Ok(Some(ExternalFpmData {
        sha256: hash_file(path)?.1,
        sidecar_sha256: if sidecar.is_file() {
            Some(hash_file(&sidecar)?.1)
        } else {
            None
        },
    }))
}

fn digest(entries: &[Entry]) -> String {
    let mut listing = String::new();
    for entry in entries {
        listing.push_str(&format!(
            "{} {} {}\n",
            entry.sha256, entry.size, entry.relative
        ));
    }
    sha256_hex(listing.as_bytes())
}

/// Fingerprint the data a native estimator rooted at `root` can read.
pub(crate) fn fingerprint(
    root: &Path,
    system: &str,
    fpm_parquet: Option<&Path>,
    packed: bool,
) -> Result<PerfDbFingerprint, AicError> {
    let (data_dir, entries) = inventory(root, system)?;
    Ok(PerfDbFingerprint {
        method: METHOD.into(),
        packed,
        system: system.into(),
        data_dir,
        sha256: digest(&entries),
        file_count: entries.len() as u64,
        size_bytes: entries.iter().map(|entry| entry.size).sum(),
        fpm_parquet: external_fpm(fpm_parquet)?,
    })
}

/// Copy the fingerprinted systems inputs into the stage. Returns the packed
/// root and, when supplied, the packed external FPM parquet location, both
/// as their final absolute paths.
pub(crate) fn pack(
    stage: &Stage,
    root: &Path,
    system: &str,
    fpm_parquet: Option<&Path>,
) -> Result<(PathBuf, Option<PathBuf>), AicError> {
    let (_, entries) = inventory(root, system)?;
    for entry in &entries {
        let target = stage
            .staging()
            .join(PACKED_SYSTEMS_DIR)
            .join(&entry.relative);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
        }
        std::fs::copy(&entry.absolute, &target).map_err(|error| io_error(&target, error))?;
        if hash_file(&target)? != (entry.size, entry.sha256.clone()) {
            return Err(invalid(format!(
                "performance data changed while packing {}",
                entry.relative
            )));
        }
    }
    let packed_parquet = match fpm_parquet {
        Some(path) => {
            let name = utf8_name(path)?;
            let directory = stage.staging().join(PACKED_FPM_DIR);
            std::fs::create_dir_all(&directory).map_err(|error| io_error(&directory, error))?;
            for source in [path.to_path_buf(), path.with_extension("metadata.json")] {
                if source == path || source.is_file() {
                    let target = directory.join(utf8_name(&source)?);
                    std::fs::copy(&source, &target).map_err(|error| io_error(&target, error))?;
                }
            }
            Some(stage.destination().join(PACKED_FPM_DIR).join(name))
        }
        None => None,
    };
    Ok((stage.destination().join(PACKED_SYSTEMS_DIR), packed_parquet))
}
