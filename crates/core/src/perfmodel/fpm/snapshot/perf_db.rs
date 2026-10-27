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
//!   `perf_data_reuse_manifest.yaml`;
//! - the hardware YAML's `data_dir`, resolved like the loader resolves it:
//!   every directory and regular file below it, or its absence (an external
//!   FPM parquet needs no data directory); and
//! - an external `fpm_parquet_path` and its `.metadata.json` sidecar.
//!
//! Each entry contributes a typed line (`dir`, `file <sha256> <size>`,
//! `dangling` for an unresolvable symlink, `absent`); lines are sorted and
//! hashed. Symlinks are followed like the loader follows them. Names that no
//! resolver reads are skipped: dot-prefixed entries (the Rust and Python
//! resolvers skip them in every directory listing, and lookups by name never
//! start with a dot) and non-UTF-8 names (lookups are UTF-8 paths). Other
//! top-level subtrees, such as other systems' data, are excluded.
//!
//! Estimators record the fingerprint when they first load a root in a
//! process ([`loaded`]); saving recomputes it and fails if the data changed
//! since. Packing copies exactly the fingerprinted entries, preserving
//! relative paths, so a packed root resolves tables exactly like the original.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use super::{Stage, invalid, io_error, normalized_absolute, sha256_hex};
use crate::AicError;

pub(crate) const METHOD: &str = "systems_tree_sha256_v1";
pub(crate) const PACKED_SYSTEMS_DIR: &str = "perf_db/systems";
pub(crate) const PACKED_FPM_DIR: &str = "perf_db/fpm";
const MAX_DEPTH: usize = 32;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PerfDbFingerprint {
    pub method: String,
    pub system: String,
    /// `data_dir` exactly as written in the hardware YAML.
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

#[derive(Clone, Debug, PartialEq)]
enum EntryKind {
    Dir,
    File { size: u64, sha256: String },
    Dangling { target: PathBuf },
    Absent,
}

struct Entry {
    /// Listing label: a systems-root-relative path, or `abs:<path>` for a
    /// data directory outside the root.
    label: String,
    absolute: PathBuf,
    kind: EntryKind,
}

impl Entry {
    fn line(&self) -> String {
        match &self.kind {
            EntryKind::Dir => format!("dir {}\n", self.label),
            EntryKind::File { size, sha256 } => format!("file {sha256} {size} {}\n", self.label),
            EntryKind::Dangling { .. } => format!("dangling {}\n", self.label),
            EntryKind::Absent => format!("absent {}\n", self.label),
        }
    }
}

struct Inventory {
    data_dir: String,
    /// Normalized data directory relative to the root; `None` when it
    /// resolves outside the root.
    data_inside: Option<PathBuf>,
    entries: Vec<Entry>,
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

fn yaml_data_dir(root: &Path, system: &str) -> Result<String, AicError> {
    let yaml = root.join(format!("{system}.yaml"));
    let text = std::fs::read_to_string(&yaml).map_err(|error| io_error(&yaml, error))?;
    let value: serde_yaml::Value = serde_yaml::from_str(&text)
        .map_err(|error| invalid(format!("cannot parse {}: {error}", yaml.display())))?;
    value
        .get("data_dir")
        .and_then(serde_yaml::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| invalid(format!("{} has no data_dir", yaml.display())))
}

/// Classify one directory entry like the loader sees it: symlinks are
/// followed and an unresolvable one is recorded but never read.
fn classify(path: &Path) -> Result<Option<EntryKind>, AicError> {
    let link = std::fs::symlink_metadata(path).map_err(|error| io_error(path, error))?;
    let metadata = if link.file_type().is_symlink() {
        match std::fs::metadata(path) {
            Ok(metadata) => metadata,
            Err(_) => {
                let target = std::fs::read_link(path).map_err(|error| io_error(path, error))?;
                return Ok(Some(EntryKind::Dangling { target }));
            }
        }
    } else {
        link
    };
    Ok(if metadata.is_dir() {
        Some(EntryKind::Dir)
    } else if metadata.is_file() {
        let (size, sha256) = hash_file(path)?;
        Some(EntryKind::File { size, sha256 })
    } else {
        None
    })
}

/// UTF-8, non-dot children in name order.
fn listed_children(dir: &Path) -> Result<Vec<(String, PathBuf)>, AicError> {
    let mut children = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|error| io_error(dir, error))? {
        let entry = entry.map_err(|error| io_error(dir, error))?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        children.push((name, entry.path()));
    }
    children.sort();
    Ok(children)
}

fn walk(dir: &Path, label: &str, depth: usize, out: &mut Vec<Entry>) -> Result<(), AicError> {
    if depth > MAX_DEPTH {
        return Err(invalid(format!(
            "performance data deeper than {MAX_DEPTH} levels at {}",
            dir.display()
        )));
    }
    for (name, path) in listed_children(dir)? {
        let child = format!("{label}/{name}");
        let Some(kind) = classify(&path)? else {
            continue;
        };
        let directory = kind == EntryKind::Dir;
        out.push(Entry {
            label: child.clone(),
            absolute: path.clone(),
            kind,
        });
        if directory {
            walk(&path, &child, depth + 1, out)?;
        }
    }
    Ok(())
}

fn inventory(root: &Path, system: &str) -> Result<Inventory, AicError> {
    let data_dir = yaml_data_dir(root, system)?;
    let mut entries = Vec::new();
    for (name, path) in listed_children(root)? {
        let Some(kind) = classify(&path)? else {
            continue;
        };
        // Top-level directories (other systems, profiles) are not inputs.
        if kind != EntryKind::Dir {
            entries.push(Entry {
                label: name,
                absolute: path,
                kind,
            });
        }
    }
    // The loader joins data_dir onto the root, so `./x`, `x/../y` and
    // absolute paths are all accepted.
    let root_normal = normalized_absolute(root)?;
    let data_path = normalized_absolute(&root_normal.join(&data_dir))?;
    let data_inside = data_path
        .strip_prefix(&root_normal)
        .ok()
        .filter(|relative| {
            relative
                .components()
                .all(|component| matches!(component, Component::Normal(_)))
        })
        .map(Path::to_path_buf);
    let label = match &data_inside {
        Some(relative) if relative.as_os_str().is_empty() => {
            return Err(invalid(format!(
                "{system}.yaml data_dir must name a directory below the systems root"
            )));
        }
        Some(relative) => relative.to_str().map(str::to_owned),
        None => data_path.to_str().map(|path| format!("abs:{path}")),
    }
    .ok_or_else(|| invalid("performance data directory must be valid UTF-8"))?;
    match std::fs::metadata(&data_path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => entries.push(Entry {
            label,
            absolute: data_path,
            kind: EntryKind::Absent,
        }),
        Err(error) => return Err(io_error(&data_path, error)),
        Ok(metadata) if metadata.is_dir() => {
            entries.push(Entry {
                label: label.clone(),
                absolute: data_path.clone(),
                kind: EntryKind::Dir,
            });
            walk(&data_path, &label, 0, &mut entries)?;
        }
        Ok(_) => {
            return Err(invalid(format!(
                "performance data path {} is not a directory",
                data_path.display()
            )));
        }
    }
    entries.sort_by(|a, b| a.label.cmp(&b.label));
    Ok(Inventory {
        data_dir,
        data_inside,
        entries,
    })
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

fn summarize(
    system: &str,
    inventory: &Inventory,
    fpm_parquet: Option<ExternalFpmData>,
) -> PerfDbFingerprint {
    let listing: String = inventory.entries.iter().map(Entry::line).collect();
    let files = inventory
        .entries
        .iter()
        .filter_map(|entry| match entry.kind {
            EntryKind::File { size, .. } => Some(size),
            _ => None,
        });
    PerfDbFingerprint {
        method: METHOD.into(),
        system: system.into(),
        data_dir: inventory.data_dir.clone(),
        sha256: sha256_hex(listing.as_bytes()),
        file_count: files.clone().count() as u64,
        size_bytes: files.sum(),
        fpm_parquet,
    }
}

/// Fingerprint the data a native estimator rooted at `root` can read now.
pub(crate) fn fingerprint(
    root: &Path,
    system: &str,
    fpm_parquet: Option<&Path>,
) -> Result<PerfDbFingerprint, AicError> {
    let inventory = inventory(root, system)?;
    Ok(summarize(system, &inventory, external_fpm(fpm_parquet)?))
}

/// The load-time fingerprint shared by a process's estimators.
pub(crate) type LoadedFingerprint = Arc<Result<PerfDbFingerprint, String>>;

/// Fingerprint `root` the first time this process loads it and reuse that
/// result afterwards, like the shared table memo reuses loaded tables. Data
/// that changes later is caught when saving recomputes the fingerprint.
/// Failures are recorded, not returned: they only prevent saving.
pub(crate) fn loaded(root: &Path, system: &str, fpm_parquet: Option<&Path>) -> LoadedFingerprint {
    type Key = (PathBuf, String, Option<PathBuf>);
    static LOADED: OnceLock<Mutex<HashMap<Key, LoadedFingerprint>>> = OnceLock::new();
    let normal = |path: &Path| normalized_absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let key = (normal(root), system.to_owned(), fpm_parquet.map(normal));
    let memo = LOADED.get_or_init(Default::default);
    if let Some(value) = memo.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Arc::clone(value);
    }
    let value = Arc::new(fingerprint(root, system, fpm_parquet).map_err(|error| error.to_string()));
    Arc::clone(
        memo.lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(key)
            .or_insert(value),
    )
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
    let inventory = inventory(root, system)?;
    if inventory.data_inside.is_none() {
        return Err(invalid(format!(
            "cannot pack performance data: {system}.yaml data_dir {:?} resolves outside the systems root {}; save without include_perf_db to reference it",
            inventory.data_dir,
            root.display()
        )));
    }
    let packed = stage.staging().join(PACKED_SYSTEMS_DIR);
    std::fs::create_dir_all(&packed).map_err(|error| io_error(&packed, error))?;
    for entry in &inventory.entries {
        let target = packed.join(&entry.label);
        match &entry.kind {
            EntryKind::Dir => {
                std::fs::create_dir_all(&target).map_err(|error| io_error(&target, error))?;
            }
            EntryKind::File { size, sha256 } => {
                if let Some(parent) = target.parent() {
                    std::fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
                }
                std::fs::copy(&entry.absolute, &target)
                    .map_err(|error| io_error(&target, error))?;
                if hash_file(&target)? != (*size, sha256.clone()) {
                    return Err(invalid(format!(
                        "performance data changed while packing {}",
                        entry.label
                    )));
                }
            }
            EntryKind::Dangling { target: link } => {
                #[cfg(unix)]
                std::os::unix::fs::symlink(link, &target)
                    .map_err(|error| io_error(&target, error))?;
                #[cfg(not(unix))]
                return Err(invalid(format!(
                    "cannot pack the unresolvable symlink {} ({})",
                    entry.label,
                    link.display()
                )));
            }
            EntryKind::Absent => {}
        }
    }
    let packed_parquet = match fpm_parquet {
        Some(path) => {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| invalid("fpm_parquet_path must name a UTF-8 file"))?;
            let directory = stage.staging().join(PACKED_FPM_DIR);
            std::fs::create_dir_all(&directory).map_err(|error| io_error(&directory, error))?;
            // The sidecar keeps its name relative to the parquet's stem.
            let sidecar = path.with_extension("metadata.json");
            let mut sources = vec![path.to_path_buf()];
            if sidecar.is_file() {
                sources.push(sidecar);
            }
            for source in sources {
                let target = directory.join(source.file_name().expect("named file"));
                std::fs::copy(&source, &target).map_err(|error| io_error(&target, error))?;
            }
            Some(stage.destination().join(PACKED_FPM_DIR).join(name))
        }
        None => None,
    };
    Ok((stage.destination().join(PACKED_SYSTEMS_DIR), packed_parquet))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn systems(root: &Path) {
        std::fs::create_dir_all(root.join("data/h200_sxm/vllm/1.0")).unwrap();
        std::fs::create_dir_all(root.join("data/b200_sxm/vllm/1.0")).unwrap();
        std::fs::write(root.join("h200_sxm.yaml"), "data_dir: data/h200_sxm/\n").unwrap();
        std::fs::write(root.join("data/h200_sxm/vllm/1.0/gemm_perf.parquet"), b"a").unwrap();
        std::fs::write(root.join("data/b200_sxm/vllm/1.0/gemm_perf.parquet"), b"b").unwrap();
    }

    #[test]
    fn fingerprint_covers_the_system_tree_and_external_fpm_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("systems");
        systems(&root);
        let parquet = dir.path().join("fpm/reviewed.parquet");
        std::fs::create_dir_all(parquet.parent().unwrap()).unwrap();
        std::fs::write(&parquet, b"rows").unwrap();
        std::fs::write(parquet.with_extension("metadata.json"), b"{}").unwrap();
        let base = fingerprint(&root, "h200_sxm", Some(&parquet)).unwrap();
        assert_eq!((base.file_count, base.size_bytes), (2, 25 + 1));
        assert_eq!(base.data_dir, "data/h200_sxm/");
        let external = base.fpm_parquet.clone().unwrap();
        assert_eq!(external.sha256, sha256_hex(b"rows"));
        assert_eq!(external.sidecar_sha256, Some(sha256_hex(b"{}")));
        // Another system's data and dot-prefixed entries no resolver reads
        // are outside the fingerprint.
        std::fs::write(root.join("data/b200_sxm/vllm/1.0/gemm_perf.parquet"), b"c").unwrap();
        std::fs::write(root.join("data/h200_sxm/vllm/1.0/.nfs0001"), b"stale").unwrap();
        std::fs::create_dir_all(root.join("data/h200_sxm/.cache")).unwrap();
        std::fs::write(root.join(".DS_Store"), b"x").unwrap();
        assert_eq!(
            fingerprint(&root, "h200_sxm", Some(&parquet)).unwrap(),
            base
        );
        // Empty directories change resolution, so they change the fingerprint.
        std::fs::create_dir_all(root.join("data/h200_sxm/vllm/0.9")).unwrap();
        let with_dir = fingerprint(&root, "h200_sxm", Some(&parquet)).unwrap();
        assert_ne!(with_dir, base);
        std::fs::remove_dir(root.join("data/h200_sxm/vllm/0.9")).unwrap();
        std::fs::write(parquet.with_extension("metadata.json"), b"{ }").unwrap();
        assert_ne!(
            fingerprint(&root, "h200_sxm", Some(&parquet)).unwrap(),
            base
        );
    }

    #[test]
    fn data_dir_resolves_like_the_loader() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("systems");
        systems(&root);
        let base = fingerprint(&root, "h200_sxm", None).unwrap();
        for spelled in ["./data/h200_sxm", "data/b200_sxm/../h200_sxm"] {
            std::fs::write(root.join("h200_sxm.yaml"), format!("data_dir: {spelled}\n")).unwrap();
            let same = fingerprint(&root, "h200_sxm", None).unwrap();
            assert_eq!((same.sha256.len(), same.file_count), (64, 2));
        }
        // A data directory outside the root is fingerprinted by absolute path
        // but cannot be packed.
        let outside = dir.path().join("elsewhere");
        std::fs::create_dir_all(outside.join("vllm/1.0")).unwrap();
        std::fs::write(outside.join("vllm/1.0/gemm_perf.parquet"), b"a").unwrap();
        std::fs::write(
            root.join("h200_sxm.yaml"),
            format!("data_dir: {}\n", outside.display()),
        )
        .unwrap();
        let absolute = fingerprint(&root, "h200_sxm", None).unwrap();
        assert_ne!(absolute.sha256, base.sha256);
        let stage = Stage::new(&dir.path().join("snapshot")).unwrap();
        let error = pack(&stage, &root, "h200_sxm", None).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("resolves outside the systems root"),
            "{error}"
        );
        // A missing data directory is a recorded absence, not an error: an
        // external FPM parquet needs no operation tables.
        std::fs::write(root.join("h200_sxm.yaml"), "data_dir: data/missing\n").unwrap();
        let absent = fingerprint(&root, "h200_sxm", None).unwrap();
        assert_eq!(absent.file_count, 1);
        std::fs::create_dir_all(root.join("data/missing")).unwrap();
        assert_ne!(fingerprint(&root, "h200_sxm", None).unwrap(), absent);
    }

    #[cfg(unix)]
    #[test]
    fn unreadable_names_and_dangling_links_are_recorded_like_the_loader_sees_them() {
        use std::os::unix::ffi::OsStrExt;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("systems");
        systems(&root);
        let base = fingerprint(&root, "h200_sxm", None).unwrap();
        let version = root.join("data/h200_sxm/vllm/1.0");
        // The loader cannot name a non-UTF-8 file, so it is skipped.
        std::fs::write(
            version.join(std::ffi::OsStr::from_bytes(b"bad\xff.parquet")),
            b"x",
        )
        .unwrap();
        assert_eq!(fingerprint(&root, "h200_sxm", None).unwrap(), base);
        // A dangling link is not readable now but would be if it resolved.
        std::os::unix::fs::symlink(dir.path().join("missing"), version.join("moe_perf.parquet"))
            .unwrap();
        let dangling = fingerprint(&root, "h200_sxm", None).unwrap();
        assert_ne!(dangling, base);
        let stage = Stage::new(&dir.path().join("snapshot")).unwrap();
        pack(&stage, &root, "h200_sxm", None).unwrap();
        let copied = fingerprint(&stage.staging().join(PACKED_SYSTEMS_DIR), "h200_sxm", None);
        assert_eq!(copied.unwrap(), dangling);
    }

    #[test]
    fn pack_copies_exactly_the_fingerprinted_entries() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("systems");
        systems(&root);
        std::fs::create_dir_all(root.join("data/h200_sxm/gemm/vllm/1.0")).unwrap();
        let parquet = dir.path().join("reviewed.parquet");
        std::fs::write(&parquet, b"rows").unwrap();
        std::fs::write(parquet.with_extension("metadata.json"), b"{}").unwrap();
        let stage = Stage::new(&dir.path().join("snapshot")).unwrap();
        let (packed_root, packed_parquet) =
            pack(&stage, &root, "h200_sxm", Some(&parquet)).unwrap();
        assert_eq!(packed_root, stage.destination().join(PACKED_SYSTEMS_DIR));
        assert_eq!(
            packed_parquet,
            Some(
                stage
                    .destination()
                    .join(PACKED_FPM_DIR)
                    .join("reviewed.parquet")
            )
        );
        let staged = stage.staging().join(PACKED_SYSTEMS_DIR);
        let staged_parquet = stage
            .staging()
            .join(PACKED_FPM_DIR)
            .join("reviewed.parquet");
        assert!(staged_parquet.with_extension("metadata.json").is_file());
        assert!(!staged.join("data/b200_sxm").exists());
        // Empty directories are part of resolution and are recreated.
        assert!(staged.join("data/h200_sxm/gemm/vllm/1.0").is_dir());
        assert_eq!(
            fingerprint(&staged, "h200_sxm", Some(&staged_parquet)).unwrap(),
            fingerprint(&root, "h200_sxm", Some(&parquet)).unwrap()
        );
        // A missing data directory packs as the hardware YAML alone.
        std::fs::write(root.join("h200_sxm.yaml"), "data_dir: data/gb300\n").unwrap();
        let stage = Stage::new(&dir.path().join("snapshot2")).unwrap();
        pack(&stage, &root, "h200_sxm", None).unwrap();
        let staged = stage.staging().join(PACKED_SYSTEMS_DIR);
        assert_eq!(
            fingerprint(&staged, "h200_sxm", None).unwrap(),
            fingerprint(&root, "h200_sxm", None).unwrap()
        );
    }

    #[test]
    fn load_time_fingerprints_are_memoized_per_root() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("systems");
        systems(&root);
        let first = loaded(&root, "h200_sxm", None);
        std::fs::write(root.join("data/h200_sxm/vllm/1.0/gemm_perf.parquet"), b"z").unwrap();
        // The memo keeps what this process loaded; a fresh fingerprint differs.
        assert!(Arc::ptr_eq(&first, &loaded(&root, "h200_sxm", None)));
        assert_ne!(
            first.as_ref().as_ref().unwrap(),
            &fingerprint(&root, "h200_sxm", None).unwrap()
        );
        assert!(loaded(&dir.path().join("missing"), "h200_sxm", None).is_err());
    }
}
