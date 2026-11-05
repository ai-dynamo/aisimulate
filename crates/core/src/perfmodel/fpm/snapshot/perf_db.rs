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
//! since. Tables that other engines parsed before the fingerprint stop being
//! shared then, because the shared table memo is keyed by paths, not content.
//! For the same reason a restore shares only tables an earlier restore loaded
//! right after verifying the same data, and only while no fingerprinted entry
//! has been written and no link followed on the way to the data retargeted
//! since ([`current`], [`verified`]): table families parse on first query, so
//! equal content alone would admit files that changed and changed back after
//! that restore.
//! Packing copies exactly the fingerprinted entries, preserving relative
//! paths, so a packed root resolves tables exactly like the original.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use serde::{Deserialize, Serialize};

use super::{Stage, invalid, io_error, normalized_absolute, sha256_hex};
use crate::AicError;
use crate::perfmodel::perf_database::{PerfDatabase, PerfTables, unshare_tables};

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
    /// [`stamp`] of the entry, and of the link itself for a symlink.
    stamp: String,
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
    entries: Vec<Entry>,
}

/// What changes whenever an entry is written, replaced or retargeted, even
/// back to its original bytes: its identity, size, and modification and
/// status change times. The status change time cannot be set back.
#[cfg(unix)]
fn stamp(metadata: &std::fs::Metadata) -> String {
    use std::os::unix::fs::MetadataExt;
    format!(
        "{}:{} {} {}.{} {}.{}",
        metadata.dev(),
        metadata.ino(),
        metadata.len(),
        metadata.mtime(),
        metadata.mtime_nsec(),
        metadata.ctime(),
        metadata.ctime_nsec()
    )
}

/// Without a status change time a rewrite can restore every field read here,
/// so these stamps never match ([`DataStamp::matches`]).
#[cfg(not(unix))]
fn stamp(metadata: &std::fs::Metadata) -> String {
    format!("{} {:?}", metadata.len(), metadata.modified().ok())
}

/// Symbolic links one resolution may follow, as Linux allows before `ELOOP`.
const MAX_LINK_HOPS: usize = 40;

/// [`stamp`]s of every symbolic link that resolving `path` follows, in order:
/// the links among its components and, in turn, those in each link's target.
/// Loaders read through `path`, so retargeting any of these links changes
/// what they read although no target is written; a retarget always creates
/// a new link.
fn link_stamps(path: &Path) -> Result<String, AicError> {
    let absolute = std::path::absolute(path).map_err(|error| io_error(path, error))?;
    let parts = |path: &Path| -> Vec<PathBuf> {
        path.components()
            .rev()
            .map(|component| PathBuf::from(component.as_os_str()))
            .collect()
    };
    let mut pending = parts(&absolute);
    let mut resolved = PathBuf::new();
    let mut hops = 0;
    let mut stamped = String::new();
    while let Some(part) = pending.pop() {
        match part.components().next() {
            Some(Component::ParentDir) => {
                resolved.pop();
            }
            Some(Component::Normal(_)) => {
                let candidate = resolved.join(&part);
                match std::fs::symlink_metadata(&candidate) {
                    Ok(link) if link.file_type().is_symlink() => {
                        hops += 1;
                        if hops > MAX_LINK_HOPS {
                            return Err(invalid(format!(
                                "too many symbolic links resolving {}",
                                path.display()
                            )));
                        }
                        stamped.push_str(&format!("{} -> ", stamp(&link)));
                        let target = std::fs::read_link(&candidate)
                            .map_err(|error| io_error(&candidate, error))?;
                        // An absolute target restarts at the root; a relative
                        // one continues from the link's directory.
                        pending.extend(parts(&target));
                    }
                    _ => resolved = candidate,
                }
            }
            // The root (or a prefix) replaces what was resolved so far.
            Some(Component::RootDir | Component::Prefix(_)) => resolved.push(&part),
            Some(Component::CurDir) | None => {}
        }
    }
    Ok(stamped)
}

/// [`link_stamps`] of `path`, then the [`stamp`] of what it resolves to.
fn path_stamp(path: &Path) -> Result<String, AicError> {
    let mut stamped = link_stamps(path)?;
    match std::fs::metadata(path) {
        Ok(metadata) => stamped.push_str(&stamp(&metadata)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => stamped.push_str("absent"),
        Err(error) => return Err(io_error(path, error)),
    }
    Ok(stamped)
}

/// The stamps of every input a fingerprint covers and of the links followed
/// on the way to them, taken before their contents were hashed. Matching stamps
/// mean no entry was written, replaced, added or removed, and no such link
/// retargeted, in between.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DataStamp(String);

impl DataStamp {
    fn matches(&self, other: &Self) -> bool {
        cfg!(unix) && self == other
    }
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
fn classify(path: &Path) -> Result<Option<(EntryKind, String)>, AicError> {
    let link = std::fs::symlink_metadata(path).map_err(|error| io_error(path, error))?;
    let (metadata, stamped) = if link.file_type().is_symlink() {
        match std::fs::metadata(path) {
            Ok(metadata) => {
                let stamped = format!("{}{}", link_stamps(path)?, stamp(&metadata));
                (metadata, stamped)
            }
            Err(_) => {
                let target = std::fs::read_link(path).map_err(|error| io_error(path, error))?;
                return Ok(Some((EntryKind::Dangling { target }, stamp(&link))));
            }
        }
    } else {
        let stamped = stamp(&link);
        (link, stamped)
    };
    Ok(if metadata.is_dir() {
        Some((EntryKind::Dir, stamped))
    } else if metadata.is_file() {
        let (size, sha256) = hash_file(path)?;
        Some((EntryKind::File { size, sha256 }, stamped))
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
        let Some((kind, stamp)) = classify(&path)? else {
            continue;
        };
        let directory = kind == EntryKind::Dir;
        out.push(Entry {
            label: child.clone(),
            absolute: path.clone(),
            kind,
            stamp,
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
        let Some((kind, stamp)) = classify(&path)? else {
            continue;
        };
        // Top-level directories (other systems, profiles) are not inputs.
        if kind != EntryKind::Dir {
            entries.push(Entry {
                label: name,
                absolute: path,
                kind,
                stamp,
            });
        }
    }
    // The loader joins data_dir onto the root verbatim and lets the OS
    // resolve it, so `./x`, `x/../y` and absolute paths are all accepted and
    // `..` follows symlinks. Read that exact path; label it by its lexical
    // normal form.
    let root_normal = normalized_absolute(root)?;
    let data_path = root.join(&data_dir);
    let data_normal = normalized_absolute(&data_path)?;
    let data_inside = data_normal
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
        None => data_normal.to_str().map(|path| format!("abs:{path}")),
    }
    .ok_or_else(|| invalid("performance data directory must be valid UTF-8"))?;
    match std::fs::metadata(&data_path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => entries.push(Entry {
            label,
            stamp: path_stamp(&data_path)?,
            absolute: data_path,
            kind: EntryKind::Absent,
        }),
        Err(error) => return Err(io_error(&data_path, error)),
        Ok(metadata) if metadata.is_dir() => {
            entries.push(Entry {
                label: label.clone(),
                absolute: data_path.clone(),
                kind: EntryKind::Dir,
                stamp: path_stamp(&data_path)?,
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
    Ok(Inventory { data_dir, entries })
}

/// The external parquet's data and the [`stamp`]s of the files it read.
fn external_fpm(path: Option<&Path>) -> Result<(Option<ExternalFpmData>, String), AicError> {
    let Some(path) = path else {
        return Ok((None, String::new()));
    };
    let mut stamped = path_stamp(path)?;
    let sha256 = hash_file(path)?.1;
    let sidecar = path.with_extension("metadata.json");
    let sidecar_sha256 = if sidecar.is_file() {
        stamped = format!("{stamped} {}", path_stamp(&sidecar)?);
        Some(hash_file(&sidecar)?.1)
    } else {
        None
    };
    Ok((
        Some(ExternalFpmData {
            sha256,
            sidecar_sha256,
        }),
        stamped,
    ))
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
    observe(root, system, fpm_parquet).map(|(fingerprint, _)| fingerprint)
}

/// [`fingerprint`] the data and stamp the entries it read.
fn observe(
    root: &Path,
    system: &str,
    fpm_parquet: Option<&Path>,
) -> Result<(PerfDbFingerprint, DataStamp), AicError> {
    let inventory = inventory(root, system)?;
    let (external, external_stamp) = external_fpm(fpm_parquet)?;
    let mut stamps: String = inventory
        .entries
        .iter()
        .map(|entry| format!("{} {}\n", entry.label, entry.stamp))
        .collect();
    stamps.push_str(&format!("root {}\n", path_stamp(root)?));
    stamps.push_str(&format!("external {external_stamp}\n"));
    Ok((
        summarize(system, &inventory, external),
        DataStamp(sha256_hex(stamps.as_bytes())),
    ))
}

/// The load-time fingerprint shared by a process's estimators.
pub(crate) type LoadedFingerprint = Arc<Result<PerfDbFingerprint, String>>;

/// The process-wide load-time fingerprint of one root, system and external
/// parquet, and the shared tables known to hold the data it describes.
#[derive(Default)]
struct Loaded {
    fingerprint: OnceLock<LoadedFingerprint>,
    /// Tables restored estimators loaded right after the data was
    /// fingerprinted again and still matched `fingerprint`, with the stamp of
    /// that data. Only these stay shared across a restore, and only while the
    /// stamp holds: the shared table memo is keyed by paths, so any other live
    /// tables for the root may have been parsed from another generation of
    /// its files, and these parse their families on first query.
    verified: Mutex<Vec<(Weak<PerfTables>, DataStamp)>>,
}

/// The process-wide load-time state for one root, system and external
/// parquet. Each key has its own cell, so concurrent first loads hash the
/// data once without holding the map lock while hashing.
fn loaded_cell(root: &Path, system: &str, fpm_parquet: Option<&Path>) -> Arc<Loaded> {
    type Key = (PathBuf, String, Option<PathBuf>);
    static LOADED: OnceLock<Mutex<HashMap<Key, Arc<Loaded>>>> = OnceLock::new();
    let normal = |path: &Path| normalized_absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let key = (normal(root), system.to_owned(), fpm_parquet.map(normal));
    Arc::clone(
        LOADED
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(key)
            .or_default(),
    )
}

/// Fingerprint `root` the first time this process loads it and reuse that
/// result afterwards, like the shared table memo reuses loaded tables. Data
/// that changes later is caught when saving recomputes the fingerprint.
/// Failures are recorded, not returned: they only prevent saving.
///
/// Tables another engine parsed before this fingerprint are no longer shared,
/// so the estimator loading next reads what was fingerprinted.
pub(crate) fn loaded(root: &Path, system: &str, fpm_parquet: Option<&Path>) -> LoadedFingerprint {
    Arc::clone(
        loaded_cell(root, system, fpm_parquet)
            .fingerprint
            .get_or_init(|| {
                let fingerprint =
                    fingerprint(root, system, fpm_parquet).map_err(|error| error.to_string());
                unshare_tables(root, system, &[]);
                Arc::new(fingerprint)
            }),
    )
}

/// Fingerprint `root` as it is now, before a restore loads it. When this
/// process has not loaded the root yet, the result is also its load-time
/// fingerprint, so the load that follows does not hash the same data again.
///
/// Shared tables stop being shared unless a restore already verified them
/// against this same data and no entry was written since ([`verified`]).
/// Their families parse on first query, possibly long after that restore,
/// so equal content alone would admit files that changed and changed back in
/// between. Any other tables for the root may have been parsed while its
/// files held other data: an estimator loaded after the load-time
/// fingerprint reuses it without hashing again. Repeated restores of
/// untouched data, such as one per replayed candidate, therefore parse the
/// tables once and reuse them afterwards.
///
/// Returns the stamp of the data to pass to [`verified`].
pub(crate) fn current(
    root: &Path,
    system: &str,
    fpm_parquet: Option<&Path>,
) -> Result<(PerfDbFingerprint, DataStamp), AicError> {
    let loaded = loaded_cell(root, system, fpm_parquet);
    let (current, stamp) = observe(root, system, fpm_parquet)?;
    let unchanged = loaded
        .fingerprint
        .get()
        .is_some_and(|loaded| loaded.as_ref().as_ref() == Ok(&current));
    let keep: Vec<_> = {
        let mut verified = loaded.verified.lock().unwrap_or_else(|e| e.into_inner());
        // A written entry never regains its stamp, so the others never match again.
        verified.retain(|(tables, known)| tables.strong_count() > 0 && known.matches(&stamp));
        if unchanged {
            verified
                .iter()
                .filter_map(|(tables, _)| tables.upgrade())
                .collect()
        } else {
            Vec::new()
        }
    };
    unshare_tables(root, system, &keep);
    loaded
        .fingerprint
        .get_or_init(|| Arc::new(Ok(current.clone())));
    Ok((current, stamp))
}

/// Record that a restore loaded `database`'s tables right after [`current`]
/// found the data under `root`, stamped `stamp`, unchanged from its load-time
/// fingerprint, so later restores may share them while the stamp holds.
pub(crate) fn verified(
    root: &Path,
    system: &str,
    fpm_parquet: Option<&Path>,
    database: &PerfDatabase,
    stamp: DataStamp,
) {
    let loaded = loaded_cell(root, system, fpm_parquet);
    let mut verified = loaded.verified.lock().unwrap_or_else(|e| e.into_inner());
    let tables = database.tables_arc();
    verified.retain(|(known, _)| {
        known.strong_count() > 0 && !std::ptr::eq(known.as_ptr(), Arc::as_ptr(tables))
    });
    verified.push((Arc::downgrade(tables), stamp));
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
    // The packed hardware YAML is a byte-exact copy, and entries are packed
    // at their normalized paths. The copied data_dir therefore names the
    // packed data only when it is spelled as plain names: `..` would need
    // the directories it passes through, and an absolute path would keep
    // naming the original data.
    let plain = Path::new(&inventory.data_dir)
        .components()
        .all(|component| matches!(component, Component::Normal(_) | Component::CurDir));
    if !plain {
        return Err(invalid(format!(
            "cannot pack performance data: {system}.yaml data_dir {:?} in {} must be a relative path of plain directory names, without `..` or an absolute root; save without include_perf_db to reference it",
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
    fn restore_check_seeds_the_load_time_fingerprint_without_replacing_it() {
        let dir = tempfile::tempdir().unwrap();
        let gemm = |root: &Path| root.join("data/h200_sxm/vllm/1.0/gemm_perf.parquet");
        // Before any load, the restore check's result is the load-time value.
        let fresh = dir.path().join("fresh");
        systems(&fresh);
        let (checked, _) = current(&fresh, "h200_sxm", None).unwrap();
        std::fs::write(gemm(&fresh), b"changed").unwrap();
        assert_eq!(loaded(&fresh, "h200_sxm", None).as_ref(), &Ok(checked));
        // After a load, a later check reports the disk but keeps what was loaded.
        let used = dir.path().join("used");
        systems(&used);
        let first = loaded(&used, "h200_sxm", None);
        std::fs::write(gemm(&used), b"changed").unwrap();
        let (now, _) = current(&used, "h200_sxm", None).unwrap();
        assert_ne!(first.as_ref(), &Ok(now));
        assert!(Arc::ptr_eq(&loaded(&used, "h200_sxm", None), &first));
    }

    #[test]
    fn stamps_change_when_data_changes_back() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("systems");
        systems(&root);
        let parquet = dir.path().join("reviewed.parquet");
        std::fs::write(&parquet, b"rows").unwrap();
        let observed = || observe(&root, "h200_sxm", Some(&parquet)).unwrap();
        let (base, stamp) = observed();
        assert_eq!(observed().1.matches(&stamp), cfg!(unix));
        // Coarse file clocks tick every few milliseconds.
        std::thread::sleep(std::time::Duration::from_millis(50));
        let gemm = root.join("data/h200_sxm/vllm/1.0/gemm_perf.parquet");
        for path in [&gemm, &parquet] {
            let bytes = std::fs::read(path).unwrap();
            std::fs::write(path, b"other").unwrap();
            std::fs::write(path, &bytes).unwrap();
            let (again, touched) = observed();
            assert_eq!(again, base);
            assert!(!touched.matches(&stamp));
        }
    }

    #[cfg(unix)]
    #[test]
    fn stamps_change_when_a_link_on_the_data_path_points_back() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("systems");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("h200_sxm.yaml"), "data_dir: data/h200_sxm/\n").unwrap();
        for tree in ["a", "b"] {
            let version = dir.path().join(tree).join("h200_sxm/vllm/1.0");
            std::fs::create_dir_all(&version).unwrap();
            std::fs::write(version.join("gemm_perf.parquet"), tree).unwrap();
        }
        // Retarget `systems/data` as `ln -sfn` does; neither tree is written.
        let point = |tree: &str| {
            let staged = root.join("data.next");
            std::os::unix::fs::symlink(dir.path().join(tree), &staged).unwrap();
            std::fs::rename(&staged, root.join("data")).unwrap();
        };
        point("a");
        let (base, stamp) = observe(&root, "h200_sxm", None).unwrap();
        // Coarse file clocks tick every few milliseconds.
        std::thread::sleep(std::time::Duration::from_millis(50));
        point("b");
        assert_ne!(observe(&root, "h200_sxm", None).unwrap().0, base);
        point("a");
        let (again, pointed_back) = observe(&root, "h200_sxm", None).unwrap();
        assert_eq!(again, base);
        assert!(!pointed_back.matches(&stamp));
    }

    #[cfg(unix)]
    #[test]
    fn stamps_change_when_a_link_behind_another_link_points_back() {
        let retarget = |link: &Path, target: &Path| {
            let staged = link.with_extension("next");
            std::os::unix::fs::symlink(target, &staged).unwrap();
            std::fs::rename(&staged, link).unwrap();
        };
        // `systems/data`, or a version inside the data tree, links to
        // `profiles/current`; only that intermediate link moves.
        for outer in ["data", "data/h200_sxm/vllm/1.0"] {
            let dir = tempfile::tempdir().unwrap();
            let root = dir.path().join("systems");
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join("h200_sxm.yaml"), "data_dir: data/h200_sxm/\n").unwrap();
            for tree in ["a", "b"] {
                let version = match outer {
                    "data" => dir.path().join(tree).join("h200_sxm/vllm/1.0"),
                    _ => dir.path().join(tree),
                };
                std::fs::create_dir_all(&version).unwrap();
                std::fs::write(version.join("gemm_perf.parquet"), tree).unwrap();
            }
            let current = dir.path().join("profiles/current");
            std::fs::create_dir_all(current.parent().unwrap()).unwrap();
            retarget(&current, &dir.path().join("a"));
            let link = root.join(outer);
            std::fs::create_dir_all(link.parent().unwrap()).unwrap();
            std::os::unix::fs::symlink(&current, &link).unwrap();
            let (base, stamp) = observe(&root, "h200_sxm", None).unwrap();
            // Coarse file clocks tick every few milliseconds.
            std::thread::sleep(std::time::Duration::from_millis(50));
            retarget(&current, &dir.path().join("b"));
            assert_ne!(observe(&root, "h200_sxm", None).unwrap().0, base, "{outer}");
            retarget(&current, &dir.path().join("a"));
            let (again, pointed_back) = observe(&root, "h200_sxm", None).unwrap();
            assert_eq!(again, base, "{outer}");
            assert!(!pointed_back.matches(&stamp), "{outer}");
        }
    }

    #[test]
    fn data_dir_resolves_like_the_loader() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("systems");
        systems(&root);
        let base = fingerprint(&root, "h200_sxm", None).unwrap();
        let stage = Stage::new(&dir.path().join("snapshot")).unwrap();
        let pack_error = |root: &Path| {
            pack(&stage, root, "h200_sxm", None)
                .unwrap_err()
                .to_string()
        };
        let yaml = |data_dir: &str| {
            std::fs::write(
                root.join("h200_sxm.yaml"),
                format!("data_dir: {data_dir}\n"),
            )
            .unwrap()
        };
        for spelled in ["./data/h200_sxm", "data/b200_sxm/../h200_sxm"] {
            yaml(spelled);
            let same = fingerprint(&root, "h200_sxm", None).unwrap();
            assert_eq!((same.sha256.len(), same.file_count), (64, 2));
        }
        // The copied YAML must name the packed data: `..` needs the
        // directories it passes through, which packing does not copy.
        assert!(pack_error(&root).contains("plain directory names"));
        // An absolute spelling of the same directory would keep naming the
        // original data.
        yaml(root.join("data/h200_sxm").to_str().unwrap());
        assert!(pack_error(&root).contains("plain directory names"));
        // A data directory outside the root is fingerprinted by absolute path
        // but cannot be packed.
        let outside = dir.path().join("elsewhere");
        std::fs::create_dir_all(outside.join("vllm/1.0")).unwrap();
        std::fs::write(outside.join("vllm/1.0/gemm_perf.parquet"), b"a").unwrap();
        yaml(outside.to_str().unwrap());
        let absolute = fingerprint(&root, "h200_sxm", None).unwrap();
        assert_ne!(absolute.sha256, base.sha256);
        assert!(pack_error(&root).contains("plain directory names"));
        // A missing data directory is a recorded absence, not an error: an
        // external FPM parquet needs no operation tables.
        yaml("data/missing");
        let absent = fingerprint(&root, "h200_sxm", None).unwrap();
        assert_eq!(absent.file_count, 1);
        std::fs::create_dir_all(root.join("data/missing")).unwrap();
        assert_ne!(fingerprint(&root, "h200_sxm", None).unwrap(), absent);
        // So is an intermediate directory that `..` passes through.
        yaml("data/missing/../h200_sxm");
        assert_eq!(fingerprint(&root, "h200_sxm", None).unwrap().file_count, 2);
        std::fs::remove_dir(root.join("data/missing")).unwrap();
        assert_eq!(fingerprint(&root, "h200_sxm", None).unwrap().file_count, 1);
    }

    #[cfg(unix)]
    #[test]
    fn data_dir_follows_symlinks_before_parent_components_like_the_loader() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("systems");
        systems(&root);
        // `data/link/..` is the link target's parent, not `data`.
        let other = dir.path().join("other");
        std::fs::create_dir_all(other.join("link_target")).unwrap();
        std::fs::create_dir_all(other.join("h200_sxm/vllm/1.0")).unwrap();
        std::fs::write(other.join("h200_sxm/vllm/1.0/gemm_perf.parquet"), b"z").unwrap();
        std::os::unix::fs::symlink(other.join("link_target"), root.join("data/link")).unwrap();
        std::fs::write(
            root.join("h200_sxm.yaml"),
            "data_dir: data/link/../h200_sxm\n",
        )
        .unwrap();
        let linked = fingerprint(&root, "h200_sxm", None).unwrap();
        std::fs::write(root.join("data/h200_sxm/vllm/1.0/gemm_perf.parquet"), b"x").unwrap();
        assert_eq!(fingerprint(&root, "h200_sxm", None).unwrap(), linked);
        std::fs::write(other.join("h200_sxm/vllm/1.0/gemm_perf.parquet"), b"y").unwrap();
        assert_ne!(fingerprint(&root, "h200_sxm", None).unwrap(), linked);
    }

    // Linux file systems accept any byte string as a name; macOS APFS rejects
    // invalid UTF-8 names before the fingerprint could skip them.
    #[cfg(target_os = "linux")]
    #[test]
    fn non_utf8_names_are_skipped_like_the_loader_skips_them() {
        use std::os::unix::ffi::OsStrExt;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("systems");
        systems(&root);
        let base = fingerprint(&root, "h200_sxm", None).unwrap();
        // The loader cannot name a non-UTF-8 file, so it is skipped.
        std::fs::write(
            root.join("data/h200_sxm/vllm/1.0")
                .join(std::ffi::OsStr::from_bytes(b"bad\xff.parquet")),
            b"x",
        )
        .unwrap();
        assert_eq!(fingerprint(&root, "h200_sxm", None).unwrap(), base);
    }

    #[cfg(unix)]
    #[test]
    fn dangling_links_are_recorded_like_the_loader_sees_them() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("systems");
        systems(&root);
        let base = fingerprint(&root, "h200_sxm", None).unwrap();
        let version = root.join("data/h200_sxm/vllm/1.0");
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
