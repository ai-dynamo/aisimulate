// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Snapshot-owned original source bytes, distinct from current runtime state.
//! Only typed filesystem fields are relocated; historical provenance strings
//! and references inside authenticated, byte-exact source manifests stay intact.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::{Stage, invalid, relative_path};
use crate::AicError;
use crate::perfmodel::fpm::{
    ForwardPassPerfModelConfig, ForwardPassPerfProvenance, LearnedFallbackKind, LearnedSourceKind,
    TemporaryMixedCompositionConfig,
};

pub(crate) fn map_source_paths(
    config: &mut ForwardPassPerfModelConfig,
    map: &mut impl FnMut(&Path) -> Result<PathBuf, AicError>,
) -> Result<(), AicError> {
    let learned = &mut config.estimator_config.fpm_learned;
    for path in learned
        .artifact_path
        .iter_mut()
        .chain(learned.checkpoint_path.iter_mut())
        .chain(learned.source.bundle_path.iter_mut())
    {
        *path = map(path)?;
    }
    if let Some(pin) = learned.source.resolved_source.as_mut() {
        for path in pin
            .path
            .iter_mut()
            .chain(pin.seed_path.iter_mut())
            .chain(pin.dependencies.iter_mut().map(|dep| &mut dep.path))
        {
            *path = map(path)?;
        }
    }
    if let Some(fallback) = learned.source.resolved_fallback.as_mut() {
        if let Some(policy) = fallback.policy.as_mut() {
            policy.original_a.manifest_path = map(&policy.original_a.manifest_path)?;
        }
        for remap in &mut fallback.dependency_remap {
            remap.path = map(&remap.path)?;
        }
    }
    Ok(())
}

pub(crate) fn map_native_pin(
    pin: &mut serde_json::Value,
    map: &mut impl FnMut(&Path) -> Result<PathBuf, AicError>,
) -> Result<(), AicError> {
    let mut provenance: ForwardPassPerfProvenance =
        serde_json::from_value(pin["provenance"].clone())
            .map_err(|e| invalid(format!("invalid native prior provenance: {e}")))?;
    for path in &mut provenance.config.systems_paths {
        *path = map(path)?;
    }
    if let Some(root) = provenance.selected_systems_root.as_mut() {
        *root = map(root)?;
    }
    if let Some(path) = provenance
        .config
        .estimator_config
        .fpm_interpolation
        .fpm_parquet_path
        .as_mut()
    {
        *path = map(path)?;
    }
    if let Some(profile) = provenance
        .config
        .estimator_config
        .op_level
        .execution_profile
        .as_mut()
    {
        profile.path = map(&profile.path)?;
    }
    pin["provenance"] = serde_json::to_value(provenance).map_err(invalid)?;
    Ok(())
}

pub(crate) fn map_owned_paths(
    config: &mut ForwardPassPerfModelConfig,
    map: &mut impl FnMut(&Path) -> Result<PathBuf, AicError>,
) -> Result<(), AicError> {
    map_source_paths(config, map)?;
    if let Some(pin) = config
        .estimator_config
        .fpm_learned
        .source
        .resolved_fallback
        .as_mut()
        .and_then(|fallback| fallback.native.as_mut())
    {
        map_native_pin(pin, map)?;
    }
    Ok(())
}

pub(crate) fn relative_owned(
    config: &mut ForwardPassPerfModelConfig,
    root: &Path,
) -> Result<(), AicError> {
    map_owned_paths(config, &mut |path| {
        let relative = path
            .strip_prefix(root)
            .map_err(|_| invalid("owned learned input escapes snapshot"))?;
        let text = relative
            .to_str()
            .ok_or_else(|| invalid("snapshot input must be UTF-8"))?;
        Ok(relative_path(text)?.to_owned())
    })
}

pub(crate) fn absolute_owned(
    config: &mut ForwardPassPerfModelConfig,
    root: &Path,
) -> Result<(), AicError> {
    map_owned_paths(config, &mut |path| {
        let text = path
            .to_str()
            .ok_or_else(|| invalid("snapshot input must be UTF-8"))?;
        Ok(root.join(relative_path(text)?))
    })
}

/// Archive the original source closure. A bundle keeps its relative tree so
/// its manifest bytes and digest remain valid. Legacy outside references use
/// separately authenticated remaps; the evidence bytes are never rewritten.
pub(crate) fn archive_source(
    stage: &mut Stage,
    config: &mut ForwardPassPerfModelConfig,
) -> Result<(), AicError> {
    let source = config
        .estimator_config
        .fpm_learned
        .source
        .resolved_source
        .clone()
        .ok_or_else(|| invalid("learned construction has no resolved source pin"))?;
    let source_root = match source.kind {
        LearnedSourceKind::Bundle => source
            .path
            .as_ref()
            .and_then(|path| path.parent())
            .map(Path::to_owned),
        LearnedSourceKind::Checkpoint => source.path.clone(),
        _ => None,
    };
    let namespace = if source.kind == LearnedSourceKind::Bundle {
        "source/bundle"
    } else {
        "source/checkpoint"
    };
    let mut paths = BTreeMap::new();
    if let Some(root) = &source_root {
        paths.insert(root.clone(), stage.destination().join(namespace));
    }
    let mut archived = BTreeMap::new();
    for (index, dependency) in source.dependencies.iter().enumerate() {
        let relative = if let Some(root) = &source_root {
            dependency
                .path
                .strip_prefix(root)
                .ok()
                .map(|suffix| Path::new(namespace).join(suffix))
        } else {
            None
        }
        .unwrap_or_else(|| {
            if source.kind == LearnedSourceKind::Artifact
                && source.path.as_ref() == Some(&dependency.path)
            {
                PathBuf::from("source/artifact.json")
            } else {
                Path::new("source/evidence")
                    .join(index.to_string())
                    .join(dependency.path.file_name().unwrap_or_default())
            }
        });
        if let Some(previous) = archived.get(&dependency.path) {
            if previous != &dependency.sha256 {
                return Err(invalid("source dependency has inconsistent hashes"));
            }
            continue;
        }
        let relative = relative
            .to_str()
            .ok_or_else(|| invalid("source path must be UTF-8"))?;
        stage.copy_recorded(
            &dependency.path,
            relative,
            &dependency.sha256,
            dependency.byte_length,
        )?;
        paths.insert(dependency.path.clone(), stage.destination().join(relative));
        archived.insert(dependency.path.clone(), dependency.sha256.clone());
    }
    // Existing portable bundles already contain these remaps in their original
    // manifest. Legacy manifests need their formerly external references pinned.
    let legacy = config
        .estimator_config
        .fpm_learned
        .source
        .resolved_fallback
        .as_ref()
        .is_some_and(|fallback| fallback.kind == LearnedFallbackKind::LegacyMixedInterpolationV1);
    if legacy {
        let remaps = super::super::source::portable_prior_remaps(config)?;
        config
            .estimator_config
            .fpm_learned
            .source
            .resolved_fallback
            .as_mut()
            .unwrap()
            .dependency_remap = remaps;
    }
    map_source_paths(config, &mut |path| {
        if let Some(mapped) = paths.get(path) {
            return Ok(mapped.clone());
        }
        // Bundle seed_path names the checkpoint directory, while dependency
        // records name its files. Preserve that directory's relative position.
        if let Some(root) = &source_root {
            if let Ok(relative) = path.strip_prefix(root) {
                return Ok(stage.destination().join(namespace).join(relative));
            }
        }
        Err(invalid(format!(
            "source closure omitted {}",
            path.display()
        )))
    })?;
    // Catalogue search roots are not live inputs after source selection. Native
    // prior roots have their own complete, independently packed dependency pin.
    config.systems_paths.clear();
    Ok(())
}

/// Relocate only the two typed filesystem-bearing parts of a mixed binding.
pub(crate) fn map_mixed_binding(
    state: &mut serde_json::Value,
    map: &mut impl FnMut(&Path) -> Result<PathBuf, AicError>,
) -> Result<(), AicError> {
    let mut policy: TemporaryMixedCompositionConfig =
        serde_json::from_value(state["binding"]["policy"].clone())
            .map_err(|e| invalid(format!("invalid mixed policy binding: {e}")))?;
    policy.original_a.manifest_path = map(&policy.original_a.manifest_path)?;
    state["binding"]["policy"] = serde_json::to_value(policy).map_err(invalid)?;
    let prior = &mut state["binding"]["prior"];
    if !prior.is_null() && prior.get("provenance").is_some() {
        map_native_pin(prior, map)?;
    }
    Ok(())
}
