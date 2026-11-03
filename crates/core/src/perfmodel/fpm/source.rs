// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Construction-time learned source resolution. Prediction never reads files.
//!
//! A bundle pins its own manifest and every input file. Catalogue lookup only
//! occurs for unresolved online requests; resolved absence is also a selection.

mod original_a;

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    ForwardPassPerfModelConfig, FpmLearnedConfig, LearnedAdaptation,
    MixedCompositionCalibrationConfig, MixedQualityGateConfig, OriginalASource,
    TemporaryMixedCompositionConfig, TemporaryMixedCompositionMode,
    TemporaryMixedCompositionScenario,
};
use crate::AicError;

pub(crate) const SOURCE_CONFIG_KEYS: &[&str] = &[
    "bundle_path",
    "source_policy",
    "fallback",
    "resolved_source",
    "resolved_fallback",
];
const MANIFEST_LIMIT: u64 = 16 * 1024 * 1024;

fn invalid(reason: impl std::fmt::Display) -> AicError {
    AicError::InvalidEngineConfig(format!("estimator_config.fpm_learned: {reason}"))
}
fn io(path: &Path, source: std::io::Error) -> AicError {
    AicError::Io {
        path: path.to_owned(),
        source,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearnedSourcePolicy {
    #[default]
    Auto,
    ExplicitOnly,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearnedFallback {
    #[default]
    Auto,
    Disabled,
}

/// Public JSON fields flattened into `estimator_config.fpm_learned`.
/// Kept separate from the historical online checkpoint fitting payload.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LearnedSourceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bundle_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "source_auto")]
    pub source_policy: LearnedSourcePolicy,
    #[serde(skip_serializing_if = "fallback_auto")]
    pub fallback: LearnedFallback,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolved_source: Option<ResolvedLearnedSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolved_fallback: Option<ResolvedLearnedFallback>,
}
fn source_auto(value: &LearnedSourcePolicy) -> bool {
    *value == LearnedSourcePolicy::Auto
}
fn fallback_auto(value: &LearnedFallback) -> bool {
    *value == LearnedFallback::Auto
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearnedSourceKind {
    None,
    Artifact,
    Checkpoint,
    Bundle,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearnedSeedKind {
    Artifact,
    TrainingCheckpoint,
    RuntimeCheckpoint,
}

/// A verified input file. Paths in resolved configuration are absolute; paths
/// in a bundle's dependency records are relative to that bundle manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedDependency {
    pub path: PathBuf,
    pub byte_length: u64,
    pub sha256: String,
    pub purpose: String,
}

/// Construction-local proof of the bytes a native source reader consumed.
/// Never serialized or accepted from configuration; no path-only cache can
/// stand in for this receipt. Readers record lengths from the verified bytes
/// or the same opened file they decode.
#[derive(Debug)]
pub(crate) struct VerifiedLearnedSource {
    pub(crate) source: ResolvedLearnedSource,
}

impl VerifiedLearnedSource {
    pub(crate) fn from_files(
        kind: LearnedSourceKind,
        path: &Path,
        seed_kind: LearnedSeedKind,
        dependencies: Vec<LearnedDependency>,
    ) -> Result<Self, AicError> {
        let path = absolute(path)?;
        let hash = dependencies
            .first()
            .ok_or_else(|| invalid("verified source has no root dependency"))?
            .sha256
            .clone();
        Ok(Self {
            source: ResolvedLearnedSource {
                kind,
                path: Some(path.clone()),
                sha256: Some(hash.clone()),
                seed_kind: Some(seed_kind),
                seed_path: Some(path),
                seed_sha256: Some(hash),
                dependencies,
                recipe: None,
                provenance: None,
            },
        })
    }
}

/// The seed's recorded offline trainer and parameters, not an instruction to
/// collect observations or fit missing stores in the runtime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedTrainingRecipe {
    pub name: String,
    pub features: Vec<String>,
    pub target: String,
    pub trainer_version: String,
    pub hyperparameters: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedSourceReference {
    pub reference: String,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedSourceProvenance {
    pub anchor_kind: String,
    pub accepted_rows: BTreeMap<String, usize>,
    pub sources: Vec<LearnedSourceReference>,
    pub evaluation_sources: Vec<LearnedSourceReference>,
    pub c_admission_lineage: Option<serde_json::Map<String, serde_json::Value>>,
}

/// Generated immutable selection, including an explicit absence pin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedLearnedSource {
    pub kind: LearnedSourceKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_kind: Option<LearnedSeedKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_sha256: Option<String>,
    #[serde(default)]
    pub dependencies: Vec<LearnedDependency>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<LearnedTrainingRecipe>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<LearnedSourceProvenance>,
}
impl ResolvedLearnedSource {
    pub fn absent() -> Self {
        Self {
            kind: LearnedSourceKind::None,
            path: None,
            sha256: None,
            seed_kind: None,
            seed_path: None,
            seed_sha256: None,
            dependencies: Vec::new(),
            recipe: None,
            provenance: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearnedFallbackKind {
    None,
    MixedInterpolationV1,
    LegacyMixedInterpolationV1,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedLearnedFallback {
    pub kind: LearnedFallbackKind,
    /// Exact constructed interpolation provenance; populated by the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependency_remap: Vec<LearnedDependencyRemap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<TemporaryMixedCompositionConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedBundleSeed {
    pub kind: LearnedSeedKind,
    pub path: PathBuf,
    pub sha256: String,
    pub adaptations: Vec<LearnedAdaptation>,
    pub covered_stores: Vec<String>,
}
/// A verified relocation of a reference inside unchanged original-A bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedDependencyRemap {
    pub original_path: PathBuf,
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedBundlePrior {
    #[serde(default)]
    pub dependency_remap: Vec<LearnedDependencyRemap>,
    pub original_a: OriginalASource,
    #[serde(default)]
    pub calibration: MixedCompositionCalibrationConfig,
    #[serde(default)]
    pub quality_gate: Option<MixedQualityGateConfig>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedBundleManifest {
    pub kind: String,
    pub schema_version: u32,
    pub producer_version: String,
    pub recipe_version: String,
    pub identity: serde_json::Value,
    pub seed: LearnedBundleSeed,
    pub training_recipe: LearnedTrainingRecipe,
    pub provenance: LearnedSourceProvenance,
    #[serde(default)]
    pub prior: Option<LearnedBundlePrior>,
    pub dependencies: Vec<LearnedDependency>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedCatalogueEntry {
    pub identity: serde_json::Value,
    pub recipe_version: String,
    pub bundle_path: PathBuf,
    pub sha256: String,
    pub default: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedCatalogue {
    pub kind: String,
    pub schema_version: u32,
    pub entries: Vec<LearnedCatalogueEntry>,
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
fn absolute(path: &Path) -> Result<PathBuf, AicError> {
    super::snapshot::normalized_absolute(path)
}
fn relative(root: &Path, path: &Path) -> Result<PathBuf, AicError> {
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(invalid(format!(
            "bundle paths must be relative without parent components: {}",
            path.display()
        )));
    }
    let joined = root.join(path);
    // Relative syntax alone does not confine symlinks to the bundle directory.
    let canonical_root = std::fs::canonicalize(root).map_err(|err| io(root, err))?;
    let canonical = std::fs::canonicalize(&joined).map_err(|err| io(&joined, err))?;
    if !canonical.starts_with(canonical_root) {
        return Err(invalid("bundle dependency escapes its root"));
    }
    Ok(joined)
}
fn read_json(path: &Path) -> Result<(Vec<u8>, serde_json::Value), AicError> {
    let file = std::fs::File::open(path).map_err(|err| io(path, err))?;
    let mut bytes = Vec::new();
    file.take(MANIFEST_LIMIT + 1)
        .read_to_end(&mut bytes)
        .map_err(|err| io(path, err))?;
    if bytes.len() as u64 > MANIFEST_LIMIT {
        return Err(invalid("manifest exceeds 16 MiB"));
    }
    // YAML's JSON-compatible value reader rejects duplicate object keys.
    let value: serde_yaml::Value = serde_yaml::from_slice(&bytes)
        .map_err(|err| invalid(format!("{}: {err}", path.display())))?;
    let value = serde_json::to_value(value).map_err(|err| invalid(err.to_string()))?;
    Ok((bytes, value))
}
fn dependency(path: &Path, purpose: impl Into<String>) -> Result<LearnedDependency, AicError> {
    let mut file = std::fs::File::open(path).map_err(|err| io(path, err))?;
    let mut digest = Sha256::new();
    let mut byte_length = 0u64;
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer).map_err(|err| io(path, err))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
        byte_length += count as u64;
    }
    Ok(LearnedDependency {
        path: absolute(path)?,
        byte_length,
        sha256: format!("{:x}", digest.finalize()),
        purpose: purpose.into(),
    })
}
fn check_hash(expected: &str, actual: &str) -> Result<(), AicError> {
    if !valid_hash(expected) || !expected.eq_ignore_ascii_case(actual) {
        return Err(invalid(
            "checksum mismatch: source_sha256 or dependency sha256 does not match contents",
        ));
    }
    Ok(())
}
fn required_dependency<'a>(
    dependencies: &'a [LearnedDependency],
    path: &Path,
    digest: &str,
) -> Result<&'a LearnedDependency, AicError> {
    let found = dependencies
        .iter()
        .find(|entry| entry.path == path)
        .ok_or_else(|| invalid(format!("dependency record missing for {}", path.display())))?;
    check_hash(digest, &found.sha256)?;
    Ok(found)
}
fn string<'a>(value: &'a serde_json::Value, key: &str) -> Result<&'a str, AicError> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid(format!("seed requires {key}")))
}

/// Inspect checkpoint file references, independently of the later native payload
/// decoder. Every referenced file must be covered by the bundle's closure.
fn checkpoint_dependencies(
    path: &Path,
    include_training_state: bool,
) -> Result<(serde_json::Value, Vec<LearnedDependency>), AicError> {
    let manifest_path = path.join("manifest.json");
    let (bytes, manifest) = read_json(&manifest_path)?;
    let mut files = vec![LearnedDependency {
        path: absolute(&manifest_path)?,
        byte_length: bytes.len() as u64,
        sha256: format!("{:x}", Sha256::digest(&bytes)),
        purpose: "checkpoint_manifest".into(),
    }];
    for (field, hash_field, purpose) in [
        ("artifact_file", "artifact_sha256", "learned_artifact"),
        ("state_file", "state_sha256", "runtime_state"),
    ] {
        if (field != "state_file" || include_training_state)
            && manifest.get(field).is_some_and(|v| !v.is_null())
        {
            let target = relative(path, Path::new(string(&manifest, field)?))?;
            let dep = dependency(&target, purpose)?;
            check_hash(string(&manifest, hash_field)?, &dep.sha256)?;
            files.push(dep);
        }
    }
    if include_training_state
        && let Some(stores) = manifest
            .get("stores")
            .and_then(serde_json::Value::as_object)
    {
        for store in stores.values() {
            let target = relative(path, Path::new(string(store, "rows_file")?))?;
            let dep = dependency(&target, "training_rows")?;
            check_hash(string(store, "rows_sha256")?, &dep.sha256)?;
            files.push(dep);
        }
    }
    Ok((manifest, files))
}

fn validate_provenance(provenance: &LearnedSourceProvenance) -> Result<(), AicError> {
    if !matches!(provenance.anchor_kind.as_str(), "a" | "b") || provenance.sources.is_empty() {
        return Err(invalid(
            "bundle requires A or B anchor provenance and immutable source references",
        ));
    }
    for reference in provenance
        .sources
        .iter()
        .chain(&provenance.evaluation_sources)
    {
        if reference.reference.is_empty() || !valid_hash(&reference.sha256) {
            return Err(invalid("invalid provenance source reference"));
        }
    }
    if provenance.sources.iter().any(|source| {
        provenance
            .evaluation_sources
            .iter()
            .any(|eval| eval.sha256.eq_ignore_ascii_case(&source.sha256))
    }) {
        return Err(invalid("training and evaluation source digests overlap"));
    }
    Ok(())
}

fn validate_bundle(
    path: &Path,
    expected: Option<&str>,
    config: &ForwardPassPerfModelConfig,
) -> Result<(ResolvedLearnedSource, Option<LearnedBundlePrior>), AicError> {
    let (bytes, value) = read_json(path)?;
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    if let Some(expected) = expected {
        check_hash(expected, &sha256)?;
    }
    let bundle: LearnedBundleManifest =
        serde_json::from_value(value).map_err(|err| invalid(format!("invalid bundle: {err}")))?;
    if bundle.kind != "aisimulate_learned_bundle"
        || bundle.schema_version != 1
        || bundle.producer_version.is_empty()
        || bundle.recipe_version.is_empty()
    {
        return Err(invalid(
            "unsupported bundle kind/version or missing producer/recipe version",
        ));
    }
    let identity = super::learned_online::canonical_learned_identity(config)?;
    if bundle.identity != identity {
        return Err(invalid(
            "bundle identity does not match canonical deployment identity",
        ));
    }
    validate_provenance(&bundle.provenance)?;
    if bundle.seed.adaptations.is_empty()
        || bundle
            .seed
            .adaptations
            .iter()
            .enumerate()
            .any(|(index, value)| bundle.seed.adaptations[..index].contains(value))
        || (bundle.seed.kind == LearnedSeedKind::Artifact
            && bundle
                .seed
                .adaptations
                .iter()
                .any(|mode| *mode != LearnedAdaptation::Frozen))
        || (bundle.seed.kind == LearnedSeedKind::RuntimeCheckpoint
            && bundle
                .seed
                .adaptations
                .contains(&LearnedAdaptation::CausalSelector))
    {
        return Err(invalid(
            "bundle advertises unsupported or duplicate adaptation capabilities",
        ));
    }
    let adaptation = config.estimator_config.fpm_learned.adaptation;
    if !bundle.seed.adaptations.contains(&adaptation)
        || (adaptation != LearnedAdaptation::Frozen
            && bundle.seed.kind == LearnedSeedKind::Artifact)
        || (adaptation == LearnedAdaptation::CausalSelector
            && bundle.seed.kind != LearnedSeedKind::TrainingCheckpoint)
    {
        return Err(invalid(
            "bundle seed does not support the requested adaptation",
        ));
    }
    if bundle.seed.kind != LearnedSeedKind::RuntimeCheckpoint
        && bundle.provenance.c_admission_lineage.is_some()
    {
        return Err(invalid(
            "pristine training seeds cannot claim resumed C admission lineage",
        ));
    }
    if bundle.seed.kind == LearnedSeedKind::RuntimeCheckpoint
        && bundle.provenance.c_admission_lineage.is_none()
    {
        return Err(invalid(
            "runtime checkpoint bundle requires C admission lineage",
        ));
    }
    let root = path
        .parent()
        .ok_or_else(|| invalid("bundle manifest needs a parent directory"))?;
    let mut dependencies = vec![LearnedDependency {
        path: absolute(path)?,
        byte_length: bytes.len() as u64,
        sha256: sha256.clone(),
        purpose: "bundle_manifest".into(),
    }];
    let mut seen = BTreeSet::new();
    for entry in &bundle.dependencies {
        let file = relative(root, &entry.path)?;
        if !seen.insert(file.clone()) {
            return Err(invalid("duplicate bundle dependency"));
        }
        let actual = dependency(&file, &entry.purpose)?;
        check_hash(&entry.sha256, &actual.sha256)?;
        if entry.byte_length != actual.byte_length || entry.purpose.is_empty() {
            return Err(invalid("invalid dependency length or purpose"));
        }
        dependencies.push(actual);
    }
    let seed_path = relative(root, &bundle.seed.path)?;
    let artifact_path;
    if bundle.seed.kind == LearnedSeedKind::Artifact {
        required_dependency(&dependencies, &seed_path, &bundle.seed.sha256)?;
        artifact_path = seed_path.clone();
    } else {
        let (manifest, closure) = checkpoint_dependencies(&seed_path, true)?;
        let schema = string(&manifest, "schema")?;
        let expected_schema = if bundle.seed.kind == LearnedSeedKind::TrainingCheckpoint {
            "aisimulate_fpm_training_checkpoint"
        } else {
            "aisimulate_fpm_online_checkpoint"
        };
        if schema != expected_schema || manifest.get("identity") != Some(&identity) {
            return Err(invalid("bundle checkpoint kind/identity mismatch"));
        }
        for dep in closure {
            required_dependency(&dependencies, &dep.path, &dep.sha256)?;
        }
        required_dependency(
            &dependencies,
            &seed_path.join("manifest.json"),
            &bundle.seed.sha256,
        )?;
        artifact_path = relative(&seed_path, Path::new(string(&manifest, "artifact_file")?))?;
        if bundle.seed.kind == LearnedSeedKind::TrainingCheckpoint {
            if manifest
                .get("sklearn_version")
                .and_then(serde_json::Value::as_str)
                != Some(bundle.training_recipe.trainer_version.as_str())
            {
                return Err(invalid(
                    "bundle trainer version differs from seed checkpoint",
                ));
            }
            let counts: BTreeMap<String, usize> = manifest
                .get("stores")
                .and_then(serde_json::Value::as_object)
                .ok_or_else(|| invalid("training checkpoint requires stores"))?
                .iter()
                .map(|(kind, store)| {
                    store
                        .get("row_count")
                        .and_then(serde_json::Value::as_u64)
                        .and_then(|n| usize::try_from(n).ok())
                        .map(|n| (kind.clone(), n))
                        .ok_or_else(|| invalid("invalid checkpoint row_count"))
                })
                .collect::<Result<_, _>>()?;
            if counts != bundle.provenance.accepted_rows {
                return Err(invalid(
                    "bundle accepted rows do not match checkpoint stores",
                ));
            }
        }
    }
    let (artifact_bytes, artifact) = read_json(&artifact_path)?;
    let parsed_digest = format!("{:x}", Sha256::digest(&artifact_bytes));
    let parsed_record = required_dependency(&dependencies, &artifact_path, &parsed_digest)?;
    if parsed_record.byte_length != artifact_bytes.len() as u64 {
        return Err(invalid(
            "parsed artifact length differs from verified dependency",
        ));
    }
    let model = super::learned::LearnedForwardPassModel::from_json(
        &serde_json::to_string(&artifact).map_err(|err| invalid(err.to_string()))?,
    )?;
    if model.metadata().get("identity") != Some(&identity) {
        return Err(invalid("bundle artifact identity mismatch"));
    }
    let recipe = &bundle.training_recipe;
    if recipe.features != model.feature_names()
        || artifact.get("target").and_then(serde_json::Value::as_str)
            != Some(recipe.target.as_str())
    {
        return Err(invalid(
            "bundle training feature order/target differs from seed",
        ));
    }
    if recipe.name.trim().is_empty()
        || recipe.trainer_version.trim().is_empty()
        || recipe.hyperparameters.is_empty()
        || model
            .metadata()
            .get("model")
            .and_then(serde_json::Value::as_str)
            != Some(recipe.name.as_str())
        || model
            .metadata()
            .get("hyperparameters")
            .and_then(serde_json::Value::as_object)
            != Some(&recipe.hyperparameters)
    {
        return Err(invalid("bundle training recipe differs from seed metadata"));
    }
    let covered: BTreeSet<_> = bundle.seed.covered_stores.iter().cloned().collect();
    let actual: BTreeSet<_> = artifact
        .get("stores")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| invalid("artifact stores missing"))?
        .keys()
        .cloned()
        .collect();
    if covered.len() != bundle.seed.covered_stores.len() || covered != actual {
        return Err(invalid("bundle covered stores do not match artifact"));
    }
    let mut prior = bundle.prior;
    if let Some(prior) = &mut prior {
        if bundle.provenance.anchor_kind != "a" {
            return Err(invalid("a bundle prior requires original A provenance"));
        }
        prior.original_a.manifest_path = relative(root, &prior.original_a.manifest_path)?;
        let digest = prior
            .original_a
            .source_sha256
            .as_deref()
            .ok_or_else(|| invalid("bundle prior requires source_sha256"))?;
        let verified = required_dependency(&dependencies, &prior.original_a.manifest_path, digest)?;
        prior.original_a.source_sha256 = Some(verified.sha256.clone());
        let mut references = BTreeSet::new();
        for remap in &mut prior.dependency_remap {
            if remap.original_path.as_os_str().is_empty()
                || !references.insert(remap.original_path.clone())
            {
                return Err(invalid("empty or duplicate prior dependency remap"));
            }
            remap.path = relative(root, &remap.path)?;
            let verified = required_dependency(&dependencies, &remap.path, &remap.sha256)?;
            remap.sha256.clone_from(&verified.sha256);
        }
    }
    Ok((
        ResolvedLearnedSource {
            kind: LearnedSourceKind::Bundle,
            path: Some(absolute(path)?),
            sha256: Some(sha256),
            seed_kind: Some(bundle.seed.kind),
            seed_path: Some(seed_path),
            seed_sha256: Some(bundle.seed.sha256.to_ascii_lowercase()),
            dependencies,
            recipe: Some(bundle.training_recipe),
            provenance: Some(bundle.provenance),
        },
        prior,
    ))
}

fn catalogue_source(
    config: &ForwardPassPerfModelConfig,
) -> Result<Option<(PathBuf, String)>, AicError> {
    if !config.unknown_identity_facts().is_empty() {
        return Ok(None);
    }
    let identity = super::learned_online::canonical_learned_identity(config)?;
    // Share the existing root transport with other native estimators; source
    // choice remains in this Rust catalogue resolver.
    #[cfg(feature = "python")]
    let roots = if config.systems_paths.is_empty() {
        crate::py::resolve_forward_pass_systems_roots()?
    } else {
        config.systems_paths.clone()
    };
    #[cfg(not(feature = "python"))]
    let roots = config.systems_paths.clone();
    for root in &roots {
        let directory = absolute(&root.join("fpm_learned"))?;
        let path = directory.join("index.json");
        if !path.try_exists().map_err(|err| io(&path, err))? {
            continue;
        }
        let (_, value) = read_json(&path)?;
        let index: LearnedCatalogue = serde_json::from_value(value)
            .map_err(|err| invalid(format!("invalid learned catalogue: {err}")))?;
        if index.kind != "aisimulate_learned_catalogue" || index.schema_version != 1 {
            return Err(invalid("unsupported learned catalogue kind/version"));
        }
        let matches: Vec<_> = index
            .entries
            .iter()
            .filter(|entry| entry.identity == identity)
            .collect();
        if matches.is_empty() {
            continue;
        }
        let defaults: Vec<_> = matches.iter().filter(|entry| entry.default).collect();
        if defaults.len() != 1 {
            return Err(invalid(
                "catalogue identity must have exactly one default bundle",
            ));
        }
        let entry = defaults[0];
        if !valid_hash(&entry.sha256) || entry.recipe_version.is_empty() {
            return Err(invalid("invalid selected catalogue entry"));
        }
        let selected = relative(&directory, &entry.bundle_path)?;
        // Verify the entry's recipe binding, not merely its content digest.
        let (_, value) = read_json(&selected)?;
        if value
            .get("recipe_version")
            .and_then(serde_json::Value::as_str)
            != Some(&entry.recipe_version)
            || value
                .pointer("/seed/kind")
                .and_then(serde_json::Value::as_str)
                != Some("training_checkpoint")
        {
            return Err(invalid(
                "catalogue default must bind the advertised pristine training recipe",
            ));
        }
        return Ok(Some((selected, entry.sha256.clone())));
    }
    Ok(None)
}

fn prior_reference(
    manifest: &Path,
    reference: &Path,
    expected: &str,
    remaps: &[LearnedDependencyRemap],
) -> Result<PathBuf, AicError> {
    if let Some(remap) = remaps.iter().find(|entry| entry.original_path == reference) {
        check_hash(expected, &remap.sha256)?;
        return absolute(&remap.path);
    }
    let path = if reference.is_absolute() {
        reference.to_owned()
    } else {
        manifest.parent().unwrap_or(Path::new(".")).join(reference)
    };
    absolute(&path)
}

fn prior_dependencies(
    policy: &TemporaryMixedCompositionConfig,
    remaps: &[LearnedDependencyRemap],
) -> Result<Vec<LearnedDependency>, AicError> {
    let path = &policy.original_a.manifest_path;
    let manifest = original_a::read(
        path,
        policy
            .original_a
            .source_sha256
            .as_deref()
            .ok_or_else(|| invalid("prior is not pinned"))?,
    )?;
    let mut dependencies = vec![manifest.manifest];
    for (references, purpose) in [
        (manifest.sources, "original_a_observations"),
        (manifest.seeds, "original_a_seed"),
    ] {
        for reference in references {
            let original = Path::new(&reference.path);
            let digest = &reference.sha256;
            let target = prior_reference(path, original, digest, remaps)?;
            let dep = dependency(&target, purpose)?;
            check_hash(digest, &dep.sha256)?;
            dependencies.push(dep);
            if let Some(checkpoint_hash) = &reference.checkpoint_manifest_sha256 {
                let checkpoint_ref = original
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join("manifest.json");
                let checkpoint = prior_reference(path, &checkpoint_ref, checkpoint_hash, remaps)?;
                let dep = dependency(&checkpoint, "original_a_checkpoint_manifest")?;
                check_hash(checkpoint_hash, &dep.sha256)?;
                dependencies.push(dep);
            }
        }
    }
    for remap in remaps {
        let dep = dependency(&remap.path, "original_a_relocated_input")?;
        check_hash(&remap.sha256, &dep.sha256)?;
        if !dependencies
            .iter()
            .any(|existing| existing.path == dep.path)
        {
            return Err(invalid("unused original-A dependency remap"));
        }
    }
    Ok(dependencies)
}

/// References inside an unchanged legacy original-A manifest, with verified
/// current targets. Portable snapshots relocate only the target side.
pub(crate) fn portable_prior_remaps(
    config: &ForwardPassPerfModelConfig,
) -> Result<Vec<LearnedDependencyRemap>, AicError> {
    let fallback = config
        .estimator_config
        .fpm_learned
        .source
        .resolved_fallback
        .as_ref()
        .ok_or_else(|| invalid("missing prior pin"))?;
    let policy = fallback
        .policy
        .as_ref()
        .ok_or_else(|| invalid("missing prior policy"))?;
    let path = &policy.original_a.manifest_path;
    let manifest = original_a::read(
        path,
        policy
            .original_a
            .source_sha256
            .as_deref()
            .ok_or_else(|| invalid("prior is not pinned"))?,
    )?;
    let mut references = BTreeMap::new();
    for values in [manifest.sources, manifest.seeds] {
        for value in values {
            let original = PathBuf::from(&value.path);
            references.insert(original.clone(), value.sha256);
            if let Some(hash) = value.checkpoint_manifest_sha256 {
                references.insert(
                    original
                        .parent()
                        .unwrap_or(Path::new("."))
                        .join("manifest.json"),
                    hash,
                );
            }
        }
    }
    references
        .into_iter()
        .map(|(original_path, sha256)| {
            let target =
                prior_reference(path, &original_path, &sha256, &fallback.dependency_remap)?;
            let dep = dependency(&target, "original_a_relocated_input")?;
            check_hash(&sha256, &dep.sha256)?;
            Ok(LearnedDependencyRemap {
                original_path,
                path: target,
                sha256,
            })
        })
        .collect()
}

impl FpmLearnedConfig {
    /// Fitting/checkpoint readers consume the effective payload, while exported
    /// canonical configuration retains the outer bundle's immutable selection.
    pub(crate) fn runtime_source_config(&self) -> Self {
        let mut config = self.clone();
        if let Some(pin) = &self.source.resolved_source {
            if pin.kind == LearnedSourceKind::Bundle {
                config.artifact_path = (pin.seed_kind == Some(LearnedSeedKind::Artifact))
                    .then(|| pin.seed_path.clone())
                    .flatten();
                config.checkpoint_path = (pin.seed_kind != Some(LearnedSeedKind::Artifact))
                    .then(|| pin.seed_path.clone())
                    .flatten();
                config.source_sha256.clone_from(&pin.seed_sha256);
            }
        }
        config.source = LearnedSourceConfig::default();
        config
    }
}

impl super::EstimatorConfig {
    pub(crate) fn resolve_prior_dependency(
        &self,
        manifest_path: &Path,
        reference: &Path,
        expected: &str,
    ) -> Result<PathBuf, AicError> {
        let remaps = self
            .fpm_learned
            .source
            .resolved_fallback
            .as_ref()
            .map(|pin| pin.dependency_remap.as_slice())
            .unwrap_or_default();
        prior_reference(manifest_path, reference, expected, remaps)
    }
    pub(crate) fn effective_mixed_composition(&self) -> Option<&TemporaryMixedCompositionConfig> {
        self.fpm_learned
            .source
            .resolved_fallback
            .as_ref()
            .and_then(|pin| pin.policy.as_ref())
            .or(self.temporary_mixed_composition.as_ref())
    }
    pub(crate) fn effective_mixed_composition_mut(
        &mut self,
    ) -> Option<&mut TemporaryMixedCompositionConfig> {
        if let Some(pin) = &mut self.fpm_learned.source.resolved_fallback {
            pin.policy.as_mut()
        } else {
            self.temporary_mixed_composition.as_mut()
        }
    }
}

/// Called only for explicit learned selection, after ordinary identity/default
/// normalization. Historical explicit sources retain their path/hash contract;
/// bundles and discovery additionally carry immutable selection records.
pub(crate) fn resolve(config: &mut ForwardPassPerfModelConfig) -> Result<(), AicError> {
    resolve_impl(config, false, None)
}

/// The returned proof belongs only to this normalization call, after source
/// validation. Legacy public sources keep their historical shape and return
/// no complete receipt until their native payload reader has consumed them.
pub(crate) fn resolve_with_receipt(
    config: &mut ForwardPassPerfModelConfig,
) -> Result<Option<VerifiedLearnedSource>, AicError> {
    resolve(config)?;
    Ok(config
        .estimator_config
        .fpm_learned
        .source
        .resolved_source
        .clone()
        .map(|source| VerifiedLearnedSource { source }))
}

/// Capture a complete dependency closure for a legacy model's snapshot or
/// construction-time diagnostics without changing its public configuration.
pub(crate) fn pin_legacy_source(config: &mut ForwardPassPerfModelConfig) -> Result<(), AicError> {
    resolve_impl(config, true, None)
}

pub(crate) fn pin_legacy_source_with_receipt(
    config: &mut ForwardPassPerfModelConfig,
    receipt: VerifiedLearnedSource,
) -> Result<(), AicError> {
    resolve_impl(config, true, Some(receipt))
}

fn resolve_impl(
    config: &mut ForwardPassPerfModelConfig,
    pin_legacy: bool,
    receipt: Option<VerifiedLearnedSource>,
) -> Result<(), AicError> {
    let learned = &config.estimator_config.fpm_learned;
    let explicit_count = [
        learned.artifact_path.is_some(),
        learned.checkpoint_path.is_some(),
        learned.source.bundle_path.is_some(),
    ]
    .into_iter()
    .filter(|v| *v)
    .count();
    if explicit_count > 1 {
        return Err(invalid(
            "bundle_path, artifact_path, and checkpoint_path are mutually exclusive",
        ));
    }
    if !pin_legacy && explicit_count == 1 && learned.source == LearnedSourceConfig::default() {
        // A historical normalized config may later replace artifact_path and
        // remove source_sha256, or add temporary_mixed_composition. Do not
        // attach new immutable pins that silently change either operation.
        let artifact = learned.artifact_path.is_some();
        let path = absolute(if artifact {
            learned.artifact_path.as_ref().expect("explicit artifact")
        } else {
            learned
                .checkpoint_path
                .as_ref()
                .expect("explicit checkpoint")
        })?;
        let input = if artifact {
            path.clone()
        } else {
            path.join("manifest.json")
        };
        let dep = dependency(&input, "legacy_source")?;
        if let Some(expected) = &learned.source_sha256 {
            check_hash(expected, &dep.sha256)?;
        }
        let learned = &mut config.estimator_config.fpm_learned;
        learned.source_sha256 = Some(dep.sha256);
        if artifact {
            learned.artifact_path = Some(path);
        } else {
            learned.checkpoint_path = Some(path);
        }
        if let Some(policy) = &mut config.estimator_config.temporary_mixed_composition {
            policy.original_a.resolve_source()?;
        }
        return Ok(());
    }
    let old_pin = learned.source.resolved_source.clone();
    let fallback_pin = learned.source.resolved_fallback.clone();
    if config
        .estimator_config
        .temporary_mixed_composition
        .is_some()
        && (fallback_pin.is_some() || learned.source.fallback != LearnedFallback::Auto)
    {
        return Err(invalid(
            "stable fallback and temporary_mixed_composition cannot both be supplied",
        ));
    }
    let mut selected = learned
        .source
        .bundle_path
        .clone()
        .map(|path| (LearnedSourceKind::Bundle, path));
    if let Some(path) = &learned.artifact_path {
        selected = Some((LearnedSourceKind::Artifact, path.clone()));
    }
    if let Some(path) = &learned.checkpoint_path {
        selected = Some((LearnedSourceKind::Checkpoint, path.clone()));
    }
    let mut expected = learned.source_sha256.clone();
    if let Some(pin) = &old_pin {
        if pin.kind == LearnedSourceKind::None {
            if pin != &ResolvedLearnedSource::absent() || selected.is_some() || expected.is_some() {
                return Err(invalid(
                    "resolved source absence conflicts with source inputs",
                ));
            }
        } else {
            let path = pin
                .path
                .as_ref()
                .ok_or_else(|| invalid("resolved source requires path"))?;
            let hash = pin
                .sha256
                .as_ref()
                .ok_or_else(|| invalid("resolved source requires sha256"))?;
            if let Some((kind, supplied)) = &selected {
                if *kind != pin.kind || absolute(supplied)? != *path {
                    return Err(invalid("source path conflicts with resolved source"));
                }
            }
            if let Some(digest) = &expected {
                check_hash(digest, hash)?;
            }
            selected = Some((pin.kind, path.clone()));
            expected = Some(hash.clone());
        }
    } else if selected.is_none()
        && learned.adaptation == LearnedAdaptation::Online
        && learned.source.source_policy == LearnedSourcePolicy::Auto
    {
        if expected.is_some() {
            return Err(invalid(
                "source_sha256 without a source path or resolved source",
            ));
        }
        if let Some((path, hash)) = catalogue_source(config)? {
            selected = Some((LearnedSourceKind::Bundle, path));
            expected = Some(hash);
        }
    }
    let (mut resolved, prior) = match selected {
        None => {
            if expected.is_some() {
                return Err(invalid("source_sha256 without a source"));
            }
            (ResolvedLearnedSource::absent(), None)
        }
        Some((LearnedSourceKind::Bundle, path)) => {
            validate_bundle(&absolute(&path)?, expected.as_deref(), config)?
        }
        Some((kind @ (LearnedSourceKind::Artifact | LearnedSourceKind::Checkpoint), path)) => {
            let path = absolute(&path)?;
            if let Some(receipt) = receipt {
                if receipt.source.kind != kind || receipt.source.path.as_ref() != Some(&path) {
                    return Err(invalid(
                        "verified source receipt differs from selected input",
                    ));
                }
                if kind == LearnedSourceKind::Checkpoint {
                    for dependency in receipt.source.dependencies.iter().skip(1) {
                        let relative_path = dependency.path.strip_prefix(&path).map_err(|_| {
                            invalid("verified checkpoint dependency is outside its source root")
                        })?;
                        // Preserve the source resolver's symlink/containment checks;
                        // avoiding a second content scan does not relax path scope.
                        relative(&path, relative_path)?;
                    }
                }
                if let Some(expected) = expected.as_deref() {
                    check_hash(expected, receipt.source.sha256.as_deref().unwrap_or(""))?;
                }
                (receipt.source, None)
            } else {
                let (seed_kind, dependencies) = if kind == LearnedSourceKind::Artifact {
                    (
                        LearnedSeedKind::Artifact,
                        vec![dependency(&path, "learned_artifact")?],
                    )
                } else {
                    let (manifest, deps) = checkpoint_dependencies(
                        &path,
                        learned.adaptation != LearnedAdaptation::Frozen,
                    )?;
                    let kind = match string(&manifest, "schema")? {
                        "aisimulate_fpm_training_checkpoint" => LearnedSeedKind::TrainingCheckpoint,
                        "aisimulate_fpm_online_checkpoint" => LearnedSeedKind::RuntimeCheckpoint,
                        _ => return Err(invalid("unsupported checkpoint schema")),
                    };
                    (kind, deps)
                };
                let hash = dependencies[0].sha256.clone();
                if let Some(expected) = expected.as_deref() {
                    check_hash(expected, &hash)?;
                }
                (
                    ResolvedLearnedSource {
                        kind,
                        path: Some(path.clone()),
                        sha256: Some(hash.clone()),
                        seed_kind: Some(seed_kind),
                        seed_path: Some(path),
                        seed_sha256: Some(hash),
                        dependencies,
                        recipe: None,
                        provenance: None,
                    },
                    None,
                )
            }
        }
        Some((LearnedSourceKind::None, _)) => {
            return Err(invalid("absence source cannot have a path"));
        }
    };
    if let Some(pin) = &fallback_pin {
        if pin.kind == LearnedFallbackKind::MixedInterpolationV1 {
            let expected_prior = prior.as_ref().ok_or_else(|| {
                invalid("resolved stable fallback is incompatible with selected source")
            })?;
            let policy = pin
                .policy
                .as_ref()
                .ok_or_else(|| invalid("resolved stable fallback requires policy"))?;
            let raw_frozen = learned.adaptation == LearnedAdaptation::Frozen
                && !config.estimator_config.correction.enabled;
            let expected_scenario = if raw_frozen {
                TemporaryMixedCompositionScenario::A
            } else {
                TemporaryMixedCompositionScenario::AC
            };
            let expected_mode = if raw_frozen {
                TemporaryMixedCompositionMode::Compose
            } else {
                TemporaryMixedCompositionMode::ComposeCalibrated
            };
            if policy.scenario != expected_scenario
                || policy.mode != expected_mode
                || policy.original_a != expected_prior.original_a
                || policy.calibration != expected_prior.calibration
                || policy.quality_gate != expected_prior.quality_gate
                || pin.dependency_remap != expected_prior.dependency_remap
            {
                return Err(invalid(
                    "resolved stable fallback differs from qualified bundle prior",
                ));
            }
        }
    }
    let learned = &mut config.estimator_config.fpm_learned;
    learned.source_sha256.clone_from(&resolved.sha256);
    match resolved.kind {
        LearnedSourceKind::Bundle => learned.source.bundle_path.clone_from(&resolved.path),
        LearnedSourceKind::Artifact => learned.artifact_path.clone_from(&resolved.path),
        LearnedSourceKind::Checkpoint => learned.checkpoint_path.clone_from(&resolved.path),
        LearnedSourceKind::None => {}
    }
    let resolved_fallback = if let Some(pin) = fallback_pin {
        if (pin.kind == LearnedFallbackKind::None) != pin.policy.is_none() {
            return Err(invalid("resolved fallback kind/policy mismatch"));
        }
        if learned.source.fallback == LearnedFallback::Disabled
            && pin.kind != LearnedFallbackKind::None
        {
            return Err(invalid(
                "resolved fallback conflicts with fallback=disabled",
            ));
        }
        let mut pin = pin;
        if let Some(policy) = &mut pin.policy {
            policy.original_a.resolve_source()?;
        }
        pin
    } else if let Some(mut policy) = config.estimator_config.temporary_mixed_composition.take() {
        policy.original_a.resolve_source()?;
        ResolvedLearnedFallback {
            kind: LearnedFallbackKind::LegacyMixedInterpolationV1,
            native: None,
            policy: Some(policy),
            dependency_remap: Vec::new(),
        }
    } else if learned.source.fallback == LearnedFallback::Auto {
        if let Some(prior) = prior {
            let frozen = learned.adaptation == LearnedAdaptation::Frozen
                && !config.estimator_config.correction.enabled;
            ResolvedLearnedFallback {
                kind: LearnedFallbackKind::MixedInterpolationV1,
                native: None,
                dependency_remap: prior.dependency_remap,
                policy: Some(TemporaryMixedCompositionConfig {
                    scenario: if frozen {
                        TemporaryMixedCompositionScenario::A
                    } else {
                        TemporaryMixedCompositionScenario::AC
                    },
                    mode: if frozen {
                        TemporaryMixedCompositionMode::Compose
                    } else {
                        TemporaryMixedCompositionMode::ComposeCalibrated
                    },
                    original_a: prior.original_a,
                    calibration: prior.calibration,
                    quality_gate: prior.quality_gate,
                }),
            }
        } else {
            ResolvedLearnedFallback {
                kind: LearnedFallbackKind::None,
                native: None,
                policy: None,
                dependency_remap: Vec::new(),
            }
        }
    } else {
        ResolvedLearnedFallback {
            kind: LearnedFallbackKind::None,
            native: None,
            policy: None,
            dependency_remap: Vec::new(),
        }
    };
    // Include original-A references in the same dependency closure consumers
    // export. Do not rewrite the manifest; remaps are separate authenticated data.
    if let Some(policy) = &resolved_fallback.policy {
        for dep in prior_dependencies(policy, &resolved_fallback.dependency_remap)? {
            if let Some(existing) = resolved
                .dependencies
                .iter()
                .find(|existing| existing.path == dep.path)
            {
                check_hash(&existing.sha256, &dep.sha256)?;
            } else if resolved.kind == LearnedSourceKind::Bundle {
                return Err(invalid(format!(
                    "bundle dependency record missing for prior input {}",
                    dep.path.display()
                )));
            } else {
                resolved.dependencies.push(dep);
            }
        }
    } else if !resolved_fallback.dependency_remap.is_empty() || resolved_fallback.native.is_some() {
        return Err(invalid("fallback absence cannot contain dependency remaps"));
    }
    if let Some(pin) = &old_pin {
        if pin != &resolved {
            return Err(invalid(
                "resolved source metadata differs from verified source",
            ));
        }
    }
    let absent = resolved.kind == LearnedSourceKind::None;
    learned.source.resolved_source = Some(resolved);
    learned.source.resolved_fallback = Some(resolved_fallback);
    if absent {
        if learned.adaptation == LearnedAdaptation::Online
            && learned.source.source_policy == LearnedSourcePolicy::Auto
        {
            return Err(AicError::LearnedSourceUnavailable {
                reason: "no compatible learned seed bundle for the requested deployment; provide a checkpoint or install a matching bundle".into(),
            });
        }
        return Err(invalid(
            "an explicit learned artifact, checkpoint, or bundle is required",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
