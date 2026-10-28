// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::{Component, PathBuf};

use bincode::Options;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use sha2::{Digest, Sha256};

use super::*;

const INITIAL_SCHEMA: &str = "aisimulate_fpm_training_checkpoint";
const RUNTIME_SCHEMA: &str = "aisimulate_fpm_online_checkpoint";

#[derive(Deserialize)]
struct Manifest {
    schema: String,
    schema_version: u32,
    artifact_file: String,
    artifact_sha256: String,
    identity: serde_json::Value,
    #[serde(default)]
    features: Vec<String>,
    #[serde(default)]
    target: String,
    #[serde(default)]
    shrinkage: String,
    #[serde(default)]
    stores: BTreeMap<Kind, SeedStore>,
    state_file: Option<String>,
    state_sha256: Option<String>,
    anchor_manifest_sha256: Option<String>,
    #[serde(skip)]
    source_sha256: String,
}

#[derive(Deserialize)]
struct SeedStore {
    rows_file: String,
    rows_sha256: String,
    row_count: usize,
    bin_thresholds: Vec<Vec<f64>>,
    n_bins_non_missing: Vec<usize>,
    missing_values_bin_idx: usize,
    loss: String,
    learning_rate: f64,
    l2_regularization: f64,
    min_samples_leaf: usize,
    max_depth: Option<usize>,
    max_leaf_nodes: Option<usize>,
}

#[derive(Serialize)]
struct SaveState<'a, L: Serialize> {
    learner: L,
    correction_json: &'a str,
    artifact_sha256: &'a str,
}

/// Encodes exactly like the derived `OnlineLearner` layout, with the learner's
/// source locations (`artifact_path`, `checkpoint_path`, `source_sha256`)
/// cleared. Loading always replaces them from the resolved configuration, so
/// clearing them makes the bytes independent of where the source lived.
struct SourceFreeLearner<'a>(&'a OnlineLearner);

impl Serialize for SourceFreeLearner<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let learner = self.0;
        let mut config = learner.config.clone();
        config.artifact_path = None;
        config.checkpoint_path = None;
        config.source_sha256 = None;
        let mut state = serializer.serialize_struct("OnlineLearner", 7)?;
        state.serialize_field("config", &config)?;
        state.serialize_field("identity_json", &learner.identity_json)?;
        state.serialize_field("anchor_manifest_sha256", &learner.anchor_manifest_sha256)?;
        state.serialize_field("stores", &learner.stores)?;
        state.serialize_field("next_row_id", &learner.next_row_id)?;
        state.serialize_field("version", &learner.version)?;
        state.serialize_field("rejected_observations", &learner.rejected_observations)?;
        state.end()
    }
}

#[derive(Deserialize)]
struct LoadState {
    learner: OnlineLearner,
    correction_json: String,
    artifact_sha256: String,
}

fn io(path: &Path, source: std::io::Error) -> AicError {
    AicError::Io {
        path: path.to_owned(),
        source,
    }
}

fn child(root: &Path, name: &str) -> Result<PathBuf, AicError> {
    let path = Path::new(name);
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(invalid(
            "checkpoint paths must be relative without parent components",
        ));
    }
    Ok(root.join(path))
}

fn hash(path: &Path) -> Result<String, AicError> {
    let mut reader = BufReader::new(File::open(path).map_err(|e| io(path, e))?);
    let mut digest = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = reader.read(&mut buf).map_err(|e| io(path, e))?;
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Hash and later parse the same opened payload, so an atomic path replacement
/// cannot substitute different bytes. Checkpoint inputs must remain immutable;
/// concurrent in-place modification of an already-open file is unsupported.
fn checked_file(root: &Path, name: &str, expected: &str) -> Result<(PathBuf, File), AicError> {
    let path = child(root, name)?;
    let mut file = File::open(&path).map_err(|e| io(&path, e))?;
    let mut digest = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf).map_err(|e| io(&path, e))?;
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
    }
    if format!("{:x}", digest.finalize()) != expected {
        return Err(invalid(format!("checksum mismatch for {name}")));
    }
    file.rewind().map_err(|e| io(&path, e))?;
    Ok((path, file))
}

fn manifest(path: &Path) -> Result<Manifest, AicError> {
    manifest_pinned(path, None)
}

fn manifest_pinned(path: &Path, expected: Option<&str>) -> Result<Manifest, AicError> {
    let p = path.join("manifest.json");
    let bytes = std::fs::read(&p).map_err(|e| io(&p, e))?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if expected.is_some_and(|hash| !hash.eq_ignore_ascii_case(&actual)) {
        return Err(invalid(
            "checkpoint manifest differs from resolved source_sha256",
        ));
    }
    let mut m: Manifest = serde_json::from_slice(&bytes)
        .map_err(|e| invalid(format!("invalid checkpoint manifest: {e}")))?;
    m.source_sha256 = actual;
    if m.schema == RUNTIME_SCHEMA && m.schema_version == 1 {
        return Err(invalid(
            "runtime checkpoint version 1 predates dynamic replay state; resume from its training seed or regenerate with this binary",
        ));
    }
    if m.schema == RUNTIME_SCHEMA && m.schema_version == 2 {
        return Err(invalid(
            "runtime checkpoint version 2 predates weighted group statistics; resume from its training seed or regenerate with this binary",
        ));
    }
    if m.schema == RUNTIME_SCHEMA && m.schema_version == 3 {
        return Err(invalid(
            "runtime checkpoint version 3 predates historical bin grouping; resume from its training seed or regenerate with this binary",
        ));
    }
    if m.schema == RUNTIME_SCHEMA && m.schema_version == 4 {
        return Err(invalid(
            "runtime checkpoint version 4 uses the research replay/configuration layout; resume from its training seed with the canonical online configuration",
        ));
    }
    if !((m.schema == INITIAL_SCHEMA && m.schema_version == 1)
        || (m.schema == RUNTIME_SCHEMA && m.schema_version == 5))
    {
        return Err(invalid("unsupported checkpoint schema/version"));
    }
    Ok(m)
}

fn manifest_model(path: &Path, m: &Manifest) -> Result<LearnedForwardPassModel, AicError> {
    let p = child(path, &m.artifact_file)?;
    let bytes = std::fs::read(&p).map_err(|e| io(&p, e))?;
    if format!("{:x}", Sha256::digest(&bytes)) != m.artifact_sha256 {
        return Err(invalid(format!(
            "checksum mismatch for {}",
            m.artifact_file
        )));
    }
    let text = std::str::from_utf8(&bytes).map_err(|e| invalid(e.to_string()))?;
    LearnedForwardPassModel::from_json(text)
}

pub(crate) fn checkpoint_model(path: &Path) -> Result<LearnedForwardPassModel, AicError> {
    let m = manifest(path)?;
    manifest_model(path, &m)
}

pub(crate) fn checkpoint_model_for_config(
    path: &Path,
    config: &ForwardPassPerfModelConfig,
) -> Result<LearnedForwardPassModel, AicError> {
    let m = manifest_pinned(
        path,
        config.estimator_config.fpm_learned.source_sha256.as_deref(),
    )?;
    if m.identity != canonical_learned_identity(config)? {
        return Err(invalid(
            "checkpoint deployment/store identity does not match resolved configuration",
        ));
    }
    manifest_model(path, &m)
}

pub(crate) fn canonical_learned_identity(
    config: &ForwardPassPerfModelConfig,
) -> Result<serde_json::Value, AicError> {
    let mut value = serde_json::to_value(config).map_err(|e| invalid(e.to_string()))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| invalid("invalid canonical identity"))?;
    for key in [
        "estimator_config",
        "estimation_mode",
        "fallback_policy",
        "systems_paths",
        "database_mode",
        "transfer_policy",
        "strict_provenance",
    ] {
        object.remove(key);
    }
    Ok(value)
}

pub(super) fn load(
    path: &Path,
    learned: &mut LearnedForwardPassModel,
    config: &FpmLearnedConfig,
    canonical: &ForwardPassPerfModelConfig,
) -> Result<OnlineLearner, AicError> {
    load_with(path, learned, config, canonical, false)
}

/// `combined_runtime` admits the adaptive expert of a causal selector saved
/// inside a state snapshot, whose frozen expert, calibration and selector
/// evidence the snapshot restores separately. Standalone configurations never
/// set it, so a runtime checkpoint alone still cannot seed a selector.
pub(super) fn load_with(
    path: &Path,
    learned: &mut LearnedForwardPassModel,
    config: &FpmLearnedConfig,
    canonical: &ForwardPassPerfModelConfig,
    combined_runtime: bool,
) -> Result<OnlineLearner, AicError> {
    let m = manifest_pinned(path, config.source_sha256.as_deref())?;
    if combined_runtime
        && !(m.schema == RUNTIME_SCHEMA && config.adaptation == LearnedAdaptation::CausalSelector)
    {
        return Err(invalid(
            "a causal selector snapshot requires its runtime checkpoint",
        ));
    }
    let actual_identity = canonical_learned_identity(canonical)?;
    if m.identity != actual_identity {
        return Err(invalid(
            "checkpoint deployment/store identity does not match resolved configuration",
        ));
    }
    let model = manifest_model(path, &m)?;
    if model.artifact() != learned.artifact() {
        return Err(invalid("predictor does not match checkpoint artifact"));
    }
    if config.adaptation == LearnedAdaptation::Frozen {
        return Err(invalid("frozen inference must not initialize a learner"));
    }
    if m.schema == RUNTIME_SCHEMA {
        if config.adaptation == LearnedAdaptation::CausalSelector && !combined_runtime {
            return Err(invalid(
                "causal_selector requires a training checkpoint; online runtime snapshots do not contain frozen calibration and selector state",
            ));
        }
        let (state_path, file) = checked_file(
            path,
            m.state_file
                .as_deref()
                .ok_or_else(|| invalid("missing state_file"))?,
            m.state_sha256
                .as_deref()
                .ok_or_else(|| invalid("missing state_sha256"))?,
        )?;
        let bytes = file.metadata().map_err(|e| io(&state_path, e))?.len();
        let mut state: LoadState = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(bytes)
            .reject_trailing_bytes()
            .deserialize_from(BufReader::new(file))
            .map_err(|e| invalid(format!("invalid runtime checkpoint: {e}")))?;
        if state.artifact_sha256 != m.artifact_sha256 {
            return Err(invalid(
                "runtime state predictor checksum differs from its manifest",
            ));
        }
        if m.anchor_manifest_sha256.as_deref() != Some(&state.learner.anchor_manifest_sha256) {
            return Err(invalid(
                "runtime state anchor identity differs from its manifest",
            ));
        }
        let mut saved = state.learner.config.clone();
        saved.artifact_path = config.artifact_path.clone();
        saved.checkpoint_path = config.checkpoint_path.clone();
        // A runtime export has a new manifest hash; its fitting controls are unchanged.
        saved.source_sha256 = config.source_sha256.clone();
        if &saved != config {
            return Err(invalid("online controls differ from saved runtime state"));
        }
        if serde_json::from_str::<serde_json::Value>(&state.learner.identity_json)
            .map_err(|e| invalid(e.to_string()))?
            != actual_identity
        {
            return Err(invalid("runtime state identity differs from its manifest"));
        }
        state.learner.config = config.clone();
        state.learner.restored_corrections = Some(state.correction_json);
        validate_state(&state.learner, learned)?;
        for (kind, store_state) in &mut state.learner.stores {
            fitting::cache_frozen_anchor_prefix(
                store_state,
                learned.store(*kind).expect("validated store"),
                config,
            );
        }
        return Ok(state.learner);
    }
    let target = match learned.target() {
        LearnedTarget::LogMs => "log_ms",
        LearnedTarget::Ms => "ms",
    };
    if m.features != learned.feature_names()
        || m.target != target
        || m.shrinkage != "included_in_leaf_values"
        || m.stores.keys().copied().collect::<Vec<_>>() != learned.store_kinds()
    {
        return Err(invalid(
            "checkpoint feature/target/store/shrinkage contract mismatch",
        ));
    }
    let mut learner = OnlineLearner {
        config: config.clone(),
        identity_json: actual_identity.to_string(),
        anchor_manifest_sha256: m.source_sha256.clone(),
        stores: BTreeMap::new(),
        next_row_id: 0,
        version: 0,
        rejected_observations: 0,
        restored_corrections: None,
    };
    for (kind, seed) in m.stores {
        let width = learned.feature_names().len();
        validate_seed(&seed, width)?;
        let (rows_path, rows_file) = checked_file(path, &seed.rows_file, &seed.rows_sha256)?;
        let anchors = read_rows(&rows_path, rows_file, &seed, width, &learned.target())?;
        if let Some(id) = anchors.ids.iter().max() {
            learner.next_row_id = learner.next_row_id.max(
                id.checked_add(1)
                    .ok_or_else(|| invalid("anchor row ID overflow"))?,
            );
        }
        let mut offsets = Vec::with_capacity(width + 1);
        offsets.push(0);
        for bins in &seed.bin_thresholds {
            offsets.push(offsets.last().unwrap() + bins.len() + 2);
        }
        let mut state = StoreState {
            anchors: Arc::new(anchors),
            frozen_anchor_prefix: Vec::new(),
            history_grouping_ms: None,
            thresholds: seed.bin_thresholds,
            offsets,
            settings: FitSettings {
                learning_rate: seed.learning_rate,
                l2_regularization: seed.l2_regularization,
                min_samples_leaf: seed.min_samples_leaf,
                max_depth: seed.max_depth,
            },
            trees: Vec::new(),
            recent: VecDeque::new(),
            pending: Vec::new(),
            counters: Counters::default(),
        };
        let store = learned
            .store(kind)
            .ok_or_else(|| invalid("missing seed store"))?;
        if store.trees.is_empty() {
            return Err(invalid("cannot fit an empty ensemble"));
        }
        for tree in &store.trees {
            if seed
                .max_leaf_nodes
                .is_some_and(|n| tree.left.iter().filter(|&&l| l < 0).count() > n)
            {
                return Err(invalid("seed tree exceeds recorded leaf budget"));
            }
        }
        fitting::bootstrap(&mut state, store, config)?;
        validate_seed_fit(&state, store)?;
        if config.history_bin_grouping {
            // Preserve the original bootstrap statistics, including their exact
            // accumulation order for the first lazy update. Only future eager
            // refreshes use grouped gradients; never change the seed predictor.
            let start = std::time::Instant::now();
            state.anchors = Arc::new(state.anchors.group_by_bins()?);
            state.history_grouping_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
            fitting::cache_frozen_anchor_prefix(&mut state, store, config);
        }
        learner.stores.insert(kind, state);
    }
    validate_state(&learner, learned)?;
    Ok(learner)
}

/// Check that rows and recorded fitting controls explain the imported leaves.
/// sklearn accumulates float32 derivatives while this updater retains f64;
/// the error allowance scales with the absolute derivative mass, including
/// cancellation within a leaf. This check never refits the seed predictor.
pub(super) fn validate_seed_fit(
    state: &StoreState,
    store: &super::super::learned::LearnedStore,
) -> Result<(), AicError> {
    if state.anchors.groups.is_some() {
        return Err(invalid(
            "seed fitting validation requires original ungrouped rows",
        ));
    }
    let mut prefix = vec![store.baseline; state.anchors.y.len()];
    for (tree_index, tree) in store.trees.iter().enumerate() {
        let mut absolute_gradients = vec![0.0; tree.value.len()];
        for (row, prediction) in prefix.iter_mut().enumerate() {
            let start = row * state.anchors.width;
            let x = &state.anchors.x[start..start + state.anchors.width];
            let leaf = leaf_index(tree, x);
            absolute_gradients[leaf] += (*prediction - state.anchors.y[row]).abs();
            *prediction += tree.value[leaf];
        }
        for (node, stats) in state.trees[tree_index].stats.iter().enumerate() {
            if tree.left[node] >= 0 {
                continue;
            }
            if stats.total.count == 0
                || (node != 0 && stats.total.count < state.settings.min_samples_leaf as u64)
            {
                return Err(invalid(
                    "seed leaf membership violates its fitted sample limit",
                ));
            }
            let denominator = stats.total.count as f64 + state.settings.l2_regularization;
            let expected = -state.settings.learning_rate * (stats.total.g / denominator);
            let actual = tree.value[node];
            let tolerance =
                5e-6 * state.settings.learning_rate * (absolute_gradients[node] / denominator)
                    + 1e-12 * actual.abs().max(expected.abs()).max(1.0);
            if !expected.is_finite()
                || !tolerance.is_finite()
                || (actual - expected).abs() > tolerance
            {
                return Err(invalid(format!(
                    "seed tree {tree_index} leaf {node} is inconsistent with rows, learning rate, or L2 regularization"
                )));
            }
        }
        if prefix.iter().any(|value| !value.is_finite()) {
            return Err(invalid("seed prefix prediction is nonfinite"));
        }
    }
    Ok(())
}

fn validate_seed(s: &SeedStore, width: usize) -> Result<(), AicError> {
    if width > usize::from(u16::MAX) + 1
        || s.loss != "squared_error"
        || !s.learning_rate.is_finite()
        || s.learning_rate <= 0.0
        || !s.l2_regularization.is_finite()
        || s.l2_regularization < 0.0
        || s.min_samples_leaf == 0
        || s.row_count == 0
        || s.max_depth == Some(0)
        || s.max_leaf_nodes.is_some_and(|n| n < 2)
        || s.bin_thresholds.len() != width
        || s.n_bins_non_missing.len() != width
        || s.missing_values_bin_idx > 255
    {
        return Err(invalid("unsupported or invalid fitted training parameters"));
    }
    for (bounds, &n) in s.bin_thresholds.iter().zip(&s.n_bins_non_missing) {
        if bounds.len() > 254
            || bounds.iter().any(|v| !v.is_finite())
            || bounds.windows(2).any(|w| w[0] >= w[1])
            || !(n == bounds.len() + 1 || (n == 0 && bounds.is_empty()))
            || n > s.missing_values_bin_idx
        {
            return Err(invalid("invalid fitted bin mapper"));
        }
    }
    Ok(())
}

fn read_rows(
    _path: &Path,
    file: File,
    seed: &SeedStore,
    width: usize,
    target: &LearnedTarget,
) -> Result<Anchors, AicError> {
    let reader = SerializedFileReader::new(file).map_err(|e| invalid(e.to_string()))?;
    if reader.metadata().file_metadata().num_rows() != seed.row_count as i64 {
        return Err(invalid("anchor row count differs from manifest"));
    }
    let columns = reader.metadata().file_metadata().schema_descr().columns();
    let find = |name: &str| {
        columns
            .iter()
            .position(|c| c.name() == name)
            .ok_or_else(|| invalid(format!("missing anchor column {name}")))
    };
    let id_col = find("row_id")?;
    let y_col = find("y")?;
    let wall_col = find("wall_time_ms")?;
    let x_cols = (0..width)
        .map(|i| find(&format!("x{i}")))
        .collect::<Result<Vec<_>, _>>()?;
    let values = seed
        .row_count
        .checked_mul(width)
        .ok_or_else(|| invalid("anchor dimensions overflow"))?;
    let mut out = Anchors {
        ids: Vec::with_capacity(seed.row_count),
        y: Vec::with_capacity(seed.row_count),
        x: Vec::with_capacity(values),
        bins: Vec::with_capacity(values),
        width,
        groups: None,
    };
    let mut ids = std::collections::HashSet::with_capacity(seed.row_count);
    for item in reader
        .get_row_iter(None)
        .map_err(|e| invalid(e.to_string()))?
    {
        let row = item.map_err(|e| invalid(e.to_string()))?;
        let id = row.get_ulong(id_col).map_err(|e| invalid(e.to_string()))?;
        if !ids.insert(id) {
            return Err(invalid("duplicate anchor row ID"));
        }
        let y = row.get_double(y_col).map_err(|e| invalid(e.to_string()))?;
        let wall = row
            .get_double(wall_col)
            .map_err(|e| invalid(e.to_string()))?;
        let expected = match target {
            LearnedTarget::LogMs => wall.ln(),
            LearnedTarget::Ms => wall,
        };
        if !y.is_finite()
            || !wall.is_finite()
            || wall <= 0.0
            || (y - expected).abs() > 1e-12 * expected.abs().max(1.0)
        {
            return Err(invalid("anchor target transform/units mismatch"));
        }
        let start = out.x.len();
        for &col in &x_cols {
            let x = row.get_double(col).map_err(|e| invalid(e.to_string()))?;
            if x.is_infinite() {
                return Err(invalid("infinite anchor feature"));
            }
            out.x.push(x);
        }
        out.bins
            .extend(bin_row(&out.x[start..], &seed.bin_thresholds));
        out.ids.push(id);
        out.y.push(y);
    }
    Ok(out)
}

pub(super) fn validate_state(
    learner: &OnlineLearner,
    model: &LearnedForwardPassModel,
) -> Result<(), AicError> {
    learner.config.validate()?;
    if learner.stores.keys().copied().collect::<Vec<_>>() != model.store_kinds() {
        return Err(invalid("runtime store mismatch"));
    }
    let lazy = learner.config.derivative_refresh == LearnedDerivativeRefresh::Lazy;
    let structural = learner.config.update_mode == LearnedUpdateMode::Structural;
    let mut publications = 0u64;
    for (&kind, state) in &learner.stores {
        let forest = model.store(kind).unwrap();
        let width = model.feature_names().len();
        let count = state.anchors.y.len();
        let dimensions = width
            .checked_mul(count)
            .ok_or_else(|| invalid("runtime dimensions overflow"))?;
        if width > usize::from(u16::MAX) + 1
            || count == 0
            || forest.trees.is_empty()
            || state.anchors.width != width
            || state.trees.len() != forest.trees.len()
            || state.recent.len() > learner.config.recent_window
            || state.pending.len() > learner.config.update_every
            || state.anchors.x.len() != dimensions
            || state.anchors.bins.len() != dimensions
            || state.anchors.ids.len() != count
            || state.thresholds.len() != width
            || state.offsets.len() != width + 1
            || state.offsets.first() != Some(&0)
        {
            return Err(invalid("inconsistent runtime training dimensions"));
        }

        let settings = &state.settings;
        if !settings.learning_rate.is_finite()
            || settings.learning_rate <= 0.0
            || !settings.l2_regularization.is_finite()
            || settings.l2_regularization < 0.0
            || settings.min_samples_leaf == 0
            || settings.max_depth == Some(0)
        {
            return Err(invalid("invalid runtime fitted parameters"));
        }
        for (feature, thresholds) in state.thresholds.iter().enumerate() {
            if thresholds.len() > 254
                || thresholds.iter().any(|v| !v.is_finite())
                || thresholds.windows(2).any(|w| w[0] >= w[1])
                || state.offsets[feature].checked_add(thresholds.len() + 2)
                    != Some(state.offsets[feature + 1])
            {
                return Err(invalid("invalid runtime bin mapper/offsets"));
            }
        }
        let logical_count = state.anchors.original_rows();
        match (&state.anchors.groups, learner.config.history_bin_grouping) {
            (Some(groups), true) => {
                let represented = groups
                    .counts
                    .iter()
                    .try_fold(0u64, |n, &c| n.checked_add(c));
                if groups.counts.len() != count
                    || groups.counts.contains(&0)
                    || represented != u64::try_from(logical_count).ok()
                    || logical_count < count
                    || groups.original_max_id >= learner.next_row_id
                    || state
                        .anchors
                        .ids
                        .iter()
                        .any(|&id| id > groups.original_max_id)
                    || groups.original_ids_sha256.len() != 64
                    || !groups
                        .original_ids_sha256
                        .bytes()
                        .all(|b| b.is_ascii_hexdigit())
                {
                    return Err(invalid(
                        "invalid grouped historical multiplicity/provenance",
                    ));
                }
                let unique: std::collections::BTreeSet<_> =
                    state.anchors.bins.chunks(width).collect();
                if unique.len() != count {
                    return Err(invalid("duplicate grouped historical bin vector"));
                }
            }
            (None, false) => {}
            _ => {
                return Err(invalid(
                    "historical grouping state differs from configured policy",
                ));
            }
        }
        let mut ids = std::collections::HashSet::new();
        for (i, &id) in state.anchors.ids.iter().enumerate() {
            if !ids.insert(id)
                || id >= learner.next_row_id
                || !state.anchors.y[i].is_finite()
                || (model.target() == LearnedTarget::Ms && state.anchors.y[i] <= 0.0)
            {
                return Err(invalid("invalid runtime anchor ID/target"));
            }
            validate_row_features(
                &state.anchors.x[i * width..(i + 1) * width],
                &state.anchors.bins[i * width..(i + 1) * width],
                state,
            )?;
        }
        let mut previous_id = None;
        for (row, committed) in state
            .recent
            .iter()
            .map(|r| (r, true))
            .chain(state.pending.iter().map(|r| (r, false)))
        {
            if !ids.insert(row.id)
                || state
                    .anchors
                    .groups
                    .as_ref()
                    .is_some_and(|g| row.id <= g.original_max_id)
                || row.id >= learner.next_row_id
                || previous_id.is_some_and(|id| id >= row.id)
            {
                return Err(invalid("invalid runtime online row order/identity"));
            }
            previous_id = Some(row.id);
            validate_row_features(&row.x, &row.bins, state)?;
            let expected = match model.target() {
                LearnedTarget::LogMs => row.wall_ms.ln(),
                LearnedTarget::Ms => row.wall_ms,
            };
            if !row.y.is_finite()
                || !row.wall_ms.is_finite()
                || row.wall_ms <= 0.0
                || (row.y - expected).abs() > 1e-12 * expected.abs().max(1.0)
                || row.gradients.len()
                    != if committed && lazy {
                        forest.trees.len()
                    } else {
                        0
                    }
                || row.gradients.iter().any(|g| !g.is_finite())
            {
                return Err(invalid("invalid runtime row target/derivatives"));
            }
        }
        let retained = logical_count
            .checked_add(state.recent.len())
            .ok_or_else(|| invalid("runtime row count overflow"))? as u64;
        let histogram_len = if structural {
            *state.offsets.last().unwrap()
        } else {
            0
        };
        for (tree_index, (t, tree)) in state.trees.iter().zip(&forest.trees).enumerate() {
            let frozen = tree_index < mutable_start(&learner.config, forest.trees.len());
            let retained = if frozen {
                logical_count as u64
            } else {
                retained
            };
            let expected_recent = if frozen { 0 } else { state.recent.len() as u64 };
            let nodes = tree.value.len();
            if t.stats.len() != nodes
                || t.regions.len() != nodes
                || t.candidates.len() != nodes
                || t.stats.iter().enumerate().any(|(node, s)| {
                    !s.total.g.is_finite()
                        || !s.total.recent_g.is_finite()
                        || s.total.recent_count > s.total.count
                        || s.total.recent_count > expected_recent
                        || s.total.count > retained
                        || s.histogram.len()
                            != if t.regions[node].len() > 1 {
                                histogram_len
                            } else {
                                0
                            }
                        || s.histogram.iter().any(|b| {
                            !b.g.is_finite()
                                || !b.recent_g.is_finite()
                                || b.recent_count > b.count
                                || b.recent_count > s.total.recent_count
                                || b.count > s.total.count
                        })
                })
                || t.stats[0].total.count != retained
                || t.stats[0].total.recent_count != expected_recent
            {
                return Err(invalid("inconsistent runtime gradient/aggregate state"));
            }
            let mut reachable = vec![false; nodes];
            let mut pending = vec![(0, 0usize)];
            while let Some((node, depth)) = pending.pop() {
                if reachable[node] {
                    return Err(invalid("runtime predictor has shared children"));
                }
                reachable[node] = true;
                if settings.max_depth.is_some_and(|limit| depth > limit) {
                    return Err(invalid("runtime predictor exceeds fitted depth limit"));
                }
                if tree.left[node] >= 0 {
                    let left = tree.left[node] as usize;
                    let right = tree.right[node] as usize;
                    if t.stats[left]
                        .total
                        .count
                        .checked_add(t.stats[right].total.count)
                        != Some(t.stats[node].total.count)
                    {
                        return Err(invalid(
                            "runtime child counts do not partition their parent",
                        ));
                    }
                    if t.stats[left]
                        .total
                        .recent_count
                        .checked_add(t.stats[right].total.recent_count)
                        != Some(t.stats[node].total.recent_count)
                    {
                        return Err(invalid(
                            "runtime child recent counts do not partition their parent",
                        ));
                    }
                    pending.push((left, depth + 1));
                    pending.push((right, depth + 1));
                }
            }
            if t.regions[0] != (0..nodes).collect::<Vec<_>>() {
                return Err(invalid("runtime root lost its reserved tree capacity"));
            }
            // Reject inactive branches before walking descendants below: a
            // malformed unused DAG could otherwise make that walk exponential.
            if reachable
                .iter()
                .enumerate()
                .any(|(node, &active)| !active && tree.left[node] >= 0)
            {
                return Err(invalid("runtime unused arena slot contains a branch"));
            }
            for node in 0..nodes {
                let region = &t.regions[node];
                if region.first() != Some(&node)
                    || region.iter().any(|&slot| slot >= nodes)
                    || region.windows(2).any(|w| w[0] >= w[1])
                {
                    return Err(invalid("invalid runtime reserved region"));
                }
                let mut descendants = vec![false; nodes];
                let mut walk = vec![node];
                while let Some(child) = walk.pop() {
                    descendants[child] = true;
                    if tree.left[child] >= 0 {
                        walk.push(tree.left[child] as usize);
                        walk.push(tree.right[child] as usize);
                    }
                }
                if descendants
                    .iter()
                    .enumerate()
                    .any(|(slot, &is_child)| is_child && region.binary_search(&slot).is_err())
                    || region
                        .iter()
                        .any(|&slot| reachable[slot] && !descendants[slot])
                {
                    return Err(invalid("runtime reserved region crosses subtree ownership"));
                }
                if tree.left[node] >= 0 {
                    let left_region = &t.regions[tree.left[node] as usize];
                    let right_region = &t.regions[tree.right[node] as usize];
                    if left_region
                        .iter()
                        .chain(right_region)
                        .any(|slot| region.binary_search(slot).is_err())
                        || left_region.iter().any(|slot| right_region.contains(slot))
                    {
                        return Err(invalid(
                            "runtime child region capacities overlap or escape their parent",
                        ));
                    }
                }
                let stats = &t.stats[node];
                if structural && region.len() > 1 {
                    for offsets in state.offsets.windows(2) {
                        let sum = stats.histogram[offsets[0]..offsets[1]]
                            .iter()
                            .try_fold(0u64, |sum, bin| sum.checked_add(bin.count));
                        let recent_sum = stats.histogram[offsets[0]..offsets[1]]
                            .iter()
                            .try_fold(0u64, |sum, bin| sum.checked_add(bin.recent_count));
                        if sum != Some(stats.total.count)
                            || recent_sum != Some(stats.total.recent_count)
                        {
                            return Err(invalid(
                                "runtime histogram counts do not match node count",
                            ));
                        }
                    }
                }
                let candidates = &t.candidates[node];
                if ((!structural || region.len() == 1) && !candidates.is_empty())
                    || candidates.iter().any(|c| {
                        let feature = usize::from(c.feature);
                        let bin = usize::from(c.bin);
                        feature >= width
                            || bin > state.thresholds[feature].len()
                            || (bin == state.thresholds[feature].len() && c.missing_left)
                    })
                    || candidates.windows(2).any(|w| {
                        (w[0].feature, w[0].bin, w[0].missing_left)
                            >= (w[1].feature, w[1].bin, w[1].missing_left)
                    })
                {
                    return Err(invalid("invalid runtime split candidates"));
                }
                if tree.left[node] >= 0 {
                    let feature = tree.feature[node] as usize;
                    let bin = state.thresholds[feature]
                        .iter()
                        .position(|&v| v == tree.threshold[node])
                        .or_else(|| {
                            (tree.threshold[node] == f64::MAX)
                                .then_some(state.thresholds[feature].len())
                        })
                        .ok_or_else(|| {
                            invalid("runtime predictor threshold is absent from its bin mapper")
                        })?;
                    if structural
                        && !candidates.contains(&Candidate::new(
                            feature,
                            bin,
                            tree.missing_left.get(node).copied().unwrap_or(true),
                        )?)
                    {
                        return Err(invalid("runtime split candidates omit the incumbent"));
                    }
                }
                if !reachable[node]
                    && (tree.left[node] >= 0
                        || stats.total.count != 0
                        || stats.total.g != 0.0
                        || stats.total.recent_g != 0.0
                        || stats.total.recent_count != 0
                        || stats.histogram.iter().any(|s| {
                            s.g != 0.0 || s.count != 0 || s.recent_g != 0.0 || s.recent_count != 0
                        })
                        || !candidates.is_empty())
                {
                    return Err(invalid("runtime unused arena slot contains active state"));
                }
            }
        }
        let counters = &state.counters;
        if Some(counters.accepted)
            != ((state.recent.len() + state.pending.len()) as u64).checked_add(counters.evicted)
            || counters.eager_updates.checked_add(counters.lazy_updates)
                != Some(counters.publications)
            || counters.lazy_fallbacks > counters.eager_updates
            || counters.batches_since_eager >= learner.config.eager_refresh_interval
            || (!lazy && (counters.batches_since_eager != 0 || counters.lazy_updates != 0))
        {
            return Err(invalid("inconsistent runtime update counters"));
        }
        publications = publications
            .checked_add(counters.publications)
            .ok_or_else(|| invalid("runtime publication count overflow"))?;
    }
    if learner.version != publications {
        return Err(invalid(
            "runtime model version differs from publication count",
        ));
    }
    Ok(())
}

fn validate_row_features(x: &[f64], bins: &[u16], state: &StoreState) -> Result<(), AicError> {
    if x.len() != state.anchors.width
        || bins.len() != state.anchors.width
        || x.iter().any(|v| v.is_infinite())
        || bins != bin_row(x, &state.thresholds)
    {
        return Err(invalid("invalid runtime row features/bin assignments"));
    }
    Ok(())
}

pub(super) fn save(
    path: &Path,
    learned: &LearnedForwardPassModel,
    learner: &OnlineLearner,
    corrections: &str,
) -> Result<(), AicError> {
    save_with(path, learned, learner, corrections, false)
}

/// `source_free` writes [`SourceFreeLearner`] instead of the learner's own
/// source locations; state snapshots use it so identical learner state
/// always produces identical bytes.
pub(super) fn save_with(
    path: &Path,
    learned: &LearnedForwardPassModel,
    learner: &OnlineLearner,
    corrections: &str,
    source_free: bool,
) -> Result<(), AicError> {
    if path.exists() {
        return Err(invalid(
            "checkpoint destination already exists; use a new snapshot directory",
        ));
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
    let staging = tempfile::Builder::new()
        .prefix(".fpm-checkpoint-")
        .tempdir_in(parent)
        .map_err(|e| io(parent, e))?;
    let model_path = staging.path().join("model.json");
    let mut model_file = File::create(&model_path).map_err(|e| io(&model_path, e))?;
    model_file
        .write_all(learned.to_json()?.as_bytes())
        .map_err(|e| io(&model_path, e))?;
    model_file.sync_all().map_err(|e| io(&model_path, e))?;
    let artifact_sha256 = hash(&model_path)?;
    let state_path = staging.path().join("state.bin");
    let mut writer = BufWriter::new(File::create(&state_path).map_err(|e| io(&state_path, e))?);
    let options = bincode::DefaultOptions::new().with_fixint_encoding();
    if source_free {
        options.serialize_into(
            &mut writer,
            &SaveState {
                learner: SourceFreeLearner(learner),
                correction_json: corrections,
                artifact_sha256: &artifact_sha256,
            },
        )
    } else {
        options.serialize_into(
            &mut writer,
            &SaveState {
                learner,
                correction_json: corrections,
                artifact_sha256: &artifact_sha256,
            },
        )
    }
    .map_err(|e| invalid(format!("cannot serialize learner state: {e}")))?;
    writer.flush().map_err(|e| io(&state_path, e))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|e| io(&state_path, e))?;
    let m = serde_json::json!({
        "schema": RUNTIME_SCHEMA, "schema_version": 5,
        "artifact_file": "model.json", "artifact_sha256": artifact_sha256,
        "state_file": "state.bin", "state_sha256": hash(&state_path)?,
        "identity": serde_json::from_str::<serde_json::Value>(&learner.identity_json).map_err(|e| invalid(e.to_string()))?,
        "anchor_manifest_sha256": learner.anchor_manifest_sha256,
    });
    let manifest_path = staging.path().join("manifest.json");
    let mut file = File::create(&manifest_path).map_err(|e| io(&manifest_path, e))?;
    serde_json::to_writer(&mut file, &m).map_err(|e| invalid(e.to_string()))?;
    file.sync_all().map_err(|e| io(&manifest_path, e))?;
    publish_checkpoint(staging.path(), path)
}

fn publish_checkpoint(staging: &Path, path: &Path) -> Result<(), AicError> {
    // Reserve the destination atomically so concurrent writers cannot replace
    // each other's complete checkpoint (or an existing empty directory).
    std::fs::create_dir(path).map_err(|e| io(path, e))?;
    if let Err(error) = std::fs::rename(staging, path) {
        // Remove only our still-empty reservation; never remove populated data.
        let _ = std::fs::remove_dir(path);
        return Err(io(path, error));
    }
    Ok(())
}

#[cfg(test)]
mod publication_tests {
    use super::*;

    #[test]
    fn source_free_learner_encoding_matches_the_derived_layout() {
        let learner = OnlineLearner {
            config: FpmLearnedConfig {
                checkpoint_path: Some("/checkpoint".into()),
                source_sha256: Some("a".repeat(64)),
                ..FpmLearnedConfig::default()
            },
            identity_json: "{}".into(),
            anchor_manifest_sha256: "b".repeat(64),
            stores: BTreeMap::new(),
            next_row_id: 7,
            version: 3,
            rejected_observations: 2,
            restored_corrections: Some("{}".into()),
        };
        let options = bincode::DefaultOptions::new().with_fixint_encoding();
        let mut cleared = learner.clone();
        cleared.config.checkpoint_path = None;
        cleared.config.source_sha256 = None;
        assert_eq!(
            options.serialize(&SourceFreeLearner(&learner)).unwrap(),
            options.serialize(&cleared).unwrap()
        );
    }

    #[test]
    fn destination_created_after_preparation_is_never_replaced() {
        let root = tempfile::tempdir().unwrap();
        let staging = root.path().join("staging");
        std::fs::create_dir(&staging).unwrap();
        std::fs::write(staging.join("manifest.json"), "complete").unwrap();
        let destination = root.path().join("competing");
        std::fs::create_dir(&destination).unwrap();
        assert!(publish_checkpoint(&staging, &destination).is_err());
        assert!(destination.is_dir());
        assert!(std::fs::read_dir(&destination).unwrap().next().is_none());
        assert_eq!(
            std::fs::read_to_string(staging.join("manifest.json")).unwrap(),
            "complete"
        );
    }

    #[test]
    fn failed_publication_removes_only_its_empty_reservation() {
        let root = tempfile::tempdir().unwrap();
        let destination = root.path().join("destination");
        assert!(publish_checkpoint(&root.path().join("missing"), &destination).is_err());
        assert!(!destination.exists());
    }

    #[test]
    fn checked_payload_retains_opened_bytes_after_atomic_path_replacement() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("state.bin");
        std::fs::write(&path, b"original").unwrap();
        let expected = hash(&path).unwrap();
        let (_, mut opened) = checked_file(root.path(), "state.bin", &expected).unwrap();
        let replacement = root.path().join("replacement");
        std::fs::write(&replacement, b"replacement").unwrap();
        std::fs::rename(replacement, &path).unwrap();
        let mut content = String::new();
        opened.read_to_string(&mut content).unwrap();
        assert_eq!(content, "original");
        assert_eq!(std::fs::read_to_string(path).unwrap(), "replacement");
    }
}
