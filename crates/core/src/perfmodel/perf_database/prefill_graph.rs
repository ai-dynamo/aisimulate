// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact, immutable data for the qualified seven-shape prefill pilot.
//! These two composite tables deliberately bypass shared-source resolution and
//! interpolation. The profile hashes every row payload and all retained inputs.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::parquet_loader::PerfReader;
use crate::common::enums::DatabaseMode;
use crate::common::error::AicError;
use crate::perfmodel::{BackendKind, DataType, EngineConfig};

pub const PROFILE_NAME: &str = "sglang_glm52_nvfp4_vr200_tp4_graph_v1";
// Identity only; no performance value is hard-coded in the consumer.
pub const PROFILE_ID: &str = "829a83e1629ba546dd4bd90e75a2e2496b7fb24ddc8b60dfbf076ba02312cbce";
pub const VERSION: &str = "0.5.18+nvinternal.rubin.0.8full.66997102";
pub const PROFILE_FILE: &str = "sglang_glm52_nvfp4_vr200_tp4_graph_v1.profile.json";
pub const CONTEXTS: [(u32, u32, u32); 7] = [
    (1, 1024, 0),
    (2, 1024, 0),
    (1, 1024, 1024),
    (1, 8192, 0),
    (2, 8192, 0),
    (1, 16384, 0),
    (1, 16384, 16384),
];
const TOKENS: [u32; 4] = [1024, 2048, 8192, 16384];
const ROLES: [&str; 2] = ["post_attention", "following_mlp"];

pub(crate) fn error(message: impl std::fmt::Display) -> AicError {
    AicError::PrefillGraphProfile(format!("{PROFILE_NAME}: {message}"))
}

pub(crate) fn validate_id(id: &str) -> Result<(), AicError> {
    if id != PROFILE_ID {
        return Err(error(
            "profile identity is not the approved immutable publication",
        ));
    }
    Ok(())
}

/// Public ISL includes prefix; admission precedes token multiplication.
pub(crate) fn public_shape(bs: u32, isl: u32, prefix: u32) -> Result<u32, AicError> {
    let new = isl
        .checked_sub(prefix)
        .filter(|&n| n > 0)
        .ok_or_else(|| error("total isl must exceed prefix"))?;
    inner_shape(bs, new, prefix)
}

/// The operation seam already subtracted prefix. Never subtract it again.
pub(crate) fn inner_shape(bs: u32, new: u32, prefix: u32) -> Result<u32, AicError> {
    if bs == 0 || new == 0 || new > u32::MAX / bs {
        return Err(error(
            "batch/new-token product must be positive and fit u32",
        ));
    }
    new.checked_add(prefix)
        .ok_or_else(|| error("total isl exceeds u32"))?;
    if !CONTEXTS.contains(&(bs, new, prefix)) {
        return Err(error(format!(
            "unqualified (batch,new,prefix)=({bs},{new},{prefix})"
        )));
    }
    bs.checked_mul(new)
        .ok_or_else(|| error("token product exceeds u32"))
}

pub(crate) fn validate_config(config: &EngineConfig) -> Result<bool, AicError> {
    match (
        &config.prefill_graph_profile,
        &config.prefill_graph_profile_id,
    ) {
        (None, None) => return Ok(false),
        (Some(name), Some(id)) if name == PROFILE_NAME => validate_id(id)?,
        _ => {
            return Err(error(
                "select the exact profile name and resolved profile ID together",
            ));
        }
    }
    let p = &config.parallel;
    let q = &config.quantization;
    if config.model_name != "nvidia/GLM-5.2-NVFP4"
        || config.system_name != "vr200_hecate"
        || config.backend != BackendKind::Sglang
        || config.backend_version.as_deref() != Some(VERSION)
        || config.database_mode != DatabaseMode::Silicon
        || config.decoder_replay
        || config.forward_model.as_deref().unwrap_or("op_level") != "op_level"
        || q.weight_dtype != Some(DataType::Bfloat16)
        || q.activation_dtype != Some(DataType::Bfloat16)
        || q.moe_dtype != Some(DataType::Nvfp4)
        || q.kv_cache_dtype != Some(DataType::Fp8)
        || (p.tp_size, p.pp_size, p.moe_tp_size, p.moe_ep_size) != (4, 1, Some(4), Some(1))
        || p.attention_dp_size.unwrap_or(1) != 1
        || p.cp_size.unwrap_or(1) != 1
        || config
            .speculative
            .as_ref()
            .and_then(|s| s.nextn)
            .unwrap_or(0)
            != 0
        || config.tolerate_dirless_version
        || config.enable_shared_layer != Some(false)
    {
        return Err(error(
            "requires the pinned GLM-5.2 NVFP4 VR200 SGLang TP4/EP1 BF16/FP8-KV op-level SILICON runtime with shared-source inheritance disabled",
        ));
    }
    Ok(true)
}

// serde_json's default fast f64 parser can move a correctly published value
// by one ULP. Preserve the profile's decimal literal for correctly rounded
// standard parsing, without changing float parsing for existing engine specs.
fn exact_f64<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    let raw = Box::<serde_json::value::RawValue>::deserialize(deserializer)?;
    raw.get().parse().map_err(serde::de::Error::custom)
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct AttentionRow {
    framework: String,
    version: String,
    device: String,
    op_name: String,
    kernel_source: String,
    measurement_protocol: String,
    batch_size: u32,
    input_seq_len: u32,
    prefix_len: u32,
    #[serde(deserialize_with = "exact_f64")]
    latency: f64,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct BoundaryRow {
    framework: String,
    version: String,
    device: String,
    op_name: String,
    kernel_source: String,
    measurement_protocol: String,
    num_tokens: u32,
    boundary_role: String,
    #[serde(deserialize_with = "exact_f64")]
    latency: f64,
}
#[derive(Deserialize)]
struct Table<R> {
    relative_path: PathBuf,
    rows: Vec<R>,
}
#[derive(Deserialize)]
struct Tables {
    attention: Table<AttentionRow>,
    communication: Table<BoundaryRow>,
}
#[derive(Deserialize)]
struct RetainedFile {
    path: PathBuf,
    sha256: String,
    size_bytes: u64,
}
#[derive(Deserialize)]
struct Profile {
    schema_version: u32,
    profile_name: String,
    tables: Tables,
    retained_files: Vec<RetainedFile>,
}

#[derive(Debug)]
struct Data {
    json: String,
    attention: BTreeMap<(u32, u32, u32), f64>,
    communication: BTreeMap<(u32, String), f64>,
    attention_rows: Vec<AttentionRow>,
    boundary_rows: Vec<BoundaryRow>,
}

pub struct PrefillGraphTable {
    systems_root: PathBuf,
    data: OnceLock<Result<Data, String>>,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn regular_path(root: &Path, relative: &Path) -> Result<PathBuf, AicError> {
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(error(
            "profile paths must be canonical bundle-relative paths",
        ));
    }
    let mut path = root.to_path_buf();
    for part in relative.components() {
        path.push(part);
        let metadata = std::fs::symlink_metadata(&path)
            .map_err(|e| error(format!("{}: {e}", path.display())))?;
        if metadata.file_type().is_symlink() {
            return Err(error(format!("symlink in profile path {}", path.display())));
        }
    }
    if !path.is_file() {
        return Err(error(format!("{} is not a regular file", path.display())));
    }
    Ok(path)
}

fn validate_schema(path: &Path, attention: bool) -> Result<(), AicError> {
    use parquet::basic::{ConvertedType, Repetition, Type};
    use parquet::file::reader::{FileReader, SerializedFileReader};
    let file = std::fs::File::open(path).map_err(error)?;
    let reader = SerializedFileReader::new(file).map_err(error)?;
    let mut expected = BTreeMap::from([
        ("framework", Type::BYTE_ARRAY),
        ("version", Type::BYTE_ARRAY),
        ("device", Type::BYTE_ARRAY),
        ("op_name", Type::BYTE_ARRAY),
        ("kernel_source", Type::BYTE_ARRAY),
        ("profile_id", Type::BYTE_ARRAY),
        ("measurement_protocol", Type::BYTE_ARRAY),
        ("latency", Type::DOUBLE),
    ]);
    if attention {
        for name in ["batch_size", "input_seq_len", "prefix_len"] {
            expected.insert(name, Type::INT64);
        }
    } else {
        expected.insert("num_tokens", Type::INT64);
        expected.insert("boundary_role", Type::BYTE_ARRAY);
    }
    let columns = reader.metadata().file_metadata().schema_descr().columns();
    if columns.len() != expected.len() {
        return Err(error("unexpected measured table columns"));
    }
    for column in columns {
        let Some(kind) = expected.remove(column.name()) else {
            return Err(error("duplicate or unknown table column"));
        };
        if column.physical_type() != kind
            || column.self_type().get_basic_info().repetition() != Repetition::REQUIRED
            || column.path().parts().len() != 1
            || (kind == Type::BYTE_ARRAY && column.converted_type() != ConvertedType::UTF8)
        {
            return Err(error(format!(
                "invalid required column type: {}",
                column.name()
            )));
        }
    }
    Ok(())
}

impl PrefillGraphTable {
    pub fn new(systems_root: &Path) -> Self {
        Self {
            systems_root: systems_root.to_path_buf(),
            data: OnceLock::new(),
        }
    }

    fn table_relative(family: &str, file: &str) -> PathBuf {
        PathBuf::from("data/vr200_hecate")
            .join(family)
            .join("sglang")
            .join(VERSION)
            .join(file)
    }

    fn read(&self) -> Result<Data, AicError> {
        let profile_path = regular_path(
            &self.systems_root,
            &Self::table_relative("sparse_attention", PROFILE_FILE),
        )?;
        let bytes = std::fs::read(&profile_path).map_err(|e| error(e))?;
        validate_id(&sha256(&bytes))?;
        let other = regular_path(
            &self.systems_root,
            &Self::table_relative("comm", PROFILE_FILE),
        )?;
        if std::fs::read(other).map_err(error)? != bytes {
            return Err(error("attention and communication profile bytes differ"));
        }
        let profile: Profile = serde_json::from_slice(&bytes).map_err(error)?;
        if profile.schema_version != 1 || profile.profile_name != PROFILE_NAME {
            return Err(error("unsupported profile schema/name"));
        }
        let attention_path = Self::table_relative(
            "sparse_attention",
            "sglang_prefill_attention_sequence_perf.parquet",
        );
        let communication_path =
            Self::table_relative("comm", "sglang_prefill_comm_norm_boundary_perf.parquet");
        if profile.tables.attention.relative_path != attention_path
            || profile.tables.communication.relative_path != communication_path
        {
            return Err(error(
                "profile tables must use their exact primary family paths",
            ));
        }
        let mut retained = BTreeSet::new();
        for file in &profile.retained_files {
            if !retained.insert(&file.path) {
                return Err(error("duplicate retained-file identity"));
            }
            let path = regular_path(&self.systems_root, &file.path)?;
            let bytes = std::fs::read(&path).map_err(error)?;
            if bytes.len() as u64 != file.size_bytes || sha256(&bytes) != file.sha256 {
                return Err(error(format!(
                    "retained input changed: {}",
                    file.path.display()
                )));
            }
        }
        // The immutable profile pins the precise runtime/provenance fields and
        // retained inventory; explicit row validation also guards loader errors.
        let attention = self.attention_rows(
            &regular_path(&self.systems_root, &attention_path)?,
            &profile.tables.attention.rows,
        )?;
        let communication = self.boundary_rows(
            &regular_path(&self.systems_root, &communication_path)?,
            &profile.tables.communication.rows,
        )?;
        Ok(Data {
            json: String::from_utf8(bytes).map_err(error)?,
            attention,
            communication,
            attention_rows: profile.tables.attention.rows,
            boundary_rows: profile.tables.communication.rows,
        })
    }

    /// Recheck disk on each engine build, even when shared DB tables are cached.
    pub(crate) fn validate(&self) -> Result<(), AicError> {
        self.read().map(|_| ()).map_err(error)
    }

    /// Copy the complete admitted inputs into engine-owned storage. Validation
    /// runs on the copy itself, so a source change during copying cannot bind
    /// unverified bytes to the profile. Lazy readers never reopen the originals.
    pub(crate) fn snapshot(&self) -> Result<tempfile::TempDir, AicError> {
        let profile: Profile = serde_json::from_str(&self.read()?.json).map_err(error)?;
        let mut paths: BTreeSet<PathBuf> = profile
            .retained_files
            .into_iter()
            .map(|file| file.path)
            .collect();
        paths.extend([
            profile.tables.attention.relative_path,
            profile.tables.communication.relative_path,
            Self::table_relative("sparse_attention", PROFILE_FILE),
            Self::table_relative("comm", PROFILE_FILE),
        ]);
        let snapshot = tempfile::Builder::new()
            .prefix("aisimulate-prefill-graph-")
            .tempdir()
            .map_err(error)?;
        for relative in paths {
            let source = regular_path(&self.systems_root, &relative)?;
            let destination = snapshot.path().join(relative);
            std::fs::create_dir_all(destination.parent().unwrap()).map_err(error)?;
            std::fs::write(&destination, std::fs::read(source).map_err(error)?).map_err(error)?;
        }
        Self::new(snapshot.path()).validate()?;
        Ok(snapshot)
    }

    fn data(&self) -> Result<&Data, AicError> {
        self.data
            .get_or_init(|| self.read().map_err(|e| e.to_string()))
            .as_ref()
            .map_err(error)
    }
    pub(crate) fn profile_json(&self) -> Result<&str, AicError> {
        Ok(&self.data()?.json)
    }
    pub(crate) fn attention(&self, bs: u32, new: u32, prefix: u32) -> Result<f64, AicError> {
        inner_shape(bs, new, prefix)?;
        self.data()?
            .attention
            .get(&(bs, new, prefix))
            .copied()
            .ok_or_else(|| error("missing exact attention cell"))
    }
    pub(crate) fn boundary(&self, tokens: u32, role: &str) -> Result<f64, AicError> {
        self.data()?
            .communication
            .get(&(tokens, role.to_string()))
            .copied()
            .ok_or_else(|| error("missing exact communication boundary cell"))
    }

    pub(crate) fn raw_view(&self, attention: bool) -> Result<Option<String>, AicError> {
        let first = self
            .systems_root
            .join(Self::table_relative("sparse_attention", PROFILE_FILE));
        let second = self
            .systems_root
            .join(Self::table_relative("comm", PROFILE_FILE));
        if !first.exists() && !second.exists() {
            return Ok(None);
        }
        let data = self.data()?;
        let rows = if attention {
            serde_json::to_value(&data.attention_rows)
        } else {
            serde_json::to_value(&data.boundary_rows)
        }
        .map_err(error)?;
        let mut out = serde_json::Map::new();
        for row in rows.as_array().ok_or_else(|| error("missing raw rows"))? {
            let mut row = row.clone();
            row.as_object_mut()
                .ok_or_else(|| error("invalid raw row"))?
                .insert("profile_id".into(), PROFILE_ID.into());
            if attention {
                let key = format!(
                    "{}|{}|{}",
                    row["batch_size"], row["input_seq_len"], row["prefix_len"]
                );
                out.insert(key, row);
            } else {
                let key = row["num_tokens"].to_string();
                let role = row["boundary_role"]
                    .as_str()
                    .ok_or_else(|| error("missing raw role"))?
                    .to_owned();
                out.entry(key)
                    .or_insert_with(|| serde_json::json!({}))
                    .as_object_mut()
                    .unwrap()
                    .insert(role, row);
            }
        }
        serde_json::to_string(&out).map(Some).map_err(error)
    }

    pub(crate) fn validate_sources(&self, db: &super::PerfDatabase) -> Result<(), AicError> {
        // Admission must recheck the caller's mutable files, even if shared
        // tables have cached data or an error from an earlier read.
        let value: Profile = serde_json::from_str(&self.read()?.json).map_err(error)?;
        for file in &value.retained_files {
            if file.path.extension().and_then(|s| s.to_str()) != Some("parquet") {
                continue;
            }
            let expected = regular_path(&self.systems_root, &file.path)?;
            let name = file
                .path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| error("invalid retained filename"))?;
            let sources = db
                .source_resolver
                .sources_for(name, &db.data_root)
                .map_err(error)?;
            if sources.len() != 1
                || sources[0].kernel_sources().is_some()
                || sources[0].path() != expected
            {
                return Err(error(format!(
                    "retained table must resolve to its exact primary file: {name}"
                )));
            }
        }
        Ok(())
    }

    fn attention_rows(
        &self,
        path: &Path,
        expected: &[AttentionRow],
    ) -> Result<BTreeMap<(u32, u32, u32), f64>, AicError> {
        validate_schema(path, true)?;
        let reader = PerfReader::open(path).map_err(error)?;
        let mut out = BTreeMap::new();
        for row in reader.rows().map_err(error)? {
            let row = row.map_err(error)?;
            let strcol = |name| row.str_owned(reader.col(name)?);
            let intcol = |name| row.u32(reader.col(name)?);
            validate_id(&strcol("profile_id").map_err(error)?)?;
            let value = AttentionRow {
                framework: strcol("framework")?,
                version: strcol("version")?,
                device: strcol("device")?,
                op_name: strcol("op_name")?,
                kernel_source: strcol("kernel_source")?,
                measurement_protocol: strcol("measurement_protocol")?,
                batch_size: intcol("batch_size")?,
                input_seq_len: intcol("input_seq_len")?,
                prefix_len: intcol("prefix_len")?,
                latency: row.f64(reader.col("latency")?)?,
            };
            let key = (value.batch_size, value.input_seq_len, value.prefix_len);
            if !expected.contains(&value)
                || !value.latency.is_finite()
                || value.latency <= 0.0
                || value.framework != "sglang"
                || value.version != VERSION
                || value.device != "NVIDIA Graphics Device"
                || value.op_name != "sglang_prefill_attention_sequence"
                || value.measurement_protocol != "joint_attention_sequence"
                || value.kernel_source != "sglang_breakable_dsa_attention_sequence"
                || !CONTEXTS.contains(&key)
                || out.insert(key, value.latency).is_some()
            {
                return Err(error(format!(
                    "mismatched, duplicate or invalid attention row {key:?}: got {value:?}, expected {:?}",
                    expected
                        .iter()
                        .find(|row| (row.batch_size, row.input_seq_len, row.prefix_len) == key)
                )));
            }
        }
        if out.len() != CONTEXTS.len() || expected.len() != CONTEXTS.len() {
            return Err(error("incomplete attention row inventory"));
        }
        Ok(out)
    }
    fn boundary_rows(
        &self,
        path: &Path,
        expected: &[BoundaryRow],
    ) -> Result<BTreeMap<(u32, String), f64>, AicError> {
        validate_schema(path, false)?;
        let reader = PerfReader::open(path).map_err(error)?;
        let mut out = BTreeMap::new();
        for row in reader.rows().map_err(error)? {
            let row = row.map_err(error)?;
            let strcol = |name| row.str_owned(reader.col(name)?);
            validate_id(&strcol("profile_id").map_err(error)?)?;
            let value = BoundaryRow {
                framework: strcol("framework")?,
                version: strcol("version")?,
                device: strcol("device")?,
                op_name: strcol("op_name")?,
                kernel_source: strcol("kernel_source")?,
                measurement_protocol: strcol("measurement_protocol")?,
                num_tokens: row.u32(reader.col("num_tokens")?)?,
                boundary_role: strcol("boundary_role")?,
                latency: row.f64(reader.col("latency")?)?,
            };
            let kernel = if value.num_tokens <= 2048 {
                "flashinfer_ar_residual_rmsnorm"
            } else {
                "native_allreduce_then_rmsnorm"
            };
            if !expected.contains(&value)
                || !value.latency.is_finite()
                || value.latency <= 0.0
                || value.framework != "sglang"
                || value.version != VERSION
                || value.device != "NVIDIA Graphics Device"
                || value.op_name != "sglang_prefill_comm_norm_boundary"
                || value.measurement_protocol != "block_5x20"
                || value.kernel_source != kernel
                || !TOKENS.contains(&value.num_tokens)
                || !ROLES.contains(&value.boundary_role.as_str())
                || out
                    .insert(
                        (value.num_tokens, value.boundary_role.clone()),
                        value.latency,
                    )
                    .is_some()
            {
                return Err(error(
                    "mismatched, duplicate or invalid communication boundary row",
                ));
            }
        }
        if out.len() != 8 || expected.len() != 8 {
            return Err(error("incomplete communication row inventory"));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reviewed_composition_roundtrips_and_rejects_coefficient_changes() {
        let ops: Vec<crate::operators::Op> = serde_json::from_str(include_str!(
            "../operators/testdata/glm52_prefill_graph_context.json"
        ))
        .unwrap();
        assert_eq!(composition_sha256(&ops).unwrap(), CONTEXT_OPS_SHA256);
        let bytes = bincode::serialize(&ops).unwrap();
        let mut roundtrip: Vec<crate::operators::Op> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(ops, roundtrip);
        assert_eq!(composition_sha256(&roundtrip).unwrap(), CONTEXT_OPS_SHA256);
        // The independent baseline full-model structural inventory is 115953106944 bytes/rank.
        assert_eq!(
            ops.iter()
                .map(crate::operators::Op::weight_bytes)
                .sum::<f64>(),
            115953106944.0
        );
        let crate::operators::Op::Elementwise(ref mut norms) = roundtrip[5] else {
            panic!("norm position")
        };
        norms.scale_factor = 3.0;
        assert_ne!(composition_sha256(&roundtrip).unwrap(), CONTEXT_OPS_SHA256);
    }

    #[test]
    fn qualified_rows_match_trace_oracles() {
        let root = std::env::var_os("AISIMULATE_PREFILL_GRAPH_SYSTEMS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                crate::perfmodel::repo_relative("python/aisimulate/src/aisimulate_core/systems")
                    .unwrap()
            });
        let table = PrefillGraphTable::new(&root);
        table.validate().unwrap();
        // Independently accepted joint-sequence mean and rank-max/block-normalized boundary mean.
        assert_eq!(table.attention(1, 1024, 0).unwrap(), 12.539125347137452);
        assert_eq!(
            table.boundary(1024, "post_attention").unwrap(),
            0.03387998938560486
        );
        for (b, n, p) in CONTEXTS {
            assert!(table.attention(b, n, p).unwrap() > 0.0);
        }
        for n in TOKENS {
            for role in ROLES {
                assert!(table.boundary(n, role).unwrap() > 0.0);
            }
        }
    }

    #[test]
    fn direct_build_keeps_snapshot_alive_with_database_views() {
        use std::sync::Arc;

        use crate::perf_database::PerfDatabase;
        use crate::perfmodel::engine::{runtime::Engine, spec::EngineSpec};

        let root = std::env::var_os("AISIMULATE_PREFILL_GRAPH_SYSTEMS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                crate::perfmodel::repo_relative("python/aisimulate/src/aisimulate_core/systems")
                    .unwrap()
            });
        let source = PrefillGraphTable::new(&root).snapshot().unwrap();
        let config: EngineConfig = serde_json::from_value(serde_json::json!({
            "schema_version": crate::ENGINE_CONFIG_SCHEMA_VERSION,
            "model_name": "nvidia/GLM-5.2-NVFP4", "system_name": "vr200_hecate",
            "systems_path": source.path(), "backend": "sglang", "backend_version": VERSION,
            "tp_size": 4, "pp_size": 1, "moe_tp_size": 4, "moe_ep_size": 1,
            "weight_dtype": "bfloat16", "activation_dtype": "bfloat16",
            "moe_dtype": "nvfp4", "kv_cache_dtype": "fp8", "enable_shared_layer": false,
            "prefill_graph_profile": PROFILE_NAME, "prefill_graph_profile_id": PROFILE_ID,
        }))
        .unwrap();
        let ops = serde_json::from_str(include_str!(
            "../operators/testdata/glm52_prefill_graph_context.json"
        ))
        .unwrap();
        let spec = EngineSpec::new(config, ops, vec![]);
        let supplied_db =
            Arc::new(PerfDatabase::load(source.path(), "vr200_hecate", "sglang", VERSION).unwrap());
        let engine = Engine::build(spec, Arc::clone(&supplied_db)).unwrap();
        assert!(!Arc::ptr_eq(engine.database(), &supplied_db));
        let private_root = engine.database().prefill_graph.systems_root.clone();
        assert_ne!(private_root, source.path());
        drop(source);
        // Frozen independent v5 ledger: admission must precede this first read.
        assert!(
            (engine.predict_prefill_latency(1, 1024, 0).unwrap() - 34.90489051212317).abs() < 1e-10
        );
        let view = engine.database().silicon_view();
        drop(engine);
        assert!(private_root.is_dir());
        assert_eq!(
            view.prefill_graph.attention(1, 1024, 0).unwrap(),
            12.539125347137452
        );
        drop(view);
        assert!(!private_root.exists());
    }

    #[test]
    fn total_isl_roundtrips_all_seven_contexts_once() {
        for (bs, new, past) in CONTEXTS {
            assert_eq!(public_shape(bs, new + past, past).unwrap(), bs * new);
        }
        for (bs, isl, past) in [
            (0, 1024, 0),
            (1, 1024, 1024),
            (1, 1024, 1025),
            (1, 2048, 0),
            (2, 1536, 512),
            (u32::MAX, 2, 0),
            (1, 16384, 16384),
        ] {
            assert!(public_shape(bs, isl, past).is_err(), "{bs}/{isl}/{past}");
        }
        assert!(inner_shape(1, u32::MAX, 1).is_err());
    }
}

// SHA256 of serde_json::to_value(context_ops), after removing only the two
// profile_id fields, encoded by serde_json::to_vec. This is a structural
// identity, generated from the reviewed model rewrite; no latency is embedded.
pub(crate) const CONTEXT_OPS_SHA256: &str =
    "8fb54aacfaeea16b94b2e671744fc87ddeab19846ce25b7ad0b2467602633199";

pub(crate) fn contains_profile_ops(ops: &[crate::operators::Op]) -> bool {
    use crate::operators::Op;
    ops.iter().any(|op| match op {
        Op::SglangPrefillAttentionSequence(_) | Op::SglangPrefillCommNormBoundary(_) => true,
        Op::Overlap(o) => contains_profile_ops(&o.group_a) || contains_profile_ops(&o.group_b),
        Op::Fallback(o) => {
            contains_profile_ops(std::slice::from_ref(&o.primary))
                || contains_profile_ops(&o.fallback)
        }
        Op::TokenScale(o) => contains_profile_ops(std::slice::from_ref(&o.op)),
        Op::FpmForward(o) => contains_profile_ops(&o.sol_ops),
        Op::Dsv41Stage(o) => contains_profile_ops(&o.children),
        _ => false,
    })
}

pub(crate) fn composition_sha256(ops: &[crate::operators::Op]) -> Result<String, AicError> {
    let mut value = serde_json::to_value(ops).map_err(error)?;
    for op in value
        .as_array_mut()
        .ok_or_else(|| error("invalid context operation list"))?
    {
        for variant in [
            "SglangPrefillAttentionSequence",
            "SglangPrefillCommNormBoundary",
        ] {
            if let Some(fields) = op
                .get_mut(variant)
                .and_then(serde_json::Value::as_object_mut)
            {
                fields.remove("profile_id");
            }
        }
    }
    Ok(sha256(&serde_json::to_vec(&value).map_err(error)?))
}

pub(crate) fn validate_spec(
    spec: &crate::perfmodel::engine::spec::EngineSpec,
) -> Result<bool, AicError> {
    use crate::operators::Op;
    let selected = validate_config(&spec.engine)?;
    if !selected {
        if contains_profile_ops(&spec.context_ops) || contains_profile_ops(&spec.generation_ops) {
            return Err(error(
                "measured composite ops require the explicitly selected profile",
            ));
        }
        return Ok(false);
    }
    if contains_profile_ops(&spec.generation_ops) {
        return Err(error(
            "prefill composite operations cannot appear in generation",
        ));
    }
    let mut attention = 0;
    let mut roles = BTreeSet::new();
    for op in &spec.context_ops {
        match op {
            Op::SglangPrefillAttentionSequence(op) => {
                validate_id(&op.profile_id)?;
                if !op.weight_bytes.is_finite() || op.weight_bytes <= 0.0 {
                    return Err(error("missing original attention weight inventory"));
                }
                attention += 1;
            }
            Op::SglangPrefillCommNormBoundary(op) => {
                validate_id(&op.profile_id)?;
                op.count()?;
                if !roles.insert(op.boundary_role.as_str()) {
                    return Err(error("duplicate boundary role"));
                }
            }
            _ => {}
        }
    }
    if attention != 1
        || roles != BTreeSet::from(ROLES)
        || composition_sha256(&spec.context_ops)? != CONTEXT_OPS_SHA256
    {
        return Err(error(
            "unsupported modified context composition; recompile the exact approved profile",
        ));
    }
    Ok(true)
}
