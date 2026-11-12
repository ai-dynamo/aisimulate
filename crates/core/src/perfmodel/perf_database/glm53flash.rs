// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GLM-5.3-Flash NoPE sparse-MLA + IndexPool attention module measurements.
//!
//! This is the only GLM-specific measured table. Every other GLM operator is
//! priced by the generic GEMM / MoE / KDA / mHC / collective tables. The
//! schema follows the DeepSeek-V4.1 module precedent (`dsv41.rs`):
//!
//! | column | meaning |
//! |---|---|
//! | `component` | always `attention` |
//! | `geometry` | canonical sorted compact JSON of the `Glm53Attention` body without `name`/`measured` |
//! | `batch_size` | requests in the timed call |
//! | `prefix` | cached tokens per request before the timed prefill (0 for decode) |
//! | `x` | new tokens per request (prefill) or absolute KV length per request (decode) |
//! | `latency` | milliseconds, local compute only (the output all-reduce is excluded) |
//! | `kernel_source`, `measurement_scope=local_compute`, `source_sha256`, `config_sha256`, `runtime_digest=sha256:...`, `used_cuda_graph`, `sample_count`, `kv_seed_regime`, `execution_profile=full` | provenance |
//!
//! Only the requested system/backend/version primary file is read: GLM
//! attention measurements are never inherited from sibling versions, declared
//! donors or other backends. A missing file or geometry is a coverage gap;
//! every malformed present file is fatal. Provenance: one source/runtime
//! identity and one latent `kv_cache_dtype` per file, one `config_sha256`
//! per checkpoint format (a version directory holds both checkpoints), and
//! one `used_cuda_graph` value per
//! geometry (shipped tables time both prefill and decode under the serving
//! CUDA graphs; prefill uses the frameworks' breakable/piecewise graphs).
//! Within one geometry the reader
//! interpolates utilization (`SOL / latency`) hierarchically over batch,
//! prefix and work (`x`), holding the boundary utilization outside the
//! measured range, and returns `SOL(query) / util`.
//!
//! The `geometry` string is the file format only. Loading validates it and
//! indexes rows by its typed form (`GeometryKey`); queries build that key
//! from the op's fields instead of serializing the op.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use rustc_hash::FxHashMap;
use serde_json::Value;

use super::SourceResolver;
use crate::common::enums::{GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::operators::glm53flash::Glm53AttentionOp;

pub const BASENAME: &str = "glm53_attention_module_perf.parquet";

type Curve = BTreeMap<u32, f64>;
type PrefixCurves = BTreeMap<u32, Curve>;
type BatchCurves = BTreeMap<u32, PrefixCurves>;
type Grids = FxHashMap<GeometryKey, BatchCurves>;

pub struct Glm53AttentionTable {
    path: Option<PathBuf>,
    grids: OnceLock<Result<Grids, String>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Backend {
    Vllm,
    Sglang,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum CheckpointFormat {
    Fp8,
    Nvfp4,
}

/// Typed form of the canonical `geometry` column: every serialized body field
/// except `name` and `measured`. Two ops have equal keys exactly when their
/// canonical geometry strings are equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GeometryKey {
    is_context: bool,
    backend: Backend,
    checkpoint_format: CheckpointFormat,
    hidden_size: u32,
    tp_size: u32,
    num_heads: u32,
    head_dim: u32,
    q_lora_rank: u32,
    kv_lora_rank: u32,
    value_head_dim: u32,
    index_n_heads: u32,
    index_head_dim: u32,
    index_topk: u32,
    index_pool: u32,
    conv_kernel: u32,
    /// Bits: finite values match exactly when their JSON spellings do; a
    /// non-finite value (`null` in JSON) matches no row.
    gate_lower_bound: u64,
    projection_quant_mode: GemmQuantMode,
    kv_cache_dtype: KvCacheQuantMode,
}

/// `None` for an op no validated row can describe: rows are `sparse_mla`
/// modules of a vllm/sglang backend and an fp8/nvfp4 checkpoint.
fn geometry_key(op: &Glm53AttentionOp) -> Option<GeometryKey> {
    if op.layer_kind != "sparse_mla" {
        return None;
    }
    let backend = match op.backend.as_str() {
        "vllm" => Backend::Vllm,
        "sglang" => Backend::Sglang,
        _ => return None,
    };
    let checkpoint_format = match op.checkpoint_format.as_str() {
        "fp8" => CheckpointFormat::Fp8,
        "nvfp4" => CheckpointFormat::Nvfp4,
        _ => return None,
    };
    Some(GeometryKey {
        is_context: op.is_context,
        backend,
        checkpoint_format,
        hidden_size: op.hidden_size,
        tp_size: op.tp_size,
        num_heads: op.num_heads,
        head_dim: op.head_dim,
        q_lora_rank: op.q_lora_rank,
        kv_lora_rank: op.kv_lora_rank,
        value_head_dim: op.value_head_dim,
        index_n_heads: op.index_n_heads,
        index_head_dim: op.index_head_dim,
        index_topk: op.index_topk,
        index_pool: op.index_pool,
        conv_kernel: op.conv_kernel,
        gate_lower_bound: op.gate_lower_bound.to_bits(),
        projection_quant_mode: op.projection_quant_mode,
        kv_cache_dtype: op.kv_cache_dtype,
    })
}

fn invalid(message: impl Into<String>) -> AicError {
    AicError::InvalidPerfData(message.into())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Canonical measured key: the op body without its display name and without
/// the (empty for sparse MLA) generic measured composition.
pub fn geometry(op: &Glm53AttentionOp) -> Result<String, AicError> {
    let mut value = serde_json::to_value(op).map_err(|e| invalid(e.to_string()))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| invalid("GLM attention geometry must be an object"))?;
    object.remove("name");
    object.remove("measured");
    let sorted: BTreeMap<_, _> = object.iter().collect();
    serde_json::to_string(&sorted).map_err(|e| invalid(e.to_string()))
}

fn validate_geometry(encoded: &str) -> Result<GeometryKey, AicError> {
    let mut named: Value = serde_json::from_str(encoded).map_err(|e| invalid(e.to_string()))?;
    named
        .as_object_mut()
        .ok_or_else(|| invalid("GLM attention geometry must be an object"))?
        .insert("name".into(), Value::String(String::new()));
    let op: Glm53AttentionOp = serde_json::from_value(named).map_err(|e| invalid(e.to_string()))?;
    op.validate()?;
    let key = geometry_key(&op)
        .filter(|_| op.measured.is_empty())
        .ok_or_else(|| invalid("GLM attention table rows describe sparse_mla modules only"))?;
    if geometry(&op)? != encoded {
        return Err(invalid(
            "GLM attention geometry has unknown or noncanonical fields",
        ));
    }
    Ok(key)
}

impl Glm53AttentionTable {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            path: Some(data_root.join(BASENAME)),
            grids: OnceLock::new(),
        }
    }

    /// Exact system/backend/version primary only (family-first or legacy
    /// layout). Declared donors, earlier siblings and cross-backend fill are
    /// never consulted for GLM attention.
    pub fn with_sources(data_root: &Path, resolver: &SourceResolver) -> Result<Self, AicError> {
        let primary = resolver
            .prioritized_sources_for(BASENAME, data_root)?
            .into_iter()
            .find(|source| source.channel == "primary");
        if primary
            .as_ref()
            .is_some_and(|source| source.source.kernel_sources().is_some())
        {
            return Err(invalid(
                "GLM attention primary kernel_sources filters are not supported",
            ));
        }
        let path = primary.map(|source| source.source.0);
        if let Some(path) = &path {
            let version = data_root.file_name();
            let backend = data_root.parent().and_then(Path::file_name);
            let parent = path.parent();
            if parent.and_then(Path::file_name) != version
                || parent.and_then(Path::parent).and_then(Path::file_name) != backend
            {
                return Err(invalid(
                    "GLM attention primary source must belong to the requested backend and version",
                ));
            }
        }
        Ok(Self {
            path,
            grids: OnceLock::new(),
        })
    }

    pub fn primary_path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Loaded and validated geometry grids; a missing file is empty.
    fn grids(&self) -> Result<&Grids, AicError> {
        self.grids
            .get_or_init(|| match &self.path {
                Some(path) => load(path).map_err(|e| format!("{}: {e}", path.display())),
                None => Ok(Grids::default()),
            })
            .as_ref()
            .map_err(|e| invalid(e.clone()))
    }

    /// Rows for this op's exact geometry. Load or validation failures are
    /// errors and take precedence over a coverage gap.
    fn batches(&self, op: &Glm53AttentionOp) -> Result<Option<&BatchCurves>, AicError> {
        let grids = self.grids()?;
        Ok(geometry_key(op).and_then(|key| grids.get(&key)))
    }

    /// Whether the validated table holds rows for this op's exact geometry
    /// (TP/heads, checkpoint, backend and phase). Load or validation
    /// failures are errors, not a coverage gap.
    pub fn covers(&self, op: &Glm53AttentionOp) -> Result<bool, AicError> {
        Ok(self.batches(op)?.is_some())
    }

    /// `sol(batch, prefix, x)` is the operator's analytical latency; it anchors
    /// utilization interpolation and boundary holds. `Ok(None)` is a coverage
    /// gap (no file or no rows for this geometry).
    pub fn query(
        &self,
        op: &Glm53AttentionOp,
        batch: u32,
        prefix: u32,
        x: u32,
        sol: &dyn Fn(f64, f64, f64) -> Result<f64, AicError>,
    ) -> Result<Option<f64>, AicError> {
        let Some(batches) = self.batches(op)? else {
            return Ok(None);
        };
        if batch == 0 || x == 0 {
            return Ok(Some(0.0));
        }
        let failure = RefCell::new(None);
        let sol_at = |b: f64, p: f64, w: f64| match sol(b, p, w) {
            Ok(value) if value.is_finite() && value > 0.0 => value,
            Ok(_) => {
                failure
                    .borrow_mut()
                    .get_or_insert(invalid("GLM attention SOL anchor is not positive"));
                f64::NAN
            }
            Err(err) => {
                failure.borrow_mut().get_or_insert(err);
                f64::NAN
            }
        };
        let util_x = |b: u32, p: u32, curve: &Curve| {
            interpolate(curve, f64::from(x), |w, latency| {
                sol_at(f64::from(b), f64::from(p), w) / latency
            })
        };
        let util_prefix = |b: u32, prefixes: &PrefixCurves| {
            interpolate(prefixes, f64::from(prefix), |p, curve| {
                util_x(b, p as u32, curve)
            })
        };
        let util = interpolate(batches, f64::from(batch), |b, prefixes| {
            util_prefix(b as u32, prefixes)
        });
        let latency = sol_at(f64::from(batch), f64::from(prefix), f64::from(x)) / util;
        if let Some(err) = failure.into_inner() {
            return Err(err);
        }
        if !latency.is_finite() || latency <= 0.0 {
            return Err(invalid(
                "GLM attention interpolation produced an invalid latency",
            ));
        }
        Ok(Some(latency))
    }
}

/// Linear interpolation of `value(coordinate, entry)` between the two
/// bracketing coordinates; outside the measured range the nearest boundary
/// value is held (utilization hold).
fn interpolate<T>(map: &BTreeMap<u32, T>, at: f64, value: impl Fn(f64, &T) -> f64) -> f64 {
    let below = map.range(..=at.floor().max(0.0) as u32).next_back();
    let above = map.range(at.ceil().max(0.0) as u32..).next();
    match (below, above) {
        (Some((&lo, a)), Some((&hi, b))) if hi > lo => {
            let t = (at - f64::from(lo)) / f64::from(hi - lo);
            (1.0 - t) * value(f64::from(lo), a) + t * value(f64::from(hi), b)
        }
        (Some((&lo, a)), _) => value(f64::from(lo), a),
        (None, Some((&hi, b))) => value(f64::from(hi), b),
        (None, None) => f64::NAN,
    }
}

fn load(path: &Path) -> Result<Grids, AicError> {
    if !path.try_exists().map_err(|e| invalid(e.to_string()))? {
        return Ok(Grids::default());
    }
    use super::parquet_loader::PerfReader;
    let reader = PerfReader::open(path)?;
    let component = reader.col("component")?;
    let geometry_col = reader.col("geometry")?;
    let batch_size = reader.col("batch_size")?;
    let prefix_col = reader.col("prefix")?;
    let x_col = reader.col("x")?;
    let latency_col = reader.col("latency")?;
    let kernel_source = reader.col("kernel_source")?;
    let measurement_scope = reader.col("measurement_scope")?;
    let source_sha256 = reader.col("source_sha256")?;
    let config_sha256 = reader.col("config_sha256")?;
    let runtime_digest = reader.col("runtime_digest")?;
    let used_cuda_graph = reader.col("used_cuda_graph")?;
    let sample_count = reader.col("sample_count")?;
    let kv_seed_regime = reader.col("kv_seed_regime")?;
    let execution_profile = reader.col("execution_profile")?;
    // One runtime/source identity per file. One <backend>/<version> directory
    // holds both checkpoints, so the configuration identity is homogeneous per
    // checkpoint_format; the CUDA-graph mode is homogeneous per geometry
    // (checkpoint, TP, phase), because serving runs prefill eagerly and decode
    // under full CUDA graphs.
    let mut runtime_identity: Option<(String, String)> = None;
    let mut configs: FxHashMap<CheckpointFormat, String> = FxHashMap::default();
    let mut graphs: FxHashMap<GeometryKey, bool> = FxHashMap::default();
    let mut validated: BTreeMap<String, GeometryKey> = BTreeMap::new();
    // One latent KV dtype per file: FP8- and BF16-KV deployments are distinct
    // identities collected by distinct runs and never share a table.
    let mut kv_cache_dtype: Option<KvCacheQuantMode> = None;
    let mut grids = Grids::default();
    for row in reader.rows()? {
        let row = row?;
        if row.str(component)? != "attention" {
            return Err(invalid("GLM attention table component must be attention"));
        }
        let encoded = row.str(geometry_col)?;
        let key = match validated.get(encoded) {
            Some(&key) => key,
            None => {
                let key = validate_geometry(encoded)?;
                if *kv_cache_dtype.get_or_insert(key.kv_cache_dtype) != key.kv_cache_dtype {
                    return Err(invalid(
                        "GLM attention table mixes FP8 and BF16 latent KV identities",
                    ));
                }
                validated.insert(encoded.to_owned(), key);
                key
            }
        };
        let is_context = key.is_context;
        let (batch, prefix, x) = (row.u32(batch_size)?, row.u32(prefix_col)?, row.u32(x_col)?);
        let latency = row.f64(latency_col)?;
        if batch == 0
            || x == 0
            || !latency.is_finite()
            || latency <= 0.0
            || row.u32(sample_count)? == 0
        {
            return Err(invalid(
                "GLM attention sample requires positive work, latency and sample_count",
            ));
        }
        if row.str(kernel_source)?.trim().is_empty()
            || row.str(measurement_scope)? != "local_compute"
            || row.str(execution_profile)? != "full"
        {
            return Err(invalid(
                "GLM attention samples require a kernel, local_compute scope and full execution",
            ));
        }
        let regime = row.str(kv_seed_regime)?;
        if !matches!(regime, "real_kv" | "n/a")
            || ((!is_context || prefix > 0) && regime != "real_kv")
        {
            return Err(invalid(
                "GLM decode/cached-prefill attention requires real KV initialization",
            ));
        }
        if !is_context && prefix != 0 {
            return Err(invalid(
                "GLM decode attention uses absolute KV length with prefix=0",
            ));
        }
        let runtime = (
            row.str_owned(source_sha256)?,
            row.str_owned(runtime_digest)?,
        );
        let config = row.str_owned(config_sha256)?;
        let graph = row.bool_strict(used_cuda_graph)?;
        if !valid_sha256(&runtime.0)
            || !valid_sha256(&config)
            || !runtime.1.strip_prefix("sha256:").is_some_and(valid_sha256)
        {
            return Err(invalid(
                "GLM attention provenance requires complete SHA256 identities",
            ));
        }
        if runtime_identity.get_or_insert_with(|| runtime.clone()) != &runtime
            || configs
                .entry(key.checkpoint_format)
                .or_insert_with(|| config.clone())
                != &config
            || *graphs.entry(key).or_insert(graph) != graph
        {
            return Err(invalid(
                "GLM attention table mixes runtime/source identities, a checkpoint's config, \
                 or one geometry's CUDA graph mode",
            ));
        }
        if grids
            .entry(key)
            .or_default()
            .entry(batch)
            .or_default()
            .entry(prefix)
            .or_default()
            .insert(x, latency)
            .is_some()
        {
            return Err(invalid(
                "duplicate GLM attention physical key and work coordinate",
            ));
        }
    }
    if grids.is_empty() {
        return Err(invalid("present GLM attention module table is empty"));
    }
    Ok(grids)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::operators::glm53flash::tests::attention;
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};

    const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    pub(crate) fn rows(
        op: &Glm53AttentionOp,
        points: &[(i64, i64, i64, f64)],
        regime: &'static str,
    ) -> Vec<Col> {
        let geometry: &'static str = geometry(op).unwrap().leak();
        let n = points.len();
        vec![
            Col::Str("component", vec!["attention"; n]),
            Col::Str("geometry", vec![geometry; n]),
            Col::I64("batch_size", points.iter().map(|p| p.0).collect()),
            Col::I64("prefix", points.iter().map(|p| p.1).collect()),
            Col::I64("x", points.iter().map(|p| p.2).collect()),
            Col::F64("latency", points.iter().map(|p| p.3).collect()),
            Col::Str("kernel_source", vec!["glm53_sparse_mla_module"; n]),
            Col::Str("measurement_scope", vec!["local_compute"; n]),
            Col::Str("source_sha256", vec![SHA; n]),
            Col::Str("config_sha256", vec![SHA; n]),
            Col::Str("runtime_digest", vec![DIGEST; n]),
            Col::Bool("used_cuda_graph", vec![false; n]),
            Col::I64("sample_count", vec![5; n]),
            Col::Str("kv_seed_regime", vec![regime; n]),
            Col::Str("execution_profile", vec!["full"; n]),
        ]
    }

    fn table(columns: &[Col]) -> (tempfile::TempDir, Glm53AttentionTable) {
        let root = tempfile::tempdir().unwrap();
        write_parquet(&root.path().join(BASENAME), columns);
        let table = Glm53AttentionTable::new(root.path().to_owned());
        (root, table)
    }

    #[test]
    fn geometry_is_canonical_and_excludes_display_name() {
        let mut op = attention("sparse_mla");
        let key = geometry(&op).unwrap();
        assert!(!key.contains("\"name\"") && !key.contains("measured"));
        assert!(key.starts_with("{\"backend\":\"vllm\",\"checkpoint_format\":\"nvfp4\""));
        validate_geometry(&key).unwrap();
        op.name = "attention_43".into();
        assert_eq!(geometry(&op).unwrap(), key);
        op.tp_size = 4;
        op.num_heads = 16;
        assert_ne!(geometry(&op).unwrap(), key);
        assert!(validate_geometry(&geometry(&attention("kda")).unwrap()).is_err());
    }

    #[test]
    fn typed_key_changes_exactly_when_canonical_geometry_does() {
        use crate::operators::op::Op;
        let op = attention("sparse_mla");
        let original = serde_json::to_value(&op).unwrap();
        let reference = geometry(&op).unwrap();
        let key = geometry_key(&op).unwrap();
        for (field, value) in original.as_object().unwrap() {
            let mut changed = original.clone();
            changed[field] = match (field.as_str(), value) {
                ("measured", _) => {
                    serde_json::to_value(vec![Op::Glm53Attention(attention("kda"))]).unwrap()
                }
                ("gate_lower_bound", _) => serde_json::json!(-4.0),
                (_, Value::Bool(v)) => Value::Bool(!v),
                (_, Value::Number(v)) => serde_json::json!(v.as_u64().unwrap() + 1),
                (_, Value::String(_)) => serde_json::json!(match field.as_str() {
                    "name" => "attention_43",
                    "layer_kind" => "kda",
                    "backend" => "sglang",
                    "checkpoint_format" => "fp8",
                    "projection_quant_mode" => "fp8_block",
                    "kv_cache_dtype" => "bfloat16",
                    _ => panic!("uncovered string field {field}"),
                }),
                _ => panic!("uncovered field {field}"),
            };
            let altered: Glm53AttentionOp = serde_json::from_value(changed).unwrap();
            let excluded = matches!(field.as_str(), "name" | "measured");
            assert_eq!(
                geometry(&altered).unwrap() == reference,
                excluded,
                "{field}"
            );
            assert_eq!(geometry_key(&altered) == Some(key), excluded, "{field}");
        }
    }

    #[test]
    fn exact_hits_interpolate_and_hold_utilization() {
        let op = attention("sparse_mla");
        let (_root, table) = table(&rows(
            &op,
            &[
                (1, 0, 10, 1.0),
                (1, 0, 20, 4.0),
                (2, 0, 10, 3.0),
                (2, 0, 20, 8.0),
            ],
            "n/a",
        ));
        // SOL = batch * x: exact hits verbatim.
        let sol = |b: f64, _p: f64, x: f64| Ok(b * x);
        let q = |b, x| table.query(&op, b, 0, x, &sol).unwrap().unwrap();
        assert!((q(1, 10) - 1.0).abs() < 1e-12);
        assert!((q(2, 20) - 8.0).abs() < 1e-12);
        // x=15, batch 1: util 10/1=10 and 20/4=5 -> 7.5; SOL 15 -> 2.0.
        assert!((q(1, 15) - 2.0).abs() < 1e-12);
        // Beyond range holds the boundary utilization (20/4=5): SOL 40 -> 8.
        assert!((q(1, 40) - 8.0).abs() < 1e-12);
        // Batch 3 holds batch-2 utilization: util(2,10)=20/3; SOL 30 -> 4.5.
        assert!((q(3, 10) - 4.5).abs() < 1e-12);
        // Unknown geometry is a coverage gap.
        let mut other = op.clone();
        other.backend = "sglang".into();
        assert!(table.query(&other, 1, 0, 10, &sol).unwrap().is_none());
        // A non-finite gate (canonical `null`) matches no row.
        let mut nan = op.clone();
        nan.gate_lower_bound = f64::NAN;
        assert!(table.query(&nan, 1, 0, 10, &sol).unwrap().is_none());
    }

    fn with_provenance(mut columns: Vec<Col>, config: &'static str, graph: bool) -> Vec<Col> {
        let n = match &columns[0] {
            Col::Str(_, values) => values.len(),
            _ => unreachable!(),
        };
        columns[9] = Col::Str("config_sha256", vec![config; n]);
        columns[11] = Col::Bool("used_cuda_graph", vec![graph; n]);
        columns
    }

    fn concat(parts: Vec<Vec<Col>>) -> Vec<Col> {
        let mut parts = parts.into_iter();
        let mut out = parts.next().unwrap();
        for part in parts {
            for (column, extra) in out.iter_mut().zip(part) {
                match (column, extra) {
                    (Col::Str(_, a), Col::Str(_, b)) => a.extend(b),
                    (Col::I64(_, a), Col::I64(_, b)) => a.extend(b),
                    (Col::F64(_, a), Col::F64(_, b)) => a.extend(b),
                    (Col::Bool(_, a), Col::Bool(_, b)) => a.extend(b),
                    _ => unreachable!(),
                }
            }
        }
        out
    }

    #[test]
    fn one_table_holds_both_checkpoints_and_phase_graph_modes() {
        const FP8_CONFIG: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let nvfp4_prefill = attention("sparse_mla");
        let mut nvfp4_decode = nvfp4_prefill.clone();
        nvfp4_decode.is_context = false;
        let mut fp8_decode = nvfp4_decode.clone();
        fp8_decode.checkpoint_format = "fp8".into();
        let parts = || {
            vec![
                with_provenance(
                    rows(&nvfp4_prefill, &[(1, 0, 10, 1.0)], "real_kv"),
                    SHA,
                    false,
                ),
                with_provenance(
                    rows(&nvfp4_decode, &[(1, 0, 10, 0.5)], "real_kv"),
                    SHA,
                    true,
                ),
                with_provenance(
                    rows(&fp8_decode, &[(1, 0, 10, 0.25)], "real_kv"),
                    FP8_CONFIG,
                    true,
                ),
            ]
        };
        let (_root, combined) = table(&concat(parts()));
        let sol = |_b: f64, _p: f64, x: f64| Ok(x);
        assert_eq!(
            combined.query(&nvfp4_prefill, 1, 0, 10, &sol).unwrap(),
            Some(1.0)
        );
        assert_eq!(
            combined.query(&fp8_decode, 1, 0, 10, &sol).unwrap(),
            Some(0.25)
        );
        // A checkpoint cannot mix configurations.
        let mut mixed = parts();
        mixed.push(with_provenance(
            rows(&fp8_decode, &[(2, 0, 10, 0.5)], "real_kv"),
            SHA,
            true,
        ));
        let (_root, bad) = table(&concat(mixed));
        assert!(bad.query(&fp8_decode, 1, 0, 10, &sol).is_err());
        // One geometry cannot mix eager and graph timings.
        let mut mixed = parts();
        mixed.push(with_provenance(
            rows(&nvfp4_decode, &[(2, 0, 10, 0.5)], "real_kv"),
            SHA,
            false,
        ));
        let (_root, bad) = table(&concat(mixed));
        assert!(bad.query(&nvfp4_decode, 1, 0, 10, &sol).is_err());
        // The runtime/source identity stays table-wide.
        let mut mixed = parts();
        let mut other = with_provenance(
            rows(&nvfp4_prefill, &[(2, 0, 10, 1.0)], "real_kv"),
            SHA,
            false,
        );
        other[8] = Col::Str("source_sha256", vec![FP8_CONFIG]);
        mixed.push(other);
        let (_root, bad) = table(&concat(mixed));
        assert!(bad.query(&nvfp4_prefill, 1, 0, 10, &sol).is_err());
    }

    #[test]
    fn malformed_or_borrowed_rows_are_fatal() {
        let op = attention("sparse_mla");
        let mut decode = op.clone();
        decode.is_context = false;
        // Decode requires real KV initialization.
        let (_root, decode_table) = table(&rows(&decode, &[(1, 0, 10, 1.0)], "n/a"));
        let sol = |_b: f64, _p: f64, x: f64| Ok(x);
        assert!(decode_table.query(&decode, 1, 0, 10, &sol).is_err());
        // KDA geometry cannot be a row of the sparse attention table.
        let (_root2, kda_table) = table(&rows(&attention("kda"), &[(1, 0, 10, 1.0)], "n/a"));
        assert!(kda_table.query(&op, 1, 0, 10, &sol).is_err());
        // Absent file is only a coverage gap.
        let empty = Glm53AttentionTable::new(tempfile::tempdir().unwrap().path().to_owned());
        assert!(empty.query(&op, 1, 0, 10, &sol).unwrap().is_none());
    }
}
