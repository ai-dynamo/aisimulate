// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! V4.1 local-compute measurements, isolated by exact geometry and runtime.
//! Integer columns use physical INT64 constrained to u32. Latency is in ms.
//! A missing file/key is a coverage gap; every malformed present file is fatal,
//! including in HYBRID mode. No V4 or cross-version source inheritance occurs.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use super::SourceResolver;
use super::axis_curve::LeafAxisCurve;
use super::parquet_loader::PerfReader;
use super::perf_interp::LeafValue;
use crate::common::error::AicError;
use crate::operators::dsv41::{Dsv41AttentionOp, Dsv41EngramOp, Dsv41LinearOp, Dsv41MhcOp};

const BASENAME: &str = "dsv41_module_perf.parquet";

pub struct Dsv41Table {
    path: Option<PathBuf>,
    grids: OnceLock<Result<Grids, String>>,
}

#[derive(Default)]
struct Grids {
    curves: BTreeMap<Key, LeafAxisCurve>,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    component: String,
    geometry: String,
    batch_size: u32,
    prefix: u32,
}

#[derive(Debug, PartialEq, Eq)]
struct Provenance {
    source_sha256: String,
    config_sha256: String,
    runtime_digest: String,
    used_cuda_graph: bool,
}

fn invalid(message: impl Into<String>) -> AicError {
    AicError::InvalidPerfData(message.into())
}

/// Serialize the measured module geometry with sorted keys.
/// Display names and the analytical KV-layout selector are not table dimensions.
/// The latter changes SOL work/bytes, not the source-pinned measured module key.
pub fn geometry<T: Serialize>(op: &T) -> Result<String, AicError> {
    let mut value = serde_json::to_value(op).map_err(|e| invalid(e.to_string()))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| invalid("V41 geometry must be an object"))?;
    object.remove("name");
    // Keep the existing measurement schema independent of a newly appended
    // analytical operator field. validate_body still compares this exact body
    // to the stored descriptor, so an explicit layout field (or any unknown
    // field) in a table remains noncanonical and is rejected.
    object.remove("kv_cache_layout");
    let sorted: BTreeMap<_, _> = object.iter().collect();
    serde_json::to_string(&sorted).map_err(|e| invalid(e.to_string()))
}

fn validate_body<T: DeserializeOwned + Serialize>(value: &Value) -> Result<(), AicError> {
    let mut named = value.clone();
    named
        .as_object_mut()
        .ok_or_else(|| invalid("V41 geometry must be an object"))?
        .insert("name".into(), Value::String(String::new()));
    let op: T = serde_json::from_value(named).map_err(|e| invalid(e.to_string()))?;
    let round_trip: Value =
        serde_json::from_str(&geometry(&op)?).map_err(|e| invalid(e.to_string()))?;
    if &round_trip != value {
        return Err(invalid("V41 geometry has unknown or noncanonical fields"));
    }
    Ok(())
}

fn validate_geometry(component: &str, encoded: &str) -> Result<Value, AicError> {
    let value: Value = serde_json::from_str(encoded).map_err(|e| invalid(e.to_string()))?;
    match component {
        "attention" => validate_body::<Dsv41AttentionOp>(&value)?,
        "mhc" => validate_body::<Dsv41MhcOp>(&value)?,
        "engram" => validate_body::<Dsv41EngramOp>(&value)?,
        "linear" => validate_body::<Dsv41LinearOp>(&value)?,
        _ => return Err(invalid(format!("unknown V41 component {component:?}"))),
    }
    let object = value.as_object().expect("validated object");
    let sorted: BTreeMap<_, _> = object.iter().collect();
    if serde_json::to_string(&sorted).map_err(|e| invalid(e.to_string()))? != encoded {
        return Err(invalid("V41 geometry must use canonical sorted JSON"));
    }
    for (key, value) in object {
        if value.is_number()
            && value.as_u64() == Some(0)
            && !matches!(key.as_str(), "compress_ratio" | "candidate_limit")
        {
            return Err(invalid(format!("V41 geometry {key} must be positive")));
        }
    }
    if component == "attention" {
        let role = value["role"].as_str().unwrap_or_default();
        if !matches!(role, "swa" | "full" | "reindex" | "reuse") {
            return Err(invalid("invalid V41 CSA2 role"));
        }
        let ratio = value["compress_ratio"].as_u64().unwrap_or(u64::MAX);
        if ratio > 2 || (role == "swa") != (ratio == 0) {
            return Err(invalid("invalid V41 CSA2 compression ratio"));
        }
    }
    Ok(value)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl Dsv41Table {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            path: Some(data_root.join(BASENAME)),
            grids: OnceLock::new(),
        }
    }

    /// Honor family-first primary discovery and explicit admission vetoes,
    /// excluding declared/sibling/cross-backend donors. A filtered primary is
    /// unsupported: never discard its requested kernel admission constraint.
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
                "V41 primary kernel_sources filters are not supported; select an unfiltered homogeneous module table",
            ));
        }
        let path = primary.map(|source| source.source.0);
        if let Some(path) = &path {
            let system_root = data_root
                .parent()
                .and_then(Path::parent)
                .ok_or_else(|| invalid("V41 data root must include system/backend/version"))?;
            let belongs_to_request = |source: &Path, root: &Path| {
                let Ok(relative) = source.strip_prefix(root) else {
                    return false;
                };
                let parts: Vec<_> = relative.components().collect();
                // Only <backend>/<version>/<file> or one family directory
                // beneath this system root. Parent traversal is never a donor.
                matches!(parts.len(), 3 | 4)
                    && parts
                        .iter()
                        .all(|part| matches!(part, Component::Normal(_)))
                    && source.file_name().is_some_and(|name| name == BASENAME)
                    && source.parent().and_then(Path::file_name) == data_root.file_name()
                    && source
                        .parent()
                        .and_then(Path::parent)
                        .and_then(Path::file_name)
                        == data_root.parent().and_then(Path::file_name)
            };
            if !belongs_to_request(path, system_root) {
                return Err(invalid(
                    "V41 primary source must belong to the requested system, backend and version",
                ));
            }
            // A whole data root may be relocated through a symlink. Resolve
            // both sides together, while rejecting a file/family symlink that
            // borrows another system's measurements. Absent data stays a gap.
            if path.try_exists().map_err(|e| invalid(e.to_string()))? {
                let resolved_source = path.canonicalize().map_err(|e| invalid(e.to_string()))?;
                let resolved_root = system_root
                    .canonicalize()
                    .map_err(|e| invalid(e.to_string()))?;
                if !belongs_to_request(&resolved_source, &resolved_root) {
                    return Err(invalid(
                        "V41 primary source resolves outside the requested system, backend and version",
                    ));
                }
            }
        }
        Ok(Self {
            path,
            grids: OnceLock::new(),
        })
    }

    /// Exact component/geometry/batch/prefix buckets; only the work axis is
    /// interpolated. Attention uses query length (context) or absolute KV length
    /// (decode). Other components use total tokens with batch=1 and prefix=0.
    pub fn query<T: Serialize>(
        &self,
        component: &str,
        op: &T,
        batch_size: u32,
        prefix: u32,
        x: u32,
        sol: &dyn Fn(f64) -> Result<f64, AicError>,
    ) -> Result<Option<LeafValue>, AicError> {
        let grids = self.grids.get_or_init(|| match &self.path {
            Some(path) => load(path).map_err(|e| format!("{}: {e}", path.display())),
            None => Ok(Grids::default()),
        });
        let grids = grids.as_ref().map_err(|e| invalid(e.clone()))?;
        let key = Key {
            component: component.into(),
            geometry: geometry(op)?,
            batch_size,
            prefix,
        };
        let Some(curve) = grids.curves.get(&key) else {
            return Ok(None);
        };
        // Preserve analytic errors across the shared scalar curve callback.
        let failure = RefCell::new(None);
        let result = curve.query(f64::from(x), &|point| match sol(point) {
            Ok(value) => value,
            Err(err) => {
                *failure.borrow_mut() = Some(err);
                f64::NAN
            }
        });
        if let Some(err) = failure.into_inner() {
            return Err(err);
        }
        let leaf = result?;
        if !leaf.latency.is_finite() || leaf.latency <= 0.0 {
            return Err(invalid("V41 interpolation produced an invalid latency"));
        }
        Ok(Some(leaf))
    }
}

fn load(path: &Path) -> Result<Grids, AicError> {
    // try_exists preserves permission/I/O failures; only absence is coverage.
    if !path.try_exists().map_err(|e| invalid(e.to_string()))? {
        return Ok(Grids::default());
    }
    let reader = PerfReader::open(path)?;
    let component = reader.col("component")?;
    let geometry = reader.col("geometry")?;
    let batch_size = reader.col("batch_size")?;
    let prefix = reader.col("prefix")?;
    let x = reader.col("x")?;
    let latency = reader.col("latency")?;
    let kernel_source = reader.col("kernel_source")?;
    let measurement_scope = reader.col("measurement_scope")?;
    let source_sha256 = reader.col("source_sha256")?;
    let config_sha256 = reader.col("config_sha256")?;
    let runtime_digest = reader.col("runtime_digest")?;
    let used_cuda_graph = reader.col("used_cuda_graph")?;
    let sample_count = reader.col("sample_count")?;
    let kv_seed_regime = reader.col("kv_seed_regime")?;
    let execution_profile = reader.col("execution_profile")?;
    let mut identity = None;
    let mut points: BTreeMap<Key, BTreeMap<u32, LeafValue>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let component = row.str(component)?;
        let encoded = row.str(geometry)?;
        let shape = validate_geometry(component, encoded)?;
        let (batch, prefix, x) = (row.u32(batch_size)?, row.u32(prefix)?, row.u32(x)?);
        let latency = row.f64(latency)?;
        if batch == 0
            || x == 0
            || !latency.is_finite()
            || latency <= 0.0
            || row.u32(sample_count)? == 0
        {
            return Err(invalid(
                "V41 sample requires positive work, latency and sample_count",
            ));
        }
        if row.str(kernel_source)?.trim().is_empty()
            || row.str(measurement_scope)? != "local_compute"
        {
            return Err(invalid(
                "V41 samples require an actual kernel and local_compute scope excluding collectives",
            ));
        }
        let profile = row.str(execution_profile)?;
        if !matches!(profile, "full" | "decoder_bounded") {
            return Err(invalid("unknown V41 execution_profile"));
        }
        let regime = row.str(kv_seed_regime)?;
        if !matches!(regime, "real_kv" | "n/a") {
            return Err(invalid("unknown V41 kv_seed_regime"));
        }
        if component == "attention" {
            let is_context = shape["is_context"].as_bool().expect("typed field");
            if (!is_context || prefix > 0) && regime != "real_kv" {
                return Err(invalid(
                    "V41 decode/cached-prefill requires real KV initialization",
                ));
            }
            if !is_context && prefix != 0 {
                return Err(invalid("V41 decode uses absolute KV length with prefix=0"));
            }
            if shape["bounded_prefill"] == true
                && (profile != "decoder_bounded"
                    || !is_context
                    || u64::from(x) > shape["window_size"].as_u64().unwrap())
            {
                return Err(invalid(
                    "V41 bounded prefill sample contradicts its execution profile or window",
                ));
            }
        } else if batch != 1 || prefix != 0 || regime != "n/a" {
            return Err(invalid(
                "V41 non-attention rows require batch=1, prefix=0 and kv_seed_regime=n/a",
            ));
        }
        let provenance = Provenance {
            source_sha256: row.str_owned(source_sha256)?,
            config_sha256: row.str_owned(config_sha256)?,
            runtime_digest: row.str_owned(runtime_digest)?,
            used_cuda_graph: row.bool_strict(used_cuda_graph)?,
        };
        if !valid_sha256(&provenance.source_sha256)
            || !valid_sha256(&provenance.config_sha256)
            || !provenance
                .runtime_digest
                .strip_prefix("sha256:")
                .is_some_and(valid_sha256)
        {
            return Err(invalid(
                "V41 provenance requires complete SHA256 identities",
            ));
        }
        match &identity {
            Some(expected) if expected != &provenance => {
                return Err(invalid(
                    "V41 table mixes runtime/source/config or CUDA graph identities",
                ));
            }
            None => identity = Some(provenance),
            _ => {}
        }
        let key = Key {
            component: component.into(),
            geometry: encoded.into(),
            batch_size: batch,
            prefix,
        };
        if points
            .entry(key)
            .or_default()
            .insert(x, LeafValue::with_power(latency, 0.0))
            .is_some()
        {
            return Err(invalid("duplicate V41 physical key and work coordinate"));
        }
    }
    if points.is_empty() {
        return Err(invalid("present V41 module table is empty"));
    }
    Ok(Grids {
        curves: points
            .into_iter()
            .map(|(key, curve)| (key, LeafAxisCurve::from_map("x", curve)))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::enums::{DatabaseMode, FmhaQuantMode, GemmQuantMode};
    use crate::operators::base::Source;
    use crate::perf_database::PerfDatabase;
    use crate::perf_database::energy_test_fixtures::{
        Col, write_energy_systems_root, write_parquet,
    };

    const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OTHER_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const DIGEST: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const LINEAR: &str = r#"{"k":32,"n":64,"quant_mode":"fp8_block"}"#;

    fn linear() -> Dsv41LinearOp {
        Dsv41LinearOp {
            name: "projection".into(),
            n: 64,
            k: 32,
            quant_mode: GemmQuantMode::Fp8Block,
        }
    }

    fn attention() -> Dsv41AttentionOp {
        Dsv41AttentionOp {
            name: "attention".into(),
            is_context: false,
            role: "reuse".into(),
            compress_ratio: 1,
            hidden_size: 5120,
            num_heads: 16,
            head_dim: 512,
            q_lora_rank: 1280,
            o_lora_rank: 1024,
            o_groups: 2,
            index_n_heads: 16,
            index_head_dim: 128,
            index_topk: 512,
            window_size: 128,
            candidate_limit: 16384,
            is_candidate_source: false,
            bounded_prefill: false,
            gemm_quant_mode: GemmQuantMode::Fp8Block,
            fmha_quant_mode: FmhaQuantMode::Fp8,
            kv_cache_layout: crate::operators::dsv41::Dsv41KvCacheLayout::LogicalFp4,
        }
    }

    fn fixture() -> Vec<Col> {
        vec![
            Col::Str("component", vec!["linear"; 2]),
            Col::Str("geometry", vec![LINEAR; 2]),
            Col::I64("batch_size", vec![1; 2]),
            Col::I64("prefix", vec![0; 2]),
            Col::I64("x", vec![10, 20]),
            Col::F64("latency", vec![1.0, 3.0]),
            Col::Str("kernel_source", vec!["observed.block32.kernel"; 2]),
            Col::Str("measurement_scope", vec!["local_compute"; 2]),
            Col::Str("source_sha256", vec![SHA; 2]),
            Col::Str("config_sha256", vec![SHA; 2]),
            Col::Str("runtime_digest", vec![DIGEST; 2]),
            Col::Bool("used_cuda_graph", vec![false; 2]),
            Col::I64("sample_count", vec![5; 2]),
            Col::Str("kv_seed_regime", vec!["n/a"; 2]),
            Col::Str("execution_profile", vec!["full"; 2]),
        ]
    }

    fn table(columns: &[Col]) -> (tempfile::TempDir, Dsv41Table) {
        let root = tempfile::tempdir().unwrap();
        write_parquet(&root.path().join(BASENAME), columns);
        let table = Dsv41Table::new(root.path().to_owned());
        (root, table)
    }

    fn lookup(table: &Dsv41Table, x: u32) -> Result<Option<LeafValue>, AicError> {
        table.query("linear", &linear(), 1, 0, x, &|x| Ok(x * x))
    }

    #[test]
    fn exact_interpolation_and_f64_sol_util_hold() {
        let (_root, table) = table(&fixture());
        for (x, expected) in [(10, 1.0), (15, 2.0), (20, 3.0), (40, 12.0), (5, 0.25)] {
            assert_eq!(lookup(&table, x).unwrap().unwrap().latency, expected);
        }
    }

    #[test]
    fn geometry_excludes_names_but_never_shapes_or_arithmetic_quantization() {
        assert_eq!(geometry(&linear()).unwrap(), LINEAR);
        let (_root, table) = table(&fixture());
        let mut op = linear();
        op.name = "another-layer".into();
        assert!(
            table
                .query("linear", &op, 1, 0, 10, &|x| Ok(x))
                .unwrap()
                .is_some()
        );
        op.k = 64;
        assert!(
            table
                .query("linear", &op, 1, 0, 10, &|x| Ok(x))
                .unwrap()
                .is_none()
        );
        op = linear();
        op.quant_mode = GemmQuantMode::Bfloat16;
        assert!(
            table
                .query("linear", &op, 1, 0, 10, &|x| Ok(x))
                .unwrap()
                .is_none()
        );
        assert!(
            table
                .query("linear", &linear(), 2, 0, 10, &|x| Ok(x))
                .unwrap()
                .is_none()
        );
        assert!(
            table
                .query("linear", &linear(), 1, 1, 10, &|x| Ok(x))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn analytical_layout_preserves_the_existing_attention_measurement_key() {
        use crate::operators::dsv41::Dsv41KvCacheLayout;
        let mut op = attention();
        let legacy = geometry(&op).unwrap();
        assert!(!legacy.contains("kv_cache_layout"));
        validate_geometry("attention", &legacy).unwrap();
        op.kv_cache_layout = Dsv41KvCacheLayout::SglangFp8Bf16;
        assert_eq!(geometry(&op).unwrap(), legacy);
        op.window_size += 1;
        assert_ne!(geometry(&op).unwrap(), legacy);
    }

    #[test]
    fn physical_layout_query_uses_legacy_table_without_erasing_measured_dimensions() {
        use crate::operators::dsv41::Dsv41KvCacheLayout;
        // The pre-layout measurement descriptor is deliberately literal: the
        // test must not generate its old input using the new key serializer.
        const LEGACY: &str = r#"{"bounded_prefill":false,"candidate_limit":16384,"compress_ratio":1,"fmha_quant_mode":"fp8","gemm_quant_mode":"fp8_block","head_dim":512,"hidden_size":5120,"index_head_dim":128,"index_n_heads":16,"index_topk":512,"is_candidate_source":false,"is_context":false,"num_heads":16,"o_groups":2,"o_lora_rank":1024,"q_lora_rank":1280,"role":"reuse","window_size":128}"#;
        let mut op = attention();
        let mut columns = attention_fixture(&op, 0, "real_kv");
        columns[1] = Col::Str("geometry", vec![LEGACY; 2]);
        let (_root, table) = table(&columns);
        op.kv_cache_layout = Dsv41KvCacheLayout::SglangFp8Bf16;
        assert_eq!(geometry(&op).unwrap(), LEGACY);
        let estimate = table
            .query("attention", &op, 1, 0, 10, &|_| {
                Err(AicError::ModelConfig(
                    "unexpected analytical fallback".into(),
                ))
            })
            .unwrap()
            .unwrap();
        assert_eq!(estimate.latency, 1.0);
        op.fmha_quant_mode = FmhaQuantMode::Bfloat16;
        assert!(
            table
                .query("attention", &op, 1, 0, 10, &|x| Ok(x))
                .unwrap()
                .is_none()
        );
        op.fmha_quant_mode = FmhaQuantMode::Fp8;
        op.head_dim = 256;
        assert!(
            table
                .query("attention", &op, 1, 0, 10, &|x| Ok(x))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn stored_analytical_layout_and_unknown_fields_remain_invalid() {
        let body: Value = serde_json::from_str(&geometry(&attention()).unwrap()).unwrap();
        for (key, value) in [
            ("kv_cache_layout", serde_json::json!("logical_fp4")),
            ("kv_cache_layout", serde_json::json!("sglang_fp8_bf16")),
            (
                "unrecognized_measurement_dimension",
                serde_json::json!(true),
            ),
        ] {
            let mut altered = body.clone();
            altered[key] = value;
            let encoded = serde_json::to_string(&altered).unwrap();
            assert!(matches!(
                validate_geometry("attention", &encoded),
                Err(AicError::InvalidPerfData(message))
                    if message.contains("unknown or noncanonical fields")
            ));
        }
    }

    #[test]
    fn missing_file_is_coverage_without_legacy_or_sibling_reuse() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("gemm_perf.parquet"), b"legacy data").unwrap();
        std::fs::write(root.path().join("dsv4_module_perf.parquet"), b"legacy data").unwrap();
        let sibling = root.path().join("other-version");
        std::fs::create_dir(&sibling).unwrap();
        write_parquet(&sibling.join(BASENAME), &fixture());
        assert!(
            lookup(&Dsv41Table::new(root.path().to_owned()), 10)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn family_first_discovery_and_explicit_veto_are_preserved() {
        use crate::config::PerfDbSources;
        let root = tempfile::tempdir().unwrap();
        let data = write_energy_systems_root(root.path());
        let family = root.path().join("data/dsv41/vllm/1.0");
        std::fs::create_dir_all(&family).unwrap();
        write_parquet(&family.join(BASENAME), &fixture());
        let resolver = SourceResolver::fixed(PerfDbSources::default());
        let table = Dsv41Table::with_sources(&data, &resolver).unwrap();
        assert_eq!(lookup(&table, 10).unwrap().unwrap().latency, 1.0);
        let resolver = SourceResolver::fixed(BTreeMap::from([(BASENAME.into(), vec![])]));
        let table = Dsv41Table::with_sources(&data, &resolver).unwrap();
        assert!(lookup(&table, 10).unwrap().is_none());
    }

    #[test]
    fn live_resolution_does_not_promote_a_sibling_when_primary_is_vetoed() {
        let root = tempfile::tempdir().unwrap();
        let data = write_energy_systems_root(root.path());
        write_parquet(&data.join(BASENAME), &fixture());
        std::fs::write(data.join("INCOMPLETE.txt"), "incomplete").unwrap();
        let sibling = root.path().join("data/dsv41/vllm/0.9");
        std::fs::create_dir_all(&sibling).unwrap();
        write_parquet(&sibling.join(BASENAME), &fixture());
        let resolver = SourceResolver::live(
            PerfDatabase::resolve_ctx(root.path(), "testsys", "vllm", "1.0", true, false).unwrap(),
        );
        let table = Dsv41Table::with_sources(&data, &resolver).unwrap();
        assert!(lookup(&table, 10).unwrap().is_none());
    }

    #[test]
    fn explicit_source_cannot_spoof_a_different_runtime_version() {
        use crate::config::PerfSource;
        let root = tempfile::tempdir().unwrap();
        let data = write_energy_systems_root(root.path());
        let wrong_path = root.path().join("data/dsv41/vllm/0.9").join(BASENAME);
        let resolver = SourceResolver::fixed(BTreeMap::from([(
            BASENAME.into(),
            vec![PerfSource(wrong_path, None)],
        )]));
        assert!(matches!(
            Dsv41Table::with_sources(&data, &resolver),
            Err(AicError::InvalidPerfData(_))
        ));
    }

    #[test]
    fn explicit_source_cannot_borrow_another_system_at_the_same_backend_version() {
        use crate::config::PerfSource;
        let root = tempfile::tempdir().unwrap();
        let data = write_energy_systems_root(root.path());
        let donor = root.path().join("other-system/dsv41/vllm/1.0");
        std::fs::create_dir_all(&donor).unwrap();
        write_parquet(&donor.join(BASENAME), &fixture());
        let resolver = SourceResolver::fixed(BTreeMap::from([(
            BASENAME.into(),
            vec![PerfSource(donor.join(BASENAME), None)],
        )]));
        assert!(matches!(
            Dsv41Table::with_sources(&data, &resolver),
            Err(AicError::InvalidPerfData(_))
        ));
    }

    #[test]
    fn explicit_same_system_legacy_and_arbitrary_family_sources_remain_valid() {
        use crate::config::PerfSource;
        let root = tempfile::tempdir().unwrap();
        let data = write_energy_systems_root(root.path());
        for location in ["data/vllm/1.0", "data/native-components/vllm/1.0"] {
            let directory = root.path().join(location);
            std::fs::create_dir_all(&directory).unwrap();
            write_parquet(&directory.join(BASENAME), &fixture());
            let resolver = SourceResolver::fixed(BTreeMap::from([(
                BASENAME.into(),
                vec![PerfSource(directory.join(BASENAME), None)],
            )]));
            let table = Dsv41Table::with_sources(&data, &resolver).unwrap();
            assert_eq!(lookup(&table, 10).unwrap().unwrap().latency, 1.0);
        }
    }

    #[test]
    fn explicit_primary_kernel_filter_cannot_admit_unrequested_kernels() {
        use crate::config::PerfSource;
        let root = tempfile::tempdir().unwrap();
        let data = write_energy_systems_root(root.path());
        write_parquet(&data.join(BASENAME), &fixture());
        for filter in [vec![], vec!["a-different-native-kernel".into()]] {
            let resolver = SourceResolver::fixed(BTreeMap::from([(
                BASENAME.into(),
                vec![PerfSource(data.join(BASENAME), Some(filter))],
            )]));
            assert!(matches!(
                Dsv41Table::with_sources(&data, &resolver),
                Err(AicError::InvalidPerfData(message)) if message.contains("kernel_sources")
            ));
        }
    }

    #[test]
    fn explicit_source_rejects_nested_families_and_parent_traversal() {
        use crate::config::PerfSource;
        let root = tempfile::tempdir().unwrap();
        let data = write_energy_systems_root(root.path());
        for location in [
            "data/nested/family/vllm/1.0",
            "data/../other-system/vllm/1.0",
        ] {
            let resolver = SourceResolver::fixed(BTreeMap::from([(
                BASENAME.into(),
                vec![PerfSource(root.path().join(location).join(BASENAME), None)],
            )]));
            assert!(matches!(
                Dsv41Table::with_sources(&data, &resolver),
                Err(AicError::InvalidPerfData(_))
            ));
        }
    }

    #[cfg(unix)]
    #[test]
    fn root_symlink_relocation_is_valid_but_cross_system_family_symlinks_are_not() {
        use crate::config::PerfSource;
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let data = write_energy_systems_root(root.path());
        write_parquet(&data.join(BASENAME), &fixture());
        let relocated = root.path().join("relocated-data");
        symlink(root.path().join("data"), &relocated).unwrap();
        let relocated_data = relocated.join("vllm/1.0");
        let resolver = SourceResolver::fixed(BTreeMap::from([(
            BASENAME.into(),
            vec![PerfSource(relocated_data.join(BASENAME), None)],
        )]));
        let table = Dsv41Table::with_sources(&relocated_data, &resolver).unwrap();
        assert_eq!(lookup(&table, 10).unwrap().unwrap().latency, 1.0);

        let donor = root.path().join("other-system/vllm/1.0");
        std::fs::create_dir_all(&donor).unwrap();
        write_parquet(&donor.join(BASENAME), &fixture());
        let family = root.path().join("data/borrowed");
        symlink(root.path().join("other-system"), &family).unwrap();
        let resolver = SourceResolver::fixed(BTreeMap::from([(
            BASENAME.into(),
            vec![PerfSource(family.join("vllm/1.0").join(BASENAME), None)],
        )]));
        assert!(matches!(
            Dsv41Table::with_sources(&data, &resolver),
            Err(AicError::InvalidPerfData(_))
        ));
    }

    #[test]
    fn present_corruption_and_missing_columns_are_not_coverage() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join(BASENAME), b"corrupt").unwrap();
        let err = lookup(&Dsv41Table::new(root.path().to_owned()), 10).unwrap_err();
        assert!(matches!(err, AicError::InvalidPerfData(_)));
        assert!(!err.is_missing_perf_data());
        let mut columns = fixture();
        columns.pop();
        let (_root, table) = table(&columns);
        assert!(matches!(
            lookup(&table, 10),
            Err(AicError::InvalidPerfData(_))
        ));
    }

    #[test]
    fn duplicate_physical_points_fail_even_across_execution_profiles() {
        let mut columns = fixture();
        columns[4] = Col::I64("x", vec![10, 10]);
        columns[14] = Col::Str("execution_profile", vec!["full", "decoder_bounded"]);
        let (_root, table) = table(&columns);
        assert!(
            lookup(&table, 10)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
    }

    #[test]
    fn runtime_source_config_and_graph_identity_cannot_mix() {
        for (index, replacement) in [
            (8, Col::Str("source_sha256", vec![SHA, OTHER_SHA])),
            (9, Col::Str("config_sha256", vec![SHA, OTHER_SHA])),
            (
                10,
                Col::Str(
                    "runtime_digest",
                    vec![
                        DIGEST,
                        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    ],
                ),
            ),
            (11, Col::Bool("used_cuda_graph", vec![false, true])),
        ] {
            let mut columns = fixture();
            columns[index] = replacement;
            let (_root, table) = table(&columns);
            assert!(
                lookup(&table, 10)
                    .unwrap_err()
                    .to_string()
                    .contains("mixes")
            );
        }
    }

    #[test]
    fn invalid_contract_rows_fail_before_an_unrelated_query() {
        for (index, replacement) in [
            (0, Col::Str("component", vec!["legacy_mhc"; 2])),
            (
                1,
                Col::Str(
                    "geometry",
                    vec![r#"{"k":32,"n":64,"quant_mode":"fp8_block","unknown":1}"#; 2],
                ),
            ),
            (2, Col::I64("batch_size", vec![0; 2])),
            (3, Col::I64("prefix", vec![1; 2])),
            (4, Col::I64("x", vec![i64::from(u32::MAX) + 1; 2])),
            (5, Col::F64("latency", vec![f64::NAN; 2])),
            (6, Col::Str("kernel_source", vec![""; 2])),
            (
                7,
                Col::Str("measurement_scope", vec!["includes_all_reduce"; 2]),
            ),
            (8, Col::Str("source_sha256", vec!["not-a-hash"; 2])),
            (10, Col::Str("runtime_digest", vec!["mutable:latest"; 2])),
            (11, Col::I64("used_cuda_graph", vec![0; 2])),
            (12, Col::I64("sample_count", vec![0; 2])),
            (13, Col::Str("kv_seed_regime", vec!["fake_kv"; 2])),
            (
                14,
                Col::Str("execution_profile", vec!["prefix_swa_replay"; 2]),
            ),
        ] {
            let mut columns = fixture();
            columns[index] = replacement;
            let (_root, table) = table(&columns);
            assert!(
                matches!(lookup(&table, 10), Err(AicError::InvalidPerfData(_))),
                "column {index}"
            );
        }
    }

    fn attention_fixture(op: &Dsv41AttentionOp, prefix: i64, regime: &'static str) -> Vec<Col> {
        let mut columns = fixture();
        // The common parquet fixture accepts static strings; these tiny test
        // geometries live for the test process only.
        let encoded: &'static str = Box::leak(geometry(op).unwrap().into_boxed_str());
        columns[0] = Col::Str("component", vec!["attention"; 2]);
        columns[1] = Col::Str("geometry", vec![encoded; 2]);
        columns[3] = Col::I64("prefix", vec![prefix; 2]);
        columns[13] = Col::Str("kv_seed_regime", vec![regime; 2]);
        columns
    }

    #[test]
    fn decode_and_cached_prefill_require_real_kv() {
        for (is_context, prefix) in [(false, 0), (true, 1024)] {
            let mut op = attention();
            op.is_context = is_context;
            let (_root, table) = table(&attention_fixture(&op, prefix, "n/a"));
            assert!(
                lookup(&table, 10)
                    .unwrap_err()
                    .to_string()
                    .contains("real KV")
            );
            let (_root, table) = self::table(&attention_fixture(&op, prefix, "real_kv"));
            assert!(
                table
                    .query("attention", &op, 1, prefix as u32, 10, &|x| Ok(x))
                    .unwrap()
                    .is_some()
            );
            assert!(
                table
                    .query("attention", &op, 1, prefix as u32 + 1, 10, &|x| Ok(x))
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn bounded_path_cannot_read_full_attention_measurements() {
        let mut op = attention();
        op.is_context = true;
        let (_root, table) = table(&attention_fixture(&op, 2048, "real_kv"));
        op.bounded_prefill = true;
        assert!(
            table
                .query("attention", &op, 1, 2048, 10, &|x| Ok(x))
                .unwrap()
                .is_none()
        );
        let (_root, table) = self::table(&attention_fixture(&op, 2048, "real_kv"));
        assert!(
            lookup(&table, 10)
                .unwrap_err()
                .to_string()
                .contains("bounded prefill")
        );
    }

    #[test]
    fn early_attention_geometry_allows_no_candidate_list() {
        // The production graph has no candidate-list limit through layer 20.
        // Layers 0/1 additionally use uncompressed SWA; full sources use 2.
        for (role, ratio) in [("swa", 0), ("full", 2), ("reuse", 2), ("reindex", 2)] {
            let mut op = attention();
            op.role = role.into();
            op.compress_ratio = ratio;
            op.candidate_limit = 0;
            let (_root, table) = table(&attention_fixture(&op, 0, "real_kv"));
            assert!(
                table
                    .query("attention", &op, 1, 0, 10, &|x| Ok(x))
                    .unwrap()
                    .is_some()
            );
        }
    }

    #[test]
    fn sol_errors_survive_the_curve_callback() {
        let (_root, table) = table(&fixture());
        let err = table
            .query("linear", &linear(), 1, 0, 40, &|_| {
                Err(AicError::MissingSystemFlops("fixture".into()))
            })
            .unwrap_err();
        assert!(matches!(err, AicError::MissingSystemFlops(_)));
    }

    #[test]
    fn native_modes_preserve_sources_and_do_not_hide_invalid_data() {
        let root = tempfile::tempdir().unwrap();
        let data = write_energy_systems_root(root.path());
        let make_db = |mode| {
            let mut db = PerfDatabase::load(root.path(), "testsys", "vllm", "1.0").unwrap();
            db.database_mode = mode;
            db
        };
        assert_eq!(
            linear()
                .query(&make_db(DatabaseMode::Hybrid), 10)
                .unwrap()
                .source,
            Source::Sol
        );
        assert!(matches!(
            linear().query(&make_db(DatabaseMode::Silicon), 10),
            Err(AicError::PerfDatabase(_))
        ));
        write_parquet(&data.join(BASENAME), &fixture());
        for mode in [DatabaseMode::Silicon, DatabaseMode::Hybrid] {
            let result = linear().query(&make_db(mode), 15).unwrap();
            assert_eq!(result.source, Source::Silicon);
            assert_eq!(result.latency_ms, 2.0);
        }
        std::fs::write(data.join(BASENAME), b"malformed").unwrap();
        assert!(matches!(
            linear().query(&make_db(DatabaseMode::Hybrid), 10),
            Err(AicError::InvalidPerfData(_))
        ));
        assert_eq!(
            linear()
                .query(&make_db(DatabaseMode::Sol), 10)
                .unwrap()
                .source,
            Source::Sol
        );
    }
}
