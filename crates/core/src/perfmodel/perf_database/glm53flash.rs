// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Measured GLM-5.3-Flash native operation queries.
//! Bounded interpolation requires actual CUDA dispatch equivalence and complete
//! measured corners. No extrapolation, cross-backend reuse or SOL fallback.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use super::SourceResolver;
use super::parquet_loader::PerfReader;
use super::perf_interp::LeafValue;
use crate::common::error::AicError;
use crate::operators::glm53flash::{
    Glm53AttentionOp, Glm53FfnOp, Glm53MhcOp, Glm53PrimitiveOp, Glm53RouterOp,
};

const BASENAME: &str = "glm53flash_module_perf.parquet";

pub struct Glm53Table {
    path: Option<PathBuf>,
    request: Option<(String, String)>,
    points: OnceLock<Result<Points, String>>,
}

type Points = BTreeMap<Key, Measurement>;

struct Measurement {
    value: LeafValue,
    dispatch: String,
    source: String,
    state: String,
    graph: bool,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    component: String,
    geometry: String,
    batch_size: u32,
    prefix: u32,
    x: u32,
}

fn invalid(message: impl Into<String>) -> AicError {
    AicError::InvalidPerfData(message.into())
}

pub fn geometry<T: Serialize>(op: &T) -> Result<String, AicError> {
    let mut value = serde_json::to_value(op).map_err(|e| invalid(e.to_string()))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| invalid("GLM53 geometry must be an object"))?;
    object.remove("name");
    object.remove("children");
    serde_json::to_string(&object.iter().collect::<BTreeMap<_, _>>())
        .map_err(|e| invalid(e.to_string()))
}

fn validate_body<T: DeserializeOwned + Serialize>(value: &Value) -> Result<T, AicError> {
    let mut named = value.clone();
    named
        .as_object_mut()
        .ok_or_else(|| invalid("GLM53 geometry must be an object"))?
        .insert("name".into(), Value::String(String::new()));
    let op: T = serde_json::from_value(named).map_err(|e| invalid(e.to_string()))?;
    let round_trip: Value =
        serde_json::from_str(&geometry(&op)?).map_err(|e| invalid(e.to_string()))?;
    if &round_trip != value {
        return Err(invalid("GLM53 geometry has unknown or noncanonical fields"));
    }
    Ok(op)
}

fn validate_geometry(component: &str, encoded: &str) -> Result<Value, AicError> {
    let value: Value = serde_json::from_str(encoded).map_err(|e| invalid(e.to_string()))?;
    match component {
        "attention" => validate_body::<Glm53AttentionOp>(&value)?.validate()?,
        "mhc" => validate_body::<Glm53MhcOp>(&value)?.validate()?,
        "ffn" => validate_body::<Glm53FfnOp>(&value)?.validate_physical()?,
        "primitive" => validate_body::<Glm53PrimitiveOp>(&value)?.validate_physical()?,
        "router" => validate_body::<Glm53RouterOp>(&value)?.validate()?,
        _ => return Err(invalid("unknown GLM53 component")),
    }
    let sorted: BTreeMap<_, _> = value.as_object().expect("typed object").iter().collect();
    if serde_json::to_string(&sorted).map_err(|e| invalid(e.to_string()))? != encoded {
        return Err(invalid("GLM53 geometry must be canonical JSON"));
    }
    Ok(value)
}

fn sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl Glm53Table {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            path: Some(data_root.join(BASENAME)),
            request: None,
            points: OnceLock::new(),
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
                "GLM53 primary kernel_sources filters are not supported; select an unfiltered homogeneous module table",
            ));
        }
        let path = primary.map(|source| source.source.0);
        if let Some(path) = &path {
            let system_root = data_root
                .parent()
                .and_then(Path::parent)
                .ok_or_else(|| invalid("GLM53 data root must include system/backend/version"))?;
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
                    "GLM53 primary source must belong to the requested system, backend and version",
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
                        "GLM53 primary source resolves outside the requested system, backend and version",
                    ));
                }
            }
        }
        let version = data_root
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| invalid("missing GLM53 requested version"))?;
        let backend = data_root
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .ok_or_else(|| invalid("missing GLM53 requested backend"))?;
        Ok(Self {
            path,
            request: Some((backend.into(), version.into())),
            points: OnceLock::new(),
        })
    }

    pub fn has_measurements(&self) -> Result<bool, AicError> {
        let points = self.points.get_or_init(|| match &self.path {
            Some(path) => {
                load(path, self.request.as_ref()).map_err(|e| format!("{}: {e}", path.display()))
            }
            None => Ok(Points::new()),
        });
        Ok(!points.as_ref().map_err(|e| invalid(e.clone()))?.is_empty())
    }

    pub fn query<T: Serialize>(
        &self,
        component: &str,
        op: &T,
        batch_size: u32,
        prefix: u32,
        x: u32,
    ) -> Result<Option<LeafValue>, AicError> {
        let points = self.points.get_or_init(|| match &self.path {
            Some(path) => {
                load(path, self.request.as_ref()).map_err(|e| format!("{}: {e}", path.display()))
            }
            None => Ok(Points::new()),
        });
        let points = points.as_ref().map_err(|e| invalid(e.clone()))?;
        let key = Key {
            component: component.into(),
            geometry: geometry(op)?,
            batch_size,
            prefix,
            x,
        };
        if let Some(measured) = points.get(&key) {
            return Ok(Some(measured.value));
        }
        interpolate(points, &key)
    }
}

// Structural partitions supplement (never replace) observed CUDA kernel names.
// In particular, selected pool/tail state and the short sparse path must not be
// smoothed across their discontinuities even if kernel symbol names coincide.
fn partition(key: &Key, shape: &Value) -> Vec<u64> {
    if key.component != "attention" {
        return vec![];
    }
    let context = shape["is_context"].as_bool().unwrap_or(false);
    if shape["layer_kind"] == "kda" {
        return if context {
            vec![u64::from(key.x).div_ceil(128), u64::from(key.prefix == 0)]
        } else {
            vec![]
        };
    }
    let pool = shape["index_pool"].as_u64().unwrap_or(4).max(1);
    let query = if context { u64::from(key.x) } else { 1 };
    let past = if context {
        u64::from(key.prefix)
    } else {
        u64::from(key.x)
    };
    let total = past + query;
    let topk = shape["index_topk"].as_u64().unwrap_or(2048);
    vec![
        u64::from(total <= topk),
        past % pool,
        total % pool,
        u64::from(past == 0),
    ]
}

fn interpolate(points: &Points, target: &Key) -> Result<Option<LeafValue>, AicError> {
    let shape: Value =
        serde_json::from_str(&target.geometry).map_err(|e| invalid(e.to_string()))?;
    let wanted = partition(target, &shape);
    let mut groups: BTreeMap<(&str, &str, &str, bool), Vec<(&Key, &Measurement)>> = BTreeMap::new();
    for (key, measurement) in points {
        if key.component == target.component
            && key.geometry == target.geometry
            && !measurement.dispatch.is_empty()
            && partition(key, &shape) == wanted
        {
            groups
                .entry((
                    &measurement.dispatch,
                    &measurement.source,
                    &measurement.state,
                    measurement.graph,
                ))
                .or_default()
                .push((key, measurement));
        }
    }
    let coordinates = |key: &Key| [key.batch_size, key.prefix, key.x];
    let target_coords = coordinates(target);
    let mut resolved = None;
    for sites in groups.values() {
        let mut bounds = [(0_u32, 0_u32); 3];
        let mut supported = true;
        for axis in 0..3 {
            let lower = sites
                .iter()
                .map(|(key, _)| coordinates(key)[axis])
                .filter(|&v| v <= target_coords[axis])
                .max();
            let upper = sites
                .iter()
                .map(|(key, _)| coordinates(key)[axis])
                .filter(|&v| v >= target_coords[axis])
                .min();
            match (lower, upper) {
                (Some(lo), Some(hi)) => bounds[axis] = (lo, hi),
                _ => {
                    supported = false;
                    break;
                }
            }
        }
        if !supported {
            continue;
        }
        let mut corners = vec![([0_u32; 3], 1.0_f64)];
        for axis in 0..3 {
            let (lo, hi) = bounds[axis];
            let mut next = Vec::new();
            for (mut point, weight) in corners {
                point[axis] = lo;
                if lo == hi {
                    next.push((point, weight));
                } else {
                    let t = f64::from(target_coords[axis] - lo) / f64::from(hi - lo);
                    next.push((point, weight * (1.0 - t)));
                    point[axis] = hi;
                    next.push((point, weight * t));
                }
            }
            corners = next;
        }
        let mut latency = 0.0;
        for (point, weight) in corners {
            match sites.iter().find(|(key, _)| coordinates(key) == point) {
                Some((_, measured)) => latency += weight * measured.value.latency,
                None => {
                    supported = false;
                    break;
                }
            }
        }
        if supported {
            // Competing state/graph/dispatch baselines are not averaged.
            if resolved.is_some() {
                return Ok(None);
            }
            resolved = Some(LeafValue::latency_only(latency));
        }
    }
    Ok(resolved)
}

fn load(path: &Path, request: Option<&(String, String)>) -> Result<Points, AicError> {
    if !path.try_exists().map_err(|e| invalid(e.to_string()))? {
        return Ok(Points::new());
    }
    let reader = PerfReader::open(path)?;
    let component = reader.col("component")?;
    let geometry = reader.col("geometry")?;
    let batch_size = reader.col("batch_size")?;
    let prefix = reader.col("prefix")?;
    let x = reader.col("x")?;
    let latency = reader.col("latency")?;
    let kernel_source = reader.col("kernel_source")?;
    let dispatch_fingerprint = reader.col("dispatch_fingerprint")?;
    let measurement_scope = reader.col("measurement_scope")?;
    let source_sha256 = reader.col("source_sha256")?;
    let config_sha256 = reader.col("config_sha256")?;
    let runtime_digest = reader.col("runtime_digest")?;
    let used_cuda_graph = reader.col("used_cuda_graph")?;
    let sample_count = reader.col("sample_count")?;
    let kv_seed_regime = reader.col("kv_seed_regime")?;
    let state_mode = reader.col("state_mode")?;
    let backend = reader.col("backend")?;
    let backend_version = reader.col("backend_version")?;
    let backend_revision = reader.col("backend_revision")?;
    let checkpoint_revision = reader.col("checkpoint_revision")?;
    let dataset_role = reader.col("dataset_role")?;
    let request_set = reader.col("request_set")?;
    let corpus_sha256 = reader.col("corpus_sha256")?;
    let evidence_sha256 = reader.col("evidence_sha256")?;
    let mut identities = BTreeMap::new();
    let mut points = Points::new();
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
            || row.u32(sample_count)? < 10
        {
            return Err(invalid(
                "GLM53 measurement needs positive work, sample count and finite latency",
            ));
        }
        let expected_scope = match (component, shape["role"].as_str()) {
            ("primitive", Some("allreduce")) => "communication",
            ("primitive", Some("logits")) => "compute_and_communication",
            _ => "local_compute",
        };
        if row.str(kernel_source)?.trim().is_empty()
            || row.str(measurement_scope)? != expected_scope
        {
            return Err(invalid(
                "GLM53 measurement needs observed native local-compute dispatch",
            ));
        }
        let graph = row.bool_strict(used_cuda_graph)?;
        let dispatch = row.str(dispatch_fingerprint)?;
        if !dispatch.is_empty() && !sha256(dispatch) {
            return Err(invalid("GLM53 native CUDA dispatch fingerprint is invalid"));
        }
        let format = shape["checkpoint_format"].as_str().unwrap_or_default();
        let expected_checkpoint = match format {
            "fp8" => "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
            "nvfp4" => "09b04e5e74bca08ca8549fc736d4cdd8624bfde3",
            _ => return Err(invalid("GLM53 checkpoint format is unqualified")),
        };
        if row.str(checkpoint_revision)? != expected_checkpoint {
            return Err(invalid("GLM53 checkpoint revision is unqualified"));
        }
        let runtime_backend = row.str(backend)?;
        if request.is_some_and(|(backend, version)| {
            backend != runtime_backend || version != row.str(backend_version).unwrap_or_default()
        }) {
            return Err(invalid(
                "GLM53 row provenance does not match the requested backend/version directory",
            ));
        }
        let expected_runtime = match runtime_backend {
            "vllm" => ("0.30.0", "ced6857afa0ea7b2e3f0846a62e1394e90f15607"),
            "sglang" => ("0.5.20", "94602c9c2b7cbdb8efd5c52802dac6a1c180089e"),
            _ => return Err(invalid("GLM53 backend is unqualified")),
        };
        if (row.str(backend_version)?, row.str(backend_revision)?) != expected_runtime
            || shape["backend"].as_str() != Some(runtime_backend)
        {
            return Err(invalid("GLM53 geometry/runtime backend revision mismatch"));
        }
        let regime = row.str(kv_seed_regime)?;
        let mode = row.str(state_mode)?;
        if component == "attention" {
            let context = shape["is_context"]
                .as_bool()
                .ok_or_else(|| invalid("missing GLM53 phase"))?;
            let tp = shape["tp_size"].as_u64().unwrap_or_default();
            if !matches!(tp, 1 | 2 | 4) || (tp == 1 && format != "nvfp4") {
                return Err(invalid("unqualified GLM53 TP/checkpoint combination"));
            }
            if context {
                if u64::from(prefix) + u64::from(x) > 131072
                    || (prefix == 0 && mode != "full_prefill")
                    || (prefix > 0 && !matches!(mode, "cached_prefill" | "chunked_prefill"))
                {
                    return Err(invalid("GLM53 prefill state/128K coordinates disagree"));
                }
            } else if prefix != 0 || mode != "decode" || x > 131071 {
                return Err(invalid(
                    "GLM53 decode needs absolute past-KV x and prefix=0",
                ));
            }
            let expected_regime = if prefix > 0 || !context {
                "real_kv"
            } else {
                "empty"
            };
            if regime != expected_regime {
                return Err(invalid(
                    "GLM53 cached/decode measurement needs real native prefix state",
                ));
            }
        } else if batch != 1 || prefix != 0 || regime != "n/a" || mode != "token_only" {
            return Err(invalid(
                "GLM53 token-only component has stateful coordinates",
            ));
        }
        if row.str(dataset_role)? != "calibration"
            || row.str(request_set)?.is_empty()
            || !sha256(row.str(corpus_sha256)?)
            || !sha256(row.str(evidence_sha256)?)
        {
            return Err(invalid(
                "GLM53 table needs content-addressed calibration evidence",
            ));
        }
        let source = row.str(source_sha256)?;
        let config = row.str(config_sha256)?;
        let digest = row.str(runtime_digest)?;
        if !sha256(source) || !sha256(config) || !digest.strip_prefix("sha256:").is_some_and(sha256)
        {
            return Err(invalid(
                "GLM53 measurement needs complete SHA256 provenance",
            ));
        }
        let identity = (
            runtime_backend.to_owned(),
            expected_runtime,
            source.to_owned(),
            config.to_owned(),
            digest.to_owned(),
        );
        if let Some(previous) = identities.insert(format.to_owned(), identity.clone()) {
            if previous != identity {
                return Err(invalid(
                    "GLM53 table mixes immutable provenance within checkpoint format",
                ));
            }
        }
        let key = Key {
            component: component.into(),
            geometry: encoded.into(),
            batch_size: batch,
            prefix,
            x,
        };
        if points
            .insert(
                key,
                Measurement {
                    value: LeafValue::with_power(latency, 0.0),
                    dispatch: dispatch.into(),
                    source: row.str(kernel_source)?.into(),
                    state: mode.into(),
                    graph,
                },
            )
            .is_some()
        {
            return Err(invalid("duplicate GLM53 measured physical key"));
        }
    }
    if points.is_empty() {
        return Err(invalid("present GLM53 measured table is empty"));
    }
    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::enums::DatabaseMode;
    use crate::config::PerfDbSources;
    use crate::operators::base::Source;
    use crate::perf_database::PerfDatabase;
    use crate::perf_database::energy_test_fixtures::{
        Col, write_energy_systems_root, write_parquet,
    };

    const SHAPE: &str = r#"{"backend":"vllm","checkpoint_format":"fp8","hc_mult":4,"hidden_size":4096,"is_context":true,"role":"pre","sinkhorn_iters":20,"tp_size":2}"#;
    const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn op() -> Glm53MhcOp {
        Glm53MhcOp {
            name: "mhc_pre_attn_0".into(),
            is_context: true,
            role: "pre".into(),
            tp_size: 2,
            backend: "vllm".into(),
            checkpoint_format: "fp8".into(),
            hidden_size: 4096,
            hc_mult: 4,
            sinkhorn_iters: 20,
        }
    }

    fn fixture() -> Vec<Col> {
        vec![
            Col::Str("component", vec!["mhc"; 2]),
            Col::Str("geometry", vec![SHAPE; 2]),
            Col::I64("batch_size", vec![1; 2]),
            Col::I64("prefix", vec![0; 2]),
            Col::I64("x", vec![128, 256]),
            Col::F64("latency", vec![3.0, 7.0]),
            Col::Str("kernel_source", vec!["fixture.native.mhc_pre"; 2]),
            Col::Str("dispatch_fingerprint", vec![""; 2]),
            Col::Str("measurement_scope", vec!["local_compute"; 2]),
            Col::Str("source_sha256", vec![SHA; 2]),
            Col::Str("dataset_role", vec!["calibration"; 2]),
            Col::Str(
                "request_set",
                vec!["authored-unit-fixture-not-performance"; 2],
            ),
            Col::Str("corpus_sha256", vec![SHA; 2]),
            Col::Str("evidence_sha256", vec![SHA; 2]),
            Col::Str("config_sha256", vec![SHA; 2]),
            Col::Str("runtime_digest", vec![DIGEST; 2]),
            Col::Bool("used_cuda_graph", vec![false; 2]),
            Col::I64("sample_count", vec![10; 2]),
            Col::Str("kv_seed_regime", vec!["n/a"; 2]),
            Col::Str("state_mode", vec!["token_only"; 2]),
            Col::Str("backend", vec!["vllm"; 2]),
            Col::Str("backend_version", vec!["0.30.0"; 2]),
            Col::Str(
                "backend_revision",
                vec!["ced6857afa0ea7b2e3f0846a62e1394e90f15607"; 2],
            ),
            Col::Str(
                "checkpoint_revision",
                vec!["eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"; 2],
            ),
        ]
    }

    fn measured(latency: f64, dispatch: &str) -> Measurement {
        Measurement {
            value: LeafValue::latency_only(latency),
            dispatch: dispatch.into(),
            source: "authored-fixture".into(),
            state: "cached_prefill".into(),
            graph: false,
        }
    }

    #[test]
    fn bounded_interpolation_requires_kernel_evidence_and_complete_corners() {
        let mut attention = crate::operators::glm53flash::tests::attention("kda");
        attention.is_context = true;
        let shape = geometry(&attention).unwrap();
        let key = |batch, prefix, x| Key {
            component: "attention".into(),
            geometry: shape.clone(),
            batch_size: batch,
            prefix,
            x,
        };
        let mut points = Points::new();
        for batch in [1, 4] {
            for prefix in [256, 512] {
                for x in [64, 128] {
                    points.insert(
                        key(batch, prefix, x),
                        measured(f64::from(batch + prefix + x), SHA),
                    );
                }
            }
        }
        let target = key(2, 384, 96);
        assert!((interpolate(&points, &target).unwrap().unwrap().latency - 482.0).abs() < 1e-10);
        assert!(interpolate(&points, &key(5, 384, 96)).unwrap().is_none());
        points.get_mut(&key(4, 512, 128)).unwrap().dispatch.clear();
        assert!(interpolate(&points, &target).unwrap().is_none());
        points.get_mut(&key(4, 512, 128)).unwrap().dispatch = SHA.into();
        points.get_mut(&key(4, 512, 128)).unwrap().graph = true;
        assert!(interpolate(&points, &target).unwrap().is_none());
    }

    #[test]
    fn interpolation_preserves_index_pool_tail_and_short_sparse_path() {
        let mut attention = crate::operators::glm53flash::tests::attention("sparse_mla");
        attention.is_context = false;
        let shape = geometry(&attention).unwrap();
        let key = |x| Key {
            component: "attention".into(),
            geometry: shape.clone(),
            batch_size: 1,
            prefix: 0,
            x,
        };
        let mut points = Points::new();
        points.insert(key(128), measured(3.0, SHA));
        points.insert(key(256), measured(7.0, SHA));
        assert_eq!(
            interpolate(&points, &key(192)).unwrap().unwrap().latency,
            5.0
        );
        assert!(interpolate(&points, &key(193)).unwrap().is_none());
        points.clear();
        points.insert(key(2044), measured(3.0, SHA));
        points.insert(key(2052), measured(7.0, SHA));
        assert!(interpolate(&points, &key(2048)).unwrap().is_none());
    }

    #[test]
    fn exact_measurements_never_interpolate_or_extrapolate() {
        let root = tempfile::tempdir().unwrap();
        write_parquet(&root.path().join(BASENAME), &fixture());
        let table = Glm53Table::new(root.path().into());
        // Authored fixture values are 3 ms at128 tokens and7 ms at256 tokens.
        assert_eq!(
            table
                .query("mhc", &op(), 1, 0, 128)
                .unwrap()
                .unwrap()
                .latency,
            3.0
        );
        for missing in [64, 192, 512] {
            assert!(table.query("mhc", &op(), 1, 0, missing).unwrap().is_none());
        }
        let mut renamed = op();
        renamed.name = "mhc_pre_attn_20".into();
        assert_eq!(geometry(&renamed).unwrap(), SHAPE);
        renamed.checkpoint_format = "nvfp4".into();
        assert!(table.query("mhc", &renamed, 1, 0, 128).unwrap().is_none());
    }

    #[test]
    fn malformed_or_duplicate_rows_are_errors_not_coverage_gaps() {
        for bad in [
            Col::F64("latency", vec![f64::NAN, 7.0]),
            Col::I64("x", vec![128, 128]),
            Col::Str("checkpoint_revision", vec!["main"; 2]),
            Col::Str("kv_seed_regime", vec!["fake_kv"; 2]),
            Col::Str("backend_version", vec!["0.27.0"; 2]),
        ] {
            let mut columns = fixture();
            let index = match &bad {
                Col::F64("latency", _) => 5,
                Col::I64("x", _) => 4,
                Col::Str("checkpoint_revision", _) => 18,
                Col::Str("kv_seed_regime", _) => 13,
                Col::Str("backend_version", _) => 16,
                _ => unreachable!(),
            };
            columns[index] = bad;
            let root = tempfile::tempdir().unwrap();
            write_parquet(&root.path().join(BASENAME), &columns);
            assert!(
                Glm53Table::new(root.path().into())
                    .has_measurements()
                    .is_err()
            );
        }
    }

    #[test]
    fn requested_version_and_explicit_source_veto_are_honored() {
        let root = tempfile::tempdir().unwrap();
        let data = write_energy_systems_root(root.path());
        write_parquet(&data.join(BASENAME), &fixture());
        let resolver = SourceResolver::fixed(PerfDbSources::default());
        // The fixture declares0.30.0, but the requested path isvllm/1.0.
        assert!(
            Glm53Table::with_sources(&data, &resolver)
                .unwrap()
                .has_measurements()
                .is_err()
        );
        let resolver = SourceResolver::fixed(BTreeMap::from([(BASENAME.into(), vec![])]));
        assert!(
            !Glm53Table::with_sources(&data, &resolver)
                .unwrap()
                .has_measurements()
                .unwrap()
        );
    }

    #[test]
    fn public_op_queries_are_strict_in_silicon_and_hybrid() {
        let root = tempfile::tempdir().unwrap();
        let _ = write_energy_systems_root(root.path());
        let data = root.path().join("data/vllm/0.30.0");
        std::fs::create_dir_all(&data).unwrap();
        write_parquet(&data.join(BASENAME), &fixture());
        for mode in [DatabaseMode::Silicon, DatabaseMode::Hybrid] {
            let mut db = PerfDatabase::load(root.path(), "testsys", "vllm", "0.30.0").unwrap();
            db.database_mode = mode;
            let result = op().query(&db, 128).unwrap();
            assert_eq!(result.source, Source::Silicon);
            assert_eq!(result.latency_ms, 3.0);
            assert!(matches!(
                op().query(&db, 192),
                Err(AicError::PerfDatabase(_))
            ));
        }
    }
}
