// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema 3 native vLLM serving units. Original implementation of the frozen
//! serving contract; source dispatch follows the pinned native policy snapshot.
//! This reader never grants runtime admission or borrows eager/FULL-only rows.

use super::glm53flash::{
    geometry, sha256, validate_geometry, validate_native_workload, validate_runtime,
};
use super::glm53flash_graph::{VLLM_REVISION, VLLM_STOCK_SOURCE, vllm_flags, vllm_pins};
use super::parquet_loader::PerfReader;
use crate::common::error::AicError;
use crate::operators::op::RuntimeContext;
use crate::operators::{Op, PerformanceResult, Source};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

fn invalid(message: impl Into<String>) -> AicError {
    AicError::InvalidPerfData(message.into())
}
fn canonical<T: Serialize>(value: &T) -> Result<String, AicError> {
    serde_json::to_string(&serde_json::to_value(value).map_err(|e| invalid(e.to_string()))?)
        .map_err(|e| invalid(e.to_string()))
}
fn digest(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    cg_mode: String,
    num_tokens: u32,
    num_reqs: Option<u32>,
    uniform_token_count: Option<u32>,
    max_query_len: Option<u32>,
    num_active_loras: u32,
    num_ubatches: u32,
}
impl Descriptor {
    fn captured(mode: &str, tokens: u32) -> Self {
        Self {
            cg_mode: mode.into(),
            num_tokens: tokens,
            num_reqs: (mode == "FULL").then_some(tokens),
            uniform_token_count: (mode == "FULL").then_some(1),
            max_query_len: None,
            num_active_loras: 0,
            num_ubatches: 1,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Candidate {
    num_tokens: u32,
    num_active_loras: u32,
    descriptors: Vec<Descriptor>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PiecewiseEntry {
    num_tokens: u32,
    num_reqs: Option<u32>,
    uniform: bool,
    has_lora: bool,
    num_active_loras: u32,
    completed: bool,
    num_graphs: u32,
    num_eager_breaks: u32,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativePolicy {
    backend: String,
    backend_version: String,
    backend_revision: String,
    source_pins: BTreeMap<String, String>,
    native_flags: BTreeMap<String, bool>,
    capture_sizes: Vec<u32>,
    max_num_reqs: u32,
    max_capture_tokens: u32,
    decode_query_len: u32,
    graphs_captured: bool,
    lora_capture_cases: Vec<u32>,
    dp_size: u32,
    tp_size: u32,
    resolved_mode: String,
    use_breakable_cg: bool,
    capture_descriptors: BTreeMap<String, Vec<Descriptor>>,
    full_graphs: Vec<Descriptor>,
    candidates: Vec<Candidate>,
    piecewise_entries: Vec<PiecewiseEntry>,
}
impl NativePolicy {
    fn validate(&self) -> Result<(), AicError> {
        if self.backend != "vllm"
            || self.backend_revision != VLLM_REVISION
            || self.source_pins != vllm_pins()
            || self.native_flags != vllm_flags()
            || self.capture_sizes.is_empty()
            || self.capture_sizes[0] == 0
            || self.capture_sizes.windows(2).any(|p| p[0] >= p[1])
            || self.max_capture_tokens != *self.capture_sizes.last().unwrap()
            || self.max_num_reqs == 0
            || self.decode_query_len != 1
            || !self.graphs_captured
            || self.lora_capture_cases != [0]
            || self.dp_size != 1
            || !matches!(self.tp_size, 2 | 4)
            || !matches!(
                self.resolved_mode.as_str(),
                "FULL_AND_PIECEWISE" | "FULL_DECODE_ONLY"
            )
            || self.use_breakable_cg != (self.resolved_mode == "FULL_AND_PIECEWISE")
        {
            return Err(invalid(
                "serving policy differs from the initialized native V2 dispatch contract",
            ));
        }
        validate_runtime(&self.backend, &self.backend_version)?;
        let full: Vec<_> = self
            .capture_sizes
            .iter()
            .copied()
            .filter(|&n| n <= self.max_num_reqs)
            .collect();
        if full.is_empty() {
            return Err(invalid("native serving policy lacks FULL descriptors"));
        }
        let full_graphs: Vec<_> = full
            .iter()
            .map(|&n| Descriptor::captured("FULL", n))
            .collect();
        let mut descriptors =
            BTreeMap::from([("FULL".into(), full_graphs.iter().rev().cloned().collect())]);
        if self.use_breakable_cg {
            descriptors.insert(
                "PIECEWISE".into(),
                self.capture_sizes
                    .iter()
                    .rev()
                    .map(|&n| Descriptor::captured("PIECEWISE", n))
                    .collect(),
            );
        }
        if self.full_graphs != full_graphs || self.capture_descriptors != descriptors {
            return Err(invalid(
                "native serving descriptor inventory differs from configured captures",
            ));
        }
        // The actual native manager serializes every candidate token count,
        // including zero. Bound iteration by the supplied finite list length.
        let candidate_max = if self.use_breakable_cg {
            self.max_capture_tokens
        } else {
            *full.last().unwrap()
        };
        if self.candidates.len() != candidate_max as usize + 1 {
            return Err(invalid("native serving candidate coverage is incomplete"));
        }
        for (tokens, candidate) in self.candidates.iter().enumerate() {
            let mut expected = Vec::new();
            if let Some(&size) = full.iter().find(|&&n| n >= tokens as u32) {
                expected.push(Descriptor::captured("FULL", size));
            }
            if self.use_breakable_cg
                && let Some(&size) = self.capture_sizes.iter().find(|&&n| n >= tokens as u32)
            {
                expected.push(Descriptor::captured("PIECEWISE", size));
            }
            // FULL_DECODE_ONLY can have configured buckets above max requests,
            // where no descriptor exists; those counts are absent natively.
            if candidate.num_tokens != tokens as u32
                || candidate.num_active_loras != 0
                || candidate.descriptors != expected
                || expected.is_empty()
            {
                return Err(invalid("native serving candidate priority differs"));
            }
        }
        let pw = if self.use_breakable_cg {
            self.capture_sizes.as_slice()
        } else {
            &[]
        };
        if self.piecewise_entries.len() != pw.len()
            || self.piecewise_entries.iter().zip(pw).any(|(entry, &size)| {
                entry.num_tokens != size
                    || entry.num_reqs.is_some()
                    || entry.uniform
                    || entry.has_lora
                    || entry.num_active_loras != 0
                    || !entry.completed
                    || entry.num_graphs == 0
            })
        {
            return Err(invalid(
                "native serving PIECEWISE entry inventory is incomplete",
            ));
        }
        Ok(())
    }
    fn select(
        &self,
        context: bool,
        batch: u32,
        query: u32,
    ) -> Result<(String, u32, u32), AicError> {
        if batch == 0 || batch > self.max_num_reqs || query == 0 || (!context && query != 1) {
            return Err(invalid(
                "native serving requires homogeneous positive request geometry",
            ));
        }
        let tokens = batch
            .checked_mul(query)
            .ok_or_else(|| invalid("native serving token count overflow"))?;
        if !context && let Some(row) = self.full_graphs.iter().find(|d| d.num_tokens >= tokens) {
            return Ok(("FULL".into(), row.num_tokens, row.num_tokens));
        }
        if self.use_breakable_cg
            && let Some(&size) = self.capture_sizes.iter().find(|&&n| n >= tokens)
        {
            return Ok(("PIECEWISE".into(), size, batch));
        }
        Ok(("NONE".into(), tokens, batch))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServingPolicy {
    schema_version: u32,
    backend: String,
    backend_version: String,
    backend_revision: String,
    checkpoint_format: String,
    checkpoint_revision: String,
    config_sha256: String,
    execution_policy_sha256: String,
    native_policy: NativePolicy,
    native_policy_sha256: String,
    runtime_digest: String,
    source_pins: BTreeMap<String, String>,
    source_sha256: String,
    timing_boundary: String,
    tp_size: u32,
}
impl ServingPolicy {
    fn validate(&self) -> Result<(), AicError> {
        validate_runtime(&self.backend, &self.backend_version)?;
        self.native_policy.validate()?;
        let revision = match self.checkpoint_format.as_str() {
            "fp8" => "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
            "nvfp4" => "09b04e5e74bca08ca8549fc736d4cdd8624bfde3",
            _ => return Err(invalid("unknown serving checkpoint")),
        };
        if self.schema_version != 3
            || self.backend != "vllm"
            || self.backend_version != "0.30.0"
            || self.backend_revision != VLLM_REVISION
            || self.source_sha256 != VLLM_STOCK_SOURCE
            || self.source_pins != vllm_pins()
            || self.checkpoint_revision != revision
            || !matches!(self.tp_size, 2 | 4)
            || self.timing_boundary != "native_metadata_to_logits_gpu_v1"
            || self.native_policy.backend != self.backend
            || self.native_policy.backend_version != self.backend_version
            || self.native_policy.backend_revision != self.backend_revision
            || self.native_policy.tp_size != self.tp_size
            || self.native_policy.source_pins != self.source_pins
            || self.native_policy_sha256 != digest(&canonical(&self.native_policy)?)
            || !sha256(&self.config_sha256)
            || !sha256(&self.execution_policy_sha256)
            || !self
                .runtime_digest
                .strip_prefix("sha256:")
                .is_some_and(sha256)
        {
            return Err(invalid(
                "serving policy source/config/runtime identity differs",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Unit {
    component: String,
    name: String,
    geometry: String,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Point {
    phase: String,
    mode: String,
    batch: u32,
    query: u32,
    prefix: u32,
    tokens: u32,
    requests: u32,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    unit: Unit,
    point: Point,
}
struct Measurement {
    latency: f64,
    count: u32,
    method: String,
    dispatch: String,
}
struct Profile {
    policy: ServingPolicy,
    rows: BTreeMap<Key, Measurement>,
    units: BTreeMap<String, BTreeSet<Unit>>,
}
type Profiles = BTreeMap<(String, u32), Profile>;
pub(super) struct ServingTable {
    path: Option<PathBuf>,
    request: (String, String),
    profile: OnceLock<Result<Option<Profiles>, String>>,
}
impl ServingTable {
    pub(super) fn new(path: Option<PathBuf>, request: (String, String)) -> Self {
        Self {
            path,
            request,
            profile: OnceLock::new(),
        }
    }
    fn profiles(&self) -> Result<Option<&Profiles>, AicError> {
        self.profile
            .get_or_init(|| match &self.path {
                Some(path) => load(path, &self.request).map_err(|e| e.to_string()),
                None => Ok(None),
            })
            .as_ref()
            .map(Option::as_ref)
            .map_err(|e| invalid(e.clone()))
    }
    pub(super) fn has_measurements(&self) -> Result<bool, AicError> {
        Ok(self.profiles()?.is_some())
    }
    pub(super) fn identities(&self) -> Result<Vec<(String, u32)>, AicError> {
        Ok(self
            .profiles()?
            .map(|p| p.keys().cloned().collect())
            .unwrap_or_default())
    }
    pub(super) fn validate_ops(&self, ops: &[Op], context: bool) -> Result<bool, AicError> {
        let Some(profiles) = self.profiles()? else {
            return Ok(false);
        };
        if !ops.iter().any(super::glm53flash_graph::contains_glm) {
            return Ok(false);
        }
        let first = ops
            .iter()
            .find_map(|op| operation(op).transpose())
            .transpose()?;
        if let Some((_, shape)) = first {
            let id = (
                shape["checkpoint_format"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                shape["tp_size"].as_u64().unwrap_or_default() as u32,
            );
            if shape["backend"] != "vllm" || !profiles.contains_key(&id) {
                return Ok(false);
            }
        }
        let mut units = BTreeSet::new();
        let mut identity = None;
        for op in ops {
            let (unit, shape) = operation(op)?.ok_or_else(|| {
                invalid("serving phases reject nested/fallback/non-GLM operations")
            })?;
            if shape["is_context"] != context || shape["backend"] != "vllm" {
                return Err(invalid("serving operation phase/backend differs"));
            }
            let current = (
                shape["checkpoint_format"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                shape["tp_size"].as_u64().unwrap_or_default() as u32,
            );
            if identity.as_ref().is_some_and(|old| old != &current) || !units.insert(unit) {
                return Err(invalid(
                    "serving phase has duplicate or mismatched operation identity",
                ));
            }
            identity = Some(current);
        }
        let identity = identity.ok_or_else(|| invalid("empty serving phase"))?;
        let profile = profiles
            .get(&identity)
            .ok_or_else(|| invalid("serving table lacks compiled checkpoint/TP"))?;
        validate_units(&units)?;
        let phase = if context { "context" } else { "generation" };
        if profile
            .units
            .get(phase)
            .is_some_and(|actual| actual != &units)
        {
            return Err(invalid(
                "compiled serving phase differs from exact measured operation names/geometries",
            ));
        }
        Ok(true)
    }
    pub(super) fn query(
        &self,
        op: &Op,
        ctx: &RuntimeContext,
    ) -> Result<Option<PerformanceResult>, AicError> {
        let Some((unit, shape)) = operation(op)? else {
            return Ok(None);
        };
        let Some(profiles) = self.profiles()? else {
            return Ok(None);
        };
        let identity = (
            shape["checkpoint_format"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            shape["tp_size"].as_u64().unwrap_or_default() as u32,
        );
        let Some(profile) = profiles.get(&identity) else {
            return Ok(None);
        };
        if shape["backend"] != profile.policy.backend {
            return Err(invalid("serving backend differs"));
        }
        let context = shape["is_context"] == true;
        let query = if context { ctx.s } else { 1 };
        let prefix = if context {
            ctx.prefix
        } else {
            ctx.s
                .checked_sub(1)
                .ok_or_else(|| invalid("invalid inclusive decode position"))?
        };
        if ctx.beam_width != 1
            || ctx.seq_imbalance_correction_scale != 1.0
            || ctx.gen_seq_imbalance_correction_scale != 1.0
            || ctx.num_image_tokens != 0
            || (!context && ctx.prefix != 0)
            || ctx.batch_size.checked_mul(query) != Some(ctx.num_tokens)
        {
            return Err(invalid(
                "serving query requires complete homogeneous RuntimeContext",
            ));
        }
        validate_bounds(context, prefix, query)?;
        validate_native_workload(
            &unit.component,
            &shape,
            prefix,
            query,
            Some(&profile.policy.backend_version),
        )?;
        let (mode, tokens, requests) =
            profile
                .policy
                .native_policy
                .select(context, ctx.batch_size, query)?;
        let key = Key {
            unit,
            point: Point {
                phase: if context { "context" } else { "generation" }.into(),
                mode,
                batch: ctx.batch_size,
                query,
                prefix,
                tokens,
                requests,
            },
        };
        if !profile
            .units
            .get(&key.point.phase)
            .is_some_and(|units| units.contains(&key.unit))
        {
            return Err(invalid(
                "serving phase or exact native operation is unmeasured",
            ));
        }
        let latency = if let Some(row) = profile.rows.get(&key) {
            row.latency
        } else {
            interpolate(&profile.rows, &key)?.ok_or_else(|| {
                invalid("serving unit lacks exact data or same-dispatch prefix brackets")
            })?
        };
        Ok(Some(PerformanceResult::with_energy(
            latency,
            0.0,
            Source::Silicon,
        )))
    }
}
fn validate_bounds(context: bool, prefix: u32, query: u32) -> Result<(), AicError> {
    if query == 0
        || (!context && (query != 1 || prefix == 0))
        || prefix.checked_add(query).is_none_or(|n| n > 131072)
    {
        return Err(invalid(
            "serving prefix/query exceeds qualified context bounds",
        ));
    }
    Ok(())
}
fn operation(op: &Op) -> Result<Option<(Unit, Value)>, AicError> {
    let (component, shape) = match op {
        Op::Glm53Attention(o) => {
            o.validate()?;
            ("attention", serde_json::to_value(o))
        }
        Op::Glm53Mhc(o) => {
            o.validate()?;
            ("mhc", serde_json::to_value(o))
        }
        Op::Glm53Ffn(o) => {
            o.validate()?;
            ("ffn", serde_json::to_value(o))
        }
        Op::Glm53Primitive(o) => {
            o.validate()?;
            ("primitive", serde_json::to_value(o))
        }
        Op::Glm53Runtime(o) => {
            o.validate()?;
            ("runtime", serde_json::to_value(o))
        }
        _ => return Ok(None),
    };
    let shape = shape.map_err(|e| invalid(e.to_string()))?;
    Ok(Some((
        Unit {
            component: component.into(),
            name: op.name().into(),
            geometry: geometry(&shape)?,
        },
        shape,
    )))
}
fn validate_units(units: &BTreeSet<Unit>) -> Result<(), AicError> {
    let mut counts = BTreeMap::new();
    let mut names = BTreeSet::new();
    for u in units {
        *counts.entry(u.component.as_str()).or_insert(0usize) += 1;
        if u.name.is_empty()
            || !names.insert(&u.name)
            || (u.component == "runtime" && u.name != "native_graph_setup")
        {
            return Err(invalid(
                "serving operations require unique original names and one setup",
            ));
        }
    }
    if counts
        != BTreeMap::from([
            ("attention", 45),
            ("ffn", 45),
            ("mhc", 93),
            ("primitive", 94),
            ("runtime", 1),
        ])
    {
        return Err(invalid(
            "serving point requires all 277 original physical operations and exactly one setup",
        ));
    }
    Ok(())
}
fn partition(key: &Key) -> Result<Vec<u64>, AicError> {
    if key.unit.component != "attention" {
        return Ok(vec![]);
    }
    let shape: Value =
        serde_json::from_str(&key.unit.geometry).map_err(|e| invalid(e.to_string()))?;
    let context = key.point.phase == "context";
    let p = u64::from(key.point.prefix);
    let q = u64::from(key.point.query);
    if shape["layer_kind"] == "kda" {
        return Ok(if context {
            vec![u64::from(p == 0)]
        } else {
            vec![]
        });
    }
    let pool = shape["index_pool"].as_u64().unwrap_or(4).max(1);
    let topk = shape["index_topk"].as_u64().unwrap_or(2048);
    let total = p + q;
    Ok(if context && p == 0 {
        vec![
            u64::from(total <= topk),
            u64::from(total >= pool),
            u64::from(total % pool != 0),
            1,
        ]
    } else {
        vec![
            u64::from(total <= topk),
            p % pool,
            total % pool,
            u64::from(p == 0),
        ]
    })
}
fn interpolate(rows: &BTreeMap<Key, Measurement>, target: &Key) -> Result<Option<f64>, AicError> {
    let wanted = partition(target)?;
    type DispatchGroups<'a> = BTreeMap<(&'a str, &'a str, u32), Vec<(&'a Key, &'a Measurement)>>;
    let mut groups = DispatchGroups::new();
    // Prefix is ordered after unit/phase/mode/B/Q. Inspect only that exact
    // slice; a full-table scan for each of 278 units scales quadratically.
    let mut lower = target.clone();
    lower.point.prefix = 0;
    lower.point.tokens = 0;
    lower.point.requests = 0;
    let mut upper = target.clone();
    upper.point.prefix = u32::MAX;
    upper.point.tokens = u32::MAX;
    upper.point.requests = u32::MAX;
    for (key, row) in rows.range(lower..=upper) {
        let mut point = key.point.clone();
        point.prefix = target.point.prefix;
        if key.unit == target.unit
            && point == target.point
            && !row.dispatch.is_empty()
            && partition(key)? == wanted
        {
            groups
                .entry((&row.dispatch, &row.method, row.count))
                .or_default()
                .push((key, row));
        }
    }
    let mut answer = None;
    for group in groups.values() {
        let lo = group
            .iter()
            .filter(|(k, _)| k.point.prefix < target.point.prefix)
            .max_by_key(|(k, _)| k.point.prefix);
        let hi = group
            .iter()
            .filter(|(k, _)| k.point.prefix > target.point.prefix)
            .min_by_key(|(k, _)| k.point.prefix);
        if let (Some((lk, lv)), Some((hk, hv))) = (lo, hi) {
            if answer.is_some() {
                return Err(invalid("ambiguous serving prefix dispatch brackets"));
            }
            let w = f64::from(target.point.prefix - lk.point.prefix)
                / f64::from(hk.point.prefix - lk.point.prefix);
            answer = Some(lv.latency * (1.0 - w) + hv.latency * w);
        }
    }
    Ok(answer)
}
fn load(path: &Path, request: &(String, String)) -> Result<Option<Profiles>, AicError> {
    if !path.try_exists().map_err(|e| invalid(e.to_string()))? {
        return Ok(None);
    }
    let reader = PerfReader::open(path)?;
    let policy_col = reader.col("graph_policy")?;
    let mut policies = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let text = row.str(policy_col)?;
        if !policies.contains_key(text) {
            let value: Value = serde_json::from_str(text).map_err(|e| invalid(e.to_string()))?;
            if value["schema_version"] == 3 {
                let policy: ServingPolicy =
                    serde_json::from_value(value).map_err(|e| invalid(e.to_string()))?;
                policy.validate()?;
                if canonical(&policy)? != text
                    || request != &(policy.backend.clone(), policy.backend_version.clone())
                {
                    return Err(invalid(
                        "serving policy is noncanonical or belongs to another runtime",
                    ));
                }
                policies.insert(text.to_owned(), policy);
            }
        }
    }
    if policies.is_empty() {
        return Ok(None);
    }
    let names = [
        "component",
        "operation_name",
        "geometry",
        "phase",
        "runtime_mode",
        "batch_size",
        "query_length",
        "prefix",
        "physical_num_tokens",
        "physical_num_requests",
        "latency",
        "contribution_count",
        "sample_count",
        "dispatch_fingerprint",
        "measurement_method",
        "graph_policy",
        "graph_policy_sha256",
        "dataset_role",
        "aggregation_policy",
        "rank_selection_sha256",
        "evidence_sha256",
        "policy_evidence_sha256",
        "measurement_scope",
    ];
    let c = names
        .iter()
        .map(|n| reader.col(n))
        .collect::<Result<Vec<_>, _>>()?;
    let mut profiles = Profiles::new();
    let mut points = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let Some(policy) = policies.get(row.str(c[15])?) else {
            continue;
        };
        if digest(row.str(c[15])?) != row.str(c[16])? {
            return Err(invalid("serving policy SHA differs"));
        }
        let id = (policy.checkpoint_format.clone(), policy.tp_size);
        let profile = profiles.entry(id.clone()).or_insert_with(|| Profile {
            policy: policy.clone(),
            rows: BTreeMap::new(),
            units: BTreeMap::new(),
        });
        if profile.policy != *policy {
            return Err(invalid(
                "serving checkpoint/TP has competing runtime policies",
            ));
        }
        let unit = Unit {
            component: row.str(c[0])?.into(),
            name: row.str(c[1])?.into(),
            geometry: row.str(c[2])?.into(),
        };
        let shape = if unit.component == "runtime" {
            let mut v: Value =
                serde_json::from_str(&unit.geometry).map_err(|e| invalid(e.to_string()))?;
            v.as_object_mut()
                .ok_or_else(|| invalid("setup geometry must be an object"))?
                .insert("name".into(), unit.name.clone().into());
            let op: crate::operators::Glm53RuntimeOp =
                serde_json::from_value(v).map_err(|e| invalid(e.to_string()))?;
            op.validate()?;
            if geometry(&op)? != unit.geometry {
                return Err(invalid("noncanonical serving setup geometry"));
            }
            serde_json::to_value(op).map_err(|e| invalid(e.to_string()))?
        } else {
            validate_geometry(&unit.component, &unit.geometry)?
        };
        let point = Point {
            phase: row.str(c[3])?.into(),
            mode: row.str(c[4])?.into(),
            batch: row.u32(c[5])?,
            query: row.u32(c[6])?,
            prefix: row.u32(c[7])?,
            tokens: row.u32(c[8])?,
            requests: row.u32(c[9])?,
        };
        let context = point.phase == "context";
        if !matches!(point.phase.as_str(), "context" | "generation")
            || shape["is_context"] != context
            || shape["backend"] != policy.backend
            || shape["checkpoint_format"] != policy.checkpoint_format
            || shape["tp_size"] != policy.tp_size
        {
            return Err(invalid(
                "serving row geometry differs from phase/runtime policy",
            ));
        }
        validate_bounds(context, point.prefix, point.query)?;
        validate_native_workload(
            &unit.component,
            &shape,
            point.prefix,
            point.query,
            Some(&policy.backend_version),
        )?;
        if policy
            .native_policy
            .select(context, point.batch, point.query)?
            != (point.mode.clone(), point.tokens, point.requests)
        {
            return Err(invalid("serving row has wrong native dispatch/padding"));
        }
        let latency = row.f64(c[10])?;
        let count = row.u32(c[11])?;
        let sample_count = row.u32(c[12])?;
        let dispatch = row.str(c[13])?;
        let method = row.str(c[14])?;
        let timing = match (point.mode.as_str(), unit.component.as_str()) {
            ("NONE", "runtime") => method == "native_runtime_cuda_events_v1" && count == 2,
            ("NONE", _) => method == "native_module_cuda_events_v1" && count == 1 && latency > 0.0,
            ("FULL" | "PIECEWISE", _) => {
                method == "native_cupti_unit_union_v1" && (count == 0) == (latency == 0.0)
            }
            _ => false,
        };
        if !latency.is_finite()
            || latency < 0.0
            || !timing
            || sample_count < 10
            || !(sha256(dispatch) || point.mode == "NONE" && dispatch.is_empty())
            || row.str(c[17])? != "calibration"
            || row.str(c[18])? != "whole_forward_slowest_rank_v1"
            || row.str(c[22])? != "native_vllm_serving_units_v1"
            || [c[19], c[20], c[21]]
                .iter()
                .any(|&i| !row.str(i).is_ok_and(sha256))
        {
            return Err(invalid(
                "serving row lacks complete measured timing/provenance",
            ));
        }
        let evidence = (
            row.str(c[19])?.to_owned(),
            row.str(c[20])?.to_owned(),
            row.str(c[21])?.to_owned(),
            sample_count,
        );
        let (old_evidence, units) = points
            .entry((id, point.clone()))
            .or_insert_with(|| (evidence.clone(), BTreeSet::new()));
        if *old_evidence != evidence || !units.insert(unit.clone()) {
            return Err(invalid(
                "serving point mixes original evidence or duplicates units",
            ));
        }
        if profile
            .rows
            .insert(
                Key { unit, point },
                Measurement {
                    latency,
                    count,
                    method: method.into(),
                    dispatch: dispatch.into(),
                },
            )
            .is_some()
        {
            return Err(invalid("duplicate serving physical key"));
        }
    }
    for ((id, point), (_, units)) in points {
        validate_units(&units)?;
        let profile = profiles.get_mut(&id).unwrap();
        if profile
            .units
            .insert(point.phase, units.clone())
            .is_some_and(|old| old != units)
        {
            return Err(invalid(
                "serving points have different exact operation manifests",
            ));
        }
    }
    Ok(Some(profiles))
}

#[cfg(test)]
mod tests {
    // TEST_ONLY authored policies/latencies, never performance observations.
    use super::*;
    const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    fn policy(piecewise: bool) -> NativePolicy {
        let full: Vec<_> = [1, 2, 4].map(|n| Descriptor::captured("FULL", n)).into();
        let mut captures = BTreeMap::from([("FULL".into(), full.iter().rev().cloned().collect())]);
        if piecewise {
            captures.insert(
                "PIECEWISE".into(),
                [8, 4, 2, 1]
                    .map(|n| Descriptor::captured("PIECEWISE", n))
                    .into(),
            );
        }
        NativePolicy {
            backend: "vllm".into(),
            backend_version: "0.30.0".into(),
            backend_revision: VLLM_REVISION.into(),
            source_pins: vllm_pins(),
            native_flags: vllm_flags(),
            capture_sizes: vec![1, 2, 4, 8],
            max_num_reqs: 4,
            max_capture_tokens: 8,
            decode_query_len: 1,
            graphs_captured: true,
            lora_capture_cases: vec![0],
            dp_size: 1,
            tp_size: 2,
            resolved_mode: if piecewise {
                "FULL_AND_PIECEWISE"
            } else {
                "FULL_DECODE_ONLY"
            }
            .into(),
            use_breakable_cg: piecewise,
            capture_descriptors: captures,
            full_graphs: full,
            candidates: (0..=if piecewise { 8 } else { 4 })
                .map(|n| {
                    let mut ds = Vec::new();
                    if let Some(b) = [1, 2, 4].into_iter().find(|b| *b >= n) {
                        ds.push(Descriptor::captured("FULL", b));
                    }
                    if piecewise {
                        ds.push(Descriptor::captured(
                            "PIECEWISE",
                            [1, 2, 4, 8].into_iter().find(|b| *b >= n).unwrap(),
                        ));
                    }
                    Candidate {
                        num_tokens: n,
                        num_active_loras: 0,
                        descriptors: ds,
                    }
                })
                .collect(),
            piecewise_entries: if piecewise {
                [1, 2, 4, 8]
                    .map(|n| PiecewiseEntry {
                        num_tokens: n,
                        num_reqs: None,
                        uniform: false,
                        has_lora: false,
                        num_active_loras: 0,
                        completed: true,
                        num_graphs: 46,
                        num_eager_breaks: 45,
                    })
                    .into()
            } else {
                vec![]
            },
        }
    }
    #[test]
    fn exact_policy_distinguishes_context_q1_from_decode_and_preserves_none_work() {
        let p = policy(true);
        p.validate().unwrap();
        assert_eq!(p.select(false, 3, 1).unwrap(), ("FULL".into(), 4, 4));
        assert_eq!(p.select(true, 3, 1).unwrap(), ("PIECEWISE".into(), 4, 3));
        assert_eq!(p.select(true, 1, 5).unwrap(), ("PIECEWISE".into(), 8, 1));
        assert_eq!(p.select(true, 2, 5).unwrap(), ("NONE".into(), 10, 2));
        for (context, b, q) in [(true, 0, 1), (true, 5, 1), (true, 1, 0), (false, 1, 2)] {
            assert!(p.select(context, b, q).is_err());
        }
    }
    #[test]
    fn full_decode_only_native_candidate_list_omits_empty_trailing_counts() {
        let p = policy(false);
        p.validate().unwrap();
        assert_eq!(p.candidates.len(), 5); // configured8, maxrequests4
        assert_eq!(p.select(true, 1, 1).unwrap(), ("NONE".into(), 1, 1));
        let mut bad = p.clone();
        bad.candidates.push(Candidate {
            num_tokens: 5,
            num_active_loras: 0,
            descriptors: vec![],
        });
        assert!(bad.validate().is_err());
    }
    #[test]
    fn native_policy_rejects_changed_source_priority_missing_entry_and_unadmitted_repair() {
        for defect in 0..5 {
            let mut p = policy(true);
            match defect {
                0 => {
                    p.source_pins.insert("unexpected.py".into(), SHA.into());
                }
                1 => p.candidates[1].descriptors.reverse(),
                2 => {
                    p.piecewise_entries.pop();
                }
                3 => p.backend_version = "0.30.0+glm53tail.eb4704514fdf".into(),
                _ => p
                    .native_flags
                    .insert("prefix_caching".into(), true)
                    .map(|_| ())
                    .unwrap(),
            }
            assert!(p.validate().is_err(), "defect {defect}");
        }
        let mut encoded = serde_json::to_value(policy(true)).unwrap();
        encoded["schema_version"] = 1.into();
        assert!(serde_json::from_value::<NativePolicy>(encoded).is_err());
    }
    fn key(prefix: u32) -> Key {
        Key {
            unit: Unit {
                component: "mhc".into(),
                name: "TEST_ONLY_mhc_0".into(),
                geometry: "{}".into(),
            },
            point: Point {
                phase: "context".into(),
                mode: "PIECEWISE".into(),
                batch: 1,
                query: 4,
                prefix,
                tokens: 4,
                requests: 1,
            },
        }
    }
    fn measured(value: f64, dispatch: &str) -> Measurement {
        Measurement {
            latency: value,
            count: 1,
            method: "native_cupti_unit_union_v1".into(),
            dispatch: dispatch.into(),
        }
    }
    #[test]
    fn prefix_only_interpolation_requires_same_unit_bucket_query_method_count_and_fingerprint() {
        let target = key(8);
        let rows = BTreeMap::from([(key(4), measured(2.0, SHA)), (key(12), measured(6.0, SHA))]);
        assert_eq!(interpolate(&rows, &target).unwrap(), Some(4.0));
        for defect in 0..9 {
            let mut high = key(12);
            let mut row = measured(6.0, SHA);
            match defect {
                0 => high.unit.name = "other_layer".into(),
                1 => high.point.mode = "FULL".into(),
                2 => high.point.query = 5,
                3 => high.point.tokens = 8,
                4 => high.point.requests = 2,
                5 => row.method = "other".into(),
                6 => row.count = 2,
                7 => row.dispatch = "".into(),
                _ => high.point.batch = 2,
            }
            let rows = BTreeMap::from([(key(4), measured(2.0, SHA)), (high, row)]);
            assert_eq!(
                interpolate(&rows, &target).unwrap(),
                None,
                "defect {defect}"
            );
        }
        assert_eq!(interpolate(&rows, &key(16)).unwrap(), None);
    }
    #[test]
    fn empty_none_dispatch_is_exact_only_and_competing_brackets_reject() {
        let mut rows = BTreeMap::from([(key(4), measured(2.0, "")), (key(12), measured(6.0, ""))]);
        assert_eq!(interpolate(&rows, &key(8)).unwrap(), None);
        rows = BTreeMap::from([
            (key(4), measured(2.0, SHA)),
            (key(12), measured(6.0, SHA)),
            (key(6), measured(2.0, "b")),
            (key(10), measured(6.0, "b")),
        ]);
        assert!(interpolate(&rows, &key(8)).is_err());
    }
    #[test]
    fn attention_state_boundaries_and_inclusive_limit_are_not_smoothed() {
        let mut lo = key(127);
        lo.unit.component = "attention".into();
        lo.unit.geometry = r#"{"layer_kind":"sparse_mla","index_pool":4,"index_topk":2048}"#.into();
        let mut hi = lo.clone();
        hi.point.prefix = 135;
        let mut middle = lo.clone();
        middle.point.prefix = 131;
        let rows = BTreeMap::from([
            (lo.clone(), measured(1.0, SHA)),
            (hi.clone(), measured(3.0, SHA)),
        ]);
        assert_eq!(interpolate(&rows, &middle).unwrap(), Some(2.0));
        middle.point.prefix = 132;
        assert_eq!(interpolate(&rows, &middle).unwrap(), None);
        assert!(validate_bounds(false, 131071, 1).is_ok());
        for (ctx, p, q) in [
            (false, 0, 1),
            (false, 131072, 1),
            (true, 131072, 1),
            (true, 0, 0),
            (true, u32::MAX, 1),
        ] {
            assert!(validate_bounds(ctx, p, q).is_err());
        }
        assert!(validate_bounds(true, 0, 131072).is_ok());
    }
}
