// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema 3 native vLLM serving units. Original implementation of the frozen
//! serving contract; source dispatch follows the pinned native policy snapshot.
//! This reader never grants runtime admission or borrows eager/FULL-only rows.

use super::glm53flash::{
    VLLM_TAIL_VERSION, geometry, sha256, validate_geometry, validate_native_workload,
    validate_runtime,
};
use super::glm53flash_graph::{VLLM_REVISION, VLLM_STOCK_SOURCE, vllm_flags, vllm_pins};
use super::parquet_loader::PerfReader;
use crate::common::error::AicError;
use crate::operators::op::RuntimeContext;
use crate::operators::{Op, PerformanceResult, Source};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const VLLM_TAIL_SOURCE: &str = "603066c63ced49b8e059ff020a372acb75539d45c9d1bff591fade9d4f51f63b";
const LOOKUP_CONTRACT: &str = "vllm_serving_bounded_p_q_v1";

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
        // Canonical effective model-source maps, not the four unchanged native
        // graph-policy files or the separate worker source/library inventory.
        let source = match self.backend_version.as_str() {
            "0.30.0" => VLLM_STOCK_SOURCE,
            VLLM_TAIL_VERSION => VLLM_TAIL_SOURCE,
            _ => return Err(invalid("unqualified native serving runtime")),
        };
        let revision = match self.checkpoint_format.as_str() {
            "fp8" => "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
            "nvfp4" => "09b04e5e74bca08ca8549fc736d4cdd8624bfde3",
            _ => return Err(invalid("unknown serving checkpoint")),
        };
        if self.schema_version != 3
            || self.backend != "vllm"
            || self.backend_revision != VLLM_REVISION
            || self.source_sha256 != source
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
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
#[derive(Debug, Clone, Serialize)]
struct Measurement {
    latency: f64,
    #[serde(rename = "contribution_count")]
    count: u32,
    #[serde(rename = "measurement_method")]
    method: String,
    #[serde(rename = "dispatch_fingerprint")]
    dispatch: String,
    sample_count: u32,
    rank_selection_sha256: String,
    evidence_sha256: String,
    policy_evidence_sha256: String,
    source_ownership_sha256: Option<String>,
}
struct Profile {
    policy: ServingPolicy,
    lookup_contract: Option<String>,
    rows: BTreeMap<Key, Measurement>,
    units: BTreeMap<String, BTreeSet<Unit>>,
}
impl Profile {
    fn lookup(&self, target: &Key) -> Result<(f64, Value), AicError> {
        let (axis, endpoints) = if self.rows.contains_key(target) {
            ("exact", vec![(target.clone(), 1.0)])
        } else if self.lookup_contract.as_deref() == Some(LOOKUP_CONTRACT) {
            self.bounded_endpoints(target)?
        } else {
            return interpolate(&self.rows, target)?
                .map(|latency| (latency, Value::Null))
                .ok_or_else(|| {
                    invalid("serving unit lacks exact data or same-dispatch prefix brackets")
                });
        };
        let mut total = 0.0;
        let mut evidence = Vec::new();
        let mut ownership = None;
        for (key, weight) in endpoints {
            let row = self
                .rows
                .get(&key)
                .ok_or_else(|| invalid("missing serving endpoint unit"))?;
            if axis != "exact" {
                let source = row
                    .source_ownership_sha256
                    .as_ref()
                    .ok_or_else(|| invalid("serving endpoint lacks original source ownership"))?;
                let identity = (source, &row.method);
                if ownership.is_some_and(|old| old != identity) {
                    return Err(invalid(
                        "serving endpoints change native source ownership or measurement method",
                    ));
                }
                ownership = Some(identity);
            }
            total += weight * row.latency;
            evidence.push(json!({"point":key.point,"weight":weight,"measurement":row}));
        }
        if !total.is_finite() || total < 0.0 {
            return Err(invalid("invalid bounded serving latency"));
        }
        Ok((
            total,
            json!({"operation_name":target.unit.name,"geometry":target.unit.geometry,
            "target":target.point,"axis":axis,"latency_ms":total,"endpoints":evidence}),
        ))
    }
    fn bounded_endpoints(&self, target: &Key) -> Result<(&'static str, Vec<(Key, f64)>), AicError> {
        let wanted = partition(target)?;
        let mut lower = target.clone();
        lower.point.query = 0;
        lower.point.prefix = 0;
        lower.point.tokens = 0;
        lower.point.requests = 0;
        let mut upper = lower.clone();
        upper.point.query = u32::MAX;
        upper.point.prefix = u32::MAX;
        upper.point.tokens = u32::MAX;
        upper.point.requests = u32::MAX;
        let mut lane = Vec::new();
        for (key, _) in self.rows.range(lower..=upper) {
            let p = &key.point;
            let t = &target.point;
            if key.unit == target.unit
                && p.phase == t.phase
                && p.mode == t.mode
                && p.batch == t.batch
                && p.requests == t.requests
                && (p.prefix == 0) == (t.prefix == 0)
                && (p.mode == "NONE" || p.tokens == t.tokens)
                && partition(key)? == wanted
            {
                lane.push(key);
            }
        }
        for axis in ["P", "Q"] {
            if axis == "Q" && (target.point.phase != "context" || target.point.prefix != 0) {
                continue;
            }
            let coordinate = |key: &Key| {
                if axis == "P" {
                    key.point.prefix
                } else {
                    key.point.query
                }
            };
            let same_axis = |key: &&Key| {
                if axis == "P" {
                    key.point.query == target.point.query
                } else {
                    key.point.prefix == target.point.prefix
                }
            };
            let lo = lane
                .iter()
                .copied()
                .filter(same_axis)
                .filter(|key| coordinate(key) < coordinate(target))
                .max_by_key(|key| coordinate(key));
            let hi = lane
                .iter()
                .copied()
                .filter(same_axis)
                .filter(|key| coordinate(key) > coordinate(target))
                .min_by_key(|key| coordinate(key));
            if let (Some(lo), Some(hi)) = (lo, hi) {
                let weight = f64::from(coordinate(target) - coordinate(lo))
                    / f64::from(coordinate(hi) - coordinate(lo));
                return Ok((axis, vec![(lo.clone(), 1.0 - weight), (hi.clone(), weight)]));
            }
        }
        Err(invalid(
            "serving unit has no enclosing measured P or P0/Q endpoints with the same native mode/padding/state; no extrapolation or fallback",
        ))
    }
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
        let (latency, _) = profile.lookup(&key)?;
        Ok(Some(PerformanceResult::with_energy(
            latency,
            0.0,
            Source::Silicon,
        )))
    }
    pub(super) fn audit(
        &self,
        context_ops: &[Op],
        generation_ops: &[Op],
        context: bool,
        point: (u32, u32, u32),
    ) -> Result<Value, AicError> {
        if !self.validate_ops(context_ops, true)? || !self.validate_ops(generation_ops, false)? {
            return Err(invalid(
                "serving audit requires both complete named model phases",
            ));
        }
        let (batch, query, prefix) = point;
        validate_bounds(context, prefix, query)?;
        let profiles = self
            .profiles()?
            .ok_or_else(|| invalid("missing serving audit table"))?;
        let mut operations = Vec::new();
        let mut policy_hash = None;
        let mut native_policy_hash = None;
        let phase = if context { "context" } else { "generation" };
        for op in if context { context_ops } else { generation_ops } {
            let (unit, shape) =
                operation(op)?.ok_or_else(|| invalid("invalid serving audit operation"))?;
            let identity = (
                shape["checkpoint_format"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                shape["tp_size"].as_u64().unwrap_or_default() as u32,
            );
            let profile = profiles
                .get(&identity)
                .ok_or_else(|| invalid("missing serving audit identity"))?;
            if profile.lookup_contract.as_deref() != Some(LOOKUP_CONTRACT) {
                return Err(invalid(
                    "serving endpoint audit requires explicit bounded lookup metadata",
                ));
            }
            validate_native_workload(
                &unit.component,
                &shape,
                prefix,
                query,
                Some(&profile.policy.backend_version),
            )?;
            let (mode, tokens, requests) =
                profile.policy.native_policy.select(context, batch, query)?;
            let target = Key {
                unit,
                point: Point {
                    phase: phase.into(),
                    mode,
                    batch,
                    query,
                    prefix,
                    tokens,
                    requests,
                },
            };
            let (_, row) = profile.lookup(&target)?;
            policy_hash = Some(digest(&canonical(&profile.policy)?));
            native_policy_hash = Some(profile.policy.native_policy_sha256.clone());
            operations.push(row);
        }
        Ok(
            json!({"schema":"glm53flash_lookup_audit_v1","graph_policy_sha256":policy_hash,
            "native_policy_sha256":native_policy_hash,"lookup_contract":LOOKUP_CONTRACT,
            "phase":phase,"target":{"batch":batch,"query":query,"prefix":prefix},"operations":operations}),
        )
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
    let lookup_col = reader.col_optional("lookup_contract");
    let ownership_col = reader.col_optional("source_ownership_sha256");
    if lookup_col.is_some() != ownership_col.is_some() {
        return Err(invalid("incomplete serving lookup metadata columns"));
    }
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
        let lookup_contract = lookup_col
            .map(|c| row.str(c).map(str::to_owned))
            .transpose()?;
        let source_ownership = ownership_col
            .map(|c| row.str(c).map(str::to_owned))
            .transpose()?;
        if lookup_contract
            .as_deref()
            .is_some_and(|s| s != LOOKUP_CONTRACT)
            || source_ownership.as_deref().is_some_and(|s| !sha256(s))
        {
            return Err(invalid(
                "unknown serving lookup contract or invalid source ownership",
            ));
        }
        let profile = profiles.entry(id.clone()).or_insert_with(|| Profile {
            policy: policy.clone(),
            lookup_contract: lookup_contract.clone(),
            rows: BTreeMap::new(),
            units: BTreeMap::new(),
        });
        if profile.policy != *policy || profile.lookup_contract != lookup_contract {
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
                    sample_count,
                    rank_selection_sha256: row.str(c[19])?.into(),
                    evidence_sha256: row.str(c[20])?.into(),
                    policy_evidence_sha256: row.str(c[21])?.into(),
                    source_ownership_sha256: source_ownership,
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
                3 => p.backend_version = "0.30.0+glm53tailref.4e4a40c2a838".into(),
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

    #[test]
    fn admitted_tail_policy_requires_exact_source_and_matching_native_version() {
        for piecewise in [false, true] {
            for (format, revision) in [
                ("fp8", "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"),
                ("nvfp4", "09b04e5e74bca08ca8549fc736d4cdd8624bfde3"),
            ] {
                for tp in [2, 4] {
                    let mut native = policy(piecewise);
                    native.backend_version = VLLM_TAIL_VERSION.into();
                    native.tp_size = tp;
                    let p = ServingPolicy {
                        schema_version: 3,
                        backend: "vllm".into(),
                        backend_version: VLLM_TAIL_VERSION.into(),
                        backend_revision: VLLM_REVISION.into(),
                        checkpoint_format: format.into(),
                        checkpoint_revision: revision.into(),
                        config_sha256: SHA.into(),
                        execution_policy_sha256: SHA.into(),
                        native_policy_sha256: digest(&canonical(&native).unwrap()),
                        native_policy: native,
                        runtime_digest: format!("sha256:{SHA}"),
                        source_pins: vllm_pins(),
                        source_sha256: VLLM_TAIL_SOURCE.into(),
                        timing_boundary: "native_metadata_to_logits_gpu_v1".into(),
                        tp_size: tp,
                    };
                    p.validate().unwrap();
                    for defect in 0..5 {
                        let mut bad = p.clone();
                        match defect {
                            0 => bad.source_sha256 = VLLM_STOCK_SOURCE.into(),
                            1 => bad.source_sha256 = SHA.into(),
                            2 => bad.backend_version = "0.30.0".into(),
                            3 => {
                                bad.native_policy.backend_version = "0.30.0".into();
                                bad.native_policy_sha256 =
                                    digest(&canonical(&bad.native_policy).unwrap());
                            }
                            _ => {
                                bad.source_pins
                                    .insert("config/compilation.py".into(), SHA.into());
                            }
                        }
                        assert!(bad.validate().is_err(), "defect {defect}");
                    }
                }
            }
        }
    }

    #[test]
    fn tail_model_source_pin_matches_packaged_python_admission_closure() {
        let runtime = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/aisimulate/collector/fpm_forward/runtime");
        let read = |relative: &str| std::fs::read_to_string(runtime.join(relative)).unwrap();
        let mut sources: BTreeMap<String, String> =
            serde_json::from_str(&read("glm53flash/runtime-source-sha256.json")).unwrap();
        for relative in [
            "glm53flash_vllm_tail_repair/candidate/expected-source-sha256.json",
            "glm53flash_vllm_kpool_candidate/v2-source-sha256.json",
        ] {
            sources
                .extend(serde_json::from_str::<BTreeMap<String, String>>(&read(relative)).unwrap());
        }
        assert_eq!(sources.len(), 37);
        assert_eq!(digest(&canonical(&sources).unwrap()), VLLM_TAIL_SOURCE);
        let build: Value = serde_json::from_str(&read(
            "glm53flash_vllm_tail_repair/candidate/actual-build-receipt.json",
        ))
        .unwrap();
        assert_eq!(build["version"], VLLM_TAIL_VERSION);
        for patch in build["patch"]["sources"].as_array().unwrap() {
            assert_eq!(
                sources[patch["source_path"].as_str().unwrap()],
                patch["patched_sha256"]
            );
        }
        assert_eq!(
            digest(&read(
                "glm53flash_vllm_tail_repair/qualification/admission-summary.json"
            )),
            "8fc691d6054f48741c248eb7937b7b4db6220ff1ea337b968ff656c56ba8cf45"
        );
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
            sample_count: 10,
            rank_selection_sha256: SHA.into(),
            evidence_sha256: SHA.into(),
            policy_evidence_sha256: SHA.into(),
            source_ownership_sha256: None,
        }
    }
    fn bounded_profile(rows: BTreeMap<Key, Measurement>) -> Profile {
        let native = policy(true);
        Profile {
            policy: ServingPolicy {
                schema_version: 3,
                backend: "vllm".into(),
                backend_version: "0.30.0".into(),
                backend_revision: VLLM_REVISION.into(),
                checkpoint_format: "fp8".into(),
                checkpoint_revision: "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a".into(),
                config_sha256: SHA.into(),
                execution_policy_sha256: SHA.into(),
                native_policy_sha256: digest(&canonical(&native).unwrap()),
                native_policy: native,
                runtime_digest: format!("sha256:{SHA}"),
                source_pins: vllm_pins(),
                source_sha256: VLLM_STOCK_SOURCE.into(),
                timing_boundary: "native_metadata_to_logits_gpu_v1".into(),
                tp_size: 2,
            },
            lookup_contract: Some(LOOKUP_CONTRACT.into()),
            rows,
            units: BTreeMap::new(),
        }
    }
    fn owned(value: f64, dispatch: &str) -> Measurement {
        let mut m = measured(value, dispatch);
        m.source_ownership_sha256 = Some(SHA.into());
        m
    }
    #[test]
    fn bounded_prefix_uses_measured_ownership_not_equal_kernel_inventory() {
        let mut hi = owned(
            6.0,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        );
        hi.count = 4; // Different original GPU activity count, same physical unit.
        let mut p = bounded_profile(BTreeMap::from([(key(4), owned(2.0, SHA)), (key(12), hi)]));
        let (value, audit) = p.lookup(&key(8)).unwrap();
        assert_eq!(value, 4.0); // Equal distances from measured 2 and 6.
        assert_eq!(audit["axis"], "P");
        assert_eq!(audit["endpoints"][0]["weight"], 0.5);
        assert_ne!(
            audit["endpoints"][0]["measurement"]["dispatch_fingerprint"],
            audit["endpoints"][1]["measurement"]["dispatch_fingerprint"]
        );
        p.lookup_contract = None;
        assert!(p.lookup(&key(8)).is_err()); // Original contract remains strict.
    }
    #[test]
    fn bounded_none_q_changes_physical_tokens_but_not_batch_or_cached_state() {
        let k = |q, prefix| {
            let mut k = key(prefix);
            k.point.query = q;
            k.point.mode = "NONE".into();
            k.point.tokens = q;
            k
        };
        let p = bounded_profile(BTreeMap::from([
            (k(8, 0), owned(2.0, "")),
            (k(16, 0), owned(6.0, "")),
        ]));
        let (value, audit) = p.lookup(&k(12, 0)).unwrap();
        assert_eq!(value, 4.0);
        assert_eq!(audit["axis"], "Q");
        assert_eq!(audit["endpoints"][0]["point"]["tokens"], 8);
        assert_eq!(audit["endpoints"][1]["point"]["tokens"], 16);
        for target in [k(4, 0), k(32, 0), k(12, 4)] {
            assert!(p.lookup(&target).is_err());
        }
        let mut wrong = k(12, 0);
        wrong.point.batch = 2;
        wrong.point.tokens = 24;
        wrong.point.requests = 2;
        assert!(p.lookup(&wrong).is_err());
        assert_eq!(p.lookup(&k(8, 0)).unwrap().1["axis"], "exact");
    }
    #[test]
    fn bounded_lookup_rejects_changed_source_method_mode_and_native_padding() {
        for defect in 0..5 {
            let mut hi = key(12);
            let mut value = owned(6.0, SHA);
            match defect {
                0 => value.source_ownership_sha256 = Some("b".repeat(64)),
                1 => value.source_ownership_sha256 = None,
                2 => value.method = "native_module_cuda_events_v1".into(),
                3 => hi.point.tokens = 8,
                _ => hi.point.mode = "NONE".into(),
            }
            let p = bounded_profile(BTreeMap::from([(key(4), owned(2.0, SHA)), (hi, value)]));
            assert!(p.lookup(&key(8)).is_err(), "defect {defect}");
        }
        let p = bounded_profile(BTreeMap::from([
            (key(0), owned(2.0, SHA)),
            (key(12), owned(6.0, SHA)),
        ]));
        assert!(p.lookup(&key(8)).is_err()); // Initialization cannot bracket cached state.
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
