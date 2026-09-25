// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Separate, source-bound native FULL graph profiles. Costs are measured
//! disjoint-node activity unions, composed as an explicit additive approximation.
//! They are never inferred from whole-forward residuals or eager module timings.

use super::glm53flash::{geometry, primary_path, sha256, validate_geometry, validate_runtime};
use super::{SourceResolver, parquet_loader::PerfReader};
use crate::common::error::AicError;
use crate::operators::op::RuntimeContext;
use crate::operators::{Op, PerformanceResult, Source};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

pub const BASENAME: &str = "glm53flash_graph_perf.parquet";
const NAMED_CONTRACT: &str = "graph_named_operations_v1";
type NamedUnit = (String, String, String);
const SG_SOURCE: &str = "401b762a863931720b2b5cdc7b64246fac11cb215dbf6ea0fd19f3db24ce7e49";
const SG_REVISION: &str = "94602c9c2b7cbdb8efd5c52802dac6a1c180089e";
const SG_DECODE: &str = "55892739b9c577ae43a60d5d31eac53f81e2b4aeca57ef5368b9c881117889d8";
const SG_BASE: &str = "03098df21a963d28f8075e0630a2bc1c356560449f9ca5f9e74dfb5f9892b7ed";
const SG_FULL: &str = "0dc52a9a581636a20f5070cbb81d921bc56e4fb3394a9a1cf601747271c6905b";
const SG_SHAPE: &str = "26e3f15209b654345a35966bd817ff8d0eb6c4c118527e78ca1e89942d5ea2c5";
pub(super) const VLLM_REVISION: &str = "ced6857afa0ea7b2e3f0846a62e1394e90f15607";
pub(super) const VLLM_STOCK_SOURCE: &str =
    "46cb601e49c399143db029d3cce33c2ee5216b8cdb6bf62385820b25fc67cba8";
#[cfg(test)]
const VLLM_REPAIR_SOURCE: &str = "06a8cb8ab3fa89d4e82428fd32112074f50249c427a467787004ecb0a870f128";
pub(super) fn vllm_pins() -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            "v1/worker/gpu/cudagraph_utils.py".into(),
            "6e9c042890603535e300a40df8ee159dbed1058a64a83ae50ff0329e332e05ff".into(),
        ),
        (
            "v1/worker/gpu/model_runner.py".into(),
            "174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce".into(),
        ),
        (
            "config/compilation.py".into(),
            "c9cec5c7200e8e559810ec8c30113ad61dab6780f9f7fb3c116bd0d9b5a43065".into(),
        ),
        (
            "compilation/breakable_cudagraph.py".into(),
            "3cc427612a08e2b9b3fee47548026400c1d0776e2d4747535e59ef5512bdf1e8".into(),
        ),
    ])
}
pub(super) fn vllm_flags() -> BTreeMap<String, bool> {
    [
        "compiled_model",
        "varlen_decode",
        "microbatch_runner",
        "speculative",
        "lora",
        "encoder_decoder",
        "async_scheduling",
        "expert_parallel",
        "prefix_caching",
        "kda_recoverssm",
    ]
    .into_iter()
    .map(|key| (key.into(), false))
    .collect()
}
fn invalid(message: impl Into<String>) -> AicError {
    AicError::InvalidPerfData(message.into())
}

/// Actual initialized native policy, not a per-query table of observed answers.
/// Ordinary uncompiled SGLang FULL (schema 1) and native V2 FULL (schema 2)
/// one-token decode. V2 PIECEWISE buckets are excluded from this query region.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphPolicy {
    pub schema_version: u32,
    pub backend: String,
    pub backend_version: String,
    pub backend_revision: String,
    pub checkpoint_format: String,
    pub checkpoint_revision: String,
    pub tp_size: u32,
    pub phase: String,
    pub runtime_mode: String,
    pub capture_sizes: Vec<u32>,
    pub disable_padding: bool,
    pub captured_req_width: u32,
    pub native_flags: BTreeMap<String, bool>,
    pub source_pins: BTreeMap<String, String>,
    pub source_sha256: String,
    pub config_sha256: String,
    pub runtime_digest: String,
    pub resolved_config_sha256: String,
    pub native_policy_receipt_sha256: String,
    pub state_layout_sha256: BTreeMap<u32, String>,
    pub capture_registry_sha256: BTreeMap<u32, Vec<String>>,
}
impl GraphPolicy {
    pub fn validate(&self) -> Result<(), AicError> {
        validate_runtime(&self.backend, &self.backend_version)?;
        let expected = if self.checkpoint_format == "fp8" {
            "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
        } else if self.checkpoint_format == "nvfp4" {
            "09b04e5e74bca08ca8549fc736d4cdd8624bfde3"
        } else {
            return Err(invalid("unqualified native graph checkpoint"));
        };
        let pins = BTreeMap::from([
            (
                "srt/model_executor/runner/decode_cuda_graph_runner.py".into(),
                SG_DECODE.into(),
            ),
            (
                "srt/model_executor/runner/base_cuda_graph_runner.py".into(),
                SG_BASE.into(),
            ),
            (
                "srt/model_executor/runner_backend/full_cuda_graph_backend.py".into(),
                SG_FULL.into(),
            ),
            (
                "srt/model_executor/runner/shape_key.py".into(),
                SG_SHAPE.into(),
            ),
        ]);
        // Exact native predicates from decode runner can_run_graph/load_batch
        // and base runner _pad_to_bucket at the pinned source hashes above.
        let flags = BTreeMap::from([
            ("enable_torch_compile".into(), false),
            ("is_encoder_decoder".into(), false),
            ("require_mlp_tp_gather".into(), false),
            ("require_attn_tp_gather".into(), false),
            ("require_mlp_sync".into(), false),
            ("enable_two_batch_overlap".into(), false),
            ("enable_pdmux".into(), false),
            ("ragged_verify_mode".into(), false),
            ("enable_prefill_cp".into(), false),
            ("enable_lora".into(), false),
            ("speculative_enabled".into(), false),
            ("metadata_glue_enabled".into(), false),
            ("reuse_output_buffer".into(), false),
        ]);
        let sg_identity = self.schema_version == 1
            && self.backend == "sglang"
            && self.backend_version == "0.5.20"
            && self.backend_revision == SG_REVISION
            && self.source_sha256 == SG_SOURCE
            && self.source_pins == pins
            && self.native_flags == flags;
        let vllm_identity = self.schema_version == 2
            && self.backend == "vllm"
            && self.backend_revision == VLLM_REVISION
            && self.backend_version == "0.30.0"
            && self.source_sha256 == VLLM_STOCK_SOURCE
            && self.source_pins == vllm_pins()
            && self.native_flags == vllm_flags()
            && !self.disable_padding;
        if !(sg_identity || vllm_identity)
            || self.checkpoint_revision != expected
            || !matches!(self.tp_size, 2 | 4)
            || self.phase != "generation"
            || self.runtime_mode != "FULL"
            || self.captured_req_width != 1
            || self.capture_sizes.is_empty()
            || self.capture_sizes[0] == 0
            || self.capture_sizes.windows(2).any(|p| p[0] >= p[1])
            || !sha256(&self.config_sha256)
            || !sha256(&self.resolved_config_sha256)
            || !sha256(&self.native_policy_receipt_sha256)
            || !self
                .runtime_digest
                .strip_prefix("sha256:")
                .is_some_and(sha256)
            || self.state_layout_sha256.keys().copied().collect::<Vec<_>>()
                != (0..self.tp_size).collect::<Vec<_>>()
            || self
                .capture_registry_sha256
                .keys()
                .copied()
                .collect::<Vec<_>>()
                != (0..self.tp_size).collect::<Vec<_>>()
            || self.state_layout_sha256.values().any(|hash| !sha256(hash))
            || self.capture_registry_sha256.values().any(|hashes| {
                hashes.len() != self.capture_sizes.len() || hashes.iter().any(|hash| !sha256(hash))
            })
        {
            return Err(invalid(
                "native graph policy is outside the exact source/config/capture identity",
            ));
        }
        Ok(())
    }
    pub fn padded_batch(&self, batch: u32) -> Result<u32, AicError> {
        self.validate()?;
        if batch == 0 {
            return Err(invalid("native graph requires a positive actual batch"));
        }
        self.capture_sizes
            .iter()
            .copied()
            .find(|&size| {
                if self.disable_padding {
                    size == batch
                } else {
                    size >= batch
                }
            })
            .ok_or_else(|| {
                invalid("batch lies outside the initialized native graph capture policy")
            })
    }
}

/// Legacy mixed composition loses request boundaries before per-op dispatch.
/// It must not silently select the eager context list for a graph deployment.
pub(crate) fn reject_mixed(
    ops: &[Op],
    db: &crate::perf_database::PerfDatabase,
) -> Result<(), AicError> {
    if matches!(
        db.database_mode,
        crate::common::enums::DatabaseMode::Silicon | crate::common::enums::DatabaseMode::Hybrid
    ) && ops.iter().any(contains_glm)
        && (db.glm53flash_graph.has_measurements()?
            || db.glm53flash_graph.sglang_prefill.has_measurements()?)
    {
        return Err(invalid(
            "native graph Ops do not support the legacy mixed-step composition without actual homogeneous dispatch coordinates",
        ));
    }
    Ok(())
}

pub(super) fn contains_glm(op: &Op) -> bool {
    match op {
        Op::Glm53Attention(_)
        | Op::Glm53Mhc(_)
        | Op::Glm53Router(_)
        | Op::Glm53Ffn(_)
        | Op::Glm53Primitive(_)
        | Op::Glm53Runtime(_) => true,
        Op::Overlap(value) => value.group_a.iter().chain(&value.group_b).any(contains_glm),
        Op::Fallback(value) => {
            contains_glm(&value.primary) || value.fallback.iter().any(contains_glm)
        }
        Op::TokenScale(value) => contains_glm(&value.op),
        _ => false,
    }
}

pub(crate) fn validate_context_ops(
    ops: &[Op],
    db: &crate::perf_database::PerfDatabase,
) -> Result<bool, AicError> {
    if !matches!(
        db.database_mode,
        crate::common::enums::DatabaseMode::Silicon | crate::common::enums::DatabaseMode::Hybrid
    ) {
        return Ok(false);
    }
    if !ops.iter().any(contains_glm) {
        return Ok(false);
    }
    if db.glm53flash_graph.sglang_prefill.has_measurements()? {
        return db.glm53flash_graph.sglang_prefill.validate_ops(ops);
    }
    db.glm53flash_graph.validate_serving_ops(ops, true)
}

/// A serving model cannot erase an entire phase and receive a zero prediction.
/// Exact canonical model identity additionally protects a spec with both lists
/// erased. Model-less probes, legacy profiles, SOL and whole-forward FPM retain
/// their existing contracts; unknown names with no GLM ops carry no GLM identity.
pub(crate) fn validate_model_ops(
    spec: &crate::perfmodel::engine::spec::EngineSpec,
    db: &crate::perf_database::PerfDatabase,
) -> Result<(), AicError> {
    validate_context_ops(&spec.context_ops, db)?;
    validate_generation_ops(&spec.generation_ops, db)?;
    if !matches!(
        db.database_mode,
        crate::common::enums::DatabaseMode::Silicon | crate::common::enums::DatabaseMode::Hybrid
    ) || spec.engine.forward_model.as_deref() == Some("fpm")
    {
        return Ok(());
    }
    db.glm53flash_graph.validate_named_model(
        &spec.engine.model_name,
        spec.engine.parallel.tp_size,
        &spec.generation_ops,
    )?;
    let prefill = &db.glm53flash_graph.sglang_prefill;
    if prefill.has_measurements()?
        && (spec
            .context_ops
            .iter()
            .chain(&spec.generation_ops)
            .any(contains_glm)
            || prefill.matches_model(&spec.engine.model_name, spec.engine.parallel.tp_size)?)
    {
        // Schema4 prices context only, but its complete GLM model cannot erase
        // either compiled phase and obtain zero. Generation keeps its original
        // table/query semantics and requires its own measured coverage.
        prefill.validate_ops(&spec.context_ops)?;
        prefill.validate_generation_contract(&spec.generation_ops, &spec.context_ops)?;
    }
    let context = spec.context_ops.iter().any(contains_glm)
        && db
            .glm53flash_graph
            .validate_serving_ops(&spec.context_ops, true)?;
    let generation = spec.generation_ops.iter().any(contains_glm)
        && db
            .glm53flash_graph
            .validate_serving_ops(&spec.generation_ops, false)?;
    let known_model = db
        .glm53flash_graph
        .matches_serving_model(&spec.engine.model_name, spec.engine.parallel.tp_size)?;
    if (context || generation || known_model) && !(context && generation) {
        return Err(invalid(
            "schema3 GLM serving requires both complete context and generation operation lists; empty phases cannot predict zero",
        ));
    }
    Ok(())
}

/// Serialized specs predate the runtime marker and may also be hand-built.
/// A graph-backed phase must own setup exactly once, outside any fallback,
/// overlap or token-scaling wrapper that could omit or duplicate its cost.
pub(crate) fn validate_generation_ops(
    ops: &[Op],
    db: &crate::perf_database::PerfDatabase,
) -> Result<bool, AicError> {
    if !matches!(
        db.database_mode,
        crate::common::enums::DatabaseMode::Silicon | crate::common::enums::DatabaseMode::Hybrid
    ) || !ops.iter().any(contains_glm)
        || !db.glm53flash_graph.has_measurements()?
    {
        return Ok(false);
    }
    if db.glm53flash_graph.validate_serving_ops(ops, false)? {
        return Ok(true);
    }
    if let Some(profile) = db.glm53flash_graph.named_profile(ops)? {
        validate_named_ops(profile, ops)?;
    }
    let markers: Vec<_> = ops
        .iter()
        .filter_map(|op| match op {
            Op::Glm53Runtime(value) => Some(value),
            _ => None,
        })
        .collect();
    if markers.len() != 1 {
        return Err(invalid(
            "graph-backed GLM generation requires exactly one native_graph_setup marker; recompile old specs",
        ));
    }
    let marker = markers[0];
    marker.validate()?;
    if marker.is_context || ops.len() < 2 {
        return Err(invalid(
            "native graph setup marker must belong to a nonempty generation phase",
        ));
    }
    for op in ops {
        let (backend, format, tp, context) = match op {
            Op::Glm53Attention(value) => (
                &value.backend,
                &value.checkpoint_format,
                value.tp_size,
                value.is_context,
            ),
            Op::Glm53Mhc(value) => (
                &value.backend,
                &value.checkpoint_format,
                value.tp_size,
                value.is_context,
            ),
            Op::Glm53Ffn(value) => (
                &value.backend,
                &value.checkpoint_format,
                value.tp_size,
                value.is_context,
            ),
            Op::Glm53Primitive(value) => (
                &value.backend,
                &value.checkpoint_format,
                value.tp_size,
                value.is_context,
            ),
            Op::Glm53Runtime(value) => (
                &value.backend,
                &value.checkpoint_format,
                value.tp_size,
                value.is_context,
            ),
            _ => {
                return Err(invalid(
                    "native graph generation does not support mixed, nested or fallback operation lists",
                ));
            }
        };
        if context
            || backend != &marker.backend
            || format != &marker.checkpoint_format
            || tp != marker.tp_size
        {
            return Err(invalid(
                "native graph setup marker differs from its generation operation identity",
            ));
        }
    }
    Ok(true)
}

const GROUP_CONTRACT: &str = "sglang_named_graph_group_v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupExecutionPolicy {
    normalization: String,
    sha256: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupCompatibility {
    execution_policy: GroupExecutionPolicy,
    native_snapshot_sha256: String,
    state_layout_sha256: String,
    source_ownership_sha256: String,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupPoint {
    batch_size: u32,
    prefix: u32,
    padded_batch_size: u32,
    native_benchmark_id: u32,
    original_point_id: u32,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupMember {
    graph_policy: GraphPolicy,
    graph_policy_sha256: String,
    evidence_sha256: String,
    rank_selection_sha256: String,
    source_plan_sha256: String,
    native_runtime_run_id: String,
    control_sha256: String,
    points: Vec<GroupPoint>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphGroup {
    contract: String,
    source_plan_sha256: String,
    shard_manifest_sha256: String,
    corpus_sha256: String,
    compatibility: GroupCompatibility,
    members: BTreeMap<String, GroupMember>,
}
fn canonical_value<T: Serialize>(value: &T) -> Result<String, AicError> {
    serde_json::to_string(&serde_json::to_value(value).map_err(|e| invalid(e.to_string()))?)
        .map_err(|e| invalid(e.to_string()))
}
fn value_sha<T: Serialize>(value: &T) -> Result<String, AicError> {
    Ok(format!(
        "{:x}",
        Sha256::digest(canonical_value(value)?.as_bytes())
    ))
}
fn same_graph_execution(a: &GraphPolicy, b: &GraphPolicy) -> bool {
    a.schema_version == b.schema_version
        && a.backend == b.backend
        && a.backend_version == b.backend_version
        && a.backend_revision == b.backend_revision
        && a.checkpoint_format == b.checkpoint_format
        && a.checkpoint_revision == b.checkpoint_revision
        && a.tp_size == b.tp_size
        && a.phase == b.phase
        && a.runtime_mode == b.runtime_mode
        && a.capture_sizes == b.capture_sizes
        && a.disable_padding == b.disable_padding
        && a.captured_req_width == b.captured_req_width
        && a.native_flags == b.native_flags
        && a.source_pins == b.source_pins
        && a.source_sha256 == b.source_sha256
        && a.config_sha256 == b.config_sha256
        && a.runtime_digest == b.runtime_digest
}
impl GraphGroup {
    fn validate(&self) -> Result<(), AicError> {
        let c = &self.compatibility;
        if self.contract != GROUP_CONTRACT
            || self.members.is_empty()
            || ![
                &self.source_plan_sha256,
                &self.shard_manifest_sha256,
                &self.corpus_sha256,
                &c.execution_policy.sha256,
                &c.native_snapshot_sha256,
                &c.state_layout_sha256,
                &c.source_ownership_sha256,
            ]
            .iter()
            .all(|s| sha256(s))
            || !matches!(
                c.execution_policy.normalization.as_str(),
                "resolved_server_args_except_random_seed_v1"
                    | "resolved_server_args_and_native_allocator_v2"
            )
        {
            return Err(invalid(
                "graph group lacks its explicit complete compatibility contract",
            ));
        }
        let baseline = &self.members.values().next().unwrap().graph_policy;
        let mut runs = BTreeSet::new();
        let mut sources = BTreeSet::new();
        let mut controls = BTreeSet::new();
        let mut points = BTreeSet::new();
        let mut original_ids = BTreeSet::new();
        for (id, member) in &self.members {
            member.graph_policy.validate()?;
            if id.is_empty()
                || member.graph_policy.schema_version != 1
                || member.graph_policy.backend != "sglang"
                || !same_graph_execution(baseline, &member.graph_policy)
                || value_sha(&member.graph_policy)? != member.graph_policy_sha256
                || ![
                    &member.evidence_sha256,
                    &member.rank_selection_sha256,
                    &member.source_plan_sha256,
                    &member.control_sha256,
                ]
                .iter()
                .all(|s| sha256(s))
                || member.native_runtime_run_id.is_empty()
                || !runs.insert(&member.native_runtime_run_id)
                || !sources.insert(&member.source_plan_sha256)
                || !controls.insert(&member.control_sha256)
                || member.points.is_empty()
            {
                return Err(invalid(
                    "graph group changed or reused an original native member identity",
                ));
            }
            let mut native_ids = BTreeSet::new();
            for point in &member.points {
                if point.prefix == 0
                    || point.prefix >= 131072
                    || member.graph_policy.padded_batch(point.batch_size)?
                        != point.padded_batch_size
                    || point.native_benchmark_id == 0
                    || point.original_point_id == 0
                    || !native_ids.insert(point.native_benchmark_id)
                    || !original_ids.insert(point.original_point_id)
                    || !points.insert((point.batch_size, point.prefix, point.padded_batch_size))
                {
                    return Err(invalid(
                        "graph group duplicates or changes original native point ownership",
                    ));
                }
            }
            if native_ids.iter().copied().ne(1..=native_ids.len() as u32) {
                return Err(invalid("graph member omits an original native benchmark"));
            }
        }
        if original_ids
            .iter()
            .copied()
            .ne(1..=original_ids.len() as u32)
        {
            return Err(invalid("graph group omits an original parent point"));
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    name: Option<String>,
    component: String,
    geometry: String,
    batch: u32,
    prefix: u32,
    padded: u32,
}
struct Row {
    latency: f64,
    dispatch: String,
    activity_count: u32,
    evidence: String,
    rank_selection: String,
    member: Option<String>,
}
struct Profile {
    named: bool,
    units: BTreeSet<NamedUnit>,
    policy: GraphPolicy,
    rows: BTreeMap<Key, Row>,
    group: Option<Arc<GraphGroup>>,
    group_hash: Option<String>,
    group_selections: Mutex<BTreeMap<(u32, u32, u32), Result<Vec<(u32, f64)>, String>>>,
}
type Profiles = BTreeMap<(String, u32), Profile>;
pub struct Glm53GraphTable {
    serving: super::glm53flash_serving::ServingTable,
    sglang_prefill: super::glm53flash_sglang_prefill::PrefillTable,
    path: Option<PathBuf>,
    request: (String, String),
    profile: OnceLock<Result<Option<Profiles>, String>>,
}
impl Glm53GraphTable {
    pub fn with_sources(root: &Path, resolver: &SourceResolver) -> Result<Self, AicError> {
        let path = primary_path(root, resolver, BASENAME)?;
        let request: (String, String) = (
            root.parent()
                .and_then(Path::file_name)
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .into(),
            root.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .into(),
        );
        Ok(Self {
            sglang_prefill: super::glm53flash_sglang_prefill::PrefillTable::new(
                primary_path(root, resolver, super::glm53flash_sglang_prefill::BASENAME)?,
                request.clone(),
                root.parent()
                    .and_then(Path::parent)
                    .and_then(Path::file_name)
                    .is_some_and(|name| name == "gb300"),
            ),
            serving: super::glm53flash_serving::ServingTable::new(path.clone(), request),
            path,
            request: (
                root.parent()
                    .and_then(Path::file_name)
                    .and_then(|s| s.to_str())
                    .unwrap_or_default()
                    .into(),
                root.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default()
                    .into(),
            ),
            profile: OnceLock::new(),
        })
    }
    fn profile(&self) -> Result<Option<&Profiles>, AicError> {
        self.profile
            .get_or_init(|| match &self.path {
                Some(path) => {
                    load(path, &self.request).map_err(|e| format!("{}: {e}", path.display()))
                }
                None => Ok(None),
            })
            .as_ref()
            .map(Option::as_ref)
            .map_err(|e| invalid(e.clone()))
    }
    fn check_schema_identities(&self) -> Result<(), AicError> {
        if let Some(legacy) = self.profile()?
            && self
                .serving
                .identities()?
                .iter()
                .any(|identity| legacy.contains_key(identity))
        {
            return Err(invalid(
                "schema3 serving and legacy graph policy compete for the same checkpoint/TP",
            ));
        }
        Ok(())
    }
    fn named_profile(&self, ops: &[Op]) -> Result<Option<&Profile>, AicError> {
        let Some(first) = ops.first() else {
            return Ok(None);
        };
        if !contains_glm(first) {
            return Ok(None);
        }
        let (_, shape) = named_op(first)?;
        let identity = (
            shape["checkpoint_format"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            shape["tp_size"].as_u64().unwrap_or_default() as u32,
        );
        Ok(self
            .profile()?
            .and_then(|p| p.get(&identity))
            .filter(|p| p.named))
    }
    fn validate_named_model(&self, model: &str, tp: u32, ops: &[Op]) -> Result<(), AicError> {
        let format = match model {
            "zai-org/GLM-5.3-Flash" => "fp8",
            "nvidia/GLM-5.3-Flash-NVFP4" => "nvfp4",
            _ => return Ok(()),
        };
        if let Some(profile) = self.profile()?.and_then(|p| p.get(&(format.into(), tp)))
            && profile.named
        {
            validate_named_ops(profile, ops)?;
        }
        Ok(())
    }
    fn matches_serving_model(&self, model: &str, tp: u32) -> Result<bool, AicError> {
        let format = match model {
            "zai-org/GLM-5.3-Flash" => "fp8",
            "nvidia/GLM-5.3-Flash-NVFP4" => "nvfp4",
            _ => return Ok(false),
        };
        if self.request.0 != "vllm" {
            return Ok(false);
        }
        self.check_schema_identities()?;
        Ok(self.serving.identities()?.contains(&(format.into(), tp)))
    }
    pub(crate) fn requires_serving_context(&self, shape: &Value) -> Result<bool, AicError> {
        if self.sglang_prefill.requires_context(shape)? {
            return Ok(true);
        }
        self.check_schema_identities()?;
        let identity = (
            shape["checkpoint_format"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            shape["tp_size"].as_u64().unwrap_or_default() as u32,
        );
        Ok(shape["backend"] == "vllm" && self.serving.identities()?.contains(&identity))
    }
    fn validate_serving_ops(&self, ops: &[Op], context: bool) -> Result<bool, AicError> {
        self.check_schema_identities()?;
        self.serving.validate_ops(ops, context)
    }
    pub fn has_measurements(&self) -> Result<bool, AicError> {
        self.check_schema_identities()?;
        Ok(self.profile()?.is_some() || self.serving.has_measurements()?)
    }
    pub(crate) fn has_prefill_measurements(&self) -> Result<bool, AicError> {
        self.sglang_prefill.has_measurements()
    }
    /// Read-only evidence from the same selector that prices SG context units.
    pub(crate) fn glm53flash_lookup_audit(
        &self,
        context: &[Op],
        generation: &[Op],
        is_context: bool,
        point: (u32, u32, u32),
    ) -> Result<Value, AicError> {
        self.check_schema_identities()?;
        if is_context && self.sglang_prefill.has_measurements()? {
            self.sglang_prefill.audit(context, generation, point)
        } else if !is_context && let Some(profile) = self.named_profile(generation)? {
            validate_named_ops(profile, generation)?;
            let (batch, query, prefix) = point;
            if query != 1 || prefix == 0 || prefix >= 131072 {
                return Err(invalid(
                    "named graph audit requires homogeneous one-token decode",
                ));
            }
            let mut operations = Vec::new();
            let policy_hash = format!(
                "{:x}",
                Sha256::digest(
                    serde_json::to_string(
                        &serde_json::to_value(&profile.policy)
                            .map_err(|e| invalid(e.to_string()))?
                    )
                    .map_err(|e| invalid(e.to_string()))?
                    .as_bytes()
                )
            );
            for op in generation {
                let (unit, _) = named_op(op)?;
                let target = Key {
                    name: Some(unit.1.clone()),
                    component: unit.0,
                    geometry: unit.2,
                    batch,
                    prefix,
                    padded: profile.policy.padded_batch(batch)?,
                };
                let selected = select_profile_rows(profile, &target)?;
                operations.push(serde_json::json!({"operation_name":op.name(),"geometry":target.geometry,
                    "latency_ms":selected.iter().map(|(_, r, w)| r.latency * w).sum::<f64>(),
                    "endpoints":selected.iter().map(|(key,row,weight)| graph_endpoint(profile, key, row, *weight)).collect::<Vec<_>>()}));
            }
            if operations.is_empty() {
                return Err(invalid("named graph audit requires complete generation"));
            }
            let mut audit = serde_json::json!({"schema":"glm53flash_lookup_audit_v1","lookup_contract":NAMED_CONTRACT,
                "phase":"generation","target":{"batch_size":batch,"query_length":query,"prefix":prefix},"operations":operations});
            if let Some(hash) = &profile.group_hash {
                audit["group_contract"] = serde_json::json!(GROUP_CONTRACT);
                audit["graph_group_sha256"] = serde_json::json!(hash);
            } else {
                audit["graph_policy_sha256"] = serde_json::json!(policy_hash);
            }
            Ok(audit)
        } else {
            self.serving.audit(context, generation, is_context, point)
        }
    }

    pub fn query(
        &self,
        op: &Op,
        ctx: &RuntimeContext,
    ) -> Result<Option<PerformanceResult>, AicError> {
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
        validate_runtime(&self.request.0, &self.request.1)?;
        self.check_schema_identities()?;
        if let Some(value) = self.sglang_prefill.query(op, ctx)? {
            return Ok(Some(value));
        }
        if let Some(value) = self.serving.query(op, ctx)? {
            return Ok(Some(value));
        }
        let shape = shape.map_err(|e| invalid(e.to_string()))?;
        let serving_selected = self.serving.has_measurements()?;
        let identity = (
            shape["checkpoint_format"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            shape["tp_size"].as_u64().unwrap_or_default() as u32,
        );
        if serving_selected
            && !self
                .profile()?
                .is_some_and(|profiles| profiles.contains_key(&identity))
        {
            return Err(invalid(
                "selected native serving dataset lacks this checkpoint/TP policy",
            ));
        }
        if shape["is_context"] == true {
            return Ok(None);
        }
        let Some(profiles) = self.profile()? else {
            return Ok(None);
        };
        let identity = (
            shape["checkpoint_format"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            shape["tp_size"].as_u64().unwrap_or_default() as u32,
        );
        let profile = profiles.get(&identity).ok_or_else(|| {
            invalid("selected native graph dataset lacks this checkpoint/TP policy")
        })?;
        if shape["backend"] != profile.policy.backend
            || shape["checkpoint_format"] != profile.policy.checkpoint_format
            || shape["tp_size"] != profile.policy.tp_size
        {
            return Err(invalid(
                "graph table does not match the requested native model identity",
            ));
        }
        if ctx.beam_width != 1
            || ctx.prefix != 0
            || ctx.batch_size != ctx.num_tokens
            || ctx.s <= 1
            || ctx.s > 131072
            || ctx.seq_imbalance_correction_scale != 1.0
            || ctx.gen_seq_imbalance_correction_scale != 1.0
        {
            return Err(invalid(
                "graph query requires complete homogeneous one-token decode RuntimeContext; mixed/token-only queries are unsupported",
            ));
        }
        let target = Key {
            name: profile.named.then(|| op.name().into()),
            component: component.into(),
            geometry: geometry(&shape)?,
            batch: ctx.batch_size,
            // RuntimeContext.s is inclusive of the current decode token.
            // The native measurement axis excludes it (ScheduleBatch
            // prepare_for_decode at the pinned source increments seq_lens).
            prefix: ctx
                .s
                .checked_sub(1)
                .ok_or_else(|| invalid("native graph decode position must be positive"))?,
            padded: profile.policy.padded_batch(ctx.batch_size)?,
        };
        let latency = select_profile_rows(profile, &target)?
            .iter()
            .map(|(_, row, weight)| row.latency * weight)
            .sum();
        Ok(Some(PerformanceResult::with_energy(
            latency,
            0.0,
            Source::Silicon,
        )))
    }
}

fn expected_names(backend: &str) -> BTreeSet<(String, String)> {
    let mut names: BTreeSet<_> = [
        ("primitive", "embedding"),
        ("primitive", "embedding_allreduce"),
        ("primitive", "final_norm"),
        ("primitive", "logits"),
        ("mhc", "mhc_expand"),
        ("mhc", "mhc_contract"),
        ("runtime", "native_graph_setup"),
    ]
    .into_iter()
    .map(|(c, n)| (c.into(), n.into()))
    .collect();
    for i in 0..45 {
        for (c, n) in [
            ("attention", "attention"),
            ("ffn", "ffn"),
            ("primitive", "attention_allreduce"),
            ("primitive", "ffn_allreduce"),
        ] {
            names.insert((c.into(), format!("{n}_{i}")));
        }
        if backend == "sglang" {
            for n in [
                "mhc_pre_attn",
                "mhc_post_attn",
                "mhc_pre_ffn",
                "mhc_post_ffn",
            ] {
                names.insert(("mhc".into(), format!("{n}_{i}")));
            }
        } else {
            names.insert((
                "mhc".into(),
                if i == 0 {
                    "mhc_pre_attn_0".into()
                } else {
                    format!("mhc_fused_attn_{i}")
                },
            ));
            names.insert(("mhc".into(), format!("mhc_fused_ffn_{i}")));
        }
    }
    if backend == "vllm" {
        names.insert(("mhc".into(), "mhc_post_ffn_44".into()));
    }
    names
}
fn validate_named_unit(
    backend: &str,
    component: &str,
    name: &str,
    shape: &Value,
) -> Result<(), AicError> {
    if !expected_names(backend).contains(&(component.into(), name.into())) {
        return Err(invalid("unknown named graph operation"));
    }
    let role = if component == "primitive" {
        Some(if name.contains("allreduce") {
            "allreduce"
        } else {
            name
        })
    } else if component == "mhc" {
        Some(if name == "mhc_expand" {
            "expand"
        } else if name == "mhc_contract" {
            "contract"
        } else if name.starts_with("mhc_fused_") {
            "fused_post_pre"
        } else if name.starts_with("mhc_pre_") {
            "pre"
        } else {
            "post"
        })
    } else {
        None
    };
    if role.is_some_and(|r| shape["role"] != r) {
        return Err(invalid(
            "named graph role differs from original call identity",
        ));
    }
    if matches!(component, "attention" | "ffn") {
        let index: u32 = name
            .rsplit('_')
            .next()
            .unwrap_or_default()
            .parse()
            .map_err(|_| invalid("invalid graph layer name"))?;
        if (component == "attention"
            && shape["layer_kind"] != if index % 4 == 3 { "sparse_mla" } else { "kda" })
            || (component == "ffn" && shape["is_dense"] != (index < 3))
        {
            return Err(invalid(
                "named graph layer geometry differs from native model topology",
            ));
        }
    }
    Ok(())
}
fn named_op(op: &Op) -> Result<(NamedUnit, Value), AicError> {
    let (component, shape) = match op {
        Op::Glm53Attention(o) => ("attention", serde_json::to_value(o)),
        Op::Glm53Mhc(o) => ("mhc", serde_json::to_value(o)),
        Op::Glm53Ffn(o) => ("ffn", serde_json::to_value(o)),
        Op::Glm53Primitive(o) => ("primitive", serde_json::to_value(o)),
        Op::Glm53Runtime(o) => ("runtime", serde_json::to_value(o)),
        _ => {
            return Err(invalid(
                "named graph requires unwrapped original physical operations",
            ));
        }
    };
    let shape = shape.map_err(|e| invalid(e.to_string()))?;
    Ok((
        (component.into(), op.name().into(), geometry(&shape)?),
        shape,
    ))
}
fn validate_named_ops(profile: &Profile, ops: &[Op]) -> Result<(), AicError> {
    let units = ops
        .iter()
        .map(|op| named_op(op).map(|(u, _)| u))
        .collect::<Result<BTreeSet<_>, _>>()?;
    if !profile.named || ops.len() != units.len() || units != profile.units {
        return Err(invalid(
            "named graph generation differs from complete measured operation names/geometries",
        ));
    }
    Ok(())
}

fn partition(key: &Key) -> Result<Vec<u32>, AicError> {
    let shape: Value = serde_json::from_str(&key.geometry).map_err(|e| invalid(e.to_string()))?;
    Ok(
        if key.component == "attention" && shape["layer_kind"] == "sparse_mla" {
            vec![
                key.prefix % 4,
                (key.prefix + 1) % 4,
                u32::from(key.prefix + 1 <= 2048),
                u32::from(key.prefix == 0),
            ]
        } else {
            vec![]
        },
    )
}
fn group_points(
    profile: &Profile,
    batch: u32,
    prefix: u32,
    padded: u32,
) -> Result<Vec<(u32, f64)>, AicError> {
    let group = profile
        .group
        .as_ref()
        .ok_or_else(|| invalid("graph group is absent"))?;
    let points: BTreeSet<u32> = group
        .members
        .values()
        .flat_map(|m| &m.points)
        .filter(|p| p.batch_size == batch && p.padded_batch_size == padded)
        .map(|p| p.prefix)
        .collect();
    if points.contains(&prefix) {
        return Ok(vec![(prefix, 1.0)]);
    }
    let mut pairs = Vec::new();
    for low in points.iter().filter(|p| **p < prefix) {
        for high in points.iter().filter(|p| **p > prefix) {
            pairs.push((*high - *low, *low, *high));
        }
    }
    pairs.sort();
    for (_, low, high) in pairs {
        let mut compatible = true;
        for (component, name, geometry) in &profile.units {
            let target = Key {
                name: Some(name.clone()),
                component: component.clone(),
                geometry: geometry.clone(),
                batch,
                prefix,
                padded,
            };
            let lk = Key {
                prefix: low,
                ..Key {
                    name: target.name.clone(),
                    component: target.component.clone(),
                    geometry: target.geometry.clone(),
                    batch,
                    prefix,
                    padded,
                }
            };
            let hk = Key {
                prefix: high,
                ..Key {
                    name: target.name.clone(),
                    component: target.component.clone(),
                    geometry: target.geometry.clone(),
                    batch,
                    prefix,
                    padded,
                }
            };
            let wanted = partition(&target)?;
            let (Some(left), Some(right)) = (profile.rows.get(&lk), profile.rows.get(&hk)) else {
                compatible = false;
                break;
            };
            if partition(&lk)? != wanted
                || partition(&hk)? != wanted
                || left.dispatch != right.dispatch
                || left.activity_count != right.activity_count
            {
                compatible = false;
                break;
            }
        }
        if compatible {
            let weight = f64::from(prefix - low) / f64::from(high - low);
            return Ok(vec![(low, 1.0 - weight), (high, weight)]);
        }
    }
    Err(invalid(
        "graph group lacks complete same-pad compatible measured point brackets",
    ))
}
fn select_profile_rows<'a>(
    profile: &'a Profile,
    target: &Key,
) -> Result<Vec<(&'a Key, &'a Row, f64)>, AicError> {
    if profile.group.is_none() {
        return select_rows(&profile.rows, target);
    }
    let coordinates = (target.batch, target.prefix, target.padded);
    let selected = profile
        .group_selections
        .lock()
        .map_err(|_| invalid("graph group selection lock failed"))?
        .entry(coordinates)
        .or_insert_with(|| {
            group_points(profile, coordinates.0, coordinates.1, coordinates.2)
                .map_err(|e| e.to_string())
        })
        .clone()
        .map_err(invalid)?;
    selected
        .into_iter()
        .map(|(prefix, weight)| {
            let key = Key {
                name: target.name.clone(),
                component: target.component.clone(),
                geometry: target.geometry.clone(),
                batch: target.batch,
                prefix,
                padded: target.padded,
            };
            profile
                .rows
                .get_key_value(&key)
                .map(|(k, r)| (k, r, weight))
                .ok_or_else(|| invalid("graph group selected endpoint lacks this named unit"))
        })
        .collect()
}
fn graph_endpoint(profile: &Profile, key: &Key, row: &Row, weight: f64) -> Value {
    let mut value = serde_json::json!({"batch_size":key.batch,"prefix":key.prefix,"padded_batch_size":key.padded,
        "latency_ms":row.latency,"weight":weight,"evidence_sha256":row.evidence,
        "rank_selection_sha256":row.rank_selection,"dispatch_fingerprint":row.dispatch,"activity_count":row.activity_count});
    if let (Some(group), Some(id)) = (&profile.group, &row.member) {
        let member = &group.members[id];
        let point = member
            .points
            .iter()
            .find(|p| {
                (p.batch_size, p.prefix, p.padded_batch_size) == (key.batch, key.prefix, key.padded)
            })
            .expect("validated member point");
        let object = value.as_object_mut().unwrap();
        object.insert("graph_member_id".into(), serde_json::json!(id));
        object.insert(
            "graph_policy_sha256".into(),
            serde_json::json!(member.graph_policy_sha256),
        );
        object.insert(
            "native_runtime_run_id".into(),
            serde_json::json!(member.native_runtime_run_id),
        );
        object.insert(
            "native_benchmark_id".into(),
            serde_json::json!(point.native_benchmark_id),
        );
        object.insert(
            "original_point_id".into(),
            serde_json::json!(point.original_point_id),
        );
    }
    value
}

fn select_rows<'a>(
    rows: &'a BTreeMap<Key, Row>,
    target: &Key,
) -> Result<Vec<(&'a Key, &'a Row, f64)>, AicError> {
    if let Some((key, row)) = rows.get_key_value(target) {
        return Ok(vec![(key, row, 1.0)]);
    }
    let wanted = partition(target)?;
    let mut groups: BTreeMap<&str, Vec<(&Key, &Row)>> = BTreeMap::new();
    for (key, row) in rows {
        if key.name == target.name
            && key.component == target.component
            && key.geometry == target.geometry
            && key.batch == target.batch
            && key.padded == target.padded
            && partition(key)? == wanted
        {
            groups.entry(&row.dispatch).or_default().push((key, row));
        }
    }
    let mut answer = None;
    for group in groups.values() {
        let lo = group
            .iter()
            .filter(|(k, _)| k.prefix < target.prefix)
            .max_by_key(|(k, _)| k.prefix);
        let hi = group
            .iter()
            .filter(|(k, _)| k.prefix > target.prefix)
            .min_by_key(|(k, _)| k.prefix);
        if let (Some((lk, lv)), Some((hk, hv))) = (lo, hi) {
            if lv.activity_count != hv.activity_count {
                continue;
            }
            if answer.is_some() {
                return Err(invalid("ambiguous native graph dispatch brackets"));
            }
            let weight = f64::from(target.prefix - lk.prefix) / f64::from(hk.prefix - lk.prefix);
            answer = Some(vec![(*lk, *lv, 1.0 - weight), (*hk, *hv, weight)]);
        }
    }
    answer.ok_or_else(|| {
        invalid("native graph unit lacks same-policy, same-pad measured history brackets")
    })
}
fn load(path: &Path, request: &(String, String)) -> Result<Option<Profiles>, AicError> {
    if !path.try_exists().map_err(|e| invalid(e.to_string()))? {
        return Ok(None);
    }
    let reader = PerfReader::open(path)?;
    let policy_col = reader.col("graph_policy")?;
    let mut has_legacy = false;
    let mut any_row = false;
    for row in reader.rows()? {
        let row = row?;
        any_row = true;
        let value: Value =
            serde_json::from_str(row.str(policy_col)?).map_err(|e| invalid(e.to_string()))?;
        match value["schema_version"].as_u64() {
            Some(1 | 2) => has_legacy = true,
            Some(3) => {}
            _ => return Err(invalid("unknown native graph/serving policy schema")),
        }
    }
    if !any_row {
        return Err(invalid("empty native graph table is not a serving policy"));
    }
    if !has_legacy {
        return Ok(None);
    }
    let names = [
        "component",
        "geometry",
        "batch_size",
        "prefix",
        "padded_batch_size",
        "latency",
        "activity_count",
        "sample_count",
        "dispatch_fingerprint",
        "graph_policy",
        "graph_policy_sha256",
        "dataset_role",
        "aggregation_policy",
        "rank_selection_sha256",
        "evidence_sha256",
        "measurement_scope",
    ];
    let cols = names
        .iter()
        .map(|name| reader.col(name))
        .collect::<Result<Vec<_>, _>>()?;
    let named_col = reader.col_optional("operation_name");
    let contract_col = reader.col_optional("graph_lookup_contract");
    let group_col = reader.col_optional("graph_group");
    let group_hash_col = reader.col_optional("graph_group_sha256");
    let member_col = reader.col_optional("graph_member_id");
    let mut groups: BTreeMap<String, (String, Arc<GraphGroup>)> = BTreeMap::new();
    let mut profiles = Profiles::new();
    let mut point_evidence = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let text = row.str(cols[9])?;
        let schema: Value = serde_json::from_str(text).map_err(|e| invalid(e.to_string()))?;
        if schema["schema_version"] == 3 {
            continue;
        }
        let parsed: GraphPolicy = serde_json::from_str(text).map_err(|e| invalid(e.to_string()))?;
        parsed.validate()?;
        let canonical = serde_json::to_string(
            &serde_json::to_value(&parsed).map_err(|e| invalid(e.to_string()))?,
        )
        .map_err(|e| invalid(e.to_string()))?;
        if text != canonical
            || format!("{:x}", Sha256::digest(text.as_bytes())) != row.str(cols[10])?
            || request != &(parsed.backend.clone(), parsed.backend_version.clone())
        {
            return Err(invalid(
                "native graph table mixes policy identities or noncanonical source evidence",
            ));
        }
        let name = row.str_optional(named_col)?;
        let contract = row.str_optional(contract_col)?;
        let named = match (contract, name) {
            (None, None) => false,
            (Some(NAMED_CONTRACT), Some(value)) if !value.is_empty() => true,
            _ => {
                return Err(invalid(
                    "graph operation names require the exact analysis lookup contract",
                ));
            }
        };
        let group_text = row.str_optional(group_col)?;
        let group_hash = row.str_optional(group_hash_col)?;
        let member_id = row.str_optional(member_col)?;
        let group = match (group_text, group_hash, member_id) {
            (None, None, None) => None,
            (Some(text), Some(hash), Some(id)) if named => {
                if !groups.contains_key(hash) {
                    let value: GraphGroup =
                        serde_json::from_str(text).map_err(|e| invalid(e.to_string()))?;
                    value.validate()?;
                    if canonical_value(&value)? != text || value_sha(&value)? != hash {
                        return Err(invalid(
                            "graph group has noncanonical or changed analysis metadata",
                        ));
                    }
                    groups.insert(hash.into(), (text.into(), Arc::new(value)));
                }
                let (encoded, group) = &groups[hash];
                let member = group
                    .members
                    .get(id)
                    .ok_or_else(|| invalid("graph row names an undeclared group member"))?;
                if encoded != text
                    || member.graph_policy != parsed
                    || member.graph_policy_sha256 != row.str(cols[10])?
                    || member.evidence_sha256 != row.str(cols[14])?
                    || member.rank_selection_sha256 != row.str(cols[13])?
                {
                    return Err(invalid(
                        "graph row changed original member policy or evidence",
                    ));
                }
                Some(Arc::clone(group))
            }
            _ => {
                return Err(invalid(
                    "graph group requires complete named analysis metadata",
                ));
            }
        };
        let identity = (parsed.checkpoint_format.clone(), parsed.tp_size);
        if profiles.get(&identity).is_some_and(|previous| {
            previous.named != named
                || previous.group_hash.as_deref() != group_hash
                || (group.is_none() && previous.policy != parsed)
        }) {
            return Err(invalid(
                "native graph identity has competing capture/runtime policies",
            ));
        }
        let component = row.str(cols[0])?;
        let encoded = row.str(cols[1])?;
        let shape = if component == "runtime" {
            let mut value: Value =
                serde_json::from_str(encoded).map_err(|e| invalid(e.to_string()))?;
            value
                .as_object_mut()
                .ok_or_else(|| invalid("runtime geometry must be object"))?
                .insert("name".into(), "native_graph_setup".into());
            let op: crate::operators::Glm53RuntimeOp =
                serde_json::from_value(value).map_err(|e| invalid(e.to_string()))?;
            op.validate()?;
            if geometry(&op)? != encoded {
                return Err(invalid("noncanonical runtime setup geometry"));
            }
            serde_json::to_value(op).map_err(|e| invalid(e.to_string()))?
        } else {
            validate_geometry(component, encoded)?
        };
        if shape["is_context"] != false
            || shape["backend"] != parsed.backend
            || shape["checkpoint_format"] != parsed.checkpoint_format
            || shape["tp_size"] != parsed.tp_size
        {
            return Err(invalid(
                "graph row differs from its physical execution policy",
            ));
        }
        if let Some(name) = name {
            validate_named_unit(&parsed.backend, component, name, &shape)?;
        }
        let batch = row.u32(cols[2])?;
        let prefix = row.u32(cols[3])?;
        let padded = row.u32(cols[4])?;
        let latency = row.f64(cols[5])?;
        let count = row.u32(cols[6])?;
        if padded != parsed.padded_batch(batch)?
            || prefix == 0
            || prefix >= 131072
            || !latency.is_finite()
            || latency < 0.0
            || (count == 0) != (latency == 0.0)
            || row.u32(cols[7])? < 10
            || (named && row.u32(cols[7])? != 10)
            || !sha256(row.str(cols[8])?)
            || row.str(cols[11])? != "calibration"
            || row.str(cols[12])? != "whole_forward_slowest_rank_v1"
            || !sha256(row.str(cols[13])?)
            || !sha256(row.str(cols[14])?)
            || row.str(cols[15])? != "disjoint_native_node_activity_union_v1"
        {
            return Err(invalid(
                "native graph row lacks complete physical work, timing or calibration evidence",
            ));
        }
        if let (Some(group), Some(id)) = (&group, member_id)
            && !group.members[id]
                .points
                .iter()
                .any(|p| (p.batch_size, p.prefix, p.padded_batch_size) == (batch, prefix, padded))
        {
            return Err(invalid(
                "graph row is outside its original member point ownership",
            ));
        }
        let evidence_key = (identity.clone(), batch, prefix, padded);
        let evidence_value = (
            row.str(cols[13])?.to_owned(),
            row.str(cols[14])?.to_owned(),
            row.u32(cols[7])?,
            member_id.map(str::to_owned),
        );
        if point_evidence
            .insert(evidence_key, evidence_value.clone())
            .is_some_and(|old| old != evidence_value)
        {
            return Err(invalid(
                "native graph point mixes rank-selection/sample/evidence receipts across operation units",
            ));
        }
        let key = Key {
            name: name.map(str::to_owned),
            component: component.into(),
            geometry: encoded.into(),
            batch,
            prefix,
            padded,
        };
        let profile = profiles.entry(identity).or_insert_with(|| Profile {
            named,
            units: BTreeSet::new(),
            policy: parsed.clone(),
            rows: BTreeMap::new(),
            group: group.clone(),
            group_hash: group_hash.map(str::to_owned),
            group_selections: Mutex::new(BTreeMap::new()),
        });
        if profile
            .rows
            .insert(
                key,
                Row {
                    latency,
                    dispatch: row.str(cols[8])?.into(),
                    activity_count: count,
                    evidence: row.str(cols[14])?.into(),
                    rank_selection: row.str(cols[13])?.into(),
                    member: member_id.map(str::to_owned),
                },
            )
            .is_some()
        {
            return Err(invalid("duplicate native graph physical key"));
        }
    }
    if profiles.is_empty() {
        return Err(invalid("empty native graph table is not a serving policy"));
    }
    for profile in profiles.values_mut() {
        if let Some(group) = &profile.group {
            let declared: BTreeSet<_> = group
                .members
                .iter()
                .flat_map(|(id, m)| {
                    m.points
                        .iter()
                        .map(move |p| (id.as_str(), p.batch_size, p.prefix, p.padded_batch_size))
                })
                .collect();
            let observed: BTreeSet<_> = profile
                .rows
                .iter()
                .map(|(k, r)| {
                    (
                        r.member.as_deref().expect("group member"),
                        k.batch,
                        k.prefix,
                        k.padded,
                    )
                })
                .collect();
            if declared != observed {
                return Err(invalid(
                    "graph group omits declared original members or points",
                ));
            }
        }
        if profile.named {
            let mut points: BTreeMap<(u32, u32, u32), BTreeSet<NamedUnit>> = BTreeMap::new();
            for key in profile.rows.keys() {
                points
                    .entry((key.batch, key.prefix, key.padded))
                    .or_default()
                    .insert((
                        key.component.clone(),
                        key.name.clone().expect("named key"),
                        key.geometry.clone(),
                    ));
            }
            for units in points.values() {
                let actual: BTreeSet<_> = units
                    .iter()
                    .map(|(c, n, _)| (c.clone(), n.clone()))
                    .collect();
                if actual != expected_names(&profile.policy.backend)
                    || actual.len() != units.len()
                    || (!profile.units.is_empty() && &profile.units != units)
                {
                    return Err(invalid(
                        "named graph point lacks complete consistent original operation inventory",
                    ));
                }
                profile.units = units.clone();
            }
        }
        for key in profile.rows.keys() {
            if !profile.rows.keys().any(|setup| {
                setup.component == "runtime"
                    && setup.batch == key.batch
                    && setup.prefix == key.prefix
                    && setup.padded == key.padded
            }) {
                return Err(invalid("native graph point lacks measured setup"));
            }
        }
    }
    Ok(Some(profiles))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::common::enums::{DatabaseMode, TransferPolicy};
    use crate::config::PerfDbSources;
    use crate::operators::{Glm53MhcOp, Glm53RuntimeOp};
    use crate::perf_database::PerfDatabase;
    use crate::perf_database::energy_test_fixtures::{
        Col, write_energy_systems_root, write_parquet,
    };
    const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    fn policy() -> GraphPolicy {
        GraphPolicy {
            schema_version: 1,
            backend: "sglang".into(),
            backend_version: "0.5.20".into(),
            backend_revision: SG_REVISION.into(),
            checkpoint_format: "fp8".into(),
            checkpoint_revision: "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a".into(),
            tp_size: 2,
            phase: "generation".into(),
            runtime_mode: "FULL".into(),
            capture_sizes: vec![1, 2, 4],
            disable_padding: false,
            captured_req_width: 1,
            native_flags: [
                "enable_torch_compile",
                "is_encoder_decoder",
                "require_mlp_tp_gather",
                "require_attn_tp_gather",
                "require_mlp_sync",
                "enable_two_batch_overlap",
                "enable_pdmux",
                "ragged_verify_mode",
                "enable_prefill_cp",
                "enable_lora",
                "speculative_enabled",
                "metadata_glue_enabled",
                "reuse_output_buffer",
            ]
            .into_iter()
            .map(|key| (key.into(), false))
            .collect(),
            source_pins: BTreeMap::from([
                (
                    "srt/model_executor/runner/decode_cuda_graph_runner.py".into(),
                    SG_DECODE.into(),
                ),
                (
                    "srt/model_executor/runner/base_cuda_graph_runner.py".into(),
                    SG_BASE.into(),
                ),
                (
                    "srt/model_executor/runner_backend/full_cuda_graph_backend.py".into(),
                    SG_FULL.into(),
                ),
                (
                    "srt/model_executor/runner/shape_key.py".into(),
                    SG_SHAPE.into(),
                ),
            ]),
            source_sha256: SG_SOURCE.into(),
            config_sha256: SHA.into(),
            runtime_digest: format!("sha256:{SHA}"),
            resolved_config_sha256: SHA.into(),
            native_policy_receipt_sha256: SHA.into(),
            state_layout_sha256: BTreeMap::from([(0, SHA.into()), (1, SHA.into())]),
            capture_registry_sha256: BTreeMap::from([
                (0, vec![SHA.into(); 3]),
                (1, vec![SHA.into(); 3]),
            ]),
        }
    }
    pub(crate) fn marker() -> Op {
        Op::Glm53Runtime(Glm53RuntimeOp {
            name: "native_graph_setup".into(),
            backend: "sglang".into(),
            checkpoint_format: "fp8".into(),
            tp_size: 2,
            is_context: false,
        })
    }
    pub(crate) fn mhc() -> Op {
        Op::Glm53Mhc(Glm53MhcOp {
            name: "mhc_pre_attn_0".into(),
            backend: "sglang".into(),
            checkpoint_format: "fp8".into(),
            tp_size: 2,
            is_context: false,
            role: "pre".into(),
            hidden_size: 4096,
            hc_mult: 4,
            sinkhorn_iters: 20,
        })
    }
    fn body(op: &Op) -> Value {
        serde_json::to_value(op)
            .unwrap()
            .as_object()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .clone()
    }
    fn literal(s: String) -> &'static str {
        Box::leak(s.into_boxed_str())
    }
    pub(crate) fn fixture() -> Vec<Col> {
        let text =
            literal(serde_json::to_string(&serde_json::to_value(policy()).unwrap()).unwrap());
        let hash = literal(format!("{:x}", Sha256::digest(text.as_bytes())));
        let setup = literal(geometry(&body(&marker())).unwrap());
        let mhc = literal(geometry(&body(&mhc())).unwrap());
        vec![
            Col::Str("component", vec!["runtime", "runtime", "mhc", "mhc"]),
            Col::Str("geometry", vec![setup, setup, mhc, mhc]),
            Col::I64("batch_size", vec![3; 4]),
            Col::I64("prefix", vec![129, 137, 129, 137]),
            Col::I64("padded_batch_size", vec![4; 4]),
            Col::F64("latency", vec![1.0, 3.0, 3.0, 5.0]),
            Col::I64("activity_count", vec![1; 4]),
            Col::I64("sample_count", vec![10; 4]),
            Col::Str("dispatch_fingerprint", vec![SHA; 4]),
            Col::Str("graph_policy", vec![text; 4]),
            Col::Str("graph_policy_sha256", vec![hash; 4]),
            Col::Str("dataset_role", vec!["calibration"; 4]),
            Col::Str(
                "aggregation_policy",
                vec!["whole_forward_slowest_rank_v1"; 4],
            ),
            Col::Str("rank_selection_sha256", vec![SHA; 4]),
            Col::Str("evidence_sha256", vec![SHA; 4]),
            Col::Str(
                "measurement_scope",
                vec!["disjoint_native_node_activity_union_v1"; 4],
            ),
        ]
    }
    pub(crate) fn db(root: &Path) -> PerfDatabase {
        let _ = write_energy_systems_root(root);
        let path = root.join("data/sglang/0.5.20");
        std::fs::create_dir_all(&path).unwrap();
        write_parquet(&path.join(BASENAME), &fixture());
        PerfDatabase::load(root, "testsys", "sglang", "0.5.20")
            .unwrap()
            .with_mode(DatabaseMode::Silicon, TransferPolicy::ALL)
    }
    #[test]
    fn named_selector_preserves_collective_occurrence_and_original_endpoints() {
        let mut rows = BTreeMap::new();
        for (name, low, high) in [
            ("embedding_allreduce", 0.8, 1.6),
            ("attention_allreduce_0", 0.004, 0.008),
        ] {
            for (prefix, latency) in [(128, low), (136, high)] {
                rows.insert(
                    Key {
                        name: Some(name.into()),
                        component: "primitive".into(),
                        geometry: "{}".into(),
                        batch: 1,
                        prefix,
                        padded: 1,
                    },
                    Row {
                        latency,
                        dispatch: SHA.into(),
                        activity_count: 1,
                        evidence: format!("{prefix}"),
                        rank_selection: SHA.into(),
                        member: None,
                    },
                );
            }
        }
        let target = Key {
            name: Some("embedding_allreduce".into()),
            component: "primitive".into(),
            geometry: "{}".into(),
            batch: 1,
            prefix: 132,
            padded: 1,
        };
        let chosen = select_rows(&rows, &target).unwrap();
        assert_eq!(
            chosen
                .iter()
                .map(|(k, _, w)| (k.prefix, *w))
                .collect::<Vec<_>>(),
            vec![(128, 0.5), (136, 0.5)]
        );
        assert!((chosen.iter().map(|(_, r, w)| r.latency * w).sum::<f64>() - 1.2).abs() < 1e-12);
        assert_eq!(chosen[0].1.evidence, "128");
        let missing = Key {
            name: Some("ffn_allreduce_0".into()),
            ..target
        };
        assert!(select_rows(&rows, &missing).is_err());
        let legacy = Key {
            name: None,
            ..missing
        };
        assert!(select_rows(&rows, &legacy).is_err());
    }
    #[test]
    fn named_inventory_retains_native_fusion_and_layer_role_identity() {
        assert_eq!(expected_names("sglang").len(), 367);
        assert_eq!(expected_names("vllm").len(), 278);
        assert!(
            validate_named_unit(
                "sglang",
                "primitive",
                "embedding_allreduce",
                &serde_json::json!({"role":"allreduce"})
            )
            .is_ok()
        );
        assert!(
            validate_named_unit(
                "sglang",
                "primitive",
                "embedding_allreduce",
                &serde_json::json!({"role":"logits"})
            )
            .is_err()
        );
        assert!(
            validate_named_unit(
                "vllm",
                "mhc",
                "mhc_post_attn_0",
                &serde_json::json!({"role":"post"})
            )
            .is_err()
        );
        assert!(
            validate_named_unit(
                "vllm",
                "mhc",
                "mhc_fused_attn_1",
                &serde_json::json!({"role":"fused_post_pre"})
            )
            .is_ok()
        );
        assert!(
            validate_named_unit(
                "sglang",
                "attention",
                "attention_3",
                &serde_json::json!({"layer_kind":"kda"})
            )
            .is_err()
        );
    }
    #[test]
    fn vllm_full_table_requires_exact_runtime_and_preserves_padding_setup_geometry() {
        for (version, source) in [
            ("0.30.0", VLLM_STOCK_SOURCE),
            ("0.30.0+glm53kpool.bf5f6b0e689d", VLLM_REPAIR_SOURCE),
        ] {
            let mut p = policy();
            p.schema_version = 2;
            p.backend = "vllm".into();
            p.backend_version = version.into();
            p.backend_revision = VLLM_REVISION.into();
            p.source_sha256 = source.into();
            p.native_flags = vllm_flags();
            p.source_pins = vllm_pins();
            let quarantined = version != "0.30.0";
            assert_eq!(p.validate().is_err(), quarantined);
            if !quarantined {
                assert_eq!(p.padded_batch(3).unwrap(), 4);
            }
            assert!(p.padded_batch(5).is_err());
            for defect in 0..4 {
                let mut bad = p.clone();
                match defect {
                    0 => bad.source_sha256 = SG_SOURCE.into(),
                    1 => bad.backend_version.push_str(".other"),
                    2 => bad
                        .native_flags
                        .insert("compiled_model".into(), true)
                        .map(|_| ())
                        .unwrap(),
                    _ => bad.disable_padding = true,
                }
                assert!(bad.validate().is_err());
            }
            let mut marker = marker();
            if let Op::Glm53Runtime(ref mut value) = marker {
                value.backend = "vllm".into();
            }
            let mut mhc = mhc();
            if let Op::Glm53Mhc(ref mut value) = mhc {
                value.backend = "vllm".into();
            }
            let mut rows = fixture();
            let setup = literal(geometry(&body(&marker)).unwrap());
            let physical = literal(geometry(&body(&mhc)).unwrap());
            rows[1] = Col::Str("geometry", vec![setup, setup, physical, physical]);
            let text = literal(serde_json::to_string(&serde_json::to_value(&p).unwrap()).unwrap());
            rows[9] = Col::Str("graph_policy", vec![text; 4]);
            rows[10] = Col::Str(
                "graph_policy_sha256",
                vec![literal(format!("{:x}", Sha256::digest(text.as_bytes()))); 4],
            );
            let root = tempfile::tempdir().unwrap();
            let _ = write_energy_systems_root(root.path());
            let path = root.path().join("data/vllm").join(version);
            std::fs::create_dir_all(&path).unwrap();
            write_parquet(&path.join(BASENAME), &rows);
            let db = PerfDatabase::load(root.path(), "testsys", "vllm", version)
                .unwrap()
                .with_mode(DatabaseMode::Silicon, TransferPolicy::ALL);
            let ctx = RuntimeContext {
                batch_size: 3,
                num_tokens: 3,
                s: 134,
                ..RuntimeContext::default()
            };
            if quarantined {
                // TEST_ONLY exact valid historical rows must not bypass quarantine.
                assert!(marker.query(&db, &ctx).is_err());
                assert!(mhc.query(&db, &ctx).is_err());
                continue;
            }
            assert_eq!(marker.query(&db, &ctx).unwrap().latency_ms, 2.0);
            assert_eq!(mhc.query(&db, &ctx).unwrap().latency_ms, 4.0);
            assert!(
                mhc.query(
                    &db,
                    &RuntimeContext {
                        batch_size: 4,
                        num_tokens: 4,
                        ..ctx
                    }
                )
                .is_err()
            );
            assert!(reject_mixed(&[mhc, marker], &db).is_err());
        }
    }
    #[test]
    fn source_bound_padding_is_not_an_observed_holdout_answer_table() {
        let mut p = policy();
        assert_eq!(p.padded_batch(3).unwrap(), 4);
        assert_eq!(p.padded_batch(2).unwrap(), 2);
        assert!(p.padded_batch(5).is_err());
        p.disable_padding = true;
        assert!(p.padded_batch(3).is_err());
        p.source_pins.insert(
            "srt/model_executor/runner/base_cuda_graph_runner.py".into(),
            SHA.into(),
        );
        assert!(p.padded_batch(2).is_err());
    }
    #[test]
    fn graph_setup_and_ops_share_real_context_but_never_cross_batch_or_pad() {
        let root = tempfile::tempdir().unwrap();
        let db = db(root.path());
        let ctx = RuntimeContext {
            batch_size: 3,
            num_tokens: 3,
            s: 134,
            ..RuntimeContext::default()
        };
        assert_eq!(marker().query(&db, &ctx).unwrap().latency_ms, 2.0);
        assert_eq!(mhc().query(&db, &ctx).unwrap().latency_ms, 4.0);
        assert!(
            mhc()
                .query(
                    &db,
                    &RuntimeContext {
                        batch_size: 4,
                        num_tokens: 4,
                        ..ctx
                    }
                )
                .is_err()
        );
        assert!(mhc().query(&db, &RuntimeContext { s: 145, ..ctx }).is_err());
        assert!(
            mhc()
                .query(
                    &db,
                    &RuntimeContext {
                        batch_size: 1,
                        ..ctx
                    }
                )
                .is_err()
        );
        if let Op::Glm53Mhc(op) = mhc() {
            assert!(op.query(&db, 3).is_err());
        }
        assert!(reject_mixed(&[mhc(), marker()], &db).is_err());
    }
    #[test]
    fn native_graph_position_preserves_inclusive_context_limit() {
        let root = tempfile::tempdir().unwrap();
        let database = db(root.path());
        let mut rows = fixture();
        rows[3] = Col::I64("prefix", vec![131063, 131071, 131063, 131071]);
        write_parquet(
            &root.path().join("data/sglang/0.5.20").join(BASENAME),
            &rows,
        );
        let ctx = RuntimeContext {
            batch_size: 3,
            num_tokens: 3,
            s: 131072,
            ..RuntimeContext::default()
        };
        assert_eq!(marker().query(&database, &ctx).unwrap().latency_ms, 3.0);
        for s in [0, 1, 131073] {
            assert!(
                marker()
                    .query(&database, &RuntimeContext { s, ..ctx })
                    .is_err()
            );
        }
    }

    #[test]
    fn marker_zero_for_sol_without_polluting_physical_work() {
        let root = tempfile::tempdir().unwrap();
        let mut db = db(root.path());
        db.database_mode = DatabaseMode::SolFull;
        let ctx = RuntimeContext {
            batch_size: 3,
            num_tokens: 3,
            s: 134,
            ..RuntimeContext::default()
        };
        assert_eq!(marker().weight_bytes(), 0.0);
        assert_eq!(marker().query(&db, &ctx).unwrap().latency_ms, 0.0);
        assert!(reject_mixed(&[mhc(), marker()], &db).is_ok());
    }
    #[test]
    fn missing_setup_bad_policy_and_holdout_rows_fail_before_query() {
        for index in [0, 4, 10, 11, 12, 13] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join("sglang/0.5.20");
            std::fs::create_dir_all(&path).unwrap();
            let mut cols = fixture();
            cols[index] = match index {
                0 => Col::Str("component", vec!["mhc"; 4]),
                4 => Col::I64("padded_batch_size", vec![3; 4]),
                10 => Col::Str("graph_policy_sha256", vec![SHA; 4]),
                11 => Col::Str("dataset_role", vec!["holdout"; 4]),
                12 => Col::Str("aggregation_policy", vec!["per_operation_tp_max_v1"; 4]),
                13 => Col::Str("rank_selection_sha256", vec![""; 4]),
                _ => unreachable!(),
            };
            write_parquet(&path.join(BASENAME), &cols);
            let table = Glm53GraphTable::with_sources(
                &path,
                &SourceResolver::fixed(PerfDbSources::default()),
            )
            .unwrap();
            assert!(
                table.has_measurements().is_err(),
                "bad column {index} admitted"
            );
        }
    }
}
