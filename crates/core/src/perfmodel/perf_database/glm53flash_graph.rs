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
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub const BASENAME: &str = "glm53flash_graph_perf.parquet";
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

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
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
}
struct Profile {
    policy: GraphPolicy,
    rows: BTreeMap<Key, Row>,
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
        let latency = if let Some(row) = profile.rows.get(&target) {
            row.latency
        } else {
            interpolate(&profile.rows, &target)?.ok_or_else(|| {
                invalid("native graph unit lacks same-policy, same-pad measured history brackets")
            })?
        };
        Ok(Some(PerformanceResult::with_energy(
            latency,
            0.0,
            Source::Silicon,
        )))
    }
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
fn interpolate(rows: &BTreeMap<Key, Row>, target: &Key) -> Result<Option<f64>, AicError> {
    let wanted = partition(target)?;
    let mut groups: BTreeMap<&str, Vec<(&Key, &Row)>> = BTreeMap::new();
    for (key, row) in rows {
        if key.component == target.component
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
            answer = Some(lv.latency * (1.0 - weight) + hv.latency * weight);
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
        let identity = (parsed.checkpoint_format.clone(), parsed.tp_size);
        if profiles
            .get(&identity)
            .is_some_and(|previous| previous.policy != parsed)
        {
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
        let evidence_key = (identity.clone(), batch, prefix, padded);
        let evidence_value = (
            row.str(cols[13])?.to_owned(),
            row.str(cols[14])?.to_owned(),
            row.u32(cols[7])?,
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
            component: component.into(),
            geometry: encoded.into(),
            batch,
            prefix,
            padded,
        };
        let profile = profiles.entry(identity).or_insert_with(|| Profile {
            policy: parsed.clone(),
            rows: BTreeMap::new(),
        });
        if profile
            .rows
            .insert(
                key,
                Row {
                    latency,
                    dispatch: row.str(cols[8])?.into(),
                    activity_count: count,
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
    for profile in profiles.values() {
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
