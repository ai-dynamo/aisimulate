// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema 4: exact, named SGLang native prefill intervals and allocator setup.
//! Independently expressed receipt contract; no native implementation is copied.
//! This reader grants neither interpolation nor runtime/measurement admission.

use super::glm53flash::{geometry, sha256, validate_geometry, validate_runtime};
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

pub(super) const BASENAME: &str = "glm53flash_sglang_prefill_perf.parquet";
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
fn model_contract() -> Value {
    json!({
        "model_class": "sglang.srt.models.glm5_next.Glm5NextForConditionalGeneration",
        "language_model_class": "sglang.srt.models.glm5_next.Glm5NextModel",
        "start_layer": 0, "end_layer": 45, "pp_size": 1, "dp_size": 1, "ep_size": 1,
        "text_only": true, "can_run_tbo": false, "dflash_capture": false,
        "layers_to_capture": [], "capture_aux_hidden_states": false,
        "input_embeds_buffer": false, "gemm_output_zero_allocator_size": 0,
        "bump_allocator_calls": 1, "bump_allocator_elements": 90,
        "bump_allocator_dtype": "torch.float32"
    })
}
fn source_pins() -> BTreeMap<String, String> {
    [
        (
            "srt/models/glm5_next.py",
            "12c5157b07fb7c6d93f34e84c43a37866d2e382e703729e2205aed9f8961f9c2",
        ),
        (
            "srt/utils/common.py",
            "52eedc9338c5d2565434d265858b5d47bcc67156d38353a5b218e91df7620e45",
        ),
        (
            "srt/managers/mm_utils.py",
            "a5e34a1af72faadf610feb7bf20ae41f3a50b2bbb821e38e6b091e43e4ad46c2",
        ),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v.into()))
    .collect()
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Policy {
    schema_version: u32,
    backend: String,
    backend_version: String,
    backend_revision: String,
    checkpoint_format: String,
    checkpoint_revision: String,
    config_sha256: String,
    tp_size: u32,
    runtime_digest: String,
    source_sha256: String,
    source_pins: BTreeMap<String, String>,
    timing_boundary: String,
    execution_policy_sha256: String,
    prefill_backend: String,
    decode_backend: String,
    native_model_contract: Value,
}
impl Policy {
    fn validate(&self) -> Result<(), AicError> {
        validate_runtime(&self.backend, &self.backend_version)?;
        let revision = match self.checkpoint_format.as_str() {
            "fp8" => "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
            "nvfp4" => "09b04e5e74bca08ca8549fc736d4cdd8624bfde3",
            _ => return Err(invalid("unknown SG prefill checkpoint")),
        };
        if self.schema_version != 4
            || self.backend != "sglang"
            || self.backend_version != "0.5.20"
            || self.backend_revision != "94602c9c2b7cbdb8efd5c52802dac6a1c180089e"
            || self.checkpoint_revision != revision
            || !matches!(self.tp_size, 2 | 4)
            || self.source_sha256
                != "401b762a863931720b2b5cdc7b64246fac11cb215dbf6ea0fd19f3db24ce7e49"
            || self.source_pins != source_pins()
            || !sha256(&self.config_sha256)
            || !sha256(&self.execution_policy_sha256)
            || !self
                .runtime_digest
                .strip_prefix("sha256:")
                .is_some_and(sha256)
            || self.timing_boundary != "embedding_to_logits_gpu_v1"
            || self.prefill_backend != "disabled"
            || self.decode_backend != "full"
            || self.native_model_contract != model_contract()
        {
            return Err(invalid(
                "SG prefill policy differs from the exact native source/model contract",
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
    batch: u32,
    query: u32,
    prefix: u32,
}
impl Point {
    fn validate(&self) -> Result<u32, AicError> {
        if self.batch == 0
            || self.query == 0
            || self
                .prefix
                .checked_add(self.query)
                .is_none_or(|n| n > 131072)
        {
            return Err(invalid(
                "SG prefill requires positive B/Q and P+Q <= 131072",
            ));
        }
        self.batch
            .checked_mul(self.query)
            .ok_or_else(|| invalid("SG prefill B*Q overflow"))
    }
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    unit: Unit,
    point: Point,
}
struct Profile {
    policy: Policy,
    rows: BTreeMap<Key, f64>,
    units: BTreeSet<Unit>,
}
type Profiles = BTreeMap<(String, u32), Profile>;
pub(super) struct PrefillTable {
    path: Option<PathBuf>,
    request: (String, String),
    gb300: bool,
    profiles: OnceLock<Result<Option<Profiles>, String>>,
}
impl PrefillTable {
    pub(super) fn new(path: Option<PathBuf>, request: (String, String), gb300: bool) -> Self {
        Self {
            path,
            request,
            gb300,
            profiles: OnceLock::new(),
        }
    }
    fn profiles(&self) -> Result<Option<&Profiles>, AicError> {
        self.profiles
            .get_or_init(|| match &self.path {
                Some(path) => load(path, &self.request, self.gb300)
                    .map_err(|e| format!("{}: {e}", path.display())),
                None => Ok(None),
            })
            .as_ref()
            .map(Option::as_ref)
            .map_err(|e| invalid(e.clone()))
    }
    pub(super) fn has_measurements(&self) -> Result<bool, AicError> {
        Ok(self.profiles()?.is_some())
    }
    pub(super) fn matches_model(&self, model: &str, tp: u32) -> Result<bool, AicError> {
        let format = match model {
            "zai-org/GLM-5.3-Flash" => "fp8",
            "nvidia/GLM-5.3-Flash-NVFP4" => "nvfp4",
            _ => return Ok(false),
        };
        Ok(self
            .profiles()?
            .is_some_and(|p| p.contains_key(&(format.into(), tp))))
    }
    pub(super) fn requires_context(&self, shape: &Value) -> Result<bool, AicError> {
        Ok(shape["is_context"] == true && self.has_measurements()?)
    }
    pub(super) fn validate_ops(&self, ops: &[Op]) -> Result<bool, AicError> {
        self.validate_phase(ops, true)
    }
    pub(super) fn validate_generation_contract(
        &self,
        ops: &[Op],
        context_ops: &[Op],
    ) -> Result<bool, AicError> {
        let selected = self.validate_phase(ops, false)?;
        if selected {
            let (_, generation) =
                operation(&ops[0])?.ok_or_else(|| invalid("invalid SG generation identity"))?;
            let (_, context) = context_ops
                .first()
                .map(operation)
                .transpose()?
                .flatten()
                .ok_or_else(|| invalid("missing SG context identity"))?;
            if identity_of(&generation) != identity_of(&context) {
                return Err(invalid(
                    "SG prefill/generation checkpoint/TP identities differ",
                ));
            }
        }
        Ok(selected)
    }
    fn validate_phase(&self, ops: &[Op], context: bool) -> Result<bool, AicError> {
        let Some(profiles) = self.profiles()? else {
            return Ok(false);
        };
        let mut units = BTreeSet::new();
        let mut identity = None;
        for op in ops {
            let (unit, shape) = operation(op)?
                .ok_or_else(|| invalid("SG prefill rejects nested/fallback/non-GLM operations"))?;
            if shape["is_context"] != context || shape["backend"] != "sglang" {
                return Err(invalid("SG prefill operation phase/backend differs"));
            }
            let id = identity_of(&shape);
            if identity.as_ref().is_some_and(|old| old != &id) || !units.insert(unit) {
                return Err(invalid(
                    "SG prefill has duplicate or mismatched operation identity",
                ));
            }
            identity = Some(id);
        }
        validate_units(&units)?;
        let profile = profiles
            .get(&identity.ok_or_else(|| invalid("empty SG prefill phase"))?)
            .ok_or_else(|| invalid("SG prefill table lacks compiled checkpoint/TP"))?;
        if context && profile.units != units {
            return Err(invalid(
                "SG prefill compiled names/geometries differ from complete measured phase",
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
        if shape["is_context"] != true {
            return Ok(None);
        }
        let Some(profiles) = self.profiles()? else {
            return Ok(None);
        };
        let profile = profiles
            .get(&identity_of(&shape))
            .ok_or_else(|| invalid("selected SG prefill table lacks checkpoint/TP"))?;
        if shape["backend"] != "sglang" {
            return Err(invalid("SG prefill backend differs"));
        }
        let point = Point {
            batch: ctx.batch_size,
            query: ctx.s,
            prefix: ctx.prefix,
        };
        let tokens = point.validate()?;
        if ctx.num_tokens != tokens
            || ctx.beam_width != 1
            || ctx.num_image_tokens != 0
            || ctx.seq_imbalance_correction_scale != 1.0
            || ctx.gen_seq_imbalance_correction_scale != 1.0
        {
            return Err(invalid(
                "SG prefill requires complete homogeneous RuntimeContext with num_tokens=B*Q",
            ));
        }
        let latency = profile.rows.get(&Key {unit, point})
            .ok_or_else(|| invalid("SG prefill lacks exact measured operation/B/Q/P; fallback and interpolation are disabled"))?;
        Ok(Some(PerformanceResult::with_energy(
            *latency,
            0.0,
            Source::Silicon,
        )))
    }
}
fn identity_of(shape: &Value) -> (String, u32) {
    (
        shape["checkpoint_format"]
            .as_str()
            .unwrap_or_default()
            .into(),
        shape["tp_size"].as_u64().unwrap_or_default() as u32,
    )
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
            o.validate_physical()?;
            ("ffn", serde_json::to_value(o))
        }
        Op::Glm53Primitive(o) => {
            o.validate_physical()?;
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
    let mut expected: BTreeSet<(String, String)> = [
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
    for layer in 0..45 {
        for (component, prefix) in [
            ("attention", "attention"),
            ("ffn", "ffn"),
            ("mhc", "mhc_pre_attn"),
            ("mhc", "mhc_post_attn"),
            ("mhc", "mhc_pre_ffn"),
            ("mhc", "mhc_post_ffn"),
            ("primitive", "attention_allreduce"),
            ("primitive", "ffn_allreduce"),
        ] {
            expected.insert((component.into(), format!("{prefix}_{layer}")));
        }
    }
    if units.len() != 367
        || units
            .iter()
            .map(|u| (u.component.clone(), u.name.clone()))
            .collect::<BTreeSet<_>>()
            != expected
    {
        return Err(invalid(
            "SG prefill requires complete 366 original named units and exactly one setup",
        ));
    }
    Ok(())
}
fn load(
    path: &Path,
    request: &(String, String),
    gb300: bool,
) -> Result<Option<Profiles>, AicError> {
    if !path.try_exists().map_err(|e| invalid(e.to_string()))? {
        return Ok(None);
    }
    if !gb300 {
        return Err(invalid("SG prefill measurements require GB300"));
    }
    let reader = PerfReader::open(path)?;
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
        "activity_count",
        "sample_count",
        "dispatch_fingerprint",
        "measurement_method",
        "prefill_policy",
        "prefill_policy_sha256",
        "dataset_role",
        "aggregation_policy",
        "rank_selection_sha256",
        "evidence_sha256",
        "policy_evidence_sha256",
        "measurement_scope",
    ];
    let c: Vec<_> = names
        .iter()
        .map(|n| reader.col(n))
        .collect::<Result<_, _>>()?;
    let mut policies = BTreeMap::new();
    let mut profiles: Profiles = BTreeMap::new();
    type Evidence = (String, String, String);
    let mut points: BTreeMap<((String, u32), Point), (Evidence, BTreeSet<Unit>)> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let text = row.str(c[16])?;
        if !policies.contains_key(text) {
            let policy: Policy = serde_json::from_str(text).map_err(|e| invalid(e.to_string()))?;
            policy.validate()?;
            if canonical(&policy)? != text
                || request != &(policy.backend.clone(), policy.backend_version.clone())
            {
                return Err(invalid(
                    "SG prefill policy is noncanonical or belongs to another runtime",
                ));
            }
            policies.insert(text.to_owned(), policy);
        }
        let policy = &policies[text];
        if row.str(c[17])? != digest(text) {
            return Err(invalid("SG prefill policy SHA mismatch"));
        }
        let id = (policy.checkpoint_format.clone(), policy.tp_size);
        let profile = profiles.entry(id.clone()).or_insert_with(|| Profile {
            policy: policy.clone(),
            rows: BTreeMap::new(),
            units: BTreeSet::new(),
        });
        if profile.policy != *policy {
            return Err(invalid(
                "SG prefill has competing policies for checkpoint/TP",
            ));
        }
        let unit = Unit {
            component: row.str(c[0])?.into(),
            name: row.str(c[1])?.into(),
            geometry: row.str(c[2])?.into(),
        };
        let shape = if unit.component == "runtime" {
            let mut shape: Value =
                serde_json::from_str(&unit.geometry).map_err(|e| invalid(e.to_string()))?;
            shape
                .as_object_mut()
                .ok_or_else(|| invalid("SG setup geometry must be an object"))?
                .insert("name".into(), unit.name.clone().into());
            let op: crate::operators::Glm53RuntimeOp =
                serde_json::from_value(shape).map_err(|e| invalid(e.to_string()))?;
            op.validate()?;
            if geometry(&op)? != unit.geometry {
                return Err(invalid("noncanonical SG setup geometry"));
            }
            serde_json::to_value(op).map_err(|e| invalid(e.to_string()))?
        } else {
            validate_geometry(&unit.component, &unit.geometry)?
        };
        let point = Point {
            batch: row.u32(c[5])?,
            query: row.u32(c[6])?,
            prefix: row.u32(c[7])?,
        };
        let tokens = point.validate()?;
        if row.str(c[3])? != "context"
            || row.str(c[4])? != "NONE"
            || shape["is_context"] != true
            || shape["backend"] != "sglang"
            || identity_of(&shape) != id
            || row.u32(c[8])? != tokens
            || row.u32(c[9])? != point.batch
        {
            return Err(invalid(
                "SG prefill row has mismatched phase/policy/physical geometry",
            ));
        }
        let latency = row.f64(c[10])?;
        let count = row.u32(c[11])?;
        let activity = row.u32(c[12])?;
        let fingerprint = row.str(c[14])?;
        if !latency.is_finite()
            || latency < 0.0
            || (latency == 0.0 && activity != 0)
            || count == 0
            || (unit.component == "runtime" && count != 1)
            || row.u32(c[13])? != 10
            || if activity == 0 {
                !fingerprint.is_empty()
            } else {
                !sha256(fingerprint)
            }
            || row.str(c[15])? != "native_sglang_prefill_events_v1"
            || row.str(c[18])? != "calibration"
            || row.str(c[19])? != "whole_forward_slowest_rank_v1"
            || row.str(c[23])? != "native_sglang_prefill_units_v1"
            || [c[20], c[21], c[22]]
                .iter()
                .any(|&i| !row.str(i).is_ok_and(sha256))
        {
            return Err(invalid(
                "SG prefill row lacks complete measured timing/provenance",
            ));
        }
        let evidence = (
            row.str(c[20])?.into(),
            row.str(c[21])?.into(),
            row.str(c[22])?.into(),
        );
        let (old, units) = points
            .entry((id, point.clone()))
            .or_insert_with(|| (evidence.clone(), BTreeSet::new()));
        if old != &evidence
            || !units.insert(unit.clone())
            || profile.rows.insert(Key { unit, point }, latency).is_some()
        {
            return Err(invalid(
                "SG prefill point mixes evidence or duplicates measured units",
            ));
        }
    }
    if profiles.is_empty() {
        return Err(invalid("selected SG prefill table is empty"));
    }
    for ((id, _), (_, units)) in points {
        validate_units(&units)?;
        let profile = profiles.get_mut(&id).unwrap();
        if !profile.units.is_empty() && profile.units != units {
            return Err(invalid(
                "SG prefill points have different operation manifests",
            ));
        }
        profile.units = units;
    }
    Ok(Some(profiles))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sg_context_coordinates_are_exact_and_not_vllm_aligned() {
        for prefix in [0, 118, 126, 131040] {
            assert_eq!(
                Point {
                    batch: 2,
                    query: 32,
                    prefix
                }
                .validate()
                .unwrap(),
                64
            );
        }
        for (batch, query, prefix) in [(0, 32, 0), (1, 0, 0), (1, 32, 131041), (u32::MAX, 2, 0)] {
            assert!(
                Point {
                    batch,
                    query,
                    prefix
                }
                .validate()
                .is_err()
            );
        }
    }
}
