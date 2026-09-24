// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Separate, source-bound native FULL graph profiles. Costs are measured
//! disjoint-node activity unions, composed as an explicit additive approximation.
//! They are never inferred from whole-forward residuals or eager module timings.

use super::glm53flash::{geometry, primary_path, sha256, validate_geometry};
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
fn invalid(message: impl Into<String>) -> AicError {
    AicError::InvalidPerfData(message.into())
}

/// Actual initialized native policy, not a per-query table of observed answers.
/// The initial reviewed implementation is ordinary, uncompiled SGLang FULL
/// one-token decode. Other native backends/policies require their own audit.
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
        if self.schema_version != 1
            || self.backend != "sglang"
            || self.backend_version != "0.5.20"
            || self.backend_revision != SG_REVISION
            || self.checkpoint_revision != expected
            || !matches!(self.tp_size, 2 | 4)
            || self.phase != "generation"
            || self.runtime_mode != "FULL"
            || self.captured_req_width != 1
            || self.source_pins != pins
            || self.native_flags != flags
            || self.capture_sizes.is_empty()
            || self.capture_sizes[0] == 0
            || self.capture_sizes.windows(2).any(|p| p[0] >= p[1])
            || self.source_sha256 != SG_SOURCE
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
    ) && ops
        .iter()
        .any(|op| matches!(op, Op::Glm53Attention(_) | Op::Glm53Runtime(_)))
        && db.glm53flash_graph.has_measurements()?
    {
        return Err(invalid(
            "native graph Ops do not support the legacy mixed-step composition without actual homogeneous dispatch coordinates",
        ));
    }
    Ok(())
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
    path: Option<PathBuf>,
    request: (String, String),
    profile: OnceLock<Result<Option<Profiles>, String>>,
}
impl Glm53GraphTable {
    pub fn with_sources(root: &Path, resolver: &SourceResolver) -> Result<Self, AicError> {
        Ok(Self {
            path: primary_path(root, resolver, BASENAME)?,
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
    pub fn has_measurements(&self) -> Result<bool, AicError> {
        Ok(self.profile()?.is_some())
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
        let shape = shape.map_err(|e| invalid(e.to_string()))?;
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
            || ctx.s == 0
            || ctx.s >= 131072
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
            prefix: ctx.s,
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
    fn fixture() -> Vec<Col> {
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
            s: 133,
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
    fn marker_zero_for_sol_without_polluting_physical_work() {
        let root = tempfile::tempdir().unwrap();
        let mut db = db(root.path());
        db.database_mode = DatabaseMode::SolFull;
        let ctx = RuntimeContext {
            batch_size: 3,
            num_tokens: 3,
            s: 133,
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
