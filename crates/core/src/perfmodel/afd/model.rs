// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified native port of heterogeneous AFD/fabric behavior; source revision and
// original paths: root THIRD_PARTY_NOTICES.md (AISimulate heterogeneous AFD).
// Source: https://github.com/LPhgh/aisimulate/tree/af56328c4ba7ffa1b8c42c8375244568c2e7bb8c

//! AFD timing from pool-specific, source-ordered execution plans.

#[path = "execution.rs"]
mod execution;

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::{AfdModelShape, AfdPoolRole};
use crate::common::enums::CommQuantMode;
use crate::common::error::AicError;
use crate::engine::AfdEngineConfig;
use crate::operators::PerformanceResult;
use crate::operators::afd_comm::{
    AfdCombineOp, AfdFCollectiveOp, AfdFabricLink, AfdTokenMapping, AfdTransferDirection,
    AfdTransferOp,
};
use crate::perfmodel::engine::Engine;
use crate::perfmodel::session::{query_context_op, query_generation_op};
use crate::{
    BackendKind, DatabaseMode, EstimationMode, EstimatorConfig, ForwardPassFallbackPolicy,
};
#[cfg(feature = "python")]
use crate::{ForwardPassPerfModel, ForwardPassPerfModelConfig, ForwardPassWorkerType};

/// Stride for sampling the KV length along the decode trace.
pub const AFD_DECODE_STRIDE: u32 = 32;

pub use super::pipeline::AfdPipelineLayout;

/// AFD phase selector for the measurement API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AfdPhase {
    Prefill,
    Decode,
}

// Pool identity uses the same types as replay's authoritative topology.
pub type AfdAPoolSpec = crate::engine::AfdAPoolConfig;
pub type AfdFPoolSpec = crate::engine::AfdFPoolConfig;

/// Explicit per-model quant overrides shared by both pools (the caller's
/// explicit choice, preserved verbatim per perfmodel-api). Absent fields
/// stay `None` so each pool's compile resolves quant fresh against its own
/// system — the quant rule. NEVER populate these from the other pool's
/// resolved values: an A-pool resolution inherited by the F pool misprices
/// it (e.g. B200 forced into nvfp4_wo).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AfdQuantOverrides {
    pub gemm_quant_mode: Option<String>,
    pub moe_quant_mode: Option<String>,
    pub fmha_quant_mode: Option<String>,
    pub kvcache_quant_mode: Option<String>,
    pub comm_quant_mode: Option<String>,
}

impl AfdQuantOverrides {
    fn resolved_comm_quant_mode(&self) -> Result<CommQuantMode, AicError> {
        self.comm_quant_mode
            .as_deref()
            .map_or(Ok(CommQuantMode::Half), |mode| {
                serde_json::from_value(serde_json::Value::String(mode.into())).map_err(|error| {
                    AicError::InvalidEngineConfig(format!("invalid AFD comm_quant_mode: {error}"))
                })
            })
    }
}

/// Estimator plumbing shared by both pool configs (perfmodel-api's
/// controls, passed through unchanged).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AfdEstimatorPlumbing {
    pub enable_shared_layer: Option<bool>,
    pub strict_provenance: bool,
    pub database_mode: DatabaseMode,
    pub transfer_policy: Option<Vec<String>>,
    pub systems_paths: Vec<PathBuf>,
    pub estimation_mode: EstimationMode,
    pub fallback_policy: ForwardPassFallbackPolicy,
    pub estimator_config: EstimatorConfig,
}

/// Construction input for [`AfdForwardPassPerfModel`].
///
/// Each pool's node width is validated against its selected perf-DB system
/// spec; a mismatch means enumeration used a different hardware definition.
#[derive(Clone, Debug)]
pub struct AfdForwardPassModelConfig {
    pub model: String,
    pub backend: BackendKind,
    pub backend_version: Option<String>,
    /// Explicit kernel selectors passed unchanged to each pool's canonical model.
    pub attention_backend: Option<String>,
    pub moe_kernel_source: Option<String>,
    /// Shared, immutable topology used by replay resource accounting and timing.
    pub topology: Arc<AfdEngineConfig>,
    pub num_layers: u32,
    pub hidden_size: u32,
    /// Routed MoE dispatch identity; models must contain routed layers.
    pub num_experts: u32,
    pub topk: u32,
    /// Source-derived facts. Required by the engine-only constructor; build
    /// resolves and verifies them against the cached model source metadata.
    pub model_shape: Option<AfdModelShape>,
    /// MTP depth: decode uses `nextn + 1` verification tokens and the aggregate
    /// model's `(num_layers + nextn) / num_layers` communication work scale.
    pub nextn: u32,
    pub quant: AfdQuantOverrides,
    pub estimator: AfdEstimatorPlumbing,
}

impl AfdForwardPassModelConfig {
    /// Establish topology invariants before compilation or querying either pool.
    fn validate(&self) -> Result<(), AicError> {
        self.topology.routing_location.ensure_supported()?;
        for (name, value) in [
            ("gpus_per_node", self.topology.gpus_per_node),
            ("f_gpus_per_node", self.topology.effective_f_gpus_per_node()),
            ("num_layers", self.num_layers),
            ("hidden_size", self.hidden_size),
            ("num_microbatches", self.topology.num_microbatches),
            ("a_pool.a_tp_size", self.topology.a_pool.a_tp_size),
            ("a_pool.n_a_nodes", self.topology.a_pool.n_a_nodes),
            (
                "a_pool.batch_size_per_a_replica",
                self.topology.a_pool.batch_size_per_a_replica,
            ),
            ("f_pool.n_f_nodes", self.topology.f_pool.n_f_nodes),
            ("f_pool.f_moe_ep_size", self.topology.f_pool.f_moe_ep_size),
        ] {
            if value == 0 {
                return Err(AicError::InvalidEngineConfig(format!(
                    "afd timing config: {name} must be a positive integer, got {value}"
                )));
            }
        }
        if self.topology.gpus_per_node % self.topology.a_pool.a_tp_size != 0 {
            return Err(AicError::InvalidEngineConfig(format!(
                "afd timing config: a_tp_size={} must divide gpus_per_node={}",
                self.topology.a_pool.a_tp_size, self.topology.gpus_per_node
            )));
        }
        if self.topology.num_microbatches > self.topology.a_pool.batch_size_per_a_replica {
            return Err(AicError::InvalidEngineConfig(
                "afd timing config: num_microbatches cannot exceed a_pool.batch_size_per_a_replica"
                    .into(),
            ));
        }
        if !self.topology.comm_overhead_factor.is_finite()
            || self.topology.comm_overhead_factor <= 0.0
        {
            return Err(AicError::InvalidEngineConfig(format!(
                "afd timing config: comm_overhead_factor must be positive finite, got {}",
                self.topology.comm_overhead_factor
            )));
        }
        let a_gpu_count = self
            .topology
            .a_pool
            .n_a_nodes
            .checked_mul(self.topology.gpus_per_node);
        let f_gpu_count = self
            .topology
            .f_pool
            .n_f_nodes
            .checked_mul(self.topology.effective_f_gpus_per_node());
        let geometry_fits = a_gpu_count.zip(f_gpu_count).is_some_and(|(a, f)| {
            a.checked_add(f).is_some()
                && (a / self.topology.a_pool.a_tp_size)
                    .checked_mul(self.topology.a_pool.batch_size_per_a_replica)
                    .is_some()
        });
        if !geometry_fits || self.nextn.checked_add(1).is_none() {
            return Err(AicError::InvalidEngineConfig(
                "afd timing config: GPU, batch, or nextn geometry exceeds u32 range".into(),
            ));
        }
        let f_gpu_count = f_gpu_count.expect("checked topology");
        if f_gpu_count % self.topology.f_pool.f_moe_ep_size != 0 {
            return Err(AicError::InvalidEngineConfig(format!(
                "f_moe_ep_size={} must divide f_gpu_count={f_gpu_count}",
                self.topology.f_pool.f_moe_ep_size
            )));
        }
        super::resolve_afd_parallelism(
            self.topology.a_pool.a_tp_size,
            self.topology.a_pool.n_a_nodes,
            self.topology.f_pool.n_f_nodes,
            self.topology.gpus_per_node,
            Some(self.topology.f_pool.f_moe_ep_size),
            Some(self.num_experts),
            self.topology.f_gpus_per_node,
        )?;
        self.quant.resolved_comm_quant_mode()?;
        if let Some(shape) = self.model_shape {
            shape.validate(self.num_layers, self.num_experts, self.topk)?;
        }
        Ok(())
    }
}

/// Per-phase layer ingredients.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AfdLayerTimes {
    pub attention_ms: f64,
    pub ffn_ms: f64,
    pub a_to_f_ms: f64,
    pub f_to_a_ms: f64,
    pub num_layers: u32,
}

/// Raw layer ingredients and complete pass latencies. Decode values are
/// averaged only after evaluating each KV sample's pipeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AfdPhaseMetrics {
    pub layer_times: AfdLayerTimes,
    /// Diagnostic latency with unscaled transfers.
    pub t_step_ms: f64,
    /// Latency with the configured transfer factor applied once per sample.
    pub billed_step_ms: f64,
}

impl AfdPhaseMetrics {
    #[cfg(test)]
    pub(crate) fn from_layer_times(
        layout: AfdPipelineLayout,
        layer_times: AfdLayerTimes,
        microbatches: u32,
        factor: f64,
    ) -> Result<Self, AicError> {
        let t_step_ms = layout.step_latency_ms(&layer_times, microbatches, 1.0)?;
        let billed_step_ms = if factor == 1.0 {
            t_step_ms
        } else {
            layout.step_latency_ms(&layer_times, microbatches, factor)?
        };
        Ok(Self {
            layer_times,
            t_step_ms,
            billed_step_ms,
        })
    }
}

/// Accumulate evaluated samples, never evaluate averaged stage times.
#[derive(Default)]
struct PhaseAverage {
    attention: f64,
    ffn: f64,
    a_to_f: f64,
    f_to_a: f64,
    step: f64,
    billed: f64,
    repeats: u32,
}

impl PhaseAverage {
    fn push(&mut self, sample: &AfdPhaseMetrics, repeat: u32) {
        let weight = f64::from(repeat);
        self.attention += sample.layer_times.attention_ms * weight;
        self.ffn += sample.layer_times.ffn_ms * weight;
        self.a_to_f += sample.layer_times.a_to_f_ms * weight;
        self.f_to_a += sample.layer_times.f_to_a_ms * weight;
        self.step += sample.t_step_ms * weight;
        self.billed += sample.billed_step_ms * weight;
        self.repeats += repeat;
    }

    fn finish(self, num_layers: u32) -> AfdPhaseMetrics {
        debug_assert!(
            self.repeats > 0,
            "decode sampling always visits at least once"
        );
        let denom = f64::from(self.repeats);
        AfdPhaseMetrics {
            layer_times: AfdLayerTimes {
                attention_ms: self.attention / denom,
                ffn_ms: self.ffn / denom,
                a_to_f_ms: self.a_to_f / denom,
                f_to_a_ms: self.f_to_a / denom,
                num_layers,
            },
            t_step_ms: self.step / denom,
            billed_step_ms: self.billed / denom,
        }
    }
}

fn for_each_decode_sample(
    isl: u32,
    osl: u32,
    mut visit: impl FnMut(u32, u32) -> Result<(), AicError>,
) -> Result<(), AicError> {
    let decode_steps = osl.saturating_sub(1).max(1);
    let mut i = 0;
    while i < decode_steps {
        let s_i = isl + i + 1;
        let repeat = if osl > 1 {
            AFD_DECODE_STRIDE.min(osl - 1 - i)
        } else {
            1
        };
        visit(s_i, repeat)?;
        i += AFD_DECODE_STRIDE;
    }
    Ok(())
}

/// One pool's compiled engine and prepared physical residency/execution counts.
struct PoolRuntime {
    engine: Arc<Engine>,
    weight_gb: f64,
}

/// The `"afd"` forward-pass perf model.
pub struct AfdForwardPassPerfModel {
    config: AfdForwardPassModelConfig,
    repetitions: Vec<u32>,
    /// Full-batch A-only layers before the circulating routed microbatches.
    dense_prefix_layers: usize,
    /// Pin memory preflight to the roots actually selected by each constructor.
    selected_systems_roots: [Option<PathBuf>; 2],
    a_pool: PoolRuntime,
    f_pool: PoolRuntime,
    a2f: AfdTransferOp,
    f2a: AfdTransferOp,
    f_ag: AfdFCollectiveOp,
    f_rs: AfdFCollectiveOp,
    a_combine: AfdCombineOp,
    token_mapping: AfdTokenMapping,
    fabric: AfdFabricLink,
}

#[cfg(any(feature = "python", test))]
fn paired_prediction_provenance(
    a_pool: serde_json::Value,
    f_pool: serde_json::Value,
) -> serde_json::Value {
    let a_version = a_pool["backend_version"].as_str().filter(|v| !v.is_empty());
    let f_version = f_pool["backend_version"].as_str().filter(|v| !v.is_empty());
    let version = a_version.filter(|a| Some(*a) == f_version);
    serde_json::json!({
        "estimator": "op_level",
        "timing_source": "afd",
        "data_version": version,
        "data_version_unavailable_reason": version.is_none().then_some(
            "AFD pools do not share a single nonempty data version; see data_identity"
        ),
        "data_identity": {"a_pool": a_pool, "f_pool": f_pool},
    })
}

impl AfdForwardPassPerfModel {
    /// The construction identity (topology, pipeline knobs, estimator
    /// plumbing) — the `AfdTimingModel` derives its pipeline knobs from it.
    pub(crate) fn config(&self) -> &AfdForwardPassModelConfig {
        &self.config
    }

    /// Resolved pair identity for the role report; computed once during setup.
    #[cfg(feature = "python")]
    pub(crate) fn prediction_provenance(&self) -> serde_json::Value {
        let identity = |a_pool: bool| {
            let engine = if a_pool {
                &self.a_pool.engine
            } else {
                &self.f_pool.engine
            };
            let db = engine.database();
            serde_json::json!({
                "model": self.config.model,
                "system": db.system,
                "backend": db.backend,
                "backend_version": db.version,
                "data_root": db.data_root,
                "systems_root": self.pool_systems_root(a_pool),
                "requested_quantization": self.config.quant,
                "effective_quantization": engine.execution_quantization(),
            })
        };
        paired_prediction_provenance(identity(true), identity(false))
    }

    pub(crate) fn pool_systems_root(&self, a_pool: bool) -> Option<&PathBuf> {
        self.selected_systems_roots[usize::from(!a_pool)].as_ref()
    }

    /// Production constructor: one `ForwardPassPerfModel::best_available`
    /// per pool system (perfmodel-api's single construction interface), then
    /// [`Self::from_engines`]. Requires the `python` feature (engine
    /// compilation crosses into Python).
    ///
    /// Per-pool identity mapping:
    /// - A pool: `tp = a_tp_size`, `attention_dp = 1`, `moe_tp = a_tp_size`,
    ///   `moe_ep = 1` (MoE dims inert but constraint-satisfying);
    /// - F pool: `tp = n_f_nodes * f_gpus_per_node`, `moe_tp = f_pool_width /
    ///   f_moe_ep_size`, `moe_ep = f_moe_ep_size` (divisibility checked).
    ///
    /// Plans require op-level engines; whole-model FPM modes reject explicitly.
    #[cfg(feature = "python")]
    pub fn build(mut config: AfdForwardPassModelConfig) -> Result<Self, AicError> {
        config.validate()?;
        // Architecture gate: reject models whose stages/residency cannot be
        // split across A/F pools before compilation turns them into
        // confusing partition errors.
        let (architecture, shape) = pyo3::Python::with_gil(|py| {
            use pyo3::prelude::PyAnyMethods;
            let module = py.import("aisimulate_core.sdk.models.helpers")?;
            let info = module
                .getattr("_get_model_info")?
                .call1((config.model.as_str(),))?;
            let architecture: Option<String> = info.get_item("architecture")?.extract()?;
            let shape = AfdModelShape {
                num_shared_experts: info.get_item("num_shared_experts")?.extract()?,
                num_moe_layers: info.get_item("num_moe_layers")?.extract()?,
            };
            let source_layers: u32 = info.get_item("layers")?.extract()?;
            let source_experts: u32 = info.get_item("num_experts")?.extract()?;
            let source_topk: u32 = info.get_item("topk")?.extract()?;
            let source_hidden: u32 = info.get_item("hidden_size")?.extract()?;
            let source_heads: u32 = info.get_item("n")?.extract()?;
            if source_hidden != config.hidden_size
                || source_heads % config.topology.a_pool.a_tp_size != 0
            {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "AFD hidden size/attention TP disagrees with model source geometry",
                ));
            }
            if (source_layers, source_experts, source_topk)
                != (config.num_layers, config.num_experts, config.topk)
            {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "AFD layer/expert/topk identity disagrees with model source metadata",
                ));
            }
            Ok::<_, pyo3::PyErr>((architecture.unwrap_or_default(), shape))
        })
        .map_err(|error| {
            AicError::InvalidEngineConfig(format!(
                "AFD could not resolve the architecture of model {:?}: {error}",
                config.model
            ))
        })?;
        crate::perfmodel::afd::validation::validate_afd_model_architecture(&architecture)?;
        shape.validate(config.num_layers, config.num_experts, config.topk)?;
        if config.model_shape.is_some_and(|saved| saved != shape) {
            return Err(AicError::InvalidEngineConfig(
                "AFD model_shape disagrees with model source metadata".into(),
            ));
        }
        config.model_shape = Some(shape);
        let a_config = pool_fpm_config(&config, PoolKind::A)?;
        let f_config = pool_fpm_config(&config, PoolKind::F)?;
        let a_fpm = ForwardPassPerfModel::best_available(a_config)?;
        let f_fpm = ForwardPassPerfModel::best_available(f_config)?;
        let a_engine = a_fpm.native_engine().ok_or_else(|| {
            AicError::InvalidEngineConfig(format!(
                "AFD requires a native per-pool engine (op lists); the A pool resolved to a \
                 regression-only model: {}",
                a_fpm
                    .provenance()
                    .map(|p| p.selection_failures.join("; "))
                    .unwrap_or_default(),
            ))
        })?;
        let f_engine = f_fpm.native_engine().ok_or_else(|| {
            AicError::InvalidEngineConfig(format!(
                "AFD requires a native per-pool engine (op lists); the F pool resolved to a \
                 regression-only model: {}",
                f_fpm
                    .provenance()
                    .map(|p| p.selection_failures.join("; "))
                    .unwrap_or_default(),
            ))
        })?;
        let selected_systems_roots = [&a_fpm, &f_fpm].map(|model| {
            model
                .provenance()
                .and_then(|p| p.selected_systems_root.clone())
        });
        let mut model = Self::from_engines(config, a_engine, f_engine)?;
        model.selected_systems_roots = selected_systems_roots;
        Ok(model)
    }

    /// Build from two already-compiled pool engines. Production callers use
    /// [`Self::build`]; this constructor is the feature-independent core the
    /// spec-fixture parity tests drive directly.
    pub fn from_engines(
        config: AfdForwardPassModelConfig,
        a_engine: Arc<Engine>,
        f_engine: Arc<Engine>,
    ) -> Result<Self, AicError> {
        config.validate()?;
        let comm_quant_mode = config.quant.resolved_comm_quant_mode()?;
        config.model_shape.ok_or_else(|| AicError::InvalidEngineConfig(
            "AFD from_engines requires source-derived model_shape; missing metadata cannot establish support".into()
        ))?;
        for (pool, engine, tp, moe_tp, moe_ep) in [
            (
                "A",
                &a_engine,
                config.topology.a_pool.a_tp_size,
                config.topology.a_pool.a_tp_size,
                1,
            ),
            (
                "F",
                &f_engine,
                config.topology.f_pool.n_f_nodes * config.topology.effective_f_gpus_per_node(),
                1,
                config.topology.f_pool.f_moe_ep_size,
            ),
        ] {
            if engine.parallelism() != (tp, 1, moe_tp, moe_ep) {
                return Err(AicError::InvalidEngineConfig(format!(
                    "AFD {pool}-pool compiled parallel identity disagrees with the configured topology"
                )));
            }
        }
        // gpus_per_node must agree with each pool's DB system spec.
        for (pool, engine, width) in [
            ("a", &a_engine, config.topology.gpus_per_node),
            ("f", &f_engine, config.topology.effective_f_gpus_per_node()),
        ] {
            let db = engine.database();
            let db_gpus = db.system_spec.node.num_gpus_per_node;
            if db_gpus != width {
                return Err(AicError::InvalidEngineConfig(format!(
                    "AFD {pool}-pool topology gpus_per_node={} disagrees with the '{}' system \
                     spec, which reports {db_gpus}",
                    width, db.system,
                )));
            }
        }
        {
            let a = a_engine.afd_plan().ok_or_else(|| {
                AicError::InvalidEngineConfig(
                    "missing A execution plan; recompile the AFD engine spec".into(),
                )
            })?;
            let f = f_engine.afd_plan().ok_or_else(|| {
                AicError::InvalidEngineConfig(
                    "missing F execution plan; recompile the AFD engine spec".into(),
                )
            })?;
            if a.role != AfdPoolRole::A
                || f.role != AfdPoolRole::F
                || a.model != config.model
                || f.model != config.model
                || a.layers != f.layers
                || a.layers.len() != config.num_layers as usize
                || a.shared_inter_size != f.shared_inter_size
                || a.hidden_size != config.hidden_size
                || f.hidden_size != config.hidden_size
                || a.num_experts != config.num_experts
                || f.num_experts != config.num_experts
                || a.topk != config.topk
                || f.topk != config.topk
                || a.context.templates.len() != f.context.templates.len()
                || a.context
                    .templates
                    .iter()
                    .zip(&f.context.templates)
                    .any(|(a, f)| a.routed != f.routed || a.attention_kind != f.attention_kind)
                || a.layers
                    .iter()
                    .filter(|&&i| a.context.templates[i].routed)
                    .count()
                    != config.model_shape.expect("validated shape").num_moe_layers as usize
            {
                return Err(AicError::InvalidEngineConfig(
                    "AFD pool execution plans disagree with source identity or geometry".into(),
                ));
            }
        }
        let plan = a_engine.afd_plan().expect("validated AFD plan");
        let mut repetitions = vec![0; plan.context.templates.len()];
        for &i in &plan.layers {
            repetitions[i] += 1;
        }
        let dense_prefix_layers = plan
            .layers
            .iter()
            .take_while(|&&i| !plan.context.templates[i].routed)
            .count();
        let prepare = |engine: Arc<Engine>| {
            let plan = engine.afd_plan().expect("validated AFD plan");
            let weight_gb = execution::physical_weights(&engine, plan, AfdPhase::Prefill)
                .max(execution::physical_weights(&engine, plan, AfdPhase::Decode));
            PoolRuntime { engine, weight_gb }
        };
        let a_pool = prepare(a_engine);
        let f_pool = prepare(f_engine);
        let a_replica_count = a_replica_count(&config);
        let f_rank_count =
            config.topology.f_pool.n_f_nodes * config.topology.effective_f_gpus_per_node();
        let shared_transfer = |direction: AfdTransferDirection, name: &str| AfdTransferOp {
            name: name.into(),
            scale_factor: 1.0,
            direction,
            hidden_size: config.hidden_size,
            a_replica_count,
            f_rank_count,
            gpus_per_node: config.topology.gpus_per_node,
            f_gpus_per_node: config.topology.f_gpus_per_node,
            num_experts: config.num_experts,
            topk: config.topk,
            comm_quant_mode,
            // Measurement-side ops carry 1.0: the pipeline evaluator applies
            // the configured factor exactly once.
            comm_overhead_factor: 1.0,
        };
        let shared_collective = |collective: AfdFCollectiveOp| AfdFCollectiveOp {
            hidden_size: config.hidden_size,
            a_replica_count,
            f_rank_count,
            gpus_per_node: config.topology.effective_f_gpus_per_node(),
            num_experts: config.num_experts,
            topk: config.topk,
            comm_quant_mode,
            ..collective
        };
        let token_mapping = AfdTokenMapping::new(
            a_replica_count,
            config.topology.a_pool.a_tp_size,
            config.topology.effective_f_gpus_per_node(),
        )?;
        let mut link_probe = crate::operators::communication::P2POp::new("afd_link", 2, 1);
        link_probe.span_gpus =
            Some(config.topology.a_pool.n_a_nodes * config.topology.gpus_per_node + f_rank_count);
        link_probe.force_scale_out =
            a_pool.engine.database().system != f_pool.engine.database().system;
        let fabric = AfdFabricLink::between(
            link_probe.fabric_parameters(&a_pool.engine.database().system_spec),
            link_probe.fabric_parameters(&f_pool.engine.database().system_spec),
        );
        let model = Self {
            dense_prefix_layers,
            selected_systems_roots: [None, None],
            token_mapping,
            fabric,
            a2f: shared_transfer(AfdTransferDirection::A2f, "afd_a2f_transfer"),
            f2a: shared_transfer(AfdTransferDirection::F2a, "afd_f2a_transfer"),
            f_ag: shared_collective(AfdFCollectiveOp::all_gather("afd_f_node_allgather", 1.0)),
            f_rs: shared_collective(AfdFCollectiveOp::reduce_scatter(
                "afd_f_node_reducescatter",
                1.0,
            )),
            a_combine: AfdCombineOp {
                name: "afd_a_side_combine".into(),
                scale_factor: 1.0,
                hidden_size: config.hidden_size,
                a_tp_size: config.topology.a_pool.a_tp_size,
                f_moe_ep_size: config.topology.f_pool.f_moe_ep_size,
                comm_quant_mode,
            },
            config,
            a_pool,
            f_pool,
            repetitions,
        };
        model.validate_communication_readiness()?;
        Ok(model)
    }

    fn validate_communication_readiness(&self) -> Result<(), AicError> {
        let check = |pool: &str,
                     db: &crate::perf_database::PerfDatabase,
                     dtype: CommQuantMode,
                     op: &str| {
            let result = match db.database_mode {
                DatabaseMode::Sol | DatabaseMode::SolFull => Ok(()),
                DatabaseMode::Empirical => db
                    .communication
                    .nccl_empirical_max_num_gpus(dtype, op)
                    .map(|_| ()),
                _ => db
                    .communication
                    .nccl_max_num_gpus(dtype, op)
                    .and_then(|bucket| {
                        bucket.map(|_| ()).ok_or_else(|| {
                            AicError::PerfDatabase(format!(
                                "no NCCL/OneCCL bucket dtype={} operation={op}",
                                dtype.name()
                            ))
                        })
                    }),
            };
            result.map_err(|e| {
                AicError::PerfDatabase(format!(
                    "AFD {pool} pool {} communication preflight ({op}, dtype={}): {e}",
                    db.system,
                    dtype.name()
                ))
            })
        };
        if self.f_ag.f_gpus_in_node() > 1
            && self.f_ag.rank_mapping == crate::operators::AfdRankMapping::OneToOne
        {
            for op in ["all_gather", "reduce_scatter"] {
                check(
                    "F",
                    self.f_pool.engine.database(),
                    self.f_ag.comm_quant_mode,
                    op,
                )?;
            }
        }
        if self.config.topology.a_pool.a_tp_size > 1 {
            check(
                "A",
                self.a_pool.engine.database(),
                CommQuantMode::Half,
                "all_gather",
            )?;
        }
        Ok(())
    }

    /// Physical resident weights per GPU, shared across phase checks.
    pub(crate) fn pool_weight_gb(&self, a_pool: bool) -> f64 {
        if a_pool {
            self.a_pool.weight_gb
        } else {
            self.f_pool.weight_gb
        }
    }

    pub fn predict_phase_metrics(
        &self,
        phase: AfdPhase,
        isl: u32,
        osl: u32,
        prefix: u32,
    ) -> Result<AfdPhaseMetrics, AicError> {
        self.measure_phase(phase, isl, osl, prefix, |_, _| {})
    }

    fn communication_work_scale(&self, phase: AfdPhase) -> f64 {
        if phase == AfdPhase::Prefill {
            1.0
        } else {
            (f64::from(self.config.num_layers) + f64::from(self.config.nextn))
                / f64::from(self.config.num_layers)
        }
    }

    pub fn predict_decode_step_ms(
        &self,
        batch_size: u32,
        s: u32,
        prefix: u32,
    ) -> Result<AfdPhaseMetrics, AicError> {
        self.measure_decode_step(batch_size, s, prefix, |_, _| {})
    }

    fn measure_decode_step(
        &self,
        batch_size: u32,
        s: u32,
        prefix: u32,
        visit: impl FnMut(&str, &PerformanceResult),
    ) -> Result<AfdPhaseMetrics, AicError> {
        if batch_size == 0 || s == 0 || prefix > s {
            return Err(AicError::InvalidEngineConfig(
                "AFD decode requires a positive batch/KV length and prefix <= KV length".into(),
            ));
        }
        let (m, _) = self.effective_microbatch_geometry(batch_size);
        self.measure_pool_plan(AfdPhase::Decode, batch_size, s, prefix, m, None, visit)
    }

    #[cfg(feature = "python")]
    pub(crate) fn predict_decode_step_with_operations(
        &self,
        batch: u32,
        s: u32,
        prefix: u32,
    ) -> Result<(AfdPhaseMetrics, Vec<AfdOpEvidence>), AicError> {
        let mut operations = Vec::new();
        let metrics = self.measure_decode_step(batch, s, prefix, |name, r| {
            operations.push(AfdOpEvidence::new(name, r))
        })?;
        Ok((metrics, operations))
    }

    #[cfg(feature = "python")]
    pub(crate) fn predict_phase_with_operations(
        &self,
        phase: AfdPhase,
        isl: u32,
        osl: u32,
        prefix: u32,
    ) -> Result<(AfdPhaseMetrics, Vec<AfdOpEvidence>), AicError> {
        let mut operations = std::collections::BTreeMap::<String, PerformanceResult>::new();
        let metrics = self.measure_phase(phase, isl, osl, prefix, |name, r| {
            let entry = operations
                .entry(name.to_owned())
                .or_insert_with(|| PerformanceResult::new(0.0, r.source));
            entry.latency_ms += r.latency_ms;
            entry.energy_wms += r.energy_wms;
            entry.source = entry.source.combine(r.source);
        })?;
        Ok((
            metrics,
            operations
                .into_iter()
                .map(|(name, r)| AfdOpEvidence::new(&name, &r))
                .collect(),
        ))
    }

    /// Split the pool-wide batch across A replicas and circulating microbatches.
    pub fn effective_microbatch_geometry(&self, pool_batch: u32) -> (u32, u32) {
        microbatch_geometry(&self.config, pool_batch)
    }
}

#[cfg(feature = "python")]
pub(crate) struct AfdOpEvidence {
    pub name: String,
    pub latency_ms: f64,
    pub energy_wms: f64,
    pub source: crate::operators::Source,
}

#[cfg(feature = "python")]
impl AfdOpEvidence {
    fn new(name: &str, r: &PerformanceResult) -> Self {
        Self {
            name: name.to_owned(),
            latency_ms: r.latency_ms,
            energy_wms: r.energy_wms,
            source: r.source,
        }
    }
}

/// Which pool a [`ForwardPassPerfModelConfig`] is derived for.
#[cfg(feature = "python")]
#[derive(Clone, Copy)]
enum PoolKind {
    A,
    F,
}

/// Derive one pool's [`ForwardPassPerfModelConfig`] from the AFD model
/// config: the pool's parallel identity + shared estimator plumbing +
/// the quant rule (only the caller's explicit overrides; never the
/// other pool's resolved values).
#[cfg(feature = "python")]
fn pool_fpm_config(
    config: &AfdForwardPassModelConfig,
    pool: PoolKind,
) -> Result<ForwardPassPerfModelConfig, AicError> {
    if matches!(
        config.estimator.estimation_mode,
        EstimationMode::FpmInterpolation | EstimationMode::FpmRegression
    ) {
        return Err(AicError::InvalidEngineConfig(
            "AFD per-pool estimation requires op_level: execution plans need granular op \
             lists; whole-model FPM modes are unsupported"
                .into(),
        ));
    }
    let (tp, moe_tp, moe_ep, system) = match pool {
        PoolKind::A => (
            config.topology.a_pool.a_tp_size,
            config.topology.a_pool.a_tp_size,
            1,
            config.topology.a_pool.system.clone(),
        ),
        PoolKind::F => {
            let f_pool_width =
                config.topology.f_pool.n_f_nodes * config.topology.effective_f_gpus_per_node();
            (
                f_pool_width,
                f_pool_width / config.topology.f_pool.f_moe_ep_size,
                config.topology.f_pool.f_moe_ep_size,
                config.topology.f_pool.system.clone(),
            )
        }
    };
    let mut fpm_config = ForwardPassPerfModelConfig::new(
        config.model.clone(),
        system,
        config.backend.clone(),
        ForwardPassWorkerType::Aggregated,
    );
    fpm_config.afd_routing_location = Some(config.topology.routing_location);
    fpm_config.afd_pool_role = Some(match pool {
        PoolKind::A => AfdPoolRole::A,
        PoolKind::F => AfdPoolRole::F,
    });
    fpm_config.enable_shared_layer = config.estimator.enable_shared_layer;
    fpm_config.strict_provenance = config.estimator.strict_provenance;
    fpm_config.backend_version = config.backend_version.clone();
    fpm_config.attention_backend = config.attention_backend.clone();
    fpm_config.moe_kernel_source = config.moe_kernel_source.clone();
    fpm_config.tp = tp;
    fpm_config.pp = 1;
    fpm_config.attention_dp = 1;
    fpm_config.moe_tp_size = Some(moe_tp);
    fpm_config.moe_ep_size = Some(moe_ep);
    // Quant rule: only the caller's explicit overrides reach the pool
    // configs; absent fields stay None for per-pool fresh resolution.
    fpm_config.gemm_quant_mode = config.quant.gemm_quant_mode.clone();
    fpm_config.moe_quant_mode = config.quant.moe_quant_mode.clone();
    fpm_config.fmha_quant_mode = config.quant.fmha_quant_mode.clone();
    fpm_config.kvcache_quant_mode = config.quant.kvcache_quant_mode.clone();
    fpm_config.comm_quant_mode = config.quant.comm_quant_mode.clone();
    fpm_config.nextn = config.nextn;
    fpm_config.database_mode = config.estimator.database_mode;
    fpm_config.transfer_policy = config.estimator.transfer_policy.clone();
    fpm_config.systems_paths = config.estimator.systems_paths.clone();
    fpm_config.estimation_mode = config.estimator.estimation_mode;
    fpm_config.fallback_policy = config.estimator.fallback_policy;
    fpm_config.estimator_config = config.estimator.estimator_config.clone();
    Ok(fpm_config)
}

/// A-node count times workers per node.
fn a_replica_count(config: &AfdForwardPassModelConfig) -> u32 {
    config.topology.a_pool.n_a_nodes * config.topology.gpus_per_node
        / config.topology.a_pool.a_tp_size
}

/// Split a pool-wide batch into `(microbatches per A replica, sequences per
/// microbatch)`. See
/// [`AfdForwardPassPerfModel::effective_microbatch_geometry`].
fn microbatch_geometry(config: &AfdForwardPassModelConfig, pool_batch_size: u32) -> (u32, u32) {
    let num_microbatches = config.topology.num_microbatches;
    // Geometry probes retain the minimum nonempty shape; actual decode queries
    // reject empty batches before evaluating any operations.
    let sequences_per_a_replica = pool_batch_size.max(1).div_ceil(a_replica_count(config));
    // This helper returns the busiest shape only. Evaluation retains the
    // exact 3 + 3 + 2 split for eight sequences. Two sequences use two
    // microbatches of size one; idle replicas contribute no F tokens.
    let m_eff = num_microbatches.min(sequences_per_a_replica);
    (m_eff, sequences_per_a_replica.div_ceil(m_eff))
}

#[cfg(test)]
mod geometry_tests {
    use super::*;
    use crate::AfdRoutingLocation;

    fn base_config() -> AfdForwardPassModelConfig {
        AfdForwardPassModelConfig {
            model: "Test/Model".into(),
            backend: BackendKind::Sglang,
            backend_version: None,
            attention_backend: None,
            moe_kernel_source: None,
            topology: Arc::new(AfdEngineConfig {
                a_pool: AfdAPoolSpec {
                    system: "h200_sxm".into(),
                    a_tp_size: 1,
                    n_a_nodes: 1,
                    batch_size_per_a_replica: 18,
                },
                f_pool: AfdFPoolSpec {
                    system: "h200_sxm".into(),
                    f_moe_ep_size: 1,
                    n_f_nodes: 1,
                },
                gpus_per_node: 1,
                f_gpus_per_node: None,
                num_microbatches: 3,
                pipeline_layout: AfdPipelineLayout::FourStage,
                comm_overhead_factor: 1.0,
                routing_location: AfdRoutingLocation::A,
            }),
            num_layers: 64,
            hidden_size: 5120,
            num_experts: 128,
            topk: 8,
            model_shape: Some(AfdModelShape {
                num_shared_experts: 0,
                num_moe_layers: 64,
            }),

            nextn: 0,
            quant: AfdQuantOverrides::default(),
            estimator: AfdEstimatorPlumbing::default(),
        }
    }

    #[test]
    fn model_config_rejects_invalid_geometry_before_pool_preparation() {
        let valid = base_config();
        valid.validate().expect("valid topology");
        for field in [
            "gpus_per_node",
            "num_layers",
            "hidden_size",
            "num_microbatches",
            "a_tp_size",
            "n_a_nodes",
            "batch_size_per_a_replica",
            "n_f_nodes",
            "f_moe_ep_size",
        ] {
            let mut cfg = valid.clone();
            match field {
                "gpus_per_node" => Arc::make_mut(&mut cfg.topology).gpus_per_node = 0,
                "num_layers" => cfg.num_layers = 0,
                "hidden_size" => cfg.hidden_size = 0,
                "num_microbatches" => Arc::make_mut(&mut cfg.topology).num_microbatches = 0,
                "a_tp_size" => Arc::make_mut(&mut cfg.topology).a_pool.a_tp_size = 0,
                "n_a_nodes" => Arc::make_mut(&mut cfg.topology).a_pool.n_a_nodes = 0,
                "batch_size_per_a_replica" => {
                    Arc::make_mut(&mut cfg.topology)
                        .a_pool
                        .batch_size_per_a_replica = 0
                }
                "n_f_nodes" => Arc::make_mut(&mut cfg.topology).f_pool.n_f_nodes = 0,
                _ => Arc::make_mut(&mut cfg.topology).f_pool.f_moe_ep_size = 0,
            }
            assert!(cfg.validate().unwrap_err().to_string().contains(field));
        }
        for mutate in [
            |cfg: &mut AfdForwardPassModelConfig| {
                Arc::make_mut(&mut cfg.topology).a_pool.a_tp_size = 3
            },
            |cfg: &mut AfdForwardPassModelConfig| {
                Arc::make_mut(&mut cfg.topology).f_pool.f_moe_ep_size = 3
            },
            |cfg: &mut AfdForwardPassModelConfig| {
                Arc::make_mut(&mut cfg.topology).num_microbatches = 19
            },
            |cfg: &mut AfdForwardPassModelConfig| {
                Arc::make_mut(&mut cfg.topology).a_pool.n_a_nodes = u32::MAX
            },
            |cfg: &mut AfdForwardPassModelConfig| {
                Arc::make_mut(&mut cfg.topology).f_pool.n_f_nodes = u32::MAX
            },
            |cfg: &mut AfdForwardPassModelConfig| {
                Arc::make_mut(&mut cfg.topology)
                    .a_pool
                    .batch_size_per_a_replica = u32::MAX
            },
            |cfg: &mut AfdForwardPassModelConfig| cfg.nextn = u32::MAX,
            |cfg: &mut AfdForwardPassModelConfig| {
                Arc::make_mut(&mut cfg.topology).comm_overhead_factor = 0.0
            },
            |cfg: &mut AfdForwardPassModelConfig| {
                Arc::make_mut(&mut cfg.topology).comm_overhead_factor = f64::NAN
            },
        ] {
            let mut cfg = valid.clone();
            // Two A replicas make the pool batch overflow in the batch-size case.
            Arc::make_mut(&mut cfg.topology).a_pool.n_a_nodes = 2;
            mutate(&mut cfg);
            assert!(cfg.validate().is_err());
        }
    }

    #[test]
    fn geometry_covers_full_partial_and_degenerate_batches() {
        let cfg = base_config();
        let geo = |pool| microbatch_geometry(&cfg, pool);
        // Full pool batch reproduces the frozen identity exactly.
        assert_eq!(geo(18), (3, 6));
        assert_eq!(geo(12), (3, 4));
        assert_eq!(geo(9), (3, 3));
        assert_eq!(geo(8), (3, 3));
        assert_eq!(geo(6), (3, 2));
        assert_eq!(geo(5), (3, 2));
        assert_eq!(geo(3), (3, 1));
        assert_eq!(geo(2), (2, 1));
        assert_eq!(geo(1), (1, 1));
        assert_eq!(geo(0), (1, 1));
        // Over-capacity: the pipeline depth still clamps at num_microbatches.
        assert_eq!(geo(36), (3, 12));
    }

    #[test]
    fn geometry_splits_pool_batch_across_a_replicas() {
        let mut cfg = base_config();
        Arc::make_mut(&mut cfg.topology).a_pool.n_a_nodes = 2;
        let geo = |pool| microbatch_geometry(&cfg, pool);
        // Two A replicas: the pool-wide batch is halved per A replica, so a full
        // pool matches the single-worker geometry at half the pool size.
        assert_eq!(geo(36), (3, 6));
        assert_eq!(geo(18), (3, 3));
        assert_eq!(geo(16), (3, 3));
        assert_eq!(geo(5), (3, 1));
        assert_eq!(geo(4), (2, 1));
        assert_eq!(geo(2), (1, 1));
    }

    #[test]
    fn geometry_single_microbatch_stays_serial() {
        let mut cfg = base_config();
        Arc::make_mut(&mut cfg.topology).num_microbatches = 1;
        assert_eq!(microbatch_geometry(&cfg, 18), (1, 18));
        assert_eq!(microbatch_geometry(&cfg, 3), (1, 3));
    }

    #[test]
    fn geometry_matches_the_frozen_identity_when_not_divisible() {
        // An uneven split retains the configured count and prices every
        // microbatch at the largest size, like the frozen measurement.
        let mut cfg = base_config();
        Arc::make_mut(&mut cfg.topology).num_microbatches = 7;
        // 18 sequences split into 7 microbatches: the largest holds 3.
        assert_eq!(microbatch_geometry(&cfg, 18), (7, 3));

        let mut cfg = base_config();
        Arc::make_mut(&mut cfg.topology)
            .a_pool
            .batch_size_per_a_replica = 7;
        Arc::make_mut(&mut cfg.topology).num_microbatches = 5;
        // 7 sequences split into 5 microbatches: the largest holds 2.
        assert_eq!(microbatch_geometry(&cfg, 7), (5, 2));
    }

    #[test]
    fn geometry_depends_on_sequence_count_not_configured_batch_size() {
        let mut cfg = base_config();
        for batch_size_per_a_replica in [3, 8, 18, 64] {
            Arc::make_mut(&mut cfg.topology)
                .a_pool
                .batch_size_per_a_replica = batch_size_per_a_replica;
            assert_eq!(microbatch_geometry(&cfg, 8), (3, 3));
            assert_eq!(microbatch_geometry(&cfg, 2), (2, 1));
        }
        Arc::make_mut(&mut cfg.topology).num_microbatches = 7;
        assert_eq!(microbatch_geometry(&cfg, 17), (7, 3));
        assert_eq!(microbatch_geometry(&cfg, 3), (3, 1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AfdRoutingLocation;
    use crate::perf_database::PerfDatabase;
    use crate::perfmodel::engine::spec::EngineSpec;
    use std::path::PathBuf;

    /// Repo systems root (committed parquet data), mirroring the readiness
    /// tests' CARGO_MANIFEST_DIR-relative access.
    fn systems_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/aisimulate/src/aisimulate_core/systems")
    }

    fn golden_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("parity_tests/perfmodel/goldens")
    }

    fn load_db(system: &str) -> Arc<PerfDatabase> {
        Arc::new(
            PerfDatabase::load(&systems_root(), system, "sglang", "0.5.14")
                .unwrap_or_else(|e| panic!("load {system} db: {e}")),
        )
    }

    fn build_engine(spec_file: &str, db: &Arc<PerfDatabase>) -> Arc<Engine> {
        let path = golden_dir().join("afd_specs").join(spec_file);
        let json = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "read {}: {e} — regenerate via gen_afd_spec_fixtures.py",
                path.display()
            )
        });
        let spec: EngineSpec =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        Arc::new(Engine::build(spec, db.clone()).expect("engine build"))
    }

    fn pure_config() -> AfdForwardPassModelConfig {
        AfdForwardPassModelConfig {
            model: "Qwen/Qwen3-30B-A3B".into(),
            backend: BackendKind::Sglang,
            backend_version: Some("0.5.14".into()),
            attention_backend: None,
            moe_kernel_source: None,
            topology: Arc::new(AfdEngineConfig {
                a_pool: AfdAPoolSpec {
                    system: "h200_sxm".into(),
                    a_tp_size: 1,
                    n_a_nodes: 1,
                    batch_size_per_a_replica: 18,
                },
                f_pool: AfdFPoolSpec {
                    system: "h200_sxm".into(),
                    f_moe_ep_size: 8,
                    n_f_nodes: 1,
                },
                gpus_per_node: 8,
                f_gpus_per_node: None,
                num_microbatches: 3,
                pipeline_layout: AfdPipelineLayout::FourStage,
                comm_overhead_factor: 1.0,
                routing_location: AfdRoutingLocation::A,
            }),
            num_layers: 48,
            hidden_size: 2048,
            num_experts: 128,
            topk: 8,
            model_shape: Some(AfdModelShape {
                num_shared_experts: 0,
                num_moe_layers: 48,
            }),

            nextn: 0,
            quant: AfdQuantOverrides::default(),
            estimator: AfdEstimatorPlumbing::default(),
        }
    }

    fn assert_close(field: &str, actual: f64, expected: f64) {
        let delta = (actual - expected).abs();
        let limit = 1e-9 * expected.abs().max(1e-300);
        assert!(
            delta <= limit,
            "{field}: actual={actual} expected={expected} abs_delta={delta:.3e} rtol=1e-9"
        );
    }

    /// Synthetic source layout: one dense layer, then 47 routed layers with a
    /// shared branch, A TP2 and two F nodes. Reuse identical compute graphs at equal token
    /// shapes so only the seven injected operations gain the hand-derived
    /// 17/16 communication work scale. In particular, the compiled TP collective
    /// and one-time embedding/head must not gain a second scale.
    #[test]
    fn mtp_communication_scales_selective_plan_without_double_scaling_local_work() {
        use super::super::plan::{AfdExecutionPlan, AfdLayerTemplate, AfdPhasePlan};
        use crate::operators::{CustomAllReduceOp, ElementwiseOp, Op};

        let db = load_db("h200_sxm");
        let mut config = pure_config();
        Arc::make_mut(&mut config.topology).a_pool.a_tp_size = 2;
        Arc::make_mut(&mut config.topology).f_pool.n_f_nodes = 2;
        Arc::make_mut(&mut config.topology).f_pool.f_moe_ep_size = 16;
        config.model_shape = Some(AfdModelShape {
            num_shared_experts: 1,
            num_moe_layers: 47,
        });
        let mut engines = Vec::new();
        for role in [AfdPoolRole::A, AfdPoolRole::F] {
            let file = if role == AfdPoolRole::A {
                "moe_pure_h200_a_sglang.json"
            } else {
                "moe_pure_h200_f_sglang.json"
            };
            let mut spec: EngineSpec = serde_json::from_str(
                &std::fs::read_to_string(golden_dir().join("afd_specs").join(file)).unwrap(),
            )
            .unwrap();
            let a_pool = role == AfdPoolRole::A;
            spec.engine.extra.insert(
                "afd_pool_role".into(),
                if a_pool { "a" } else { "f" }.into(),
            );
            if a_pool {
                spec.engine.parallel.tp_size = 2;
                spec.engine.parallel.moe_tp_size = Some(2);
                spec.context_ops = vec![
                    Op::Elementwise(ElementwiseOp::new("attention", 4096.)),
                    Op::CustomAllReduce(CustomAllReduceOp::new("afd_attn_ar", 1., 2048, 2)),
                    Op::Elementwise(ElementwiseOp::new("shared", 256.)),
                    Op::Elementwise(ElementwiseOp::new("embedding", 4096.)),
                    Op::Elementwise(ElementwiseOp::new("logits", 4096.)),
                ];
            } else {
                spec.engine.parallel.tp_size = 16;
                spec.engine.parallel.moe_ep_size = Some(16);
                let mut moe = spec
                    .context_ops
                    .iter()
                    .find(|op| matches!(op, Op::Moe(_)))
                    .unwrap()
                    .clone();
                moe.set_scale_factor(1.);
                if let Op::Moe(op) = &mut moe {
                    op.moe_ep_size = 16;
                }
                spec.context_ops = vec![moe];
            }
            spec.generation_ops = spec.context_ops.clone();
            for op in &mut spec.generation_ops {
                op.set_scale_factor(17. / 16.);
            }
            let template = |routed| AfdLayerTemplate {
                attention_kind: "global".into(),
                routed,
                ops: if a_pool {
                    vec![0, 1]
                } else if routed {
                    vec![0]
                } else {
                    vec![]
                },
                rank_ops: if a_pool && routed { vec![2] } else { vec![] },
                work_scale: 1.,
            };
            let phase = AfdPhasePlan {
                templates: vec![template(false), template(true)],
                prefix: if a_pool { vec![3] } else { vec![] },
                suffix: if a_pool { vec![4] } else { vec![] },
            };
            let mut generation = phase.clone();
            for t in &mut generation.templates {
                t.work_scale = 17. / 16.;
            }
            spec.afd_plan = Some(AfdExecutionPlan {
                role,
                model: config.model.clone(),
                hidden_size: 2048,
                num_experts: 128,
                topk: 8,
                shared_inter_size: 16,
                layers: std::iter::once(0)
                    .chain(std::iter::repeat_n(1, 47))
                    .collect(),
                context: phase,
                generation,
            });
            engines.push(Arc::new(Engine::build(spec, db.clone()).unwrap()));
        }
        for layout in [AfdPipelineLayout::FourStage, AfdPipelineLayout::TwoStage] {
            Arc::make_mut(&mut config.topology).pipeline_layout = layout;
            Arc::make_mut(&mut config.topology).comm_overhead_factor = 1.75;
            config.nextn = 0;
            let plain = AfdForwardPassPerfModel::from_engines(
                config.clone(),
                engines[0].clone(),
                engines[1].clone(),
            )
            .unwrap();
            config.nextn = 3;
            let mtp = AfdForwardPassPerfModel::from_engines(
                config.clone(),
                engines[0].clone(),
                engines[1].clone(),
            )
            .unwrap();
            assert_eq!(
                mtp.predict_phase_metrics(AfdPhase::Prefill, 128, 2, 64)
                    .unwrap(),
                plain
                    .predict_phase_metrics(AfdPhase::Prefill, 128, 2, 64)
                    .unwrap(),
            );
            let reference = plain.predict_decode_step_ms(288, 129, 0).unwrap();
            let actual = mtp.predict_decode_step_ms(72, 129, 0).unwrap();
            assert_close(
                "A transfer",
                actual.layer_times.a_to_f_ms,
                reference.layer_times.a_to_f_ms * 17. / 16.,
            );
            assert_close(
                "F return",
                actual.layer_times.f_to_a_ms,
                reference.layer_times.f_to_a_ms * 17. / 16.,
            );
            assert!(actual.billed_step_ms > reference.billed_step_ms);
            let mut reference_ops = std::collections::BTreeMap::new();
            let mut actual_ops = std::collections::BTreeMap::new();
            plain
                .measure_plan(AfdPhase::Decode, 24, 72, 129, 0, 3, None, |name, r| {
                    reference_ops.insert(name.to_owned(), r.clone());
                })
                .unwrap();
            mtp.measure_plan(AfdPhase::Decode, 24, 18, 129, 0, 3, None, |name, r| {
                actual_ops.insert(name.to_owned(), r.clone());
            })
            .unwrap();
            let mut injected = 0;
            for (name, reference) in &reference_ops {
                let actual = &actual_ops[name];
                let factor = if name == "afd_attn_ar" || !name.starts_with("afd_") {
                    1.
                } else {
                    injected += 1;
                    17. / 16.
                };
                assert!(reference.latency_ms > 0., "{name}");
                assert_close(name, actual.latency_ms, reference.latency_ms * factor);
                assert_eq!(actual.energy_wms, reference.energy_wms * factor);
                assert_eq!(actual.source, reference.source);
            }
            assert_eq!(injected, 7);
        }
    }

    #[test]
    fn direct_model_construction_and_queries_reject_invalid_inputs() {
        let db = load_db("h200_sxm");
        let a_engine = build_engine("moe_pure_h200_a_sglang.json", &db);
        let f_engine = build_engine("moe_pure_h200_f_sglang.json", &db);
        for field in ["num_layers", "num_microbatches", "a_tp_size"] {
            let mut config = pure_config();
            match field {
                "num_layers" => config.num_layers = 0,
                "num_microbatches" => Arc::make_mut(&mut config.topology).num_microbatches = 0,
                _ => Arc::make_mut(&mut config.topology).a_pool.a_tp_size = 0,
            }
            let error =
                AfdForwardPassPerfModel::from_engines(config, a_engine.clone(), f_engine.clone())
                    .err()
                    .expect("invalid model must reject before querying");
            assert!(error.to_string().contains(field));
        }
        let model =
            AfdForwardPassPerfModel::from_engines(pure_config(), a_engine, f_engine).unwrap();
        assert!(model.predict_decode_step_ms(0, 1024, 0).is_err());
        for phase in [AfdPhase::Prefill, AfdPhase::Decode] {
            for (isl, osl, prefix) in [(0, 128, 0), (1024, 0, 0), (1024, 128, 1025)] {
                assert!(
                    model
                        .predict_phase_metrics(phase, isl, osl, prefix)
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn legacy_spec_without_a_plan_requires_recompilation() {
        let db = load_db("h200_sxm");
        let mut spec: EngineSpec = serde_json::from_str(
            &std::fs::read_to_string(golden_dir().join("afd_specs/moe_pure_h200_a_sglang.json"))
                .unwrap(),
        )
        .unwrap();
        spec.afd_plan = None;
        spec.engine.extra.remove("afd_pool_role");
        let a = Arc::new(Engine::build(spec, db.clone()).unwrap());
        let f = build_engine("moe_pure_h200_f_sglang.json", &db);
        let error = AfdForwardPassPerfModel::from_engines(pure_config(), a, f)
            .err()
            .unwrap();
        assert!(error.to_string().contains("recompile"));
    }

    #[test]
    fn compiled_pool_identity_and_missing_model_facts_reject() {
        let db = load_db("h200_sxm");
        let a = build_engine("moe_pure_h200_a_sglang.json", &db);
        let f = build_engine("moe_pure_h200_f_sglang.json", &db);
        let mut missing = pure_config();
        missing.model_shape = None;
        let error = AfdForwardPassPerfModel::from_engines(missing, a.clone(), f.clone())
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("requires source-derived model_shape")
        );
        let mut wrong_model = pure_config();
        wrong_model.model = "Other/CompatibleShape".into();
        let error = AfdForwardPassPerfModel::from_engines(wrong_model, a.clone(), f)
            .err()
            .unwrap();
        assert!(error.to_string().contains("source identity"));
        let path = golden_dir().join("afd_specs/moe_pure_h200_f_sglang.json");
        let mut spec: EngineSpec =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        spec.afd_plan.as_mut().unwrap().model = "Other/CompatibleShape".into();
        spec.engine.model_name = "Other/CompatibleShape".into();
        let wrong_source = Arc::new(Engine::build(spec.clone(), db.clone()).unwrap());
        let error = AfdForwardPassPerfModel::from_engines(pure_config(), a.clone(), wrong_source)
            .err()
            .unwrap();
        assert!(error.to_string().contains("source identity"));
        spec.engine.parallel.tp_size = 4;
        spec.engine.parallel.moe_ep_size = Some(4);
        let mismatched = Arc::new(Engine::build(spec, db).unwrap());
        let error = AfdForwardPassPerfModel::from_engines(pure_config(), a, mismatched)
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("F-pool compiled parallel identity")
        );
    }

    /// Invariant behind the continuous-batching geometry clamp: at the
    /// saturation point (with at least one sequence per configured microbatch)
    /// the per-pass evaluation and the frozen stride-averaged identity price
    /// the same step. An uneven split still uses the largest microbatch size.
    #[test]
    fn saturated_continuous_step_matches_the_frozen_identity() {
        let db = load_db("h200_sxm");
        let a_engine = build_engine("moe_pure_h200_a_sglang.json", &db);
        let f_engine = build_engine("moe_pure_h200_f_sglang.json", &db);
        let mut cfg = pure_config();
        Arc::make_mut(&mut cfg.topology).num_microbatches = 7;
        let model =
            AfdForwardPassPerfModel::from_engines(cfg, a_engine, f_engine).expect("afd model");

        let isl = 1024;
        // a_replica_count = 1 * 8 / 8 = 1, so the saturated pool-wide batch is
        // exactly batch_size_per_a_replica.
        let pool = 144;
        assert_eq!(model.effective_microbatch_geometry(pool), (7, 3));

        // osl=2 collapses integrate_decode's stride quadrature
        // (AFD_DECODE_STRIDE = 32) to a SINGLE sample at s = isl + 1, which is
        // the only point where the stride-averaged frozen identity and the
        // instantaneous per-pass evaluation are the same number.
        let frozen = model
            .predict_phase_metrics(AfdPhase::Decode, isl, 2, 0)
            .expect("frozen decode");
        let continuous = model
            .predict_decode_step_ms(pool, isl + 1, 0)
            .expect("continuous decode");
        assert_close("t_step_ms", continuous.t_step_ms, frozen.t_step_ms);
        assert_close(
            "billed_step_ms",
            continuous.billed_step_ms,
            frozen.billed_step_ms,
        );
        assert_close(
            "attention_ms",
            continuous.layer_times.attention_ms,
            frozen.layer_times.attention_ms,
        );
        assert_close(
            "ffn_ms",
            continuous.layer_times.ffn_ms,
            frozen.layer_times.ffn_ms,
        );
    }

    /// One sequence per A replica runs all stages serially. Both measurement and
    /// scheduler paths use the sequential sum of compute and transfer times.
    #[test]
    fn single_microbatch_uses_the_sequential_cadence() {
        let db = load_db("h200_sxm");
        let a_engine = build_engine("moe_pure_h200_a_sglang.json", &db);
        let f_engine = build_engine("moe_pure_h200_f_sglang.json", &db);
        let mut cfg = pure_config();
        Arc::make_mut(&mut cfg.topology).num_microbatches = 1;
        let model =
            AfdForwardPassPerfModel::from_engines(cfg, a_engine, f_engine).expect("afd model");

        let times = model
            .predict_phase_metrics(AfdPhase::Decode, 1024, 128, 0)
            .expect("layer times")
            .layer_times;
        let serial = times.attention_ms + times.a_to_f_ms + times.ffn_ms + times.f_to_a_ms;
        let metrics = model
            .predict_phase_metrics(AfdPhase::Decode, 1024, 2, 0)
            .unwrap();
        let sample = model.predict_decode_step_ms(144, 1025, 0).unwrap();
        assert_eq!(metrics.t_step_ms, sample.t_step_ms);
        // The stride-averaged ingredient sum gives the same serial total.
        let full = model
            .predict_phase_metrics(AfdPhase::Decode, 1024, 128, 0)
            .unwrap();
        assert_close(
            "serial step",
            full.t_step_ms,
            serial * f64::from(times.num_layers),
        );
    }

    /// Supported pure MoE DP+EP graph queries both phases on the SGLang DB.
    #[test]
    fn sglang_pure_moe_has_finite_positive_layer_times() {
        let db =
            Arc::new(PerfDatabase::load(&systems_root(), "h200_sxm", "sglang", "0.5.14").unwrap());
        let mut config = pure_config();
        config.backend = BackendKind::Sglang;
        config.backend_version = Some("0.5.14".into());
        let model = AfdForwardPassPerfModel::from_engines(
            config,
            build_engine("moe_pure_h200_a_sglang.json", &db),
            build_engine("moe_pure_h200_f_sglang.json", &db),
        )
        .unwrap();
        for phase in [AfdPhase::Prefill, AfdPhase::Decode] {
            let times = model
                .predict_phase_metrics(phase, 1024, 128, 0)
                .unwrap()
                .layer_times;
            assert_eq!(times.num_layers, 48);
            for time in [
                times.attention_ms,
                times.ffn_ms,
                times.a_to_f_ms,
                times.f_to_a_ms,
            ] {
                assert!(time.is_finite() && time > 0.0);
            }
        }
    }

    #[test]
    fn paired_provenance_preserves_each_pool_and_only_publishes_a_shared_version() {
        for (a_version, f_version, shared) in [
            ("0.5.14", "0.5.14", Some("0.5.14")),
            ("0.5.14", "0.5.13", None),
            ("", "", None),
        ] {
            let a = serde_json::json!({"system": "h200_sxm", "backend_version": a_version, "data_root": "/a/data"});
            let f = serde_json::json!({"system": "b200_sxm", "backend_version": f_version, "data_root": "/f/data"});
            let provenance = paired_prediction_provenance(a.clone(), f.clone());
            assert_eq!(provenance["estimator"], "op_level");
            assert_eq!(provenance["timing_source"], "afd");
            assert_eq!(provenance["data_version"].as_str(), shared);
            assert_eq!(provenance["data_identity"]["a_pool"], a);
            assert_eq!(provenance["data_identity"]["f_pool"], f);
            assert_eq!(
                provenance["data_version_unavailable_reason"].is_null(),
                shared.is_some()
            );
        }
    }

    /// Hetero gate: with the F pool compiled against b200_sxm, the
    /// attention terms still come from the A pool's h200 DB while the FFN
    /// terms come from the F pool's b200 DB (the per-pool
    /// quant/data resolution rule).
    #[test]
    fn hetero_f_pool_reads_its_own_system_db() {
        let a_db = load_db("h200_sxm");
        let f_db = load_db("b200_sxm");
        let a_engine = build_engine("moe_pure_h200_a_sglang.json", &a_db);
        let mut config = pure_config();
        Arc::make_mut(&mut config.topology).f_pool.system = "b200_sxm".into();
        let hetero = AfdForwardPassPerfModel::from_engines(
            config,
            a_engine.clone(),
            build_engine("moe_hetero_b200_f_sglang.json", &f_db),
        )
        .unwrap();
        let homo = AfdForwardPassPerfModel::from_engines(
            pure_config(),
            a_engine,
            build_engine("moe_pure_h200_f_sglang.json", &a_db),
        )
        .unwrap();
        for phase in [AfdPhase::Prefill, AfdPhase::Decode] {
            let actual = hetero
                .predict_phase_metrics(phase, 1024, 128, 0)
                .unwrap()
                .layer_times;
            let reference = homo
                .predict_phase_metrics(phase, 1024, 128, 0)
                .unwrap()
                .layer_times;
            assert_close(
                "A compute remains on A DB",
                actual.attention_ms,
                reference.attention_ms,
            );
            assert!(actual.ffn_ms > 0.0 && reference.ffn_ms > 0.0);
            assert_eq!(hetero.f_pool.engine.database().system, "b200_sxm");
        }
    }

    /// Enforcement: a topology enumerated against the wrong node
    /// width must fail loudly instead of pricing against a different SKU.
    #[test]
    fn gpus_per_node_mismatch_rejects() {
        for pool in ["a", "f"] {
            let mut a_db = load_db("h200_sxm");
            let mut f_db = load_db("h200_sxm");
            let mismatched_db = if pool == "a" { &mut a_db } else { &mut f_db };
            Arc::get_mut(mismatched_db)
                .unwrap()
                .tables_mut()
                .system_spec
                .node
                .num_gpus_per_node = 4;
            // Keep topology and compiled rank identity consistent so this reaches
            // the per-pool hardware check rather than an earlier validation.
            let result = AfdForwardPassPerfModel::from_engines(
                pure_config(),
                build_engine("moe_pure_h200_a_sglang.json", &a_db),
                build_engine("moe_pure_h200_f_sglang.json", &f_db),
            );
            match result {
                Err(AicError::InvalidEngineConfig(message)) => assert_eq!(
                    message,
                    format!(
                        "AFD {pool}-pool topology gpus_per_node=8 disagrees with the \
                         'h200_sxm' system spec, which reports 4"
                    ),
                ),
                Err(error) => panic!("expected invalid {pool}-pool topology, got {error}"),
                Ok(_) => panic!("accepted invalid {pool}-pool topology"),
            }
        }
    }

    /// Hand timelines: samples [4,1,1,1] and [1,1,4,1], m=4, L=2
    /// each finish at 35 (unscaled) or 37 (factor=2). Their averaged stages
    /// instead finish at 24.5 or 26.5, so billing must average evaluated passes.
    #[test]
    fn stride_average_evaluates_each_pipeline_before_averaging() {
        let mut average = PhaseAverage::default();
        for (attention_ms, ffn_ms, repeat) in [(4., 1., 32), (1., 4., 31)] {
            let sample = AfdPhaseMetrics::from_layer_times(
                AfdPipelineLayout::FourStage,
                AfdLayerTimes {
                    attention_ms,
                    ffn_ms,
                    a_to_f_ms: 1.,
                    f_to_a_ms: 1.,
                    num_layers: 2,
                },
                4,
                2.,
            )
            .unwrap();
            assert_eq!(sample.t_step_ms, 35.);
            assert_eq!(sample.billed_step_ms, 37.);
            average.push(&sample, repeat);
        }
        let metrics = average.finish(2);
        assert_eq!(metrics.t_step_ms, 35.);
        assert_eq!(metrics.billed_step_ms, 37.);
        assert_ne!(
            AfdPipelineLayout::FourStage
                .step_latency_ms(&metrics.layer_times, 4, 2.)
                .unwrap(),
            metrics.billed_step_ms
        );
    }

    fn wire_config() -> AfdTimingConfig {
        serde_json::from_value(serde_json::json!({
            "model": "Qwen/Qwen3-30B-A3B", "isl": 1024, "osl": 128,
            "num_layers": 48, "hidden_size": 2048, "num_experts": 128, "topk": 8,
            "model_shape": {"num_shared_experts": 0, "num_moe_layers": 48},
        }))
        .unwrap()
    }

    #[test]
    fn provider_reads_one_shared_topology_and_has_no_rank_identity_fields() {
        let topology = pure_config().topology;
        let json = serde_json::to_value(wire_config()).unwrap();
        for key in [
            "a_pool",
            "f_pool",
            "gpus_per_node",
            "pipeline_layout",
            "nextn",
            "backend",
            "comm_quant_mode",
            "max_seq_len",
        ] {
            assert!(json.get(key).is_none(), "duplicate provider input {key}");
        }
        let config = wire_config()
            .into_perf_config(Arc::clone(&topology), BackendKind::Sglang, 2)
            .unwrap();
        assert!(Arc::ptr_eq(&topology, &config.topology));
        assert_eq!(config.nextn, 2);
        assert_eq!(config.backend, BackendKind::Sglang);
    }

    #[test]
    fn communication_precision_has_one_explicit_source() {
        let topology = pure_config().topology;
        for (mode, expected) in [
            (None, CommQuantMode::Half),
            (Some("half"), CommQuantMode::Half),
            (Some("fp8"), CommQuantMode::Fp8),
            (Some("int8"), CommQuantMode::Int8),
        ] {
            let mut wire = wire_config();
            wire.quant.comm_quant_mode = mode.map(str::to_owned);
            let config = wire
                .into_perf_config(Arc::clone(&topology), BackendKind::Sglang, 0)
                .unwrap();
            assert_eq!(config.quant.resolved_comm_quant_mode().unwrap(), expected);
            assert_eq!(config.quant.comm_quant_mode.as_deref(), mode);
            #[cfg(feature = "python")]
            for pool in [PoolKind::A, PoolKind::F] {
                assert_eq!(
                    pool_fpm_config(&config, pool)
                        .unwrap()
                        .comm_quant_mode
                        .as_deref(),
                    mode
                );
            }
        }
        let mut wire = wire_config();
        wire.quant.comm_quant_mode = Some("invalid".into());
        assert!(
            wire.into_perf_config(topology, BackendKind::Sglang, 0)
                .unwrap_err()
                .to_string()
                .contains("comm_quant_mode")
        );
    }

    #[test]
    fn kernel_selectors_round_trip_without_pinning_defaults() {
        let mut wire = wire_config();
        wire.estimator.enable_shared_layer = Some(false);
        wire.estimator.strict_provenance = true;
        let value = serde_json::to_value(&wire).unwrap();
        assert!(value.get("attention_backend").is_none());
        assert!(value.get("moe_kernel_source").is_none());
        wire.attention_backend = Some("fa3".into());
        wire.moe_kernel_source = Some("sglang_fused_moe_triton".into());
        let restored: AfdTimingConfig =
            serde_json::from_value(serde_json::to_value(&wire).unwrap()).unwrap();
        let config = restored
            .into_perf_config(pure_config().topology, BackendKind::Sglang, 0)
            .unwrap();
        assert_eq!(config.estimator.enable_shared_layer, Some(false));
        assert!(config.estimator.strict_provenance);
        assert_eq!(config.attention_backend, wire.attention_backend);
        assert_eq!(config.moe_kernel_source, wire.moe_kernel_source);
    }

    #[cfg(feature = "python")]
    #[test]
    fn pool_configs_preserve_kernel_selectors_and_canonical_validation() {
        let mut config = pure_config();
        config.backend = BackendKind::Sglang;
        config.estimator.enable_shared_layer = Some(false);
        config.estimator.strict_provenance = true;
        config.attention_backend = Some("fa3".into());
        config.moe_kernel_source = Some("sglang_fused_moe_triton".into());
        for pool in [PoolKind::A, PoolKind::F] {
            let model = pool_fpm_config(&config, pool).unwrap();
            assert_eq!(model.enable_shared_layer, Some(false));
            assert!(model.strict_provenance);
            assert_eq!(
                model.afd_pool_role,
                Some(match pool {
                    PoolKind::A => AfdPoolRole::A,
                    PoolKind::F => AfdPoolRole::F,
                })
            );
            assert_eq!(model.attention_backend, config.attention_backend);
            assert_eq!(model.moe_kernel_source, config.moe_kernel_source);
            model.validate().unwrap();
        }
        config.moe_kernel_source = Some(" \t".into());
        let error = pool_fpm_config(&config, PoolKind::F)
            .unwrap()
            .validate()
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("moe_kernel_source cannot be empty")
        );
    }

    #[test]
    fn into_perf_config_rejects_invalid_workload_and_rank_routing() {
        for (isl, osl) in [(0, 128), (1024, 0)] {
            let mut wire = wire_config();
            wire.isl = isl;
            wire.osl = osl;
            assert!(
                wire.into_perf_config(pure_config().topology, BackendKind::Sglang, 0)
                    .is_err()
            );
        }
        let mut topology = (*pure_config().topology).clone();
        topology.routing_location = AfdRoutingLocation::F;
        assert!(matches!(
            wire_config().into_perf_config(Arc::new(topology), BackendKind::Sglang, 0),
            Err(AicError::NotImplemented(_))
        ));
    }
    #[test]
    fn independent_f_width_drives_collectives_buffers_and_gpu_accounting() {
        let a_db = load_db("h200_sxm");
        let mut f_db = load_db("h200_sxm");
        Arc::get_mut(&mut f_db)
            .unwrap()
            .tables_mut()
            .system_spec
            .node
            .num_gpus_per_node = 4;
        let mut cfg = pure_config();
        let topology = Arc::make_mut(&mut cfg.topology);
        topology.f_gpus_per_node = Some(4);
        topology.f_pool.n_f_nodes = 2; // same eight F ranks; four GPUs per node
        let model = AfdForwardPassPerfModel::from_engines(
            cfg,
            build_engine("moe_pure_h200_a_sglang.json", &a_db),
            build_engine("moe_pure_h200_f_sglang.json", &f_db),
        )
        .unwrap();
        assert_eq!(model.f_ag.f_gpus_in_node(), 4);
        assert_eq!(model.a2f.num_f_nodes(), 2);
        assert_eq!(model.token_mapping.busiest(7), 14); // eight TP1 senders / four NICs
        let (peak, held) = model.f_plan_buffers(7);
        assert!(peak > 0 && held > 0);
        let topology = &model.config.topology;
        assert_eq!(
            crate::engine::afd_total_gpus(
                &topology.a_pool,
                &topology.f_pool,
                topology.gpus_per_node,
                topology.effective_f_gpus_per_node(),
            ),
            16
        );
    }

    #[test]
    fn prepared_fabric_selects_pool_span_and_forces_cross_system_scale_out() {
        // Two eight-GPU nodes: total span 16. A 16-GPU rack fits exactly;
        // a 12-GPU rack spills; different systems cannot share even a 72-GPU rack.
        for (cross_system, rack, bw, latency) in [
            (false, 16, 100e9, 10e-6),
            (false, 12, 10e9, 30e-6),
            (true, 72, 10e9, 30e-6),
        ] {
            let mut a_db = load_db("h200_sxm");
            let mut f_db = load_db(if cross_system { "b200_sxm" } else { "h200_sxm" });
            for db in [&mut a_db, &mut f_db] {
                let node = &mut Arc::get_mut(db).unwrap().tables_mut().system_spec.node;
                node.intra_node_bw = 1000e9;
                node.inter_node_bw = 100e9;
                node.num_gpus_per_rack = Some(rack);
                node.inter_rack_bw = Some(10e9);
                node.p2p_latency = 10e-6;
                node.inter_rack_latency = Some(30e-6);
            }
            let mut cfg = pure_config();
            Arc::make_mut(&mut cfg.topology).f_pool.system = f_db.system.clone();
            let model = AfdForwardPassPerfModel::from_engines(
                cfg,
                build_engine("moe_pure_h200_a_sglang.json", &a_db),
                build_engine(
                    if cross_system {
                        "moe_hetero_b200_f_sglang.json"
                    } else {
                        "moe_pure_h200_f_sglang.json"
                    },
                    &f_db,
                ),
            )
            .unwrap();
            assert_eq!(model.fabric.inter_node_bw, bw);
            assert_eq!(model.fabric.p2p_latency, latency);
            assert_eq!(model.fabric.peer, None);
        }
    }
}

fn default_true() -> bool {
    true
}

/// Wire configuration of the `"afd"` timing provider: the
/// JSON payload the compiler emits as `timing_model.config` for AFD roles.
/// Converts 1:1 into [`AfdForwardPassModelConfig`]; `isl`/`osl` additionally
/// seed the provider's frozen measurement identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AfdTimingConfig {
    pub model: String,
    #[serde(default)]
    pub backend_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe_kernel_source: Option<String>,
    /// Measured workload point (the frozen measurement identity).
    pub isl: u32,
    pub osl: u32,
    pub num_layers: u32,
    pub hidden_size: u32,
    #[serde(default)]
    pub num_experts: u32,
    #[serde(default)]
    pub topk: u32,
    #[serde(default)]
    pub model_shape: Option<AfdModelShape>,
    /// Decode-identity choice: `true` (default) keeps the frozen
    /// stride-averaged decode step for fixed synthetic workloads; `false`
    /// (trace/agentic) evaluates the pipeline instantaneously per pass.
    #[serde(default = "default_true")]
    pub frozen_decode_identity: bool,
    #[serde(default)]
    pub quant: AfdQuantOverrides,
    #[serde(default)]
    pub estimator: AfdEstimatorPlumbing,
}

impl AfdTimingConfig {
    /// Consume provider inputs once, reading rank-owned identity and topology.
    pub fn into_perf_config(
        self,
        topology: Arc<AfdEngineConfig>,
        backend: BackendKind,
        nextn: u32,
    ) -> Result<AfdForwardPassModelConfig, AicError> {
        for (name, value) in [("isl", self.isl), ("osl", self.osl)] {
            if value == 0 {
                return Err(AicError::InvalidEngineConfig(format!(
                    "afd timing config: {name} must be positive"
                )));
            }
        }
        let config = AfdForwardPassModelConfig {
            model: self.model,
            backend,
            backend_version: self.backend_version,
            attention_backend: self.attention_backend,
            moe_kernel_source: self.moe_kernel_source,
            topology,
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            num_experts: self.num_experts,
            topk: self.topk,
            model_shape: self.model_shape,
            nextn,
            quant: self.quant,
            estimator: self.estimator,
        };
        config.validate()?;
        Ok(config)
    }
}
